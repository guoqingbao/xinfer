# Kernel & Runtime Optimization Roadmap

This document tracks the profiling work and optimization plan for bringing xInfer
to vLLM/SGLang-level performance on CUDA, starting with FP8 dense models and the
Qwen3.5/3.8 hybrid (GatedDeltaNet + full attention) architecture.

All numbers below were measured on the two-H800-PCIe development host
(SM90, CUDA 13.0, 80 GB HBM each) with:

```bash
cargo build --release --features "cuda,nccl,flashinfer,cutlass"
```

Local kernel development used Cargo path patches (not committed) pointing
`candle-*` at a local checkout and `attention-rs` at the kernel workspace. The
committed `Cargo.toml` pins the upstream revisions instead; all numbers below
were produced with the same revisions that the PRs pin.

## Benchmark setup

```bash
# Decode throughput (ignore EOS so every request decodes max_tokens)
./target/release/xinfer --m /data/Qwen3.8-27B-FP8 --d 0 --batch 32 \
  --ignore-eos --max-tokens 128 --temperature 0 \
  --prompts "Write a detailed essay about the history of computing."

# Prefill throughput (8k-token prompt)
P=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*800)")
./target/release/xinfer --m /data/Qwen3.8-27B-FP8 --d 0 --batch 1 \
  --max-tokens 2 --temperature 0 --prompts "$P"
```

Profiling: Nsight Systems (`--disable-cuda-graph` for kernel-level timelines,
CUDA graphs hide individual kernels by default), Nsight Compute for DRAM/SM
utilization of individual kernels.

## Target model

`Qwen3.8-27B-FP8`: 64 layers, 48 GatedDeltaNet layers + 16 full-attention
layers, hidden 5120, FFN 17408, 24 Q / 4 KV heads, head_dim 256 for full
attention, 16/48 K/V heads for GDN, vocab 248320, FP8 E4M3 weights with
128x128 block scales (`weight_scale_inv` in BF16), BF16 lm_head and embeddings.

Per decode step the engine reads **~27.3 GB of weights** regardless of batch
size (gate/up 11.4 GB, down 5.7 GB, GDN projections 5.5 GB, attention 1.7 GB,
lm_head 2.5 GB). At the measured ~1.85 TB/s HBM bandwidth this is a **~14.7 ms
floor per step**, i.e. ~68 tok/s at batch 1 and ~2.1k tok/s at batch 32 before
any other work. Every measurement below should be read against that roofline.

## Baseline → current

| Benchmark | Before | After | Δ |
|---|---|---|---|
| 27B FP8 decode, batch 1 | 48.65 tok/s | **52.22 tok/s** | +7.3% |
| 27B FP8 decode, batch 8 | 315.45 tok/s | **366.84 tok/s** | +16.3% |
| 27B FP8 decode, batch 16 | 561.22 tok/s | **649.66 tok/s** | +15.8% |
| 27B FP8 decode, batch 32 | 1103.02 tok/s | **1303.51 tok/s** | +18.2% |
| 27B FP8 decode, batch 64 | — | 2558 tok/s | — |
| 27B FP8 decode, batch 128 | — | 3604 tok/s | — |
| Qwen3-4B FP8 decode, batch 1 | 181.3 tok/s | **224.2 tok/s** | +24% |
| Qwen3-4B FP8 decode, batch 32 | 3306 tok/s | **4077 tok/s** | +23% |
| 27B Q4_K_M GGUF decode, batch 1 | — | 61.9 tok/s | — |
| 27B Q4_K_M GGUF decode, batch 8 | — | 203.6 tok/s | — |
| 27B FP8 prefill, 8k tokens (opt-in fast GDN) | 3063 tok/s | **7593 tok/s** | +148% |
| 27B FP8 prefill, 8k tokens, default GDN | 3063 tok/s | **3994 tok/s** | +30% |
| 27B FP8 prefill, 20k tokens (fast GDN) | — | 7428 tok/s | — |

Roofline position: batch 1 is at ~84% of the all-GEMM floor (19.2 ms actual vs
16.0 ms floor including lm_head); batch 32 is at ~65% (24.5 ms/step actual vs
16.0 ms floor). The remaining gap is almost entirely non-GEMM kernels.

## Profiling findings (before changes)

Batch-1 decode, per token:

- FP8 GEMMs: 14.0 ms, each individual GEMM at 87–95% of HBM bandwidth. The
  DeepGEMM/FlashInfer blockscale kernels are already near-optimal — **the GEMM
  path was not the problem**.
- lm_head: 1.4 ms (BF16, 2.5 GB read, ~93% of peak bandwidth).
- cast/norm chain: BF16→F32 cast, `rmsnorm_f32`, F32→BF16 cast for **every**
  norm (two per layer plus Q/K norms), ~1.9 ms/token.
- SiLU+multiply in dense MLP: `usilu` + `bmul` plus two `narrow(...).contiguous()`
  copies of the packed gate/up output per layer.
- GDN host-side copies: ~8 `ucopy_bf16` launches per GDN layer from
  `narrow+contiguous` splits of the packed qkv projection, a `cat` of q/k/v,
  and non-contiguous conv-output slices.
- Batch 32: GDN decode kernel 3.4 ms/step, per-token-group activation
  quantization 0.7 ms/step, GDN bookkeeping copies ~1 ms/step.

Prefill (8k tokens), before changes:

- Sequential GDN recurrence: 1.0 s of 2.9 s (34%).
- Causal conv varlen: 0.20 s. The kernel walked the whole sequence per
  channel: ~80 GB/s effective, **~20x below bandwidth**.
- CUTLASS FP8 GEMM: 0.61 s (~716 TFLOPS, ~36% MFU).
- GDN host copies: 0.29 s.
- FlashAttention prefill itself was only 37 ms.

## Completed optimizations

1. **Token-parallel causal conv1d prefill** (`attention-rs/src/kernels/src/gdn.cu`).
   One block per token, 16-byte vectorized channel groups, history read from
   `x` (or the carried state at sequence boundaries), plus a separate state
   commit kernel launched afterwards to avoid read/write races. Verified
   bit-exact against the reference (max diff ~1e-7, float rounding). 0.20 s →
   ~0.02 s on 8k prefill.

2. **Chunked FlashInfer GDN prefill (opt-in, SM90-only)**
   (`gdn_flashinfer_prefill.cu`, already present behind
   `SM90_LOWER_PRECISION_GDN_PREFILL`). On typical inputs the final state is
   bit-identical to the reference recurrence and output p99 is 2.4e-4 (0.3% of
   range), but the project's differential probe shows non-exact state for some
   input distributions, so the chunked kernel stays opt-in:
   `XINFER_GDN_PREFILL=fast` (or the legacy variable = 1). It is roughly +58%
   prefill throughput (3063 → 4837 tok/s on its own, 7601 tok/s with the other
   changes below). Non-SM90 devices fall back automatically. Promoting it to
   the default needs a precision-policy decision.

3. **Fused RMSNorm and fused residual-add RMSNorm**
   (`attention-rs/src/kernels/src/norm.cu`, `attention-rs/src/norm.rs`,
   `src/models/layers/others.rs`). Reads/writes BF16/F16 activations, keeps
   reduction and weight in F32, and optionally writes the dtype-rounded
   residual sum. Verified **bit-identical** to the reference
   (cast→candle rmsnorm_f32→cast) for dims 128/256/5120. Replaced three
   launches per norm with one; wired into `NormX::forward` and
   `NormX::forward_add` (Qwen3 / Qwen3.5 decoder layers fuse the post-attention
   add). ~1.9 ms → ~0.65 ms of norm time per batch-1 token at the time of the
   change. The fused add-norm is wired into Qwen3, Qwen3.5, LLaMA, GLM4,
   DeepSeek, MiniMax and Qwen3.5/Qwen3 MoE decoder layers. Gemma, Phi4 and
   DFlash use different layer structures and were deliberately left on the
   reference path. Metal has no plain fused add-RMSNorm kernel yet:
   `attention_rs::norm` falls back to separate add + cast + Candle RMSNorm
   there, so CUDA is where the win lands (Metal kernel is a roadmap item).

4. **Fused dense-MLP SiLU-and-mul** (`src/models/layers/mlp.rs`,
   `MergedParallelColumnLinear::forward_packed`). The packed gate/up output is
   consumed in place; removed two `contiguous` copies and the separate
   `silu`/`mul` launches per layer. attention-rs' CUDA kernel now also handles
   F32, so GGUF models (which keep activations in F32) use the same fused
   path; the kernel has a float4 fast path with a scalar fallback.

5. **Packed GDN decode path** (`src/models/layers/deltanet.rs`,
   `attention-rs/src/gdn.rs`). Keeps `in_proj_qkv` packed, feeds it directly to
   the conv update, L2-normalizes q/k from strided slices
   (`l2_norm_range`), and passes v as a strided view
   (`gated_delta_rule_decode_slots_gqa_strided_v`). Removed ~7 of ~8 copies per
   GDN layer. Verified bit-exact against the compact reference.

6. **Coalesced GDN decode kernel** (`gdn_decode_smem_kernel`). One block per
   `(batch, value_head)`, whole 128x128 FP32 state in 64 KiB shared memory,
   coalesced per-row loads/stores, two recurrence passes in shared memory.
   Used for batch ≥ 20 on SM80+; the register kernel remains for small
   batches, where more blocks hide latency better. +2.5% batch-32 decode
   throughput.

7. **External per-token-group FP8 activation quantization**
   (`attention-rs/src/fp8_linear.rs`). Both decode and prefill now quantize
   activations with attention-rs' vectorized `per_token_group_quant_fp8_v2`
   kernel and call the FP8-input blockscale runner, instead of relying on the
   runner's internal quantizer. At M=8192 the outputs are **bit-identical** to
   the CUTLASS path (verified with a torch reference; rel. error 3.7e-3 for
   both vs a dequantized FP32 reference, i.e. pure FP8 quantization error).
   Prefill 6609 → 7601 tok/s; decode +1%.

## Remaining roadmap

The full, continuously updated opportunity backlog with file/function-level
details is in [`docs/performance_opportunities.md`](performance_opportunities.md).

Priorities are ordered by measured/estimated impact at batch 32 unless noted.
The batch-32 step is ~24.5 ms; ~16 ms is the unavoidable weight-read floor.

### P0 — CUDA graph coverage for large batches

`planned_graph_capture_batches` captures every exact batch up to 32 and the
hybrid GDN path requires exact sizes (`mamba_slot_mapping` cannot be padded
safely), so batch 64/128 currently run eagerly. Batches 64/128 are the
throughput regime. Options:

- Capture a sparse set of exact large sizes (48/64/96/128) when
  `max_num_seqs` exceeds 32. Memory is the constraint: on the 27B model the
  graph pool already holds ~57 GB with 32 graphs.
- Alternatively make GDN graph replay batch-size agnostic by allocating dummy
  slots and valid dummy KV blocks for padded rows, so one captured graph can
  serve a range of batch sizes.
- Expected: +10–20% at batch 64/128.

### P0 — Fuse FP8 activation quantization into producers

`per_token_group_quant_fp8_v2` costs ~0.55 ms/step at batch 32 (one launch per
FP8 GEMM). Most GEMM inputs are produced by our own kernels:

- norm → qkv / GDN qkv/z projections (rms_norm output),
- silu-and-mul → down_proj,
- gated-rmsnorm-silu-mul → GDN out_proj.

Add optional FP8 + scale outputs to these kernels and a pre-quantized input
path through `LinearX` (`LnFp8::forward_prequant`). Saves the per-GEMM quant
launch and one BF16 read/write per projection (~0.5 ms/step, ~2.5%).

### P1 — GDN decode at the DRAM floor

The shared-memory kernel is at ~3.1 ms/step vs a ~2.75 ms DRAM-bound floor
(state traffic is 48 layers × 201 MB logical per step, partly served by L2).
Next steps: `cp.async`/TMA-staged loads, two heads per block to increase
occupancy, or transposed (V-major) state layout to shorten the per-thread
stride. Expected: 0.3–0.5 ms/step.

### P1 — Merge GDN `in_proj_a`/`in_proj_b` BF16 GEMVs

Two cuBLASLt split-K BF16 GEMVs per GDN layer (`[T,48] = [T,5120]·[5120,48]`)
cost ~0.68 ms/step including the split-K reduction. Pack A and B into one
`[T,96]` projection (one launch), or replace both with a small fused GEMV
kernel sized for 48/96 output columns. Expected: 0.3–0.5 ms/step.

### P1 — Sampling and logits

`fast_argmax_f32` costs ~0.19 ms per step (both batch 1 and 32) for a 248k
vocabulary. A fused argmax in the lm_head epilogue or a persistent two-pass
reduction can cut this by 5–10x. Low absolute impact but it is per token, not
per batch, so it matters for latency-oriented batch-1 serving.

### P1 — GGUF quantized GEMV throughput

The GGUF path is far from its weight-read floor: Qwen3.8-27B-Q4_K_M reads
~16 GB of quantized weights per decode step (~8.6 ms at 1.85 TB/s, i.e.
~930 tok/s at batch 8), but measures 204 tok/s. The dequantizing matvec
kernels, not the activation path, are the bottleneck. Profile the Q4_K/IQ
matvec kernels and consider a fused dequant+MMA (MMQ-style) kernel for larger
decode batches.

### P1 — Metal parity for fused norm

The new fused RMSNorm/add-RMSNorm kernels are CUDA-only. On Metal the path
falls back to a separate add, an F32 cast (when the weight is F32), Candle's
RMSNorm and a cast back. A Metal fused-add-rmsnorm kernel in
`metal-kernels`/`metal_flash` would bring the same launch savings to Apple
silicon; it needs the same dtype/offset semantics as the CUDA version.

### P1 — Batch-1 tail

Batch 1 has ~3.1 ms/token outside GEMMs (84% of roofline). Remaining items are
small launches: residual add after MLP (one `badd` per layer), Q/K norm + RoPE
(already fused separately but two norm launches per attention layer), and the
GDN register kernel. Fusing the MLP residual add across the layer boundary into
the next layer's input norm (fused add-norm already exists) is the next
structural step.

### P2 — Speculative decoding

MTP (Qwen3.8 ships `mtp.safetensors`) and DFlash2 draft models are already
supported (`--draft-model`, `--num-speculative-tokens`). These are the largest
wins for interactive batch-1 latency; they need dedicated tuning/benchmarking
after the kernel work above.

### P2 — NVFP4 (next phase)

The software NVFP4 path (Hopper) needs the same treatment: profile
`nvfp4_gemm` / requantization, check gate/up merge and block-scale swizzle
overhead, and compare against the FP8 path on the same model. Hardware NVFP4
kernels (SM100/SM120) must remain gated by compute capability.

### P2 — Long-context / KV cache

KV cache quantization (`fp8`, `turbo8`, `turbo4`, `turbo3`) is already
supported; measure its interaction with the new prefill/decode kernels and
tune split-K/partition sizes for 32k+ contexts. PD disaggregation and chunked
prefill are already implemented — benchmark them after the kernel work.

## Platform notes

- The new norm/conv/GDN kernels use plain CUDA vector loads and shared memory;
  no architecture-specific instructions. They compile for SM70+.
- The 64 KiB-shared GDN decode kernel is gated to SM80+ and batch ≥ 20 at
  runtime; older archs and small batches fall back to the register kernel.
- Hardware FP8 paths remain gated as before (`flashinfer`/`cutlass`, SM89/90+);
  the software paths are unchanged.
- Metal is unaffected: `attention_rs::norm` falls back to Candle ops when the
  device is not CUDA, and the new GDN kernels are CUDA-only with the Metal
  wrappers untouched.

## Validation

- `tests/norm_fused.rs` — fused vs cast→rmsnorm_f32→cast, bit-identical.
- `tests/gdn_conv_parallel.rs` — token-parallel conv vs CPU reference
  (out/state/snapshots), lengths 1..130, kernel sizes 2–4.
- `tests/gdn_strided.rs` — strided L2 norm and strided-v GDN decode vs the
  compact reference, bit-identical.
- `tests/gdn_fi_diff.rs` — chunked FlashInfer GDN prefill vs reference.
- `examples/fp8_paths_probe.rs` + torch reference — FlashInfer/CUTLASS/quant
  path equivalence at M=512 and M=8192.
- `tests/run_precision_suite.sh --include-optin-gdn` — full differential suite.

## Reproducing the delta

```bash
# Baseline: build pinned upstream crates (comment out [patch] in Cargo.toml)
cargo build --release --features "cuda,nccl,flashinfer,cutlass"

# Current: local candle + attention.rs via [patch]
cargo build --release --features "cuda,nccl,flashinfer,cutlass"

# Optional runtime switches
XINFER_GDN_PREFILL=fast                  # opt-in chunked FlashInfer GDN prefill (SM90)
XINFER_GDN_PREFILL=exact                 # reference sequential GDN prefill
SM90_LOWER_PRECISION_GDN_PREFILL=1       # legacy alias for the fast path
```
