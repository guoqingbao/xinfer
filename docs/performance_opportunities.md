# xInfer Performance Opportunities Backlog

This is the working backlog for making xInfer competitive with vLLM/SGLang.
Every item lists the measured bottleneck, the proposed change, the exact files
and functions involved, validation, and risk. It complements
`docs/kernel_optimization.md` (which records the work already merged/PR'd:
fused RMSNorm/add-norm, fused SiLU-and-mul incl. F32, packed GDN decode,
coalesced GDN decode, token-parallel conv prefill, external FP8 quant).

Measurement host: 2×H800 PCIe (SM90, 80 GB, ~1.85 TB/s measured copy/read
bandwidth), CUDA 13.0. Target model for most numbers: `Qwen3.8-27B-FP8`
(64 layers: 48 GDN + 16 full attention, hidden 5120, FFN 17408, FP8 128×128
block scales, BF16 lm_head/embedding).

---

## 0. Roofline and current position

Per decode step the model reads ~27.3 GB of weights (gate/up 11.4 GB, down
5.7 GB, GDN projections 5.5 GB, attention 1.7 GB, lm_head 2.5 GB). At
1.85 TB/s that is **~14.7 ms/step regardless of batch size**, plus lm_head is
included in that number; the practical floor is **~16 ms/step**.

| Workload | Current | Floor | Efficiency |
|---|---|---|---|
| 27B FP8 decode b1 | 52.2 tok/s (19.2 ms/token) | ~62.5 tok/s (16 ms) | 84% |
| 27B FP8 decode b32 | 1303 tok/s (24.5 ms/step) | ~2000 tok/s (16 ms) | 65% |
| 27B FP8 prefill 8k, default GDN | 3994 tok/s | GEMM/compute bound | — |
| 27B FP8 prefill 8k, `XINFER_GDN_PREFILL=fast` | 7593 tok/s | — | — |
| 27B Q4_K_M GGUF b8 | 203.6 tok/s | ~930 tok/s | 22% |
| 27B NVFP4 (software) b8 | 51.6 tok/s | FP8 parity ~360 tok/s | 14% |

Measured decode-only kernel breakdown, 27B FP8 batch 32, per step (eager,
`--disable-cuda-graph`; CUDA-graph replay is ~24.5 ms/step):

| Kernel | ms/step | Share | Notes |
|---|---|---|---|
| FP8 GEMMs (DeepGEMM blockscale) | 14.6 | 62% | each 87–95% of HBM peak; at floor |
| GDN decode (`gdn_decode_smem_kernel`) | 3.12 | 13% | DRAM/L2 bound, ~2.75 ms floor |
| lm_head (`nvjet_*`, BF16) | 1.41 | 6% | at ~93% peak, unavoidable bytes |
| per-token FP8 quant (`per_token_group_quant_fp8_v2`) | 0.54 | 2.3% | one launch per FP8 GEMM |
| fused RMSNorm | 0.62 | 2.6% | 163 launches/step, launch-bound |
| GDN in_proj_a/b BF16 GEMV + splitK reduce | 0.68 | 2.9% | 2 cuBLASLt calls per GDN layer |
| `ucopy_bf16` | 0.30 | 1.3% | ~90 launches/step, attribution needed |
| fused SiLU-and-mul | 0.22 | 0.9% | |
| gated RMSNorm + SiLU | 0.20 | 0.8% | |
| argmax / sampling | 0.19 | 0.8% | per step, not per token |
| GDN L2 norm (`l2_norm_range`) | 0.18 | 0.8% | 2 launches/layer |
| conv update | 0.16 | 0.7% | |
| residual add (`badd_bf16`) | 0.16 | 0.7% | one per layer after MLP |
| FlashInfer paged decode + KV append | 0.19 | 0.8% | |
| `is_i64_f32` casts | 0.12 | 0.5% | ~16/step, source unknown |

Batch-1 per token is ~16 ms of GEMM/lm_head and ~3.1 ms of everything else.
Large-batch throughput is limited by the ~8.5 ms/step of non-GEMM work above.

Prefill (8k tokens) after the merged work, `XINFER_GDN_PREFILL=fast`:
FP8 blockscale GEMM ~0.61 s is now the largest item (~40%), chunked GDN
prefill ~0.03 s, conv ~0.02 s, attention ~0.04 s, remaining host copies and
elementwise work ~0.2 s. With the default reference GDN recurrence, GDN
prefill dominates again (~1.0 s).

---

## 1. Prioritized opportunity table

| # | Area | Opportunity | Expected gain | Effort/risk |
|---|---|---|---|---|
| K1 | kernel | Fuse FP8 activation quant into norm/SiLU/gated-RMSNorm producers | 0.5–0.7 ms/step b32; ~0.6 ms/token b1 | Medium |
| K2 | kernel | GDN decode to DRAM floor (TMA/cp.async, 2 heads/block) | 0.3–0.6 ms/step | Medium |
| K3 | kernel | Merge GDN `in_proj_a`/`in_proj_b` into one GEMM/GEMV | 0.3–0.5 ms/step | Low |
| S1 | structural | CUDA graph coverage beyond batch 32 (exact GDN sizes) | +10–20% at b64/b128 | High |
| S2 | structural | Remove per-step host tensor build + replay copies | latency/jitter; 0.2–1 ms/step eager | Medium |
| K4 | kernel | Sampling/argmax + logits dtype | 0.1–0.2 ms/step | Low |
| K5 | kernel | Fuse QK-norm + QK-norm/RoPE launches; multi-row norm | 0.1–0.2 ms/step | Low |
| K6 | kernel/struct | Fuse final residual add across layer boundary | ~0.15 ms/step | Medium |
| K7 | kernel | Attribute/remove remaining ucopy, is_i64 casts, copy2d | 0.2–0.4 ms/step | Low |
| G1 | format | GGUF quantized GEMV/MMQ (next major PR) | b8 204 → 500–900 tok/s | High |
| G2 | format | NVFP4 software path (Hopper) | b8 52 → 200+ tok/s | High |
| P1 | prefill | Make chunked GDN prefill default (precision work) | default prefill 3994 → ~7500 tok/s | Medium |
| P2 | prefill | Tune prefill FP8 GEMM (DeepGEMM/CUTLASS configs) | 0.61 s → 0.4–0.5 s | Medium |
| S3 | scheduler | Mixed prefill/decode scheduling, chunk tuning | TTFT/ITL stability | Medium |
| S4 | speculative | MTP/DFlash tuning and verification kernels | 1.5–2.5× b1 decode | Medium |
| S5 | MoE | Profile/optimize MoE routing, grouped GEMM, offload | model-dependent | Medium |
| S6 | multi-GPU | TP/NCCL overlap, graph + collectives, PD | near-linear scaling | High |
| S7 | memory | KV dtype (fp8/turbo), allocator, prefix cache tuning | larger batch/context | Low–Medium |
| S8 | Metal | Fused add-RMSNorm and GGUF kernels on Metal | Metal parity | Medium |

---

## 2. Kernel-level opportunities

### K1. Fuse FP8 activation quantization into producer epilogues

**Bottleneck.** `per_token_group_quant_fp8_v2_kernel` runs once per FP8 GEMM
(304 launches/step, ~0.54 ms at batch 32, ~0.6 ms per batch-1 token). Almost
every GEMM input is produced by one of our own kernels:

- decoder input norm → full-attn QKV, GDN `in_proj_qkv`/`in_proj_z`
- fused SiLU-and-mul → MLP `down_proj`
- gated RMSNorm + SiLU → GDN `out_proj`
- attention output → attention `o_proj` (harder; FlashInfer epilogue)

**Change.** Add quantizing variants that write `(BF16 output, FP8 payload,
column-major per-128-group scales)` in one launch:

- `rms_norm_quant`, `add_rms_norm_quant` in
  `attention-rs/src/kernels/src/norm.cu` + `attention-rs/src/norm.rs`
- `silu_and_mul_quant` in `attention-rs/src/kernels/src/silu_and_mul.cu`
- `gated_rmsnorm_{silu,sigmoid}_mul_quant` in `attention-rs/src/gdn.rs`
  (fused kernel already exists in `gdn.cu`)

Then add a pre-quantized input path through the projection stack:
`LinearX::forward_prequant` in `src/models/layers/linear.rs`
(`LnFp8::forward` currently re-quantizes internally), and thread the
`(fp8, scales)` pair through `TensorParallelColumnLinear` /
`MergedParallelColumnLinear` in `src/models/layers/distributed.rs`.
Call sites: `src/models/layers/attention.rs` (`forward_with_kv`),
`src/models/layers/mlp.rs` (`MLP::forward`), `src/models/layers/deltanet.rs`
(`forward_decode_packed`).

**Important details.** The runner expects column-major scales with
`scale_stride = round_up(m, 4)` (`attention-rs/src/fp8_linear.rs`,
`fp8_matmul_flashinfer_with_input_scale`); the producer must use the same
layout, and scratch buffers must come from `get_grouped_gemm_scratch`
(graph-domain aware). Keep a fallback when the projection is not FP8.

**Validation.** Compare logits before/after at temperature 0 on ~20 prompts
and run `tests/run_precision_suite.sh`; add an FP8 GEMM probe with
pre-quantized input. Risk: layout/stride mismatch and precision drift.

### K2. GDN decode at the DRAM floor

**Bottleneck.** `gdn_decode_smem_kernel` (`attention-rs/src/kernels/src/gdn.cu`,
launched from `gdn.rs::gated_delta_rule_decode_slots_gqa`) is 3.12 ms/step at
batch 32. State traffic is `batch × 48 heads × 128×128 × 4 B × 2` per layer
(~201 MB logical/layer, ~9.7 GB/step); ncu measured 63 MB read + 43 MB write
per launch (L2 absorbs part), 34% DRAM, 15% SM. Floor ~2.75 ms.

**Change.**
- Prefetch state with `cp.async`/TMA into the shared-memory tile instead of
  synchronous coalesced loads; overlap the two passes.
- Put two value heads per block (2×64 KiB shared is too much for SM80; use
  half-state tiling or 32×128 tiles) to raise occupancy from 3 blocks/SM.
- Consider a V-major `[V,K]` state layout so each thread reads contiguous K
  for one column; keep the existing K-major layout as fallback.
- Fuse the L2 normalization of q/k into this kernel (compute per-head norm in
  shared, reuse across value heads) to drop the two `l2_norm_range` launches
  (0.18 ms/step) and their reads.
- Fuse `fused_gdn_gating` (0.08 ms) or the gated RMSNorm into the same launch.

Files: `attention-rs/src/kernels/src/gdn.cu` (`gdn_decode_smem_kernel`,
`launch_gdn_decode_smem`), `attention-rs/src/gdn.rs`
(`gated_delta_rule_decode_slots_gqa`, `l2_norm_range`).

**Validation.** Differential vs the register kernel and vs the reference
recurrence; keep SM70/75 on the register kernel. Watch shared-memory limits
per SM (64 KiB is the whole budget on SM75).

### K3. Merge GDN `in_proj_a` / `in_proj_b`

**Bottleneck.** Each GDN layer runs two tiny BF16 GEMVs (`[T,48] = [T,5120]·
[5120,48]`) through cuBLASLt with split-K reduction: ~0.68 ms/step at batch
32 (two `nvjet_*_splitK` + two `splitKreduce_kernel` per layer).

**Change.** At load time concatenate the two weights into `[96,5120]` and
split the output, so one GEMM/GEMV covers both (same as the existing GGUF
`PackedQkvzBa`/`in_proj_ba` concept). For decoding, a small custom kernel
(one block per token, warp per 16 output columns) avoids the split-K reduce.
Files: `src/models/layers/deltanet.rs` (`project_inputs`,
`project_inputs_decode_packed`, `GdnProjection::SplitQkvZaLegacy`),
`src/models/layers/distributed.rs`, `attention-rs/src/gdn.rs`.

**Validation.** Bit-identical projection outputs (just concatenated matmul);
precision suite GDN probe. GGUF already packs this in some variants.

### K4. Sampling and logits

**Bottleneck.** `src/core/runner.rs::sample` (line ~1752) always casts logits
to F32, then `LogitsProcessor::sample_with_strategy`
(`src/utils/logits_processor.rs:258`) runs the CUDA sampler. Greedy uses
`fast_argmax_f32` at ~0.19 ms/step for the 248k vocabulary (both b1 and b32);
`Sampling::ArgMax` also goes through the generic candle path
(`sample_argmax`, `logits.argmax(...).to_vec1()`), forcing a D2H sync.

**Change.** Add a CUDA argmax/top-k epilogue specialized for the vocabulary
width that writes FP32 token ids directly (one small D2H of `batch` u32), and
fuse argmax into the lm_head projection epilogue for greedy decode. Avoid the
BF16→F32 logits cast by sampling from F32 logits when the head already
produces F32 (`VocabParallelLinear::forward`), or from BF16 with accumulation
in F32. For default temperature sampling, keep the existing
`attention-rs/src/sampler.rs::sample_cuda` but trim the sort
(`attention-rs/src/sort.rs`, `src/kernels/src/fast_topk.cu`, `sort.cu`).
Files: `src/core/runner.rs`, `src/utils/logits_processor.rs`,
`src/models/layers/distributed.rs`, attention-rs sampler/sort kernels.

**Validation.** Compare sampled ids at temperature 0 and distributions at
temperature >0 against the current sampler; check penalized/logit-processor
paths (frequency/presence penalties, guided decoding) still use F32.

### K5. Remaining norm/attention prep launches

**Bottleneck.** Fused RMSNorm is launch-bound: 163 launches/step cost
0.62 ms at batch 32 although the data is only ~100 MB/step. Q and K norms are
two separate launches per full-attention layer
(`src/models/layers/attention.rs::prepare_qk_for_rope`), followed by
`fused_rope_partial_tok_major_bf16_kernel`.

**Change.** (a) Add a `rms_norm_pair` kernel normalizing Q and K in one launch
(shared weight-sized prologue; different head counts). (b) Fuse Q/K norm into
the RoPE kernel (`attention-rs/src/kernels/src/fused_rope.cu`,
`attention-rs/src/fused_rope.rs`) so `prepare_qk_for_rope` disappears for the
common per-head RMSNorm case. (c) For small batches, process multiple rows per
block in `rms_norm.cu` to reduce block count/latency.

**Validation.** Bit-identical to the separate kernels (norm tests already do
this); attention probes.

### K6. Fuse the final residual add across the layer boundary

**Bottleneck.** Every decoder layer ends with `residual + mlp_output`
(`badd_bf16`, 0.16 ms/step at batch 32) and the next layer starts with an
input RMSNorm. The post-attention add is already fused into the norm.

**Change.** Let a decoder layer return `(mlp_output, residual)` and have the
next layer's `input_layernorm.forward_add(mlp_output, residual)` fuse both the
add and the norm; the final layer performs the add before the model norm.
Files: `src/models/layers/others.rs` (`NormX::forward_add` already exists) and
the decoder loops in `qwen3_5.rs`, `qwen3.rs`, `llama.rs`, `glm4*.rs`,
`deepseek3.rs`, `qwen3_moe.rs`, `qwen3_5_moe.rs`, `minimax.rs`.
Note the GDN/MoE layers are called inside the model's layer loop in the same
file; thread the pair through rather than changing `forward` in place.

**Validation.** Bit-identical residual sums (the fused kernel already rounds
the same way); run logits probes.

### K7. Attribute and remove remaining copies/casts

**Bottleneck.** After the GDN refactor, `ucopy_bf16` is still ~0.30 ms/step
(~90 launches/step) at batch 32, plus `is_i64_f32` ~0.12 ms (~16/step) and
`copy2d_bf16` ~0.1 ms. These are small kernels whose launch cost dominates.

**How to attribute.** Profile with `--disable-cuda-graph`, export sqlite, and
correlate `correlationId` with CUDA API calls, or add NVTX ranges
(`--features nvtx`) around candidate host blocks. Candidates:

- full-attention Q/K/V views and `reshape`/`contiguous` in
  `src/models/layers/attention.rs` (`forward_with_kv`, `prepare_qk_for_rope`);
- mask/position setup in `src/models/layers/mask.rs` and
  `src/models/layers/rotary_emb.rs`;
- `src/core/runner.rs::prepare_decode` (positions/context_lens) and
  `prepare_mamba_slot_mapping` (I64 tensor build);
- `src/models/layers/deltanet.rs` (`finish_gdn_output` reshape, z dtype
  cast), `src/models/layers/mlp.rs` (activation reshape).

**Change.** Cache device-side positions/mask/slot tensors and update them in
place; keep attention outputs as views where possible; convert I64 slots once
per step instead of per layer; avoid F32↔BF16 casts when dtypes already match.

### K8. Attention decode / KV append

FlashInfer paged decode + append is only ~0.19 ms/step at batch 32, so it is
not a priority for this model. Re-check after other gains for:
- long contexts (32k+) where split-KV/partition sizes matter
  (`attention-rs/src/flashinfer.rs`, `paged_attention.rs`);
- MLA models (`attention-rs/src/mla.rs`) and DSA/FlashMLA paths
  (`XINFER_ENABLE_FLASHMLA`), which were not benchmarked here;
- sliding-window models and FP8/TurboQuant KV cache paths.

---

## 3. Structural/runtime opportunities

### S1. CUDA graph coverage beyond batch 32

**Bottleneck.** `src/utils/graph.rs::planned_graph_capture_batches` captures
exact sizes 1..=32. Hybrid GDN models require exact sizes because
`mamba_slot_mapping` cannot be padded safely (`GraphCapturer::replay`,
`require_exact_batch`). Batches 64/128 therefore run eagerly (measured 2558
and 3604 tok/s at 27B FP8). Capturing 32 graphs already consumes a large graph
pool (the 27B log reported ~57 GB pool usage at batch 32); capturing 1..128
naively will OOM.

**Options.**
1. Sparse exact captures (e.g. 40/48/56/64/80/96/112/128) with a memory guard
   based on `report_graph_pool_usage` and free VRAM; replay picks the next
   captured size only for non-Mamba models.
2. Make GDN graphs batch-agnostic: replay with padded rows whose
   `mamba_slot_mapping` points at dedicated scratch slots and whose KV block
   tables point at private dummy blocks, so a single graph can serve a range.
3. Capture-on-demand for large batches with an LRU of graphs.

Files: `src/utils/graph.rs` (capture/replay), `src/core/runner.rs`
(`prepare_decode`, `prepare_mamba_slot_mapping`), `src/core/engine.rs`
(decide capture at startup), `src/models/qwen3_5.rs`
(`get_mamba_slots_for_sequences`).

**Validation.** Compare outputs/graph vs eager at b64/b128, check peak memory
with `report_graph_pool_usage`, run 1000+ steps to catch pool fragmentation.

### S2. Per-step host work and H2D copies

**Bottleneck.** `src/core/runner.rs::prepare_decode` (line ~1565) builds
`input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`,
`mamba_slot_mapping` from host `Vec`s every step (new device allocations and
H2D copies). `GraphCapturer::replay` then zeroes and copies them again into
static buffers (`prev` tensors). CPU gaps showed up as ~30.9 ms wall vs
~25.5 ms kernel time in eager batch-32 profiling.

**Change.** Allocate pinned host staging buffers and device buffers once,
fill in place, and issue one async H2D copy per step; in replay copy only the
changed slices (no `zero_` of the full block table), and keep block tables on
device with in-place page-id updates from the allocator. For batch-1 latency,
precompute positions from a length counter on device.

Files: `src/core/runner.rs`, `src/utils/graph.rs`,
`src/core/block_manager.rs`, `src/utils/kvcache_allocator.rs`.

**Validation.** CPU timeline in nsys should show no per-step allocations;
measure TTFT and inter-token latency at b1 and b64.

### S3. Scheduler / continuous batching

**Bottleneck.** Chunked prefill (`--prefill-chunk-size`, default 8k, min 1k,
max 32k) and continuous batching exist, but a long prefill can delay decode
steps; `src/core/scheduler.rs` and `Engine::prepare_step` decide batch
composition. We have not measured mixed prefill+decode (ITL spikes).

**Change.** Benchmark a mixed workload (one 32k prompt arriving during 32
decodes). Tune chunk size, `max_num_batched_tokens`, decode-first ordering,
and the KV-usage swap threshold (`KVCACHE_SWAP_THRESHOLD` in
`src/core/engine.rs`). Consider prioritizing decode over new prefills once
ITL exceeds a bound. Also verify why some 27B runs reported
`max 20 parallel reqs` (kvcache allocator/engine cap) and whether that is the
intended `max_num_parallel_reqs`.

### S4. Speculative decoding (MTP / DFlash)

MTP (`src/speculative/mtp.rs`, `src/models/qwen3_5_mtp.rs`) and DFlash2
(`src/speculative/dflash.rs`) are implemented but were not benchmarked here.
The model directory ships `mtp.safetensors` and a DFlash2 draft
(`/data/Qwen3.8-27B-DFlash2`, `-FP8`). Work:

- measure acceptance rate and tokens/verify for `--num-speculative-tokens`
  3/5/7 at b1 and b8;
- profile the verify forward (batch = seqs × (k+1)); GDN rollback
  (`GatedDeltaNet::rollback_mtp_verify_at`) and conv/recurrent snapshots add
  bandwidth that the normal decode path does not have;
- tune tie-breaking/sampling in `src/speculative/verify.rs`;
- ensure CUDA graph capture covers MTP verify (`mtp_capturer`) and that the
  GDN exact-batch constraint is honored.

This is the largest latency win for interactive b1 (expect 1.5–2.5× if
acceptance is good).

### S5. MoE models

Only a smoke test was run (Qwen3.5-35B-A3B-FP8 b8 = 410 tok/s). Profile
before changing:
- routing/top-k (`src/models/layers/moe.rs::route`, `src/models/layers/other`),
  permutation/scatter kernels, grouped GEMM dispatch thresholds
  (`FusedMoe`, `FusedMoeGGUF`, `FusedMoeISQ` in `src/models/layers/moe.rs`);
- attention-rs kernels `moe_gemm.cu`, `moe_gemv.cu`, `moe_gemm_gguf.cu`,
  `moe_gguf_small_m.cu`, `moe_wmma_gguf.cu`, `moe_w2_*`;
- shared-expert output add and gate/up merge (`MLP::merge_nvfp4_gate_up`,
  GGUF packing) — the same SiLU/quant fusion ideas apply;
- expert offload (`XINFER_MOE_OFFLOAD`, `src/models/layers/moe_offload/`,
  `src/models/layers/moe_w2_delta.rs`) — measure PCIe bound and overlap.

### S6. Multi-GPU / multi-node / PD

TP paths use NCCL all-reduce (`src/models/layers/distributed.rs`,
`src/core/runner.rs`). Not benchmarked in this session. Items:
- 2×H800 TP for 27B FP8 (`--d 0,1`) vs single GPU: measure compute/bandwidth
  scaling and the GDN head sharding;
- overlap all-reduce with the next projection / fused residual-add all-reduce;
- CUDA graph capture with NCCL collectives (graph-compatible comms);
- multi-node (`src/utils/multi_node.rs`): TCP forward coordination and NCCL ID
  exchange overhead;
- PD disaggregation (`--pd-server`/`--pd-client`, `src/transfer/`): KV transfer
  bandwidth and prefill/decode balance.

### S7. KV cache, allocator, prefix cache

`src/utils/kvcache_allocator.rs` (open in the editor) sizes the KV pool and
mamba cache from `--kv-fraction`/`--mamba-fraction`; block size is 64 on CUDA
(32 on Metal). Opportunities:
- tune block size/page size for long context (attention kernel partition
  sizes and prefix-cache granularity);
- test `--kvcache-dtype fp8|turbo8|turbo4|turbo3` for memory vs quality; MLA
  models only support fp8 and TurboQuant auto-falls back without
  flashinfer/flashattn;
- CPU swap path in `src/core/block_manager.rs` and the swap threshold in
  `src/core/engine.rs` — measure preemption stalls under memory pressure;
- prefix cache hashing/eviction (`src/core/prefix_cache.rs`): long multi-turn
  benchmarks, eviction policy, block reuse across requests.

### S8. Server / tokenizer / streaming

TTFT includes HTTP + tokenization. Measure `/tokenize`, chat-template
rendering (`src/utils/chat_template.rs`, minijinja), and incremental
detokenization in the stream path (`src/core/engine.rs` stream decoders).
Consider a tokenizer thread pool, avoiding locks around the engine, and
batching stream sends. Relevant when b1 latency work (speculative decoding)
lands.

---

## 4. Format-specific work

### G1. GGUF quantized GEMV/MMQ (next major PR)

**Measured.** Qwen3.8-27B-Q4_K_M: b1 61.9 tok/s, b8 203.6 tok/s. Weight bytes
~16 GB/step → ~8.6 ms floor → ~930 tok/s at b8. Efficiency ~22%. F32
SiLU fusion only gained ~1.6%, confirming the matvec path dominates.

**Plan.**
1. Profile with nsys (`--disable-cuda-graph`) and ncu on the top GGUF
   matvec kernels: candle `candle-kernels/src/quantized.cu`
   (`mul_mat_vec_q`, `dequantize_mul_mat_vec`, `mul_mat_vec_q_*` variants)
   and attention-rs `src/kernels/src/gguf_gemm.cu`,
   `moe_gemm_gguf.cu`, `moe_gguf_small_m.cu`.
2. For batch-1/low batch, implement a specialized warp-per-row GEMV with
   vectorized quant block loads (`int4`), one block per output tile, and
   `__ldg`/L2-friendly layouts. Keep the existing packed QKV/gate-up merges
   (`attention.rs::try_pack_gguf_qkv`, `mlp.rs::merge_gguf_gate_up`).
3. For larger decode batches, implement an MMQ-style kernel (dequantize Q4_K
   blocks to int8 and use `mma.sync` int8 tensor cores), with a threshold vs
   the GEMV kernel. This is the approach llama.cpp uses; candle's
   `mul_mat_vec_q` is bandwidth-limited by 4-bit unpacking.
4. IQ formats: candle's IQ matmul is likely slower; evaluate dequantizing IQ
   weights to Q8_0/F16 at load when VRAM allows
   (`src/utils/gguf_varbuilder.rs`, `linear.rs::QLinear`), or add IQ-specific
   warp kernels.
5. lm_head for GGUF may be quantized or dense; verify the logits projection
   path (`VocabParallelLinear::load_no_bias` + QLinear) and fuse argmax (K4).

**Validation.** Token-for-token comparison vs current output at temp 0 is too
strict for kernel changes; compare logits on a fixed batch of prompts
(max/mean diff), run the GGUF models in the precision probes, and benchmark
b1/b8/b32 across Q4_K_M and IQ formats.

### G2. Software NVFP4 (Hopper)

**Measured.** 27B NVFP4 b8 = 51.6 tok/s vs 366.6 for FP8, i.e. ~7× slower.
This is the explicit "next phase" format.

**Plan.** Profile `attention_rs::nvfp4_linear::nvfp4_matmul`
(`attention-rs/src/nvfp4_linear.rs`) and kernels
`nvfp4_gemm.cu`, `nvfp4_quant.cu`, `nvfp4_quant_flashinfer.cu` on SM90
(software path):
- determine whether the kernel dequantizes the full weight to BF16 per call
  (3× traffic) or uses LUT unpacking; test `XINFER_NVFP4_FORCE_LUT=1`;
- fuse dequantization into the GEMM mainloop (no BF16 materialization),
  vectorize E2M1 unpacking (`lop3`/LUT), apply block scales in the epilogue;
- reuse the merged gate/up path (`MLP::merge_nvfp4_gate_up`) and its swizzled
  scales; avoid re-swizzling per call;
- consider FP8 activation quantization before NVFP4 weights (W4A8) if
  accuracy allows, which lets the fast FP8 blockscale GEMM read 4-bit weights
  after an on-the-fly dequant;
- keep hardware NVFP4 (SM100/SM120, `nvfp4_gemm_cutlass.cu`,
  `nvfp4_gemm_flashinfer*`) untouched and gated.

**Validation.** NVFP4 precision probe (`compare_nvfp4_probe.py`), logits vs
the software reference, and the Hopper fallback test path in the suite.

### G3. Chunked GDN prefill accuracy → default

The chunked FlashInfer kernel gives 7593 vs 3994 tok/s prefill but the
project's opt-in GDN probe expects exact state/output for that path; some
input distributions differ (state rel error up to 0.4 in the probe, final
state bit-identical for typical inputs). Options:
- understand whether the probe's expected `allowed=0` can be relaxed for the
  documented lower-precision path;
- or improve the kernel's accumulation (FP32 where it uses lower precision)
  or add periodic state checkpoints to bound drift;
- then flip the default in `src/utils/env.rs`
  (`sm90_lower_precision_gdn_prefill`) and update docs/tests.

This is the highest prefill-throughput item once precision is settled.

### G4. Prefill FP8 GEMM tuning

With fast GDN prefill, the blockscale GEMM is ~0.61 s of 1.54 s at 8k
(~716 TFLOPS, ~36% MFU). Try:
- DeepGEMM/TRT-LLM runner configuration selection for M=8192 and the
  N/K shapes (5120×17408, 34816×5120, 6144×5120, 14336×5120), including
  cluster shapes and swapAB thresholds (`attention-rs/src/fp8_linear.rs`,
  `flashinfer_bmm_fp8.cu`);
- whether FlashInfer's own FP8 attention/prefill kernels can replace the
  current path for the 16 full-attention layers;
- persistent CUTLASS scheduler settings in `fp8_gemm_cutlass.cu` as a
  fallback.

---

## 5. Platform coverage and Metal

- New CUDA kernels are plain CUDA and compile for SM70+. The shared-memory
  GDN decode kernel is gated to SM80+ and batch ≥ 20; the register kernel is
  the fallback. Keep that policy when changing tiling.
- Add compile-only CI smoke builds with `CUDA_COMPUTE_CAP` = 70/75/80/89/90/
  100/120 and run the precision suite on each available arch. The suite picks
  the lowest detected capability.
- FP8 hardware paths remain gated to SM89/SM90+; software FP8 fallbacks must
  keep working (V100 path).
- **Metal parity (future):** there is no plain fused add-RMSNorm kernel in
  `attention-rs/src/metal-kernels` (only GDN's gated RMSNorm). Add
  `rmsnorm`/`add_rmsnorm` Metal kernels and dispatch from
  `attention-rs/src/norm.rs`; currently Metal falls back to add → cast →
  Candle RMSNorm → cast. The Metal PRs (#119/#511) added Metal
  `silu_and_mul`, cast/transpose, and GGUF prefill kernels — continue that
  work for decode.

---

## 6. Profiling cookbook

Nsight Systems is extracted at
`/opt/nsys/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys`.

```bash
NS=/opt/nsys/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys
# Kernel-level decode timeline (graphs disabled)
$NS profile -t cuda -o /tmp/prof --force-overwrite true \
  ./target/release/xinfer --m <model> --d 0 --batch 32 --ignore-eos \
  --max-tokens 64 --temperature 0 --disable-cuda-graph --prompts "..."

# CUDA graph replay kernel nodes
$NS profile -t cuda --cuda-graph-trace=node -o /tmp/prof_graph ...

# Aggregate kernels in a time window (sqlite at /tmp/prof.sqlite)
python3 - <<'PY'
import sqlite3
db=sqlite3.connect('/tmp/prof.sqlite'); c=db.cursor()
t0,t1=c.execute("select min(start),max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
rows=c.execute("""select s.value,k.end-k.start from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id=k.shortName where k.end > ?""",(t1-2_000_000_000,)).fetchall()
from collections import defaultdict
agg=defaultdict(lambda:[0,0])
for nm,dt in rows: agg[nm.split('<')[0][:90]][0]+=1; agg[nm.split('<')[0][:90]][1]+=dt
for nm,(n,t) in sorted(agg.items(), key=lambda x:-x[1][1])[:20]:
    print(f"{t/1e6:8.2f}ms {n:6d} {nm}")
PY
```

Ncu example (GDN state traffic):

```bash
ncu --kernel-name regex:gdn_decode_smem --launch-count 1 \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
  ./target/release/xinfer ... --disable-cuda-graph ...
```

Benchmark commands:

```bash
# decode scaling
for B in 1 8 16 32 64 128; do
  ./target/release/xinfer --m /data/Qwen3.8-27B-FP8 --d 0 --batch $B \
    --ignore-eos --max-tokens 128 --temperature 0 --prompts "Write ..."
done
# prefill (8k)
P=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*800)")
./target/release/xinfer --m /data/Qwen3.8-27B-FP8 --d 0 --batch 1 \
  --max-tokens 2 --temperature 0 --prompts "$P"
```

Gotchas: CUDA graphs hide individual kernels unless
`--cuda-graph-trace=node`; `torch` bandwidth microbenchmarks need the loop
count multiplied in (a 4 GB copy timed over N iterations reports bytes/N);
`/data` root is a stale NFS handle but subdirectories work; export
`http_proxy/https_proxy=http://127.0.0.1:7101` before cargo git fetches.

---

## 7. Validation gates for every optimization

1. `cargo build --release --features "cuda,nccl,flashinfer,cutlass"` and,
   for portable changes, `cuda,nccl` only plus a compile with
   `CUDA_COMPUTE_CAP=70/75`.
2. `bash tests/run_precision_suite.sh` — all standard probes must pass. The
   opt-in GDN FlashInfer prefill is known to fail the exactness probe
   (`--include-optin-gdn`) and stays opt-in until G3 is resolved.
3. Model matrix smoke tests: FP8 dense (4B/27B), FP8 MoE (35B-A3B), NVFP4
   (27B), GGUF Q4_K_M/IQ, and at least one LLaMA/GLM-class model if available.
4. Logits diffs (max/mean) against the pre-change binary on a fixed prompt
   set for temp-0 sensitivity, plus a short generation sanity check.
5. Benchmarks before/after at b1/b8/b32 and prefill 2k/8k/20k; report both
   eager (`--disable-cuda-graph`) and graph numbers for decode.

Keep every optimization independently revertible (env switch or small
commit) until it has passed the gates; the existing pattern is one PR per
major kernel family.

---

## 8. Suggested PR sequence

1. **GGUF quantized GEMV/MMQ** (G1) — largest single-model gap.
2. **Activation-quant fusion** (K1) plus the remaining norm/sampling
   launches (K4/K5).
3. **CUDA graphs beyond 32** (S1) — large-batch throughput.
4. **GDN decode microkernel + a/b merge** (K2/K3).
5. **Chunked GDN prefill precision/default** (G3).
6. **NVFP4 software path** (G2).
7. **Speculative/MTP tuning** (S4), then MoE (S5), multi-GPU/PD (S6).
8. **Metal fused norm** and remaining Metal parity (S8).
