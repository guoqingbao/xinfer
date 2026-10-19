# Speculative Decoding (MTP & DFlash2)

xInfer speeds up autoregressive decode with **speculative decoding**: a small set of candidate tokens is proposed each step, then verified by the target model in a single batched forward pass. Accepted drafts advance the sequence without extra per-token decode rounds.

Two modes are supported:

| Mode | When to use | CLI |
|------|-------------|-----|
| **Built-in MTP** | Target model ships MTP prediction heads (e.g. Qwen3.5 / Qwen3.6 with MTP weights) | `--num-speculative-tokens N` |
| **DFlash2** | External DFlash2 draft model (separate safetensors checkpoint) | `--draft-model <id_or_path>` |

**Rule:** If `--draft-model` is set, xInfer uses **DFlash2** (DFlash1 checkpoints are not supported). Otherwise, `--num-speculative-tokens` enables **built-in MTP** when the target model has MTP layers.

---

## Built-in MTP

MTP (Multi-Token Prediction) uses lightweight heads bundled in the target checkpoint. Each decode step:

1. Target model produces one **anchor** token and a hidden state.
2. MTP head drafts `N` tokens autoregressively (no extra KV growth).
3. Target model verifies `[anchor, draft₀, …, draftₙ₋₁]` in one forward pass.
4. Matching prefix is accepted; GDN/Mamba state is rolled back on partial rejection (Qwen3.5 hybrid models).

### CLI

```bash
# Qwen3.5 35B with 3 speculative tokens per step
xinfer --m Qwen/Qwen3.5-35B-A3B --d 0,1 \
  --num-speculative-tokens 3 --ui-server
```

Typical values: **3–7** draft tokens. Higher values increase acceptance variance and KV verify width.

### Requirements

- Target weights must include MTP layers (`mtp.*` or GGUF `nextn.*` tensors).
- Model types with MTP support today: **Qwen3.5**, **Qwen3.5 MoE**, **Qwen3-VL** (Qwen3.5 text backbone).
- Recommended build: `./build.sh --release --features cuda,nccl,flashinfer,cutlass`

---

## DFlash2

[DFlash2](https://github.com/z-lab/dflash) uses a **separate draft model** that consumes projected intermediate hidden states from selected target layers. Draft tokens are chosen with a top-k candidate lattice (`selector_top_k` in the draft config).

### CLI

```bash
# Qwen3.8 target + DFlash2 draft (HuggingFace id or local path)
xinfer --m Qwen/Qwen3.8-... --d 0,1 \
  --draft-model <your-dflash2-draft-repo-or-path> \
  --num-speculative-tokens 7 --ui-server
```

`--draft-model` accepts either:

- A **HuggingFace model id** (downloads `config.json` + safetensors), or
- A **local directory** containing `config.json` and `model.safetensors` (or sharded index).

`--num-speculative-tokens` is the number of draft tokens proposed per step (excluding the anchor). If omitted, xInfer uses the draft model config (`block_size - 1`).

Draft weights must be **safetensors** (GGUF draft models are not supported). The checkpoint must be **DFlash2** (`architectures` contains `DFlash2` or `dflash_config.selector_top_k` is set).

### Requirements

- Target: **Qwen3.5**, **Qwen3.5 MoE**, **Qwen3-VL**, or **Qwen3.8** (dense) today.
- Draft: matching-family **DFlash2** checkpoint.
- Recommended build: `cuda,nccl,flashinfer,cutlass` (CUDA graphs optional; disable with `--disable-cuda-graph` for debugging).

---

## Python API

```python
from xinfer import Engine, EngineConfig

# Built-in MTP
cfg = EngineConfig(
    model_id="Qwen/Qwen3.5-35B-A3B",
    num_speculative_tokens=3,
)

# DFlash2 (draft_model enables DFlash2 instead of MTP)
cfg = EngineConfig(
    model_id="Qwen/Qwen3.8-...",
    draft_model="<dflash2-draft-id-or-path>",
    num_speculative_tokens=7,
)

engine = Engine(cfg, "bf16")
```

---

## Flags reference

| Flag | Description |
|------|-------------|
| `--num-speculative-tokens N` | Draft tokens per decode step. Enables **MTP** when the target has MTP heads and `--draft-model` is unset. With `--draft-model`, sets DFlash2 draft width (optional; defaults from draft config). |
| `--draft-model <id_or_path>` | External **DFlash2** draft model. Enables DFlash2 and disables built-in MTP. |

**Removed:** `--mtp` (use `--num-speculative-tokens`), `--draft-model-id` / `--draft-model-path` (use `--draft-model`).

### Speculative-decoding env vars

| Var | default | effect |
|---|---|---|
| `XINFER_SPEC_FF` | off | fast-forward decode: after the sampled base token, append the full grammar-forced run deterministically (no per-token sampling) |
| `XINFER_SPEC_PARALLEL_DRAFT` | off | skip the single-sequence gate so concurrent sequences each draft (batched verify) |
| `XINFER_DFLASH_PARALLEL_SLOTS` | 2 | cap on concurrent DFlash drafter slots (sequences beyond it fall back to plain decode) |
| `XINFER_SPEC_CONTEXT_WINDOW` | 4096 | DFlash projected-context window (0 = unbounded full history) |
| `XINFER_SPEC_GRAPH` | on | capture the DFlash verify into a CUDA graph (0 = force the eager draft) |
| `XINFER_VOB_SAMPLING` | off | sample against a VOB bitset mask (8x less data than the F32 path) via the fused `sample_cuda_vob` kernel |

### Drafting modalities

- **MTP (built-in):** the target model's MTP head drafts K tokens, verified against the target logits.
- **DFlash (external):** a separate drafter model. DFlash2 checkpoints carry the grouped conv + candidate-selector lattice; DFlash1 has neither and drafts via plain argmax (`has_v2_components` feature-detects). YARN rope scaling keeps the drafter in the base model's position space.
- **Fast-forward (`XINFER_SPEC_FF`):** the lowest-priority path; appends the grammar-forced run after the base token.

The grammar mask on the drafts/verify runs on the CPU (VOB or F32 allow-mask offloaded to the fused sampler). The GPU-resident PDA table (fused `fused_project`/`fused_sample`) draft masking and the matcher-gated single-writer ingress are not present on this branch; the grammar mask here is the CPU path.

---

## Tips & limitations

- **Throughput:** Speedup depends on draft acceptance rate and model size. Monitor logs for `MTP Stats` / `DFlash2 Stats` acceptance summaries.
- **Memory:** Verify passes append up to `N+1` tokens per step; ensure KV budget via `--kv-fraction` and `--max-num-seqs`.
- **Hybrid models:** Qwen3.5 GDN layers require state rollback on rejected drafts; this is handled automatically.
- **Batching:** Speculative decode is optimized for `batch_size=1` decode steps; larger batches use a batched DFlash2 path.
- **PD disaggregation:** Do not combine speculative decode with PD client/server modes on hybrid Mamba models.

---

## Quick test

```bash
# 1. Health check
curl -s http://localhost:8000/v1/models

# 2. Short completion (MTP or DFlash2 server)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

See also [test_model.md](./test_model.md) for full model validation workflows.

### Data-movement cardinality
- F32 path: `batch * vocab` F32 values for the mask (the ~4 bytes/token).
- VOB path: `batch * ceil(vocab/32)` U32 words (the ~8x less data).
- The mask/vob is built on the GPU (no CPU round-trip for the VOB->allow expansion,
  which the attention-rs `vob_to_allow` GPU kernel does).

### Trigger
- Every decode step where a grammar is active + the sampling strategy is top-k/top-p.

### Sync/async
- Synchronous GPU kernel launch (the fused sampler runs on the CUDA stream, no host
  sync until the sample is consumed).

### Data lifecycle
- The mask/VOB is per-step (the freed after the sample). The grammar FSM state is
  persistent across steps (owned by the guidance matcher, not the spec-decode path).

### Locking
- The sampler is locked (the `fast_sampler` mutex in the logits processor, the one
  fused-sampler instance per runner).

### Citations / references
- The VOB (the vector of bits) grammar mask: the llguidance `SimpleVob` (the
  <https://github.com/guidance-ai/llguidance>, the compact allowed-token set).
- The fused masked sampling: the attention-rs `sample_cuda_masked` / `sample_cuda_vob`
  kernels (the local <https://github.com/guoqingbao/attention.rs>).

---

## 2. DFlash1 support

### What it does
DFlash2 checkpoints carry the grouped dynamic conv + the candidate-selector lattice.
DFlash1 checkpoints have neither. The `has_v2_components()` feature-detects the v2
components; when absent, the drafter falls back to plain-argmax drafting (the no
selector walk).

 DFlash1 + DFlash2 load now (the prior code bailed on non-DFlash2).

### Trigger
- At drafter load (the `DFlashDrafter::new`), when `--draft-model` points at a DFlash1
  checkpoint.

### Citations / references
- The DFlash drafter: the z-lab DFlash (the <https://github.com/z-lab/dflash>), the
  draft-model speculative decoding.

---

## 3. YARN rope scaling

### What it does
The drafter's rope embeddings must live in the same position space as the base model, or
long-context drafts drift + the acceptance rate collapses. `apply_static_rope_scaling`
feeds the CLI `--yarn-scaling-factor` through the existing dynamic-YARN machinery (the
`derive_yarn_parameters` theta adjustment + the mscale) into the drafter config, so the
drafter's rope matches the base. MTP needs no change (it inherits the base Config).

### Trigger
- At drafter init, when `--yarn-scaling-factor` is set.

### Citations / references
- YARN: "YaRN: Efficient Scaling of Transformer Models via Position Interpolation",
  arXiv:2309.00071 (the rope position interpolation for long context).

---

## 4. `XINFER_SPEC_FF` fast-forward

### What it does
The lowest-priority decode path. After the sampled base token, the full grammar-forced
(ff) run is appended deterministically (no per-token sampling), because the grammar
has already forced those tokens. The next draft anchors on the last ff token.

### Diagram
```
  base token (sampled)  -->  ff run (grammar-forced, deterministic)  -->  next draft anchors here
  [sampled]                [ff1, ff2, ... ffN]                        [next base]
```

### Trigger
- `XINFER_SPEC_FF=1` + no MTP/DFlash active + not a prefill step.

### Data-movement cardinality
- The ff run is appended to the sequence's token (the N forced tokens, no extra KV
  beyond the normal append).

### Citations / references
- The grammar-forced fast-forward: the llguidance `compute_ff_tokens` (the deterministic
  continuation the grammar forces, the <https://github.com/guidance-ai/llguidance>).

---

## 5. The spec-ff stats

The per-seq speculative report (`spec_stats.rs`) counts the ff continuations alongside the
MTP/DFlash propose/accept, so the end-of-sequence report shows how many tokens came from
the fast-forward path.

## 6. The PDA deps

The attention-rs / pushdown-rs / llguidance revs are bumped to the PDA-capable set (the
`dpda` feature on llguidance). The PDA deps are data-only on this branch: the PDA machine
is present, but the PDA masking path is not wired here.

## 7. The dynamic draft-length lookup + the per-drafter acceptance profile (the §2 session work)

This session added the QoS-driven dynamic speculative draft length + the per-drafter
acceptance tracking, on top of the existing VOB/masked sampling + the DFlash1 + the
YARN + the `XINFER_SPEC_FF` fast-forward.

### The `DynamicSdLookup` (the dense `batch_size -> K` table)

The `DynamicSdLookup` (the `speculative/dynamic_sd.rs`) is the dense batch-size-to-K
table for the dynamic speculative draft length. The config is a list of
`(range_start, range_end, k)` triples (the `num_speculative_tokens_per_batch_size`);
the lookup expands them into a dense 1-indexed table where index `batch_size` stores
the K to use for that runtime batch size. Gaps between ranges are filled by carrying
forward the previous K; the tail after the last range is filled by carrying forward
the last K; every K is clamped to the `max_speculative_tokens`. The small batch maps
to the large K (the latency win); the large batch maps to the small K (the
throughput win). The `DynamicSdLookup::build` returns `None` when the config is empty
or invalid (no dynamic SD, the static draft length).

The `dynamic_sd_lookup` test proves the carry-forward gap fill (the golden: the
`[(1,16,3),(32,128,2)]` schedule maps batch 17-31 to K=3, the batch 32-128 to K=2)
+ the clamp (the K above the max is clamped) + the invalid-config rejection (the
empty / the no-start-at-1 / the overlapping ranges all return `None`).

### The `DrafterProfile` (the per-drafter acceptance tracking)

The `DrafterProfile` (the `qos_histogram.rs`) tracks the per-drafter acceptance-rate
EMA + the per-batch curve for each drafter kind (the MTP, the DFlash, the chained).
The MTP (the model's own head) has a stable acceptance rate across batch size; the
DFlash (the separate drafter) has a batch-varying acceptance rate. The
`DrafterProfile::select` picks the best drafter for the workload class (the
coding-agent traffic favors the DFlash, the chatbot traffic favors the MTP, the
mixed traffic favors the chained). The is the signal for the dynamic drafter
selection (the §2 business logic: serve right drafter for the right workload).

### The sampling-override actuator

The `AdaptivePlan.sampling` override (the per-class top-k/top-p width) is applied at
the runner's `sample` (the `last_sampling_override` field on the `Scheduler`, set in
the `prepare_step_view`). The QoS output-length CDF drives the width: the short-output
output class gets the tight top-k (the speed), the long-output class gets the wide
top-p (the quality). When the override is `None` (no QoS), the model's
configured sampling is used unchanged.

### The drafter-chaining seam

The MTP `draft_tokens_gpu` returns the `(draft_tokens, final_hidden)` — the
`final_hidden` (the post-draft hidden state) is the anchor the DFlash drafter
consumes as its `target_hidden`. The chaining seam (the `mtp.rs` `run_mtp_decode`
exposing the `final_hidden`) makes the linear-then-parallel drafter chain possible
(the MTP's stable-acceptance draft anchors the DFlash's batch-varying draft, the
combined K1+K2 verified in a single pass).

## Known gaps (the production-grade todos)

- **The VOB-bitset sampling is wired on this branch (the §2.3 session work).** The
  `build_vob_words` (the per-seq VOB bitset, the `guided_decoding.rs`) + the
  `sample_with_vob` (the fused bitwise-AND sampler, the `logits_processor.rs`) + the
  `XINFER_VOB_SAMPLING` env gate (the `env.rs`) are present. The VOB path is selected
  when the grammar is active (the `guided_positions` non-empty) + the `XINFER_VOB_SAMPLING`
  env is set; otherwise the F32 masked-sampling path (`sample_with_strategy_masked`) runs
  unchanged. The `test_vob_is_8x_less_data_per_token` test proves the VOB bitset is 8×
  less data per token than the F32 mask (the 32 bits/token vs the 4 bits/token).
- **The fast-forward (`XINFER_SPEC_FF`) is grammar-only.** It appends the grammar-forced
  run (`ff_tokens`) after the sampled base token. For non-grammar sequences there is no
  forced run, so the fast-forward is a no-op (the base token only). It is most useful for
  tool-call / structured-output decoding.
- **The PDA draft validation is CPU-only on this branch.** The `pda_validate_draft` (the
  CPU pushdown-automaton walk) is present for transient recovery. The GPU-resident PDA
  table (the fused `fused_project` / `fused_sample` kernels) + the matcher-gated
  single-writer ingress are not present on this branch.

## Actor integration (the `SchedD` speculative paths)

On the actor branch, the speculative decode paths are driven by `SchedD::handle_step`:

```
SchedD::handle_step:
  a. scheduler.schedule()
       // For MTP/DFlash: pre_allocate_mtp_blocks(scheduled_ids, num_spec + 1)
       //   Extends block_table to cover the verify positions
  b. Forward pass selection:
       use_mtp    = mtp_enabled && !use_dflash && !is_prefill && parallel_ok
       use_dflash = dflash_enabled && !is_prefill && dflash_ok
       use_spec_ff = !use_mtp && !use_dflash && env::spec_ff() && !is_prefill
  c. fwd.run_mtp(sequences) / fwd.run_dflash(sequences) / fwd.run_spec_ff(sequences)
       // Each sends the appropriate MessageType to the 4 GPU runners
       // Returns Vec<Vec<u32>> (accepted tokens per sequence)
  d. fwd.gate_commit(seq_ids, outputs, use_spec_ff)
       // Commits the produced tokens to the grammar FSM
       // Returns the gated prefix (FSM-legal tokens only)
  e. scheduler.postprocess(scheduled_ids, gated)
       // Appends gated tokens to sequences
  f. Delivery loop (StreamD)
  g. clear_finished + check_canceled
```

The `FwdProtocol` methods for speculative paths:
- `run_mtp(Vec<DecodeSequence>) -> Response<Result<Vec<Vec<u32>>, String>>`
- `run_dflash(Vec<DecodeSequence>) -> Response<Result<Vec<Vec<u32>>, String>>`
- `run_spec_ff(Vec<DecodeSequence>) -> Response<Result<Vec<Vec<u32>>, String>>`
- `gate_commit(Vec<usize>, Vec<Vec<u32>>, bool) -> Response<Vec<Vec<u32>>>`
- `fetch_spec_seq_stats(usize) -> Response<SpecSeqStatsData>` (the runner-fetch primitive,
  still available for a direct query)

The per-sequence speculative stats (`SpecSeqStatsData`) are captured by the `SchedD` at
sequence finish (the capture-before-drop: the `SchedD` pulls the stats from the runner into
its own `spec_stats` map BEFORE the runner's `Finished` message drops them), then surfaced to
the engine via `SchedMessage::GetSpecStats` + cached in the engine's `seq_spec_stats` map for
the server's end-of-sequence report. This avoids the race where a late fetch would read an
already-dropped runner-side stat.
