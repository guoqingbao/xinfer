# Workspace budget: reserve only what is actually allocated

The KV-cache allocator subtracts a *workspace reserve* from the available
VRAM before sizing the KV pool:

```
GPU Memory Budget: 3.85 GB available → 0.25 GB workspace reserve (SplitK 48M + Transient 80M) → 3.60 GB for caches
```

That reserve has to match what the runtime actually allocates. When it is too
large the KV pool shrinks for no reason; when it is too small the first
prefill chunk fails with `CUDA_ERROR_OUT_OF_MEMORY`.

This document describes two defects in that accounting, both of which reserve
memory for a workspace that is never created.

## Defect 1: the CUTLASS workspace was reserved unconditionally

`compute_workspace_budget()` reserved 512 MiB whenever the crate was built
with the `cutlass` feature:

```rust
let cutlass_bytes: u64 = if cfg!(feature = "cutlass") {
    512 * 1024 * 1024
} else {
    0
};
```

`cfg!` is a compile-time constant, so this reserved the workspace for every
build that enabled the feature — including:

- F16 models, which never dispatch to a CUTLASS kernel at all;
- quantized models on GPUs below SM90, where the CUTLASS path is not taken;
- models whose quantized linears are served by a different kernel.

The same condition gated the MoE activation pool, so that pool was
over-estimated for the same reason.

### Fix

Reserve the workspace only when the loaded model actually dispatches to the
CUTLASS kernels, that is: the checkpoint is FP8 / NVFP4 / MXFP4 quantized
**and** the device is SM90 or newer.

## Defect 2: the FlashInfer workspace was reserved for a backend that never started

Two places independently decide whether FlashInfer is usable:

- `KVCacheAllocator::uses_flashinfer_workspace` decides whether to reserve
  the 640 MiB FlashInfer workspace;
- `core/runner.rs::skip_flashinfer_init` decides whether to actually
  initialize the backend.

The two lists differed. The runner applied a GQA group size check that the
allocator did not:

```rust
// FlashInfer prefill/decode kernels only support specific GQA group
// sizes (see attention-rs flashinfer.rs); e.g. Qwen3.8-Flash-Next has
// 24 Q heads / 2 KV heads (group 12) which is unsupported.
```

So a model with an unsupported group size — the Qwen3.8-Flash-Next geometry is
the documented example — would reserve 640 MiB that no allocation ever used.

### Fix

Extract the predicate into one shared function,
`kvcache_allocator::flashinfer_gqa_group_supported`, and call it from the
allocator, from the runner, and from the acceleration report. A single
predicate cannot drift.

## Impact

On a build with `cuda,nccl,flashinfer,cutlass`, the two defects together
could reserve up to 1.15 GB of VRAM for workspaces that are never allocated.
On a 16 GB card that is 7% of the device, taken directly out of the KV pool.

For the Qwen3.8-27B-NVFP4 deployment that motivated this change (4x V100
16 GB, `--kv-fraction 0.6`, `--prefill-chunk-size 4096`), the savings from
not reserving the CUTLASS workspace on SM70 let the KV pool grow, and the
all-reduce output buffer that caused the original `CUDA_ERROR_OUT_OF_MEMORY`
fits without shrinking the prefill chunk further.

## Verification

The new unit tests cover:

- the GQA group predicate for a supported and an unsupported geometry;
- the replicated-KV case, where the shard count exceeds the KV head count;
- that the CUTLASS workspace is not reserved for an unquantized model;
- that it is not reserved when the `cutlass` feature is absent.

The GQA predicate is exercised against the same limits the FlashInfer kernels
use: group sizes `{1,2,3,4,5,6,8,16,32,64}`.

## Related work

The same "reserve only what is allocated" principle appears in
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), which introduced
`VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT=1` to remove a duplicate copy of the
packed weights and reported the resulting change as:

| Per-GPU measurement | Before | After |
|---|---:|---:|
| Model loading | 10.07 GiB | 6.47 GiB |
| Available KV budget | 10.57 GiB | 12.68 GiB |
| Logical KV capacity | 922,965 tokens | 1,107,285 tokens |

That change removes a redundant weight copy; the two changes here remove
workspace reservations that were never used at all.
