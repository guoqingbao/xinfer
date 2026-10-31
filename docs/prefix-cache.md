# Prefix cache and CPU KV offload

Prefix cache lets xInfer reuse KV cache blocks from prior requests when a new
prompt shares a prefix. This accelerates consecutive requests with overlapping
history (for example, chat sessions that replay the same system + earlier turns).

On CUDA, cached prefix blocks displaced by GPU KV pressure can be copied to
the CPU KV swap pool and restored when that prefix is requested again. This
uses the existing swap pool; it does not offload model weights. Prefix cache is
enabled by default in the CLI, and CPU prefix offload has no separate switch.
Use `--disable-prefix-cache` to disable both reuse and this offload path.

```powershell
$env:XINFER_MOE_OFFLOAD = '0.2'
target/release/xinfer.exe --m F:/models/Qwen3.8-35B-Distill-GGUF --kvcache-dtype turbo4 --ui-server
```

The MoE variable in this example controls expert weights independently; it is
not required for prefix KV offload.

## How it works
- Finished sequences contribute full KV blocks to a global prefix cache.
- New requests find the longest cached prefix (block-aligned) and reuse those blocks.
- Remaining tokens are prefetched as usual, with KV writes continuing after the cached prefix.
- Under GPU KV pressure, unused cached GPU blocks are copied to the CPU swap
  pool in batches. Matching CPU blocks are copied back to new GPU blocks on a
  later request. If the CPU pool is full, older offloaded prefixes are dropped
  by LRU and recomputed if requested again. A small portion of CPU swap is
  reserved for live-sequence preemption.

Prefix cache is block-granular: only full KV blocks are reused. If the common
prefix ends mid-block, the tail of that block is recomputed. When a prompt is
fully cached at block boundaries, the last block is recomputed to ensure a
non-empty prefill step for correct sampling.

## Flags
- Prefix cache is **enabled by default**. Use `--disable-prefix-cache` to turn it off.
- `--prefix-cache-max-tokens <N>`: cap cache size in tokens (rounded down to block size).
- `--cpu-mem-fold <F>`: size the CUDA CPU KV swap pool as a multiple of GPU KV
  blocks (default `0.5`). Prefix offloads and live-sequence swaps share it.

If `--prefix-cache-max-tokens` is not set, defaults are:
- Normal mode: up to 100% of GPU KV blocks, reclaimed when active requests need space
- PD server: 80% of GPU KV blocks
- PD client: 50% of GPU KV blocks

These are upper bounds, not GPU memory reservations. An explicitly smaller
cache cap may evict prefixes even while GPU KV space remains free. If a CPU
copy fails or CPU swap is unavailable, that cached block is dropped and must
be recomputed on its next use. Offload requires a CUDA build; it was exercised
on a single CUDA GPU. Multi-GPU behavior has not been verified.

## Hybrid Mamba Snapshot Stride
For hybrid Mamba models (for example Qwen3.5), prefix reuse also needs a
compatible Mamba snapshot at the matched boundary.

Use environment variable `XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS` to control
sparse snapshot capture during decode (a larger stride reduces snapshot memory use):
- Default: `1` blocks
- Minimum valid value: `1` (capture every block)
- Effective snapshot boundary in tokens: `block_size * stride`

Example with default `block_size=64` and stride `8`:
- Decode snapshot boundary is every `512` tokens.
- Effective hybrid prefix reuse is aligned to the nearest captured boundary.

This setting only sparsifies decode-time snapshot capture. Prompt/prefill
snapshot capture remains dense. When KV prefix blocks are offloaded, their
Mamba/GDN snapshot identities are retained and associated with the new GPU
blocks when the KV is restored. A hybrid prefix is reusable only if a
compatible recurrent-state snapshot exists at its reuse boundary.

## Notes
- Prefix cache uses the same KV memory pool as active sequences. A larger cache
  reduces the maximum number of concurrent tokens available for new requests.
- Cached KV reuse is automatic; no `session_id` is required.
- Sliding window attention limits how much cached context is effectively used.
- `XINFER_MOE_OFFLOAD` independently controls MoE expert weights; it does not
  enable prefix KV offload. See [MoE expert offload](moe_offload.md).
- With `--disable-prefix-cache`, the prefix offload path is absent and the
  non-prefix paths follow main. With default CLI settings, prefix caching and
  pressure-triggered offload are active without an extra flag.

## CPU swap with prefix cache

When prefix cache is enabled, live sequence preemption is partial:

- Leading prefix/shared blocks stay on GPU and keep their sequence reference.
- Only the sequence-owned suffix blocks are copied to CPU swap.
- Swap-in reuses the retained GPU prefix blocks, allocates new suffix blocks,
  and copies only the suffix KV back from CPU.

This is different from prefix-cache offload. Prefix-cache offload only helps when
an evicted cached prefix is requested again later. Partial sequence swap helps
immediately under memory pressure because it avoids recomputing the active
sequence suffix after preemption, even when no future request reuses that suffix.
For hybrid Mamba/GDN models, the active recurrent state remains tied to the
swapped sequence lifecycle and is not released until the sequence finishes or is
cancelled. On CUDA, Mamba/GDN prefix snapshots use a separate two-tier cache:
device snapshots are the fast path, and snapshots evicted from device memory are
spilled to CPU so a later KV prefix hit can promote the matching recurrent state
back instead of falling back to a shorter partial hit. Metal uses unified memory,
so this CPU spill tier is disabled there. The CUDA CPU snapshot tier is 4x the
device snapshot capacity. Prompt/prefill snapshots and final decode-boundary
snapshots are protected from ordinary decode-time snapshot churn. When the CPU
snapshot tier is full, LRU eviction frees at least 10% of the tier in one batch
before accepting new GDN/Mamba snapshot offloads.

## Inspecting cache hits

The scheduler's `CPU swap used` field measures the shared CPU KV pool. An
`Offloaded ... evicted prefix cache block(s) to CPU` line confirms offload;
`CPU promoted` in an extended-prefix line confirms that saved blocks were
restored. A prefix hit alone does not prove CPU offload: the KV may have stayed
on GPU. For hybrid models, also look for `Restored mamba prefix state`.

Chat completion responses include the prefix-cache hit count under
`usage.prompt_tokens_details.cached_tokens` (OpenAI extension). The field is
omitted when no hits occurred, so existing single-turn responses keep their
shape:

```json
"usage": {
  "prompt_tokens": 499,
  "completion_tokens": 16,
  "total_tokens": 515,
  "prompt_tokens_details": { "cached_tokens": 480 }
}
```

In Python (offline batch), call `engine.get_num_cached_tokens_for_seq(seq_id)`
on the `seq_id` returned in each `GenerationOutput`.

For models that emit `<think>…</think>` reasoning blocks, responses also
include `usage.completion_tokens_details.reasoning_tokens` so clients can
attribute completion cost across reasoning vs final-answer output:

```json
"usage": {
  "prompt_tokens": 12,
  "completion_tokens": 256,
  "total_tokens": 268,
  "completion_tokens_details": { "reasoning_tokens": 192 }
}
```
