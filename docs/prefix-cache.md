# Prefix Cache (KV Reuse)

Prefix cache lets xInfer reuse KV cache blocks from prior requests when a new
prompt shares a prefix. This accelerates consecutive requests with overlapping
history (for example, chat sessions that replay the same system + earlier turns).

## How it works
- Finished sequences contribute full KV blocks to a global prefix cache.
- New requests find the longest cached prefix (block-aligned) and reuse those blocks.
- Remaining tokens are prefetched as usual, with KV writes continuing after the cached prefix.

Prefix cache is block-granular: only full KV blocks are reused. If the common
prefix ends mid-block, the tail of that block is recomputed. When a prompt is
fully cached at block boundaries, the last block is recomputed to ensure a
non-empty prefill step for correct sampling.

## Flags
- Prefix cache is **enabled by default**. Use `--disable-prefix-cache` to turn it off.
- `--prefix-cache-max-tokens <N>`: cap cache size in tokens (rounded down to block size).

If `--prefix-cache-max-tokens` is not set, defaults are:
- Normal mode: ~50% of GPU KV blocks
- PD server: ~75% of GPU KV blocks
- PD client: ~35% of GPU KV blocks

## Hybrid Mamba Snapshot Stride
For hybrid Mamba models (for example Qwen3.5), prefix reuse also needs a
compatible Mamba snapshot at the matched boundary.

Use environment variable `XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS` to control
sparse snapshot capture during decode (larger stride side usefull for limited GPU memory):
- Default: `1` blocks
- Minimum valid value: `1` (capture every block)
- Effective snapshot boundary in tokens: `block_size * stride`

Example with default `block_size=64` and stride `8`:
- Decode snapshot boundary is every `512` tokens.
- Effective hybrid prefix reuse is aligned to the nearest captured boundary.

This setting only sparsifies decode-time snapshot capture. Prompt/prefill
snapshot capture remains dense.

## Notes
- Prefix cache uses the same KV memory pool as active sequences. A larger cache
  reduces the maximum number of concurrent tokens available for new requests.
- Cached KV reuse is automatic; no `session_id` is required.
- Sliding window attention limits how much cached context is effectively used.

## Inspecting cache hits

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
