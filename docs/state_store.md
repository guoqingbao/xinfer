# Persistent Inference State Store

The state store persists the engine's inference state (the KV cache, the
GDN/mamba recurrent state, and the prefix-cache index) so a restart can
warm-start instead of re-prefilling, and so a-instance / cross-host
migration can move live state. It is optional: with no `--state-store` the
engine behaves exactly as before.

## Backends (selected by URL scheme)

One flag, `--state-store <URL>`, selects the backend by scheme:

| scheme | backend | notes |
|---|---|---|
| `/path` or `file:///path` | `FsStateStore` | CPU bounce buffer; always available |
| `gds:///path` or `nvme:///path` | `NvmeStateStore` | GPUDirect-Storage zero-copy DMA; needs the `gds` cargo feature + `libcufile` + a GDS-capable NVMe |
| `s3://bucket/prefix` | `S3StateStore` | object store (AWS / MinIO / Ceph / RustFS); needs the `s3` cargo feature + `rust-s3` |

The FS3 endpoint / region / credentials come from the standard `AWS_*`
environment variables (`AWS_ENDPOINT_URL` `AWS_REGION`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`); the bucket and key prefix come from the URL.

## Configuration

| env var | default | effect |
|---|---|---|
| `--state-store` (CLI) | unset | the store URL; unset = no persistence |
| `XINFER_STATE_COMPRESS` | off | zstd-compress the serialized state before write |
| `XINFER_STATE_KEY` | unset | AES-256-GCM encrypt the state; the key is SHA-256'd to 32 bytes |
| `XINFER_STATESTORE_STRICT` | off | reject a warm-load whose `version_stamp` (runtime + model + dtype + block size) does not match the current process |
| `XINFER_STATE_TTL_MS` | 7 days | prune states not accessed within this window |
| `XINFER_STATE_MAX_BYTES` | 8 GiB | space watermark; oldest states are evicted past it |
| `XINFER_PLE_NO_MMAP` | off | read the PLE n-gram table into heap instead of mmap (required on SM121-class unified-memory GPUs, where an mmap competes with model + KV for the same pool) |

## Lifecycle

- **Boot** — `LLMEngine::warm_load` lists the store, version-checks each record
  (discarding under `XINFER_STATESTORE_STRICT` on mismatch), re-seeds the
  scheduler prefix cache, and logs a load summary (count, bytes, ms).
- **Run** — the store is passive; no per-token traffic.
- **Shutdown** — `LLMEngine::checkpoint` captures the live KV + GDN + prefix
  state and writes one record, logging a save summary.
- **Prune** — at boot, `StateStore::prune` drops records older than the TTL and
  evicts LRU records past the space watermark so the store cannot fill the disk.

## Correctness notes

- The KV and GDN are captured at a step boundary (quiescent), and the GDN
  snapshot length is aligned to the same token count as the KV so the
  two never desync on restore.
- `InferenceState` equality ignores `last_accessed` (a bookkeeping timestamp),
  so round-trip tests compare logical content.
- The prefix cache is re-seeded by hash, not by physical block id, so a restore
  onto a differently-sized pool is safe.