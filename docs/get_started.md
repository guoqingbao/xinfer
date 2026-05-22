# Get Started

This guide walks through building and running xInfer across CUDA/Metal, different model formats, multi-rank, PD Disaggregation, and OpenAI-compatible APIs. Commands assume repo root.

## Build & features
- **Backends**: `--features cuda[,nccl,graph,flashinfer,cutlass]` or `--features metal`. CPU-only is supported but slow.
- **Quant/accel toggles**: `--kvcache-dtype fp8|turbo8|turbo4|turbo3` (KV cache quantization), `flashattn` or `flashinfer` (Ampere+), `--prefix-cache` (prefix KV reuse).
- **Python bindings**: add feature `python` when building wheels (`./build.sh --features python`).

### Build (CUDA)
```shell
# Remove `nccl` for single-gpu usage
# Remove `flashattn`, `flashinfer` and `cutlass` for V100 or older hardware
./build.sh --release --features cuda,nccl,flashinfer,cutlass
```

### Build (Metal)
```shell
cargo build --release --features metal
```

## Model formats
- **Safetensors (HF layout)**: `--m <hf_id>` for cached download, or `--w <local_dir>` for offline weights + configs.
- **Safetensors (HF layout) + ISQ**: in-situ quantize into GGUF with args `--isq <q4k|q2k|q6k|...>`.
- **GGUF**: `--f <gguf_file>`; no configs needed.
- **Vision-Language** (Qwen3-VL, Gemma3, Mistral3-VL): require image tokens; use `--ui-server` for uploads or send image_url/base64 in the request.


## 3) Run patterns (single host)
- **CUDA text model (chat/server)**  
  ```bash
  target/release/xinfer --m Qwen/Qwen2.5-7B-Instruct --max-model-len 131072 \
    --kv-fraction 0.6 --ui-server
  ```
- **Metal (Mac) text model**  
  ```bash
  target/release/xinfer --m meta-llama/Llama-3-8b --max-model-len 32768 --ui-server
  ```
- **GGUF quantized**  
  ```bash
  target/release/xinfer --f /path/model-Q4_K_M.gguf --max-model-len 65536  ```
- **Embeddings** (same server; OpenAI `/v1/embeddings`)  
  ```bash
  target/release/xinfer --m Qwen/Qwen2.5-7B-Instruct  # curl -d '{"input":"hello","embedding_type":"mean"}' http://localhost:8000/v1/embeddings
  ```
- **Multimodal**  
  ```bash
  # Update image in the Chat UI
  target/release/xinfer --m Qwen/Qwen3-VL-8B-Instruct --ui-server  ```

Common runtime knobs: `--max-model-len`, `--max-num-seqs`, `--kv-fraction` (CUDA KV share), `--cpu-mem-fold` (CPU swap ratio), `--port`, `--kvcache-dtype` (fp8/turbo8/turbo4/turbo3), `--prefix-cache`, `--prefix-cache-max-tokens`, `--ui-server`, `--batch` (perf test).

Reasoning defaults to enabled when a request omits `thinking` / `enable_thinking`. Use `--disable-reasoning` on the Rust CLI to make the default be disabled instead; explicit request values still override the server default.

## 4) Multi-rank (single node)
- **NCCL multi-GPU**  
  ```bash
  target/release/xinfer --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --max-num-seqs 2 --kv-fraction 0.5
  ```
- **Graph capture**: CUDA graph is auto-enabled with `cuda` feature. Use `--disable-cuda-graph` at runtime to skip graph capture.

## 5) PD Disaggregation (prefill/decoding split)
- **PD server (prefill host, usually memory-rich)**  
  ```bash
  target/release/xinfer --pd-server --port 8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507  ```
- **PD client (decode host)**  
  ```bash
  target/release/xinfer --server --pd-client --pd-url 0.0.0.0:8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507  ```
- Same weights/config on both ends; Local IPC used automatically on same node CUDA, TCP when `--pd-url` is set. Monitor logs for transfer and swap events.

## Prefix cache
- Enabled by default (CUDA/Metal). Disable with `--disable-prefix-cache`. Prefix reuse is automatic; no `session_id` required.
- Use `--prefix-cache-max-tokens` to cap the cache size (rounded down to block size).
- Tune `--max-model-len`, `--kv-fraction`, `--cpu-mem-fold`; avoid overcommitting KV or cache will swap/evict.

## APIs (OpenAI-style)
- Chat: `POST /v1/chat/completions` (supports `stream=true`, images for VL models).
- Embeddings: `POST /v1/embeddings` (`embedding_type=mean|last`, `encoding_format=float|base64`).
- Models: `GET /v1/models`; Usage: `GET /v1/usage?session_id=...`.
- UI: add `--ui-server` to expose the built-in web UI on port 8001.

## Troubleshooting & tuning
- Use `--log` to view loading/progress; watch for “swap” messages (KV pressure).
- If OOM on Metal, lower `--max-model-len` and batch; on CUDA, reduce `--kv-fraction` or `--max-num-seqs`.
- For GGUF/ISQ, keep `--max-num-seqs` moderate to avoid bandwidth bottlenecks; `--kvcache-dtype fp8` is supported on all CUDA GPUs (SM70+) and Metal.
- Use the chat logger to monitor detailed interactions between client and xInfer.

```shell
# Log into files (in folder ./log)
export XINFER_CHAT_LOGGER=1
```
