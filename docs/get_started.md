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
- **Safetensors (HF layout)**: `--m <hf_id>` for cached download, or `--m <local_dir>` for offline weights + configs. `--w <local_dir>` still works as a legacy alias.
- **Safetensors (HF layout) + ISQ**: in-situ quantize into GGUF with args `--isq <q4k|q2k|q6k|...>`.
- **GGUF**: multiple loading modes:
  - `--m <local.gguf>` — load a single local GGUF file.
  - `--m <local_dir>` — auto-detect the main GGUF file in a directory (picks the largest non-mmproj `.gguf`). Multi-shard files (`*-00001-of-00007.gguf`) and auxiliary files (`mmproj-*.gguf` for vision towers) are auto-discovered.
  - `--f <local.gguf>` — load a single local GGUF file (alternative syntax).
  - `--m <hf_id> --f <remote_file.gguf>` — download a GGUF file from a Hugging Face repo.
  - `--m <hf_id> --f <subfolder>` — auto-detect all GGUF files in a HuggingFace repo subfolder (e.g. `--m unsloth/GLM-5.2-GGUF --f UD-Q2_K_XL`).
- **Vision-Language** (Qwen3-VL, Qwen3.5-VL, Gemma3, Gemma4, Mistral3-VL): require image tokens; use `--ui-server` for uploads or send image_url/base64 in the request. For GGUF multimodal models, place the `mmproj-*.gguf` vision tower file in the same directory as the main model file — it will be auto-detected.


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
- **GGUF quantized (single file)**  
  ```bash
  target/release/xinfer --m /path/model-Q4_K_M.gguf --max-model-len 65536
  ```
- **GGUF quantized (folder with auto-detection)**  
  ```bash
  target/release/xinfer --m /path/GLM-5.2-GGUF/ --ui-server
  ```
- **Remote GGUF quantized (multi-shard subfolder)**  
  ```bash
  target/release/xinfer --d 0,1,2,3 --m unsloth/GLM-5.2-GGUF --f UD-Q2_K_XL --ui-server
  ```
- **Remote GGUF quantized (single file)**  
  ```bash
  target/release/xinfer --m unsloth/Qwen3-0.6B-GGUF --f Qwen3-0.6B-Q4_K_M.gguf --ui-server
  ```
- **Embeddings** (same server; OpenAI `/v1/embeddings`)  
  ```bash
  target/release/xinfer --m Qwen/Qwen2.5-7B-Instruct  # curl -d '{"input":"hello","embedding_type":"mean"}' http://localhost:8000/v1/embeddings
  ```
- **Multimodal**  
  ```bash
  # Update image in the Chat UI
  target/release/xinfer --m Qwen/Qwen3-VL-8B-Instruct --ui-server
  ```

Common runtime knobs: `--max-model-len`, `--max-num-seqs`, `--kv-fraction` (CUDA KV share), `--cpu-mem-fold` (CPU swap ratio), `--port`, `--kvcache-dtype` (fp8/turbo8/turbo4/turbo3), `--prefix-cache`, `--prefix-cache-max-tokens`, `--ui-server`, `--batch` (perf test).

Reasoning defaults to enabled when a request omits `thinking` / `enable_thinking`. Use `--disable-reasoning` on the Rust CLI to make the default be disabled instead; explicit request values still override the server default.

## 4) Multi-rank (single node)
- **NCCL multi-GPU**  
  ```bash
  target/release/xinfer --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --max-num-seqs 2 --kv-fraction 0.5
  ```
- **Graph capture**: CUDA graph is auto-enabled with `cuda` feature. Use `--disable-cuda-graph` at runtime to skip graph capture.

## 4.1) Multi-node tensor parallelism

Distribute tensor-parallel inference across multiple machines via TCP-based NCCL bootstrap. Node 0 is the coordinator (runs scheduler + API); worker nodes run forward-only daemon loops.

**Requirements:**
- All nodes must have the same model weights available locally (same path or HuggingFace cache).
- All nodes must be reachable via TCP on the `--master-port` (default 29500) and `--master-port + 1` (forward coordination).
- Build with `--features cuda,nccl,flashinfer,cutlass` on all nodes.

**Example: 2 nodes × 4 GPUs = 8-way tensor parallelism**

```bash
# Node 0 (master, at 192.168.1.100): runs scheduler + API server
xinfer --m /data/DeepSeek-R1/ --d 0,1,2,3 \
  --num-nodes 2 --node-rank 0 \
  --master-addr 192.168.1.100 --master-port 29500 \
  --ui-server

# Node 1 (worker, at 192.168.1.101): runs forward-only daemon
xinfer --m /data/DeepSeek-R1/ --d 0,1,2,3 \
  --num-nodes 2 --node-rank 1 \
  --master-addr 192.168.1.100 --master-port 29500
```

**How it works:**
1. Node 0 generates a NCCL unique ID and distributes it to worker nodes via TCP.
2. Each node spawns local runner subprocesses with global NCCL ranks (`node_rank × local_gpus + local_rank`).
3. All ranks join a single global NCCL communicator for all-reduce / all-gather operations.
4. Node 0 broadcasts forward-pass commands to worker nodes via TCP; NCCL synchronizes the actual tensor computations.
5. Only node 0 runs the scheduler and API server; worker nodes are stateless forward engines.

**CLI flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--num-nodes` | 1 | Total number of nodes |
| `--node-rank` | 0 | This node's rank (0 = master) |
| `--master-addr` | _(required)_ | Master node's IP address |
| `--master-port` | 29500 | TCP port for NCCL bootstrap |

**NCCL environment variables** (optional tuning):
```bash
export NCCL_IB_DISABLE=1      # Disable InfiniBand if unavailable
export NCCL_SOCKET_IFNAME=eth0 # Specify network interface
export NCCL_DEBUG=INFO          # Enable NCCL debug logging
```

## 5) PD Disaggregation (prefill/decoding split)
- **PD server (prefill host, usually memory-rich)**  
  ```bash
  target/release/xinfer --pd-server --port 8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507
  ```
- **PD client (decode host)**  
  ```bash
  target/release/xinfer --server --pd-client --pd-url 0.0.0.0:8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507
  ```
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
