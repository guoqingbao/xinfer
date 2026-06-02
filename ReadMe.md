<p align="center">
  <img src="logo.svg" alt="xInfer" width="400"><br>
  <b>Blazing-fast LLM inference in pure Rust.</b> No PyTorch. No Python runtime. Just fast, portable, production-ready inference.<br>
  <a href="./ReadMe.md">English</a> | <a href="./ReadMe-CN.md">简体中文</a>
</p>

---

## ✨ Why xInfer?

| | Feature | Details |
|---|---|---|
| **0️⃣** | Zero Python dependencies | Pure Rust backend — no PyTorch, no CUDA Python bindings |
| **⚡** | Fast | Native Flash Attention, FlashInfer, CUDA Graphs, continuous batching, prefix caching, PD disaggregation. Up to **181 tok/s** decode for `30B+` models on consumer GPUs |
| **🪶** | Tiny footprint | Core scheduling + attention logic in **< 5 000 lines** of Rust |
| **🌍** | Cross-platform | CUDA (Linux/Windows), Metal (macOS). Same binary, same API |
| **🏭** | Production-ready | OpenAI/Anthropic-compatible APIs, built-in ChatGPT-style Web UI, MCP tool calling, structured outputs, embedding + tokenizer endpoints |
| **🗜️** | Aggressive KV compression | TurboQuant (`2–4 bit` KV cache) extends context up to **4.3×** with minimal quality loss. Run `30B+` MoE models with **millions of context** on single 24/32 GB GPUs |
| **🔥** | V100 + NVFP4 | First-ever NVFP4 + low-bit KV cache on V100 — no hardware FP4 needed, coherent output on legacy GPUs |
| **🐍** | Lightweight Python bindings | Optional PyO3 wheel when you need a Python entry point |

---

## 🚀 Quick Start

### 📦 Install

**Option 1 — Install DEB or Python package**
```bash
curl -sSL https://guoqingbao.github.io/xinfer/install.sh | bash
```

**Option 2 — npm**
```bash
npm install -g xinfer-ai
```

---

### ▶️ Run

**Using HuggingFace Model ID:**
```bash
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4 --ui-server
```

**Using local model path:**
```bash
xinfer --w /home/Qwen3.6-35B-A3B --d 0,1 --ui-server
```

**Python usage:**
```bash
# python3 -m xinfer.chat
python3 -m xinfer.server --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4 --ui-server
```

> **Tip:** Open `http://IP:8001` for the built-in chat UI, or use `http://IP:8000/v1/` as your API `Base URL`.

---

### 🗜️ KV Cache Compression

Add `--kvcache-dtype` to compress KV cache and extend context length:

| Flag (`--kvcache-dtype`) | Compression | Quality | GPU Requirement |
|---|---|---|---|
| _(default)_ | 1× (BF16) | Baseline | All |
| `fp8` | **2×** | Near-lossless | SM70+ / Apple M1 |
| `turbo8` | **2.6×** | 79–100% throughput | SM70+ / Apple M1|
| `turbo4` | **3.7×** | Best balance | SM70+ / Apple M1|
| `turbo3` | **4.7×** | Max compression | SM70+ |

---

## 📈 Performance

> Tested on **V100-32G**, **A100-40G**, **Hopper-80G** and **RTX 5090**

| Model | Format | Size | Decoding Speed |
|---|---|---|---|
| Ministral-3-3B (**Multimodal**) | ISQ (BF16→Q4K) | 3B | **193.67** tokens/s |
| Qwen3-VL-8B-Instruct (**Multimodal**) | Q8_0 | 8B | **112.51** tokens/s |
| Llama-3.1-8B | ISQ (BF16→Q4K) | 8B | **133.10** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **139.25** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **77.48** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **46.02** tokens/s |
| **Qwen3-30B-A3B** | NVFP4 | **30B (MoE)** | **181.59** tokens/s (**RTX 5090**) |
| **Qwen3-30B-A3B** | NVFP4 | **30B (MoE)** | **72.86** tokens/s (**V100, Software FP4**) |
| **Qwen3.5-27B** (**Multimodal**) | Q4_K_M | **27B (Dense)** | **49.33** tokens/s |
| **Qwen3.5-27B/Qwen3.6-27B** | FP8 | **27B (Dense)** | **45** tokens/s (**Hopper**) |
| **Qwen3.6-35B-A3B** (**Multimodal**) | FP8 | **35B (MoE)** | **110** tokens/s (**Hopper**) |
| **GLM4.7 Flash** | NVFP4 | **30B (MoE)** | **79** tokens/s (**Hopper, Software FP4**) |
| **Gemma4-31B** | ISQ (BF16→Q4K) | **31B (Dense)** | **47** tokens/s (**Hopper**) |
| **Gemma4-26B-A4B** | NVFP4 | **26B (MoE)** | **137.23** tokens/s (**RTX 5090**) |
| **MiniMax-M2.5** | NVFP4 | **229B (MoE)** | **64.50** tokens/s (**Hopper, Software FP4, TP=2**) |

<details>
<summary><b>Apple Silicon (M4)</b></summary>

| Model | Batch Size | Output Tokens | Time (s) | Throughput (tokens/s) |
|---|---|---|---|---|
| Qwen3-0.6B (BF16) | 128 | 63488 | 83.13s | 763.73 |
| Qwen3-0.6B (BF16) | 32 | 15872 | 23.53s | 674.43 |
| Qwen3-0.6B (BF16) | 1 | 456 | 9.23s | 49.42 |
| Qwen3-4B (Q4_K_M) | 1 | 1683 | 52.62s | 31.98 |
| Qwen3-8B (Q2_K) | 1 | 1300 | 80.88s | 16.07 |
| Qwen3.5-4B (Q3_K_M) | 1 | 1592 | 69.04s | 23.06 |
| Qwen3.5-2B (NVFP4) | 1 | 1883 | 60.76s | 30.99 |
| Qwen3.5-2B (NVFP4) | 2 | 3942 | 81.96s | 48.10 |

</details>

[Full benchmarks →](docs/performance.md)

---

## 🧠 Supported Models

* ✅ LLaMa (LLaMa2, LLaMa3, **LLaMa4**, IQuest-Coder)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Qwen2/Qwen3 MoE
* ✅ Qwen3 Next
* ✅ Qwen3.5/3.6 Dense/MoE (27B, 35B, 122B, 397B, Multimodal model)
* ✅ Mistral v1, v2
* ✅ Mistral-3-VL Reasoning (3B, 8B, 14B, Multimodal model)
* ✅ GLM4 (0414, **Not ChatGLM**)
* ✅ GLM4 MoE (4.6/4.7)
* ✅ GLM4.7 Flash
* ✅ DeepSeek V3/R1/V3.2
* ✅ Phi3 / Phi4 (Phi-3, Phi-4, Phi-4-mini, etc.)
* ✅ Gemma3/**Gemma4** (Multimodal model)
* ✅ Qwen3-VL (Dense, Multimodal model)
* ✅ MiroThinker-v1.5 (30B, 235B)

**Formats:** Safetensors (BF16, `FP8-blockwise`, GPTQ, AWQ, MXFP4, `NVFP4`) | GGUF (all quant types) | `ISQ` (on-the-fly quantization)

---

### TurboQuant KV Cache — Run 30B+ Models on Consumer GPUs

TurboQuant compresses KV cache to 2–4 bits via Walsh-Hadamard transform rotation + per-head absmax quantization. Max context tokens with `turbo4`:

| Model | KV budget | BF16 | turbo4 | Gain |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** (NVFP4) | 7 GB (24 GB GPU) | 700k | **2.7M** | **3.9×** |
| | 15 GB (32 GB GPU) | 1.5M | **5.8M** | **3.9×** |
| **Qwen3.6-27B** (FP8) | 7 GB | 112k | **434k** | **3.9×** |
| | 15 GB | 240k | **930k** | **3.9×** |
| **Qwen3-30B-A3B** (Q4_K_M) | 7 GB | 74k | **281k** | **3.8×** |
| | 15 GB | 160k | **602k** | **3.8×** |
| **Gemma4-26B-A4B** (NVFP4) | 7 GB | 32k | **125k** | **3.9×** |
| | 15 GB | 70k | **271k** | **3.9×** |

> Hybrid models (Qwen3.6) have fewer full attention layers, making TurboQuant especially effective. MLA models (DeepSeek, GLM4.7 Flash) use `fp8` instead. The KV budget in the table is the theoretical maximum; actual usage can only utilize up to 90% of the KV budget (`--kv-fraction 0.9`), leaving room for runtime and batching buffers.

```bash
# 35B MoE on single 24/32 GB GPU
xinfer --m unsloth/Qwen3.6-35B-A3B-NVFP4 --kvcache-dtype turbo4

# Production precision
xinfer --m Qwen/Qwen3.6-35B-A3B-FP8 --kvcache-dtype fp8

# 27B Dense + turbo4
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4

# 30B MoE GGUF + turbo4
xinfer --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kvcache-dtype turbo4

# Metal/MacOS
xinfer --m unsloth/Qwen3.5-4B-GGUF --f Qwen3.5-4B-Q4_K_M.gguf
```

---

## 📘 Usage
> For **Python installaion**, running model with `python3 -m xinfer.server` 

> For Docker builds, refer to [**Run xInfer in Docker →**](docs/docker.md)

### Running Models

> **Tip:** By default, xInfer starts an OpenAI-compatible API server at `http://localhost:8000`. Add `--ui-server` to also launch the built-in ChatGPT-style Web UI at `http://localhost:8001`.

```bash
# FP8 model (sm90+ with cutlass) + web UI
xinfer --m Qwen/Qwen3.6-27B-FP8 --ui-server

# Unquantized Safetensors (multi-GPU)
xinfer --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --kvcache-dtype fp8

# ISQ on-the-fly quantization
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k

# NVFP4 model
xinfer --m unsloth/Qwen3.6-27B-NVFP4

# MXFP4
xinfer --m olka-fi/Qwen3.5-4B-MXFP4

# GGUF model (4-bit KvCache)
xinfer --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf --kvcache-dtype turbo4

# FP8 on Metal
xinfer --m Qwen/Qwen3.5-4B-FP8

# Gemma4 26B (NVFP4)
xinfer --m unsloth/gemma-4-26b-a4b-it-NVFP4

# MLA model (GLM4.7 Flash)
xinfer --m GadflyII/GLM-4.7-Flash-NVFP4

# Interactive CLI chat
xinfer --i --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf
```

<details>
<summary><b>ISQ (on-the-fly quantization) + KV cache compression</b></summary>

```bash
# ISQ Q4K + FP8 KV cache
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k --kvcache-dtype fp8

# ISQ Q4K + TurboQuant KV cache
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k --kvcache-dtype turbo4

# Metal ISQ
xinfer --w /path/Qwen3-4B --isq q6k
```

</details>

<details>
<summary><b>GGUF models</b></summary>

```bash
# Single GPU — GGUF
xinfer --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf

# Multi-GPU — GGUF
xinfer --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

</details>

<details>
<summary><b>TurboQuant KV cache (2–4 bit) — see <a href="#turboquant-kv-cache--run-30b-models-on-consumer-gpus">TurboQuant section</a></b></summary>

```bash
# turbo4: 4-bit K+V — 3.7× compression, best tradeoff
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4

# turbo3: 3-bit K + 4-bit V — 4.7× compression
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo3

# turbo8: FP8 K + 4-bit V — 2.6× compression, highest quality
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo8

# 35B MoE (NVFP4 + turbo4) — fits on single 24 GB GPU
xinfer --m unsloth/Qwen3.6-35B-A3B-NVFP4 --kvcache-dtype turbo4

# 30B MoE (GGUF Q4_K_M + turbo4) — consumer GPU
xinfer --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kvcache-dtype turbo4
```

</details>

<details>
<summary><b>Multimodal models (Qwen3-VL, Gemma4, Mistral3-VL)</b></summary>

```bash
# Upload images via built-in Chat UI or send image_url in API requests

# Qwen3.6 35B MoE (FP8, multimodal)
xinfer --m Qwen/Qwen3.6-35B-A3B-FP8 --ui-server

# Qwen3-VL 8B (GGUF)
xinfer --m unsloth/Qwen3-VL-8B-Instruct-GGUF --f Qwen3-VL-8B-Instruct-Q8_0.gguf --ui-server

# Gemma4 26B MoE (NVFP4, multimodal)
xinfer --m unsloth/gemma-4-26b-a4b-it-NVFP4 --ui-server

# Mistral-3 VL 3B (BF16, multimodal)
xinfer --m mistralai/Ministral-3-3B --ui-server
```

</details>

---

## 📘 Build from source code

**Option 1 — Cargo**
```bash
# Prerequisites: Rust compiler, CUDA Toolkit (optional) or Metal Xcode command line tool
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get install -y git build-essential libssl-dev pkg-config

export XINFER_REPO="https://github.com/guoqingbao/xinfer"
# MacOS/Metal: replace features to `metal`
# SM_70/SM_75 (e.g., V100): remove `flashinfer` and `cutlass` features
cargo install --git $XINFER_REPO xinfer --features cuda,nccl,flashinfer,cutlass
```

**Option 2 — Docker**
```bash
# Turing/V100 (sm_70/sm_75): remove `flashinfer` and `cutlass` features
./build_docker.sh "cuda,nccl,flashinfer,cutlass"
```

See [Docker guide →](docs/docker.md)


<details>
<summary><b>Build Python wheel from source</b></summary>

```bash
pip install maturin maturin[patchelf]

# FlashInfer backend (SM80+)
./build.sh --release --features cuda,nccl,flashinfer,cutlass,python

# Flash Attention backend
./build.sh --release --features cuda,nccl,flashattn,cutlass,python

# macOS Metal
maturin build --release --features metal,python

# Install
pip install target/wheels/xinfer*.whl --force-reinstall
```

</details>

See [more Python examples →](python/ReadMe.md)

---

## 🔀 Prefill-Decode Disaggregation

Split prefill (prompt processing) and decode (token generation) across GPUs or machines. Eliminates decode stalls during long-context prefilling. PD Server and PD Client must use **same** KvCache type (`--kvcache-dtype`). API request(s) must send to PD Client and the PD Server only process internal prefill requests sent from PD Client.

| Mode | Config | Use Case |
|---|---|---|
| Local IPC | _(default, no flag)_ | Same machine, CUDA |
| File IPC | `--pd-url file:///path` | Docker containers, shared volume |
| Remote TCP | `--pd-url tcp://host:port` | Different machines |

**Local IPC** (multirank)
```bash
# PD Server (prefill GPU, default port 7000)
xinfer --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --pd-server

# PD Client (decode GPU + API)
xinfer --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
```

**Multinode** (tcp mode)
```bash
# Server machine (192.168.1.100)
target/release/xinfer --d 0,1 --m Qwen/... --pd-server --pd-url tcp://0.0.0.0:8100

# Client machine
target/release/xinfer --d 0,1 --w /path/... --pd-client --pd-url tcp://192.168.1.100:8100 --ui-server --port 8000
```

> Metal/macOS requires `--pd-url` (no LocalIPC support).

<details>
<summary><b>Multi-container (file:// mode)</b></summary>

```bash
mkdir -p /tmp/pd-sockets

# Server container
docker run --gpus '"device=0,1"' -v /tmp/pd-sockets:/sockets ...
target/release/xinfer --d 0,1 --m Qwen/... --pd-server --pd-url file:///sockets

# Client container
docker run --gpus '"device=2,3"' -v /tmp/pd-sockets:/sockets ...
target/release/xinfer --d 0,1 --w /path/... --pd-client --pd-url file:///sockets --ui-server --port 8000
```

</details>

---

## 🔌 MCP Tool Calling

```bash
xinfer --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --mcp-config ./mcp.json
```

[MCP documentation →](docs/mcp_tool_calling.md)

---

## 🔌 Structured Outputs

Constraint-based generation via llguidance — Lark grammars, regex, JSON Schema.

[Structured outputs documentation →](docs/guided_decoding.md)

---

## 📚 Documentation

| Guide | Description |
|---|---|
| [Get Started](docs/get_started.md) | Build, run, and configure |
| [Docker](docs/docker.md) | Container builds and deployment |
| [Performance](docs/performance.md) | Full benchmark tables |
| [Prefix Cache](docs/prefix-cache.md) | Automatic KV cache reuse |
| [Multimodal](docs/multimodal.md) | Vision-language models |
| [Embedding](docs/embeddings.md) | Text embedding API |
| [Tokenizer API](docs/tokenizer_api.md) | Tokenize / detokenize endpoints |
| [Tool Parsing](docs/tool_parsing.md) | Tool call detection and parsing |
| [MCP Integration](docs/mcp_tool_calling.md) | Model Context Protocol |
| [Guided Decoding](docs/guided_decoding.md) | Structured outputs |
| [Rust Crate](docs/rust_crate.md) | Use as a library |
| [Add a Model](docs/add_model.md) | Port a new architecture (AI-assisted) |
| [Test a Model](docs/test_model.md) | Validate model quality (AI-assisted) |

**Using Agents under xInfer backend:** [xbot](docs/xbot.md) · [OpenCode](docs/opencode.md) · [Kilo Code](docs/kilocode.md) · [Claude Code](docs/claude_code.md) · [Goose](docs/goose.md)

---

## ⚙️ CLI Reference

| Flag | Description |
|---|---|
| `--m` | HuggingFace model ID (auto-download) |
| `--w` | Local Safetensors model path |
| `--f` | GGUF file path (or filename when `--m` is given) |
| `--d` | Device IDs (e.g. `--d 0,1`) |
| `--ui-server` | API server + built-in ChatGPT-style web UI |
| `--server` | API server only (no web UI) |
| `--i` | Interactive CLI chat |
| `--isq` | On-the-fly quantization: `q2k`, `q3k`, `q4k`, `q5k`, `q6k`, `q8_0` |
| `--kvcache-dtype` | KV cache quantization: `fp8`, `turbo8`, `turbo4`, `turbo3` |
| `--max-num-seqs` | Max concurrent requests (default: 32, macOS: 8) |
| `--max-tokens` | Max tokens per response (default: 16384) |
| `--kv-fraction` | GPU memory fraction for KV cache |
| `--cpu-mem-fold` | CPU swap memory ratio (default: 0.2) |
| `--pd-server` | Run as PD prefill server |
| `--pd-client` | Run as PD decode client |
| `--pd-url` | PD connection URL (`tcp://`, `http://`, `file://`) |
| `--disable-prefix-cache` | Disable prefix caching |
| `--prefix-cache-max-tokens` | Cap prefix cache size |
| `--prefill-chunk-size` | Cap prefill chunk size (default: CUDA 8K, Metal: 4k) |
| `--disable-cuda-graph` | Disable CUDA graph capture |
| `--yarn-scaling-factor` | YARN RoPE context extension factor |
| `--temperature` | Sampling temperature (0–1) |
| `--top-k` / `--top-p` | Top-k / nucleus sampling |
| `--presence-penalty` | Penalize repeated tokens (−2 to 2) |
| `--frequency-penalty` | Penalize frequent tokens (−2 to 2) |
| `--mcp-config` | MCP servers JSON config |
| `--mcp-command` / `--mcp-args` | Single MCP server command + args |

---

## 📽️ Demo

<video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>

---

## 🛠️ Roadmap

* [x] Batched inference (Metal)
* [x] GGUF format support
* [x] FlashAttention (CUDA)
* [x] CUDA Graph
* [x] OpenAI-compatible API (streaming support)
* [x] Continuous batching
* [x] Multi-gpu inference (Safetensors, GPTQ, AWQ, GGUF)
* [x] Speedup prompt processing on Metal/macOS
* [x] Chunked Prefill
* [x] Prefix cache (available on `CUDA` when `prefix-cache` enabled)
* [x] Model loading from hugginface hub
* [ ] Model loading from ModelScope (China)
* [x] Prefix cache for Metal/macOS
* [x] FP8 KV Cache (CUDA, all backends including FlashInfer on SM80+)
* [x] FP8 KV Cache (Metal)
* [x] FP8 KV Cache (with FlashInfer, SM80+)
* [x] TurboQuant KV Cache (2-4 bit compression with WHT rotation)
* [x] FP8 Models (CUDA: MoE, Dense; Metal: Dense)
* [ ] Additional model support (Kimi K2, GLM 5.1 etc.)
* [x] CPU KV Cache Offloading
* [x] Prefill-decode Disaggregation (CUDA)
* [x] Prefill-decode Disaggregation (Metal)
* [x] Built-in ChatGPT-like Web Server
* [x] Embedding API
* [x] Tokenize/Detokenize API
* [x] MCP Integration & Tool Calling
* [x] Prefix Caching
* [x] Claude/Anthropic-compatible API Server
* [x] Support CUDA 13
* [x] **Support FlashInfer backend**
* [x] **Support DeepGEMM backend (Hopper)**
* [x] **MXFP4/NVFP4 Model Support**
* [x] **Support Turboquant (4-bit, 3-bit) KvCache**
* [ ] TentorRT-LLM

---

## 📚 References

- [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
- Python nano-vllm

## Star History

<a href="https://www.star-history.com/?repos=guoqingbao%2Fxinfer&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=guoqingbao/xinfer&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=guoqingbao/xinfer&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=guoqingbao/xinfer&type=date&legend=top-left" />
 </picture>
</a>

**Like this project? Give it a ⭐ and contribute!**
