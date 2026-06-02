<p align="center">
  <img src="logo.svg" alt="xInfer" width="400"><br>
  <b>纯 Rust 实现的极速 LLM 推理引擎。</b>无需 PyTorch，无需 Python 运行时，开箱即用。<br>
  <a href="./ReadMe.md">English</a> | <a href="./ReadMe-CN.md">简体中文</a>
</p>

---

## ✨ 为什么选择 xInfer？

| | 特性 | 详情 |
|---|---|---|
| **0️⃣** | 零 Python 依赖 | 纯 Rust 后端 — 不需要 PyTorch 或 CUDA Python 绑定 |
| **⚡** | 极致性能 | 原生 Flash Attention、FlashInfer、CUDA Graphs、持续批处理、前缀缓存、PD 分离。消费级 GPU 上 `30B+` 模型解码速度高达 **181 tok/s** |
| **🪶** | 极简内核 | 核心调度 + 注意力逻辑仅 **< 5000 行** Rust 代码 |
| **🌍** | 跨平台 | CUDA（Linux/Windows）、Metal（macOS），统一二进制，统一 API |
| **🏭** | 生产就绪 | OpenAI/Anthropic 兼容 API、内置 ChatGPT 风格 Web UI、MCP 工具调用、结构化输出、Embedding + 分词器端点 |
| **🗜️** | 极致 KV 压缩 | TurboQuant（`2–4 位` KV 缓存）以极小的质量损失将上下文扩展至 **4.3 倍**。单卡 24/32 GB GPU 即可运行 `30B+` MoE 模型并支持**百万级上下文** |
| **🔥** | V100 + NVFP4 | 业界首创：V100 上运行 NVFP4 + 低位 KV 缓存推理 — 无需硬件 FP4，旧 GPU 重获新生 |
| **🐍** | 轻量 Python 绑定 | 需要 Python 入口时可选 PyO3 wheel 包 |

---

## 🚀 快速开始

### 📦 安装

**方式 1 — 安装 DEB 或 Python包**
```bash
curl -sSL https://guoqingbao.github.io/xinfer/install.sh | bash
```

**方式 2 — npm**
```bash
npm install -g xinfer-ai
```
install.sh 和 npm 会自动检测 GPU 的 CUDA 计算能力并下载对应的预编译二进制文件。

---

### ▶️ 运行

**使用 HuggingFace 模型 ID：**
```bash
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4 --ui-server
```

**使用本地模型路径：**
```bash
xinfer --w /home/Qwen3.6-35B-A3B --d 0,1 --ui-server
```

**Python 使用方式：**
```bash
python3 -m xinfer.server --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4 --ui-server
```

> **提示：** 浏览器打开 `http://IP:8001` 即可使用内置对话界面，或使用 `http://IP:8000/v1/` 作为 API 服务 `Base URL`。

---

### 🗜️ KV 缓存压缩

添加 `--kvcache-dtype` 参数压缩 KV 缓存，扩展上下文长度：

| 参数（`--kvcache-dtype`） | 压缩比 | 质量 | GPU 要求 |
|---|---|---|---|
| _（默认）_ | 1×（BF16） | 基线 | 全部 |
| `fp8` | **2×** | 近无损 | SM70+ / Apple M1 |
| `turbo8` | **2.6×** | 79–100% 基线吞吐 | SM70+ / Apple M1|
| `turbo4` | **3.7×** | 最佳平衡 | SM70+ / Apple M1|
| `turbo3` | **4.7×** | 最大压缩 | SM70+ |

---

## 📈 性能

> 测试平台：**V100-32G**、**A100-40G**、**Hopper-80G** 及 **RTX 5090**

| 模型 | 格式 | 大小 | 输出速度 |
|---|---|---|---|
| Ministral-3-3B (**多模态**) | ISQ (BF16→Q4K) | 3B | **193.67** tokens/s |
| Qwen3-VL-8B-Instruct (**多模态**) | Q8_0 | 8B | **112.51** tokens/s |
| Llama-3.1-8B | ISQ (BF16→Q4K) | 8B | **133.10** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **139.25** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **77.48** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **46.02** tokens/s |
| **Qwen3-30B-A3B** | NVFP4 | **30B (MoE)** | **181.59** tokens/s (**RTX 5090**) |
| **Qwen3-30B-A3B** | NVFP4 | **30B (MoE)** | **72.86** tokens/s (**V100, Software FP4**) |
| **Qwen3.5-27B** (**多模态**) | Q4_K_M | **27B (Dense)** | **49.33** tokens/s |
| **Qwen3.5-27B/Qwen3.6-27B** | FP8 | **27B (Dense)** | **45** tokens/s (**Hopper**) |
| **Qwen3.6-35B-A3B** (**多模态**) | FP8 | **35B (MoE)** | **110** tokens/s (**Hopper**) |
| **GLM4.7 Flash** | NVFP4 | **30B (MoE)** | **79** tokens/s (**Hopper, Software FP4**) |
| **Gemma4-31B** | ISQ (BF16→Q4K) | **31B (Dense)** | **47** tokens/s (**Hopper**) |
| **Gemma4-26B-A4B** | NVFP4 | **26B (MoE)** | **137.23** tokens/s (**RTX 5090**) |
| **MiniMax-M2.5** | NVFP4 | **229B (MoE)** | **64.50** tokens/s (**Hopper, Software FP4, TP=2**) |

<details>
<summary><b>Apple Silicon (M4)</b></summary>

| 模型 | 并发数 | 输出 Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
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

[完整性能测试 →](docs/performance.md)

---

## 🧠 支持的模型

* ✅ LLaMa 系列（LLaMa2、LLaMa3、**LLaMa4**、IQuest-Coder）
* ✅ Qwen 系列（Qwen2、Qwen3）
* ✅ Qwen2/Qwen3 MoE 系列
* ✅ Qwen3-Next 系列
* ✅ Qwen3.5/3.6 Dense/MoE 系列（27B、35B、122B、397B、多模态）
* ✅ Mistral v1、v2
* ✅ Mistral-3-VL Reasoning（3B、8B、14B、多模态）
* ✅ GLM4（0414 版本，**非 ChatGLM**）
* ✅ GLM4 MoE（4.6/4.7）
* ✅ GLM4.7 Flash
* ✅ DeepSeek V3/R1/V3.2
* ✅ Phi3 / Phi4（Phi-3、Phi-4、Phi-4-mini 等）
* ✅ Gemma3/**Gemma4**（多模态）
* ✅ Qwen3-VL（Dense、多模态）
* ✅ MiroThinker-v1.5（30B、235B）

**格式：** Safetensors（BF16、`FP8-blockwise`、GPTQ、AWQ、MXFP4、`NVFP4`）| GGUF（所有量化类型）| `ISQ`（即时量化）

---

### TurboQuant KV 缓存 — 消费级 GPU 运行 30B+ 模型

TurboQuant 通过 Walsh-Hadamard 变换旋转 + 逐头 absmax 量化，将 KV 缓存压缩至 2–4 位。使用 `turbo4` 的最大上下文容量：

| 模型 | KV 预算 | BF16 | turbo4 | 提升 |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B**（NVFP4） | 7 GB（24 GB GPU） | 70 万 | **270 万** | **3.9×** |
| | 15 GB（32 GB GPU） | 150 万 | **580 万** | **3.9×** |
| **Qwen3.6-27B**（FP8） | 7 GB | 11.2 万 | **43.4 万** | **3.9×** |
| | 15 GB | 24 万 | **93 万** | **3.9×** |
| **Qwen3-30B-A3B**（Q4_K_M） | 7 GB | 7.4 万 | **28.1 万** | **3.8×** |
| | 15 GB | 16 万 | **60.2 万** | **3.8×** |
| **Gemma4-26B-A4B**（NVFP4） | 7 GB | 3.2 万 | **12.5 万** | **3.9×** |
| | 15 GB | 7 万 | **27.1 万** | **3.9×** |

> 混合架构模型（Qwen3.6）全注意力层数远少于总层数，TurboQuant 压缩效果尤为显著。MLA 模型（DeepSeek、GLM4.7 Flash）请搭配 `fp8` 使用。表中 KV 预算为理论最大值，实际可用量最高为 KV 预算的 90%（`--kv-fraction 0.9`），需为运行时和批处理预留缓冲空间。

```bash
# 35B MoE 单卡 24/32 GB 即可运行
xinfer --m unsloth/Qwen3.6-35B-A3B-NVFP4 --kvcache-dtype turbo4

# FP8 精度
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

## 📘 使用方法
> **Python包安装后**请使用 `python3 -m xinfer.server` 方式运行

> Docker 内构建请参考 [**在 Docker 中运行 xInfer →**](docs/docker.md)

### 运行模型

> **提示：** 默认启动 OpenAI 兼容 API 服务（`http://localhost:8000`）。添加 `--ui-server` 可同时启动内置 ChatGPT 风格 Web UI（`http://localhost:8001`）。

```bash
# FP8 模型（sm90+ 需启用 cutlass）+ Web UI
xinfer --m Qwen/Qwen3.6-27B-FP8 --ui-server

# 未量化 Safetensors（多卡）
xinfer --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --kvcache-dtype fp8

# ISQ 即时量化
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k

# NVFP4 模型
xinfer --m unsloth/Qwen3.6-27B-NVFP4

# MXFP4
xinfer --m olka-fi/Qwen3.5-4B-MXFP4

# GGUF 模型（4 位 KV 缓存）
xinfer --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf --kvcache-dtype turbo4

# FP8 Metal
xinfer --m Qwen/Qwen3.5-4B-FP8

# Gemma4 26B（NVFP4）
xinfer --m unsloth/gemma-4-26b-a4b-it-NVFP4

# MLA 模型（GLM4.7 Flash）
xinfer --m GadflyII/GLM-4.7-Flash-NVFP4

# 交互式 CLI 对话
xinfer --i --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf
```

<details>
<summary><b>ISQ 即时量化 + KV 缓存压缩</b></summary>

```bash
# ISQ Q4K + FP8 KV 缓存
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k --kvcache-dtype fp8

# ISQ Q4K + TurboQuant KV 缓存
xinfer --m Qwen/Qwen3.6-35B-A3B --isq q4k --kvcache-dtype turbo4

# Metal ISQ
xinfer --w /path/Qwen3-4B --isq q6k
```

</details>

<details>
<summary><b>GGUF 模型</b></summary>

```bash
# 单卡 — GGUF
xinfer --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf

# 多卡 — GGUF
xinfer --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

</details>

<details>
<summary><b>TurboQuant KV 缓存（2–4 位压缩）— 详见 <a href="#turboquant-kv-缓存--消费级-gpu-运行-30b-模型">TurboQuant 专区</a></b></summary>

```bash
# turbo4: 4位 K+V — 3.7× 压缩，最佳平衡
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo4

# turbo3: 3位 K + 4位 V — 4.7× 压缩
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo3

# turbo8: FP8 K + 4位 V — 2.6× 压缩，最高质量
xinfer --m Qwen/Qwen3.6-27B-FP8 --kvcache-dtype turbo8

# 35B MoE（NVFP4 + turbo4）— 24 GB 单卡即可运行
xinfer --m unsloth/Qwen3.6-35B-A3B-NVFP4 --kvcache-dtype turbo4

# 30B MoE（GGUF Q4_K_M + turbo4）— 消费级 GPU
xinfer --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kvcache-dtype turbo4
```

</details>

<details>
<summary><b>多模态模型（Qwen3-VL, Gemma4, Mistral3-VL）</b></summary>

```bash
# 通过内置 Chat UI 上传图片或在 API 请求中传入 image_url

# Qwen3.6 35B MoE（FP8，多模态）
xinfer --m Qwen/Qwen3.6-35B-A3B-FP8 --ui-server

# Qwen3-VL 8B（GGUF）
xinfer --m unsloth/Qwen3-VL-8B-Instruct-GGUF --f Qwen3-VL-8B-Instruct-Q8_0.gguf --ui-server

# Gemma4 26B MoE（NVFP4，多模态）
xinfer --m unsloth/gemma-4-26b-a4b-it-NVFP4 --ui-server

# Mistral-3 VL 3B（BF16，多模态）
xinfer --m mistralai/Ministral-3-3B --ui-server
```

</details>

---
## 📘 从源码编译安装
**方式 1 — Cargo**
```bash
# 依赖项: Rust 编译器、CUDA 工具链（可选）、Metal 需安装 Xcode 命令行工具（可选）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get install -y git build-essential libssl-dev pkg-config

export XINFER_REPO="https://github.com/guoqingbao/xinfer"
# macOS/Metal：将特性替换为 `--features metal`
# SM_70/SM_75（如 V100）：去掉 `flashinfer` 和 `cutlass` 编译选项
cargo install --git $XINFER_REPO xinfer --features cuda,nccl,flashinfer,cutlass
```

**方式 2 — Docker**
```bash
# Turing/V100 (sm_70/sm_75)：去掉 `flashinfer` 和 `cutlass` 编译选项
./build_docker.sh "cuda,nccl,flashinfer,cutlass"
```

参考 [Docker 指南 →](docs/docker.md)

<details open>
<summary><b>从源码编译 Python wheel</b></summary>

```bash
pip install maturin maturin[patchelf]

# FlashInfer 后端（SM80+）
./build.sh --release --features cuda,nccl,flashinfer,cutlass,python

# Flash Attention 后端
./build.sh --release --features cuda,nccl,flashattn,cutlass,python

# macOS Metal
maturin build --release --features metal,python

# 安装
pip install target/wheels/xinfer_ai-*.whl --force-reinstall
```

</details>

---

## 🔀 Prefill-Decode 分离（PD 分离）

将预填充（prompt 处理）和解码（token 生成）拆分到不同 GPU 或机器。消除长上下文预填充时的解码卡顿。PD 服务器与 PD 客户端必须使用相同 KvCache 数据类型（`--kvcache-dtype`）。对话请求需发送至 PD 客户端，PD 服务端只处理 PD 客户端发来的预填充请求。

| 模式 | 配置 | 适用场景 |
|---|---|---|
| 本地 IPC | _（默认，无需参数）_ | 同一机器，CUDA |
| 文件 IPC | `--pd-url file:///path` | Docker 容器，共享卷 |
| 远程 TCP | `--pd-url tcp://host:port` | 不同机器 |

**单机多卡部署**（无需指定 pd-url，默认使用 CUDA IPC）
```bash
# PD 服务器（预填充 GPU，默认端口 7000）
xinfer --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --pd-server

# PD 客户端（解码 GPU + API 服务）
xinfer --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
```

**多机部署**（tcp 模式）
```bash
# 服务器机器（192.168.1.100）
target/release/xinfer --d 0,1 --m Qwen/... --pd-server --pd-url tcp://0.0.0.0:8100

# 客户端机器
target/release/xinfer --d 0,1 --w /path/... --pd-client --pd-url tcp://192.168.1.100:8100 --ui-server --port 8000
```

> Metal/macOS 不支持 Local IPC，必须指定 `--pd-url`。

<details>
<summary><b>多容器部署（file:// 模式）</b></summary>

```bash
mkdir -p /tmp/pd-sockets

# 服务器容器
docker run --gpus '"device=0,1"' -v /tmp/pd-sockets:/sockets ...
target/release/xinfer --d 0,1 --m Qwen/... --pd-server --pd-url file:///sockets

# 客户端容器
docker run --gpus '"device=2,3"' -v /tmp/pd-sockets:/sockets ...
target/release/xinfer --d 0,1 --w /path/... --pd-client --pd-url file:///sockets --ui-server --port 8000
```

</details>

---

## 🔌 MCP 工具调用

```bash
xinfer --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --mcp-config ./mcp.json
```

[MCP 文档 →](docs/mcp_tool_calling.md)

---

## 🔌 结构化输出

通过 llguidance 实现约束生成 — Lark 语法、正则表达式、JSON Schema。

[结构化输出文档 →](docs/guided_decoding.md)

---

## 📚 文档

| 指南 | 说明 |
|---|---|
| [快速开始](docs/get_started.md) | 编译、运行与配置 |
| [Docker](docs/docker.md) | 容器构建与部署 |
| [性能测试](docs/performance.md) | 完整性能表 |
| [前缀缓存](docs/prefix-cache.md) | 自动 KV 缓存复用 |
| [多模态](docs/multimodal.md) | 视觉语言模型 |
| [Embedding](docs/embeddings.md) | 文本嵌入 API |
| [分词器 API](docs/tokenizer_api.md) | Tokenize / Detokenize 端点 |
| [工具调用解析](docs/tool_parsing.md) | 工具调用检测与解析 |
| [MCP 集成](docs/mcp_tool_calling.md) | Model Context Protocol |
| [引导解码](docs/guided_decoding.md) | 结构化输出 |
| [Rust 库](docs/rust_crate.md) | 作为 Rust 库使用 |
| [添加模型](docs/add_model.md) | 移植新架构（AI 辅助） |
| [测试模型](docs/test_model.md) | 验证模型质量（AI 辅助） |

**在 xInfer 后端下使用 Agent：** [xbot](docs/xbot.md) · [OpenCode](docs/opencode.md) · [Kilo Code](docs/kilocode.md) · [Claude Code](docs/claude_code.md) · [Goose](docs/goose.md)

---

## ⚙️ 命令行参数

| 参数 | 说明 |
|---|---|
| `--m` | HuggingFace 模型 ID（自动下载） |
| `--w` | 本地 Safetensors 模型路径 |
| `--f` | GGUF 文件路径（或配合 `--m` 使用时为文件名） |
| `--d` | 设备 ID（如 `--d 0,1`） |
| `--ui-server` | API 服务 + ChatGPT 风格内置 Web UI |
| `--server` | 仅 API 服务（无 Web UI） |
| `--i` | 交互式 CLI 对话 |
| `--isq` | 即时量化：`q2k`、`q3k`、`q4k`、`q5k`、`q6k`、`q8_0` |
| `--kvcache-dtype` | KV 缓存量化：`fp8`、`turbo8`、`turbo4`、`turbo3` |
| `--max-num-seqs` | 最大并发请求数（默认 32，macOS 为 8） |
| `--max-tokens` | 单次最大输出 token 数（默认 16384） |
| `--kv-fraction` | GPU 显存用于 KV 缓存的比例 |
| `--cpu-mem-fold` | CPU 交换显存比例（默认 0.2） |
| `--pd-server` | 作为 PD 预填充服务器运行 |
| `--pd-client` | 作为 PD 解码客户端运行 |
| `--pd-url` | PD 连接 URL（`tcp://`、`http://`、`file://`） |
| `--disable-prefix-cache` | 禁用前缀缓存 |
| `--prefix-cache-max-tokens` | 前缀缓存大小上限 |
| `--prefill-chunk-size` | 预填充分块大小 (默认: CUDA 8K, Metal: 4k) |
| `--disable-cuda-graph` | 禁用 CUDA 图捕获 |
| `--yarn-scaling-factor` | YARN RoPE 上下文扩展因子 |
| `--temperature` | 采样温度（0–1） |
| `--top-k` / `--top-p` | Top-k / 核采样 |
| `--presence-penalty` | 重复惩罚（−2 到 2） |
| `--frequency-penalty` | 高频惩罚（−2 到 2） |
| `--mcp-config` | MCP 服务器 JSON 配置 |
| `--mcp-command` / `--mcp-args` | 单个 MCP 服务器命令及参数 |

---

## 📽️ 演示

<video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>

---

## 🛠️ 开发计划

* [x] Metal 平台支持批量推理
* [x] 支持 GGUF 格式
* [x] CUDA 平台 Flash Attention 支持
* [x] CUDA Graph
* [x] OpenAI API 兼容服务器（支持流式输出）
* [x] 持续批处理
* [x] 多卡并行推理（Safetensors 模型、GPTQ/AWQ 及 GGUF 量化模型）
* [x] Metal/macOS 平台 Prompt 处理加速
* [x] 分块预填充（Chunked Prefill）
* [x] 前缀缓存（使用 `prefix-cache` 参数）
* [x] 从 HuggingFace Hub 下载并加载模型
* [ ] 从 ModelScope 下载并加载（中国大陆地区）
* [x] Metal/macOS 平台前缀缓存
* [x] FP8 KV Cache（CUDA，所有后端，FlashInfer 支持 SM80+）
* [x] FP8 KV Cache（Metal）
* [x] FP8 KV Cache（FlashInfer，SM80+）
* [x] TurboQuant KV Cache（2-4 位压缩，WHT 旋转量化）
* [x] FP8 模型（CUDA: MoE, Dense; Metal: Dense）
* [ ] 支持更多模型类型（Kimi K2、GLM 5.1 等）
* [x] CPU KV Cache 卸载
* [x] PD（Prefill/Decode）分离（CUDA）
* [x] PD（Prefill/Decode）分离（Metal）
* [x] 内置 ChatGPT 风格 Web 网页服务
* [x] Embedding API
* [x] Tokenize/Detokenize API
* [x] MCP 集成与工具调用
* [x] 公共前缀缓存
* [x] Claude/Anthropic API 兼容服务器
* [x] 支持 CUDA 13
* [x] **支持 FlashInfer 后端**
* [x] **支持 DeepGEMM 后端（Hopper）**
* [x] **MXFP4/NVFP4 模型支持**
* [x] **支持 Turboquant（4 位、3 位）KvCache**
* [ ] TentorRT-LLM

---

## 📚 参考项目

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

**喜欢这个项目？欢迎 ⭐ 收藏和参与贡献！**
