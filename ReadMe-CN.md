# 🚀 **vLLM.rs** – 用 Rust 实现的极简 vLLM

一个极速 ⚡、轻量的 🦀**Rust 实现版 vLLM**。

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a>
</p>

## ✨ 主要特性

* 🔧 **纯 Rust 后端** – 完全**不依赖 PyTorch**
* 🚀 **高性能** (支持**前缀缓存、PD分离**)
* 🧠 **极简核心** – 核心逻辑仅 **<3000 行** Rust 代码
* 💻 **跨平台支持** – 支持 **CUDA**（Linux/Windows）与 **Metal**（macOS）
* 🤖 **内置API 服务与ChatGPT风格网页** – Rust 原生实现的聊天与 API/Web 服务
* 🔌 **MCP集成** – Model Context Protocol 工具调用支持
* 📊 **Embedding与分词器API** – 完整的文本处理支持
* 🐍 **轻量 Python 接口** – 使用 PyO3 构建的 Python 聊天接口

---

## 📈 性能

### 💬 对话性能

> **A100** (单卡, 40G)

| 模型 | 格式 | 大小 | 输出速度 |
|------------------|---------------|----------|------------------------|
| Ministral-3-3B (Multimodal) | BF16 | 3B | **118.49** tokens/s |
| Ministral-3-3B (Multimodal) | ISQ (BF16->Q4K) | 3B | **171.92** tokens/s |
| Qwen3-VL-8B-Instruct (**Multimodal**) | Q8_0 | 8B | **105.31** tokens/s |
| Llama-3.1-8B | ISQ (BF16->Q4K) | 8B | **120.74** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **124.87** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **41.36** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)**| **97.16** tokens/s  |
| **Qwen3.5-27B** | Q4_K_M | **27B (Dense)**| **45.20** tokens/s  |
| **Qwen3.5-27B/Qwen3.6-27B** | FP8 | **27B (Dense)**| **42** tokens/s (**Hopper**)  |
| **Qwen3.5-35B-A3B** | FP8 | **35B (MoE)**| **97** tokens/s (**Hopper**)  |
| **GLM4.7 Flash** | NVFP4 | **30B (MoE)**| **79** tokens/s (**Hopper**)  |
| **Gemma4-31B** | ISQ (BF16->Q4K) | **31B (Dense)**| **41** tokens/s (**Hopper**)  |
| **Gemma4-26B-A4B** | NVFP4 | **26B (MoE)**| **82** tokens/s (**Hopper**)  |
| **MiniMax-M2.5** | NVFP4 | **229B (MoE)**| **62** tokens/s (**Hopper, TP=2**)  |

> vLLM.rs 在 **Metal (Apple Silicon, M4)** 上的性能

  <details>

   | 模型 | 并发数 | 输出Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
   |------------------|--------|--------|---------|-------------|
   | Qwen3-0.6B (BF16) |  128  | 63488       | 83.13s    | 763.73     |
   | Qwen3-0.6B (BF16) |  32      | 15872       | 23.53s    | 674.43    |
   | Qwen3-0.6B (BF16) | 1       | 456       | 9.23s    | 49.42       |
   | Qwen3-4B (Q4_K_M)  | 1       | 1683       | 52.62s    | 31.98     |
   | Qwen3-8B (Q2_K)  | 1       | 1300       | 80.88s    | 16.07     |
   | Qwen3.5-4B (Q3_K_M)  | 1       | 1592       | 69.04s | 23.06    |
   | Qwen3.5-2B (NVFP4)  | 1       | 1883       | 60.76s | 30.99    |
   | Qwen3.5-2B (NVFP4)  | 2       | 3942       | 81.96s | 48.10    |
  </details>

查看 [**完整性能测试 →**](docs/performance.md)

## 🧠 支持的模型架构

* ✅ LLaMa 系列（LLaMa2、LLaMa3, **LLaMa4**, IQuest-Coder）
* ✅ Qwen 系列（Qwen2、Qwen3）
* ✅ Qwen2/Qwen3 Moe 系列
* ✅ Qwen3-Next 系列
* ✅ Qwen3.5/3.6 Dense/MoE 系列（27B, 35B, 122B, 397B, 多模态）
* ✅ Mistral v1, v2
* ✅ Mistral-3 VL Reasoning (3B, 8B, 14B, 多模态)
* ✅ GLM4 (0414版本, **非ChatGLM**)
* ✅ GLM4 MoE (4.6/4.7)
* ✅ GLM4.7 Flash
* ✅ DeepSeek V3/R1/V3.2
* ✅ Phi3 / Phi4 (Phi-3, Phi-4, Phi-4-mini等)
* ✅ Gemma3/**Gemma4** (多模态)
* ✅ Qwen3-VL (Dense, 多模态)
* ✅ MiroThinker-v1.5 (30B, 235B)

支持 **Safetensor** (包含GPTQ, AWQ, MXFP4, NVFP4, FP8-blockwise 量化格式) 和 **GGUF** 格式。

所有模型均支持FP8 KvCache加速（`--fp8-kvcache`），兼容所有注意力后端：`flashinfer`（SM80+）、`flashattn`、以及 Paged Attention（V100/SM70+）。

所有模型还支持 **TurboQuant KV缓存压缩**（`--kvcache-dtype turbo4` / `turbo3` / `turbo8`），通过Walsh-Hadamard变换和逐头absmax量化实现2-4位KV缓存压缩，大幅降低显存占用。

---
## 📚 文档
- [快速开始](docs/get_started.md)
- [Docker构建](docs/docker.md)
- [工具调用解析](docs/tool_parsing.md)
- [MCP集成与工具调用](docs/mcp_tool_calling.md)
- [引导解码/结构化输出](docs/guided_decoding.md)
- [在xbot中使用vLLM.rs后端](docs/xbot.md)
- [OpenCode使用vLLM.rs后端](docs/open_code.md)
- [Kilo Code使用vLLM.rs后端](docs/kilocode.md)
- [Claude Code使用vLLM.rs后端](docs/claude_code.md)
- [Goose AI Agent使用vLLM.rs后端](docs/goose.md)
- [Embedding](docs/embeddings.md)
- [多模态 (Qwen3-VL, Gemma3, Mistral3-VL)](docs/multimodal.md)
- [前缀缓存](docs/prefix-cache.md)
- [Rust库](docs/rust_crate.md)
- [Tokenize/Detokenize](docs/tokenize.md)
- [性能测试](docs/performance.md)
- [模型测试 (AI辅助)](docs/test_model.md)
- [为本项目添加新模型架构（AI辅助）](docs/add_model.md)

## 📘 使用方法（Python）
### 📦 使用 pip 安装
- 💡 **CUDA 计算能力 < 8.0**（例如 V100）需要**手动编译** （不支持 `flashattn`；或可使用 **Rust 模式**）。
- 💡 **预编译包** 默认启用了 `flashinfer` 后端，在SM80+上已原生支持 **FP8 KV Cache**。

> 🍎 Metal（macOS）
```shell
python3 -m pip install vllm_rs
````

> 🟩 CUDA（Linux）

#### Ampere / Ada（SM80+）

```shell
#（可选）安装 NCCL
apt-get install -y libnccl2 libnccl-dev
python3 -m pip install vllm_rs
```

#### Hopper（SM90+）/ Blackwell（SM120+）

从 [Release Assets](https://github.com/guoqingbao/vllm.rs/releases/) 下载 wheel，解压后安装 `.whl` 包。


### 🌐✨ API Server + ChatGPT风格内置网页
   💡使用`--ui-server`会同时启动ChatGPT风格网页, 此时无需其它客户端。

   💡如长文本请求导致当前生成过程卡顿，请使用 **Rust PD Server**方案 （见**PD分离**）

   💡前缀缓存为自动匹配公共前缀，无需 `session_id`。

   💡如希望未传 `thinking` / `enable_thinking` 的请求默认关闭推理，可在 Rust CLI 启动时增加 `--disable-reasoning`。

  <details open>
    <summary>单卡 + GGUF模型</summary>

  ```bash
  # CUDA
  python3 -m vllm_rs.server --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf --ui-server --prefix-cache
  # Metal/MacOS (MacOS Tahoe之前的系统可能会存在生成过慢问题)
  python3 -m vllm_rs.server --m unsloth/Qwen3.5-4B-GGUF --f Qwen3.5-4B-Q3_K_M.gguf --ui-server --prefix-cache
   ```
  </details>

   <details open>
    <summary>多卡 + 本地GGUF模型</summary>

   ```bash
   python3 -m vllm_rs.server --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --ui-server --prefix-cache
   ```
  </details>

  </details>

   <details open>
    <summary>将未量化模型加载为GGUF模型</summary>

   ```bash
   # 同时将权重量化为Q4K格式，启用最长上下文：
   python3 -m vllm_rs.server --w /path/Qwen3.5-122B-A10B --isq q4k --d 0,1 --port 8000 --max-model-len 262144 --max-num-seqs 1 --ui-server --prefix-cache
   ```
  </details>


  <details open>
    <summary>FP8模型</summary>

_FP8-Blockwise格式:_
```bash
# CUDA (MoE, Dense) sm90+ 设备需打开`cutlass`特性以支持FP8硬件加速
python3 -m vllm_rs.server --m Qwen/Qwen3.6-27B-FP8 --ui-server --prefix-cache --fp8-kvcache
# MacOS/Metal (Dense)
python3 -m vllm_rs.server --m Qwen/Qwen3-4B-Instruct-2507-FP8 --ui-server --prefix-cache
```

_MXFP4 格式:_
```bash
python3 -m vllm_rs.server --m olka-fi/Qwen3.5-4B-MXFP4 --ui-server --prefix-cache
```

_NVFP4 格式:_
```bash
python3 -m vllm_rs.server --m unsloth/Qwen3.6-27B-NVFP4 --ui-server --prefix-cache
```
  </details>

<details open>
    <summary>多模态模型 (Qwen3 VL, +图片)</summary>

```bash
# 使用内置ChatUI上传或提及图片url (格式 '.bmp', '.gif', '.jpeg', '.png', '.tiff', or '.webp')
python3 -m vllm_rs.server --m Qwen/Qwen3.5-35B-A3B-FP8 --ui-server --prefix-cache
```

  <details>
    <summary>运行GPTQ/AWQ Marlin兼容模型</summary>

```bash
python3 -m vllm_rs.server --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin
```
  </details>

查看 [**更多Python示例 →**](python/ReadMe.md)



## 📘 使用方法（Rust）

### CUDA平台安装 (CUDA 11+, 12+, 13.0)

> 方案 1：安装进Docker：
   <details>

```bash
cd vllm.rs
# 将 `sm_80` 更改至你当前的硬件特性，如 sm_75 (V100), sm_80 (A100), sm_86 (RTX4090), sm_90 (Hopper), sm_100/sm_120 (Blackwell); 将 CUDA 版本号 `12.9.0` 更改为与当前主机驱动匹配的版本; 将最后一个参数 `0` 更改为 `1` 启用Rust中国区镜像（适用于中国大陆）
./build_docker.sh "cuda,nccl,graph,flashinfer,cutlass,python" sm_80 12.9.0 0

# 还可以使用 `flash attention` 后端, 以及传入 `--prod` 以构建生产镜像
./build_docker.sh --prod "cuda,nccl,graph,flashattn,cutlass,python" sm_90 13.0.0
```
   </details>

参考 [**如何通过Docker运行 vLLM.rs 服务 →**](docs/docker.md)

> 方案 2：手动安装：

   <details open>

安装 Rust 工具链
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装构建依赖：
```sh
sudo apt-get update
sudo apt-get install -y git build-essential libssl-dev pkg-config
```

安装 CUDA Toolkit：
```sh
# CUDA 12.9 （版本号<= 本机驱动版本）
apt-get update
apt-get install -y \
  cuda-nvcc-12-9 \
  cuda-nvrtc-dev-12-9 \
  libcublas-dev-12-9 \
  libcurand-dev-12-9

# NCCL
apt-get install -y libnccl2 libnccl-dev
```
编译 vLLM.rs
```shell
# 只有单卡的情况下去掉 `nccl`
# V100及较老的机型(SM70)去掉 `flashattn/flashinfer` 和 `cutlass`特性
# SM80+(A100, RTX4090, H100等)推荐使用flashinfer，支持FP8 KVCache
# 默认安装进/usr/local/bin，使用`--dst`更改安装目录
./build.sh --install --features cuda,nccl,graph,flashinfer,cutlass

# 使用Flash Attention后端
./build.sh --install --features cuda,nccl,graph,flashattn,cutlass
```
  </details>

### MacOS/Metal平台安装

安装 [Xcode 命令行工具](https://mac.install.guide/commandlinetools/)

使用`metal`特性安装
```shell
cargo install --features metal
```

### 运行方式

默认启动 **API 服务模式**（端口 8000）。使用 `--i` 启用交互模式 🤖，`--ui-server` 启用带 Web UI 的服务模式 🌐，`--m` 指定Huggingface模型，或`--w` 指定本地Safetensors模型路径 或`--f` 指定GGUF模型文件：

> 单卡/多卡推理
  <details open>
    <summary>单卡推理</summary>

   ```bash
   # CUDA （将 `--i`替换成 `--ui-server`则启用网页版本）
   vllm-rs --i --m unsloth/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf --kv-fraction 0.8
   # Metal/MacOS (MacOS Tahoe之前的系统可能会存在生成过慢问题，使用更小的`--max-model-len` 或 `--kv-fraction`减少显存占用)
   vllm-rs --i --m unsloth/Qwen3.5-4B-GGUF --f Qwen3.5-4B-Q3_K_M.gguf
   ```
  </details>

  <details open>
    <summary>多卡未量化模型</summary>

   ```bash
   vllm-rs --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --ui-server --prefix-cache --fp8-kvcache
   ```
  </details>

  <details open>
    <summary>FP8/FP4模型</summary>

  _FP8格式:_
   ```bash
   vllm-rs --d 0,1 --w /path/Qwen3-Coder-30B-A3B-Instruct-FP8/ --ui-server --prefix-cache --fp8-kvcache
    # Or Qwen3.6-27B-FP8 / Qwen3.6-35B-A3B-FP8
   vllm-rs --m Qwen/Qwen3-Coder-Next-FP8 --ui-server --d 0,1 --prefix-cache
   ```

  _MXFP4格式 (CUDA):_
  ```bash
  vllm-rs --m olka-fi/Qwen3.5-4B-MXFP4 --ui-server --prefix-cache
  ```

  _NVFP4格式:_
  ```bash
vllm-rs --m unsloth/Qwen3.6-27B-NVFP4 --ui-server --prefix-cache
# MacOS/Metal
vllm-rs --m AxionML/Qwen3.5-2B-NVFP4 --ui-server --prefix-cache
  ```
  </details>

   <details open>
    <summary>多卡量化模型</summary>

   ```bash
   vllm-rs --ui-server --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --prefix-cache
   ```
  </details>

   <details open>
    <summary>未量化模型运行为Q4K量化模型，同时使用FP8 KVCache</summary>

   ```bash
   # 使用flashinfer后端（SM80+推荐）
   vllm-rs --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --server --port 8000 --fp8-kvcache
   # V100/SM70+ 使用paged attention后端
   # 编译时去除`flashinfer` 和 `flashattn`
   ```
  </details>

---

## 🔌 LLGuidance 支持（结构化输出与约束）

vLLM.rs 现在支持通过 llguidance 库实现结构化输出和约束生成：

- **自定义约束**：允许客户端通过 structured_outputs 或 response_format 提交 Lark/Regex/JSON Schema 约束

查看 [**结构化输出文档 →**](docs/llguidance-integration.md)

---

## 🔌 MCP集成 (工具调用)
通过Model Context Protocol让LLM调用外部工具。

```bash
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --prefix-cache --mcp-config ./mcp.json
```
查看 [**MCP文档 →**](docs/mcp_tool_calling.md)

---

## 🔀 Prefill-decode 分离（PD分离）

  <details>
    <summary>启动PD服务器</summary>
   Metal/MacOS平台或PD服务器与PD客户端不在同一OS，服务器与客户端需要同时指定`--pd-url`（例如0.0.0.0:8100）

   无需指定`port`，因为此服务器不直接接收用户请求，KvCache大小由`--max-model-len`和`--max-num-seqs`控制。
   ```bash
   # PD服务器使用`flashinfer` 或 `flashattn` 加快处理长文本prefill（PD服务器启动非量化模型可获得最佳吞吐率）
   vllm-rs --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --pd-server
   ```

   PD服务器还可使用预编译Python包启动 (依赖：pip install vllm_rs)
   ```bash
   python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --pd-server
   ```
  </details>

  <details>
    <summary>启动PD客户端</summary>

   ```bash
   vllm-rs --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
   ```

  PD客户端还可使用预编译Python包启动 (依赖：pip install vllm_rs)
  ```bash
   python3 -m vllm_rs.server --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
   ```
  </details>

  <details>
    <summary>单机多个Dockers/多机配置</summary>

   PD Server与Client启动时的模型及Rank数量（卡数）需要一致，可为相同模型的不同格式（例如服务器未量化Safetensor, 客户端GGUF）
   如果指定了 `--pd-url`（例如 server端: 0.0.0.0:8100, client端: server_ip:8100），PD 服务器/客户端将尝试绑定或连接到该地址，
   客户端将尝试使用指定的 URL 连接到服务器（Metal平台不支持LocalIPC, 必须提供pd-url）。在这种情况下，服务器和客户端可以部署在不同的机器上。
   单机多卡，PD服务器与客户端运行于不同Docker，需要配置Docker启动参数 `--ipc=host`
  </details>

---

## 📽️ 演示视频

🎉 观看项目运行演示：
<video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>


## 🔨 从源代码编译安装（可选）

> ⚠️ 启用 Flash Attention（CUDA）时，首次编译可能需要较长时间。

> ⚠️ 启用 前缀缓存或多GPU推理时，需要同时编译`Runner`（使用`build.sh`编译 或 `run.sh`运行）

### 🛠️ 环境要求
* 构建 Python 接口需安装 [Maturin](https://github.com/PyO3/maturin)

### 编译步骤
1. **安装 Maturin**

```bash
sudo apt install libssl-dev pkg-config -y # 编译依赖 (Linux)
pip install maturin
pip install maturin[patchelf]  # Linux/Windows 平台
```

2. **构建 Python 包**

```bash
# Naive CUDA (不带NCCL，只能用于单卡推理) 
maturin build --release --features cuda,graph,python

# CUDA Paged Attention（V100/SM70+，支持FP8 KV Cache）
./build.sh --release --features cuda,nccl,graph,python

# CUDA FlashInfer后端（SM80+，支持FP8 KV Cache）
./build.sh --release --features cuda,nccl,graph,flashinfer,cutlass,python

# CUDA Flash Attention后端
./build.sh --release --features cuda,nccl,graph,flashattn,cutlass,python

# macOS（Metal，支持FP8 KV Cache，但不支持多GPU推理）
maturin build --release --features metal,python

```

3. **安装构建好的包与依赖**

```bash
pip install target/wheels/vllm_rs-*-cp38-abi3-*.whl --force-reinstall
```


### ⚙️ 命令行参数说明

| 参数          | 描述                                     |
| ----------- | -------------------------------------- |
| `--m`       | Hugginface模型ID (用于下载)               |
| `--w`       | Safetensor模型路径           |
| `--f`       | 当指定Model ID时为GGUF文件名，或未指定时为GGUF本地文件路径                 |
| `--d`       | 设备 ID，例如 `--d 0`                       |
| `--max-num-seqs`   | 同时处理的最大请求数（默认 `32`, macOS平台为`8`）   |
| `--max-tokens`     | 单次最大输出 token 数（默认 `16384`，上限为模型支持的最大长度） |
| `--batch`     | 仅用于性能 (启用后会忽略 `max-num-seqs` 与 `prompts`) |
| `--prompts` | 输入的 prompt，多个使用 \| 分隔 |
| `--dtype`   | KV 缓存数据类型：`bf16`（默认）、`f16` 或 `f32`     |
| `--isq`   | 将未量化模型加载为GGUF量化模型，可选`q2k`, `q4k`  等   |
| `--temperature`   | 采样温度 (sampling temperature)，控制输出"随机性/创造性"的一个超参数，介于0-1之间  |
| `--top-k`   | top-k 控制模型在每一步只从前 k 个最高概率的词里挑选，k 越小 → 越稳定；k 越大 → 越随机   |
| `--top-p`   | top-p 采样根据概率阈值选择动态数量的候选，范围是 [0,1]，常用在 0.8 ~ 0.95   |
| `--presence-penalty` | 出现惩罚，控制模型是否避免再次提及`已经出现过的词`。<br> 数值范围 [-2, 2]，正值越大 → 越倾向引入新词汇；负值 → 越倾向重复已出现的词 |
| `--frequency-penalty` | 频率惩罚，控制模型是否减少`高频重复词`的出现。<br> 数值范围 [-2, 2]，正值越大 → 重复次数越多的词惩罚越强；负值 → 越鼓励重复使用同一词 |
| `--server`       | 显式启动API服务（未指定 `--i`、`--prompts` 或 `--batch` 时，这是默认行为）        |
| `--i`            | 交互式CLI对话模式                                        |
| `--fp8-kvcache`       | 使用FP8 KV Cache（兼容所有后端：flashinfer SM80+、paged attention V100+、Metal）|
| `--kvcache-dtype`     | KV缓存量化：`fp8`、`turbo8`（FP8 K + 4位V）、`turbo4`（4位K+V）、`turbo3`（3位K + 4位V）。TurboQuant模式使用原生flash attention。|
| `--cpu-mem-fold`       | CPU KV Cache大小 (与GPU KV Cache的百分比，默认 0.2，取值0.1 - 10.0)              |
| `--pd-server`       | 使用PD分离模式时，指定当前实例为PD服务器（此服务器仅用于Prefill）            |
| `--pd-client`       | 使用PD分离模式时，指定当前实例为PD客户端（此客户端将长的上下文Prefill请求发送给PD服务器处理）|
| `--pd-url`       |  使用PD分离模式时，PD服务器实例如指定pd-url，则通过TCP/IP通信（适用于PD服务器与客户端在不同服务器） |
| `--ui-server`       |  服务模式: 启动API服务，同时启动ChatGPT风格的内置对话网页服务 |
| `--kv-fraction`       |  用于控制KVCache使用量 (模型加载后剩余可用GPU显存的百分比) |
| `--prefix-cache`   | 启用前缀缓存，用于多轮对话 |
| `--prefix-cache-max-tokens`   | 限制前缀缓存大小（按 block size 向下取整） |
| `--yarn-scaling-factor`       | YARN RoPE缩放因子，用于扩展上下文窗口（例如：`4.0` 扩展4倍上下文） |

### MCP配置参数

| 参数 | 描述 |
|------|------|
| `--mcp-command` | 单个MCP服务器可执行文件路径 |
| `--mcp-args` | MCP服务器参数（逗号分隔） |
| `--mcp-config` | 多个MCP服务器的JSON配置文件路径 |

## 📌 项目状态

> 🚧 **项目仍在积极开发中，接口与功能可能发生变更。**

## 🛠️ 开发计划（TODO）

* [x] Metal 平台支持批量推理
* [x] 支持 GGUF 格式
* [x] CUDA 平台 Flash Attention 支持
* [x] CUDA Graph
* [x] OpenAI API 兼容服务器（支持流式输出）
* [x] 持续批处理
* [x] 多卡并行推理（Safetensors模型、GPTQ/AWQ及GGUF量化模型）
* [x] Metal/macOS平台Prompt处理加速
* [x] 分块预填充（Chunked Prefill）
* [x] 前缀缓存 (使用`prefix-cache`参数)
* [x] 从Hugginface Hub下载并加载模型
* [ ] 从ModelScope下载并加载 (中国大陆地区)
* [x] Metal/macOS平台前缀缓存
* [x] FP8 KV Cache (CUDA，所有后端，FlashInfer支持SM80+)
* [x] FP8 KV Cache (Metal)
* [x] FP8 KV Cache (FlashInfer，SM80+)
* [x] TurboQuant KV Cache（2-4位压缩，WHT旋转量化）
* [x] FP8 模型 (CUDA: MoE, Dense; Metal: Dense)
* [ ] 支持更多模型类型（Kimi K2, GLM 5.1等）
* [x] CPU KV Cache 卸载
* [x] PD（Prefill/Decode）分离（CUDA）
* [x] PD（Prefill/Decode）分离（Metal）
* [x] 内置 ChatGPT风格 Web 网页服务
* [x] Embedding API
* [x] Tokenize/Detokenize API
* [x] MCP集成与工具调用
* [x] 公共前缀缓存
* [x] Claude/Anthropic API 兼容服务器
* [x] 支持CUDA 13
* [x] **支持FlashInfer后端**
* [x] **支持DeepGEMM后端 (Hopper)**
* [x] **MXFP4/NVFP4模型支持**
* [ ] TentorRT-LLM 后端

## 📚 参考项目

参考：

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm 项目

---

💡 **喜欢这个项目？欢迎 ⭐ 收藏和参与贡献！**
