# Adding New Model Architectures (AI-Assisted)

vLLM.rs ships with a **Cursor Agent Skill** that guides AI coding assistants through the full workflow of adapting a new HuggingFace model architecture to this project. It covers safetensors and GGUF formats, Dense and MoE architectures, and quantization formats (MXFP4, FP8, ISQ, etc.).

## Prerequisites

- [Cursor IDE](https://cursor.sh/) with Agent mode enabled (for other Agents, mention the skill file manually)
- The vLLM.rs repository cloned locally
- (Optional) A local copy of [attention.rs](https://github.com/guoqingbao/attention.rs) if custom kernels are needed

## How It Works

The skill lives at `.cursor/skills/add-model/SKILL.md` and is **automatically activated** when you ask the agent to add, port, or support a new model. It walks the agent through six phases:

| Phase | What happens |
|-------|-------------|
| **0 — Gather info** | Collects model config, weight tensor info, and GGUF metadata. Can read from a local model path, HuggingFace Hub, or user-provided files. |
| **1 — Analyze** | Compares the new architecture against existing models, identifies closest template, classifies as Dense / MoE / Hybrid / Multimodal. |
| **2 — Kernels** | Determines if new CUDA + Metal kernels are needed in `attention.rs`. If so, clones the repo locally, adds both backends, and points `Cargo.toml` to the local copy. |
| **3 — Implement** | Creates the model file (`src/models/<arch>.rs`), handles HF and GGUF weight paths, MoE construction, packed expert layouts. |
| **4 — Register** | Wires the model into `mod.rs`, `config.rs`, `utils/mod.rs`, `runner.rs`, and `parser.rs`. |
| **5 — Build** | Compiles with the correct platform features and fixes all errors/warnings. |
| **6 — Test** | Starts the server, sends OpenAI-compatible requests, debugs common issues (NaN, crashes, wrong output). |

## Quick Start

Open the project in Cursor and ask the agent:

```
Add support for the model google/gemma-4-26B-A4B-it
```

The agent will read the skill, then ask you for any missing information (config URL, tensor info, etc.). If you have the model downloaded locally, provide the path:

```
Add support for the model at /path/to/my-model/
```

The agent will inspect the local weights directly to extract config and tensor info.

## What You Can Provide

The skill accepts multiple input forms — provide whichever is convenient:

- **HuggingFace model ID** — e.g. `google/gemma-4-26B-A4B-it`
- **Local safetensors directory (prefered)** — path containing `config.json` + `*.safetensors`
- **Local GGUF file** — a single `.gguf` file
- **Manual config + tensor info** — paste `config.json` contents (or the path/url) and weight tensor names/shapes (can be obtained by clicking model weights in Hugginface model/files repo)

## Supported Configurations

| Dimension | Supported variants |
|-----------|--------------------|
| **Format** | Safetensors (HF), GGUF, GPTQ, AWQ |
| **Architecture** | Dense, MoE, Hybrid MoE (parallel dense+MoE), Multimodal (vision+language) |
| **Quantization** | ISQ (in-situ: Q4K, Q8_0, etc.), FP8, MXFP4, NVFP4, GPTQ, AWQ |
| **Attention** | GQA, MQA, MHA, Sliding Window, Multi-head Latent Attention (MLA) |
| **Platform** | CUDA (Linux/Windows), Metal (macOS) |

## Multi-GPU Models

For models that require multi-GPU inference, the agent uses `run.sh` which builds the `runner` binary alongside the main server:

```bash
./run.sh --release --features "cuda,flashinfer,cutlass,nccl" -- \
    --model <model_id> --port 8000 --tensor-parallel <num_gpus>
```

When retrying after a failed load, the agent will kill all `vllm-rs` and `runner` processes and verify GPU memory is freed before reloading.

## Debugging Tips

If the model loads but produces incorrect output, the skill guides the agent through:

1. Comparing logits against the Python HuggingFace reference implementation
2. Checking RoPE parameters (`rope_theta`, `partial_rotary_factor`, `head_dim`)
3. Verifying activation functions (`Silu` vs `GeluPytorchTanh`)
4. Inspecting MoE routing (expert selection, normalization, scaling)
5. Validating weight loading prefixes (HF vs GGUF tensor name mappings)

## File Reference

| File | Role |
|------|------|
| `.cursor/skills/add-model/SKILL.md` | The skill definition (read by the AI agent) |
| `src/models/<arch>.rs` | Model implementation (created by the agent) |
| `src/models/layers/moe.rs` | Shared MoE layer (FusedMoe, FusedMoeGGUF, ISQ, MXFP4) |
| `src/utils/mod.rs` | Config parsing and architecture registration (revised by the agent) |
| `src/core/runner.rs` | Model enum and dispatch macros (revised by the agent) |
