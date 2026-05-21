# Model Testing (AI-Assisted)

vLLM.rs ships with a **Cursor Agent Skill** that automates the process of testing LLM models for correctness, output quality, and inference performance. It handles GPU detection, model loading, API testing, and result summarization.

## Prerequisites

- [Cursor IDE](https://cursor.sh/) with Agent mode enabled (for other Agents, mention the skill file manually)
- The vLLM.rs repository cloned locally
- One or more NVIDIA GPUs with sufficient memory for the target models
- Python 3 with the requests library installed

## How It Works

The skill lives at .cursor/skills/test-model/SKILL.md and is **automatically activated** when you ask the agent to test, benchmark, or validate models. It walks the agent through five phases:

| Phase | What happens |
|-------|-------------|
| **0 - Gather models** | Collects model list from a local folder scan or user-provided HuggingFace IDs. Auto-detects supported architectures and quantization formats. |
| **1 - Estimate resources** | Queries nvidia-smi for available GPUs and free memory. Estimates per-model memory requirements and assigns GPUs accordingly. |
| **2 - Build** | Compiles the project with run.sh --features cuda,nccl,flashinfer,cutlass --release (builds both vllm-rs and the runner binary). |
| **3 - Create test script** | Writes test_model.py, an OpenAI-compatible API test that sends 1k+ input tokens, requests 1k+ output tokens, and measures throughput and quality. |
| **4 - Test each model** | Iteratively starts the server, waits for readiness, runs the test script (with and without reasoning/thinking), records results, kills the server, and moves to the next model. |
| **5 - Summarize** | Produces a markdown table with per-model results: format, GPUs, throughput, and quality verdict. |

## Quick Start

Open the project in Cursor and ask the agent:

```
Test all models in /data/
```

The agent will scan the directory, identify compatible models, and test each one automatically.

You can also specify individual models:

```
Test models AxionML/Qwen3.5-2B-NVFP4, Qwen/Qwen3-4B
```

Or mix local paths and HuggingFace IDs:

```
Test /data/Qwen3.5-27B-FP8 and AxionML/Qwen3.5-9B-NVFP4
```

## What Gets Tested

For each model, the skill tests:

| Aspect | Details |
|--------|---------|
| **Loading** | Server starts and model is accessible via /v1/models |
| **Inference (no thinking)** | 1k+ input tokens, 2k output tokens, thinking=false |
| **Inference (with thinking)** | Same prompt with thinking=true / reasoning enabled |
| **Output quality** | Coherence check, repetition detection (3-gram analysis) |
| **Performance** | End-to-end throughput in tokens/second |

## Supported Model Formats

| Format | Detection method |
|--------|-----------------|
| **BF16 / FP16** (safetensors) | No quantization_config in config.json |
| **FP8** (blockwise) | quant_method: fp8 |
| **MXFP4** | quant_method: mxfp4 or format contains mxfp4 |
| **NVFP4** (modelopt) | quant_method: modelopt with quant_algo: NVFP4 |
| **NVFP4** (compressed-tensors) | format contains nvfp4 |
| **GGUF** | .gguf file present in directory |
| **GPTQ / AWQ** | quant_method: gptq or awq |

## GPU Assignment

The agent automatically assigns GPUs based on available memory:

- **Single GPU**: Models that fit in one GPU's free memory
- **Multi-GPU**: Large models that require 2+ GPUs (uses --d 0,1 etc.)
- **Skip**: Models that exceed all available GPU memory

## Debugging Failures

If a model fails to load or produce output, the skill guides the agent through:

1. Checking server logs for panics or weight-loading errors
2. Enabling RUST_BACKTRACE=1 for stack traces
3. Temporarily using guard.step().unwrap() in engine.rs for crash debugging
4. Identifying common issues (MLX format incompatibility, quantization detection, OOM)

## Example Output (Decoding performance)

| # | Model | Format | GPUs | thinking=false | thinking=true | Quality |
|---|-------|--------|------|----------------|---------------|---------|
| 1 | Qwen3-4B-FP8 | FP8 | 1 | 1329 in / 2048 out, 168 tok/s | 1329 in / 2048 out, 169 tok/s | OK |
| 2 | Qwen3.5-27B | BF16 | 1 | 1342 in / 2048 out, 28 tok/s | 1342 in / 2048 out, 27 tok/s | OK |
| 3 | Qwen3.5-35B-A3B-GGUF | Q3_K_M | 1 | 1342 in / 2048 out, 95 tok/s | 1342 in / 2048 out, 98 tok/s | OK |

## File Reference

| File | Role |
|------|------|
| .cursor/skills/test-model/SKILL.md | The skill definition (read by the AI agent) |
| test_model.py | OpenAI API test script (created by the skill) |
| run.sh | Build script for both vllm-rs and runner binaries |
| src/core/engine.rs | Engine loop, guard.step() location for debug |
| src/models/layers/deltanet.rs | DeltaNet layer with per-weight quantization detection |
