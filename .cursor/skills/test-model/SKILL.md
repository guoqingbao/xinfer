---
name: test-model
description: >-
  Test LLM models served by xinfer for correctness, output quality, and
  performance. Use when the user asks to test, benchmark, validate, or verify
  models — either from a local folder path or HuggingFace model IDs. Supports
  all xinfer-compatible formats: BF16, FP8, MXFP4, NVFP4, GGUF, GPTQ, AWQ,
  ISQ, Dense, MoE, and Multimodal architectures.
---

# Test Model — Validate and Benchmark LLM Models on xinfer

## Phase 0: Gather Model List

Collect the models to test. The user provides **one or both** of:

| Input | Format | Example |
|-------|--------|---------|
| **Local folder** | Absolute path to a directory containing model weights | `/data/models` or `/data/Qwen3.5-27B-FP8` |
| **HuggingFace IDs** | Comma-separated model IDs | `AxionML/Qwen3.5-2B-NVFP4, Qwen/Qwen3-4B` |

### Detecting models in a local folder

If the user provides a **parent directory** (not a single model), scan it to find testable models:

```bash
# List subdirectories that look like model folders
for d in /data/*/; do
  if [ -f "$d/config.json" ] || ls "$d"/*.gguf 2>/dev/null | head -1 >/dev/null; then
    echo "$d"
  fi
done
```

For each candidate directory, determine the model type by reading `config.json`:

```python
import json, os, sys, glob

def detect_model(path):
    """Detect model type and quantization from a local directory."""
    config_path = os.path.join(path, "config.json")
    gguf_files = glob.glob(os.path.join(path, "*.gguf"))

    info = {"path": path, "name": os.path.basename(path.rstrip("/"))}

    if gguf_files:
        info["format"] = "gguf"
        info["gguf_file"] = os.path.basename(gguf_files[0])
        return info

    if not os.path.exists(config_path):
        return None

    cfg = json.load(open(config_path))
    arch = (cfg.get("architectures") or ["Unknown"])[0]

    supported = [
        "LlamaForCausalLM", "MistralForCausalLM", "Ministral3ForConditionalGeneration",
        "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
        "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "Qwen3VLForConditionalGeneration",
        "Gemma3ForConditionalGeneration", "Gemma3ForCausalLM",
        "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
        "Phi3ForCausalLM", "Phi4ForCausalLM",
        "Glm4ForCausalLM", "Glm4MoeForCausalLM",
    ]
    if arch not in supported:
        info["skip"] = f"Unsupported architecture: {arch}"
        return info

    info["arch"] = arch
    info["format"] = "safetensors"

    qcfg = cfg.get("quantization_config", {})
    qm = qcfg.get("quant_method", "")
    if qm in ("fp8", "modelopt", "compressed-tensors"):
        algo = qcfg.get("quant_algo", "")
        fmt = qcfg.get("format", "")
        if algo and ("nvfp4" in algo.lower() or "fp4" in algo.lower()):
            info["quant"] = "nvfp4"
        elif "nvfp4" in fmt.lower():
            info["quant"] = "nvfp4"
        elif "mxfp4" in fmt.lower():
            info["quant"] = "mxfp4"
        elif qm == "fp8":
            info["quant"] = "fp8"
        else:
            info["quant"] = qm
    elif qm in ("gptq", "awq"):
        info["quant"] = qm
    elif qm == "mxfp4":
        info["quant"] = "mxfp4"
    else:
        info["quant"] = "bf16"

    return info
```

Present the detected models to the user as a table and confirm before proceeding.

---

## Phase 1: Estimate GPU Requirements and Detect Hardware

### Detect available GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
```

Parse the output to get `gpu_id`, `name`, `total_mb`, `free_mb` for each GPU.

### Estimate model memory

Use these rough heuristics for memory estimation (single-GPU, including KV cache overhead):

| Format | Estimate (GB) |
|--------|---------------|
| BF16 / FP16 | `params_B * 2.2` |
| FP8 | `params_B * 1.2` |
| MXFP4 / NVFP4 | `params_B * 0.8` |
| GGUF Q4_K_M | `params_B * 0.7` |
| GGUF Q3_K_M | `params_B * 0.55` |
| GGUF Q2_K | `params_B * 0.45` |
| MoE (A3B active) | Use active params for compute, total params for weight memory |

Extract parameter count from the model name when possible (e.g. `Qwen3.5-27B` → 27B).
For MoE models with `A3B` in the name, the weight memory uses total params but fits better than dense.

### GPU assignment rules

1. If a model fits in one GPU's free memory, use `--d <gpu_id>` with the GPU that has the most free memory.
2. If a model needs 2 GPUs, use `--d <id1>,<id2>` with the two GPUs with the most free memory.
3. If a model exceeds all available GPU memory, report it as skipped and move to the next.
4. For models explicitly specified as multi-GPU by the user, respect that.

---

## Phase 2: Build the Project

Build using `build.sh`:

```bash
cd <project_root>
./build.sh --install --features cuda,nccl,flashinfer,cutlass
```

Verify the build succeeds (exit code 0). The `Error: Must provide model_id or weight_path` message after build is expected — it means the binary compiled correctly.

If the build fails, check and fix compilation errors before proceeding.

Important, if you build on CUDA with cargo build, make sure always build xinfer binaries.
---

## Phase 3: Create the Test Script

Create `test_model.py` in the project root with the following capabilities:

- Accept `--port` to specify the API server port
- Accept `--wait` for server readiness timeout
- Test both `thinking=false` and `thinking=true` modes
- Send a prompt with **at least 1024 input tokens** and request **at least 2048 output tokens**
- Measure end-to-end throughput (completion_tokens / total_time)
- Check output quality: detect excessive 3-gram repetition, too-short responses
- Report prompt tokens, completion tokens, time, throughput, and quality verdict
- Print a summary table at the end

The prompt should be a substantive multi-topic question (algorithms, data structures, etc.) padded with context tokens to reach the 1k+ input requirement. Use `max_tokens: 2048` and `temperature: 0.7`. Set request timeout to 300s.

For thinking mode, add `"extra_body": {"thinking": true}` to the payload.

Quality checks:
- Response must be at least 100 characters
- 3-gram repetition: flag if any trigram appears more than `max(10, 5% of total trigrams)` times

---

## Phase 4: Test Each Model

For each model, execute this sequence:

### Step 1: Kill previous instances

```bash
pkill -9 -f 'xinfer' 2>/dev/null
sleep 3
```

Always wait 3 seconds after killing to ensure GPU memory is released.

### Step 2: Start the server

Build the server command based on model type:

| Model source | Command pattern |
|-------------|-----------------|
| Local safetensors | `./target/release/xinfer --w <path> --ui-server --d <gpus> --port 7000` |
| Local GGUF | `./target/release/xinfer --w <dir> --f <file.gguf> --ui-server --d <gpus> --port 7000` |
| HuggingFace ID | `./target/release/xinfer --m <hf_id> --ui-server --d <gpus> --port 7000` |

Run the server in the background with `RUST_BACKTRACE=1` for debugging.

### Step 3: Wait for server readiness

Poll `GET /v1/models` every 2-3 seconds until it returns HTTP 200, with a timeout of:
- Small models (< 10B): 120s
- Medium models (10-40B): 300s
- Large models (> 40B) or HF downloads: 600s

### Step 4: Run the test script

```bash
python3 test_model.py --port 7000
```

### Step 5: Handle failures

If the server fails to start or the test script returns errors:

1. **Check server logs** for panics or errors
2. **Common issues and fixes**:

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `MLX-quantized models` panic | Incompatible NVFP4 packing | Skip model; use modelopt/compressed-tensors variant |
| `Unable to load ... projection weights` | DeltaNet weights not detected as quantized | Check `is_weight_quantized` in `deltanet.rs` |
| `CUDA out of memory` | Model too large for GPU | Try with more GPUs or skip |
| Server starts but API times out | Model too slow on prefill | Increase test timeout to 600s |
| `failed to fill whole buffer` | Runner process crashed | Check runner logs, enable `RUST_BACKTRACE=full` |

3. **Debug with unwrap**: If the model crashes during inference, temporarily change `guard.step()` to `guard.step().unwrap()` in `src/core/engine.rs` to get a full stack trace. **Revert after debugging.**

4. If a model cannot be fixed, record the failure reason and continue to the next model.

---

## Phase 5: Summarize Results

After all models are tested, produce a summary table:

```
## Test Results

| # | Model | Format | GPUs | thinking=false | thinking=true | Quality |
|---|-------|--------|------|----------------|---------------|---------|
| 1 | Qwen3.5-27B-FP8 | FP8 | 1 | 1342 in / 2048 out, 42.2 tok/s | 1342 in / 2048 out, 42.2 tok/s | OK |
| 2 | ... | ... | ... | ... | ... | ... |

### Notes
- Model X: SKIPPED — reason
- Model Y: FAILED — error description
```

Include for each model:
- Model name and quantization format
- Number of GPUs used
- Input/output token counts and throughput for both thinking modes
- Quality verdict (OK / ISSUES / FAILED / SKIPPED)

---

## Quick Reference

### Key files

| File | Purpose |
|------|---------|
| `test_model.py` | OpenAI API test script (created by this skill) |
| `src/core/engine.rs` | Engine loop; `guard.step()` for debug |
| `src/models/layers/deltanet.rs` | DeltaNet layer; quantization detection |
| `src/models/layers/linear.rs` | Linear layer loaders (FP8, MXFP4, NVFP4) |
| `build.sh` | Build script (compiles xinfer) |

### Build features

| Feature set | When to use |
|-------------|-------------|
| `cuda,nccl,flashinfer,cutlass` | SM80+ (Ampere/Ada/Hopper), recommended |
| `cuda,nccl,flashattn,cutlass` | Alternative to flashinfer |
| `cuda,nccl` | V100 (SM70), no flash attention |
| `metal` | macOS Apple Silicon |

### Server flags

| Flag | Purpose |
|------|---------|
| `--w <path>` | Local model weight directory |
| `--f <file>` | GGUF filename within the weight directory |
| `--m <hf_id>` | HuggingFace model ID (auto-downloads) |
| `--d <ids>` | GPU device IDs (e.g. `0` or `0,1`) |
| `--port <n>` | API server port |
| `--disable-prefix-cache` | Disable prefix caching (on by default) |
| `--ui-server` | Enable built-in ChatGPT-like web UI |
| `--isq <fmt>` | In-situ quantization (q2k, q3k, q4k, q5k, q6k, q8_0) |
| `--kvcache-dtype <mode>` | KV cache quantization: fp8, turbo8, turbo4, turbo3 |
