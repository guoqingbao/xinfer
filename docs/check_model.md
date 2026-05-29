# Model Compatibility Checking (AI-Assisted)

xInfer ships with a **Cursor Agent Skill** that validates model compatibility before loading. It checks config correctness, weight tensor format, quantization consistency, and multi-GPU (tensor-parallel) divisibility — catching issues that would otherwise cause runtime panics or silent precision bugs.

## Prerequisites

- [Cursor IDE](https://cursor.sh/) with Agent mode enabled (for other Agents, mention the skill file manually)
- The xInfer repository cloned locally
- A model to check: HuggingFace URL/ID, local path, or pasted config + tensor info

## How It Works

The skill lives at `.cursor/skills/check-model/SKILL.md` and is **automatically activated** when you ask the agent to check, validate, audit, or verify a model. It walks the agent through five phases:

| Phase | What happens |
|-------|-------------|
| **0 — Gather info** | Collects model config and tensor info from HuggingFace URL, local path, or user-provided data. |
| **1 — Parse config** | Identifies architecture, extracts attention/MoE/hybrid parameters, normalizes quantization config. |
| **2 — Validate tensors** | Checks that tensor names, shapes, and dtypes match the expected format for the detected quantization. |
| **3 — Multi-rank analysis** | Tests TP divisibility for world_size 1, 2, 4, and 8 across all sharded components. |
| **4 — Report** | Produces a structured summary with OK/WARN/ERROR flags for each check. |

## Quick Start

Open the project in Cursor and ask the agent:

```
Check this model: https://huggingface.co/AxionML/Qwen3.5-27B-NVFP4/blob/main/config.json
```

You can also provide tensor info by clicking a `.safetensors` file in the HuggingFace model page and copying the tensor tree:

```
Check this model for multi-GPU compatibility:
<paste config.json URL>
<paste tensor info>
```

Or check a local model:

```
Check the model at /data/Qwen3.5-122B-A10B-NVFP4/ for 4-GPU loading
```

## What Gets Checked

### Tensor Format Validation

For each quantization format, the skill verifies tensor naming and shapes:

| Format | Expected tensors per linear layer |
|--------|----------------------------------|
| **BF16/FP16** | `weight` only |
| **FP8** | `weight` (U8) + `weight_scale` or `weight_scale_inv` (F32) |
| **NVFP4 (ModelOpt)** | `weight` (U8) + `weight_scale` (F8_E4M3) + `weight_scale_2` (F32) + `input_scale` (F32) |
| **NVFP4 (compressed-tensors)** | `weight_packed` (U8) + `weight_scale` (F8_E4M3) + `weight_global_scale` (F32) + `input_global_scale` (F32) |
| **MXFP4** | `weight_packed` or `blocks` (U8) + `weight_scale` or `scales` (U8) |

### Quantization Ignore List

For models with mixed precision (e.g., NVFP4 MoE with BF16 attention), the skill parses the `ignore` list from `quantization_config` and verifies that:
- Ignored layers have BF16/FP16 weights only (no quantized tensors)
- Non-ignored layers have the expected quantized tensor structure
- The `ignore` list supports literal paths, regex patterns (`re:...`), and glob wildcards

### Multi-Rank Divisibility

For each candidate GPU count (1, 2, 4, 8), the skill checks:

| Component | What must divide evenly |
|-----------|------------------------|
| Full attention Q heads | `num_attention_heads % world_size` |
| Full attention KV heads | `num_kv_heads % world_size` (or `world_size % num_kv_heads` for replicated mode) |
| GDN K/V heads | `linear_num_key_heads % world_size` and `linear_num_value_heads % world_size` |
| GDN projections | All projection output dims by `world_size` |
| MoE intermediate | `moe_intermediate_size % world_size` |
| Shared expert | `shared_expert_intermediate_size % world_size` |
| FP8 block alignment | Per-rank boundaries aligned to `weight_block_size` |
| FP4 scale alignment | Per-rank inner dims are multiples of group_size (16 for NVFP4, 32 for MXFP4) |

### Loader Path Compatibility

The skill checks tensor naming against xinfer's loader priority order:

| Component | Tensor name priority (first match wins) |
|-----------|---------------------------------------|
| FP4 packed weights | `weight_packed` > `weight` > `blocks` |
| FP4 scales | `weight_scale` > `scales` |
| FP4 global scale | `weight_global_scale` (inverted) > `weight_scale_2` (direct) |
| FP4 input scale | `input_scale` (direct) > `input_global_scale` (inverted) |
| FP8 scale | `weight_scale` > `weight_scale_inv` |

## Example Output

```
## Model Summary
Architecture: Qwen3_5MoeForConditionalGeneration
Quantization: nvfp4 (compressed-tensors)
Layers: 48 (36 linear_attention + 12 full_attention)
Hidden: 3072, Q heads: 32, KV heads: 2, GDN K: 16, V: 64

## Tensor Format
[OK]  Linear attention: BF16 (in ignore list)
[OK]  Full attention: NVFP4 (weight_packed + weight_scale + weight_global_scale)
[OK]  MoE experts: NVFP4 per-expert (weight_packed)
[OK]  Shared expert: NVFP4 (weight_packed)

## Multi-Rank Compatibility
| Component           | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---------------------|-------|--------|--------|--------|
| Q heads (32)        |  OK   |   16   |    8   |    4   |
| KV heads (2)        |  OK   |    1   | repl(2)| repl(4)|
| GDN K heads (16)    |  OK   |    8   |    4   |    2   |
| GDN V heads (64)    |  OK   |   32   |   16   |    8   |
| MoE inter (1024)    |  OK   |  512   |  256   |  128   |
| Overall             |  OK   |   OK   |   OK   |   OK   |
```

## Integration with Add Model

The [add-model skill](add_model.md) automatically invokes the check-model skill after implementing a new architecture, ensuring the model loads correctly on single and multi-GPU configurations before testing.

## File Reference

| File | Role |
|------|------|
| `.cursor/skills/check-model/SKILL.md` | The skill definition (read by the AI agent) |
| `src/models/layers/distributed.rs` | TP column/row linear, merged chunk loading, KV head sharding |
| `src/models/layers/linear.rs` | FP8/NVFP4/MXFP4 linear layer loaders |
| `src/models/layers/deltanet.rs` | GatedDeltaNet loading, per-weight quantization detection |
| `src/models/layers/attention.rs` | Full attention QKV loading and packed QKV for FP8 |
| `src/models/layers/moe.rs` | MoE expert loading for all quantization formats |
| `src/utils/config.rs` | QuantConfig normalization, ignore list parsing |
