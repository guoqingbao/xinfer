---
name: check-model
description: >-
  Check model compatibility with xinfer before loading. Validates config.json,
  weight tensor shapes and naming, quantization format correctness, and
  multi-rank (tensor-parallel) divisibility. Use when the user asks to check,
  validate, audit, or verify a model will load correctly — from a HuggingFace
  URL/config, local path, or pasted tensor info.
---

# Check Model — Pre-Load Compatibility Audit for xinfer

## Phase 0: Gather Model Information

Collect model config and tensor info. Accept **any** of:

| Input | How to use |
|-------|-----------|
| **HuggingFace config URL** | Fetch `config.json` from the URL (e.g. `https://huggingface.co/<id>/blob/main/config.json`) |
| **HuggingFace model ID** | Fetch config from `https://huggingface.co/<id>/raw/main/config.json` |
| **Local model path** | Read `<path>/config.json` directly |
| **Pasted config JSON** | Parse inline |
| **Tensor info** | User pastes tensor names/shapes/dtypes from HuggingFace safetensor viewer or provides local weights |

If tensor info is missing, ask the user to provide it. They can get it by clicking any `.safetensors` file in the HuggingFace model page and copying the tensor tree.

For local models, extract tensor info with:

```python
import json, struct, sys, glob, os
path = sys.argv[1]
for sf in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
    with open(sf, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for k, v in sorted(header.items()):
        if k != "__metadata__":
            print(f"{k}\t{v.get('shape')}\t{v.get('dtype')}")
```

---

## Phase 1: Parse Config and Identify Model Type

Extract from `config.json`:

### Core parameters

| Field | Required | Notes |
|-------|----------|-------|
| `architectures` | Yes | Determines model type and loader path |
| `hidden_size` | Yes | Or nested under `text_config` for multimodal |
| `num_attention_heads` | Yes | Q heads for full attention |
| `num_key_value_heads` | Yes | KV heads for GQA |
| `head_dim` | If available | Defaults to `hidden_size / num_attention_heads` |
| `num_hidden_layers` | Yes | Total layer count |
| `vocab_size` | Yes | Embedding table size |

### Hybrid (Qwen3.5/Qwen3Next) parameters

| Field | When present | Notes |
|-------|-------------|-------|
| `layer_types` | Qwen3.5/Qwen3Next | Array of `"linear_attention"` / `"full_attention"` |
| `linear_num_key_heads` | Hybrid models | GDN K heads (may differ from V heads) |
| `linear_num_value_heads` | Hybrid models | GDN V heads |
| `linear_key_head_dim` | Hybrid models | Per-head K dimension |
| `linear_value_head_dim` | Hybrid models | Per-head V dimension |
| `linear_conv_kernel_dim` | Hybrid models | Conv1d kernel size (typically 4) |
| `full_attention_interval` | Hybrid models | How often full attention appears |

### MoE parameters

| Field | When present | Notes |
|-------|-------------|-------|
| `num_experts` | MoE models | Expert count per layer |
| `num_experts_per_tok` | MoE models | Top-K routing |
| `moe_intermediate_size` | MoE models | Per-expert FFN hidden dim |
| `shared_expert_intermediate_size` | Some MoE | Shared expert dim |

### Quantization config

| Field | Notes |
|-------|-------|
| `quantization_config.quant_method` | `"modelopt"`, `"compressed-tensors"`, `"fp8"`, `"gptq"`, `"awq"` |
| `quantization_config.quant_algo` | For modelopt: `"NVFP4"`, `"FP4"` |
| `quantization_config.format` | For compressed-tensors: `"nvfp4-pack-quantized"`, `"mxfp4-pack-quantized"` |
| `quantization_config.config_groups` | Weight/activation quant specs |
| `quantization_config.ignore` | Layers excluded from quantization (stored as BF16/FP16) |
| `quantization_config.weight_block_size` | FP8 block dimensions (e.g. `[128, 128]`) |

### Normalize quant_method

Apply the same normalization as `QuantConfig::normalize_compressed_tensors()`:

| Raw `quant_method` | `quant_algo` / `format` | Normalized |
|--------------------|------------------------|------------|
| `modelopt` | `NVFP4` or `FP4` | `nvfp4` |
| `modelopt` | (detect from config_groups) | `nvfp4` |
| `compressed-tensors` | format contains `nvfp4` | `nvfp4` |
| `compressed-tensors` | format contains `mxfp4` | `mxfp4` |
| `fp8` | - | `fp8` |
| `gptq` | - | `gptq` |
| `awq` | - | `awq` |

---

## Phase 2: Validate Tensor Format Against Quantization Config

For each layer type, check that tensor names and dtypes match the expected format.

### 2a. Determine which layers are quantized vs skipped

Parse the `ignore` list from `quantization_config`. Layers in the ignore list should have BF16/FP16 weights (`weight` tensor only). Layers NOT in the ignore list should have quantized tensors.

The ignore list supports:
- Literal paths: `"model.language_model.layers.0.linear_attn.in_proj_qkv"`
- Regex patterns: `"re:.*linear_attn.*"`
- Glob-style wildcards: `"model.visual*"`, `"mtp.layers.0*"`

### 2b. Format-specific tensor checks

#### Unquantized (BF16/FP16)

Expected tensors per linear layer:
- `weight` — dtype BF16 or F16, shape `[out_dim, in_dim]`
- `bias` (optional) — dtype BF16 or F16

**Check**: No extra scale/packed tensors should be present.

#### FP8 (quant_method == "fp8")

Expected tensors per linear layer:
- `weight` — dtype U8 (F8_E4M3), shape `[out_dim, in_dim]`
- `weight_scale` or `weight_scale_inv` — dtype F32, shape `[out_dim/by, in_dim/bx]` where `[by, bx]` = `weight_block_size` (default `[128, 128]`)
- `bias` (optional)

**Check**: `weight_block_size` must have exactly 2 elements. Scale dimensions must match `ceil(out_dim/by) x ceil(in_dim/bx)`.

#### NVFP4 — ModelOpt format (quant_method == "modelopt" + quant_algo == "NVFP4")

Expected tensors per quantized linear layer:
- `weight` — dtype U8, shape `[out_dim, in_dim/2]` (packed FP4, 2 values per byte)
- `weight_scale` — dtype F8_E4M3 (U8), shape `[out_dim, in_dim/16]` (group_size=16)
- `weight_scale_2` — dtype F32, scalar (global weight scale, direct multiplier)
- `input_scale` — dtype F32, scalar (activation scale)

**Check**: `weight` shape dim1 must be exactly `in_dim/2`. Scale dim1 must be `in_dim/16`.

#### NVFP4 — Compressed-tensors format (quant_method == "compressed-tensors" + nvfp4 format)

Expected tensors per quantized linear layer:
- `weight_packed` — dtype U8, shape `[out_dim, in_dim/2]`
- `weight_scale` — dtype F8_E4M3 (U8), shape `[out_dim, in_dim/16]`
- `weight_global_scale` — dtype F32, scalar or `[1]` (divisor, inverted at load time)
- `input_global_scale` — dtype F32, scalar or `[1]` (divisor, inverted at load time)

**Check**: Same shape rules as ModelOpt, but different tensor names.

#### MXFP4 (quant_method == "mxfp4" or compressed-tensors with mxfp4)

Expected tensors per quantized linear layer:
- `weight_packed` or `blocks` — dtype U8, shape `[out_dim, in_dim/2]`
- `weight_scale` or `scales` — dtype U8 (F8_E8M0), shape `[out_dim, in_dim/32]` (group_size=32)

**Check**: Scale dim1 must be `in_dim/32`.

#### GGUF

GGUF models are self-contained (no `config.json`). Weight tensor names use `blk.{i}` prefix mapped to `model.layers.{i}`. Quantization is per-tensor via GGML dtypes (Q4_K, Q6_K, Q8_0, etc.).

**Check**: Not applicable for safetensors checks. GGUF has its own loader path via `QLinear` / `QMatMul`.

### 2c. Loader path tensor name resolution

The xinfer loaders try tensor names in priority order. Verify the model's tensors match at least one:

| Component | Tensor name priority (first match wins) |
|-----------|---------------------------------------|
| NVFP4/MXFP4 packed weights | `weight_packed` > `weight` > `blocks` |
| NVFP4/MXFP4 scales | `weight_scale` > `scales` |
| NVFP4 global scale | `weight_global_scale` (inverted) > `weight_scale_2` (direct) |
| NVFP4 input scale | `input_scale` (direct) > `input_global_scale` (inverted) |
| FP8 scale | `weight_scale` > `weight_scale_inv` |

**Flag any mismatch** where the model uses a tensor name not in the priority list.

### 2d. Hybrid model (GDN) quantization detection

For Qwen3.5/Qwen3Next models with quantization config, the `GatedDeltaNet` layer has its own quantization detection (`is_weight_quantized`) that checks each linear_attn sublayer independently:

| quant_method | Detection logic |
|-------------|----------------|
| `fp8` | Has `weight_scale` or `weight_scale_inv` |
| `mxfp4` | Has `weight_packed` or `blocks` |
| `nvfp4` | (`weight_packed` or `blocks`) AND (`weight_scale` or `scales`) **OR** (`weight_scale_2` or `input_scale`) AND (`weight_scale` or `scales`) |

If a linear_attn sublayer is in the ignore list and has only BF16 `weight`, the detection returns false, and the layer loads as unquantized. **Verify** this matches the tensor info.

---

## Phase 3: Multi-Rank Divisibility Analysis

For each candidate `world_size` in `[1, 2, 4, 8]`, check all TP-sharded dimensions.

### 3a. Full Attention

| Component | Global dim | Shard dim | Divisibility requirement |
|-----------|-----------|-----------|------------------------|
| Q projection | `num_attention_heads * head_dim` | dim 0 | `num_attention_heads % world_size == 0` |
| K/V projection | `num_kv_heads * head_dim` | dim 0 | `num_kv_heads >= world_size`: `num_kv_heads % world_size == 0`; `num_kv_heads < world_size`: `world_size % num_kv_heads == 0` (replicated KV mode) |
| O projection | `num_attention_heads * head_dim` | dim 1 | Same as Q |

For quantized (FP8/NVFP4/MXFP4) Q/K/V:
- Column linear shard dim 0: per-rank `out_dim / world_size` must be cleanly divisible
- For FP8: per-rank start must be aligned to `weight_block_size[0]` (default 128)
- For NVFP4: per-rank output must be divisible (no block alignment needed for dim 0 shard)

### 3b. GatedDeltaNet (Linear Attention)

| Component | Global dim | Requirement |
|-----------|-----------|-------------|
| `num_v_heads` | `linear_num_value_heads` | `% world_size == 0` |
| `num_k_heads` | `linear_num_key_heads` | `% world_size == 0` |
| `in_proj_qkv` (merged) | Q=`key_dim_global`, K=`key_dim_global`, V=`value_dim_global` | Each chunk `% world_size == 0` |
| `in_proj_z` | `value_dim_global` | `% world_size == 0` |
| `in_proj_b/a` | `num_v_heads_global` | `% world_size == 0` |
| `A_log / dt_bias` | `num_v_heads_global` | `% world_size == 0` |
| `conv1d` (Q block) | `key_dim_global` | `key_dim / world_size` channels per rank |
| `conv1d` (V block) | `value_dim_global` | `% world_size == 0` |
| `out_proj` | `value_dim_global` | Row linear dim 1 `% world_size == 0` |

Where:
- `key_dim_global = linear_num_key_heads * linear_key_head_dim`
- `value_dim_global = linear_num_value_heads * linear_value_head_dim`

### 3c. MoE Experts

| Component | Global dim | Shard dim | Requirement |
|-----------|-----------|-----------|-------------|
| gate/up_proj | `moe_intermediate_size` | dim 0 | `% world_size == 0` |
| down_proj | `moe_intermediate_size` | dim 1 | `% world_size == 0` |

For NVFP4/MXFP4 MoE:
- gate/up packed dim0: `moe_intermediate_size / world_size` per rank
- down packed dim1: `(moe_intermediate_size / pack_factor) / world_size` per rank

### 3d. Shared Expert MLP

Same rules as standard MLP with `shared_expert_intermediate_size`:
- Column linear (gate/up): `shared_expert_intermediate_size % world_size == 0`
- Row linear (down): `shared_expert_intermediate_size % world_size == 0`

### 3e. NVFP4/MXFP4 Scale Alignment

For NVFP4 (group_size=16): after sharding, verify `per_rank_in_dim % 16 == 0` for dim-1 shards.
For MXFP4 (group_size=32): verify `per_rank_in_dim % 32 == 0` for dim-1 shards.
For FP8: verify per-rank boundaries align to `weight_block_size`.

### 3f. Embedding / LM Head

- `embed_tokens`: replicated (not sharded), no divisibility constraint.
- `lm_head`: replicated, no constraint. But if `tie_word_embeddings` is true, verify `lm_head` doesn't exist as a separate tensor (should reuse `embed_tokens.weight`).

---

## Phase 4: Report Findings

Present results in a structured format:

### Model Summary
```
Architecture: Qwen3_5MoeForConditionalGeneration
Model Type: qwen3_5_moe (Hybrid MoE with linear attention)
Quantization: nvfp4 (compressed-tensors format)
Layers: 48 (36 linear_attention + 12 full_attention)
Hidden size: 3072
Full attention: 32 Q heads, 2 KV heads, head_dim=256
Linear attention: 16 K heads, 64 V heads, head_dim=128
MoE: 256 experts, top-8, intermediate=1024
Shared expert: intermediate=1024
```

### Tensor Format Validation
```
[OK] Linear attention layers (BF16, in ignore list)
[OK] Full attention layers (NVFP4 compressed-tensors: weight_packed + weight_scale + weight_global_scale)
[OK] MoE experts (NVFP4 compressed-tensors: per-expert weight_packed)
[OK] Shared expert MLP (NVFP4 compressed-tensors: weight_packed)
[WARN] lm_head: in ignore list, stored as BF16
```

### Multi-Rank Compatibility
```
| Component | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|-----------|-------|--------|--------|--------|
| Full attn Q heads (32) | OK | 16 | 8 | 4 |
| Full attn KV heads (2) | OK | 1 | repl(2) | repl(4) |
| GDN K heads (16) | OK | 8 | 4 | 2 |
| GDN V heads (64) | OK | 32 | 16 | 8 |
| MoE inter (1024) | OK | 512 | 256 | 128 |
| Overall | OK | OK | OK | OK |
```

### Issues Found

Flag any problems:
- `[ERROR]` — Will fail to load (missing tensors, wrong names, indivisible dims)
- `[WARN]` — May cause issues (unusual format, edge case)
- `[INFO]` — Informational (features detected, fallback paths)

---

## Phase 5: Common Issues Reference

### Known tensor name mismatches

| Model source | Packed weight name | xinfer loader support |
|-------------|-------------------|---------------------|
| ModelOpt NVFP4 | `weight` (U8) | Single-GPU: OK. Multi-GPU merged chunks: requires `weight` fallback in `load_merged_chunks` |
| Compressed-tensors NVFP4 | `weight_packed` | OK everywhere |
| Legacy MXFP4/NVFP4 | `blocks` | OK (final fallback) |

### GatedDeltaNet TP-safe loading

The `in_proj_qkv` tensor requires special merged-chunk loading for multi-GPU:
- `MergedParallelColumnLinear::load_merged_chunks` splits Q, K, V independently
- For quantized models (FP8/NVFP4/MXFP4), each chunk must be sharded within the quantized domain
- For BF16 (ignore-listed layers), falls through to the unquantized path

### Replicated KV heads

When `num_kv_heads < world_size`:
- `kv_head_shard` uses replicated mode: `ranks_per_kv_head = world_size / num_kv_heads`
- Each KV head is shared by `ranks_per_kv_head` consecutive ranks
- Requires `world_size % num_kv_heads == 0`

---

## Key Source Files

| File | Relevance |
|------|-----------|
| `src/models/layers/distributed.rs` | TP column/row linear, `load_merged_chunks`, `kv_head_shard` |
| `src/models/layers/linear.rs` | `LnFp8`, `LnNvfp4`, `LnMxfp4` loaders, tensor name resolution |
| `src/models/layers/deltanet.rs` | `GatedDeltaNet` loading, `is_weight_quantized`, projection sharding |
| `src/models/layers/attention.rs` | Full attention QKV loading, packed QKV for FP8 |
| `src/models/layers/moe.rs` | `FusedMoeNvfp4`, `FusedMoeMxfp4`, `FusedMoeFp8` expert loading |
| `src/utils/config.rs` | `QuantConfig`, `normalize_compressed_tensors`, `should_skip_module` |
