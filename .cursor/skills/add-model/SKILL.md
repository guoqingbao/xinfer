---
name: add-model
description: >-
  Adapt and port new LLM model architectures to this xinfer project.
  Use when the user asks to add, port, support, or adapt a new model
  (e.g. Llama, Gemma, Qwen, GPT-OSS, DeepSeek, or any HuggingFace
  architecture) including safetensors and GGUF formats, Dense and MoE
  architectures, and quantization formats (MXFP4, NVFP4, FP8, ISQ).
---

# Add Model — Adapt New LLM Architectures to xinfer

## Phase 0: Gather Required Information

Before starting, the agent **must** collect the inputs below. If any are missing, **ask the user explicitly**.

**Resolution order** (try top to bottom, stop at the first that succeeds):

1. **Local model path** (preferred) — If the user provides a local directory containing safetensors, read `config.json` and inspect weight tensors directly from disk. If the user provides a `.gguf` file, extract metadata and tensor info from the GGUF header (GGUF is self-contained; there is no separate `config.json`). No further input is needed in either case.
2. **HuggingFace model ID** — Look for the model in the local HuggingFace Hub cache (`~/.cache/huggingface/hub/`). If not cached, fetch `config.json` from the HuggingFace model repo. If that also fails, fall back to step 3.
3. **Manual input** — Ask the user to provide:
   - `config.json` contents (for safetensors models) **or** GGUF metadata (for GGUF models).
   - Weight tensor info (names, shapes, dtypes). The user can obtain this by clicking a weight file in the HuggingFace model repo.

**Additionally**, ask the user if they can provide the **Python reference implementation** (`modeling_<arch>.py` from HuggingFace Transformers). This is not strictly required, but significantly improves accuracy — it clarifies the exact forward pass, attention variants, MoE routing, activation functions, and normalization order that config fields alone cannot fully describe.

| Input | How to obtain |
|-------|---------------|
| **HuggingFace model ID** (e.g. `google/gemma-4-26B-A4B-it`) | User provides, or infer from context |
| **Model config** (`config.json`) | Fetch from HF: `https://huggingface.co/<id>/blob/main/config.json`. Not needed if local model path is provided. |
| **HF tensor info** (weight names + shapes) | User provides, or read from local safetensors with `scripts/inspect_weights.py` (create the script if it doesn't exist) |
| **GGUF metadata + tensor info** (if GGUF support needed) | User provides, or extract from local `.gguf` with `scripts/inspect_gguf.py` (create the script if it doesn't exist) |
| **Python reference implementation** (optional but recommended) | Fetch `modeling_<arch>.py` from the HuggingFace Transformers GitHub repo |
| **Local model path** (optional) | User provides path containing `config.json` + `*.safetensors` or `*.gguf` |

### Extracting info from local weights

For safetensors (Python required):

```python
# scripts/inspect_weights.py
import json, sys, struct
path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
for k, v in sorted(header.items()):
    if k != "__metadata__":
        print(f"{k}\t{v.get('shape')}\t{v.get('dtype')}")
```

For GGUF, use the `gguf_helper` CLI or Python `gguf` library to list tensors and metadata keys.

---

## Phase 1: Analyze the Architecture

Read and compare:

1. **`config.json`** — identify: `architectures`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `hidden_act`, `vocab_size`, `rope_theta`, `partial_rotary_factor`, `sliding_window`, `tie_word_embeddings`.
2. **MoE fields** (if any): `num_experts`, `num_experts_per_tok` / `top_k_experts`, `moe_intermediate_size`, `norm_topk_prob`, `routed_scaling_factor`.
3. **Special features**: `attention_k_eq_v`, `layer_types` array, `layer_scalar`, `final_logit_softcapping`, `attn_logit_softcapping`, dual `head_dim`, dual RoPE, per-expert scaling, shared experts, etc.
4. **Python reference** — understand the forward pass, especially:
   - Attention: GQA/MQA, head dim, RoPE variant, sliding window, QK norm, V norm
   - MLP: standard SiLU-gate, GeluTanh, custom SwiGLU, etc.
   - MoE: router structure, expert computation, parallel dense+MoE, renormalization
   - Layer structure: pre-norm vs post-norm, residual connections, layer scalar

Identify which **existing model** in `src/models/` is the closest match. Use it as a starting template.

### Architecture classification

| Type | Characteristics | Example models |
|------|----------------|----------------|
| **Dense** | Standard transformer, no MoE | Llama, Gemma3, Phi4 |
| **MoE** | Router + experts per layer | Qwen3-MoE, Gemma4-MoE |
| **Hybrid MoE** | Dense MLP + MoE in parallel per layer | Gemma4 (dense+MoE parallel) |
| **Multimodal** | Vision encoder + language model | Gemma3-VL, Qwen3-VL |

---

## Phase 2: Check if New Kernels Are Needed

If the model requires operators not in `attention.rs` or `src/models/layers/`:

1. Clone `attention.rs` to a local sibling directory (if not already present):
   ```bash
   cd .. && git clone https://github.com/guoqingbao/attention.rs.git
   ```
   Then switch to the commit that this project relying on.
2. Point `xinfer/Cargo.toml` to local `attention.rs`:
   ```toml
   # In [dependencies], change the git URL to:
   attention-rs = { path = "../attention.rs", ... }
   ```
3. Add **both CUDA and Metal** kernel implementations:
   - CUDA kernel: `attention.rs/src/kernels/src/<name>.cu`
   - Metal kernel: `attention.rs/src/metal-kernels/src/<name>.metal`
   - FFI bindings: `attention.rs/src/kernels/src/ffi.rs`
   - Rust wrapper: `attention.rs/src/<name>.rs`
   - Update `attention.rs/src/lib.rs`: add `pub mod <name>;`
   - Update `attention.rs/src/kernels/build.rs`: add `rerun-if-changed` for `.cu`
   - Update `attention.rs/src/metal-kernels/build.rs`: add to `METAL_SOURCES`
4. For CUDA BF16 kernels (Ampere+ only), guard with `#ifndef NO_BF16_KERNEL` and provide F16 fallback or dummy stubs for older GPUs.
5. Verify compilation: `cargo check --features metal` (macOS) or `cargo check --features cuda` (Linux).

---

## Phase 3: Implement the Model

### 3a. Create model file

Create `src/models/<arch>.rs`. Follow the established pattern from the closest existing model:

**Key struct pattern:**
```
pub struct <Arch>DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    moe: Option<<Arch>MoE>,      // if MoE
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    // ... additional norms for hybrid MoE
}

pub struct <Arch>ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<<Arch>DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    // ...
}
```

**Required public methods on `<Arch>ForCausalLM`:**
```rust
pub fn new(vb: &VarBuilderX, comm: Rc<Comm>, config: &Config, dtype: DType,
           is_rope_i: bool, device: &Device,
           progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>) -> Result<Self>

pub fn forward(&self, input_ids: &Tensor, positions: &Tensor,
               kv_caches: Option<&Vec<(Tensor, Tensor)>>,
               input_metadata: &InputMetadata,
               _embeded_inputs: bool) -> Result<Tensor>

pub fn forward_embedding(&self, ..same args..) -> Result<Tensor>

pub fn get_vocab_size(&self) -> usize
```

The 5th `_embeded_inputs: bool` argument is required by the `model_call!` macro.

### 3b. Weight loading paths

The model must support **two VarBuilder types** via `vb.is_qvar_builder()`:

| Path | VarBuilder | Weight prefix pattern |
|------|-----------|----------------------|
| **HF safetensors** | `Either::Left` | `language_model.model.layers.{i}` (multimodal) or `model.layers.{i}` |
| **GGUF** | `Either::Right` | `model.layers.{i}` (maps from `blk.{i}`) |

For `norm` and `lm_head`:

| Component | HF prefix | GGUF prefix |
|-----------|-----------|-------------|
| Final norm | `language_model.model.norm` | `model.norm` |
| LM head (untied) | `lm_head` | `model.output` |
| LM head (tied) | `language_model.model.embed_tokens` | `model.embed_tokens` |

### 3c. MoE construction

For models with MoE, the `Gemma4MoE` / `FusedMoe` selection pattern:

```rust
let moe = if is_qvar_builder {
    FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?
} else if quant_config == "mxfp4" {
    FusedMoeMxfp4::new(config, vb.pp("mlp"), comm.clone(), dtype)?
} else if config.quant.is_some() {
    FusedMoeISQ::new(config, vb.pp("mlp"), comm.clone(), dtype)?
} else {
    FusedMoe::new(config, vb.pp("mlp"), comm.clone(), dtype)?
};
```

**Important**: If the model's router gate is NOT at the standard `mlp.gate` path (e.g. Gemma4 uses `router.proj`), use `FusedMoe::new_with_gate(config, gate_vb, experts_vb, ...)` and `FusedMoeISQ::new_with_gate(...)`.

For GGUF models with **packed** `ffn_gate_up_exps` (instead of separate `ffn_gate_exps` + `ffn_up_exps`), `FusedMoeGGUF::new()` auto-detects and handles this.

### 3d. Packed expert layout

If the model uses packed `gate_up_proj` with a non-standard layout, add the architecture name to the layout resolvers in `src/models/layers/moe.rs`:

- `resolve_packed_gate_up_layout()` — `InterPacked` if shape is `[experts, 2*intermediate, hidden]`
- `resolve_packed_down_layout()` — `HiddenInter` if shape is `[experts, hidden, intermediate]`

---

## Phase 4: Register the Model

### 4a. `src/models/mod.rs`
```rust
pub mod <arch>;
```

### 4b. `src/utils/config.rs`
Add variant to `ModelType` enum:
```rust
pub enum ModelType {
    // ...
    <Arch>,
}
```

### 4c. `src/utils/mod.rs` — `get_arch_rope` function
Add mappings in order:

1. **GGUF canonical_arch**: `"<gguf_arch>" => "<HFArchitectureName>".to_string()`
2. **Architecture to ModelType + chat template**: Add match arm for HF architecture names
3. **Rope type selection**: `("<arch_lower>", false)` in the rope map
4. **HF config parsing** (multimodal wrapper): If the model wraps `text_config`, add handler to extract nested config, MoE config, and `rope_parameters`
5. **GGUF extra_config_json**: If the model has special GGUF metadata (sliding_window_pattern, layer_types, etc.), add extraction block
6. **GGUF MoE config**: Add `mod_cfg` construction from GGUF expert metadata
7. **`hidden_act` override**: If the model uses a non-Silu activation, override after config construction
8. **`require_model_penalty()`**: Add architecture names
9. **Multimodal GGUF warning**: Add if applicable

### 4d. `src/core/runner.rs`
1. Add `use crate::models::<arch>::<Arch>ForCausalLM;`
2. Add `<Arch>(Arc<<Arch>ForCausalLM>)` to `Model` enum
3. Add `<Arch> => <Arch>ForCausalLM` to `build_model!` macro
4. Add `<Arch> => EmbedInputs` to `graph_wrapper!` macro (or `ImageData` for multimodal)
5. Add `<Arch> => false` to both `model_call!` invocations (or `true` / image handling for multimodal)
6. Add `ModelType::<Arch>` to `disable_flash_attn` if needed
7. Add `Model::<Arch>(model) => model.get_vocab_size()` to `get_vocab_size`

### 4e. `src/server/parser.rs`
1. Add `ModelType::<Arch>` to `ToolConfig::for_model_type()`
2. Add `ModelType::<Arch>` to `parser_name_for_model()`
3. Add `ModelType::<Arch>` to structured output format handling

---

## Phase 5: Build and Verify

### Build commands

| Platform | Command |
|----------|---------|
| **macOS (Metal)** | `cargo build --release --features metal` |
| **CUDA (basic)** | `cargo build --release --features "cuda,flashinfer"` |
| **CUDA (full)** | `cargo build --release --features "cuda,flashinfer,nccl"` |
| **CUDA (sm90+)** | Add `cutlass` feature: `--features "cuda,flashinfer,nccl,cutlass"` |

If permission errors occur on `target/`, use: `CARGO_TARGET_DIR=/tmp/xinfer-check cargo check --features metal`

### Verify compilation
```bash
# Check xinfer compiles
cargo check --features metal   # or cuda

# Check attention.rs compiles (if modified)
cd ../attention.rs && cargo check --features metal   # or cuda
```

Fix all errors and warnings before proceeding.

---

## Phase 6: Test the Model

### Start the server

**Single-GPU:**
```bash
# Metal (macOS)
cargo run --release --features metal -- --m <model_id_or_path> --port 8080 # or use --w to specify local model path

# CUDA
cargo run --release --features "cuda,flashinfer,cutlass" -- --m <model_id_or_path> --port 8080
```

**Multi-GPU (CUDA):**
```bash
# --d used to specify device ids
xinfer --m <model_id_or_path> --port 8080 --d 0,1
```

### Before retrying model loading

Always kill all previous instances and verify GPU memory is freed:
```bash
# Kill all xinfer and runner processes
pkill -f xinfer;
sleep 2

# Check GPU memory (CUDA)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Check GPU memory (Metal)
sudo powermetrics --samplers gpu_power -i 1000 -n 1 | grep 'GPU'
```

Ensure the target GPU(s) have sufficient free memory for the model before loading.

### Send test requests

```bash
# Basic completion, depend on the api server port started, default 8000
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_id>",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 64,
    "temperature": 0.7
  }' | python3 -m json.tool

# Streaming
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_id>",
    "messages": [{"role": "user", "content": "Write a haiku about Rust."}],
    "max_tokens": 64,
    "stream": true
  }'
```

### Debugging common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| **Panic at weight loading** | Wrong VarBuilder prefix or tensor shape mismatch | Check HF vs GGUF prefix mapping; verify tensor shapes match config |
| **NaN/Inf in output** | Missing activation, wrong dtype, or norm misconfiguration | Verify `hidden_act`, check if model needs GeluPytorchTanh vs Silu; check norm dtype (F32 for GGUF/ISQ) |
| **Gibberish output** | Wrong RoPE params, wrong head_dim, or tied embeddings misconfigured | Verify `rope_theta`, `partial_rotary_factor`, `head_dim`, `tie_word_embeddings` |
| **Server crash on decode** | KV cache shape mismatch or sliding window misconfigured | Verify `head_dim` per attention layer matches KV cache allocation |
| **CUDA OOM** | Model too large for GPU | Use ISQ quantization (`--quant q4k`), or use multi-GPU with `--tensor-parallel` |
| **Slow performance** | Flash attention disabled or wrong features | Ensure `flashinfer` feature is enabled; check `disable_flash_attn` isn't matching your model |

### Precision validation

Compare outputs against the Python reference:
1. Run the same prompt through both the Python HF model and xinfer
2. Compare logits for the first few tokens (top-5 should match)
3. For MoE models, verify expert routing produces the same top-k experts

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `src/models/<arch>.rs` | Model implementation |
| `src/models/mod.rs` | Module registration |
| `src/models/layers/moe.rs` | MoE layer (FusedMoe, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4) |
| `src/models/layers/attention.rs` | Attention layer (handles GQA, RoPE, sliding window) |
| `src/models/layers/mlp.rs` | Dense MLP layer |
| `src/models/layers/rotary_emb.rs` | RoPE implementations (standard, scaling, YaRN) |
| `src/models/layers/mod.rs` | VarBuilderX definition |
| `src/utils/config.rs` | Config struct, ModelType enum, MoEConfig |
| `src/utils/mod.rs` | Config parsing (HF + GGUF), architecture registration, chat templates |
| `src/core/runner.rs` | Model enum, build_model!, model_call!, graph_wrapper! macros |
| `src/server/parser.rs` | Tool parsing, structured output per model type |
| `Cargo.toml` | Feature flags: metal, cuda, flashinfer, nccl, cutlass |
| `build.sh` | Build script |
