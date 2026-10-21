# LoRA Multi-Tenant Serving

xInfer serves a single base model to many tenants, each with its own LoRA
(Low-Rank Adaptation) adapter. The base model's weights are shared across all
tenants (no per-tenant model duplication); each tenant's adapter delta is
applied per-request. This is the Punica pattern: one base, many lightweight
adapters, selected at request time.

## What is LoRA

A LoRA adapter is a low-rank decomposition of a weight delta. For a base
weight `W` of shape `[out_features, in_features]`, the adapter stores two small
matrices:

- `A` of shape `[in_features, rank]` (the down-projection)
- `B` of shape `[rank, out_features]` (the up-projection)

The effective weight for a tenant is `W' = W + scaling * (B @ A)`, where
`scaling = alpha / rank`. Because the `rank` is small (typically 4-64), the
adapter is a tiny fraction of the base model's parameters. At inference, the
delta is computed as `scaling * (x @ A) @ B` and added to the base output
`x @ W` — two small matrix multiplies instead of a full-weight update.

Adapters are trained offline (with Hugging Face `peft` or similar) and shipped
as an adapter directory: an `adapter_config.json` (the rank, the alpha, the
target modules) + an `adapter_model.safetensors` (the `lora_A`/`lora_B`
tensors). xInfer loads that directory at startup through the common model
loader.

## How to make a LoRA adapter

1. Fine-tune your base model with a LoRA library (for example, Hugging Face
   `peft`):

   ```python
   from peft import LoraConfig, get_peft_model
   lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
   model = get_peft_model(base_model, lora_config)
   # ... train ...
   model.save_pretrained("./my-tenant-adapter")
   ```

2. The saved directory contains `adapter_config.json` (the rank/alpha/targets)
   + `adapter_model.safetensors` (the A/B weights). xInfer reads the
   `adapter_config.json`; if your tooling writes a differently-named config,
   point the loader at a directory layout it understands.

## How to use it in xInfer

```bash
# Load the base model + two LoRA adapters at startup
xinfer --m Qwen/Qwen3-8B \
  --lora finance=/adapters/finance \
  --lora legal=hf-ai/legal-qa-lora \
  --lora-max-tenants 32 \
  --lora-max-bytes 536870912 \
  --lora-eviction-policy lru \
  --lora-pin finance \
  --server
```

- `--lora NAME=SOURCE` (repeatable): load a LoRA adapter. `NAME` is the
  API-facing tenant key; `SOURCE` is a Hugging Face id or a local directory,
  resolved by the common model loader (the same download + cache + local-dir
  path the base model uses). When no `NAME=` prefix is given, the source
  string is used as both the name and the source.
- `--lora-max-tenants N`: the residency count budget (default 16).
- `--lora-max-bytes N`: the residency byte budget (default 256 MiB).
- `--lora-eviction-policy {lru,fifo}`: the eviction policy (default `lru`).
- `--lora-pin NAME` (repeatable): mark a tenant as pinned (never evicted, the
  SLA guarantee).
- `--lora-allow-base-fallback`: when set, an unknown adapter name falls back to
  the base model instead of returning a 404.

Per-request selection uses the OpenAI-compatible `model` field as a
`base:adapter` composite:

```json
{ "model": "Qwen/Qwen3-8B:finance", "messages": [...] }
```

- `model: "Qwen/Qwen3-8B"` (no `:`) runs the base model with no adapter.
- `model: "Qwen/Qwen3-8B:finance"` runs the base model + the `finance` adapter.
- The left of the `:` must match the exposed base model id, or the request is a
  400. The right must be a loaded adapter name, or it is a 400 (or a base-model
  fallback when `--lora-allow-base-fallback` is set).

**Additive multi-adapter stacking.** The right of the `:` may be a `+`-separated
list of adapters (e.g. `base:style+domain`). All listed adapters are composed
additively on one forward pass:

```
out = base(x) + sum_j scaling_j * (x @ A_j) @ B_j
```

where `j` ranges over the stacked adapters. This is the mistral.rs `LoraLinear`
fast path (the stacked-bank + batched-GEMM + sum-over-adapter-dim). A single
adapter (no `+`) is the prior behavior. On CUDA, the stacked fused kernel
computes the composition in two launches; otherwise the candle additive
composition is used. Each stacked adapter must be a loaded adapter name (the
same validation as the single-adapter form).

The `GET /v1/models` endpoint lists the base model id + one composite
`base:adapter` id per loaded adapter. Each adapter entry carries a `parent`
field (the base model id, the vLLM/SGLang convention) so the adapter-to-base
relationship is explicit; the base entry has `parent: null`. A client
discovers the selectable adapter names from this listing, then composes any
subset with the `+` form above.

## How it works in the runtime

The architecture respects xInfer's static-model + dynamic-request design:

```
startup:
  --lora NAME=SOURCE ...  →  init_lora_registry
    → for each (name, source): the common model loader resolves the source
      (HF download + cache + local dir), reads the adapter_config.json + the
      adapter_model.safetensors, builds the per-module LoraAdapter set
    → LoraRegistry { name: [LoraAdapter{module, A, B, scaling}), ...] }
    → set_global_registry(Arc<Mutex<registry>>)   (process-wide; the wrapped
                                                  layers read it)
    → startup log: "LoRA: N adapter(s) loaded (...), budget ... / ... bytes,
       policy ..., pinned ..."

per-request:
   request.model = "base:finance" (or "base:style+domain" for a)
     → server parses the composite, validates the base + the adapter name(s),
       sets SamplingParams.lora_tenant (single) or lora_tenants (the stack)
     → the tenant(s) travel on the Sequence into the runner
     → runner.run(): builds the per-row tenant mapping (the LoraExecution)
         - single tenant in the batch: set_active_tenant("finance")
         - multiple tenants in the batch: set_lora_execution(LoraExecution{
             row_tenants: [...], row_stacks: [...]})
     → model.forward()
         → each wrapped linear (q/k/v/o_proj, the gate/up/down_proj, the
           fused qkv/gate_up, the lm_head, the embedding, the MoE experts):
             base = x @ W
             if a tenant is active + it has an adapter for this module:
                 out = base + scaling * (x @ A) @ B     (the TP-sharded A/B)
             if a row carries a tenant stack (the additive composition):
                 out = base + sum_j scaling_j * (x @ A_j) @ B_j
             else:
                 out = base
```

The key pieces:

- **The adapter registry** (`set_global_registry`): the `LoraRegistry` is stored
  process-wide so the model layers can consult it without threading the
  registry through every model constructor. This is the DRY injection point —
  the wrapped linears read the global registry + the per-request tenant, with
  no per-model signature changes.
- **The per-request tenant selection**: the single-tenant case uses a thread-
  local (`set_active_tenant`/`active_tenant`); the multi-tenant batch uses the
  `LoraExecution` (the per-row tenant mapping). Both are safe because the
  forward pass is serialized by the actor model (the `FwdD` actor, the
  `IpcChannel` sole socket writer), so only one execution context is active at
  a time.
- **The marked layers** (`with_lora_module`): the attention q/k/v/o, the MLP
  gate/up/down, the fused qkv/gate_up, the lm_head, the embedding, + the MoE
  experts are marked with their module name. The `forward` applies the active
  tenant's delta for that module when present. Unmarked layers (the empty
  module name) take the fast path (no registry lookup).
- **The GPU tensors**: the A/B are loaded onto the device (the
  `safetensors::load` into the `device`), so the delta is computed on the GPU
  (no CPU round-trip). The `scaling` is applied via a `broadcast_mul` with a
  scalar tensor.

## What is implemented

The LoRA feature covers the full multi-tenant serving surface:

- **Adapter loading** — via the common model loader (the HF download + the
  cache + the local dir), the PEFT `adapter_config.json` parse (the per-module
  rank/alpha, the target/exclude modules, the peft_type check, the
  inference_dtype), the safetensors A/B load, the cast to the model's compute
  dtype.
- **The stacked A/B bank** — the `StackedLoraBank` type (the `[max_loras, ...]`
  stacked A/B, the slot-based `set_lora`/`get_lora`) is implemented, tested, +
  wired into the `LoraRegistry` (the registry carries an optional bank, the
  `register` populates the tenant's bank slot, the `unregister` frees it, the
  `init_lora_registry` enables the bank sized to the tenant budget). This is
  the S-LoRA memory model that scales to a large adapter catalog.
- **The TP sharding** — the A/B are sharded to match the base linear's TP
  layout (the Column-parallel shards B on the out dim, the Row-parallel shards
  A on the in dim, the Replicated is full). The `LoraParallelism` enum + the
  per-rank slicing in the `delta`.
- **The fused-module expansion** — the `qkv_proj`/`gate_up_proj` fused sites
  get the combined constituent A/B (the uniform-rank column-concat, the
  variable-rank block-fill via the `b_block_fill`).
- **The MoE expert LoRA** — the `MoeLoraState` (the 3D per-expert A/B) + the
  `FusedMoe` per-expert dense path (instead of the fused `moe_gemm` kernel when a
  tenant is active). The mixed-tenant batch is supported: the per-expert tokens
  are grouped by tenant (the `LoraExecution.row_tenants`), each group using
  its own tenant's `MoeLoraState` (the `global_moe_lora_for`).
- **The MLA correction** — the `MlaAttention` re-derives the absorbed
  `w_uk`/`w_uv_t` from the effective `kv_b_proj` weight (the base + the delta)
  when a tenant is active.
- **The embedding + lm_head LoRA** — the `VocabParallelLinear` (the lm_head)
  wrapping + the `apply_embedding_lora` in every model's `embed_forward`.
- **The QLoRA** — the quantized adapter A/B storage (the FP8 storage, the on-
  the-fly dequantize in the delta).
- **The batched multi-tenant** — the `LoraExecution` per-row tenant mapping,
  one forward pass serves many tenants.
- **The additive multi-adapter stacking** — a request may compose several
  adapters on one forward pass (`base + sum_j scaling_j * (x @ A_j) @ B_j`).
  The `model` field accepts a `+`-separated adapter list (e.g.
  `base:style+domain`); the runner builds a per-row tenant stack
  (`LoraExecution.row_stacks`); the fused stacked kernel (CUDA) or the candle
  additive composition (the fallback) applies it.
- **The fused CUDA kernel** — the attention-rs shrink/expand pair computes the
  multi-tenant delta in two kernel launches (the S-LoRA throughput path),
  replacing the per-tenant per-layer candle matmul pair. The stacked variant
  composes a per-row adapter list via a CSR slot layout.
- **The residency management** — the byte budget, the count budget, the LRU/FIFO
  eviction, the lease, the pin, the two-tier LRU, the alias→generation no-
  downtime swap, the on-demand load.
- **The operator surface** — the repeatable `--lora` CLI, the `/v1/models`
  composite listing, the `base:adapter` request parsing, the startup + runtime
  logging, the `/v1/lora` management RPC (list/load/unload).
- **The adversarial fixes** — the CUDA-graph disable when LoRA is active, the
  dtype-correct delta, the DFlash gating.

The explicit non-goals: the quantized-base serving (the AWQ/GPTQ base; the QLoRA
adapter storage is done), the LoRA training (xInfer is an inference runtime; the
adapters are trained externally + loaded here).

### Known limitations (the honest boundaries)

- **The MoE expert LoRA is per-tenant + mixed-tenant batched.** The
  `GLOBAL_MOE_LORA` is keyed by tenant (the `HashMap<tenant, MoeLoraState>`);
  the `global_moe_lora_for(tenant)` reads a specific tenant's state. The
  `FusedMoe::forward_with_routing_moe_lora` groups each expert's tokens by
  tenant (the `LoraExecution.row_tenants`) + uses each tenant's own
  `MoeLoraState`, so one forward pass serves a batch mixing multiple tenants
  correctly (no cross-tenant contamination). Proven by the
  `lora_moe_per_tenant_isolation` test.
- **The CUDA graph is disabled when any LoRA adapter is configured.** The
  runner (runner.rs) disables the graph globally when `econfig.lora.adapters` is
  non-empty, because the graph would bake in the LoRA branch at capture time.
  This is correct but means a LoRA-enabled instance does not use the decode
  CUDA graph (the slight perf cost, the correctness guarantee). A per-tenant
  graph capture is the future optimization.
- **The QLoRA dequantize is on-the-fly.** The quantized A/B are dequantized to
  the activation dtype on every delta compute (lora.rs `dequantize`). This is
  correct but adds a dequant cost per forward; a pre-dequantized cache is the
  future optimization.
- **The stacked bank is the registry's dense storage backend.** The
  `StackedLoraBank` is wired into the `LoraRegistry` (the `with_stacked_bank`,
  the `register`/`unregister` populate + free the bank slots). The per-tenant
  `Vec<LoraAdapter>` remains the serving source of truth; the bank is the
single-allocation dense backend that makes the thousands-of-adapters density
   memory-efficient.

## Interaction with the other features

### The dtype handling across the stack (the main model + the LoRA + the DFlash)

The dtype correctness is the hardest part of the LoRA integration, because three
components with different dtypes run in the same forward pass:

- **The main model** — the weights may be quantized (NVFP4, FP8, MXFP4, GGUF,
  ISQ), but the activations run in the model's compute dtype (F16 for the
  `is_f16_mode` models, BF16 for the others, F32 for the higher-precision
  models). The quantized linears (the `LnNvfp4`, the `LnFp8`, the `QLinear`)
  dequantize internally + produce the activation-dtype output.
- **The LoRA adapter** — the A/B are stored in the checkpoint's dtype (usually
  F32 for PEFT, or FP8/NF4 for QLoRA). They are cast to the model's compute
  dtype at load (the `load_adapters_from_paths`), + re-cast to the activation
  dtype at the time (the `delta` method) to match the input.
- **The DFlash drafter** — a separate model with its own dtype (the
  `DFlashDrafter.dtype`), cast at its inputs (the `to_dtype(self.dtype)`
  on the target hidden states + the noise embedding). The DFlash is gated off
  when a LoRA tenant is active (the drafter would not match the LoRA-applied
  target), so the DFlash + LoRA dtypes never mix in one pass.

The delta is computed as `scaling * (x @ A) @ B`, where `x` is the activation
(the activation's compute dtype). The dtype discipline that prevents the crash:

1. **The A/B are cast to the activation dtype before the matmul.** The `delta`
   method does `a_shard.to_dtype(x2.dtype())` + `b_shard.to_dtype(x2.dtype())`
   (lora.rs), so the `muls run in the activation dtype regardless of the
   A/B storage dtype. This handles the NVFP4-base + F16-activation case (the
   A/B loaded in F16, the activation F16, the matmul F16).
2. **The scale is cast in the activation dtype, not F32.** The `scaling` is
   a Rust `f32`; `Tensor::new(self.scaling, ...)` infers F32. A F32 scalar
   broadcast-multiplied against an F16 activation is a dtype mismatch (the
   crash: "dtype mismatch in mul, lhs: F16, rhs: F32"). The fix casts the
   scale to the activation dtype: `Tensor::new(self.scaling, up.device())?
   .to_dtype(up.dtype())?` before the `broadcast_mul`. This is applied in all
   three delta paths (the `LoraAdapter::delta`, the `MoeLoraState::gate_up_
   delta`/`down_delta`, the `LoraAdapter::weight_delta`).
3. **The QLoRA dequantize targets the activation dtype.** The `dequantize`
   method casts the quantized A/B (the FP8/NF4) to the `target_dtype` (the
   activation dtype passed in), so the dequantized A/B match the activation.

The invariant: every operand in the delta matmul chain (the `x`, the `A`, the
`B`, the `scale`) is in the activation dtype at the point of use. The base
linear's output is also in the activation dtype (the quantized linears
dequantize to the activation dtype), so the `base + delta` add is same-dtype.

The per example from the crash log (the Qwen3.5 NVFP4 base, the F16
activations, the F32-stored LoRA):
- The A/B load in F32 (the PEFT storage), cast to F16 at load (the model's
  compute dtype).
- At delta time: the `x` is F16 (the activation), the A/B are F16 (the load
  cast), the matmul is F16, the scale is cast to F16 (the fix), the `base +
  delta` is F16 + F16. No mixed-dtype op. No crash.

### Speculative decoding

The MTP path (the target-in heads) sees the LoRA delta (the MTP head consumes
the target model's hidden states, which are computed through the wrapped
q/k/v_proj). The DFlash path (the separate drafter model) does not see the
LoRA delta (the drafter is a distinct model, not the target's wrapped linears).
So when a LoRA tenant is active, the DFlash path is disabled (the runner gates
it on `active_tenant().is_none()`), and the decode falls back to the MTP/plain
path (which does get the LoRA delta). A DFlash + LoRA combination requires a
per-tenant DFlash drafter (the extension, not the current path).

### Memory management

The `LoraRegistry` manages residency with multiple bounds:
- **The count budget** (`max_tenants`, default 16): the max concurrent resident
  tenants. When exceeded, the LRU tenant is evicted (the GPU memory freed).
- **The byte budget** (`max_bytes`, default 256 MiB): the max total GPU bytes of
  resident adapter A/B. When exceeded, the LRU tenant is evicted.
- **The two-tier LRU** — the `registered` tier (the weights on GPU, the large
  capacity) + the `active` tier (the in-batch bank slots, the `max_active`
  cap). Eviction from active frees a bank slot (the weights stay registered);
  eviction from registered unloads the weights (the GPU memory freed).
- **The lease** (`acquire_lease`/`release_lease`): a tenant held by an in-
  flight request is leased (not evicted mid-forward).
- **The pin** (`pin`/`unpin`): a pinned tenant (the SLA guarantee) is never
  evicted, even under budget pressure.
- **The alias→generation** (`swap_alias`/`generation_for`/`drop_unreferenced_
  generations`): a tenant's alias points to an immutable generation; a swap is
  atomic (the in-flight requests keep the old generation, the new requests get
  the new one). The no-downtime adapter deploy.

The base model weights are unchanged (the LoRA delta is computed on-the-fly,
not a merged weight copy). The per-tenant adapter memory is small (the
`rank × in_features` + the `rank × out_features` per target module).

## Verification

The unit tests in `src/core/lora.rs` prove the core logic:

- `lora_delta_is_low_rank_product` — a known A/B produces the expected delta
  (the inclusive: the low-rank product is exact).
- `lora_zero_adapter_is_identity` — a zero A leaves the base unchanged (the
  exclusive: no spurious delta).
- `lora_registry_lookup_and_unregister` — the registered tenant is found, the
  unregistered is not, the unregister removes it.
- `lora_active_tenant_thread_local` — the set + the get round-trip, the clear.
- `lora_tp_sharding_recombines_to_unsharded_oracle` — the per-rank Column
  deltas, recombined, equal the unsharded oracle (the TP correctness).
- `lora_tp_row_sharding_sums_to_unsharded_oracle` — the per-rank Row deltas,
  summed (the all-reduce), equal the unsharded oracle (the Row TP correctness).
- `lora_batched_delta_tenant_delta_is_per_row` — a batch with 2 tenants applies
  each tenant's delta to its own rows (no cross-tenant contamination).
- `lora_registry_byte_budget_evicts` / `lora_registry_lease_protects_from_
  eviction` / `lora_registry_pin_protects_from_eviction` — the residency
  bounds.
- `lora_registry_two_generation_swap` — the no-downtime swap keeps the in-
  flight generation.
- `lora_moe_expert_delta_is_per_expert` / `lora_moe_per_tenant_isolation` —
  the MoE expert delta + the per-tenant isolation.
- `lora_lm_head_delta_applies_to_logits` — the lm_head delta.
- `lora_end_to_end_wrapped_layer_applies_delta` — the wrapped layer applies the
  delta end-to-end.
- `lora_qlora_dequantize_applies_scale` — the QLoRA dequantize.
- `lora_fused_variable_rank_block_fill` — the variable-rank fused B block-fill.
- `lora_site_registry_matching` — the site registry matches the adapter targets.
- `lora_leaf_alias_site_matching` — the runtime `o_proj` matches the checkpoint
  `out_proj` via the leaf alias.
- `lora_spec_decode_interaction_dflash_gating` — the DFlash is disabled when a
  tenant is active.
- `lora_stacked_kernel_math_matches_naive_sum` — the CPU reference for the
  stacked kernel index arithmetic (the CSR slot layout + the additive
  composition equals the naive per-adapter sum).
- `lora_stacked_composition_is_additive_sum` — the end-to-end additive
  composition through `batched_delta` (a row with a 2-adapter stack gets the
  sum of both deltas; a row with no stack gets the zero delta).

The end-to-end (the adapter actually changes the model's output for the
selected tenant) requires a trained adapter + a GPU, so it is verified by the
operator (the `--lora` + the `base:adapter` request; the output differs from
the base model by the adapter's effect). For stacking, the operator sends a
`base:a+b` request and confirms the output reflects the additive composition of
both adapters.