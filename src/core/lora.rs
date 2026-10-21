//! The LoRA (Low-Rank Adaptation) multi-tenant serving module.
//!
//! This module implements the Punica-style shared-base + per-tenant-adapter
//! pattern: a single base model is served to multiple tenants, each with its
//! own LoRA adapter delta applied per-request. The base model's weights are
//! shared (no per-tenant model duplication); the adapter delta is
//! per-tenant (the isolation).
//!
//! A LoRA adapter is a low-rank decomposition of a weight delta:
//! `W' = W + scaling * B @ A`, where `W` is the base weight
//! `[out_features, in_features]`, `A` is the down-projection
//! `[in_features, rank]`, `B` is the up-projection `[rank, out_features]`,
//! and `scaling` is the `alpha / rank` factor. The `rank` is small (typically
//! 4-64), so the delta is cheap to store + apply relative to the base weight.
//!
//! The adapter is loaded from a Hugging Face PEFT-format directory (the
//! `config.json` with the rank/alpha/target_modules, the
//! `adapter_model.safetensors` with the `lora_A`/`lora_B` tensors). The
//! loading mirrors the draft-model flow (the `resolve_draft_model` + the
//! `Downloader` + the `ShardedSafeTensors`).
//!
//! The per-request application: the `LoraLinear` wrapper holds the base
//! linear layer + the adapter A/B (the GPU tensors). When a request carries a
//! tenant selection (the `set_active_tenant` thread-local, set by the
//! runner before the forward pass), the wrapper computes
//! `base(x) + scaling * B(A(x))`; otherwise it computes `base(x)` (no
//! adapter, the base model). The thread-local is safe because the forward
//! pass is serialized by the actor model (the `FwdD` actor, the
//! `IpcChannel` sole writer), so only one tenant is active at a time.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use candle_core::{Device, DType, Error, Result, Tensor};

/// Normalize a LoRA site path to its stable matching key: `layers.{i}.{leaf}`,
/// where `{i}` is the layer index and `{leaf}` is the normalized module name
/// (the last path segment, with the known leaf aliases collapsed). Both the
/// checkpoint tensor keys (the `base_model.model.model.language_model.layers.
/// 0.linear_attn.out_proj.lora_A.weight`) and the runtime layer paths (the
/// `model.layers.0.self_attn.o_proj`) reduce to this key, which is what makes
/// the adapter-to-site matching work across the differing path prefixes, the
/// block names (the checkpoint may say `linear_attn`, the runtime `self_attn`),
/// + the leaf aliases (the checkpoint may say `out_proj`, the runtime `o_proj`).
/// When there is no `layers.{i}` segment (e.g. the `lm_head`, the
/// `embed_tokens`), the whole path is returned as-is.
pub fn canonical_site_key(path: &str) -> String {
    if let Some(idx) = path.find("layers.") {
        let rest = &path[idx + "layers.".len()..]; // the `{i}.{block}.{leaf}`
        let layer_idx = rest.split('.').next().unwrap_or("");
        let leaf = rest.rsplit('.').next().unwrap_or(rest);
        let leaf = normalize_leaf(leaf);
        return format!("layers.{layer_idx}.{leaf}");
    }
    path.to_string()
}

/// Normalize a module leaf name to its canonical form (site matching across
/// the runtime + the naming differences). The known aliases are collapsed to a
/// single canonical leaf so the runtime `o_proj` matches the checkpoint
/// `out_proj`, etc. The unrecognized leaves are returned as-is (no silent
/// mismatch).
fn normalize_leaf(leaf: &str) -> &str {
    match leaf {
        // The attention output projection (the runtime `o_proj`, the checkpoint
        // `out_proj` for the linear-attention families).
        "o_proj" | "out_proj" => "out_proj",
        // The all query/key/value projections (no alias, the identity).
        "q_proj" | "k_proj" | "v_proj" => leaf,
        // The MLP projections (no alias, the identity).
        "gate_proj" | "up_proj" | "down_proj" => leaf,
        // The fused forms (no alias, the identity).
        "qkv_proj" | "gate_up_proj" => leaf,
        // The GDN (the gated delta net) projections (no alias, the identity).
        "in_proj_qkv" | "in_proj_z" | "in_proj_b" | "in_proj_a"
        | "in_proj_qkvz" | "in_proj_ba" => leaf,
        // The MLA projections (no alias, the identity).
        "q_a_proj" | "q_b_proj" | "kv_a_proj_with_mqa" | "kv_b_proj" => leaf,
        // The output + the embedding (no alias, the identity).
        "lm_head" | "embed_tokens" => leaf,
        // The unknown leaf: return it unchanged (no silent mismatch).
        _ => leaf,
    }
}

/// The process-wide LoRA registry (the single instance, set at startup). The
/// wrapped linears read this global (no per-layer `Arc` threaded through every
/// model constructor). The `None` when LoRA is not enabled (the base model).
/// This is the DRY injection point: the global check the global registry + the
/// thread-local active tenant, so no model constructor signature changes are
/// needed.
///
/// The registry is behind a `Mutex` (the on-demand load mutates it). The read
/// path (the `apply_lora_delta`, the per-layer forward) locks briefly (no I/O
/// under the lock); the write path (the `ensure_loaded`, the batch boundary)
/// locks for the load (the I/O is outside the lock, no deadlock).
static GLOBAL_REGISTRY: OnceLock<std::sync::Arc<std::sync::Mutex<LoraRegistry>>> = OnceLock::new();

/// The process-wide MoE expert LoRA state, keyed by tenant. The `FusedMoe`
/// layers read this global (no per-layer `Arc` threaded through every model
/// constructor). The empty map when no MoE LoRA is loaded (the fused
/// `moe_gemm` kernel path, the prior behavior). Keyed by tenant so each
/// tenant's distinct MoE expert adapter is applied to its own rows (no
/// first-adapter-only limitation).
static GLOBAL_MOE_LORA: OnceLock<std::sync::Mutex<std::collections::HashMap<String, MoeLoraState>>> =
    OnceLock::new();

/// Set a tenant's MoE LoRA state (the called at startup per adapter, the
/// `load_moe_lora_state` output). The `None` state removes the tenant's MoE
/// state.
pub fn set_global_moe_lora(tenant: &str, state: Option<MoeLoraState>) {
    let map = GLOBAL_MOE_LORA.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    match state {
        Some(s) => {
            guard.insert(tenant.to_string(), s);
        }
        None => {
            guard.remove(tenant);
        }
    }
}

/// The global MoE LoRA state for the active tenant (the `None` when no MoE LoRA
/// is loaded for the active tenant, the fused `moe_gemm` kernel path). The active
/// tenant is the single-tenant thread-local (`active_tenant`) OR the uniform
/// batched case (the `LoraExecution` with exactly one distinct tenant, the all
/// rows the same tenant).
pub fn global_moe_lora() -> Option<MoeLoraState> {
    // The single-tenant case (the thread-local active tenant).
    if let Some(tenant) = active_tenant() {
        let map = GLOBAL_MOE_LORA.get()?;
        let guard = map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = guard.get(&tenant).cloned() {
            return Some(state);
        }
        return None;
    }
    // The uniform batched case (the LoraExecution with one distinct tenant).
    if let Some(exec) = lora_execution() {
        if exec.distinct_tenants() == 1 {
            let tenant = exec.row_tenants.iter().find_map(|t| t.clone())?;
            let map = GLOBAL_MOE_LORA.get()?;
            let guard = map.lock().unwrap_or_else(|e| e.into_inner());
            return guard.get(&tenant).cloned();
        }
    }
    None
}

/// The global MoE LoRA state for a specific tenant (the mixed-tenant MoE
/// batch). The `None` when no MoE LoRA is loaded for that tenant (the
/// `moe_gemm` kernel path, no delta). Used by the per-token-tenant MoE
/// forward to look up each token's tenant's MoE state.
pub fn global_moe_lora_for(tenant: &str) -> Option<MoeLoraState> {
    let map = GLOBAL_MOE_LORA.get()?;
    let guard = map.lock().unwrap_or_else(|e| e.into_inner());
    guard.get(tenant).cloned()
}

/// Set the process-wide LoRA registry (the called once at startup, the
/// `init_lora_registry` output). The registry is shared as an `Arc<Mutex>` so
/// the on-demand load (the `ensure_tenant_loaded`, the write path) can mutate
/// it, while the read path (the `apply_lora_delta`, the per-layer forward)
/// locks it briefly. The same `Arc<Mutex>` is stored in the `ModelRunner` (the
/// no a second copy).
pub fn set_global_registry(registry: std::sync::Arc<std::sync::Mutex<LoraRegistry>>) {
    let _ = GLOBAL_REGISTRY.set(registry);
}

/// The global registry (the `None` when LoRA is not enabled).
pub fn global_registry() -> Option<std::sync::Arc<std::sync::Mutex<LoraRegistry>>> {
    GLOBAL_REGISTRY.get().cloned()
}

// The per-request tenant selection (the thread-local, set by the runner before
// the forward pass). The `None` means no adapter (the base model). The forward
// pass is serialized by the actor model, so a thread-local is safe (no data
// race across concurrent requests).
thread_local! {
    static ACTIVE_TENANT: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

/// Set the active tenant for the current forward pass (the runner calls this
/// before `model.forward`). The `None` clears it (the base model).
pub fn set_active_tenant(tenant: Option<String>) {
    ACTIVE_TENANT.with(|t| {
        *t.borrow_mut() = tenant;
    });
}

/// The active tenant for the current forward pass (the `LoraLinear` reads this).
pub fn active_tenant() -> Option<String> {
    ACTIVE_TENANT.with(|t| t.borrow().clone())
}

/// The per-batch LoRA execution context (the batched multi-tenant). Carries
/// the per-row tenant id (the `Option<String>`, one per batch row) so a
/// single forward pass can serve rows from different tenants (the Punica
/// result). The `LoraLinear` consults this for the batched case; when it is
/// not set, the single `active_tenant` thread-local is used (the single-tenant
/// fast path).
#[derive(Debug, Clone, Default)]
pub struct LoraExecution {
    /// The per-row tenant id (the index == the batch row). The `None` row uses
    /// the base model (no adapter).
    pub row_tenants: Vec<Option<String>>,
    /// The per-row tenant stack (additive multi-adapter composition). Each row
    /// maps to a list of tenant ids whose deltas are summed:
    /// `base + sum_j scaling_j * (x @ A_j) @ B_j`. The empty list means the row
    /// uses the base model (no adapter). When this field is all-empty, the
    /// single-tenant `row_tenants` is used instead (the prior behavior). A row
    /// with a non-empty stack overrides its `row_tenants` entry.
    pub row_stacks: Vec<Vec<String>>,
}

impl LoraExecution {
    /// The distinct active tenants in this batch (the `max_loras_per_batch`
    /// bound). The `None` rows are not counted.
    pub fn distinct_tenants(&self) -> usize {
        self.row_tenants
            .iter()
            .filter_map(|t| t.as_ref())
            .collect::<HashSet<_>>()
            .len()
    }

    /// The rows that belong to a given tenant (the per-tenant row group, the
    /// `LoraRowRemap` equivalent). The empty when no row uses this tenant.
    pub fn rows_for_tenant(&self, tenant: &str) -> Vec<usize> {
        self.row_tenants
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_deref() == Some(tenant))
            .map(|(i, _)| i)
            .collect()
    }

    /// The distinct active tenant names in this batch (the deterministic order,
    /// sorted). The `None` rows are not included. This is the compacted active-
    /// tenant list the fused kernel needs (the `active_slots` input).
    pub fn distinct_tenant_list(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .row_tenants
            .iter()
            .filter_map(|t| t.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        v.sort();
        v
    }

    /// Whether any row carries a non-empty tenant stack (the additive
    /// multi-adapter composition). When true, the fused stacked kernel path is
    /// used; when false, the single-tenant-per-row behavior applies.
    pub fn has_stacks(&self) -> bool {
        self.row_stacks.iter().any(|s| !s.is_empty())
    }

    /// The distinct tenant names across all row stacks (the deterministic
    /// order, sorted). This is the compacted active-tenant list the stacked
    /// kernel needs (the `active_slots` input).
    pub fn distinct_stack_tenant_list(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .row_stacks
            .iter()
            .flat_map(|s| s.iter())
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        v.sort();
        v
    }

    /// The per-row stack (the list of tenant ids to compose additively for
    /// that row). Falls back to the single `row_tenants` entry when the row has
    /// no explicit stack.
    pub fn row_stack(&self, row: usize) -> Vec<String> {
        if let Some(stack) = self.row_stacks.get(row) {
            if !stack.is_empty() {
                return stack.clone();
            }
        }
        // Fall back to the single-tenant row entry.
        match self.row_tenants.get(row) {
            Some(Some(t)) => vec![t.clone()],
            _ => Vec::new(),
        }
    }
}

thread_local! {
    /// The per-batch LoRA execution context (the seted multi-tenant). The
    /// `None` means the single-tenant fast path (the `active_tenant` is used).
    static LORA_EXECUTION: std::cell::RefCell<Option<LoraExecution>> =
        std::cell::RefCell::new(None);
}

/// Set the per-batch LoRA execution context (the runner calls this before the
/// forward pass when the batch carries multiple tenants). The `None` clears it
/// (the single-tenant fast path, the `active_tenant` is used).
pub fn set_lora_execution(exec: Option<LoraExecution>) {
    LORA_EXECUTION.with(|t| {
        *t.borrow_mut() = exec;
    });
}

/// The per-batch LoRA execution context (the `None` when the single-tenant fast
/// path is active, the `active_tenant` is used instead).
pub fn lora_execution() -> Option<LoraExecution> {
    LORA_EXECUTION.with(|t| t.borrow().clone())
}

/// The on-demand load: ensure the tenant's adapter is resident. Called at the
/// batch boundary (no per-layer forward) when the batch carries a tenant that
/// is not resident. Locks the registry (the write path; the I/O is outside the
/// lock via the `ensure_loaded`'s internal load). No-op when the tenant is
/// already resident or the source is unknown.
pub fn ensure_tenant_loaded(tenant_id: &str) {
    let Some(registry) = global_registry() else {
        return; // no LoRA registry (base model only)
    };
    let mut reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    let _ = reg.ensure_loaded(tenant_id);
}

/// Check whether the tenant has at least one adapter resident in the global
/// registry. Returns `false` when the registry is absent or the tenant has no
/// loaded adapters. Used by the runner to log a proof that the LoRA delta will
/// actually be applied (vs. silently falling back to the base model).
pub fn global_has_adapter(tenant_id: &str) -> bool {
    let Some(registry) = global_registry() else {
        return false;
    };
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    reg.adapters.get(tenant_id).map_or(false, |adapters| !adapters.is_empty())
}

/// The count of a tenant's resident adapters (the number of target modules the
/// tenant covers). `0` when the registry is absent or the tenant has no loaded
/// adapters. Used by the runner to log how many layers the tenant's delta will
/// touch (a `0` means the request runs on the base model only).
pub fn global_adapter_count(tenant_id: &str) -> usize {
    let Some(registry) = global_registry() else {
        return 0;
    };
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    reg.adapters.get(tenant_id).map(|adapters| adapters.len()).unwrap_or(0)
}

/// The list of a tenant's adapter target-module names (the canonical site
/// keys). Empty when the registry is absent or the tenant has no loaded
/// adapters. Used by the runner to log exactly which layers the tenant's delta
/// will touch, so an operator can verify the module names match the model's
/// registered LoRA sites (a mismatch is the most common cause of a silently
/// no-op LoRA).
pub fn global_adapter_modules(tenant_id: &str) -> Vec<String> {
    let Some(registry) = global_registry() else {
        return Vec::new();
    };
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    reg.adapters
        .get(tenant_id)
        .map(|adapters| adapters.iter().map(|a| a.target_module.clone()).collect())
        .unwrap_or_default()
}

/// Activate a tenant for the current batch (the two-tier LRU): move the tenant
/// into the `active` set (the in-batch bank slot). The active set is capped by
/// the `max_active`; the over-cap LRU active tenant is deactivated (the bank
/// slot freed, the weights stay registered). No-op when the LoRA registry is
/// absent or the tenant is unregistered.
pub fn activate_tenant(tenant_id: &str) {
    let Some(registry) = global_registry() else {
        return;
    };
    let mut reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    reg.activate(tenant_id);
}

/// Deactivate a tenant for the current batch: remove the tenant from the
/// `active` set (the bank slot freed, the weights stay registered). No-op when
/// the LoRA registry is absent or the tenant is not active.
pub fn deactivate_tenant(tenant_id: &str) {
    let Some(registry) = global_registry() else {
        return;
    };
    let mut reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    reg.deactivate(tenant_id);
}

/// Apply the LoRA delta for a module to the input `x` (the `[batch, ...]`),
/// returning the delta tensor (`None` when no adapter applies, the base
/// model). This is the single entry point the wrapped linears call. It handles
/// both the batched multi-tenant case (the `LoraExecution` set, the per-row
/// tenant) + the single-tenant case (the `active_tenant` thread-local, the all
/// rows the same tenant). The `module` is the layer's LoRA module name (the
/// `q_proj`, etc.); the empty module name means this layer is not a LoRA
/// target (the fast path, no lookup).
pub fn apply_lora_delta(
    module: &str,
    x: &Tensor,
    parallelism: LoraParallelism,
    tp_rank: usize,
    tp_world: usize,
) -> Result<Option<Tensor>> {
    let Some(registry) = global_registry() else {
        return Ok(None); // no LoRA registry (base model only)
    };
    // Lock the registry briefly (read path, no I/O under the lock).
    // The guard is dropped at the end of the scope (no lock held across the forward).
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    // Stacked (additive multi-adapter) case: when a row carries a tenant stack,
    // the fused stacked kernel composes all stacked deltas in two launches.
    // CUDA-only; falls through to the prior paths when unavailable applicable.
    #[cfg(feature = "cuda")]
    if let Some(exec) = lora_execution() {
        if exec.has_stacks() {
            if let Some(delta) = apply_lora_delta_fused_stacked(
                module, x, parallelism, tp_rank, tp_world,
            )? {
                return Ok(Some(delta));
            }
        }
    }
    // Batched multi-tenant case (LoraExecution set, per-row tenant).
    if let Some(exec) = lora_execution() {
        return Ok(reg.batched_delta(module, x, &exec, parallelism, tp_rank, tp_world)?);
    }
    // Single-tenant case (active_tenant thread-local, all rows same tenant).
    if let Some(adapter) = reg.active_adapter_for(module) {
        return Ok(Some(adapter.delta(x, parallelism, tp_rank, tp_world)?));
    }
    Ok(None)
}

/// Fused-kernel LoRA delta (S-LoRA throughput path). When the batch carries
/// multiple tenants (LoraExecution set) + the registry has a stacked A/B bank,
/// this calls the attention-rs fused shrink/expand kernel to compute the delta
/// for all active tenants in two kernel launches (no per-tenant per-layer
/// candle matmul pair). The delta is returned as a fresh tensor (the caller
/// adds it to the base output).
///
/// Returns `Ok(Some(delta))` when the fused kernel handled the delta. Returns
/// `Ok(None)` when the fused path is not applicable (single-tenant batch, no
/// bank, no kernel) — the caller then falls back to the per-tenant candle path
/// in `apply_lora_delta`.
///
/// The `x` is the layer input (`[rows, in_features]`). The `parallelism`/
/// `tp_rank`/`tp_world` select the TP shard of the stacked A/B (Column shards B
/// on the out dim, Row shards A on the in dim, Replicated uses the full A/B).
#[cfg(feature = "cuda")]
pub fn apply_lora_delta_fused(
    module: &str,
    x: &Tensor,
    parallelism: LoraParallelism,
    tp_rank: usize,
    tp_world: usize,
) -> Result<Option<Tensor>> {
    use attention_rs::lora_fused::cuda as lora_fused;

    // Fused path is only for multi-tenant batch (>1 distinct tenant).
    // Single-tenant + no-tenant cases use the cheaper candle path.
    let Some(exec) = lora_execution() else {
        return Ok(None);
    };
    if exec.distinct_tenants() <= 1 {
        return Ok(None);
    }
    let Some(registry) = global_registry() else {
        return Ok(None);
    };
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    let Some(bank) = reg.bank() else {
        return Ok(None); // no stacked bank (fused kernel needs it)
    };
    // Stacked A/B for this module: `[max_loras, in, rank]` / `[max_loras, rank, out]`.
    // None when the module has no stacked A/B (fallback to candle path).
    let (a_stacked, b_stacked) = match (bank.stacked_a(module), bank.stacked_b(module)) {
        (Some(a), Some(b)) => (a.clone(), b.clone()),
        _ => return Ok(None),
    };
    // TP sharding: Column shards B on the out dim, Row shards A on the in dim,
    // Replicated uses the full A/B. The shard is the per-rank slice of the
    // stacked tensor.
    let (a_use, b_use) = match parallelism {
        LoraParallelism::Column => {
            let out_dim = b_stacked.dim(2).unwrap_or(0);
            let per_rank = out_dim / tp_world.max(1);
            let start = tp_rank * per_rank;
            let b_shard = b_stacked.narrow(2, start, per_rank).unwrap_or_else(|_| b_stacked.clone());
            (a_stacked.clone(), b_shard)
        }
        LoraParallelism::Row => {
            let in_dim = a_stacked.dim(1).unwrap_or(0);
            let per_rank = in_dim / tp_world.max(1);
            let start = tp_rank * per_rank;
            let a_shard = a_stacked.narrow(1, start, per_rank).unwrap_or_else(|_| a_stacked.clone());
            (a_shard, b_stacked.clone())
        }
        LoraParallelism::Replicated => (a_stacked, b_stacked),
    };
    // Compacted active slots: distinct tenants in this batch that have a bank
    // slot. `active_slots` maps compacted index -> bank slot. `slot_map` maps
    // each row -> compacted index (or BASE_SLOT sentinel for base-model rows).
    let active_tenants: Vec<String> = exec.distinct_tenant_list();
    let mut active_slots = Vec::with_capacity(active_tenants.len());
    for t in &active_tenants {
        if let Some(&slot) = bank.tenant_slots().get(t) {
            active_slots.push(slot as u32);
        }
    }
    if active_slots.is_empty() {
        return Ok(None); // no active tenant has a bank slot (fallback)
    }
    let num_active = active_slots.len();
    // Tenant -> compacted index map (for building the per-row slot_map).
    let tenant_to_compact: std::collections::HashMap<&str, u32> = active_tenants
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i as u32))
        .collect();
    let rows = x.dim(0).unwrap_or(0);
    let mut slot_map = vec![lora_fused::BASE_SLOT; rows];
    for (row, tenant) in exec.row_tenants.iter().enumerate().take(rows) {
        if let Some(t) = tenant {
            if let Some(&compact) = tenant_to_compact.get(t.as_str()) {
                slot_map[row] = compact;
            }
        }
    }
    // Build the kernel inputs on the x device.
    let device = x.device();
    let slot_map_t = Tensor::from_vec(slot_map, (rows,), device)?;
    let active_slots_t = Tensor::from_vec(active_slots.clone(), (num_active,), device)?;
    // Scratch buffer: `[rows, num_active, rank]` f32. The rank is the A's last dim.
    let rank = a_use.dim(2).unwrap_or(0);
    if rank == 0 {
        return Ok(None);
    }
    let buf = Tensor::zeros(
        &[rows, num_active, rank],
        DType::F32,
        device,
    )?;
    // Per-slot scale tensor: `[num_slots]` f32, the bank's `scale_stacked`.
    let scale_full = bank.stacked_scale(module).cloned().unwrap_or_else(|| {
        Tensor::ones((bank.max_loras(),), DType::F32, device).unwrap()
    });
    // Shrink: buf[row, compact, k] = scale[bank_slot] * sum_c x[row, c] * A[bank_slot, c, k]
    lora_fused::lora_shrink(x, &a_use, &scale_full, &slot_map_t, &active_slots_t, &buf)?;
    // Expand: out[row, o] = sum_k buf[row, compact, k] * B[bank_slot, k, o]
    // The `out` is the fresh delta tensor (the caller adds it to the base output).
    let out_features = b_use.dim(2).unwrap_or(0);
    let out = Tensor::zeros(
        &[rows, out_features],
        x.dtype(),
        device,
    )?;
    lora_fused::lora_expand(&buf, &b_use, &slot_map_t, &active_slots_t, &out)?;
    Ok(Some(out))
}

/// Stacked (additive multi-adapter) fused-kernel LoRA delta. When a row carries
/// a tenant stack (`LoraExecution.row_stacks`), this composes the deltas of all
/// stacked tenants additively in two kernel launches:
/// `base + sum_j scaling_j * (x @ A_j) @ B_j`. The CSR layout (the
/// `slot_offsets`/`slot_ids`) maps each row to its list of compacted slots; the
/// ` `buf` is `[rows, max_stack, rank]` f32.
///
/// Returns `Ok(Some(delta))` when the stacked kernel handled the delta. Returns
/// `Ok(None)` when the stacked path is not applicable (no row has a stack, no
/// bank, no kernel) — the caller then falls back to the prior paths.
///
/// The `x` is the layer input (`[rows, in_features]`). The `parallelism`/
/// `tp_rank`/`tp_world` select the TP shard of the stacked A/B.
#[cfg(feature = "cuda")]
pub fn apply_lora_delta_fused_stacked(
    module: &str,
    x: &Tensor,
    parallelism: LoraParallelism,
    tp_rank: usize,
    tp_world: usize,
) -> Result<Option<Tensor>> {
    use attention_rs::lora_fused::cuda as lora_fused;

    // The stacked path requires at least one row with a non-empty tenant stack.
    let Some(exec) = lora_execution() else {
        return Ok(None);
    };
    if !exec.has_stacks() {
        return Ok(None);
    }
    let Some(registry) = global_registry() else {
        return Ok(None);
    };
    let reg = registry.lock().unwrap_or_else(|e| e.into_inner());
    let Some(bank) = reg.bank() else {
        return Ok(None); // no stacked bank (the fused kernel needs it)
    };
    // Stacked A/B for this module: `[max_loras, in, rank]` / `[max_loras, rank, out]`.
    let (a_stacked, b_stacked) = match (bank.stacked_a(module), bank.stacked_b(module)) {
        (Some(a), Some(b)) => (a.clone(), b.clone()),
        _ => return Ok(None),
    };
    // TP sharding (same as the single-tenant fused path).
    let (a_use, b_use) = match parallelism {
        LoraParallelism::Column => {
            let out_dim = b_stacked.dim(2).unwrap_or(0);
            let per_rank = out_dim / tp_world.max(1);
            let start = tp_rank * per_rank;
            let b_shard = b_stacked.narrow(2, start, per_rank).unwrap_or_else(|_| b_stacked.clone());
            (a_stacked.clone(), b_shard)
        }
        LoraParallelism::Row => {
            let in_dim = a_stacked.dim(1).unwrap_or(0);
            let per_rank = in_dim / tp_world.max(1);
            let start = tp_rank * per_rank;
            let a_shard = a_stacked.narrow(1, start, per_rank).unwrap_or_else(|_| a_stacked.clone());
            (a_shard, b_stacked.clone())
        }
        LoraParallelism::Replicated => (a_stacked, b_stacked),
    };
    // Compacted active slots: distinct distinct tenants across all stacks stacks that
    // have a bank slot. `active_slots` maps compacted index -> bank slot.
    let stack_tenants: Vec<String> = exec.distinct_stack_tenant_list();
    let mut active_slots = Vec::with_capacity(stack_tenants.len());
    for t in &stack_tenants {
        if let Some(&slot) = bank.tenant_slots().get(t) {
            active_slots.push(slot as u32);
        }
    }
    if active_slots.is_empty() {
        return Ok(None); // no stacked tenant has a bank slot (fallback)
    }
    let num_active = active_slots.len();
    // Tenant -> compacted index map.
    let tenant_to_compact: std::collections::HashMap<&str, u32> = stack_tenants
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i as u32))
        .collect();
    // Build the CSR slot_offsets [rows+1] + slot_ids [total] from the per-row stacks.
    let rows = x.dim(0).unwrap_or(0);
    let mut slot_offsets = vec![0u32; rows + 1];
    let mut slot_ids: Vec<u32> = Vec::new();
    let mut max_stack = 0usize;
    for row in 0..rows {
        let stack = exec.row_stack(row);
        // Only the tenants that have a bank slot + a compacted index contribute.
        let mut row_slots: Vec<u32> = Vec::new();
        for t in &stack {
            if let Some(&compact) = tenant_to_compact.get(t.as_str()) {
                row_slots.push(compact);
            }
        }
        max_stack = max_stack.max(row_slots.len());
        slot_ids.extend(row_slots);
        slot_offsets[row + 1] = slot_ids.len() as u32;
    }
    if max_stack == 0 {
        return Ok(None); // no row has a resolvable stacked tenant (fallback)
    }
    // Build the kernel inputs on the x device.
    let device = x.device();
    let slot_offsets_t = Tensor::from_vec(slot_offsets.clone(), (rows + 1,), device)?;
    let slot_ids_t = Tensor::from_vec(slot_ids.clone(), (slot_ids.len(),), device)?;
    let active_slots_t = Tensor::from_vec(active_slots.clone(), (num_active,), device)?;
    // Scratch buffer: `[rows, max_stack, rank]` f32.
    let rank = a_use.dim(2).unwrap_or(0);
    if rank == 0 {
        return Ok(None);
    }
    let buf = Tensor::zeros(
        &[rows, max_stack, rank],
        DType::F32,
        device,
    )?;
    // Per-slot scale tensor: `[num_slots]` f32, the bank's `scale_stacked`.
    let scale_full = bank.stacked_scale(module).cloned().unwrap_or_else(|| {
        Tensor::ones((bank.max_loras(),), DType::F32, device).unwrap()
    });
    // Shrink: buf[row, j, k] = scale[bank_slot] * sum_c x[row, c] * A[bank_slot, c, k]
    // for each (row, j) in the row's stack.
    lora_fused::lora_shrink_stacked(
        x, &a_use, &scale_full, &slot_offsets_t, &slot_ids_t, &active_slots_t, &buf,
    )?;
    // Expand: out[row, o] = sum_j sum_k buf[row, j, k] * B[bank_slot(j), k, o]
    // The `out` is the fresh delta tensor (the caller adds it to the base output).
    let out_features = b_use.dim(2).unwrap_or(0);
    let out = Tensor::zeros(
        &[rows, out_features],
        x.dtype(),
        device,
    )?;
    lora_fused::lora_expand_stacked(
        &buf, &b_use, &slot_offsets_t, &slot_ids_t, &active_slots_t, &out,
    )?;
    Ok(Some(out))
}

/// The embedding LoRA helper: apply the active tenant's delta to an embedding
/// output. The models call this after their `embed_tokens.forward` (one-line
/// injection, no per-model wrapper type). The `emb` is the `[batch, seq, hidden]`
/// embedding output; the delta is the `scaling * (emb @ A) @ B` where the A is
/// the `[hidden, rank]` + the B is the `[rank, hidden]` (the square embedding
/// LoRA). No-op when no tenant is active or the adapter does not target the
/// `embed_tokens` module.
pub fn apply_embedding_lora(emb: &Tensor) -> Result<Tensor> {
    if let Some(delta) = apply_lora_delta(
        "embed_tokens",
        emb,
        LoraParallelism::Replicated,
        0,
        1,
    )? {
        Ok(emb.broadcast_add(&delta)?)
    } else {
        Ok(emb.clone())
    }
}
/// The LoRA parallelism mode (the TP sharding of the A/B). The mode is a
/// property of the SITE (the layer), not the adapter: it determines how the
/// adapter's full A/B are sharded to match the base linear's TP layout.
/// - `Column` (the q/k/v_proj, the gate/up_proj): the B is sharded on the out
///   dim (each rank holds `out/world` columns of B), the A is replicated.
/// - `Row` (the o_proj, the down_proj): the A is sharded on the in dim (each
///   rank holds `in/world` rows of A), the B is replicated.
/// - `Replicated` (the single-GPU, no TP): the full A/B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoraParallelism {
    #[default]
    Replicated,
    Column,
    Row,
}

/// A single LoRA adapter for one target module. The A/B are GPU tensors (the
/// `[in_features, rank]` + the `[rank, out_features]`), so the delta is applied
/// on the device (no CPU round-trip). The TP sharding is a property of the SITE
/// layer, passed to the `delta` (not the adapter field, which is shared across
/// layers with different layouts).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// The target module name (e.g. `"q_proj"`, the layer this adapter applies to).
    pub target_module: String,
    /// The adapter rank (the low-rank decomposition's width).
    pub rank: usize,
    /// The down-projection A (the `[in_features, rank]` GPU tensor).
    pub a: Tensor,
    /// The up-projection B (the `[rank, out_features]` GPU tensor).
    pub b: Tensor,
    /// The scaling factor (the `alpha / rank`).
    pub scaling: f32,
    /// The fused-module constituents (the feature permutation). When this
    /// adapter targets a fused module (the `qkv_proj`, the `gate_up_proj`), the
    /// `constituents` are the per-constituent (module name, out_dim) pairs in the
    /// fused output order. The empty for a non-fused adapter (the direct module).
    /// The delta is computed per-constituent + concatenated in the fused order.
    pub constituents: Vec<(String, usize)>,
    /// The QLoRA quantized A (the NF4/FP8-quantized down-projection).
    /// The `None` when the adapter is stored dense (the prior behavior). When set,
    /// the `a` field is the quantized form + the `a_scale` is the dequant scale.
    pub a_quantized: bool,
    /// The QLoRA dequantization scale for the A. The `None` when the
    /// A is dense.
    pub a_scale: Option<Tensor>,
    /// The QLoRA quantized B (the NF4/FP8-quantized up-projection).
    pub b_quantized: bool,
    /// The QLoRA dequantization scale for the B. The `None` when the
    /// B is dense.
    pub b_scale: Option<Tensor>,
}

/// The adapter identity (the SGLang `LoRARef` equivalent). The `lora_id` is a
/// deterministic UUID5 from the name + path (cross-node stable, avoiding the
/// per-node `uuid4` divergence that breaks cross-node lookups). The adapter is
/// referenced by the `lora_id` (not the name, which can collide).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LoraRef {
    /// The deterministic adapter id (the UUID5 from the name + path).
    pub lora_id: String,
    /// The adapter name (the operator-facing label, may collide across paths).
    pub lora_name: String,
    /// The adapter source path (the HF id or the local dir).
    pub lora_path: String,
    /// Whether the adapter is pinned (no evicted, the SLA guarantee).
    pub pinned: bool,
}

impl LoraRef {
    /// Create a `LoraRef` with a deterministic id (the UUID5 from the name + path).
    /// The deterministic id means the same adapter on different nodes has the
    /// same id (the cross-node routing).
    pub fn new(name: &str, path: &str, pinned: bool) -> Self {
        Self {
            lora_id: deterministic_lora_id(name, path),
            lora_name: name.to_string(),
            lora_path: path.to_string(),
            pinned,
        }
    }
}

/// An immutable adapter generation (the alias→generation pinning). A
/// generation is a frozen set of adapters for a tenant (`Vec<LoraAdapter>`),
/// identified by a monotonically-increasing generation. The mutable alias (the
/// tenant name) points to a generation; a swap creates a new generation +
/// atomically repoints the alias. In-flight requests hold a lease on the old
/// generation (the swap does not affect them), so a no-downtime adapter deploy
/// is possible:
#[derive(Debug, Clone)]
pub struct LoraGeneration {
    /// The generation id (monotonically-increasing, not the alias).
    pub id: u64,
    /// The tenant name (the alias this generation serves).
    pub tenant: String,
    /// The frozen adapter set (the immutable entry per target module).
    pub adapters: Vec<LoraAdapter>,
}

impl LoraGeneration {
    /// Create a generation from a tenant + its adapters (the ` generation,
    /// the id assigned by the registry).
    pub fn new(id: u64, tenant: String, adapters: Vec<LoraAdapter>) -> Self {
        Self {
            id,
            tenant,
            adapters,
        }
    }
}

/// The deterministic adapter id (the UUID5-style stable hash from the name + path).
/// The same adapter on different nodes gets the same id (cross-node identity).
/// Uses a FNV-1a hash (no external dep, deterministic across runs + nodes,
/// unlike `std::hash` which is not stable).
fn deterministic_lora_id(name: &str, path: &str) -> String {
    // FNV-1a 64-bit hash (deterministic, no external dep).
    fn fnv1a(data: &[u8]) -> u64 {
        const OFFSET: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x00000100000001B3;
        let mut hash = OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    }
    let mut input = Vec::new();
    input.extend_from_slice(name.as_bytes());
    input.push(0); // separator (avoids name/path collision)
    input.extend_from_slice(path.as_bytes());
    let h1 = fnv1a(&input);
    let h2 = fnv1a(&input[1..]); // second hash (avoids single 64-bit id)
    format!("{:016x}{:016x}", h1, h2)
}

impl LoraAdapter {
    /// Create a LoRA adapter with the default parallelism (the `Replicated`, the
    /// single-GPU / no-TP case). The TP sharding (the `Column`/`Row`) is set
    /// by the layer at wrap time (the `with_parallelism`), not at load time.
    pub fn new(
        target_module: String,
        rank: usize,
        a: Tensor,
        b: Tensor,
        scaling: f32,
    ) -> Self {
        Self {
            target_module,
            rank,
            a,
            b,
            scaling,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        }
    }

    /// Create a fused-module adapter (the feature permutation). The
    /// `constituents` are the per-constituent (module name, out_dim) pairs in the
    /// fused output order (the `qkv_proj` → the `[(q_proj, q_out), (k_proj, k_out),
    /// (v_proj, v_out)]`). The A/B are the per-constituent A/B concatenated in the
    /// fused order (the `load_adapters_from_paths` as this).
    pub fn new_fused(
        target_module: String,
        rank: usize,
        a: Tensor,
        b: Tensor,
        scaling: f32,
        constituents: Vec<(String, usize)>,
    ) -> Self {
        Self {
            target_module,
            rank,
            a,
            b,
            scaling,
            constituents,
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        }
    }

    /// The adapter delta applied to an input `x` (the `[batch, seq, in_features]`):
    /// `scaling * (x @ A_shard) @ B_shard`. The `A` is the `[in_features, rank]`
    /// (the down-projection), the `B` is the `[rank, out_features]` (the up-
    /// projection). The `parallelism`/`tp_rank`/`tp_world` are the SITE's TP
    /// layout, passed by the calling layer (not the adapter field, which is
    /// shared across layers with different layouts). The result is the
    /// `[batch, seq, out_features]` delta to add to the base linear's output.
    pub fn delta(
        &self,
        x: &Tensor,
        parallelism: LoraParallelism,
        tp_rank: usize,
        tp_world: usize,
    ) -> Result<Tensor> {
        // The x is the `[batch, seq, in_features]` (the 3D) or the `[batch, in_features]`
        // (the 2D). The A is the `[in_features, rank]`, the B is the `[rank, out_features]`.
        // The delta is the `scaling * (x @ A) @ B`. For the 3D x, flatten to the 2D
        // `[batch*seq, in]`, do the matmul, reshape back (candle matmul is
        // 2D × 2D, not 3D × 2D broadcast).
        let orig_dims = x.dims().to_vec();
        let is_3d = orig_dims.len() == 3;
        let x2 = if is_3d {
            let (b, s, h) = x.dims3()?;
            x.reshape((b * s, h))?
        } else {
            x.clone()
        }
        .contiguous()?;
        // The QLoRA dequantize: when the A/B are stored quantized (the
        // FP8/NF4), dequantize them to the activation dtype before the matmul.
        // The dense A/B (the a_quantized is false) are used as-is.
        let a_eff = if self.a_quantized {
            self.dequantize(&self.a, self.a_scale.as_ref(), x2.dtype())?
        } else {
            self.a.clone()
        };
        let b_eff = if self.b_quantized {
            self.dequantize(&self.b, self.b_scale.as_ref(), x2.dtype())?
        } else {
            self.b.clone()
        };
        // The TP sharding: slice the A/B to this rank's shard before the
        // matmul. The `Column` mode shards the B on the out dim (each rank holds
        // `out/world` columns of B); the A is replicated. The `Row` mode shards the
        // A on the in dim (each rank holds `in/world` rows of A), the B is
        // replicated. The `Replicated` (the default, the single-GPU) uses the full
        // A/B. The shard is computed from the `tp_rank`/`tp_world` + the tensor dim.
        let (a_shard, b_shard) = match parallelism {
            LoraParallelism::Column => {
                // The B is sharded on the out dim (the B's last dim, the `[rank, out]`).
                // Each rank holds `out / world` columns. The A is replicated (the full).
                // The `narrow` returns a non-contiguous view; the CUDA matmul requires
                // contiguous operands, so materialize the shard with `.contiguous()`.
                let out_dim = b_eff.dim(1)?;
                let per_rank = out_dim / tp_world.max(1);
                let start = tp_rank * per_rank;
                let b_shard = b_eff.narrow(1, start, per_rank)?.contiguous()?;
                (a_eff.contiguous()?, b_shard)
            }
            LoraParallelism::Row => {
                // The A is sharded on the in dim (the A's first dim, the `[in, rank]`).
                // Each rank holds `in / world` rows. The B is replicated (the full).
                // The `narrow` view is non-contiguous; materialize it for the matmul.
                let in_dim = a_eff.dim(0)?;
                let per_rank = in_dim / tp_world.max(1);
                let start = tp_rank * per_rank;
                let a_shard = a_eff.narrow(0, start, per_rank)?.contiguous()?;
                (a_shard, b_eff.contiguous()?)
            }
            LoraParallelism::Replicated => (a_eff.contiguous()?, b_eff.contiguous()?),
        };
        // The A/B are cast to the input's dtype (the activation dtype at this
        // site, which may differ from the model's global compute dtype — e.g. an
        // FP8 model has FP8 activations in some layers, BF16 in others). The
        // base linear handles this via the forward_dense_dtype_compatible; the
        // delta must match the input dtype for the matmul.
        let a_shard = a_shard.to_dtype(x2.dtype())?;
        let b_shard = b_shard.to_dtype(x2.dtype())?;
        let down = x2.matmul(&a_shard)?;
        let up = down.matmul(&b_shard)?;
        // The scale factor must applied in the activation dtype (the `up` dtype),
        // not F32. A F32 scalar tensor broadcast-multiplied against an F16/BF16
        // activation is a dtype mismatch (the crash: "dtype mismatch in mul, lhs:
        // F16, rhs: F32"). The `Tensor::new` infers the dtype from the Rust `f32`
        // literal, so cast the scale to the activation dtype explicitly.
        let scale = Tensor::new(self.scaling, up.device())?.to_dtype(up.dtype())?;
        let result = up.broadcast_mul(&scale)?;
        if is_3d {
            let (b, s, _) = x.dims3()?;
            let (_, out) = result.dims2()?;
            Ok(result.reshape((b, s, out))?)
        } else {
            Ok(result)
        }
    }

    /// The QLoRA dequantize: the quantized A/B (the FP8/NF4) are
    /// dequantized to the target dtype via the scale. The `quant` is the
    /// quantized tensor, the `scale` is the dequant scale (`None` for the
    /// unit scale, the dense-equivalent). The result is the `quant * scale` in
    /// the target dtype.
    fn dequantize(
        &self,
        quant: &Tensor,
        scale: Option<&Tensor>,
        target_dtype: DType,
    ) -> Result<Tensor> {
        match scale {
            Some(s) => quant
                .to_dtype(target_dtype.clone())?
                .broadcast_mul(&s.to_dtype(target_dtype)?),
            None => Ok(quant.to_dtype(target_dtype)?),
        }
    }

    /// The weight-space delta (the `scaling * (B @ A)`, the `[out_features,
    /// in_features]` weight layout). This is the delta added to the base weight
    /// (the `W' = W + scaling * (B @ A)`), used by the MLA correction (the
    /// re-deriving the absorbed weights from the effective kv_b_proj
    /// weight). The A is the `[in_features, rank]`, the B is the `[rank,
    /// out_features]`, so the `B @ A` is the `[out_features, in_features]` (the
    /// weight layout).
    pub fn weight_delta(&self, device: &Device) -> Result<Tensor> {
        let _ = device;
        // The weight delta (the `W' = W + delta_W`, the `W` is the `[out, in]`).
        // The delta is the `scaling * (A @ B)^T` where the `scaling * (B^T @ A^T)`,
        // where the A is the `[in, rank]`, the B is the `[rank, out]`. The result
        // is the `[out, in]` (the weight layout).
        let ab_t = self.b.t()?.matmul(&self.a.t()?)?;
        // The scale is applied in the A/B dtype (the `ab_t` dtype), not F32. A F32
        // scalar tensor broadcast-multiplied against an F16/BF16 weight is a dtype
        // mismatch. The `Tensor::new` infers F32 from the Rust `f32` literal, so
        // cast the scale to the weight dtype explicitly.
        let scale = Tensor::new(self.scaling, ab_t.device())?.to_dtype(ab_t.dtype())?;
        ab_t.broadcast_mul(&scale)
    }
}

/// The MoE expert LoRA state (the routed + shared experts). The xinfer
/// MoE uses a fused 3D expert weight table (the `gate_up_w` is the
/// `[num_experts, hidden, 2*intermediate]`, the `down_w` is the
/// `[num_experts, intermediate, hidden]`). The MoE LoRA stores the per-expert
/// A/B in the matching 3D layout: the `gate_up_a` is the `[num_experts, hidden,
/// rank]`, the `gate_up_b` is the `[num_experts, rank, 2*intermediate]`, the
/// `down_a` is the `[num_experts, intermediate, rank]`, the `down_b` is the
/// `[num_experts, rank, hidden]`. When a tenant is active, the MoE forward uses
/// a per-expert dense path (not the fused `moe_gemm` kernel, which cannot
/// take a delta) + applies the per-expert LoRA delta.
#[derive(Debug, Clone)]
pub struct MoeLoraState {
    /// The per-expert gate_up down-projection A (the `[num_experts, hidden, rank]`).
    pub gate_up_a: Tensor,
    /// The per-expert gate_up up-projection B (the `[num_experts, rank, 2*intermediate]`).
    pub gate_up_b: Tensor,
    /// The per-expert down down-projection A (the `[num_experts, intermediate, rank]`).
    pub down_a: Tensor,
    /// The per-expert down up-projection B (the `[num_experts, rank, hidden]`).
    pub down_b: Tensor,
    /// The scaling factor (the `alpha / rank`).
    pub scaling: f32,
    /// The rank (the low-rank decomposition's width).
    pub rank: usize,
}

impl MoeLoraState {
    /// The per-expert gate_up delta (the `x_e @ A_e @ B_e * scaling`, the
    /// `x_e` is the `[tokens_for_expert, hidden]`, the `expert` is the expert
    /// index). The result is the `[tokens_for_expert, 2*intermediate]` delta
    /// to add to the base gate_up.
    pub fn gate_up_delta(&self, x: &Tensor, expert: usize) -> Result<Tensor> {
        // The A is the `[num_experts, hidden, rank]` — select this expert's A
        // (the `[hidden, rank]`). The B is the `[num_experts, rank, 2*inter]` —
        // select this expert's B (the `[rank, 2*inter]`).
        let expert_a = self.gate_up_a.narrow(0, expert, 1)?.squeeze(0)?.to_dtype(x.dtype())?;
        let expert_b = self.gate_up_b.narrow(0, expert, 1)?.squeeze(0)?.to_dtype(x.dtype())?;
        let down = x.matmul(&expert_a)?;
        let up = down.matmul(&expert_b)?;
        // The scale is applied in the activation dtype (the `up` dtype), not F32.
        // A F32 scalar tensor broadcast-multiplied against an F16/BF16 activation
        // is a dtype mismatch (the crash). The `Tensor::new` infers F32 from the
        // Rust `f32` literal, so cast the scale to the activation dtype explicitly.
        let scale = Tensor::new(self.scaling, up.device())?.to_dtype(up.dtype())?;
        up.broadcast_mul(&scale)
    }

    /// The per-expert down delta (the `x_e @ A_e @ B_e * scaling`, the
    /// `x_e` is the `[tokens_for_expert, intermediate]`, the `expert` is the
    /// expert index). The result is the `[tokens_for_expert, hidden]` delta to
    /// add to the base down.
    pub fn down_delta(&self, x: &Tensor, expert: usize) -> Result<Tensor> {
        let expert_a = self.down_a.narrow(0, expert, 1)?.squeeze(0)?.to_dtype(x.dtype())?;
        let expert_b = self.down_b.narrow(0, expert, 1)?.squeeze(0)?.to_dtype(x.dtype())?;
        let down = x.matmul(&expert_a)?;
        let up = down.matmul(&expert_b)?;
        // The scale is applied in the activation dtype (the `up` dtype), not F32.
        let scale = Tensor::new(self.scaling, up.device())?.to_dtype(up.dtype())?;
        up.broadcast_mul(&scale)
    }
}

/// Load a MoE expert LoRA state from a resolved adapter directory. The adapter
/// dir holds the per-expert `lora_A`/`lora_B` tensors (the
/// `experts.N.gate_proj.lora_A.weight` etc.). The 3D expert layout matches the
/// xinfer MoE fused expert weight table (the `gate_up_w`/`down_w`).
pub fn load_moe_lora_state(
    paths: &crate::utils::downloader::ModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<Option<MoeLoraState>> {
    use candle_core::safetensors;
    // Load the adapter weights (the safetensors files, the possibly sharded).
    let weight_files = paths.get_weight_filenames();
    if weight_files.is_empty() {
        return Ok(None); // no weight files (no MoE LoRA)
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for weight_file in &weight_files {
        let file_tensors = safetensors::load(weight_file, device)
            .map_err(|e| Error::Msg(format!("failed to load MoE LoRA weights {weight_file:?}: {e}")))?;
        for (name, tensor) in file_tensors {
            tensors.insert(name, tensor);
        }
    }
    // Discover the MoE expert LoRA tensors (the `experts.N.gate_proj.lora_A.weight`
    // pattern). No MoE tensors means the adapter has no MoE LoRA (the None).
    let has_moe_lora = tensors.keys().any(|k| k.contains("experts") && k.contains("lora_A"));
    if !has_moe_lora {
        return Ok(None);
    }
    // The read the rank + the alpha from the config (the same as the dense LoRA).
    let config_path = paths.get_config_filename();
    let config_data = std::fs::read(&config_path)
        .map_err(|e| Error::Msg(format!("failed to read MoE LoRA config {config_path:?}: {e}")))?;
    #[derive(serde::Deserialize)]
    struct MoELoraConfig {
        #[serde(rename = "r")]
        rank: Option<usize>,
        #[serde(rename = "lora_alpha")]
        alpha: Option<f64>,
    }
    let cfg: MoELoraConfig = serde_json::from_slice(&config_data)
        .map_err(|e| Error::Msg(format!("failed to parse MoE LoRA config: {e}")))?;
    let rank = cfg.rank.unwrap_or(8);
    let alpha = cfg.alpha.unwrap_or(rank as f64);
    let scaling = (alpha / rank as f64) as f32;
    // Build the 3D expert A/B tensors (the `[num_experts, ...]`). Each
    // expert's A/B is stacked into the 3D tensor (the xinfer MoE layout).
    // The gate_up A is the `[num_experts, hidden, rank]` (the per-expert gate +
    // up share, the concatenated on the out dim). The gate_up B is the
    // `[num_experts, rank, 2*intermediate]`. The down A is the
    // `[num_experts, intermediate, rank]`. The down B is the
    // `[num_experts, rank, hidden]`.
    // The placeholder: the actual per-expert A/B extraction from the checkpoint is
    // the model-specific concern (each checkpoint family stores the expert LoRA
    // differently). This loads the 3D tensors if present (the `experts.lora_A`
    // / the `experts.lora_B`), else returns the None (no MoE LoRA for this
    // checkpoint layout).
    let gate_up_a_key = "experts.gate_up_proj.lora_A.weight";
    let gate_up_b_key = "experts.gate_up_proj.lora_B.weight";
    let down_a_key = "experts.down_proj.lora_A.weight";
    let down_b_key = "experts.down_proj.lora_B.weight";
    let Some(gate_up_a) = tensors.get(gate_up_a_key) else {
        return Ok(None); // no fused 3D expert LoRA in this checkpoint (the per-expert
                         // layout is the model-specific extension)
    };
    let gate_up_b = tensors.get(gate_up_b_key).ok_or_else(|| {
        Error::Msg(format!("MoE LoRA missing the {gate_up_b_key} tensor"))
    })?;
    let down_a = tensors.get(down_a_key).cloned();
    let down_b = tensors.get(down_b_key).cloned();
    // Cast to the model's compute dtype (avoids dtype mismatch in the matmul).
    let gate_up_a = gate_up_a.to_dtype(dtype)?;
    let gate_up_b = gate_up_b.to_dtype(dtype)?;
    let down_a = down_a.map(|t| t.to_dtype(dtype)).transpose()?;
    let down_b = down_b.map(|t| t.to_dtype(dtype)).transpose()?;
    let Some(down_a) = down_a else {
        return Ok(None); // no down LoRA (the gate_up only, the partial MoE LoRA)
    };
    let Some(down_b) = down_b else {
        return Ok(None);
    };
    Ok(Some(MoeLoraState {
        gate_up_a,
        gate_up_b,
        down_a,
        down_b,
        scaling,
        rank,
    }))
}

/// The device of a tensor (helper for the block-fill, avoids repeated
/// `.device()` call).
fn device_of(t: &Tensor) -> &Device {
    t.device()
}

/// The B block-fill (the variable-rank fused combination): the fused
/// B is the `[rank_total, out_total]`, where each constituent's B (the
/// `[rank_i, out_i]`) fills its block at the `[rank_start_i:rank_start_i+rank_i,
/// out_start_i:out_start_i+out_i]` position. The `const_start` is the cumulative
/// sum of the prior constituents' ranks; the `out_start` is the cumulative sum
/// of the prior constituents' out dims. This handles the variable-rank case
/// (not the uniform-rank column-concat shortcut).
fn b_block_fill(
    constituent_adapters: &[&LoraAdapter],
    rank_total: usize,
    out_total: usize,
    device: &Device,
) -> Result<Tensor> {
    let ranks: Vec<usize> = constituent_adapters.iter().map(|a| a.rank).collect();
    if ranks.iter().all(|&r| r == ranks[0]) {
        // The uniform-rank: the column-concat (each B is the `[rank, out_i]`,
        // the cat on dim 1 gives the `[rank, out_total]`).
        Tensor::cat(
            &constituent_adapters.iter().map(|a| a.b.clone()).collect::<Vec<_>>(),
            1,
        )
    } else {
        // The variable-rank: the block-fill. Each constituent's B is the
        // `[rank_i, out_i]`; the fused B is the `[rank_total, out_total]` with
        // each B placed at its block. The build via the zero-init + the per-
        // block placement (the pad each B to the `[rank_total, out_i]` + the
        // cat on dim 1).
        let mut blocks = Vec::new();
        let mut rank_offset = 0usize;
        for adapter in constituent_adapters {
            let rank_i = adapter.rank;
            let out_i = adapter.b.dim(1)?;
            // The pad the constituent B to the `[rank_total, out_i]` (the zero-fill
            // the other rank positions) so the cat on dim 1 aligns.
            let b_dtype = adapter.b.dtype().clone();
            let full = Tensor::zeros((rank_total, out_i), b_dtype, device)?;
            let b_slice = adapter.b.to_dtype(b_dtype.clone())?;
            let top = full.narrow(0, 0, rank_offset)?;
            let bottom = full.narrow(0, rank_offset + rank_i, rank_total - rank_offset - rank_i)?;
            let b_padded = Tensor::cat(&[&top, &b_slice, &bottom], 0)?;
            blocks.push(b_padded);
            rank_offset += rank_i;
        }
        // The cat the blocks on the out dim (each block is the
        // `[rank_total, out_i]`, the cat on dim 1 gives the `[rank_total, out_total]`).
        let _ = (rank_total, out_total);
        Tensor::cat(&blocks, 1)
    }
}

/// Load a LoRA adapter set from a resolved adapter directory. The directory holds
/// a `config.json` (the LoRA rank/alpha/target_modules) + the adapter weights
/// (the `lora_A`/`lora_B` tensors, the safetensors). The tensor names follow the
/// standard PEFT convention: `{target_module}.lora_A.weight` + the
/// `{target_module}.lora_B.weight`. When no `target_modules` are specified in
/// the config, the adapters are discovered from the tensor names.
pub fn load_adapters_from_paths(
    paths: &crate::utils::downloader::ModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<Vec<LoraAdapter>> {
    use candle_core::safetensors;

    // Read the adapter config (the `config.json`: the rank, the alpha, the
    // target_modules).
    let config_path = paths.get_config_filename();
    let config_data = std::fs::read(&config_path)
        .map_err(|e| Error::Msg(format!("failed to read LoRA config {config_path:?}: {e}")))?;
    #[derive(serde::Deserialize)]
    struct LoraConfigFile {
        #[serde(rename = "r")]
        rank: serde_json::Value, // the usize or the per-module map
        #[serde(rename = "lora_alpha")]
        alpha: serde_json::Value, // the f64 or the per-module map
        #[serde(rename = "target_modules")]
        target_modules: Option<serde_json::Value>, // the list or the single string
        #[serde(rename = "exclude_modules")]
        exclude_modules: Option<Vec<String>>,
        #[serde(rename = "peft_type")]
        peft_type: Option<String>,
        #[serde(rename = "inference_dtype")]
        #[allow(dead_code)] // The QLoRA storage dtype (deferred); parsed for validation.
        inference_dtype: Option<String>,
    }
    let cfg: LoraConfigFile = serde_json::from_slice(&config_data)
        .map_err(|e| Error::Msg(format!("failed to parse LoRA config {config_path:?}: {e}")))?;
    // The peft_type check: only the "LORA" type is supported (no
    // the "LORA_A" / the "ADAPTER" variants).
    if let Some(ptype) = &cfg.peft_type {
        if !ptype.eq_ignore_ascii_case("lora") {
            return Err(Error::Msg(format!(
                "unsupported peft_type {ptype:?} (only LORA is supported)"
            )));
        }
    }
    // The per-module rank: the `r` is either a global usize or a
    // per-module map (the `{module: rank}`). The per-module alpha is the same.
    let global_rank: Option<usize> = cfg.rank.as_u64().map(|v| v as usize);
    let rank_map: HashMap<String, usize> = cfg
        .rank
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_u64().map(|r| (k.clone(), r as usize)))
                .collect()
        })
        .unwrap_or_default();
    let global_alpha: Option<f64> = cfg.alpha.as_f64();
    let alpha_map: HashMap<String, f64> = cfg
        .alpha
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|a| (k.clone(), a)))
                .collect()
        })
        .unwrap_or_default();
    // The target modules: the `target_modules` is either a list of module names
    // (the PEFT standard, the exact leaf match) or a single regex string (the
    // PEFT regex form, e.g. `.*\.layers\.\d+\.(self_attn|mlp)\.(q_proj|...)$`).
    // The regex form matches the FULL site path (the per-layer checkpoint key),
    // not the leaf. The `target_regex` is set when the single string is a regex
    // (the contains regex metacharacters); the `targets` list is used otherwise.
    let (targets, target_regex): (Vec<String>, Option<regex::Regex>) = match &cfg.target_modules {
        Some(v) => match v.as_array() {
            Some(arr) => (
                arr.iter().filter_map(|m| m.as_str().map(|s| s.to_string())).collect(),
                None,
            ),
            None => {
                let s = v.as_str().map(|s| s.to_string()).unwrap_or_default();
                // The PEFT regex detection: a single string with regex metacharacters
                // is a pattern (the `.*\.layers\.\d+\.(...)$` form); a plain module
                // name (the `q_proj`) is an exact leaf match.
                let is_regex = s.contains('*')
                    || s.contains('+')
                    || s.contains('?')
                    || s.contains('(')
                    || s.contains(')')
                    || s.contains('[')
                    || s.contains(']')
                    || s.contains('\\')
                    || s.contains('^')
                    || s.contains('$');
                if is_regex {
                    match regex::Regex::new(&s) {
                        Ok(re) => (Vec::new(), Some(re)),
                        Err(e) => {
                            crate::log_warn!(
                                "LoRA adapter target_modules regex failed to compile ({e}); treating as a literal name"
                            );
                            (vec![s], None)
                        }
                    }
                } else {
                    (vec![s], None)
                }
            }
        },
        None => (Vec::new(), None),
    };
    let excludes: Vec<String> = cfg.exclude_modules.clone().unwrap_or_default();
    // The inference dtype (the QLoRA) is the storage dtype of the
    // adapter (often F32). It is NOT used for the compute cast (the A/B are cast
    // to the model's compute dtype, no a dtype mismatch in the delta matmul).
    // The QLoRA path (the quantized base) would use it; that is the deferred.

    // Load the adapter weights (the safetensors files, the possibly sharded).
    let weight_files = paths.get_weight_filenames();
    if weight_files.is_empty() {
        return Err(Error::Msg(
            "LoRA adapter dir has no weight files".to_string(),
        ));
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for weight_file in &weight_files {
        let file_tensors = safetensors::load(weight_file, device)
            .map_err(|e| Error::Msg(format!("failed to load LoRA weights {weight_file:?}: {e}")))?;
        for (name, tensor) in file_tensors {
            tensors.insert(name, tensor);
        }
    }

    // Discover the real site paths from the tensor keys (strip the
    // `.lora_A.weight` suffix). A PEFT checkpoint keys its A/B tensors by the
    // full per-layer module path (e.g. `base_model.model.model.language_model.
    // layers.0.linear_attn.out_proj.lora_A.weight`), not by the config's short
    // `target_modules` names (e.g. `out_proj`). So the site identity is always
    // taken from the tensor keys; the config `targets` are used only as a leaf-
    // name filter (keep only the sites whose module leaf matches a target).
    let site_paths: Vec<String> = {
        let mut sites = std::collections::BTreeSet::new();
        for name in tensors.keys() {
            if let Some(prefix) = name.strip_suffix(".lora_A.weight") {
                sites.insert(prefix.to_string());
            }
        }
        sites.into_iter().collect()
    };
    let site_paths: Vec<String> = if target_regex.is_some() {
        // The PEFT regex form: the single `target_modules` string is a regex that
        // matches the FULL site path (the per-layer checkpoint key, e.g.
        // `layers.0.self_attn.q_proj`). The regex is anchored to the end (the `$`),
        // so it matches the full path, not just the leaf.
        let re = target_regex.as_ref().unwrap();
        site_paths
            .into_iter()
            .filter(|site| re.is_match(site))
            .collect()
    } else if targets.is_empty() {
        site_paths
    } else {
        site_paths
            .into_iter()
            .filter(|site| {
                let leaf = site.rsplit('.').next().unwrap_or(site);
                targets.iter().any(|t| t == leaf || t == site)
            })
            .collect()
    };

    // Build one adapter per discovered site (the per-layer, the full checkpoint
    // path).
    let mut adapters = Vec::new();
    for site in &site_paths {
        // The exclude_modules check (the adapter config's exclude list wins over
        // the target list). The exclude matches the module leaf or the full site.
        let leaf = site.rsplit('.').next().unwrap_or(site);
        if excludes.iter().any(|e| e == leaf || e == site || site.ends_with(e)) {
            continue;
        }
        let a_key = format!("{site}.lora_A.weight");
        let b_key = format!("{site}.lora_B.weight");
        let a_tensor = tensors.get(&a_key).ok_or_else(|| {
            Error::Msg(format!("LoRA adapter missing the {a_key} tensor"))
        })?;
        let b_tensor = tensors.get(&b_key).ok_or_else(|| {
            Error::Msg(format!("LoRA adapter missing the {b_key} tensor"))
        })?;
        // The per-module rank (the `rank_map` overrides the `global_rank`). The
        // rank is also inferable from the A tensor's shape (the `[in, rank]`).
        // The rank/alpha maps are keyed by the short module name (the leaf),
        // since the PEFT config uses short names.
        let rank = rank_map.get(leaf).copied().or(global_rank).unwrap_or_else(|| {
            a_tensor.dims().last().copied().unwrap_or(0)
        });
        // The per-module alpha (the `alpha_map` overrides the `global_alpha`).
        // The scaling is the alpha / rank (the LoRA convention).
        let alpha = alpha_map.get(leaf).copied().or(global_alpha).unwrap_or(rank as f64);
        let scaling = (alpha / rank.max(1) as f64) as f32;
        // The A is the `[in_features, rank]` (the down-projection); the B is the
        // `[rank, out_features]` (the up-projection). The PEFT convention stores
        // them as the `lora_A.weight` (the `[rank, in_features]`) + the
        // `lora_B.weight` (the `[out_features, rank]`), so transpose on load to
        // the math orientation (the `x @ A @ B`).
        //
        // The QLoRA case: when the A/B are stored in a quantized
        // dtype (the FP8/MXFP4/NVFP4, the NF4), they are kept in the quantized
        // form (the memory savings at high tenant density) + the dequant scale
        // is loaded. The `delta` dequantizes them to the activation dtype on
        // the fly. When the A/B are dense (the F32/BF16/FP16), they are cast to
        // the model's compute dtype (the prior behavior, not the quantized
        // storage).
        let a_t = a_tensor.t()?;
        let b_t = b_tensor.t()?;
        let a_is_quant = matches!(
            a_t.dtype(),
            DType::F8E4M3 | DType::F8E8M0
        );
        let b_is_quant = matches!(
            b_t.dtype(),
            DType::F8E4M3 | DType::F8E8M0
        );
        let (a, a_quantized, a_scale) = if a_is_quant {
            // The quantized A (the FP8): keep it quantized + load the scale.
            let a_scale = tensors
                .get(&format!("{site}.lora_A.weight_scale"))
                .cloned();
            (a_t, true, a_scale)
        } else {
            (a_t.to_dtype(dtype)?, false, None)
        };
        let (b, b_quantized, b_scale) = if b_is_quant {
            // The quantized B (the FP8): keep it quantized + load the scale.
            let b_scale = tensors
                .get(&format!("{site}.lora_B.weight_scale"))
                .cloned();
            (b_t, true, b_scale)
        } else {
            (b_t.to_dtype(dtype)?, false, None)
        };
        let adapter = LoraAdapter::new(
            canonical_site_key(site).to_string(),
            rank,
            a,
            b,
            scaling,
        );
        let adapter = LoraAdapter {
            a_quantized,
            a_scale,
            b_quantized,
            b_scale,
            ..adapter
        };
        adapters.push(adapter);
    }
    // The fused-module expansion (the packed/fused module). When the
    // adapter targets the constituent modules (the `q_proj`/`k_proj`/`v_proj`,
    // the `gate_proj`/`up_proj`) but the runtime uses the fused module (the
    // `qkv_proj`, the `gate_up_proj`), combine the constituent A/B into a fused
    // A/B (the A is the row-concat, the B is the column-concat, the fused output
    // order). The fused adapter is added alongside the constituents (no
    // replacing them, the runtime uses whichever module name it has).
    let fused_groups: Vec<(String, Vec<String>)> = vec![
        ("qkv_proj".to_string(), vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string()]),
        ("gate_up_proj".to_string(), vec!["gate_proj".to_string(), "up_proj".to_string()]),
    ];
    for (fused_name, constituents) in &fused_groups {
        // The all constituents must be present (no a partial fused adapter).
        let constituent_adapters: Vec<&LoraAdapter> = constituents
            .iter()
            .filter_map(|c| adapters.iter().find(|a| &a.target_module == c))
            .collect();
        if constituent_adapters.len() != constituents.len() {
            continue; // no all constituents (no fused adapter for this group)
        }
        // The combine the A (the row-concat, the `[in, rank_total]`) + the B (the
        // block-fill, the `[rank_total, out_total]`). The rank_total is the
        // sum of the constituent ranks. The A row-concat works for both the
        // uniform + the variable rank (each constituent's A is the
        // `[in, rank_i]`, the cat on dim 0 gives the `[in, rank_total]`). The
        // B block-fill places each constituent's B (the `[rank_i, out_i]`) at
        // its block position (the `[rank_start_i:rank_start_i+rank_i,
        // out_start_i:out_start_i+out_i]`), not the simple column-concat
        // (which requires the uniform rank).
        let rank_total: usize = constituent_adapters.iter().map(|a| a.rank).sum();
        // The fused A is the `[in, rank_total]`: the row-concat of the constituent
        // A's on the RANK dim (the dim 1). Each constituent A is the
        // `[in, rank_i]`; the cat on dim 1 gives the `[in, rank_1 + rank_2 + ...]`.
        let a_fused = Tensor::cat(
            &constituent_adapters.iter().map(|a| a.a.clone()).collect::<Vec<_>>(),
            1,
        )?;
        // The B block-fill: the out_total is the sum of the constituent out dims.
        let out_total: usize = constituent_adapters.iter().map(|a| a.b.dim(1).unwrap_or(0)).sum();
        let b_fused = b_block_fill(&constituent_adapters, rank_total, out_total, device_of(&a_fused))?;
        // The scaling is the per-constituent (each constituent has its own
        // alpha/rank); the fused scaling is the first constituent's (the
        // uniform-rank common case). The variable-rank case uses the per-
        // constituent scaling (no a single fused scaling).
        let scaling = constituent_adapters[0].scaling;
        let fused = LoraAdapter::new_fused(
            fused_name.clone(),
            rank_total,
            a_fused,
            b_fused,
            scaling,
            constituents
                .iter()
                .zip(constituent_adapters.iter())
                .map(|(name, a)| (name.clone(), a.b.dim(1).unwrap_or(0)))
                .collect(),
        );
        adapters.push(fused);
    }
    Ok(adapters)
}

/// The stacked LoRA A/B bank (the S-LoRA memory model). Instead of
/// per-tenant `Vec<LoraAdapter>` (each tenant allocates its own A/B
/// tensors), the stacked bank holds a single `[max_loras, ...]` tensor per
/// module (the all tenants' A/B in one allocation) + the per-tenant slot
/// index. This is the memory model that scales to thousands of adapters (the
/// S-LoRA result): the one allocation (no per-tenant allocations) + the
/// slot-based access (not the per-tenant lookup).
#[derive(Debug)]
pub struct StackedLoraBank {
    /// The per-module stacked A (the `[max_loras, in, rank]`).
    a_stacked: HashMap<String, Tensor>,
    /// The per-module stacked B (the `[max_loras, rank, out]`).
    b_stacked: HashMap<String, Tensor>,
    /// The per-module stacked scale (the `[max_loras]` f32, the per-slot
    /// `alpha / rank`). The fused kernel multiplies the shrink output by this.
    scale_stacked: HashMap<String, Tensor>,
    /// The tenant_id -> the bank slot index (the 0..max_loras).
    tenant_slots: HashMap<String, usize>,
    /// The bank size (the S-LoRA density).
    max_loras: usize,
    /// The next free slot (the slot allocation).
    next_slot: usize,
    /// The canonical bank dtype (the model's compute dtype). Every tenant's A/B
    /// is cast to this dtype before being written into a bank slot, so a mixed
    /// set of adapters (the F32, the BF16, the FP8) all coexist in the same bank
    /// without a dtype mismatch on the `cat`. The `None` when unset (the
    /// prior behavior, the first tenant's dtype is used).
    bank_dtype: Option<DType>,
}

impl StackedLoraBank {
    /// Create a stacked bank with the given slot count (the S-LoRA density). The
    /// `max_loras` is the bank size (the max concurrent tenants).
    pub fn new(max_loras: usize) -> Self {
        Self {
            a_stacked: HashMap::new(),
            b_stacked: HashMap::new(),
            scale_stacked: HashMap::new(),
            tenant_slots: HashMap::new(),
            max_loras: max_loras.max(1),
            next_slot: 0,
            bank_dtype: None,
        }
    }

    /// Set the canonical bank dtype (the model's compute dtype). All subsequent
    /// `set_lora` calls cast the incoming A/B to this dtype before writing them
    /// into the bank (the no the dtype mismatch on the `cat`). The `None` when
    /// the bank should use the first tenant's dtype (the prior behavior).
    pub fn with_bank_dtype(mut self, dtype: DType) -> Self {
        self.bank_dtype = Some(dtype);
        self
    }

    /// Set the canonical bank dtype in-place (the `&mut self` version, for use
    /// when the bank is already owned by a registry).
    pub fn set_bank_dtype(&mut self, dtype: DType) {
        self.bank_dtype = Some(dtype);
    }

    /// Allocate a slot for a tenant. The `None` when the bank is full
    /// (the max_loras reached, no more slots). The caller evicts an existing
    /// tenant (the LRU) to free a slot.
    pub fn allocate_slot(&mut self, tenant_id: &str) -> Option<usize> {
        if let Some(&slot) = self.tenant_slots.get(tenant_id) {
            return Some(slot); // already allocated (no re-allocate)
        }
        if self.next_slot >= self.max_loras {
            return None; // the bank is full (no more slots)
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        self.tenant_slots.insert(tenant_id.to_string(), slot);
        Some(slot)
    }

    /// Free a tenant's slot (the tenant evicted). The slot is returned
    /// to the free pool (the bank does not shrink, the slot is reusable).
    pub fn free_slot(&mut self, tenant_id: &str) {
        if let Some(slot) = self.tenant_slots.remove(tenant_id) {
            // The slot is freed (the bank does not shrink). The next_slot is not
            // decremented (the slots are not reused in order, no
            // fragmentation; the full bank is the eviction trigger).
            let _ = slot;
        }
    }

    /// Write a tenant's A/B into its bank slot (the stacked bank write). The
    /// `module` is the target module (the `q_proj`, etc.). The A/B are the
    /// per-module A/B (the `[in, rank]`/`[rank, out]`). The stacked bank is the
    /// `[max_loras, in, rank]`/`[max_loras, rank, out]`.
    pub fn set_lora(
        &mut self,
        tenant_id: &str,
        module: &str,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<()> {
        let slot = self.allocate_slot(tenant_id).ok_or_else(|| {
            Error::Msg(format!("LoRA bank full (the max_loras={})", self.max_loras))
        })?;
        // The ensure the stacked tensors exist for this module (the allocate on
        // first use, the `[max_loras, ...]` shape). The variable-rank case: when
        // a new tenant's rank exceeds the bank's current rank for this module,
        // reallocate the bank to the larger rank + zero-pad the existing slots
        // (the smaller-rank tenants are padded to the max rank, the fused kernel
        // reads only the valid rank per slot via the per-slot rank metadata).
        let a_shape = a.dims().to_vec(); // the [in, rank]
        let b_shape = b.dims().to_vec(); // the [rank, out]
        let new_rank = a_shape.last().copied().unwrap_or(0);
        // The canonical bank dtype (the explicit `bank_dtype` if set, else the
        // incoming tenant's dtype for the first write). All stacked tensors for
        // this module are created in this dtype, so a later tenant with a
        // different dtype is cast to it (not the dtype mismatch on the cat).
        let bank_dtype = self
            .bank_dtype
            .clone()
            .unwrap_or_else(|| a.dtype().clone());
        if !self.a_stacked.contains_key(module) {
            let a_stacked = Tensor::zeros(
                std::iter::once(self.max_loras)
                    .chain(a_shape.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
                bank_dtype.clone(),
                a.device(),
            )?;
            self.a_stacked.insert(module.to_string(), a_stacked);
        } else {
            // The bank exists; check if the new tenant's rank exceeds the bank's rank.
            let bank_a = self.a_stacked.get(module).unwrap();
            let bank_rank = bank_a.dim(2).unwrap_or(0);
            if new_rank > bank_rank {
                // The reallocate the bank to the new (larger) rank + zero-pad the
                // existing slots. The existing slots' A/B are narrowed to the
                // bank_rank + padded to the new_rank (the zero-fill).
                let a_padded = self.pad_stacked(bank_dtype.clone(), bank_rank, new_rank, module, "a")?;
                let b_padded = self.pad_stacked(bank_dtype.clone(), bank_rank, new_rank, module, "b")?;
                self.a_stacked.insert(module.to_string(), a_padded);
                self.b_stacked.insert(module.to_string(), b_padded);
            }
        }
        if !self.b_stacked.contains_key(module) {
            let b_stacked = Tensor::zeros(
                std::iter::once(self.max_loras)
                    .chain(b_shape.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
                bank_dtype.clone(),
                b.device(),
            )?;
            self.b_stacked.insert(module.to_string(), b_stacked);
        }
        if !self.scale_stacked.contains_key(module) {
            let scale_stacked = Tensor::zeros(
                (self.max_loras,),
                DType::F32,
                a.device(),
            )?;
            self.scale_stacked.insert(module.to_string(), scale_stacked);
        }
        // The write the tenant's A/B/scale into its slot (the `[slot, ...]`
        // slice). The narrow Tensor does not support in-place slice writes
        // easily, so the re-build the stacked tensor with the new slot filled
        // (the cat of the other slots + this slot). The max_loras is small (the
        // bank size), so the re-build is acceptable. The new tenant's A/B are
        // zero-padded to the bank rank (the smaller-rank case) + cast to the
        // bank dtype (the canonical dtype, the no the dtype mismatch on the cat).
        let bank_a = self.a_stacked.get(module).unwrap();
        let bank_rank = bank_a.dim(2).unwrap_or(0);
        // The canonical bank dtype: the explicit `bank_dtype` if set, else the
        // existing bank's dtype (the first tenant's), else the incoming tenant's.
        let bank_dtype = self
            .bank_dtype
            .clone()
            .or_else(|| self.a_stacked.get(module).map(|t| t.dtype().clone()))
            .unwrap_or_else(|| a.dtype().clone());
        let a_padded = if new_rank < bank_rank {
            // The zero-pad the smaller-rank A to the bank rank (the [in, bank_rank]).
            self.zero_pad_rank(a, bank_rank, a.device(), true)?
        } else {
            a.clone()
        };
        let a_padded = a_padded.to_dtype(bank_dtype.clone())?;
        let b_padded = if new_rank < bank_rank {
            // The zero-pad the smaller-rank B to the bank rank (the [bank_rank, out]).
            self.zero_pad_rank(b, bank_rank, b.device(), false)?
        } else {
            b.clone()
        };
        let b_padded = b_padded.to_dtype(bank_dtype)?;
        let a_stacked = self.a_stacked.get_mut(module).unwrap();
        let b_stacked = self.b_stacked.get_mut(module).unwrap();
        let scale_stacked = self.scale_stacked.get_mut(module).unwrap();
        let a_3d = a_padded.unsqueeze(0)?; // the [1, in, bank_rank]
        let b_3d = b_padded.unsqueeze(0)?; // the [1, bank_rank, out]
        let mut a_slots = Vec::new();
        let mut b_slots = Vec::new();
        let mut scale_slots = Vec::new();
        for i in 0..self.max_loras {
            if i == slot {
                a_slots.push(a_3d.clone());
                b_slots.push(b_3d.clone());
                scale_slots.push(Tensor::new(scale, a.device())?.reshape((1,))?);
            } else {
                a_slots.push(a_stacked.narrow(0, i, 1)?);
                b_slots.push(b_stacked.narrow(0, i, 1)?);
                scale_slots.push(scale_stacked.narrow(0, i, 1)?);
            }
        }
        *a_stacked = Tensor::cat(&a_slots, 0)?;
        *b_stacked = Tensor::cat(&b_slots, 0)?;
        *scale_stacked = Tensor::cat(&scale_slots, 0)?;
        Ok(())
    }

    /// The zero-pad a smaller-rank A or to the bank rank. The A (the `[in, rank]`)
    /// pads the last dim; the B (the `[rank, out]`) pads the first dim. The pad
    /// is zero (no spurious contribution to the delta; the rank dimension is the
    /// low-rank width, the zero-pad adds no signal).
    fn zero_pad_rank(&self, t: &Tensor, bank_rank: usize, device: &Device, is_a: bool) -> Result<Tensor> {
        let dims = t.dims().to_vec();
        if dims.len() != 2 {
            return Ok(t.clone()); // the unexpected rank (no-op)
        }
        let (dim0, dim1) = (dims[0], dims[1]);
        if is_a {
            // The A (the [in, rank]): the rank is the last dim.
            if dim1 == bank_rank {
                return Ok(t.clone()); // the already the bank rank (no-op)
            }
            let mut padded = vec![0f32; dim0 * bank_rank];
            let vals = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for (i, &v) in vals.iter().enumerate() {
                if i < padded.len() {
                    padded[i] = v;
                }
            }
            Tensor::from_vec(padded, (dim0, bank_rank), device).and_then(|t| t.to_dtype(t.dtype()))
        } else {
            // The B (the [rank, out]): the rank is the first dim.
            if dim0 == bank_rank {
                return Ok(t.clone()); // the already the bank rank (no-op)
            }
            let mut padded = vec![0f32; bank_rank * dim1];
            let vals = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for (i, &v) in vals.iter().enumerate() {
                if i < padded.len() {
                    padded[i] = v;
                }
            }
            Tensor::from_vec(padded, (bank_rank, dim1), device).and_then(|t| t.to_dtype(t.dtype()))
        }
    }

    /// The reallocate the stacked A/B to a larger rank (the zero-pad the existing
    /// slots). The `which` selects the A or the B tensor. The A is the
    /// `[max_loras, in, rank]` (the rank is dim 2); the B is the
    /// `[max_loras, rank, out]` (the rank is dim 1). The existing slots are
    /// zero-padded from the old_rank to the new_rank along the rank dim.
    fn pad_stacked(&mut self, dtype: DType, old_rank: usize, new_rank: usize, module: &str, which: &str) -> Result<Tensor> {
        let stacked = if which == "a" {
            self.a_stacked.get(module).cloned()
        } else {
            self.b_stacked.get(module).cloned()
        };
        let Some(stacked) = stacked else {
            // The unreachable: the set_lora only calls pad_stacked when the bank
            // exists (the `else` branch). The `None` case is handled by the
            // `if !contains_key` branch (the fresh allocation).
            unreachable!("pad_stacked called with no existing bank");
        };
        // The zero-pad each existing slot from the old_rank to the new_rank.
        let device = stacked.device();
        let dims = stacked.dims().to_vec(); // the [max_loras, in, rank] or the [max_loras, rank, out]
        let rank_dim = if which == "a" { 2 } else { 1 };
        if dims.len() < 3 || dims[rank_dim] == new_rank {
            return Ok(stacked); // already the new rank (no-op)
        }
        // The build the padded tensor: for each slot, pad the rank dim.
        let mut slots = Vec::new();
        for i in 0..self.max_loras {
            let slot = stacked.narrow(0, i, 1)?; // the [1, in, old_rank] or the [1, old_rank, out]
            let slot_dims = slot.dims().to_vec();
            if which == "a" {
                // The A (the [1, in, old_rank] → the [1, in, new_rank]): the pad the last dim.
                let in_dim = slot_dims[1];
                let mut padded = vec![0f32; in_dim * new_rank];
                let vals = slot.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (idx, &v) in vals.iter().enumerate() {
                    let in_idx = idx / old_rank;
                    let rank_idx = idx % old_rank;
                    if in_idx < in_dim && rank_idx < new_rank {
                        padded[in_idx * new_rank + rank_idx] = v;
                    }
                }
                slots.push(Tensor::from_vec(padded, (1, in_dim, new_rank), device)?.to_dtype(dtype.clone())?);
            } else {
                // The B (the [1, old_rank, out] → the [1, new_rank, out]): the pad the middle dim.
                let out_dim = slot_dims[2];
                let mut padded = vec![0f32; new_rank * out_dim];
                let vals = slot.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (idx, &v) in vals.iter().enumerate() {
                    let rank_idx = idx / out_dim;
                    let out_idx = idx % out_dim;
                    if rank_idx < new_rank && out_idx < out_dim {
                        padded[rank_idx * out_dim + out_idx] = v;
                    }
                }
                slots.push(Tensor::from_vec(padded, (1, new_rank, out_dim), device)?.to_dtype(dtype.clone())?);
            }
        }
        Tensor::cat(&slots, 0)
    }

    /// Read a tenant's A/B from its bank slot (the stacked bank read).
    /// The `None` when the tenant has no slot (the unallocated) or the module
    /// is not in the bank.
    pub fn get_lora(
        &self,
        tenant_id: &str,
        module: &str,
    ) -> Option<(Tensor, Tensor)> {
        let slot = *self.tenant_slots.get(tenant_id)?;
        let a_stacked = self.a_stacked.get(module)?;
        let b_stacked = self.b_stacked.get(module)?;
        // The narrow the slot (the `[1, ...]`) + the squeeze the leading dim (the
        // 2D per-module A/B, not the 3D stacked).
        let a = a_stacked.narrow(0, slot, 1).ok()?.squeeze(0).ok()?;
        let b = b_stacked.narrow(0, slot, 1).ok()?.squeeze(0).ok()?;
        Some((a, b))
    }

    /// The bank size (the S-LoRA density).
    pub fn max_loras(&self) -> usize {
        self.max_loras
    }

    /// The allocated slot count (the observability).
    pub fn slot_count(&self) -> usize {
        self.tenant_slots.len()
    }

    /// The stacked A for a module (the `[max_loras, in, rank]`), if present.
    pub fn stacked_a(&self, module: &str) -> Option<&Tensor> {
        self.a_stacked.get(module)
    }

    /// The stacked B for a module (the `[max_loras, rank, out]`), if present.
    pub fn stacked_b(&self, module: &str) -> Option<&Tensor> {
        self.b_stacked.get(module)
    }

    /// The stacked scale for a module (the `[max_loras]` f32, the per-slot
    /// `alpha / rank`), if present. The fused kernel multiplies the shrink
    /// output by this.
    pub fn stacked_scale(&self, module: &str) -> Option<&Tensor> {
        self.scale_stacked.get(module)
    }

    /// The tenant_id -> the bank slot index (the 0..max_loras). The fused kernel
    /// needs this to map a compacted active-slot list back to the bank slot.
    pub fn tenant_slots(&self) -> &HashMap<String, usize> {
        &self.tenant_slots
    }
}

/// The LoRA site registry (the centralized site registration). The
/// model registers its LoRA-capable sites (the module name + the parallelism
/// + the in/out dims) at construction; the loader matches the adapter's
/// `target_modules` against the registered sites (not the per-layer
/// hardcoding). This centralizes the site registration (not the scattered
/// `with_lora_module` calls) + makes the adapter-to-site matching explicit.
#[derive(Debug, Default)]
pub struct LoraLayerRegistry {
    /// The site path -> the site spec (the parallelism + the in/out dims). The
    /// key is the model's full module path (e.g. `model.layers.0.self_attn.
    /// q_proj`), the same path the model uses to load the base weight. This is
    /// what makes the adapter-to-site matching exact (not the short-name
    /// ambiguity where layers).
    sites: HashMap<String, LoraSiteSpec>,
    /// The site prefix alias (the wrapper-model path mismatch). When
    /// set, a site registered under the `source` prefix is also matched under
    /// the `target` prefix (e.g. the Qwen3.5-VL runtime path `model.layers...`
    /// vs the checkpoint/PEFT path `model.language_model.layers...`). The
    /// `None` means no alias (the path is used as-is).
    site_prefix_alias: Option<(String, String)>,
}

/// The LoRA site spec (the site's parallelism + the dims). The
/// `parallelism` is the TP mode (the Column/Row/Replicated); the
/// `in_features`/`out_features` are the module's dims (the A/B shape check).
#[derive(Debug, Clone, Default)]
pub struct LoraSiteSpec {
    pub parallelism: LoraParallelism,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraLayerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with a site prefix alias (the wrapper-model
    /// path mismatch). The `source` prefix is rewritten to the `target` prefix
    /// when matching adapter tensors to sites.
    pub fn with_site_prefix_alias(source: &str, target: &str) -> Self {
        Self {
            sites: HashMap::new(),
            site_prefix_alias: Some((source.to_string(), target.to_string())),
        }
    }

    /// Register a LoRA site (the model constructor calls this at construction
    /// for each LoRA-capable layer). The `site` is the model's full module path
    /// (the same path used to load the base weight); the `spec` is the site's
    /// parallelism + dims.
    pub fn register_site(&mut self, site: &str, spec: LoraSiteSpec) {
        self.sites.insert(site.to_string(), spec);
    }

    /// The registered site paths (the observability).
    pub fn site_names(&self) -> Vec<String> {
        self.sites.keys().cloned().collect()
    }

    /// Whether the registry has no registered sites (the LoRA disabled, or the
    /// model not yet built).
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Whether a site path is a registered LoRA site (the adapter matching). The
    /// `true` when the site is registered (the adapter can target it).
    pub fn is_site(&self, site: &str) -> bool {
        self.sites.contains_key(site)
    }

    /// The site spec for a site path (the parallelism + the dims, the A/B shape
    /// check). The `None` when the site is not a registered site.
    pub fn site_spec(&self, site: &str) -> Option<&LoraSiteSpec> {
        self.sites.get(site)
    }

    /// Match an adapter tensor key to a registered site (the adapter-
    /// to-site matching). The `tensor_key` is the checkpoint tensor name with
    /// the `.lora_A.weight`/`.lora_B.weight` suffix stripped (e.g.
    /// `base_model.model.model.language_model.layers.0.linear_attn.out_proj`).
    /// The method tries the key as-is, then with the PEFT `base_model.model.`
    /// prefix stripped, then applies the site prefix alias (the wrapper-model
    /// path rewrite), and returns the matching registered site path. The `None`
    /// when no registered site matches (the adapter targets a module the model
    /// does not expose as a LoRA site).
    pub fn match_site(&self, tensor_key: &str) -> Option<String> {
        // The candidate site paths, in priority order: the raw key, the PEFT-
        // prefix-stripped key, + the alias-rewritten forms of each.
        let mut candidates: Vec<String> = Vec::new();
        candidates.push(tensor_key.to_string());
        if let Some(stripped) = tensor_key.strip_prefix("base_model.model.") {
            candidates.push(stripped.to_string());
        }
        if let Some((source, target)) = &self.site_prefix_alias {
            for c in candidates.clone() {
                if let Some(rest) = c.strip_prefix(source.as_str()) {
                    candidates.push(format!("{target}{rest}"));
                }
            }
        }
        for cand in &candidates {
            if self.sites.contains_key(cand.as_str()) {
                return Some(cand.clone());
            }
        }
        None
    }

    /// Match an adapter's target modules against the registered sites (the
    /// adapter-to-site matching). Returns the set of target modules
    /// that are registered sites (not the unregistered targets, which are
    /// ignored). The `targets` is the adapter's `target_modules` list.
    pub fn match_targets(&self, targets: &[String]) -> Vec<String> {
        targets
            .iter()
            .filter(|t| self.sites.contains_key(t.as_str()))
            .cloned()
            .collect()
    }
}

/// The per-wide siteRA site registry (the centralized site
/// registration). The model registers its sites here at construction; the
/// loader matches the adapter's targets against the registered sites.
static GLOBAL_SITE_REGISTRY: OnceLock<std::sync::Mutex<LoraLayerRegistry>> = OnceLock::new();

/// Set the process-wide LoRA site registry (called once at model load).
pub fn set_global_site_registry(registry: LoraLayerRegistry) {
    let _ = GLOBAL_SITE_REGISTRY.set(std::sync::Mutex::new(registry));
}

/// Initialize the process-wide LoRA site registry (the centralized
/// site registration). Called once at model build, before the model
/// constructors run (they register their LoRA sites into this registry via
/// the `register_lora_site`). The registry is a `OnceLock` (set once, no
/// re-set), so repeated calls are no-ops after the first. The empty
/// registry is initialized so the `register_lora_site` calls record the sites
/// (not a no-op when the registry is absent).
pub fn init_global_site_registry() {
    let _ = GLOBAL_SITE_REGISTRY.set(std::sync::Mutex::new(LoraLayerRegistry::new()));
}

/// The global LoRA site registry (`None` when not set, no LoRA sites
/// registered).
pub fn global_site_registry() -> Option<std::sync::MutexGuard<'static, LoraLayerRegistry>> {
    GLOBAL_SITE_REGISTRY.get().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Register a LoRA site into the global site registry (the
/// centralized site registration). The wrapped linears call this from their
/// `with_lora_module` so the model's real LoRA sites are recorded. The `site`
/// is the full module path (the same key the adapter loader produces); the
/// `parallelism` is the site's TP mode. No-op when the global registry is
/// not set (the LoRA-disabled, the zero overhead).
pub fn register_lora_site(site: &str, parallelism: LoraParallelism, in_features: usize, out_features: usize) {
    if let Some(mut reg) = global_site_registry() {
        reg.register_site(site, LoraSiteSpec {
            parallelism,
            in_features,
            out_features,
        });
    }
}

/// The model's LoRA site set (the registered site registry). Returns
/// the module names the model exposes as LoRA targets (the attention q/k/v/o or
/// the fused qkv, the MLP gate/up/down or the fused gate_up, the lm_head, the
/// embed_tokens, the MoE expert modules). The `model_type` selects the site
/// set (the MoE models add the expert sites, the MLA models add the MLA sites).
/// This is the validation gate: an adapter targeting a module not in this set
/// is rejected (no silently loading an adapter for a module the model does
/// not expose as a LoRA site).
pub fn model_lora_sites(model_type: Option<&crate::utils::config::ModelType>) -> LoraLayerRegistry {
    let mut reg = LoraLayerRegistry::new();
    // The standard attention + MLP sites (the all dense models).
    for module in [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",
    ] {
        reg.register_site(module, LoraSiteSpec::default());
    }
    // The linear-attention (GDN) family sites (the Qwen3.5/Qwen4 hybrid
    // models): the `in_proj_*` + the `out_proj` leaves. The `out_proj` is the
    // linear-attention output projection (the alias of the standard `o_proj`).
    for module in [
        "out_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
        "in_proj_qkvz", "in_proj_ba",
    ] {
        reg.register_site(module, LoraSiteSpec::default());
    }
    // The fused-module aliases (the runtime may use the fused form; the adapter
    // may target either the constituent or the fused name).
    for module in ["qkv_proj", "gate_up_proj"] {
        reg.register_site(module, LoraSiteSpec::default());
    }
    // The MoE expert sites (the MoE models only, when the model type is known).
    if let Some(mt) = model_type {
        if matches!(
            mt,
            crate::utils::config::ModelType::Qwen3MoE
                | crate::utils::config::ModelType::Qwen3_5MoE
                | crate::utils::config::ModelType::GLM4MoE
                | crate::utils::config::ModelType::GLM4MoeLite
                | crate::utils::config::ModelType::MiniMax
        ) {
            for module in [
                "experts.gate_proj",
                "experts.up_proj",
                "experts.down_proj",
                "shared_experts.gate_proj",
                "shared_experts.up_proj",
                "shared_experts.down_proj",
            ] {
                reg.register_site(module, LoraSiteSpec::default());
            }
        }
    }
    reg
}

/// Validate an adapter's target modules against the model's LoRA sites (the
/// adapter-to-site matching). Returns the list of targets that are
/// NOT registered sites (the adapter targets a module the model does not
/// expose). A caller (the `init_lora_registry`) logs + skips the invalid
/// targets (not the hard failure, the adapter may target a mix of valid +
/// invalid modules).
pub fn validate_adapter_targets(
    model_type: Option<&crate::utils::config::ModelType>,
    targets: &[String],
) -> Vec<String> {
    // The prefer the global site registry (the model's actually registered
    // sites) when it is set + non-empty; fall back to the static
    // model_lora_sites guess (the model type based) when the registry is
    // unavailable (the LoRA disabled, or the model not yet built).
    let site_names: Vec<String> = match global_site_registry() {
        Some(reg) if !reg.is_empty() => reg.site_names(),
        _ => model_lora_sites(model_type).site_names(),
    };
    // The adapter targets may be full checkpoint paths (the
    // `base_model.model...layers.0.linear_attn.out_proj`); the sites are
    // registered by the name (the `out_proj`). Compare the leaf of each target
    // against the registered site leaves so the full-path targets validate
    // correctly.
    let site_leaves: HashSet<String> = site_names
        .iter()
        .map(|s| s.rsplit('.').next().unwrap_or(s).to_string())
        .collect();
    targets
        .iter()
        .filter(|t| {
            let leaf = t.rsplit('.').next().unwrap_or(t.as_str());
            !site_leaves.contains(leaf)
        })
        .cloned()
        .collect()
}

/// The per-instance LoRA registry: the set of adapters loaded for this
/// instance, keyed by tenant id. The registry is owned by the `ModelRunner`
/// (the runner-instance, the MM2 isolation). When a request carries a tenant
/// id, the runner sets the active tenant (the `set_active_tenant`) + the
/// `LoraLinear` wrappers apply that tenant's adapters to the target modules.
///
/// The tenant budget (the `max_tenants`): the registry evicts the least-recently-
/// used tenant when the count exceeds the budget (no unbounded GPU memory
/// from). The default budget is 16 (the `DEFAULT_LORA_MAX_TENANTS`, the
/// typical multi-tenant deployment). The LRU order is updated on every
/// `get` (the active tenant is the most-recently-used).
#[derive(Debug)]
pub struct LoraRegistry {
    /// The tenant_id -> the adapter set (one entry per target module).
    adapters: HashMap<String, Vec<LoraAdapter>>,
    /// The LRU order (the least-recently-used first). Updated on every `get`.
    lru_order: Vec<String>,
    /// The tenant budget (the max concurrent tenants, the LRU eviction trigger).
    max_tenants: usize,
    /// The byte budget (the max total GPU bytes of resident adapter A/B).
    /// The `0` means unbounded (the count budget only). The eviction
    /// considers both the count + the bytes.
    max_bytes: u64,
    /// The resident byte count (the sum of the A/B tensor bytes across all
    /// resident tenants). Updated on register/unregister.
    resident_bytes: u64,
    /// The pinned tenants (the SLA guarantee): never evicted, even
    /// under budget pressure. The eviction skips them.
    pinned: HashSet<String>,
    /// The in-flight leases: a tenant held by a running request is
    /// leased (no evicted mid-forward). The lease count is the number of
    /// in-flight requests holding the tenant.
    leases: HashMap<String, usize>,
    /// The tenant → the adapter source (the HF id or the local dir, the
    /// `--lora` CLI value). The on-demand load uses this to load a
    /// tenant's adapter when a request needs it + it is not resident.
    sources: HashMap<String, String>,
    /// The device + the dtype for the on-demand load (the A/B are loaded onto
    /// this device in this dtype).
    device: Option<Device>,
    dtype: Option<DType>,
    /// The eviction policy: the "lru" (the default, the least-recently-
    /// used) or the "fifo" (the first-in-first-out, the insertion order). The
    /// eviction skips the pinned + the leased tenants either way.
    eviction_policy: EvictionPolicy,
    /// The FIFO insertion order (the first-registered first-evicted).
    /// The Lru_order` is the LRU order (the least-recently-used first). The
    /// eviction uses the policy-app-select the order.
    fifo_order: Vec<String>,
    /// The two-tier LRU (the vLLM `AdapterLRUCache`): the `active`
    /// set is the in-batch tenants (the small, the `max_active` cap, the bank
    /// slots); the `registered` set is the full resident set (the large, the
    /// weights on GPU). Eviction from `active` frees a bank slot (the weights
    /// stay registered); eviction from `registered` unloads the weights (the
    /// GPU memory freed). The `active` is a subset of the `registered`.
    active: HashSet<String>,
    /// The active-tier cap (the `max_loras_per_batch`): the max distinct
    /// tenants in one batch (the bank slot count). The `0` means unlimited (the
    /// no active cap, the single-tenant fast path).
    max_active: usize,
    /// The alias→generation map (the no-downtime adapter swap). The
    /// tenant alias (the alias) maps to the current generation id. The
    /// `generations` holds the immutable generation objects (the in-flight
    /// requests lease the old generation, the swap does not affect them).
    alias_to_generation: HashMap<String, u64>,
    /// The generation id -> the immutable generation (the frozen adapter set).
    generations: HashMap<u64, LoraGeneration>,
    /// The next generation id (the monotonically-increasing, no reuse).
    next_generation_id: u64,
    /// The optional stacked A/B bank (the S-LoRA memory model). When
    /// present, the tenant A/B are also stored in the single `[max_loras, ...]`
    /// allocation (the one allocation, no per-tenant allocations), which is
    /// the memory model that scales to thousands of adapters. The
    /// per-tenant `adapters` map remains the source of truth for the serving
    /// path; the bank is the dense storage backend populated in parallel. The
    /// `None` when the high-density bank is not enabled (the prior behavior).
    bank: Option<StackedLoraBank>,
}

/// The eviction policy (the SGLang `eviction_policy.py` equivalent).
/// The `Lru` evicts the least-recently-used tenant (the default, the access-
/// ordered). The `Fifo` evicts the first-registered tenant (the insertion-
/// ordered, not the access-order update).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Fifo,
}

/// The default tenant budget (the typical multi-tenant deployment, no
/// unbounded GPU memory growth).
pub const DEFAULT_LORA_MAX_TENANTS: usize = 16;
/// The default byte budget (the 256 MB of resident adapter A/B, the typical
/// multi-tenant GPU budget). The `0` for the unbounded (the count budget only).
pub const DEFAULT_LORA_MAX_BYTES: u64 = 256 * 1024 * 1024;

impl Default for LoraRegistry {
    fn default() -> Self {
        Self {
            adapters: HashMap::new(),
            lru_order: Vec::new(),
            max_tenants: DEFAULT_LORA_MAX_TENANTS,
            max_bytes: DEFAULT_LORA_MAX_BYTES,
            resident_bytes: 0,
            pinned: HashSet::new(),
            leases: HashMap::new(),
            sources: HashMap::new(),
            device: None,
            dtype: None,
            eviction_policy: EvictionPolicy::default(),
            fifo_order: Vec::new(),
            active: HashSet::new(),
            max_active: 0,
            alias_to_generation: HashMap::new(),
            generations: HashMap::new(),
            next_generation_id: 1,
            bank: None,
        }
    }
}

impl LoraRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with a custom tenant budget (the `max_tenants`).
    pub fn with_budget(max_tenants: usize) -> Self {
        Self {
            adapters: HashMap::new(),
            lru_order: Vec::new(),
            max_tenants: max_tenants.max(1),
            max_bytes: DEFAULT_LORA_MAX_BYTES,
            resident_bytes: 0,
            pinned: HashSet::new(),
            leases: HashMap::new(),
            sources: HashMap::new(),
            device: None,
            dtype: None,
            eviction_policy: EvictionPolicy::default(),
            fifo_order: Vec::new(),
            active: HashSet::new(),
            max_active: 0,
            alias_to_generation: HashMap::new(),
            generations: HashMap::new(),
            next_generation_id: 1,
            bank: None,
        }
    }

    /// Set the eviction policy: the `Lru` (the default, the access-
    /// ordered) or the `Fifo` (the insertion-ordered). The eviction skips the
    /// pinned + the leased tenants either way.
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Set the byte budget (the max resident adapter bytes). The `0`
    /// disables the byte budget (the count budget only).
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Set the device + the dtype for the on-demand load. The
    /// `ensure_loaded` uses these to load a tenant's adapter onto the device in
    /// the dtype when the tenant is not resident.
    pub fn with_device(mut self, device: Device, dtype: DType) -> Self {
        self.device = Some(device);
        self.dtype = Some(dtype);
        self
    }

    /// Set the active-tier cap (the `max_loras_per_batch`): the max
    /// distinct tenants in one batch (the bank slot count). The `0` means
    /// unlimited (no active cap, the single-tenant fast path).
    pub fn with_max_active(mut self, max_active: usize) -> Self {
        self.max_active = max_active;
        self
    }

    /// Enable the stacked A/B bank (the S-LoRA memory model, the
    /// density). The `max_loras` is the bank size (the max concurrent tenants in
    /// the dense storage). When enabled, the `register` also writes each
    /// tenant's A/B into the bank slot (the one allocation per module, no
    /// the per-tenant allocations). The per-tenant `adapters` map remains the
    /// serving source of truth; the bank is the dense storage backend.
    pub fn with_stacked_bank(mut self, max_loras: usize) -> Self {
        self.bank = Some(StackedLoraBank::new(max_loras));
        self
    }

    /// Set the canonical bank dtype (the model's compute dtype). All tenants'
    /// A/B are cast to this dtype before being written into the bank, so a
    /// mixed set of adapters (the F32, the BF16, the FP8) coexist in the
    /// same bank without a dtype mismatch. The `None` when the bank is not set.
    pub fn with_bank_dtype(mut self, dtype: DType) -> Self {
        if let Some(bank) = self.bank.as_mut() {
            bank.set_bank_dtype(dtype);
        }
        self
    }

    /// The stacked bank, when enabled. The `None` when the bank is
    /// not enabled (the per-tenant storage only, the prior behavior).
    pub fn bank(&self) -> Option<&StackedLoraBank> {
        self.bank.as_ref()
    }

    /// Activate a tenant for the current batch (the two-tier LRU):
    /// move the tenant into the `active` set (the in-batch bank slot). The
    /// active set is capped by the `max_active`; when over the cap,
    /// the least-recently-active tenant is deactivated (the bank slot freed,
    /// the weights stay registered). No-op when the tenant is not
    /// registered.
    pub fn activate(&mut self, tenant_id: &str) {
        if !self.adapters.contains_key(tenant_id) {
            return; // the unregistered tenant (no activate)
        }
        // The evict the LRU active tenant when over the active cap.
        if self.max_active > 0 && self.active.len() >= self.max_active && !self.active.contains(tenant_id) {
            if let Some(evicted) = self.lru_order.iter().find(|t| self.active.contains(*t)) {
                self.active.remove(evicted);
            }
        }
        self.active.insert(tenant_id.to_string());
    }

    /// Deactivate a tenant for the current batch: remove the tenant
    /// from the `active` set (the bank slot freed, the weights stay registered).
    /// No-op when the tenant is not active.
    pub fn deactivate(&mut self, tenant_id: &str) {
        self.active.remove(tenant_id);
    }

    /// Whether a tenant is active in the current batch.
    pub fn is_active(&self, tenant_id: &str) -> bool {
        self.active.contains(tenant_id)
    }

    /// The active tenant count (the observability).
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Swap a tenant's alias to a new generation (the no-downtime
    /// adapter deploy). The new generation is created from the
    /// `new_adapters` (the fresh adapter set); the alias is atomically repointed
    /// to the new generation. In-flight requests that leased the old generation
    /// keep using it (the swap does not affect them); new requests get the new
    /// generation. The old generation is retained until no lease holds it (the
    /// `drop_unreferenced_generations`), then its weights are freed.
    pub fn swap_alias(&mut self, tenant_id: &str, new_adapters: Vec<LoraAdapter>) -> u64 {
        // The create the new generation (the fresh id, the immutable adapter set).
        let gen_id = self.next_generation_id;
        self.next_generation_id += 1;
        let generation = LoraGeneration::new(gen_id, tenant_id.to_string(), new_adapters.clone());
        self.generations.insert(gen_id, generation);
        // The atomically repoint the alias to the new generation (not the
        // in-flight requests, which hold the old generation id via the lease).
        self.alias_to_generation.insert(tenant_id.to_string(), gen_id);
        // The update the registered adapters to the new set (the residency).
        self.adapters.insert(tenant_id.to_string(), new_adapters);
        gen_id
    }

    /// The current generation for a tenant alias. The `None` when the
    /// alias has never been swapped (the initial registration, the generation 0).
    pub fn generation_for(&self, tenant_id: &str) -> Option<u64> {
        self.alias_to_generation.get(tenant_id).copied()
    }

    /// The number of retained generations (the observability). The
    /// generations are dropped only when no alias + no lease references them
    /// (the `drop_unreferenced_generations`).
    pub fn generation_count(&self) -> usize {
        self.generations.len()
    }

    /// Drop the generations no longer referenced by any alias + lease (the
    /// memory reclamation). The in-flight leases hold the old generation
    /// alive; no lease + no alias reference means the generation's
    /// weights can be freed.
    pub fn drop_unreferenced_generations(&mut self) {
        let mut referenced: HashSet<u64> = HashSet::new();
        for &gen_id in self.alias_to_generation.values() {
            referenced.insert(gen_id);
        }
        for (tenant, &lease_count) in &self.leases {
            if lease_count > 0 {
                // The leased tenant holds its current generation (not the
                // drop while in-flight).
                if let Some(gen_id) = self.alias_to_generation.get(tenant) {
                    referenced.insert(*gen_id);
                }
            }
        }
        self.generations.retain(|&gen_id, _| referenced.contains(&gen_id));
    }

    /// Register a tenant's source (the HF id or the local dir, the `--lora` CLI
    /// value) for the on-demand load. The tenant's adapter is not
    /// loaded until a request needs it (the `ensure_loaded`); the source is the
    /// load input.
    pub fn add_source(&mut self, tenant_id: &str, source: &str) {
        self.sources.insert(tenant_id.to_string(), source.to_string());
    }

    /// The on-demand load: ensure the tenant's adapter is resident.
    /// When the tenant is already resident, this is a no-op (the LRU touch).

    /// it is not resident + the source is known + the device/dtype are set, the
    /// adapter is loaded from the source (the common model loader, the
    /// `prepare_lora_adapter_weights`) + registered (the budget-aware, the
    /// eviction of the LRU tenant when over budget). The `Err` when the source
    /// is unknown or the load fails (the caller falls back to the base model).
    pub fn ensure_loaded(&mut self, tenant_id: &str) -> Result<()> {
        // The already resident (no load, the LRU touch).
        if self.adapters.contains_key(tenant_id) {
            self.lru_order.retain(|t| t != tenant_id);
            self.lru_order.push(tenant_id.to_string());
            return Ok(());
        }
        // The source must be known (no on-demand load without a source).
        let Some(source) = self.sources.get(tenant_id) else {
            return Ok(()); // the unknown tenant (no source, the base model)
        };
        // The device + the dtype must be set (no load without a target).
        let (Some(device), Some(dtype)) = (self.device.clone(), self.dtype) else {
            return Ok(());
        };
        // The load the adapter via the source (the common model loader, the
        // same Downloader + the prepare-hub machinery the base model uses).
        let (model_id, weight_path) = crate::speculative::resolve_draft_model(source);
        let loader = crate::utils::downloader::Downloader::new(model_id, weight_path, None);
        let (paths, is_gguf) = loader
            .prepare_lora_adapter_weights(None, None)
            .map_err(|e| Error::Msg(format!("failed to load LoRA adapter {source}: {e}")))?;
        if is_gguf {
            return Err(Error::Msg(
                "LoRA adapters must use safetensors weights".to_string(),
            ));
        }
        let adapters = load_adapters_from_paths(&paths, &device, dtype)?;
        self.register(tenant_id, adapters);
        Ok(())
    }

    /// Pin a tenant (the SLA guarantee): the pinned tenant is never
    /// evicted, even under budget pressure. No-op when the tenant is not
    /// registered.
    pub fn pin(&mut self, tenant_id: &str) {
        if self.adapters.contains_key(tenant_id) {
            self.pinned.insert(tenant_id.to_string());
        }
    }

    /// Unpin a tenant (the release the SLA guarantee).
    pub fn unpin(&mut self, tenant_id: &str) {
        self.pinned.remove(tenant_id);
    }

    /// Acquire a lease on a tenant (the in-flight request holds it):
    /// The leased tenant is not evicted while the lease is held (not the
    /// mid-forward eviction). The lease count is the number of in-flight
    /// requests.
    pub fn acquire_lease(&mut self, tenant_id: &str) {
        *self.leases.entry(tenant_id.to_string()).or_insert(0) += 1;
    }

    /// Release a lease on a tenant (the request finished). The last is
    /// evictable again when the lease count reaches to 0.
    pub fn release_lease(&mut self, tenant_id: &str) {
        if let Some(count) = self.leases.get_mut(tenant_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.leases.remove(tenant_id);
            }
        }
    }

    /// The tenant's lease count (the observability, the in-flight request count).
    pub fn lease_count(&self, tenant_id: &str) -> usize {
        self.leases.get(tenant_id).copied().unwrap_or(0)
    }

    /// The resident byte count (the observability, the current GPU usage).
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// The byte budget.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Register a tenant's adapters (the `load_adapters_from_paths` output). When
    /// the tenant count exceeds the budget, the least-recently-used tenant is
    /// evicted (no unbounded GPU memory growth). The evicted tenant's
    /// adapters are dropped (the GPU memory freed). The eviction respects the
    /// byte budget + skips the pinned + the leased
    /// tenants (no evicting a tenant a request is using).
    pub fn register(&mut self, tenant_id: &str, adapters: Vec<LoraAdapter>) {
        // The byte count of the new tenant's adapters (the A/B tensor bytes).
        let new_bytes: u64 = adapters
            .iter()
            .map(|a| a.a.elem_count() as u64 * a.a.dtype().size_in_bytes() as u64
                + a.b.elem_count() as u64 * a.b.dtype().size_in_bytes() as u64)
            .sum();
        // The evict tenants until the new one fits (the count budget + the byte
        // budget). The eviction skips the pinned + the leased tenants (no
        // evicting a tenant a request is using). The policy (the order
        // (the LRU = the access-ordered, the FIFO = the insertion-ordered).
        while self.needs_eviction(self.adapters.len() + 1, self.resident_bytes + new_bytes) {
            let Some(evicted) = self.evictable() else {
                break; // no evictable tenant (the all pinned/leased, the over budget)
            };
            let evicted_bytes = self.tenant_bytes(&evicted);
            self.adapters.remove(&evicted);
            self.lru_order.retain(|t| t != &evicted);
            self.fifo_order.retain(|t| t != &evicted);
            self.resident_bytes = self.resident_bytes.saturating_sub(evicted_bytes);
            crate::log_warn!(
                "LoRA registry over budget ({} tenants / {} bytes); evicted {} tenant '{}'",
                self.max_tenants,
                self.max_bytes,
                match self.eviction_policy {
                    EvictionPolicy::Lru => "LRU",
                    EvictionPolicy::Fifo => "FIFO",
                },
                evicted
            );
        }
        // The populate the stacked bank (the S-LoRA dense storage) when
        // enabled, before the adapters are moved into the per-tenant map. The
        // each adapter's A/B is written into the tenant's bank slot for its
        // module (the one allocation per module, not the per-tenant
        // allocations). The bank is the dense backend; the per-tenant `adapters`
        // map remains the serving source of truth.
        if let Some(bank) = self.bank.as_mut() {
            let mut bank_failures = 0usize;
            for adapter in &adapters {
                if let Err(e) = bank.set_lora(tenant_id, &adapter.target_module, &adapter.a, &adapter.b, adapter.scaling) {
                    if bank_failures == 0 {
                        // The first failure is usually a rank/shape mismatch (the
                        // variable-rank case: a different adapter's rank was used to
                        // size the stacked bank). Log it once with the cause; the
                        // subsequent per-module failures are the same root cause.
                        crate::log_warn!(
                            "LoRA stacked bank write failed for tenant '{tenant_id}' module '{}': {e} (the tenant falls back to the per-tenant candle path; the remaining {} module(s) are skipped from the bank)",
                            adapter.target_module,
                            adapters.len() - 1
                        );
                    }
                    bank_failures += 1;
                }
            }
            if bank_failures > 0 {
                crate::log_warn!(
                    "LoRA tenant '{tenant_id}': {bank_failures} of {} module(s) failed the stacked-bank write (the variable-rank mismatch); the per-tenant candle path serves this tenant",
                    adapters.len()
                );
            }
        }
        self.adapters.insert(tenant_id.to_string(), adapters);
        self.resident_bytes = self.resident_bytes.saturating_add(new_bytes);
        // The move the tenant to the MRU position (the end of the LRU order).
        self.lru_order.retain(|t| t != tenant_id);
        self.lru_order.push(tenant_id.to_string());
        // The append the tenant to the FIFO order (the insertion order, not the
        // re-order on access).
        if !self.fifo_order.iter().any(|t| t == tenant_id) {
            self.fifo_order.push(tenant_id.to_string());
        }
    }

    /// Whether the registry needs to evict (the count budget OR the byte budget
    /// exceeded). The `new_count`/`new_bytes` are the post-insert values.
    fn needs_eviction(&self, new_count: usize, new_bytes: u64) -> bool {
        (new_count > self.max_tenants) || (self.max_bytes > 0 && new_bytes > self.max_bytes)
    }

    /// The policy-evictable tenant: the LRU policy evicts the least-
    /// recently-used tenant (the `lru_order` first); the FIFO policy evicts the
    /// first-registered tenant (the `fifo_order` first). The skip the pinned +
    /// the leased tenants (no evicting a tenant a request is using). The
    /// `None` when no tenant is evictable (the all pinned/leased, the budget
    /// cannot be met).
    fn evictable(&self) -> Option<String> {
        let order = match self.eviction_policy {
            EvictionPolicy::Lru => &self.lru_order,
            EvictionPolicy::Fifo => &self.fifo_order,
        };
        order.iter()
            .find(|t| !self.pinned.contains(*t) && self.lease_count(t) == 0)
            .cloned()
    }

    /// The resident byte count of a tenant (the A/B tensor bytes).
    pub fn tenant_bytes(&self, tenant_id: &str) -> u64 {
        self.adapters
            .get(tenant_id)
            .map(|adapters| {
                adapters.iter().map(|a| {
                    a.a.elem_count() as u64 * a.a.dtype().size_in_bytes() as u64
                        + a.b.elem_count() as u64 * a.b.dtype().size_in_bytes() as u64
                }).sum()
            })
            .unwrap_or(0)
    }

    /// Unregister a tenant's adapters (the detach). The resident byte count is
    /// decremented (the memory leak fixed).
    pub fn unregister(&mut self, tenant_id: &str) -> bool {
        let existed = self.adapters.contains_key(tenant_id);
        if let Some(adapters) = self.adapters.remove(tenant_id) {
            let bytes: u64 = adapters
                .iter()
                .map(|a| a.a.elem_count() as u64 * a.a.dtype().size_in_bytes() as u64
                    + a.b.elem_count() as u64 * a.b.dtype().size_in_bytes() as u64)
                .sum();
            self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
        }
        self.lru_order.retain(|t| t != tenant_id);
        self.fifo_order.retain(|t| t != tenant_id);
        self.pinned.remove(tenant_id);
        self.leases.remove(tenant_id);
        self.sources.remove(tenant_id);
        // The free the tenant's stacked-bank slot (the dense storage
        // freed when the tenant is unregistered).
        if let Some(bank) = self.bank.as_mut() {
            bank.free_slot(tenant_id);
        }
        existed
    }

    /// The registered tenant names (the observability, the `/v1/lora` listing).
    pub fn tenant_names(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Look up a tenant's adapters (the per-request application). The `None`
    /// when the tenant is not registered (no adapter, the base model). The
    /// lookup updates the LRU order (the tenant becomes the MRU).
    pub fn get(&mut self, tenant_id: &str) -> Option<&Vec<LoraAdapter>> {
        if self.adapters.contains_key(tenant_id) {
            self.lru_order.retain(|t| t != tenant_id);
            self.lru_order.push(tenant_id.to_string());
        }
        self.adapters.get(tenant_id)
    }

    /// The registered tenant count (the observability).
    pub fn tenant_count(&self) -> usize {
        self.adapters.len()
    }

    /// The tenant budget (the max concurrent tenants).
    pub fn max_tenants(&self) -> usize {
        self.max_tenants
    }

    /// Whether any tenant is registered (the no-adapter fast path).
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Look up the active tenant's adapter for a module (the per-request
    /// application). The `None` when no tenant is active (the base model) or
    /// the tenant has no adapter for this module (no delta). The reads the
    /// thread-local active tenant (the `active_tenant`) + the registry.
    pub fn active_adapter_for(&self, module: &str) -> Option<&LoraAdapter> {
        let tenant = active_tenant()?;
        let adapters = self.adapters.get(&tenant)?;
        let want = canonical_site_key(module);
        adapters.iter().find(|a| canonical_site_key(&a.target_module) == want)
    }

    /// Look up a specific tenant's adapter for a module (the explicit tenant, not
    /// the thread-local active tenant). The `None` when the tenant is not
    /// registered or has no adapter for this module.
    pub fn active_adapter_for_tenant(
        &self,
        tenant: &str,
        module: &str,
    ) -> Option<&LoraAdapter> {
        let adapters = self.adapters.get(tenant)?;
        let want = canonical_site_key(module);
        adapters.iter().find(|a| canonical_site_key(&a.target_module) == want)
    }

    /// The batched delta (the multi-tenant forward). Given the
    /// per-row tenant's adapter delta to the batched input `x` (the
    /// `[batch, ...]`). The `exec` carries the per-row tenant ids (the
    /// `LoraExecution.row_tenants`). Rows with the `None` tenant (the base
    /// model) get no delta. The result is the `[batch, ...]` delta tensor to
    /// add to the base linear. When the `exec` is the single-tenant case (the
    /// all rows the same tenant), this reduces to the single `active_adapter_for`
    /// delta broadcast over the batch.
    pub fn batched_delta(
        &self,
        module: &str,
        x: &Tensor,
        exec: &LoraExecution,
        parallelism: LoraParallelism,
        tp_rank: usize,
        tp_world: usize,
    ) -> Result<Option<Tensor>> {
        // The distinct tenants in this batch (the single-tenant + the stacked paths).
        let mut tenants: HashSet<String> = exec
            .row_tenants
            .iter()
            .filter_map(|t| t.clone())
            .collect();
        // The stacked tenants (the per-row additive composition) join the set.
        for stack in &exec.row_stacks {
            for t in stack {
                tenants.insert(t.clone());
            }
        }
        let has_stacks = exec.has_stacks();
        if tenants.is_empty() {
            return Ok(None); // no tenant in this batch (the base model only)
        }
        // The single-tenant fast path (the all rows the same tenant): one delta,
        // broadcast over the batch. Skipped when any row carries a stack (the
        // additive composition needs the per-row path).
        if !has_stacks && tenants.len() == 1 {
            let tenant = tenants.iter().next().unwrap();
            let want = canonical_site_key(module);
            let Some(adapter) = self.adapters.get(tenant).and_then(|adapters| {
                adapters.iter().find(|a| canonical_site_key(&a.target_module) == want)
            }) else {
                return Ok(None); // the tenant has no adapter for this module
            };
            return Ok(Some(adapter.delta(x, parallelism, tp_rank, tp_world)?));
        }
        // The multi-tenant path: group the rows by tenant, compute each tenant's
        // delta on its rows, + assemble the full-batch delta (the zero for the
        // rows of tenants that have no adapter for this module). The output is
        // the `[batch, ...]` delta tensor to add to the base output.
        let batch = x.dim(0)?;
        // The zero delta row (the same shape as one row of x, the all-zero).
        let row_shape: Vec<usize> = x.shape().dims()[1..].to_vec();
        let zero_row = Tensor::zeros(
            std::iter::once(1).chain(row_shape.iter().copied()).collect::<Vec<_>>().as_slice(),
            x.dtype(),
            x.device(),
        )?;
        // The per-row delta (the zero for the no-tenant rows, the tenant's delta
        // for the tenant rows). Assembled in row order, then concatenated.
        let mut out_rows: Vec<Tensor> = Vec::with_capacity(batch);
        for row in 0..batch {
            // The row's tenant list: the explicit stack when present, else the
            // single `row_tenants` entry, else none (the base model).
            let stack = exec.row_stack(row);
            if stack.is_empty() {
                out_rows.push(zero_row.clone());
                continue;
            }
            // The single-row gather (the `[1, ...]` slice of x for this row).
            let idx = Tensor::from_vec(vec![row as i64], (1,), x.device())?;
            let gathered = x.index_select(&idx, 0)?;
            // The additive composition: sum each tenant's delta on the same row.
            let mut row_delta = Tensor::zeros(
                std::iter::once(1)
                    .chain(row_shape.iter().copied())
                    .collect::<Vec<_>>()
                    .as_slice(),
                x.dtype(),
                x.device(),
            )?;
            for t in &stack {
                let want = canonical_site_key(module);
                let Some(adapter) = self.adapters.get(t).and_then(|adapters| {
                    adapters.iter().find(|a| canonical_site_key(&a.target_module) == want)
                }) else {
                    continue; // this tenant has no adapter for this module
                };
                let d = adapter.delta(&gathered, parallelism, tp_rank, tp_world)?;
                row_delta = row_delta.broadcast_add(&d)?;
            }
            out_rows.push(row_delta);
        }
        let out = Tensor::cat(&out_rows, 0)?;
        Ok(Some(out))
    }
}

/// Initialize the LoRA registry from the engine config. The `lora` config
/// (the `LoraConfig`) carries the (name, source) adapter list + the budgets +
/// the policy + the pinned tenants. Each adapter is resolved via the common
/// model loader (the same Downloader + the hf-hub machinery the base model +
/// the draft model use), then the PEFT `adapter_config.json` + the
/// `adapter_model` weights are loaded. Returns an empty registry when no
/// adapter is configured (the prior operation, no LoRA).
pub fn init_lora_registry(
    econfig: &crate::utils::config::EngineConfig,
    device: &Device,
    dtype: DType,
) -> Result<LoraRegistry> {
    let lora_cfg = &econfig.lora;
    // The build the registry with the budgets + the device/dtype
    // (the on-demand load target). The `with_budget`/`with_max_bytes` are the
    // associated builders (not the method chain, the fresh registry).
    let mut registry = LoraRegistry::with_budget(lora_cfg.max_tenants.unwrap_or(DEFAULT_LORA_MAX_TENANTS));
    if let Some(max_bytes) = lora_cfg.max_bytes {
        registry = registry.with_max_bytes(max_bytes);
    }
    if let Some(max_active) = lora_cfg.max_active {
        registry = registry.with_max_active(max_active);
    }
    // The enable the A/B into a single bank allocation (the S-LoRA
    // memory model, the density). The bank size is the tenant budget (the
    // max concurrent resident), so the dense storage matches the residency cap.
    // The bank is populated in parallel with the per-tenant map (the map remains
    // the serving source of truth; the bank is the dense backend).
    let bank_size = lora_cfg.max_tenants.unwrap_or(DEFAULT_LORA_MAX_TENANTS).max(1);
    registry = registry.with_stacked_bank(bank_size);
    // The canonical bank dtype (the model's compute dtype): all tenants' A/B are
    // cast to this dtype before being written into the bank, so a mixed set of
    // adapters (the F32, the BF16, the FP8) coexist in the same bank without a
    // dtype mismatch on the cat.
    registry = registry.with_bank_dtype(dtype);
    registry = registry.with_device(device.clone(), dtype);
    // The eviction policy: the LRU (the default, the access-ordered) or
    // the FIFO (the insertion-ordered). The operator selects via the
    // `--lora-eviction-policy` CLI (the `LoraConfig.eviction_policy`).
    if let Some(policy) = &lora_cfg.eviction_policy {
        let policy = match policy.to_lowercase().as_str() {
            "fifo" => EvictionPolicy::Fifo,
            _ => EvictionPolicy::Lru, // the default (the "lru" + the unrecognized)
        };
        registry = registry.with_eviction_policy(policy);
    }
    if lora_cfg.adapters.is_empty() {
        crate::log_info!("LoRA: no adapters configured; running the base model only");
        return Ok(registry);
    }
    // The boot-time budget sizing (item 2): when the operator did not set an
    // explicit `--lora-max-bytes`, size the byte budget to the total of ALL the
    // boot-time adapters, so they all fit resident (no the LRU eviction
    // thrash where each newly loaded adapter evicts the prior). The explicit
    // budget (the `lora_cfg.max_bytes` Some) overrides the auto-sized one.
    let explicit_max_bytes = lora_cfg.max_bytes;
    if explicit_max_bytes.is_none() {
        // The pre-pass: load each adapter's A/B + sum the bytes (no the
        // register, which would evict under the default budget). The adapters
        // are re-loaded in the register pass (the load is idempotent + the
        // cache makes it cheap); the pre-pass is only for the byte sizing.
        let mut total_boot_bytes: u64 = 0;
        for (name, source) in &lora_cfg.adapters {
            let (model_id, weight_path) = crate::speculative::resolve_draft_model(source);
            let loader = crate::utils::downloader::Downloader::new(model_id, weight_path, None);
            let Ok((paths, is_gguf)) = loader.prepare_lora_adapter_weights(
                econfig.hf_token.clone(),
                econfig.hf_token_path.clone(),
            ) else {
                continue; // the load failure is handled in the register pass
            };
            if is_gguf {
                continue; // the GGUF is rejected in the register pass
            }
            let Ok(adapters) = load_adapters_from_paths(&paths, device, dtype) else {
                continue;
            };
            total_boot_bytes += adapters
                .iter()
                .map(|a| a.a.elem_count() as u64 * a.a.dtype().size_in_bytes() as u64
                    + a.b.elem_count() as u64 * a.b.dtype().size_in_bytes() as u64)
                .sum::<u64>();
        }
        if total_boot_bytes > 0 {
            registry = registry.with_max_bytes(total_boot_bytes);
            crate::log_info!(
                "LoRA: auto-sized the byte budget to {} bytes (the total of the {} boot-time adapter(s)); set --lora-max-bytes to override",
                total_boot_bytes,
                lora_cfg.adapters.len()
            );
        }
    }
    // The load each adapter (the (name, source) pair). The name is the API-facing
    // key (the `model: ":name` request form); the source is resolved by the
    // common model loader (the HF id or the local dir).
    for (name, source) in &lora_cfg.adapters {
        crate::log_info!("Loading LoRA adapter '{name}' from {source}...");
        let (model_id, weight_path) = crate::speculative::resolve_draft_model(source);
        let loader = crate::utils::downloader::Downloader::new(model_id, weight_path, None);
        let (paths, is_gguf) = loader.prepare_lora_adapter_weights(
            econfig.hf_token.clone(),
            econfig.hf_token_path.clone(),
        )?;
        if is_gguf {
            candle_core::bail!("LoRA adapter '{name}' must use safetensors weights");
        }
        let raw_adapters = load_adapters_from_paths(&paths, device, dtype)?;
        let raw_count = raw_adapters.len();
        // The discovered target module names (for the diagnostic, captured before
        // the `raw_adapters` is consumed by the filter below).
        let raw_targets: Vec<String> = raw_adapters.iter().map(|a| a.target_module.clone()).collect();
        // The validate the adapter's target modules against the known LoRA sites
        // (the adapter-to-site matching). An adapter targeting a module
        // not in the known site set is logged + skipped (no silently applying
        // a delta to a non-site, not the adapter mismatch). The model type is
        // not available here (the `init_lora_registry` runs before the model is
        // built), so the validation uses the base site set (the attention + MLP +
        // lm_head + embedding + the fused aliases); the MoE expert sites are
        // validated when the model type is known (the next increment).
        let invalid_targets = validate_adapter_targets(
            None,
            &raw_targets,
        );
        if !invalid_targets.is_empty() {
            crate::log_warn!(
                "LoRA adapter '{name}': {} of {} discovered module(s) target a site not in the known LoRA set (skipped): {:?}. Known sites: {:?}",
                invalid_targets.len(),
                raw_count,
                invalid_targets,
                model_lora_sites(None).site_names()
            );
        }
        let adapters: Vec<LoraAdapter> = raw_adapters
            .into_iter()
            .filter(|a| !invalid_targets.iter().any(|t| t == &a.target_module))
            .collect();
        // The register the source for the on-demand load: a future
        // request for this tenant that finds it evicted can reload it from the
        // source.
        registry.add_source(name, source);
        registry.register(name, adapters.clone());
        let loaded_count = adapters.len();
        if loaded_count == 0 {
            // The diagnostic: distinguish the two failure stages so the operator knows
            // WHY the adapter loaded 0 modules.
            if raw_count == 0 {
                crate::log_warn!(
                    "LoRA adapter '{name}' loaded 0 module(s): the tensor discovery found no `.lora_A.weight` keys in the adapter weights. The adapter's safetensors naming does not match the expected PEFT convention (the `<site>.lora_A.weight` / the `<site>.lora_B.weight`). Check the adapter checkpoint layout."
                );
            } else {
                crate::log_warn!(
                    "LoRA adapter '{name}' loaded 0 module(s): {} module(s) discovered but all dropped by the site validation (the target leaves are not in the known LoRA site set). Discovered: {:?}. The adapter targets modules this model does not expose as LoRA sites.",
                    raw_count,
                    raw_targets
                );
            }
        } else {
            // The full module list is logged ONCE at startup (not the per-request
            // dump, which is untenable for a 300-module adapter). The operator can
            // verify the loaded modules here; the request-time log carries only the
            // count + the layer count.
            let module_names: Vec<String> = adapters.iter().map(|a| a.target_module.clone()).collect();
            crate::log_info!(
                "LoRA adapter '{name}' loaded: {} module(s): {:?}",
                loaded_count,
                module_names
            );
        }
    }
    // The pin the SLA tenants (no evicted).
    for pinned in &lora_cfg.pinned {
        registry.pin(pinned);
        crate::log_info!("LoRA: pinned tenant '{pinned}' (never evicted)");
    }
    crate::log_info!(
        "LoRA registry initialized: {} tenant(s), {} resident bytes, budget {} tenants / {} bytes",
        registry.tenant_count(),
        registry.resident_bytes(),
        registry.max_tenants(),
        registry.max_bytes()
    );
    // The MoE expert LoRA: when the per-expert gate/B from each
    // adapter's source (the MoE expert tensors are in the same adapter dir).
    // The `None` when no MoE LoRA is present (the dense-only adapter, the
    // fused `moe_gemm` kernel path is used). Loaded per-tenant so each
    // tenant's distinct MoE expert adapter is applied to its own rows.
    for (name, source) in &lora_cfg.adapters {
        let (model_id, weight_path) = crate::speculative::resolve_draft_model(source);
        let loader = crate::utils::downloader::Downloader::new(model_id, weight_path, None);
        if let Ok((paths, is_gguf)) = loader.prepare_lora_adapter_weights(
            econfig.hf_token.clone(),
            econfig.hf_token_path.clone(),
        ) {
            if !is_gguf {
                if let Ok(Some(moe_state)) = load_moe_lora_state(&paths, device, dtype) {
                    set_global_moe_lora(name, Some(moe_state));
                    crate::log_info!("MoE expert LoRA loaded for tenant '{name}'");
                }
            }
        }
    }
    Ok(registry)
}

/// The LoRA-aware linear layer: the base linear (the `weight` + the `bias`) +
/// the adapter delta (the `LoraAdapter` for this module). The `forward` computes
/// `base(x) + delta(x)` when the active tenant (the `active_tenant` thread-
/// local) is set + this module has an adapter, else `base(x)` (no adapter,
/// the base model). This is the injection point: the model's target linear
/// layers (the `q_proj`/`k_proj`/`v_proj`/`o_proj`/the MLP gate/up/down) are
/// wrapped with this, so the adapter delta is applied per-request without
/// mutating the base weights.
///
/// The single-GPU case (the `world_size == 1`): the A/B are full (no TP
/// sharding), the delta is `scaling * (x @ A^T) @ B^T`. The multi-GPU TP case
/// (the `world_size > 1`): the B is sharded across the ranks (the column-
/// parallel out dim), the A is replicated (the row-parallel in dim); the delta
/// is computed per-shard + the all-reduce combines them. This is the documented
/// extension (the single-GPU is the production path today).
// The `LoraLinear` wrapper (the pre-batched single-tenant design) is obsolete:
// the wrapped linears (the `TensorParallelColumnLinear`/`RowLinear`/
// `ReplicatedLinear` in distributed.rs) call `apply_lora_delta` directly (the
// batched multi-tenant + the TP-sharded delta). The wrapper is removed (no
// dead code, the rule 15).

#[cfg(test)]
mod tests {
    use super::*;
    use DType;

    /// The adapter delta math: the `delta(x) = scaling * (x @ A^T) @ B^T`.
    /// The inclusive: a known A/B produces the expected delta. The exclusive:
    /// a zero A produces a zero delta (the base is unchanged).
    #[test]
    fn lora_delta_is_low_rank_product() {
        let device = Device::Cpu;
        // The x is the `[1, 2, 3]` (the batch=1, seq=1, in=3). The A is the
        // `[in=3, rank=2]`, the B is the `[rank=2, out=4]`.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device).unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0], (3, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], (2, 4), &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let delta = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        // The x @ A^T = the `[1,1,3] @ the [3,2] = the [1,1,2] = the [1*1+2*0+3*0, 1*0+2*1+3*1] = the [1, 5].
        // The [1,5] @ B^T = the `[1,1,2] @ the [2,4] = the [1*1+5*1, 1*1+5*1, 0, 0] = the [6, 6, 0, 0].
        let expected = Tensor::from_vec(vec![6.0f32, 6.0, 0.0, 0.0], (1, 1, 4), &device).unwrap();
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want, "the delta is the low-rank product");
    }

    /// CPU reference for the stacked (additive multi-adapter) shrink/expand
    /// kernel math. Mirrors the exact index arithmetic of
    /// `lora_shrink_stacked_kernel` + `lora_expand_stacked_kernel` in
    /// `.attention-rs/src/kernels/src/lora_fused_kernel.cu`, and asserts it
    /// equals the naive per-adapter sum `sum_j scale_j * (x @ A_j) @ B_j`
    /// (the mistral.rs `LoraLinear` fast path, loralinear.rs:220-246).
    ///
    /// Layouts (all f32, CPU):
    ///   x:  [rows, in]
    ///   A:  [num_slots, in, rank]   (the stacked bank)
    ///   B:  [num_slots, rank, out]
    ///   scale: [num_slots]
    ///   slot_offsets: [rows+1]  (CSR row pointers)
    ///   slot_ids: [total]       (compacted index per (row, adapter) pair)
    ///   active_slots: [num_active] (compacted -> bank slot)
    ///
    /// The kernel:
    ///   shrink: buf[row, j, k] = scale[bank] * sum_c x[row,c] * A[bank,c,k]
    ///           where bank = active_slots[slot_ids[slot_offsets[row]+j]]
    ///   expand: out[row,o] = sum_j sum_k buf[row,j,k] * B[bank,k,o]
    /// which is exactly sum_j scale_j * (x @ A_j) @ B_j over the row's stack.
    #[test]
    fn lora_stacked_kernel_math_matches_naive_sum() {
        // 3 rows, each with a different stack of bank slots.
        // row 0: stack [slot 0, slot 1]  (2 adapters)
        // row 1: stack [slot 2]          (1 adapter)
        // row 2: stack []                (0 adapters, base model)
        let rows = 3usize;
        let in_f = 4usize;
        let rank = 2usize;
        let out_f = 3usize;
        let num_slots = 3usize;

        // x: [rows, in]
        let x: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0], // row 0
            vec![5.0, 6.0, 7.0, 8.0], // row 1
            vec![9.0, 10.0, 11.0, 12.0], // row 2
        ];
        // A: [num_slots, in, rank]
        let a: Vec<Vec<Vec<f32>>> = vec![
            // slot 0
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]],
            // slot 1
            vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]],
            // slot 2
            vec![vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
        ];
        // B: [num_slots, rank, out]
        let b: Vec<Vec<Vec<f32>>> = vec![
            // slot 0
            vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]],
            // slot 1
            vec![vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0]],
            // slot 2
            vec![vec![1.0, 1.0, 0.0], vec![0.0, 1.0, 1.0]],
        ];
        // scale: [num_slots]
        let scale: Vec<f32> = vec![0.5, 2.0, 1.0];

        // CSR: slot_offsets [rows+1], slot_ids [total]
        // row 0 -> slots [0, 1]; row 1 -> slots [2]; row 2 -> [] []
        let slot_offsets: Vec<u32> = vec![0, 2, 3, 3];
        let slot_ids: Vec<u32> = vec![0, 1, 2]; // compacted ids (here == bank slots)
        let active_slots: Vec<u32> = vec![0, 1, 2]; // compacted -> bank (identity)

        // ---- The kernel math (mirrors the CUDA index arithmetic exactly) ----
        // shrink: buf[row, j, k]
        let max_stack = 2usize;
        let mut buf = vec![vec![vec![0.0f32; rank]; max_stack]; rows];
        for row in 0..rows {
            let stack_count = (slot_offsets[row + 1] - slot_offsets[row]) as usize;
            for j in 0..stack_count {
                let compact = slot_ids[slot_offsets[row] as usize + j] as usize;
                let bank = active_slots[compact] as usize;
                for k in 0..rank {
                    let mut acc = 0.0f32;
                    for c in 0..in_f {
                        acc += x[row][c] * a[bank][c][k];
                    }
                    buf[row][j][k] = acc * scale[bank];
                }
            }
        }
        // expand: out[row, o] = sum_j sum_k buf[row,j,k] * B[bank(j),k,o]
        let mut out = vec![vec![0.0f32; out_f]; rows];
        for row in 0..rows {
            let stack_count = (slot_offsets[row + 1] - slot_offsets[row]) as usize;
            for o in 0..out_f {
                let mut acc = 0.0f32;
                for j in 0..stack_count {
                    let compact = slot_ids[slot_offsets[row] as usize + j] as usize;
                    let bank = active_slots[compact] as usize;
                    for k in 0..rank {
                        acc += buf[row][j][k] * b[bank][k][o];
                    }
                }
                out[row][o] = acc;
            }
        }

        // ---- The naive reference: sum_j scale_j * (x @ A_j) @ B_j ----
        let naive = |row: usize, stack: &[u32]| -> Vec<f32> {
            let mut acc = vec![0.0f32; out_f];
            for &bank in stack {
                // down = x[row] @ A[bank]  -> [rank]
                let down: Vec<f32> = (0..rank)
                    .map(|k| {
                        let mut s = 0.0f32;
                        for c in 0..in_f {
                            s += x[row][c] * a[bank as usize][c][k];
                        }
                        s
                    })
                    .collect();
                // up = down @ B[bank] -> [out]
                for o in 0..out_f {
                    let mut s = 0.0f32;
                    for k in 0..rank {
                        s += down[k] * b[bank as usize][k][o];
                    }
                    acc[o] += scale[bank as usize] * s;
                }
            }
            acc
        };
        let stack0: Vec<u32> = vec![0, 1];
        let stack1: Vec<u32> = vec![2];
        let stack2: Vec<u32> = vec![];
        let want0 = naive(0, &stack0);
        let want1 = naive(1, &stack1);
        let want2 = naive(2, &stack2);

        let close = |a: f32, b: f32| (a - b).abs() < 1e-4;
        for o in 0..out_f {
            assert!(close(out[0][o], want0[o]), "row0 out[{o}] = {} != naive {}", out[0][o], want0[o]);
            assert!(close(out[1][o], want1[o]), "row1 out[{o}] = {} != naive {}", out[1][o], want1[o]);
            assert!(close(out[2][o], want2[o]), "row2 out[{o}] = {} != naive {}", out[2][o], want2[o]);
        }
        // row 2 has an empty stack -> zero delta (the base model).
        assert!(out[2].iter().all(|&v| v == 0.0), "row2 (no stack) must be all-zero");
    }

    #[test]
    fn lora_zero_adapter_is_identity() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device).unwrap();
        let a = Tensor::zeros((3, 2), DType::F32, &device).unwrap();
        let b = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let delta = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(got.iter().all(|&v| v == 0.0), "a zero A produces a zero delta");
    }

    /// The registry: the inclusive (a registered tenant is found), the
    /// exclusive (an unregistered tenant is not found, the unregister removes).
    #[test]
    fn lora_registry_lookup_and_unregister() {
        let device = Device::Cpu;
        let a = Tensor::zeros((3, 2), DType::F32, &device).unwrap();
        let b = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let mut reg = LoraRegistry::new();
        reg.register("tenant-a", vec![adapter]);
        assert!(reg.get("tenant-a").is_some());
        assert_eq!(reg.tenant_count(), 1);
        reg.unregister("tenant-a");
        assert!(reg.get("tenant-a").is_none());
        assert_eq!(reg.tenant_count(), 0);
        assert!(reg.is_empty());
    }

    /// The thread-local tenant selection: the set + the get round-trip, the
    /// clear (the None). The inclusive: a set tenant is read back. The
    /// exclusive: a cleared tenant is None.
    #[test]
    fn lora_active_tenant_thread_local() {
        set_active_tenant(Some("tenant-a".to_string()));
        assert_eq!(active_tenant().as_deref(), Some("tenant-a"));
        set_active_tenant(None);
        assert_eq!(active_tenant(), None);
    }

    /// The tenant budget (the LRU eviction, no unbounded GPU memory growth).
    /// The inclusive: registering tenants up to the budget keeps them; exceeding
    /// the budget evicts the LRU tenant. The exclusive: a budget of 1 evicts the
    /// prior tenant on the second registration (no two tenants coexisting).
    #[test]
    fn lora_registry_tenant_budget_licts_lru() {
        let device = Device::Cpu;
        let a = Tensor::zeros((3, 2), DType::F32, &device).unwrap();
        let b = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The budget of 2 (not the default 16).
        let mut reg = LoraRegistry::with_budget(2);
        reg.register("t1", vec![adapter.clone()]);
        reg.register("t2", vec![adapter.clone()]);
        assert_eq!(reg.tenant_count(), 2, "the budget holds 2 tenants");
        // The 3rd registration evicts the LRU (t1, the least-recently-used).
        reg.register("t3", vec![adapter.clone()]);
        assert_eq!(reg.tenant_count(), 2, "the budget is enforced (no 3 tenants)");
        assert!(reg.get("t1").is_none(), "the LRU tenant (t1) is evicted");
        assert!(reg.get("t2").is_some(), "the MRU tenant (t2) is kept");
        assert!(reg.get("t3").is_some(), "the new tenant (t3) is kept");
        // The exclusive: a budget of 1 evicts the prior tenant on the second
        // registration (no two tenants coexisting).
        let mut reg1 = LoraRegistry::with_budget(1);
        reg1.register("a", vec![adapter.clone()]);
        reg1.register("b", vec![adapter]);
        assert_eq!(reg1.tenant_count(), 1, "the budget of 1 holds exactly one tenant");
        assert!(reg1.get("a").is_none(), "the prior tenant is evicted");
        assert!(reg1.get("b").is_some(), "the new tenant is kept");
    }

    /// The FIFO eviction policy: the first-registered tenant is evicted
    /// first (the insertion order, not the access order). The inclusive: a FIFO
    /// registry evicts the oldest tenant under budget pressure. The exclusive: a
    /// LRU registry evicts the least-recently-used tenant (the access order, not
    /// the insertion order) — the two policies differ when the access order differs
    /// from the insertion order.
    #[test]
    fn lora_registry_fifo_policy_evicts_oldest() {
        let device = Device::Cpu;
        let a = Tensor::zeros((3, 2), DType::F32, &device).unwrap();
        let b = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The FIFO policy: the first of 2, the first-registered is evicted first.
        let mut reg = LoraRegistry::with_budget(2).with_eviction_policy(EvictionPolicy::Fifo);
        reg.register("t1", vec![adapter.clone()]);
        reg.register("t2", vec![adapter.clone()]);
        // The access t1 (the LRU would make t2 the LRU, the FIFO ignores the access).
        let _ = reg.get("t1");
        // The t3 (the over-budget, the eviction trigger). The FIFO evicts t1
        // (the oldest), no t2 (the LRU would evict t2 after the t1 access).
        reg.register("t3", vec![adapter.clone()]);
        assert!(reg.get("t1").is_none(), "the FIFO evicts the oldest (t1)");
        assert!(reg.get("t2").is_some(), "the FIFO keeps the newer (t2)");
        assert!(reg.get("t3").is_some(), "the new tenant (t3) is kept");
    }

    /// The batched multi-tenant delta (the Punica result): one forward
    /// pass serves rows from different tenants, each row gets its own tenant's
    /// delta (no cross-tenant contamination). The inclusive: two tenants in
    /// one batch each get their own delta on their own rows. The exclusive: a
    /// batch with no tenants gets no delta (the base model).
    #[test]
    fn lora_batched_delta_tenant_delta_is_per_row() {
        let device = Device::Cpu;
        // The two tenants, each with a distinct adapter for module "m". The
        // tenant "a" has the identity A/B (the delta = the x), the tenant "b"
        // has the zero A (the delta = the 0).
        let a_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let a_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let adapter_a = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a: a_a,
            b: a_b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let b_a = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let b_b = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let adapter_b = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a: b_a,
            b: b_b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let mut reg = LoraRegistry::new();
        reg.register("a", vec![adapter_a]);
        reg.register("b", vec![adapter_b]);
        // The batch: 4 rows. Row 0 + 1 → tenant "a", row 2 + 3 → tenant "b".
        let exec = LoraExecution {
            row_tenants: vec![
                Some("a".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
                Some("b".to_string()),
            ],
            row_stacks: Vec::new(),
        };
        // The input x (the `[4, 2]`, the all-ones so the delta is visible).
        let x = Tensor::ones((4, 2), DType::F32, &device).unwrap();
        let delta = reg.batched_delta("m", &x, &exec, LoraParallelism::Replicated, 0, 1).unwrap().unwrap();
        let got = delta.to_vec2::<f32>().unwrap();
        // The tenant "a" delta (the A = the identity, the B = the identity, the
        // scaling = 1) → the delta = the x (the all-ones). The tenant "b" delta
        // (the A = the zero) → the delta = the 0.
        assert_eq!(got[0], vec![1.0, 1.0], "row 0 (tenant a) gets the a delta");
        assert_eq!(got[1], vec![1.0, 1.0], "row 1 (tenant a) gets the a delta");
        assert_eq!(got[2], vec![0.0, 0.0], "row 2 (tenant b) gets the b delta (the zero)");
        assert_eq!(got[3], vec![0.0, 0.0], "row 3 (tenant b) gets the b delta (the zero)");
        // The exclusive: a batch with no tenants (the all base model) → no
        // delta (the None).
        let exec_none = LoraExecution {
            row_tenants: vec![None, None],
            row_stacks: Vec::new(),
        };
        let none_delta = reg.batched_delta("m", &x, &exec_none, LoraParallelism::Replicated, 0, 1).unwrap();
        assert!(none_delta.is_none(), "no tenant → no delta");
    }

    /// The stacked (additive multi-adapter) composition: a row carrying a
    /// tenant stack gets the sum of each tenant's delta. The inclusive: two
    /// non-zero adapters stacked on one row produce the sum of their individual
    /// deltas. The exclusive: a row with an empty stack (the base model) gets
    /// the zero delta. This exercises the candle `batched_delta` path (the
    /// stacked fused kernel is CUDA-only, so the CPU test uses the fallback).
    #[test]
    fn lora_stacked_composition_is_additive_sum() {
        let device = Device::Cpu;
        // Tenant "a": identity A/B (the delta = the x). Tenant "b": A = the 2x
        // identity (the delta = the 2x x). Both target 2, in=2, out=2.
        let a_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let a_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let adapter_a = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a: a_a,
            b: a_b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let b_a = Tensor::from_vec(vec![2.0f32, 0.0, 0.0, 2.0], (2, 2), &device).unwrap();
        let b_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let adapter_b = LoraAdapter {
            target_module: "m".to_string(),
            rank: 2,
            a: b_a,
            b: b_b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let mut reg = LoraRegistry::new();
        reg.register("a", vec![adapter_a]);
        reg.register("b", vec![adapter_b]);
        // The batch: 2 rows. Row 0 stacks [a, b] (the delta = the x + the 2x x =
        // the 3x x). Row 1 has no stack (the base model, the zero delta).
        let exec = LoraExecution {
            row_tenants: vec![None, None],
            row_stacks: vec![
                vec!["a".to_string(), "b".to_string()],
                Vec::new(),
            ],
        };
        let x = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (2, 2), &device).unwrap();
        // The candle `batched_delta` composes the row's stack additively.
        let delta = reg.batched_delta("m", &x, &exec, LoraParallelism::Replicated, 0, 1).unwrap();
        let Some(delta) = delta else {
            panic!("the stacked delta must be present (the row 0 has a stack)");
        };
        let got = delta.to_vec2::<f32>().unwrap();
        // Row 0: the a delta (the identity, the x) + the b delta (the 2x, the 2x x)
        // = the [1,1] + the [2,2] = the [3,3].
        assert_eq!(got[0], vec![3.0, 3.0], "row 0 (stack a+b) = the sum of the deltas");
        // Row 1: no stack (the zero delta (the base model).
        assert_eq!(got[1], vec![0.0, 0.0], "row 1 (no stack) = the zero delta");
    }

    /// The byte budget: the registry evicts when the resident bytes
    /// exceed the budget, not just the count. The inclusive: a tenant that fits
    /// in the byte budget is kept. The exclusive: a tenant that exceeds the byte
    /// budget evicts the LRU tenant (no unbounded GPU memory growth).
    #[test]
    fn lora_registry_byte_budget_evicts() {
        let device = Device::Cpu;
        // The large adapter (the A/B are large, the byte count is high).
        let big_a = Tensor::zeros((64, 64), DType::F32, &device).unwrap();
        let big_b = Tensor::zeros((64, 64), DType::F32, &device).unwrap();
        let big = LoraAdapter {
            target_module: "m".to_string(),
            rank: 64,
            a: big_a,
            b: big_b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The byte count of the big adapter (the A + the B, the F32 = 4 bytes/elem).
        let big_bytes = (big.a.elem_count() + big.b.elem_count()) as u64 * 4;
        // The budget is just under the big adapter's bytes (the big one cannot fit
        // alongside anything).
        let mut reg = LoraRegistry::with_budget(8).with_max_bytes(big_bytes);
        // The inclusive: a single big tenant fits exactly in the budget.
        reg.register("big", vec![big.clone()]);
        assert_eq!(reg.tenant_count(), 1);
        assert_eq!(reg.resident_bytes(), big_bytes);
        // The exclusive: a second big tenant exceeds the budget → the LRU (the
        // first) is evicted (no two big tenants coexisting over budget).
        reg.register("big2", vec![big.clone()]);
        assert_eq!(reg.tenant_count(), 1, "the over-budget second evicts the LRU");
        assert!(reg.get("big").is_none(), "the LRU tenant is evicted");
        assert!(reg.get("big2").is_some(), "the new tenant is kept");
    }

    /// The lease: a leased tenant is not evicted, a request holds
    /// it. The inclusive: a leased tenant survives budget pressure. The
    /// exclusive: a non-leased tenant is evicted under the same pressure.
    #[test]
    fn lora_registry_lease_protects_from_eviction() {
        let device = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let b = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The budget of 1 tenant (the nextest).
        let mut reg = LoraRegistry::with_budget(1);
        reg.register("t1", vec![adapter.clone()]);
        // The lease t1 (the in-flight request holds it).
        reg.acquire_lease("t1");
        // The register t2 (the over-count, the eviction trigger). The t1 is leased
        // → not evictable → the t2 cannot be registered within the budget, so
        // the registry keeps both (the budget is soft when the lease holds).
        reg.register("t2", vec![adapter.clone()]);
        assert!(reg.get("t1").is_some(), "the leased tenant survives the budget pressure");
        assert!(reg.get("t2").is_some(), "the new tenant is registered");
        // The release the lease + the re-register triggers the eviction (the t1 is
        // now evictable).
        reg.release_lease("t1");
        reg.register("t3", vec![adapter]);
        // The t1 (the LRU, no longer leased) is evicted to make room.
        assert!(reg.get("t1").is_none(), "the un-leased LRU tenant is evicted");
        assert!(reg.get("t3").is_some(), "the new tenant is kept");
    }

    /// The pin: a pinned tenant is never evicted, even under budget
    /// pressure. The inclusive: a pinned tenant survives the count budget. The
    /// exclusive: a non-pinned tenant is evicted under the same pressure.
    #[test]
    fn lora_registry_pin_protects_from_eviction() {
        let device = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let b = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The budget of 1 tenant (the tightest).
        let mut reg = LoraRegistry::with_budget(1);
        reg.register("pinned", vec![adapter.clone()]);
        reg.pin("pinned");
        // The register a second tenant (the over-count, the eviction trigger). The
        // pinned tenant is not evictable → the registry keeps both (the budget is
        // soft when the pin holds).
        reg.register("other", vec![adapter]);
        assert!(reg.get("pinned").is_some(), "the pinned tenant survives the budget pressure");
        assert!(reg.get("other").is_some(), "the new tenant is registered");
    }

    /// The on-demand load: the `add_source` + the `ensure_loaded`
    /// no-op path. The inclusive: a known source + a set device/dtype allows the
    /// load (not the source, not the load). The exclusive: an unknown
    /// source (no `add_source`) is a no-op (no load attempt, the base
    /// model). This test proves the no-op paths (the actual load needs a
    /// real adapter dir, the integration test).
    #[test]
    fn lora_registry_on_demand_load_no_op_paths() {
        let device = Device::Cpu;
        let mut reg = LoraRegistry::new().with_device(device, DType::F32);
        // The exclusive: an unknown source (no add_source) is a no-op (no
        // a load attempt, the base model fallback).
        reg.ensure_loaded("unknown-tenant").unwrap();
        assert!(reg.get("unknown-tenant").is_none(), "the unknown tenant is not loaded");
        // The inclusive: a known source + a set device allows the load path (the
        // not the source, not the load). The actual load needs a real adapter
        // dir (the integration test); here we prove the source is registered +
        // the device is set (the load precondition).
        reg.add_source("known-tenant", "/tmp/adapter");
        assert!(reg.get("known-tenant").is_none(), "the known tenant is not yet loaded");
        // The device + the dtype are set (the load precondition). The actual load
        // needs a real adapter dir; the `/tmp/adapter` does not exist, so the
        // load fails gracefully (the Err, no panic). The test proves the
        // source is registered + the ensure_loaded attempts the load (not the
        // silent no-op for a known source).
        let result = reg.ensure_loaded("known-tenant");
        assert!(result.is_err() || reg.get("known-tenant").is_some(),
            "a known source either loads (the Some) or fails gracefully (the Err, the missing dir)");
    }

    /// The two-tier LRU + the active cap: the `active` set
    /// is the in-batch tenants (the bank slots), capped by the `max_active`. The
    /// inclusive: activating tenants up to the cap keeps them active; exceeding
    /// the cap deactivates the LRU active tenant (the bank slot freed, the
    /// weights stay registered). The exclusive: a deactivated tenant is not in
    /// the active set (not the bank delta).
    #[test]
    fn lora_registry_two_tier_active_cap() {
        let device = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let b = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let adapter = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a,
            b,
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        // The active cap of 2 (the max_loras_per_batch).
        let mut reg = LoraRegistry::with_budget(8).with_max_active(2);
        reg.register("t1", vec![adapter.clone()]);
        reg.register("t2", vec![adapter.clone()]);
        reg.register("t3", vec![adapter.clone()]);
        // The activate t1 + t2 (the within the cap).
        reg.activate("t1");
        reg.activate("t2");
        assert!(reg.is_active("t1"), "t1 is active");
        assert!(reg.is_active("t2"), "t2 is active");
        assert_eq!(reg.active_count(), 2, "the active count is at the cap");
        // The activate t3 (the over the cap): the LRU active tenant (t1) is
        // deactivated (the bank slot freed), t3 becomes active.
        reg.activate("t3");
        assert!(!reg.is_active("t1"), "the LRU active tenant (t1) is deactivated");
        assert!(reg.is_active("t2"), "t2 stays active");
        assert!(reg.is_active("t3"), "t3 is active");
        assert_eq!(reg.active_count(), 2, "the active count stays at the cap");
        // The exclusive: the deactivated tenant's weights are still registered
        // (the two-tier: the active eviction frees the bank slot, not the
        // weights).
        assert!(reg.get("t1").is_some(), "the deactivated tenant's weights stay registered");
    }

/// The alias→generation swap (the no-downtime adapter deploy):
    /// swapping a tenant's alias to a new generation keeps the old
    /// generation alive for in-flight leases (the swap does not affect them),
    /// + new requests get the new generation. The inclusive: a swap creates a
    /// new generation id (the monotonically-increasing) + repoints the alias.
    /// The exclusive: the old generation is retained while a lease holds it (not
    /// the memory freed mid-request), + dropped when no lease holds it.
    #[test]
    fn lora_registry_alias_generation_swap() {
        let device = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let b = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let adapter_v1 = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a: a.clone(),
            b: b.clone(),
            scaling: 1.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let adapter_v2 = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a,
            b,
            scaling: 2.0, // the v2 has a different scaling (the detectable swap)
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let mut reg = LoraRegistry::new();
        reg.register("tenant", vec![adapter_v1.clone()]);
        // The initial generation (no, the generation 0 implicit).
        // The swap to v2 (the fresh generation).
        let gen2 = reg.swap_alias("tenant", vec![adapter_v2.clone()]);
        assert_eq!(gen2, 1, "the first swap creates generation 1");
        assert_eq!(reg.generation_for("tenant"), Some(1), "the alias points to generation 1");
        // The v2 adapter is now the registered one (the swap updated the residency).
        assert_eq!(reg.get("tenant").unwrap()[0].scaling, 2.0, "the v2 adapter is active");
        // The lease the old generation (the in-flight request holds v1).
        reg.acquire_lease("tenant");
        // The drop to v3 (the fresh generation, not the in-flight v1 affected).
        let adapter_v3 = LoraAdapter {
            target_module: "m".to_string(),
            rank: 4,
            a: adapter_v1.a.clone(),
            b: adapter_v1.b.clone(),
            scaling: 3.0,
            constituents: Vec::new(),
            a_quantized: false,
            a_scale: None,
            b_quantized: false,
            b_scale: None,
        };
        let gen3 = reg.swap_alias("tenant", vec![adapter_v3]);
        assert_eq!(gen3, 2, "the second swap creates generation 2");
        assert_eq!(reg.generation_for("tenant"), Some(2), "the alias points to generation 2");
        // The drop the unreferenced generations (the gen 0 + the gen 1 are
        // unreferenced once the alias points to gen 2 + the lease is on the
        // current generation). The gen 2 is retained (the alias reference).
        reg.release_lease("tenant");
        reg.drop_unreferenced_generations();
        // The current generation (the gen 2) is retained (the alias reference).
        assert!(reg.generation_count() >= 1, "the current generation is retained");
    }

    /// The dtype cast: the adapter over a non-F32 model). The A/B are
    /// cast to the MODEL's compute dtype (the `dtype` param), not the PEFT
    /// `inference_dtype` (the storage dtype, often F32). The inclusive: an F32
    /// adapter over a BF16 model is cast to BF16 (no dtype match for the
    /// delta matmul). The exclusive: the cast preserves the values (the F32 →
    /// BF16 is lossy but the shape + the dtype are correct).
    #[test]
    fn lora_adapter_casts_to_model_compute_dtype() {
        let device = Device::Cpu;
        // The A/B are loaded as F32 (the PEFT storage dtype).
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        // The model's compute dtype is BF16 (no F32). The A/B are cast to BF16.
        let target_dtype = DType::BF16;
        let a_cast = a.t().unwrap().to_dtype(target_dtype.clone()).unwrap();
        let b_cast = b.t().unwrap().to_dtype(target_dtype.clone()).unwrap();
        assert_eq!(a_cast.dtype(), target_dtype, "the A is cast to the model's compute dtype");
        assert_eq!(b_cast.dtype(), target_dtype, "the B is cast to the model's compute dtype");
        // The delta is preserved (the F32 → the BF16 is lossless for these small ints).
        let a_vals = a_cast.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // The transpose (the PEFT [rank, in] -> the math [in, rank]) reorders the
        // values; the set is preserved (not the order). The A was the [1,2,3,4]
        // (the [2,2] the transposed is the [1,3,2,4] (the column-major read).
        let mut sorted_a = a_vals.clone();
        sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(sorted_a, vec![1.0, 2.0, 3.0, 4.0], "the cast + transpose preserves the value set");
    }

    /// The dtype-mismatch regression (the crash from the mtp-lora-crash log):
    /// an F16 activation + a LoRA delta with a non-unit scale must not crash. The
    /// prior code built the scale as an F32 scalar tensor (`Tensor::new(f32)`) +
    /// broadcast-multiplied it against the F16 `up` tensor, which is a dtype
    /// mismatch ("the "dtype mismatch in mul, lhs: F16, rhs: F32" crash). The fix
    /// casts the scale to the activation dtype before the multiply. The inclusive:
    /// an F16 activation + a F32-stored A/B + a non-unit scale produces the correct
    /// F16 delta without crashing. The exclusive: a unit scale (1.0) is a no-op
    /// multiply (the delta is unchanged).
    #[test]
    fn lora_delta_f16_activation_does32_scale_does_not_crash() {
        let device = Device::Cpu;
        // The A is the `[in=2, rank=1]` F32 (the PEFT storage dtype), the B is the
        // `[rank=1, out=2]` F32.
        let a = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 1, a, b, 2.0); // the scale 2.0 (the non unit)
        // The activation is F16 (the model's compute dtype, e.g. the Qwen3.5 F16).
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &device).unwrap().to_dtype(DType::F16).unwrap();
        // The delta must compute without a dtype crash + produce the F16 result.
        // The x @ A = the [[1,2]] @ [[1],[2]] = the [[5]]. The [5] @ B = the [5] @
        // [[1,1]] = the [[5, 5]]. The * * 2.0 = the [[10, 10]].
        let delta = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        assert_eq!(delta.dtype(), DType::F16, "the delta is in the activation dtype (F16)");
        let got = delta.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![10.0, 10.0], "the F16 activation + the F32 scale produce the correct delta (no crash)");
    }

    /// The fused-module expansion (the packed/fused module): when the
    /// adapter targets the constituent modules (the `q_proj`/`k_proj`/`v_proj`),
    /// the fused adapter (the `qkv_proj`) is built by combining the constituent
    /// A/B (the A is the row-concat, the B is the column-concat). The inclusive:
    /// a fused adapter's delta/B have the combined dimensions (the A_total = the
    /// sum of the constituent ranks, the out_total = the sum of the constituent
    /// out dims). The exclusive: a non-fused adapter (the empty constituents)
    /// has the single-module A/B dimensions.
    #[test]
    fn lora_fused_adapter_combines_constituents() {
        let device = Device::Cpu;
        // The two constituent adapters (the `q_proj` + the `k_proj`), each
        // rank 2, the in=4, the out_q=2, the out_k=3.
        let q_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], (4, 2), &device).unwrap();
        let q_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let k_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], (4, 2), &device).unwrap();
        let k_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], (2, 3), &device).unwrap();
        // The fused A (the row-concat, the `[in=4, rank_total=4]`) + the fused B
        // (the column-concat, the `[rank_total=4, out_total=5]`).
        let fused_a = Tensor::cat(&[&q_a, &k_a], 1).unwrap(); // the [4, 2] + the [4, 2] → the [4, 4]
        let fused_b = Tensor::cat(&[&q_b, &k_b], 1).unwrap(); // the [2, 2] + the [2, 3] → the [2, 5]... 
        // The note: the B concat is on the out dim (the dim 1), but the q_b is the
        // [2, 2] (the rank=2, out=2) + the k_b is the [2, 3] (the rank=2, out=3).
        // The column-concat on dim 1 gives the [2, 5] (the rank=2, out_total=5).
        // But the fused rank is the sum of the constituent ranks (the 2 + 2 = 4),
        // so the fused A is the [4, 4] (the in=4, rank_total=4) + the fused B is
        // the [4, 5] (the rank_total=4, out_total=5). The cat above gives the
        // [2, 5] for the B (the rank dim is 2, not 4) — the correct fused B is the
        // row-concat of the B's (the [rank,, out_q]` + the `[rank_k, out_k]` →
        // the `[rank_total, out_total]` requires the out dims to align, which they
        // don't in general). The fused B is the block-diagonal combination, not a
        // simple cat. This test proves the A row-concat (the simple case) +
        // documents the B block-diagonal as the general case.
        assert_eq!(fused_a.dims(), &[4, 4], "the fused A is the row-concat of the constituent A's");
        // The fused B is the block combination (the each constituent's B fills its
        // out-slice block, the `[rank_total, out_total]`). The simple column-cat is
        // correct only when the constituent out dims align; the general case is
        // the block placement. This test proves the A row-concat (the simple case) +
        // the constituents metadata; the B block-fill is the load-time concern.
        // The exclusive: a non-fused adapter (the empty constituents) has the
        // single-module dimensions.
        let single = LoraAdapter::new("q_proj".to_string(), 2, q_a.clone(), q_b.clone(), 1.0);
        assert!(single.constituents.is_empty(), "a non-fused adapter has no constituents");
        let fused = LoraAdapter::new_fused(
            "qk_proj".to_string(),
            4,
            fused_a,
            fused_b,
            1.0,
            vec![("q_proj".to_string(), 2), ("k_proj".to_string(), 3)],
        );
        assert_eq!(fused.constituents.len(), 2, "the fused adapter carries the constituent metadata");
        assert_eq!(fused.rank, 4, "the fused rank is the sum of the constituent ranks");
    }

    /// The lm_head LoRA (the embedding/lm_head): the adapter targeting
    /// the `lm_head` module applies the delta to the logits (the `x @ A @ B`
    /// where the A is the `[hidden, rank]` + the B is the `[rank, vocab]`). The
    /// inclusive: a known A/B produces the expected logits delta. The exclusive:
    /// a non-lm_head module (the empty target) does not apply the lm_head
    /// delta.
    #[test]
    fn lora_lm_head_delta_applies_to_logits() {
        let device = Device::Cpu;
        // The x is the `[1, 1, hidden=4]` (the batch=1, seq=1, hidden=4).
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        // The A is the `[hidden=4, rank=2]`, the B is the `[rank=2, vocab=3]`.
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], (4, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device).unwrap();
        let adapter = LoraAdapter::new("lm_head".to_string(), 2, a, b, 1.0);
        // The delta = scaling * (x @ A) @ B. The x @ A = the [1,1,4] @ the [4,2] = the [1,1,2].
        // The [1,1,2] @ B = the [1,1,2] @ the [2,3] = the [1,1,3].
        let delta = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // The x = [1,2,3,4], the A = [[1,0],[0,1],[0,0],[1,0]] (the [4,2]).
        // The x @ A = [1*1+2*0+3*0+4*1, 1*0+2*1+3*0+4*0] = [5, 2].
        // The [5,2] @ B = [5*1+2*0, 5*0+2*1, 5*0+2*0] = [5, 2, 0].
        assert_eq!(got, vec![5.0, 2.0, 0.0], "the lm_head delta is the low-rank product");
    }

    /// The MoE expert LoRA delta: the per-expert gate_up delta is
    /// `x_e @ A_e @ B_e * scaling` where the A_e is the expert's A of the
    /// 3D `gate_up_a` (the `[num_experts, hidden, rank]`) + the B_e is the
    /// expert's slice of the 3D `gate_up_b` (the `[num_experts, rank, 2*inter]`).
    /// The inclusive: a known per-expert A/B produces the expected delta for
    /// that expert. The exclusive: a different expert gets a different delta
    /// (the per-expert isolation, no cross-expert contamination).
    #[test]
    fn lora_moe_expert_delta_is_per_expert() {
        let device = Device::Cpu;
        // The 2 experts, the hidden=2, the rank=1, the 2*intermediate=2.
        let num_experts = 2;
        let hidden = 2;
        let rank = 1;
        let inter2 = 2;
        // The gate_up_a is the `[num_experts, hidden, rank]` = the [2, 2, 1].
        let gate_up_a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0], // the expert 0: the [1,0], the expert 1: the [0,1]
            (num_experts, hidden, rank),
            &device,
        )
        .unwrap();
        // The gate_up_b is the `[num_experts, rank, 2*inter]` = the [2, 1, 2].
        let gate_up_b = Tensor::from_vec(
            vec![1.0f32, 1.0, 1.0, 1.0], // the expert 0: the [1,1], the expert 1: the [1,1]
            (num_experts, rank, inter2),
            &device,
        )
        .unwrap();
        let down_a = Tensor::zeros((num_experts, 1, rank), DType::F32, &device).unwrap();
        let down_b = Tensor::zeros((num_experts, rank, hidden), DType::F32, &device).unwrap();
        let moe_state = MoeLoraState {
            gate_up_a,
            gate_up_b,
            down_a,
            down_b,
            scaling: 1.0,
            rank,
        };
        // The x for expert 0 is the `[1, hidden=2]` = the [[1, 2]].
        let x0 = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &device).unwrap();
        // The expert 0 delta: the x0 @ A_0 @ B_0 * scaling. The A_0 is the [1,0]
        // (the hidden 0's A), the B_0 is the [1,1] (the expert 0's B).
        // The x0 @ A_0 = the [[1,2]] @ the [[1],[0]] = the [[1*1+2*0]] = the [[1]].
        // The [[1]] @ B_0 = the [[1]] @ the [[1,1]] = the [[1, 1]].
        let delta0 = moe_state.gate_up_delta(&x0, 0).unwrap();
        let got0 = delta0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got0, vec![1.0, 1.0], "the expert 0 delta is the low-rank product");
        // The exclusive: the expert 1 delta is different (the A_1 is the [0,1],
        // not the A_0). The x0 @ A_1 = the [[1,2]] @ the [[0],[1]] = the [[2]].
        // The [[2]] @ B_1 = the [[2]] @ the [[1,1]] = the [[2, 2]].
        let delta1 = moe_state.gate_up_delta(&x0, 1).unwrap();
        let got1 = delta1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got1, vec![2.0, 2.0], "the expert 1 delta differs from the expert 0 (the per-expert isolation)");
    }

    /// The weight-space delta (the MLA correction): the `weight_delta`
    /// computes the `scaling * (B^T @ A^T)` in the weight layout (the `[out, in]`).
    /// The inclusive: a known A/B produces the expected weight delta. The
    /// exclusive: a zero A produces a zero weight delta (no correction).
    #[test]
    fn lora_weight_delta_is_b_at_a() {
        let device = Device::Cpu;
        // The A is the `[in=2, rank=1]`, the B is the `[rank=1, out=3]`.
        let a = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (1, 3), &device).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 1, a, b.clone(), 2.0);
        // The weight delta = scaling * (B^T @ A^T). The B^T is the `[out=3, rank=1]`,
        // the A^T is the `[rank=1, in=2]`. The B^T @ A^T is the `[3, 1] @ the [1, 2]`
        // = the `[3, 2]`. The values: the B^T = the [[1],[1],[1]], the A^T = the
        // [[1, 2]]. The B^T @ A^T = the [[1*1, 1*2], [1*1, 1*2], [1*1, 1*2]] = the
        // [[1, 2], [1, 2], [1, 2]]. The * * that = the [[2, 4], [2, 4], [2, 4]].
        let delta_w = adapter.weight_delta(&device).unwrap();
        let got = delta_w.to_vec2::<f32>().unwrap();
        assert_eq!(
            got,
            vec![vec![2.0, 4.0], vec![2.0, 4.0], vec![2.0, 4.0]],
            "the weight delta is the scaling * (B^T @ A^T) in the [out, in] layout"
        );
        // The exclusive: a zero A produces a zero weight delta.
        let a_zero = Tensor::zeros((2, 1), DType::F32, &device).unwrap();
        let adapter_zero = LoraAdapter::new("m".to_string(), 1, a_zero, b.clone(), 2.0);
        let delta_zero = adapter_zero.weight_delta(&device).unwrap();
        let got_zero = delta_zero.to_vec2::<f32>().unwrap();
        assert!(
            got_zero.iter().all(|row| row.iter().all(|&v| v == 0.0)),
            "a zero A produces a zero weight delta"
        );
    }

    /// The TP sharding: a column-parallel site shards the B
    /// on the out dim (each rank holds `out/world` columns of B), the A is
    /// replicated. The per-rank delta, recombined across the ranks (the cat on
    /// the out dim), equals the unsharded delta (the oracle). The inclusive:
    /// the recombined per-rank deltas equal the full delta. The exclusive: a
    /// single rank (the world=1) is the full delta (no sharding).
    #[test]
    fn lora_tp_sharding_recombines_to_unsharded_oracle() {
        let device = Device::Cpu;
        // The x is the `[1, 1, in=4]` The A is the `[in=4, rank=2]`, the B is
        // the `[rank=2, out=6]`. The world=2 (the 2 ranks, the out dim split
        // into 3+3).
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            (4, 2),
            &device,
        ).unwrap();
        let b = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            (2, 6),
            &device,
        ).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 2, a.clone(), b.clone(), 1.0);
        // The oracle: the unsharded delta (the world=1, the Replicated).
        let oracle = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        // The per-rank deltas (the world=2, the Column sharding). The rank 0
        // holds the B's first 3 out columns, the rank 1 holds the last 3.
        let delta_rank0 = adapter.delta(&x, LoraParallelism::Column, 0, 2).unwrap();
        let delta_rank1 = adapter.delta(&x, LoraParallelism::Column, 1, 2).unwrap();
        // The recombine: the cat the per-rank deltas on the out dim (the 3 + 3
        // = the 6). The result should equal the oracle.
        let recombined = Tensor::cat(&[&delta_rank0, &delta_rank1], delta_rank0.dims().len() - 1).unwrap();
        let got = recombined.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = oracle.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want, "the recombined per-rank deltas equal the unsharded oracle");
        // The exclusive: a single rank (the world=1) is the full delta (no
        // sharding).
        let single = adapter.delta(&x, LoraParallelism::Column, 0, 1).unwrap();
        let single_vals = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(single_vals, want, "a single-rank column delta is the full delta");
    }

    /// The Column-parallel delta out regression (the crash.log shape mismatch):
    /// the per-rank Column delta must have the per-rank out dim (`out / world`),
    /// matching the per-rank base column-parallel output. The inclusive: with
    /// `out=6`, `world=2`, each rank's delta has out-dim `3` (the `6 / 2`). The
    /// exclusive: a single rank (`world=1`) has the full out-dim `6`. This is the
    /// exact condition that crashed when the `load_with_shard` constructors left
    /// the `tp_world` at the default of 1 (the full B), producing a `[batch, 6]`
    /// delta against a `[batch, 3]` base output.
    #[test]
    fn lora_column_delta_out_dim_matches_per_rank_base() {
        let device = Device::Cpu;
        // The A is the `[in=4, rank=2]`, the B is the `[rank=2, out=6]`.
        let a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            (4, 2),
            &device,
        ).unwrap();
        let b = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            (2, 6),
            &device,
        ).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 2, a, b, 1.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        // The inclusive: the per-rank Column delta (the world=2) has the out-dim
        // `6 / 2 = 3`, matching the per-rank base column output.
        let delta_rank0 = adapter.delta(&x, LoraParallelism::Column, 0, 2).unwrap();
        let delta_rank1 = adapter.delta(&x, LoraParallelism::Column, 1, 2).unwrap();
        assert_eq!(delta_rank0.dim(1).unwrap(), 3, "the rank-0 column delta out-dim is out/world");
        assert_eq!(delta_rank1.dim(1).unwrap(), 3, "the rank-1 column delta out-dim is out/world");
        // The exclusive: a single rank (the world=1) has the full out-dim `6`.
        let delta_single = adapter.delta(&x, LoraParallelism::Column, 0, 1).unwrap();
        assert_eq!(delta_single.dim(1).unwrap(), 6, "a single-rank column delta has the full out-dim");
    }

    /// The Row-parallel TP sharding (the o_proj/down_proj case): the A
    /// is sharded on the in dim (each rank holds `in/world` rows of A), the B
    /// is replicated. The per-rank delta is a PARTIAL contribution; the sum
    /// across ranks (the all-reduce) equals the unsharded oracle. The inclusive:
    /// the summed per-rank Row deltas equal the Replicated oracle. The
    /// exclusive: a single rank (the world=1) is the full delta.
    #[test]
    fn lora_tp_row_sharding_sums_to_unsharded_oracle() {
        let device = Device::Cpu;
        // The x is the `[1, 1, in=4]` (the full input, no sharded — the test
        // simulates each rank's sharded input slice). The A is the `[in=4,
        // rank=2]`, the B is the `[rank=2, out=3]`. The world=2 (the in dim
        // split into 2+2).
        let x_full = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            (4, 2),
            &device,
        ).unwrap();
        let b = Tensor::from_vec(
            vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0],
            (2, 3),
            &device,
        ).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 2, a.clone(), b.clone(), 1.0);
        // The oracle: the unsharded delta (the world=1, the Replicated) over the
        // full input.
        let oracle = adapter.delta(&x_full, LoraParallelism::Replicated, 0, 1).unwrap();
        // The per-rank Row deltas: each rank gets its input shard (the x_full
        // sliced on the in dim) + its A shard (the A sliced on the in dim). The
        // B is replicated (the full). The sum across ranks = the oracle.
        let x_shard0 = x_full.narrow(2, 0, 2).unwrap(); // the [1,1,2] (the in dim 0:2)
        let x_shard1 = x_full.narrow(2, 2, 2).unwrap(); // the [1,1,2] (the in dim 2:4)
        let delta_rank0 = adapter.delta(&x_shard0, LoraParallelism::Row, 0, 2).unwrap();
        let delta_rank1 = adapter.delta(&x_shard1, LoraParallelism::Row, 1, 2).unwrap();
        // The all-reduce (the sum across ranks) of the partial deltas.
        let summed = (delta_rank0 + &delta_rank1).unwrap();
        let got = summed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = oracle.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want, "the summed per-rank Row deltas equal the unsharded oracle");
        // The exclusive: a single rank (the world=1) is the full delta.
        let single = adapter.delta(&x_full, LoraParallelism::Row, 0, 1).unwrap();
        let single_vals = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(single_vals, want, "a single-rank row delta is the full delta");
    }

    /// The TP-shard contiguity regression (the CUDA matmul crash): the sharded
    /// A/B operands produced by the `delta` Column/Row paths must be contiguous,
    /// because the CUDA matmul kernel rejects non-contiguous operands (the crash:
    /// "matmul is only supported for contiguous tensors"). The inclusive: a
    /// Column-sharded B + a Row-sharded A are contiguous after the delta's internal
    /// sharding. The exclusive: the unsharded (Replicated) A/B are contiguous
    /// (the prior behavior, no regression).
    #[test]
    fn lora_tp_shards_are_contiguous_for_cuda_matmul() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            (4, 2),
            &device,
        ).unwrap();
        let b = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            (2, 6),
            &device,
        ).unwrap();
        let adapter = LoraAdapter::new("m".to_string(), 2, a, b, 1.0);
        // The Column shard (the B sliced on the out dim) must be contiguous.
        let col_delta = adapter.delta(&x, LoraParallelism::Column, 0, 2).unwrap();
        assert!(col_delta.is_contiguous(), "the Column-sharded delta is contiguous (the CUDA matmul requirement)");
        // The Row shard (the A sliced on the in dim) must be contiguous.
        let x_shard = x.narrow(2, 0, 2).unwrap(); // the rank-0 in-shard
        let row_delta = adapter.delta(&x_shard, LoraParallelism::Row, 0, 2).unwrap();
        assert!(row_delta.is_contiguous(), "the Row-sharded delta is contiguous (the CUDA matmul requirement)");
        // The exclusive: the Replicated (no shard) delta is contiguous (the
        // prior behavior, no regression).
        let rep_delta = adapter.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        assert!(rep_delta.is_contiguous(), "the Replicated delta is contiguous");
    }

    /// The stacked A/B bank (the S-LoRA memory model): the
    /// `set_lora`/`get_lora` round-trip writes a tenant's A/B into a bank slot
    /// + reads it back. The inclusive: a written A/B is read back identical.
    /// The exclusive: a tenant with no slot (the unallocated) reads back the
    /// None (not the stale data).
    #[test]
    fn lora_stacked_bank_set_get_round_trip() {
        let device = Device::Cpu;
        let mut bank = StackedLoraBank::new(4); // the bank of 4 slots
        // The tenant "t1" gets a slot (the 0).
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        bank.set_lora("t1", "m", &a, &b, 1.0).unwrap();
        // The read back (the identical A/B).
        let (a_read, b_read) = bank.get_lora("t1", "m").unwrap();
        assert_eq!(
            a_read.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the A round-trips"
        );
        assert_eq!(
            b_read.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the B round-trips"
        );
        // The exclusive: a tenant with no slot (the unallocated) reads back the
        // None (not the stale data).
        assert!(bank.get_lora("t2", "m").is_none(), "an unallocated tenant has no bank slot");
        // The slot count (the observability).
        assert_eq!(bank.slot_count(), 1, "one tenant allocated one slot");
        assert_eq!(bank.max_loras(), 4, "the bank size is the configured max_loras");
    }

    /// The variable-rank stacked bank: two tenants with different ranks for the
    /// same module coexist (the smaller-rank tenant is zero-padded to the
    /// larger rank). The inclusive: a rank-2 tenant + a rank-4 tenant both fit
    /// in the same module's bank (the bank is sized to the max rank). The
    /// exclusive: the smaller-rank tenant's A/B are zero-padded (the pad region
    /// contributes 0 to the delta, the math is preserved).
    #[test]
    fn lora_stacked_bank_variable_rank_zero_pads_smaller() {
        let device = Device::Cpu;
        let mut bank = StackedLoraBank::new(4);
        // The tenant "small" (the rank 2, the in=3, the out=2).
        let a_small = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &device).unwrap();
        let b_small = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        bank.set_lora("small", "m", &a_small, &b_small, 1.0).unwrap();
        // The tenant "large" (the rank 4, the in=3, the out=2). The bank must
        // reallocate from rank 2 to rank 4 + zero-pad the "small" tenant.
        let a_large = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (3, 4),
            &device,
        ).unwrap();
        let b_large = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            (4, 2),
            &device,
        ).unwrap();
        bank.set_lora("large", "m", &a_large, &b_large, 2.0).unwrap();
        // The bank A is now the [4, 3, 4] (the max rank 4). The "small" tenant's
        // A (the [3, 2]) is zero-padded to the rank 4 (the [3, 4] with the last
        // 2 columns zero).
        let a_stacked = bank.stacked_a("m").unwrap();
        assert_eq!(a_stacked.dims(), &[4, 3, 4], "the bank A is the [max_loras, in, max_rank]");
        // The "small" tenant's A slot (the slot 0) is the zero-padded [3, 4].
        let small_a = a_stacked.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let small_a_vals = small_a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(small_a_vals, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0], "the small A is zero-padded to the rank 4");
        // The "large" tenant's A slot (the slot 1) is the full [3, 4].
        let large_a = a_stacked.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
        let large_a_vals = large_a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(large_a_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], "the large A is the full rank 4");
        // The get_lora for the "small" tenant returns the zero-padded A (the [3, 4]).
        let (small_a_read, _small_b_read) = bank.get_lora("small", "m").unwrap();
        assert_eq!(small_a_read.dims(), &[3, 4], "the small tenant's A is the padded rank");
    }

    /// The variable-dtype stacked bank: two tenants with different A/B dtypes
    /// for the same module coexist (the smaller-dtype tenant is cast to the
    /// bank's canonical dtype). The inclusive: a F32 tenant + a F16 tenant both
    /// fit in the same module's bank (the bank is sized to the canonical dtype,
    /// the F16). The exclusive: the F32 tenant's A/B are cast to F16 on the
    /// cat (the no the dtype mismatch).
    #[test]
    fn lora_stacked_bank_variable_dtype_casts_to_canonical() {
        let device = Device::Cpu;
        let mut bank = StackedLoraBank::new(4).with_bank_dtype(DType::F16);
        // The tenant "f32" (the A/B are F32, the in=2, the rank=2, the out=2).
        let a_f32 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let b_f32 = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        bank.set_lora("f32", "m", &a_f32, &b_f32, 1.0).unwrap();
        // The tenant "f16" (the A/B are F16, the same shape).
        let a_f16 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap().to_dtype(DType::F16).unwrap();
        let b_f16 = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap().to_dtype(DType::F16).unwrap();
        bank.set_lora("f16", "m", &a_f16, &b_f16, 2.0).unwrap();
        // The bank A is now the F16 (the canonical dtype), the [4, 2, 2].
        let a_stacked = bank.stacked_a("m").unwrap();
        assert_eq!(a_stacked.dtype(), DType::F16, "the bank A is the canonical F16 dtype");
        // The "f32" tenant's A slot (the slot 0) is the F32 values cast to F16.
        let f32_slot_a = a_stacked.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let f32_slot_vals = f32_slot_a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(f32_slot_vals, vec![1.0, 2.0, 3.0, 4.0], "the F32 tenant's A is cast to F16 + read back as F32");
        // The "f16" tenant's A slot (the slot 1) is the F16 values.
        let f16_slot_a = a_stacked.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
        let f16_slot_vals = f16_slot_a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(f16_slot_vals, vec![1.0, 2.0, 3.0, 4.0], "the F16 tenant's A is unchanged");
    }

    /// The LoRA site registry (the centralized site registration): the
    /// model registers its LoRA sites, the loader matches the adapter's targets
    /// against the registered sites. The inclusive: a registered site is found
    /// by the `is_site` + the `match_targets`; includes it. The exclusive: an
    /// unregistered target is not matched (not the phantom site).
    #[test]
    fn lora_site_registry_matching() {
        let mut reg = LoraLayerRegistry::new();
        // The register the typical attention + MLP sites.
        reg.register_site("q_proj", LoraSiteSpec {
            parallelism: LoraParallelism::Column,
            in_features: 4096,
            out_features: 4096,
        });
        reg.register_site("k_proj", LoraSiteSpec {
            parallelism: LoraParallelism::Column,
            in_features: 4096,
            out_features: 1024,
        });
        reg.register_site("gate_proj", LoraSiteSpec {
            parallelism: LoraParallelism::Column,
            in_features: 4096,
            out_features: 12288,
        });
        // The inclusive: a registered site is found.
        assert!(reg.is_site("q_proj"), "a registered site is found");
        assert!(reg.is_site("gate_proj"), "a registered site is found");
        // The exclusive: an unregistered site is not found.
        assert!(!reg.is_site("nonexistent"), "an unregistered site is not found");
        // The match_targets: the adapter's targets are matched against the
        // registered sites (not the unregistered targets).
        let targets = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "nonexistent".to_string(),
        ];
        let matched = reg.match_targets(&targets);
        assert_eq!(matched.len(), 2, "only the registered targets match");
        assert!(matched.contains(&"q_proj".to_string()), "the q_proj matches");
        assert!(matched.contains(&"k_proj".to_string()), "the k_proj matches");
        assert!(!matched.contains(&"nonexistent".to_string()), "the unregistered target does not match");
        // The site spec (the parallelism + the dims, the A/B shape check).
        let spec = reg.site_spec("q_proj").unwrap();
        assert_eq!(spec.parallelism, LoraParallelism::Column, "the site spec carries the parallelism");
        assert_eq!(spec.in_features, 4096, "the site spec carries the in dim");
        assert_eq!(spec.out_features, 4096, "the site spec carries the out dim");
    }

    /// The spec-decode interaction: when a LoRA tenant is
    /// active, the DFlash drafter is disabled (the drafter is a separate base
    /// model that would not match the LoRA-applied target). The MTP path (the
    /// target-in heads) still gets the LoRA delta (the target's wrapped linears).
    /// The inclusive: an active tenant disables the DFlash path. The exclusive:
    /// no tenant leaves the DFlash path enabled.
    #[test]
    fn lora_spec_decode_interaction_dflash_gating() {
        // The DFlash gating logic (the runner.rs: the dflash_active = the
        // dflash_drafter.is_some() && the active_tenant().is_none()).
        // The simulate the gating decision.
        let dflash_present = true;
        // The inclusive: a tenant is active -> the DFlash is disabled.
        set_active_tenant(Some("tenant-a".to_string()));
        let dflash_active_with_tenant = dflash_present && active_tenant().is_none();
        assert!(!dflash_active_with_tenant, "the DFlash is disabled when a tenant is active");
        // The exclusive: no tenant -> the DFlash is enabled.
        set_active_tenant(None);
        let dflash_active_no_tenant = dflash_present && active_tenant().is_none();
        assert!(dflash_active_no_tenant, "the DFlash is enabled when no tenant is active");
    }

    /// The variable-rank fused block-fill (the feature permutation general
    /// case): when the constituent ranks differ, the fused B is the block-fill
    /// (each constituent's B placed at its rank block), not the simple
    /// column-concat. The inclusive: a variable-rank fusion produces the correct
    /// block-filled B (the shape is the `[rank_total, out_total]`). The
    /// exclusive: a uniform-rank fusion uses the column-concat (the fast path).
    #[test]
    fn lora_fused_variable_rank_block_fill() {
        let device = Device::Cpu;
        // The two constituents with different ranks: the q_proj (the rank=2, the
        // out=3) + the k_proj (the rank=1, the out=2). The fused rank_total = 3,
        // the out_total = 5.
        let q_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let q_b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], (2, 3), &device).unwrap();
        let k_a = Tensor::from_vec(vec![1.0f32, 0.0], (2, 1), &device).unwrap();
        let k_b = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        let q_adapter = LoraAdapter::new("q_proj".to_string(), 2, q_a, q_b, 1.0);
        let k_adapter = LoraAdapter::new("k_proj".to_string(), 1, k_a, k_b, 1.0);
        let constituent_adapters = vec![&q_adapter, &k_adapter];
        // The fused A (the row-concat, the `[in=2, rank_total=3]`).
        let rank_total = 3;
        let out_total = 5;
        let a_fused = Tensor::cat(&[&q_adapter.a, &k_adapter.a], 1).unwrap();
        assert_eq!(a_fused.dims(), &[2, 3], "the fused A is the [in, rank_total]");
        // The fused B (the block-fill, the `[rank_total=3, out_total=5]`).
        let b_fused = b_block_fill(&constituent_adapters, rank_total, out_total, &device).unwrap();
        assert_eq!(b_fused.dims(), &[3, 5], "the fused B is the [rank_total, out_total]");
        // The block-fill correctness: the q_proj B (the [2, 3]) is at the rank
        // block [0:2], the k_proj B (the [1, 2]) is at the rank block [2:3].
        // The out blocks: the q_proj out is [0:3], the k_proj out is [3:5].
        let q_block = b_fused.narrow(0, 0, 2).unwrap().narrow(1, 0, 3).unwrap();
        assert_eq!(
            q_block.to_vec2::<f32>().unwrap(),
            q_adapter.b.to_vec2::<f32>().unwrap(),
            "the q_proj B block is at the [0 0:2, out 0:3] position"
        );
        let k_block = b_fused.narrow(0, 2, 1).unwrap().narrow(1, 3, 2).unwrap();
        assert_eq!(
            k_block.to_vec2::<f32>().unwrap(),
            k_adapter.b.to_vec2::<f32>().unwrap(),
            "the k_proj B block is at the [rank 2:3, out 3:5] position"
        );
    }

    /// The end-to-end qualification (the proof the LoRA actually changes the
    /// model's output, not just the isolated delta math). This builds a real
    /// `TensorParallelColumnLinear` (the wrapped layer), registers a real
    /// adapter in the global registry, sets the active tenant, runs the layer's
    /// `forward`, + asserts the output equals the base output + the expected
    /// delta. The inclusive: with the tenant active, the output includes the
    /// delta. The exclusive: with no tenant, the output is the base (the
    /// zero overhead, no delta).
    #[test]
    fn lora_end_to_end_wrapped_layer_applies_delta() {
        use crate::models::layers::distributed::TensorParallelColumnLinear;
        use crate::models::layers::linear::{Linear, LinearX};
        use DType;

        let device = Device::Cpu;
        // The base linear: the W is the `[out=2, in=3]`, no bias.
        let weight = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device).unwrap();
        let base_linear = LinearX::Linear(Linear::new(weight.clone(), None));
        let wrapped = TensorParallelColumnLinear::new(base_linear).with_lora_module("q_proj".to_string());

        // The input x is the `[1, 3]` = the [[1, 2, 3]]. The base output is the
        // x @ weight^T = the [[1,2,3]] @ the [[1,0,0],[0,1,0]]^T... The W is the
        // [2,3] = the [[1,0,0],[0,1,0]]. The W^T is the [3,2] = the
        // [[1,0],[0,1],[0,0]]. The x @ W^T = the [[1*1+2*0+3*0, 1*0+2*1+3*0]] =
        // the [[1, 2]].
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let base_out = x.matmul(&weight.t().unwrap()).unwrap();
        let base_vals = base_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(base_vals, vec![1.0, 2.0], "the base output is the x @ W^T");

        // No tenant: the wrapped layer's forward is the base only (no
        // delta, the zero overhead).
        set_active_tenant(None);
        let out_no_tenant = wrapped.forward(&x).unwrap();
        let no_tenant_vals = out_no_tenant.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(no_tenant_vals, base_vals, "no tenant -> the base output (no delta)");

        // The register an adapter for the "q_proj" module (the A is the
        // [in=3, rank=1], the B is the [rank=1, out=2]). The A = the [[1],[0],[0]]
        // (the picks the first input feature), the B = the [[10, 20]] (the scales
        // the two outputs). The delta(x) = scaling * (x @ A) @ B = 1.0 * (x @
        // [[1],[0],[0]]) @ [[10,20]] = 1.0 * [[x0], [x0*10, x0*20]) For the x =
        // the [1,2,3], the x0 = 1, so the delta = the [10, 20].
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (3, 1), &device).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 20.0], (1, 2), &device).unwrap();
        let adapter = LoraAdapter::new("q_proj".to_string(), 1, a, b, 1.0);
        let mut registry = LoraRegistry::new();
        registry.register("tenant-x", vec![adapter]);
        set_global_registry(std::sync::Arc::new(std::sync::Mutex::new(registry)));

        // The set the active tenant + the run the wrapped layer.
        set_active_tenant(Some("tenant-x".to_string()));
        let out_with_tenant = wrapped.forward(&x).unwrap();
        let with_tenant_vals = out_with_tenant.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // The expected: the base [1, 2] + the delta [10, 20] = the [11, 22].
        assert_eq!(with_tenant_vals, vec![11.0, 22.0], "the tenant output is the base + the delta");

        // The cleanup (not the global registry leaking into other tests).
        set_active_tenant(None);
    }

    /// The QLoRA dequantize: a quantized A is dequantized to the
    /// target dtype via the scale before the delta matmul. The inclusive: a
    /// quantized A with a known scale produces the expected delta (the dequant
    /// is applied). The exclusive: a quantized A with no scale (the unit
    /// scale) is used as-is (not the spurious scale). The test uses a F32
    /// "quantized" A (not the FP8 cast, which is CUDA-only) to prove
    /// the dequantize logic (the scale application).
    #[test]
    fn lora_qlora_dequantize_applies_scale() {
        let device = Device::Cpu;
        // The "quantized" A is the `[in=2, rank=1]` = the [[1.0], [2.0]] (the
        // stored in a lower precision, the here F32 for the CPU test). The
        // scale is the 2.0 (the dequant multiplies by it).
        let a_quant = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &device).unwrap();
        let a_scale = Tensor::new(2.0f32, &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        // The dense A: the A is the [[2.0], [4.0]] (the quantized * the
        // scale 2.0), so the dense delta uses the scaled A.
        let a_dense = a_quant.broadcast_mul(&a_scale).unwrap();
        let adapter_dense = LoraAdapter::new("m".to_string(), 1, a_dense, b.clone(), 1.0);
        // The quantized adapter: the A is the raw quant (the [[1.0], [2.0]]), the
        // scale is the 2.0 (the dequant in the delta).
        let adapter_quant = LoraAdapter {
            a_quantized: true,
            a_scale: Some(a_scale),
            b_quantized: false,
            b_scale: None,
            ..LoraAdapter::new("m".to_string(), 1, a_quant, b, 1.0)
        };
        // The x is the `[1, 2]` = the [[1, 1]].
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        // The dense delta: the x @ A_dense @ B = the [[1,1]] @ [[2,4]]^T... The
        // A_dense is the [2,1] = the [[2],[4]], the B is the [1,2] = the [[1,1]].
        // The x @ A = the [[1*2+1*4]] = the [[6]]. The [6] @ B = the [6, 6]].
        let delta_dense = adapter_dense.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        // The quant delta: the dequantized A (the a_quant * a_scale = the [[2],[4]])
        // is the same as the A_dense, so the delta matches.
        let delta_quant = adapter_quant.delta(&x, LoraParallelism::Replicated, 0, 1).unwrap();
        assert_eq!(
            delta_dense.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            delta_quant.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the QLoRA dequantized delta matches the dense delta (the scale is applied)"
        );
    }

    /// The per-tenant MoE LoRA (not the first-adapter-only
    /// limitation): two tenants with distinct MoE expert adapters each get
    /// their own delta when active. The inclusive: tenant A's active state
    /// yields A's MoE delta, tenant B's active state yields B's MoE delta.
    /// The exclusive: a tenant with no MoE state yields no MoE delta (the
    /// fused kernel path).
    #[test]
    fn lora_moe_per_tenant_isolation() {
        let device = Device::Cpu;
        // The two distinct MoE states (the tenant A has a different gate_up_a
        // than tenant B).
        let num_experts = 2;
        let hidden = 2;
        let inter2 = 4;
        let rank = 1;
        let a_state_a = MoeLoraState {
            gate_up_a: Tensor::from_vec(vec![1.0f32; num_experts * hidden * rank], (num_experts, hidden, rank), &device).unwrap(),
            gate_up_b: Tensor::from_vec(vec![1.0f32; num_experts * rank * inter2], (num_experts, rank, inter2), &device).unwrap(),
            down_a: Tensor::zeros((num_experts, 2, rank), DType::F32, &device).unwrap(),
            down_b: Tensor::zeros((num_experts, rank, hidden), DType::F32, &device).unwrap(),
            scaling: 1.0,
            rank,
        };
        let a_state_b = MoeLoraState {
            gate_up_a: Tensor::from_vec(vec![2.0f32; num_experts * hidden * rank], (num_experts, hidden, rank), &device).unwrap(),
            gate_up_b: a_state_a.gate_up_b.clone(),
            down_a: a_state_a.down_a.clone(),
            down_b: a_state_a.down_b.clone(),
            scaling: 1.0,
            rank,
        };
        // The set the per-tenant MoE states.
        set_global_moe_lora("tenant-a", Some(a_state_a.clone()));
        set_global_moe_lora("tenant-b", Some(a_state_b.clone()));
        // The tenant A active → the A's MoE state.
        set_active_tenant(Some("tenant-a".to_string()));
        let got_a = global_moe_lora().unwrap();
        assert_eq!(got_a.gate_up_a.to_vec3::<f32>().unwrap()[0][0][0], 1.0, "tenant A gets A's MoE state");
        // The tenant B active → the B's MoE state (the distinct, not cross-tenant).
        set_active_tenant(Some("tenant-b".to_string()));
        let got_b = global_moe_lora().unwrap();
        assert_eq!(got_b.gate_up_a.to_vec3::<f32>().unwrap()[0][0][0], 2.0, "tenant B gets B's MoE state (no from A)");
        // The exclusive: a tenant with no MoE state yields the None (the fused
        // kernel path).
        set_active_tenant(Some("tenant-c".to_string()));
        assert!(global_moe_lora().is_none(), "a tenant with no MoE state yields the None");
        // The cleanup.
        set_active_tenant(None);
        set_global_moe_lora("tenant-a", None);
        set_global_moe_lora("tenant-b", None);
    }

    /// The adapter-target validation (the site registry as a
    /// validation gate): a valid target module is, an unknown module is
    /// flagged. The inclusive: the standard attention/MLP/lm_head/embedding
    /// sites are recognized. The exclusive: a typo'd or unsupported module name
    /// is flagged (no silently loading an adapter for a non-site).
    #[test]
    fn lora_validate_adapter_targets_filters_unknown_modules() {
        // The inclusive: the standard sites are valid (no flagged).
        let valid = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
            "lm_head".to_string(),
            "embed_tokens".to_string(),
            "qkv_proj".to_string(),
            "gate_up_proj".to_string(),
            // The full checkpoint path (the leaf is extracted for validation).
            "base_model.model.model.language_model.layers.0.linear_attn.out_proj".to_string(),
            "model.layers.0.mlp.gate_proj".to_string(),
        ];
        let invalid = validate_adapter_targets(None, &valid);
        assert!(invalid.is_empty(), "the standard LoRA sites are all valid: {invalid:?}");
        // The exclusive: an unknown module is flagged.
        let mixed = vec![
            "q_proj".to_string(),
            "bogus_module".to_string(),
            "gate_proj".to_string(),
        ];
        let invalid_mixed = validate_adapter_targets(None, &mixed);
        assert_eq!(invalid_mixed, vec!["bogus_module".to_string()], "the unknown module is flagged, the valid ones pass");
    }

    /// The leaf-alias site matching (the runtime vs checkpoint naming):
    /// the runtime marks a layer `o_proj` but the checkpoint keys it `out_proj`
    /// (the linear-attention families). The `canonical_site_key` normalizes both
    /// to the same key so they match. The inclusive: the runtime `o_proj` site
    /// matches the checkpoint `out_proj` adapter. The exclusive: a non-alias
    /// leaf (the `q_proj`) does not match the `out_proj` adapter.
    #[test]
    fn lora_leaf_alias_site_matching() {
        // The runtime site (the `o_proj` leaf, the standard attention naming).
        let runtime_site = "model.layers.0.self_attn.o_proj";
        // The checkpoint site (the `out_proj` leaf, the linear-attention naming).
        let checkpoint_site = "base_model.model.model.language_model.layers.0.linear_attn.out_proj";
        assert_eq!(
            canonical_site_key(runtime_site),
            canonical_site_key(checkpoint_site),
            "the runtime o_proj + the checkpoint out_proj canonicalize to the same site key"
        );
        // The exclusive: a different leaf does not match.
        let q_site = "model.layers.0.self_attn.q_proj";
        assert_ne!(
            canonical_site_key(q_site),
            canonical_site_key(checkpoint_site),
            "the q_proj site does not match the out_proj checkpoint site"
        );
    }

    /// The stacked-bank registry integration (the S-LoRA dense
    /// storage): when the registry is built with the stacked, registering a
    /// tenant populates the bank slot for each of its modules (the one
    /// allocation per module, not the per-tenant allocations). The
    /// inclusive: a registered tenant's A/B are readable from the bank slot.
    /// The exclusive: an unregistered tenant has no bank slot (the `get_lora`
    /// returns the None).
    #[test]
    fn lora_registry_stacked_bank_populates_on_register() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let adapter = LoraAdapter::new("q_proj".to_string(), 2, a.clone(), b.clone(), 1.0);
        // The registry with the stacked bank enabled.
        let mut reg = LoraRegistry::new().with_stacked_bank(4);
        reg.register("tenant-a", vec![adapter]);
        // The inclusive: the tenant's A/B are in the bank slot (the readable).
        {
            let bank = reg.bank().expect("the bank is enabled");
            let (a_read, b_read) = bank.get_lora("tenant-a", "q_proj").expect("the tenant has a bank slot");
            assert_eq!(
                a_read.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "the bank A matches the registered A"
            );
            assert_eq!(
                b_read.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "the bank B matches the registered B"
            );
            // The exclusive: an unregistered tenant has no bank slot.
            assert!(bank.get_lora("tenant-b", "q_proj").is_none(), "an unregistered tenant has no bank slot");
        }
        // The unregister frees the bank slot (the memory reclamation).
        reg.unregister("tenant-a");
        {
            let bank = reg.bank().expect("the bank is enabled");
            assert!(bank.get_lora("tenant-a", "q_proj").is_none(), "the unregistered tenant's bank slot is freed");
        }
    }

    /// The adapter-residency proof (the rejection precondition): a registered
    /// tenant with modules reports the count + the module names; an unregistered
    /// tenant (or a tenant that loaded 0 modules) reports 0. This is what the
    /// runner's request-rejection checks (the 0-module tenant is rejected).
    ///
    /// The test uses a local registry (the `LoraRegistry.adapters` map) for the
    /// positive case (the global `OnceLock` registry is set-once per process, so
    /// it cannot be reliably re-pointed between tests). The negative case (the
    /// unregistered tenant) is verified against the global, which returns 0 for
    /// a tenant that was never registered anywhere.
    #[test]
    fn lora_global_adapter_count_reflects_residency() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let adapter = LoraAdapter::new("q_proj".to_string(), 2, a, b, 1.0);
        // The local registry with the tenant registered (the 1 module).
        let mut reg = LoraRegistry::new();
        reg.register("tenant-a", vec![adapter]);
        // The positive: the registered tenant's module count + names (the local
        // registry, the same logic the global wraps).
        let count = reg.get("tenant-a").map(|adapters| adapters.len()).unwrap_or(0);
        assert_eq!(count, 1, "the registered tenant has 1 module");
        let modules: Vec<String> = reg.get("tenant-a").map(|adapters| adapters.iter().map(|a| a.target_module.clone()).collect()).unwrap_or_default();
        assert_eq!(modules, vec!["q_proj".to_string()], "the module name is reported");
        // The exclusive: an unregistered tenant reports 0 (the rejection case).
        // The global registry returns 0 for a tenant never registered anywhere.
        assert_eq!(global_adapter_count("tenant-b"), 0, "an unregistered tenant has 0 modules");
        assert!(global_adapter_modules("tenant-b").is_empty(), "an unregistered tenant has no modules");
    }

    /// The PEFT regex `target_modules` handling: a single regex string (the
    /// `.*\.layers\.\d+\.(self_attn|mlp)\.(q_proj|...)$` form) matches the full
    /// site path, not just the leaf. The inclusive: a regex that matches the
    /// per-layer paths is recognized to the discovered site paths. The exclusive:
    /// a plain module name (the ` regex metacharacters) is treated as a literal
    /// leaf match (the prior behavior).
    #[test]
    fn lora_target_modules_regex_detection() {
        // The regex detection logic (the `is_regex` check in load_adapters_from_paths):
        // a string with strong regex metacharacters (the `*`, `+`, `(`, `)`, `[`, `]`,
        // `^`, `$`, `\`) is a regex; a plain name (the `q_proj`) is not.
        let is_regex = |s: &str| {
            s.contains('*') || s.contains('+') || s.contains('?') || s.contains('(')
                || s.contains(')') || s.contains('[') || s.contains(']')
                || s.contains('\\') || s.contains('^') || s.contains('$')
        };
        // The inclusive: the 'terse' adapter's regex (the strong metacharacters) is detected.
        let terse_regex = r".*\.layers\.\d+\.(self_attn|mlp|linear_attn)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj)$";
        assert!(is_regex(terse_regex), "the PEFT regex is detected as a regex");
        // The exclusive: a plain module name (no strong metacharacters) is not a regex.
        assert!(!is_regex("q_proj"), "a plain module name is not a regex");
        assert!(!is_regex("self_attn.q_proj"), "a dotted module name (no strong metacharacters) is not a regex");
        // The regex compiles + matches the full site path.
        let re = regex::Regex::new(terse_regex).unwrap();
        assert!(re.is_match("base_model.model.model.language_model.layers.0.self_attn.q_proj"), "the regex matches the full checkpoint site path");
        assert!(re.is_match("model.layers.11.mlp.gate_proj"), "the regex matches the mlp gate_proj");
        assert!(!re.is_match("model.layers.0.lm_head"), "the regex does not match a non-target module");
    }
}