// src/utils/kvcache_allocator.rs
//!
//! KVCache Allocation Module
//!
//! This module provides a centralized, robust entry point for determining
//! available GPU memory, calculating KVCache blocks, and allocating KV cache tensors.
//!
//! # Usage
//!
//! ```ignore
//! // After model loading, create allocator and plan allocation
//! let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
//! let available_memory = allocator.get_available_memory(&device_ids)?;
//! let allocation = allocator.plan_allocation(available_memory)?;
//! // Allocate tensors
//! let (gpu_cache, cpu_cache) = allocator.init_kv_cache(&allocation, dtype, &device)?;
//! ```

use super::{gemma4_per_layer_cache_config, qwen3_hybrid_layer_types, resolve_qwen3_hybrid_config};
use crate::utils::config::{Config, EngineConfig};
use candle_core::{DType, Device, Result, Tensor};
use std::fmt;

/// Reserved memory constants - used for post-allocation warnings
const CUDA_RESERVED_BYTES: u64 = 512 * 1024 * 1024; // 512 MB recommended minimum remaining memory
/// Minimum activation reserve when model-aware computation yields a very small value.
const MIN_ACTIVATION_RESERVE_BYTES: u64 = 256 * 1024 * 1024; // 256 MB floor
const SIZE_IN_MB: f64 = (1024 * 1024) as f64;
const SIZE_IN_GB: f64 = 1024.0 * 1024.0 * 1024.0;
const DEFAULT_HYBRID_MAMBA_FRACTION: f64 = 0.20;
const MAX_HYBRID_MAMBA_FRACTION: f64 = 0.35;
const HYBRID_MAMBA_ACTIVE_SLOT_MULTIPLIER: usize = 3;
const HYBRID_MAMBA_MIN_ACTIVE_SLOTS: usize = 8;

/// Per-category GPU memory budget computed deterministically from model config.
#[derive(Debug, Clone)]
pub struct GpuMemoryBudget {
    /// FlashInfer float+int workspace (fixed, 0 if disabled)
    pub flashinfer_bytes: u64,
    /// CUTLASS workspace (fixed, 0 if disabled)
    pub cutlass_bytes: u64,
    /// MoE activation pool peak estimate
    pub moe_pool_bytes: u64,
    /// Per-layer Flash split-K workspace total
    pub flash_splitk_bytes: u64,
    /// Transient activation overhead (largest forward-pass intermediate)
    pub transient_bytes: u64,
    /// Total workspace reserve (sum of above)
    pub total_bytes: u64,
}

impl GpuMemoryBudget {
    pub fn report(&self, min_available_before: u64) {
        let mut parts = Vec::new();
        if self.flashinfer_bytes > 0 {
            parts.push(format!(
                "FlashInfer {:.0}M",
                self.flashinfer_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.cutlass_bytes > 0 {
            parts.push(format!(
                "CUTLASS {:.0}M",
                self.cutlass_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.moe_pool_bytes > 0 {
            parts.push(format!(
                "MoE pool {:.0}M",
                self.moe_pool_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.flash_splitk_bytes > 0 {
            parts.push(format!(
                "SplitK {:.0}M",
                self.flash_splitk_bytes as f64 / SIZE_IN_MB
            ));
        }
        parts.push(format!(
            "Transient {:.0}M",
            self.transient_bytes as f64 / SIZE_IN_MB
        ));
        crate::log_warn!(
            "GPU Memory Budget: {:.2} GB available → {:.2} GB workspace reserve ({}) → {:.2} GB for caches",
            min_available_before as f64 / SIZE_IN_GB,
            self.total_bytes as f64 / SIZE_IN_GB,
            parts.join(" + "),
            (min_available_before.saturating_sub(self.total_bytes)) as f64 / SIZE_IN_GB
        );
    }
}

/// Represents the result of KVCache allocation planning
#[derive(Debug, Clone)]
pub struct KVCacheAllocation {
    /// Number of GPU blocks for KVCache
    pub num_gpu_blocks: usize,
    /// Number of CPU blocks for KVCache swap
    pub num_cpu_blocks: usize,
    /// Maximum number of concurrent sequences
    pub max_num_seqs: usize,
    /// Maximum model context length
    pub max_model_len: usize,
    /// Total GPU memory allocated for KVCache in bytes
    pub kvcache_memory_bytes: usize,
    /// Maximum number of batched tokens
    pub max_num_batched_tokens: usize,
}

/// Error types for KVCache allocation
#[derive(Debug, Clone)]
pub enum KVCacheError {
    /// Not enough GPU memory to allocate KVCache
    InsufficientGpuMemory {
        available_mb: f64,
        required_mb: f64,
        reserved_mb: f64,
    },
    /// Invalid configuration parameters
    InvalidConfiguration { message: String },
    /// Platform-specific error (e.g., CUDA/Metal not available)
    PlatformError { message: String },
}

impl fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVCacheError::InsufficientGpuMemory {
                available_mb,
                required_mb,
                reserved_mb,
            } => {
                write!(
                    f,
                    "Insufficient GPU memory for KVCache allocation.\n\
                     Available: {:.2} MB, Required: {:.2} MB, Reserved: {:.2} MB.\n\
                     Tips: Try reducing --max-model-len or --max-num-seqs, \
                     or free GPU resources.",
                    available_mb, required_mb, reserved_mb
                )
            }
            KVCacheError::InvalidConfiguration { message } => {
                write!(f, "Invalid KVCache configuration: {}", message)
            }
            KVCacheError::PlatformError { message } => {
                write!(f, "Platform error: {}", message)
            }
        }
    }
}

impl std::error::Error for KVCacheError {}

/// Main allocator struct for platform-aware KVCache memory planning
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KVCacheAllocator {
    // Model parameters
    num_hidden_layers: usize,
    /// Number of layers that need KV cache (excludes GDN/Mamba layers)
    num_kv_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_shards: usize,
    block_size: usize,
    // User constraints (None = auto-decide)
    user_max_model_len: Option<usize>,
    user_max_num_seqs: Option<usize>,
    config_model_len: usize,
    kv_fraction: f64,
    kvcache_dtype: crate::utils::config::KvCacheDtype,
    cpu_mem_fold: f32,
    dtype_size: usize,
    model_dtype_size: usize,
    hybrid_mamba_slot_bytes: Option<usize>,
    hybrid_num_gdn_layers: usize,
    is_mla: bool,
    mla_kv_lora_rank: usize,
    mla_qk_rope_head_dim: usize,
    /// Per-layer KV cache config: (num_kv_heads, head_dim) per KV layer.
    /// When set, overrides uniform num_kv_heads/head_dim for cache allocation.
    per_layer_cache_config: Option<Vec<(usize, usize)>>,
    // MoE config for workspace budget computation
    num_attention_heads: usize,
    hidden_size: usize,
    moe_intermediate_size: usize,
    moe_num_experts_per_tok: usize,
    is_moe: bool,
    prefill_chunk_size: usize,
}

impl KVCacheAllocator {
    fn kv_heads_per_shard_for(&self, num_kv_heads: usize) -> usize {
        if self.num_shards == 0 {
            return 1;
        }

        if num_kv_heads >= self.num_shards {
            num_kv_heads / self.num_shards
        } else {
            1
        }
    }

    fn kv_heads_per_shard(&self) -> usize {
        self.kv_heads_per_shard_for(self.num_kv_heads)
    }

    fn layer_kv_config(&self, layer_idx: usize) -> (usize, usize) {
        self.per_layer_cache_config
            .as_ref()
            .and_then(|configs| configs.get(layer_idx).copied())
            .unwrap_or((self.num_kv_heads, self.head_dim))
    }

    fn layer_flash_key_value_block_shape(&self, layer_idx: usize) -> (usize, usize, usize) {
        let (num_kv_heads, head_dim) = self.layer_kv_config(layer_idx);
        (
            self.block_size,
            self.kv_heads_per_shard_for(num_kv_heads),
            head_dim,
        )
    }

    fn layer_key_block_shape(
        &self,
        layer_idx: usize,
        cache_dtype: DType,
    ) -> (usize, usize, usize, usize) {
        let (_, kv_heads, head_dim) = self.layer_flash_key_value_block_shape(layer_idx);
        let element_size = cache_dtype.size_in_bytes();
        let x = 16 / element_size;
        (kv_heads, head_dim / x, self.block_size, x)
    }

    fn layer_value_block_shape(&self, layer_idx: usize) -> (usize, usize, usize) {
        let (_, kv_heads, head_dim) = self.layer_flash_key_value_block_shape(layer_idx);
        (kv_heads, head_dim, self.block_size)
    }

    /// Create a new KVCacheAllocator from engine and model configs
    pub fn new(econfig: &EngineConfig, config: &Config, dtype: DType) -> Self {
        let configured_num_shards = econfig.num_shards.unwrap_or(1);
        if configured_num_shards == 0 {
            crate::log_warn!(
                "EngineConfig.num_shards=0 is invalid; defaulting to 1 for KVCache allocation"
            );
        }
        let num_shards = configured_num_shards.max(1);
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let fp8_kvcache = econfig.kvcache_dtype.is_fp8_keys();
        let dtype_size = if fp8_kvcache {
            1
        } else {
            dtype.size_in_bytes()
        };
        let model_dtype_size = dtype.size_in_bytes();

        let kv_fraction = econfig
            .kv_fraction
            .unwrap_or(if cfg!(feature = "flashattn") {
                0.7
            } else {
                0.6
            }) as f64;

        let config_model_len = econfig
            .config_model_len
            .unwrap_or(config.max_position_embeddings);

        // For hybrid models (e.g., Qwen3.5), only count full-attention layers
        let num_kv_layers = if let Some(block_types) = qwen3_hybrid_layer_types(config) {
            block_types
                .iter()
                .filter(|t| t.as_str() == "full_attention")
                .count()
        } else {
            config.num_hidden_layers
        };

        let (hybrid_mamba_slot_bytes, hybrid_num_gdn_layers) = if let Some(block_types) =
            qwen3_hybrid_layer_types(config)
        {
            let num_gdn_layers = block_types
                .iter()
                .filter(|t| t.as_str() == "linear_attention")
                .count();
            if num_gdn_layers == 0 {
                (None, 0)
            } else {
                let hybrid = resolve_qwen3_hybrid_config(config);
                let shard_count = num_shards.max(1);
                if hybrid.num_v_heads % shard_count != 0 || hybrid.num_k_heads % shard_count != 0 {
                    crate::log_warn!(
                            "Hybrid mamba heads are not divisible by num_shards (v_heads={}, k_heads={}, shards={}); memory estimate uses floor division.",
                            hybrid.num_v_heads,
                            hybrid.num_k_heads,
                            shard_count
                        );
                }
                let num_v_heads = std::cmp::max(1, hybrid.num_v_heads / shard_count);
                let num_k_heads = std::cmp::max(1, hybrid.num_k_heads / shard_count);
                let conv_window = hybrid.conv_kernel_size.saturating_sub(1);
                let d_conv = num_k_heads
                    .saturating_mul(hybrid.key_head_dim)
                    .saturating_mul(2)
                    .saturating_add(num_v_heads.saturating_mul(hybrid.value_head_dim));
                let per_layer_conv_bytes = d_conv
                    .saturating_mul(conv_window)
                    .saturating_mul(model_dtype_size);
                let per_layer_recurrent_bytes = num_v_heads
                    .saturating_mul(hybrid.key_head_dim)
                    .saturating_mul(hybrid.value_head_dim)
                    .saturating_mul(DType::F32.size_in_bytes());
                let per_slot_bytes = num_gdn_layers
                    .saturating_mul(per_layer_conv_bytes.saturating_add(per_layer_recurrent_bytes));
                if per_slot_bytes == 0 {
                    (None, num_gdn_layers)
                } else {
                    (Some(per_slot_bytes), num_gdn_layers)
                }
            }
        } else {
            (None, 0)
        };

        let (is_mla, mla_kv_lora_rank, mla_qk_rope_head_dim) = {
            let extra: Option<serde_json::Value> = config
                .extra_config_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok());
            if let Some(ref extra) = extra {
                let kv_lora_rank = extra.get("kv_lora_rank").and_then(|v| v.as_u64());
                if let Some(rank) = kv_lora_rank {
                    let rope_dim = extra
                        .get("qk_rope_head_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(64) as usize;
                    (true, rank as usize, rope_dim)
                } else {
                    (false, 0, 0)
                }
            } else {
                (false, 0, 0)
            }
        };

        let kvcache_dtype = if is_mla && econfig.kvcache_dtype.is_turboquant() {
            crate::log_warn!(
                "TurboQuant ({:?}) is not supported for MLA models (kv_lora_rank={}). \
                 MLA uses compressed KV cache layout incompatible with TurboQuant. \
                 Falling back to auto KV cache dtype.",
                econfig.kvcache_dtype,
                mla_kv_lora_rank,
            );
            crate::utils::config::KvCacheDtype::Auto
        } else {
            econfig.kvcache_dtype
        };

        let per_layer_cache_config = match gemma4_per_layer_cache_config(config) {
            Some(configs) if configs.len() == num_kv_layers => Some(configs),
            Some(_) => {
                crate::log_warn!(
                    "Ignoring Gemma4 heterogeneous KV cache config because it does not match num_kv_layers={}.",
                    num_kv_layers
                );
                None
            }
            None => None,
        };

        let is_moe = config.moe_cfg.is_some()
            && config
                .moe_cfg
                .as_ref()
                .is_some_and(|m| m.num_experts.unwrap_or(0) > 1);
        let moe_intermediate_size = config
            .moe_cfg
            .as_ref()
            .map(|m| m.moe_intermediate_size)
            .unwrap_or(0);
        let moe_num_experts_per_tok = config
            .moe_cfg
            .as_ref()
            .map(|m| m.num_experts_per_tok)
            .unwrap_or(0);

        Self {
            num_hidden_layers: config.num_hidden_layers,
            num_kv_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_shards,
            block_size: econfig.block_size,
            user_max_model_len: econfig.max_model_len,
            user_max_num_seqs: if econfig.max_model_len.is_some() {
                Some(econfig.max_num_seqs)
            } else {
                None // Auto-decide if max_model_len not specified
            },
            config_model_len,
            kv_fraction: if econfig.max_model_len.is_some() && econfig.kv_fraction.is_none() {
                0.95
            } else {
                kv_fraction
            },
            kvcache_dtype,
            cpu_mem_fold: econfig.cpu_mem_fold.unwrap_or(0.2),
            dtype_size,
            model_dtype_size,
            hybrid_mamba_slot_bytes,
            hybrid_num_gdn_layers,
            is_mla,
            mla_kv_lora_rank,
            mla_qk_rope_head_dim,
            per_layer_cache_config,
            num_attention_heads: config.num_attention_heads,
            hidden_size: config.hidden_size,
            moe_intermediate_size,
            moe_num_experts_per_tok,
            is_moe,
            prefill_chunk_size: econfig.effective_prefill_chunk_size(),
        }
    }

    pub fn resolved_kvcache_dtype(&self) -> crate::utils::config::KvCacheDtype {
        self.kvcache_dtype
    }

    /// Compute the deterministic workspace budget from model and engine config.
    ///
    /// This replaces the flat `ACTIVATION_RESERVE_BYTES` with a model-aware
    /// calculation that accounts for all known persistent and transient GPU
    /// memory consumers that are not tracked in KV cache or mamba budgets.
    pub fn compute_workspace_budget(&self) -> GpuMemoryBudget {
        // FlashInfer workspace: 512 MiB float + 128 MiB int (GPU) + 128 MiB pinned host (not GPU)
        // Graph capture duplicates the GPU buffers, but those are tracked separately.
        let flashinfer_bytes: u64 = if cfg!(feature = "flashinfer") {
            (512 + 128) * 1024 * 1024 // float + int buffers
        } else {
            0
        };

        // CUTLASS workspace: 512 MiB dedicated buffer
        let cutlass_bytes: u64 = if cfg!(feature = "cutlass") {
            512 * 1024 * 1024
        } else {
            0
        };

        // MoE activation pool: only for NVFP4/MXFP4 MoE models with cutlass
        // Peak = prefill_chunk_size × topk × max(hidden, 2×intermediate) × dtype_size
        // Covers: gathered, rep_out (largest buffers); act_packed and act_scales are smaller.
        let moe_pool_bytes: u64 =
            if cfg!(feature = "cutlass") && self.is_moe && self.moe_num_experts_per_tok > 0 {
                let topk = self.moe_num_experts_per_tok;
                let size_m = self.prefill_chunk_size * topk;
                let hidden = self.hidden_size / self.num_shards.max(1);
                let inter = self.moe_intermediate_size / self.num_shards.max(1);
                let largest_dim = hidden.max(2 * inter);
                // gathered: size_m × hidden × dtype, rep_out: size_m × inter × dtype,
                // act_packed: size_m × hidden/2, act_scales: ~size_m × hidden/16
                let gathered = size_m * hidden * self.model_dtype_size;
                let rep_out = size_m * inter * self.model_dtype_size;
                let act_packed = size_m * hidden / 2;
                let act_scales = size_m * (hidden / 16 + 128);
                // Use sum of the four main pool buffers
                let pool_total = gathered + rep_out + act_packed + act_scales;
                // Also account for per-call transient output: size_m × largest_dim × dtype
                let transient_output = size_m * largest_dim * self.model_dtype_size;
                (pool_total + transient_output) as u64
            } else {
                0
            };

        // Flash split-K workspace: per KV layer, 64 seqs × q_heads × 8 splits × (head_dim+2) × 4 bytes
        let flash_splitk_bytes: u64 = if cfg!(feature = "flash") || cfg!(feature = "flashattn") {
            let q_heads_per_shard = self.num_attention_heads / self.num_shards.max(1);
            let splits = 8usize; // flash::NUM_SPLITS
            let per_layer = 64 * q_heads_per_shard * splits * (self.head_dim + 2) * 4;
            (per_layer * self.num_kv_layers) as u64
        } else {
            0
        };

        // Transient activation overhead: largest intermediate tensor during forward pass.
        // Approximately 2 × prefill_chunk_size × hidden_size × dtype_size (for gate+up projections).
        let transient_bytes =
            (2 * self.prefill_chunk_size * self.hidden_size * self.model_dtype_size) as u64;

        let total_bytes = flashinfer_bytes
            + cutlass_bytes
            + moe_pool_bytes
            + flash_splitk_bytes
            + transient_bytes;
        let total_bytes = total_bytes.max(MIN_ACTIVATION_RESERVE_BYTES);

        GpuMemoryBudget {
            flashinfer_bytes,
            cutlass_bytes,
            moe_pool_bytes,
            flash_splitk_bytes,
            transient_bytes,
            total_bytes,
        }
    }

    /// Set per-layer KV cache configuration for models with heterogeneous head dims
    /// (e.g., Gemma4 with SWA head_dim=256 and full-attention head_dim=512).
    /// Each entry is (num_kv_heads, head_dim) for one KV layer.
    pub fn set_per_layer_cache_config(&mut self, configs: Vec<(usize, usize)>) {
        assert_eq!(configs.len(), self.num_kv_layers);
        self.per_layer_cache_config = Some(configs);
    }

    pub fn plan(&self, device_ids: &[usize], econfig: &mut EngineConfig) -> Result<()> {
        match self.get_available_memory(device_ids) {
            Ok(available_before_reserve) => {
                let workspace_budget = self.compute_workspace_budget();
                workspace_budget.report(available_before_reserve);
                let activation_reserve = workspace_budget
                    .total_bytes
                    .min(available_before_reserve.saturating_sub(1));
                let cache_available = available_before_reserve.saturating_sub(activation_reserve);
                let mut kv_budget = cache_available;
                let mut mamba_budget = 0u64;
                let mut mamba_budget_slots = 0usize;
                let mut mamba_budget_enabled = false;

                if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                    let requested_fraction = econfig
                        .mamba_fraction
                        .map(|f| f as f64)
                        .unwrap_or(DEFAULT_HYBRID_MAMBA_FRACTION)
                        .clamp(0.0, MAX_HYBRID_MAMBA_FRACTION);
                    if requested_fraction > 0.0 {
                        mamba_budget_enabled = true;

                        let mut target_budget =
                            ((available_before_reserve as f64) * requested_fraction) as u64;
                        let min_one_slot = slot_bytes as u64;
                        if target_budget > 0 && target_budget < min_one_slot {
                            crate::log_warn!(
                                "Hybrid mamba budget {:.2} MB is smaller than one slot {:.2} MB; bumping to one slot.",
                                target_budget as f64 / SIZE_IN_MB,
                                min_one_slot as f64 / SIZE_IN_MB
                            );
                            target_budget = min_one_slot;
                        }
                        if target_budget >= cache_available {
                            candle_core::bail!(
                                "Hybrid mamba budget ({:.2} GB) plus workspace reserve ({:.2} GB) leaves no memory for KV cache. Reduce mamba_fraction or max_model_len.",
                                target_budget as f64 / SIZE_IN_GB,
                                activation_reserve as f64 / SIZE_IN_GB
                            );
                        }

                        mamba_budget = target_budget;
                        kv_budget = cache_available.saturating_sub(mamba_budget);
                        mamba_budget_slots = if slot_bytes == 0 {
                            0
                        } else {
                            (mamba_budget as usize / slot_bytes).max(1)
                        };
                    }
                }

                match self.plan_allocation(kv_budget, device_ids.len()) {
                    Ok(allocation) => {
                        self.apply_to_config(&allocation, econfig);

                        if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                            if !mamba_budget_enabled {
                                econfig.mamba_slot_bytes = 0;
                                econfig.mamba_memory_bytes = 0;
                                econfig.mamba_cache_capacity = None;
                                return Ok(());
                            }
                            let active_mamba_capacity = if econfig.prefix_cache.unwrap_or(false) {
                                econfig
                                    .max_num_seqs
                                    .saturating_mul(HYBRID_MAMBA_ACTIVE_SLOT_MULTIPLIER)
                                    .max(HYBRID_MAMBA_MIN_ACTIVE_SLOTS)
                                    .min(mamba_budget_slots)
                            } else {
                                mamba_budget_slots
                            };
                            if allocation.max_num_seqs > active_mamba_capacity {
                                crate::log_warn!(
                                    "Clamping max_num_seqs from {} to {} due to hybrid mamba slot capacity.",
                                    allocation.max_num_seqs,
                                    active_mamba_capacity
                                );
                                econfig.max_num_seqs = active_mamba_capacity;
                            }
                            let prefix_budget_slots =
                                mamba_budget_slots.saturating_sub(active_mamba_capacity);
                            econfig.mamba_slot_bytes = slot_bytes;
                            econfig.mamba_memory_bytes = mamba_budget as usize;
                            econfig.mamba_cache_capacity = Some(active_mamba_capacity);
                            crate::log_warn!(
                                "Hybrid Mamba Allocation: {} active slot(s), {} prefix slot budget, {} total slot budget, {:.2} GB budget, {:.2} MB/slot, {} linear-attention layer(s), model dtype {} bytes",
                                active_mamba_capacity,
                                prefix_budget_slots,
                                mamba_budget_slots,
                                mamba_budget as f64 / SIZE_IN_GB,
                                slot_bytes as f64 / SIZE_IN_MB,
                                self.hybrid_num_gdn_layers,
                                self.model_dtype_size
                            );
                        } else {
                            econfig.mamba_slot_bytes = 0;
                            econfig.mamba_memory_bytes = 0;
                            econfig.mamba_cache_capacity = None;
                        }
                        Ok(())
                    }
                    Err(e) => {
                        crate::log_error!("KVCache allocation failed: {}", e);
                        candle_core::bail!("KVCache allocation failed: {}", e)
                    }
                }
            }
            Err(e) => {
                crate::log_error!("Failed to get available memory: {:?}", e);
                Err(e)
            }
        }?;

        Ok(())
    }
    /// Calculate per-block memory size in bytes.
    /// For turbo4/turbo3, only TQ buffers are counted (standard cache is a 1-block dummy).
    /// For turbo8, standard FP8 cache + TQ V buffers.
    pub fn per_block_bytes(&self) -> usize {
        use crate::utils::config::KvCacheDtype;

        let tq_full = matches!(
            self.kvcache_dtype,
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3
        );

        let base = if tq_full {
            // turbo4/turbo3: standard K/V cache is NOT allocated per-block
            0
        } else if self.is_mla {
            self.block_size
                * (self.mla_kv_lora_rank + self.mla_qk_rope_head_dim)
                * self.dtype_size
                * self.num_kv_layers
        } else if let Some(ref configs) = self.per_layer_cache_config {
            let mut total = 0usize;
            for &(kv_heads, hd) in configs {
                let heads_per_shard = self.kv_heads_per_shard_for(kv_heads);
                total += self.block_size * heads_per_shard * hd * self.dtype_size * 2;
            }
            total
        } else {
            self.block_size
                * self.kv_heads_per_shard()
                * self.head_dim
                * self.dtype_size
                * 2 // K and V
                * self.num_kv_layers
        };

        let tq_extra = match self.kvcache_dtype {
            KvCacheDtype::Turbo8 => {
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4          // v_absmax: f32 per head per token
                    + self.block_size * heads * (hd / 2) // v_quant: packed 4-bit
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            KvCacheDtype::Turbo4 => {
                // K and V each: absmax (f32 per head) + quant (hd/2 packed u8)
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4 * 2          // k_absmax + v_absmax
                    + self.block_size * heads * (hd / 2) * 2 // k_quant + v_quant
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            KvCacheDtype::Turbo3 => {
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4 * 2              // k_absmax + v_absmax
                    + self.block_size * heads * ((hd * 3 + 7) / 8) // k_quant (3-bit packed)
                    + self.block_size * heads * (hd / 2) // v_quant (4-bit packed)
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            _ => 0,
        };

        base + tq_extra
    }

    /// Calculate required memory for given parameters
    pub fn calculate_required_memory(&self, num_seqs: usize, model_len: usize) -> usize {
        let blocks_per_seq = (model_len + self.block_size - 1) / self.block_size;
        let num_blocks = num_seqs * blocks_per_seq;
        num_blocks * self.per_block_bytes()
    }

    /// Query available GPU memory for a single device (platform-aware)
    /// Returns usable memory AFTER applying kv_fraction (reserved memory check is done post-allocation)
    #[cfg(feature = "cuda")]
    pub fn get_rank_available_memory(&self, device_id: usize) -> Result<u64> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::cuda_backend::CudaDevice;

        // Create a CUDA context for that device
        let _ = CudaDevice::new(device_id)?;

        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;

            sys::lib()
                .cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuMemGetInfo_v2 failed: {e:?}")))?;

            // Apply kv_fraction only - reserved memory check is done post-allocation
            let usable = (free as f64 * self.kv_fraction) as u64;

            crate::log_warn!(
                "GPU {}: total {:.2} GB, free {:.2} GB, kv_fraction {:.0}%, Max usable cache budget {:.2} GB",
                device_id,
                total as f64 / SIZE_IN_GB,
                free as f64 / SIZE_IN_GB,
                self.kv_fraction * 100.0,
                usable as f64 / SIZE_IN_GB
            );

            Ok(usable)
        }
    }

    /// Query available GPU memory for a single device (non-CUDA platforms)
    #[cfg(not(feature = "cuda"))]
    pub fn get_rank_available_memory(&self, _device_id: usize) -> Result<u64> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_all();

        #[cfg(feature = "metal")]
        let avail_mem = {
            let device = metal::Device::system_default().expect("No Metal device found");
            let max_mem = device.recommended_max_working_set_size();
            let alloc_mem = device.current_allocated_size();
            std::cmp::max(max_mem.saturating_sub(alloc_mem), sys.available_memory())
        };

        #[cfg(not(feature = "metal"))]
        let avail_mem = sys.available_memory();

        // Apply kv_fraction only - reserved memory check is done post-allocation
        let usable = (avail_mem as f64 * self.kv_fraction) as u64;

        crate::log_warn!(
            "Memory: available {:.2} GB, kv_fraction {:.0}%, Max usable cache budget {:.2} GB",
            avail_mem as f64 / SIZE_IN_GB,
            self.kv_fraction * 100.0,
            usable as f64 / SIZE_IN_GB
        );

        Ok(usable)
    }

    /// Query available GPU memory across all given device_ids
    /// Returns the MINIMUM usable memory across all devices
    pub fn get_available_memory(&self, device_ids: &[usize]) -> Result<u64> {
        let mut min_memory: Option<u64> = None;

        for &device_id in device_ids {
            let mem = self.get_rank_available_memory(device_id)?;
            min_memory = Some(min_memory.map_or(mem, |m| std::cmp::min(m, mem)));
        }

        min_memory.ok_or_else(|| candle_core::Error::msg("No device IDs provided"))
    }

    /// Auto-decide optimal (max_num_seqs, max_model_len) within memory budget.
    ///
    /// When total KV tokens <= config_model_len, use all tokens as max_model_len
    /// with max_num_seqs=1 (maximize context for a single sequence).
    /// When total KV tokens > config_model_len, use the candidate list to pick
    /// how many concurrent sequences to support at various context lengths.
    fn auto_decide_params(
        &self,
        available_memory: u64,
    ) -> std::result::Result<(usize, usize), KVCacheError> {
        let per_block = self.per_block_bytes();
        let total_blocks = available_memory as usize / per_block;
        let total_tokens = total_blocks * self.block_size;

        if total_blocks == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        if total_tokens <= self.config_model_len {
            return Ok((1, total_tokens));
        }

        // More KV capacity than one full context — use candidates to decide
        // how many concurrent sequences to support.
        let candidates = [
            self.config_model_len,
            self.config_model_len / 2,
            self.config_model_len / 4,
            self.config_model_len / 8,
            16 * 1024,
            8 * 1024,
            4 * 1024,
            1024,
        ];

        for &max_len in candidates.iter() {
            if max_len == 0 {
                continue;
            }
            let blocks_per_seq = (max_len + self.block_size - 1) / self.block_size;
            if total_blocks >= blocks_per_seq {
                let max_possible_seqs = total_blocks / blocks_per_seq;
                let max_seqs = std::cmp::min(max_possible_seqs, 8);
                return Ok((max_seqs, max_len));
            }
        }

        // Should not reach here since total_tokens > config_model_len,
        // but handle gracefully.
        Ok((1, total_tokens))
    }

    /// Calculate allocation plan given the minimum available memory across ranks
    /// This is the main entry point - call AFTER collecting all rank memory reports
    pub fn plan_allocation(
        &self,
        min_available_memory: u64,
        num_shards: usize,
    ) -> std::result::Result<KVCacheAllocation, KVCacheError> {
        let per_block = self.per_block_bytes();
        if per_block == 0 {
            return Err(KVCacheError::InvalidConfiguration {
                message: format!(
                    "Invalid KVCache block size: per_block=0 (num_kv_heads={}, num_shards={}, head_dim={}, block_size={}, dtype_size={}, num_kv_layers={})",
                    self.num_kv_heads,
                    self.num_shards,
                    self.head_dim,
                    self.block_size,
                    self.dtype_size,
                    self.num_kv_layers
                ),
            });
        }

        let (max_num_seqs, max_model_len) = if let (Some(max_num_seqs), Some(max_model_len)) =
            (self.user_max_num_seqs, self.user_max_model_len)
        {
            let required_bytes = self.calculate_required_memory(max_num_seqs, max_model_len);
            if required_bytes as u64 > min_available_memory {
                return Err(KVCacheError::InsufficientGpuMemory {
                    available_mb: min_available_memory as f64 / SIZE_IN_MB,
                    required_mb: required_bytes as f64 / SIZE_IN_MB,
                    reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
                });
            }
            (max_num_seqs, max_model_len)
        } else {
            self.auto_decide_params(min_available_memory)?
        };

        // Always allocate blocks from ALL available memory.
        // max_num_seqs and max_model_len are scheduling limits enforced at runtime,
        // not memory reservations. Using the full budget lets a single sequence use
        // up to max_model_len tokens from the shared pool, rather than artificially
        // capping the pool to max_num_seqs * max_model_len.
        let num_gpu_blocks = min_available_memory as usize / per_block;

        if num_gpu_blocks == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: min_available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        // Max usable KVCache tokens = num_blocks * block_size
        let max_num_batched_tokens = num_gpu_blocks * self.block_size;
        let kvcache_memory_bytes = num_gpu_blocks * per_block;

        // CPU blocks for swap
        #[cfg(feature = "cuda")]
        let num_cpu_blocks = (num_gpu_blocks as f32 * self.cpu_mem_fold) as usize;
        #[cfg(not(feature = "cuda"))]
        let num_cpu_blocks = 1;

        // Final validation
        if num_gpu_blocks == 0 || max_num_seqs == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: min_available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        let allocation = KVCacheAllocation {
            num_gpu_blocks,
            num_cpu_blocks,
            max_num_seqs,
            max_model_len,
            kvcache_memory_bytes,
            max_num_batched_tokens,
        };

        crate::log_warn!(
            "KVCache Allocation: {} GPU blocks ({:.2} GB x {}), max usable kvcache tokens {} ({}k bytes per token), scheduling limits [{} seqs x {} tokens]",
            num_gpu_blocks,
            kvcache_memory_bytes as f64 / SIZE_IN_GB,
            num_shards,
            max_num_batched_tokens,
            per_block / 1024 / self.block_size,
            max_num_seqs,
            max_model_len
        );

        Ok(allocation)
    }

    /// Apply allocation result to EngineConfig
    pub fn apply_to_config(&self, allocation: &KVCacheAllocation, econfig: &mut EngineConfig) {
        econfig.num_blocks = allocation.num_gpu_blocks;
        econfig.max_num_seqs = allocation.max_num_seqs;
        econfig.max_model_len = Some(allocation.max_model_len);
        econfig.kvcache_memory_bytes = allocation.kvcache_memory_bytes;
        econfig.max_num_batched_tokens = allocation.max_num_batched_tokens;
    }

    /// Check if auto-decide mode is needed
    pub fn needs_auto_decide(&self) -> bool {
        self.user_max_model_len.is_none()
    }

    //==========================================================================
    // Tensor Allocation Methods
    //==========================================================================

    /// Initialize KV cache tensors on GPU and CPU
    ///
    /// # Arguments
    /// * `allocation` - The allocation plan from `plan_allocation()`
    /// * `dtype` - Data type for the cache (will use U8 for FP8)
    /// * `device` - The GPU device to allocate on
    /// * `pd_config` - Optional P/D config for sync allocation
    ///
    /// # Returns
    /// Tuple of (GPU KV cache, CPU KV cache) - each is a Vec of (key_tensor, value_tensor) per layer
    pub fn init_kv_cache(
        &self,
        allocation: &KVCacheAllocation,
        dtype: DType,
        device: &Device,
        pd_config: Option<&crate::transfer::PdConfig>,
    ) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
        let num_gpu_blocks = allocation.num_gpu_blocks;
        let num_cpu_blocks = allocation.num_cpu_blocks;

        #[cfg(not(feature = "cuda"))]
        let sync_alloc = true;

        #[allow(unused)]
        #[cfg(feature = "cuda")]
        let sync_alloc = if let Some(p_cfg) = pd_config {
            matches!(p_cfg.role, crate::transfer::PdRole::Server)
        } else {
            false
        };

        #[cfg(not(feature = "cuda"))]
        let _ = pd_config;

        #[cfg(all(feature = "flashattn", not(feature = "flashinfer"), feature = "cuda"))]
        if self.kvcache_dtype.is_fp8_keys() {
            let sm = device
                .as_cuda_device()
                .ok()
                .and_then(|d| attention_rs::cuda_utils::sm_version(d))
                .unwrap_or(0);
            if sm != 90 {
                tracing::warn!(
                    "FP8 KV cache with FlashAttention requires SM90 (Hopper), \
                     detected SM{sm}. Will fall back to native flash kernels.",
                    sm = sm,
                );
            }
        }

        let cache_dtype = if self.kvcache_dtype.is_fp8_keys() {
            DType::U8
        } else {
            dtype
        };
        if self.kvcache_dtype.is_turboquant() {
            crate::log_warn!(
                "TurboQuant mode: {:?}, standard cache dtype {:?} (dummy for turbo4/turbo3)",
                self.kvcache_dtype,
                cache_dtype
            );
        } else {
            crate::log_warn!(
                "KV cache dtype: {:?}, cache dtype {:?}",
                self.kvcache_dtype,
                cache_dtype
            );
        }

        if self.is_mla {
            // MLA cache: (ckv_cache, kpe_cache) per layer
            // ckv_cache: [num_blocks, block_size, 1, kv_lora_rank]
            // kpe_cache: [num_blocks, block_size, 1, qk_rope_head_dim]
            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for _ in 0..self.num_kv_layers {
                let ckv_blocks = Tensor::empty(
                    (num_gpu_blocks, self.block_size, 1, self.mla_kv_lora_rank),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let kpe_blocks = Tensor::empty(
                    (
                        num_gpu_blocks,
                        self.block_size,
                        1,
                        self.mla_qk_rope_head_dim,
                    ),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((ckv_blocks, kpe_blocks));
            }
            for _ in 0..self.num_kv_layers {
                let ckv_blocks = Tensor::zeros(
                    (num_cpu_blocks, self.block_size, 1, self.mla_kv_lora_rank),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let kpe_blocks = Tensor::zeros(
                    (
                        num_cpu_blocks,
                        self.block_size,
                        1,
                        self.mla_qk_rope_head_dim,
                    ),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((ckv_blocks, kpe_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        } else if cfg!(feature = "flash")
            || cfg!(feature = "flashinfer")
            || cfg!(feature = "flashattn")
            || cfg!(feature = "metal")
        {
            // For turbo4/turbo3, the standard K/V cache is unused — all data lives
            // in TQ buffers. Allocate 1-block dummies so Option<Tensor> remains Some
            // (the forward path checks is_some to trigger store dispatch).
            let tq_full = matches!(
                self.kvcache_dtype,
                crate::utils::config::KvCacheDtype::Turbo4
                    | crate::utils::config::KvCacheDtype::Turbo3
            );
            let std_gpu_blocks = if tq_full { 1 } else { num_gpu_blocks };
            let std_cpu_blocks = if tq_full { 1 } else { num_cpu_blocks };

            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for layer_idx in 0..self.num_kv_layers {
                let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
                let key_blocks = Tensor::empty(
                    (std_gpu_blocks, self.block_size, kv_heads, hd),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (std_gpu_blocks, self.block_size, kv_heads, hd),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for layer_idx in 0..self.num_kv_layers {
                let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
                let key_blocks = Tensor::zeros(
                    (std_cpu_blocks, self.block_size, kv_heads, hd),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (std_cpu_blocks, self.block_size, kv_heads, hd),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }

            if self.kvcache_dtype.is_turboquant() {
                self.init_turboquant_cache(num_gpu_blocks, device, sync_alloc)?;
            }

            Ok((gpu_cache, cpu_cache))
        } else {
            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for layer_idx in 0..self.num_kv_layers {
                let kshape = self.layer_key_block_shape(layer_idx, cache_dtype);
                let vshape = self.layer_value_block_shape(layer_idx);
                let key_blocks = Tensor::empty(
                    (num_gpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (num_gpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for layer_idx in 0..self.num_kv_layers {
                let kshape = self.layer_key_block_shape(layer_idx, cache_dtype);
                let vshape = self.layer_value_block_shape(layer_idx);
                let key_blocks = Tensor::zeros(
                    (num_cpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_cpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        }
    }

    fn init_turboquant_cache(
        &self,
        num_gpu_blocks: usize,
        device: &Device,
        sync_alloc: bool,
    ) -> candle_core::Result<()> {
        use crate::utils::config::KvCacheDtype;

        let tq_mode = match self.kvcache_dtype {
            KvCacheDtype::Turbo8 => attention_rs::TurboquantMode::Turbo8,
            KvCacheDtype::Turbo4 => attention_rs::TurboquantMode::Turbo4,
            KvCacheDtype::Turbo3 => attention_rs::TurboquantMode::Turbo3,
            _ => return Ok(()),
        };

        let mut tq_layers = Vec::new();
        for layer_idx in 0..self.num_kv_layers {
            let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
            let bs = self.block_size;

            let v_absmax = Tensor::empty(
                (num_gpu_blocks, bs, kv_heads),
                candle_core::DType::F32,
                device,
                Some(sync_alloc),
            )?;
            let v_quant = Tensor::empty(
                (num_gpu_blocks, bs, kv_heads, hd / 2),
                candle_core::DType::U8,
                device,
                Some(sync_alloc),
            )?;

            let (k_absmax, k_quant) = match tq_mode {
                attention_rs::TurboquantMode::Turbo4 => {
                    let ka = Tensor::empty(
                        (num_gpu_blocks, bs, kv_heads),
                        candle_core::DType::F32,
                        device,
                        Some(sync_alloc),
                    )?;
                    let kq = Tensor::empty(
                        (num_gpu_blocks, bs, kv_heads, hd / 2),
                        candle_core::DType::U8,
                        device,
                        Some(sync_alloc),
                    )?;
                    (Some(ka), Some(kq))
                }
                attention_rs::TurboquantMode::Turbo3 => {
                    let ka = Tensor::empty(
                        (num_gpu_blocks, bs, kv_heads),
                        candle_core::DType::F32,
                        device,
                        Some(sync_alloc),
                    )?;
                    let k_bytes_per_head = (hd * 3 + 7) / 8;
                    let kq = Tensor::empty(
                        (num_gpu_blocks, bs, kv_heads, k_bytes_per_head),
                        candle_core::DType::U8,
                        device,
                        Some(sync_alloc),
                    )?;
                    (Some(ka), Some(kq))
                }
                _ => (None, None),
            };

            tq_layers.push(attention_rs::TurboquantLayerCache {
                k_absmax,
                k_quant,
                v_absmax,
                v_quant,
            });
        }

        crate::log_warn!(
            "Initialized TurboQuant {} cache: {} layers, {} blocks",
            self.kvcache_dtype,
            self.num_kv_layers,
            num_gpu_blocks,
        );
        attention_rs::init_turboquant_cache(tq_mode, tq_layers, self.block_size);
        Ok(())
    }

    /// Allocate CPU-side TQ buffers for swap. Returns None if not TQ mode.
    pub fn init_cpu_tq_cache(
        &self,
        num_cpu_blocks: usize,
    ) -> candle_core::Result<Option<Vec<crate::core::runner::CpuTqLayerCache>>> {
        use crate::utils::config::KvCacheDtype;
        let tq_mode = match self.kvcache_dtype {
            KvCacheDtype::Turbo8 => attention_rs::TurboquantMode::Turbo8,
            KvCacheDtype::Turbo4 => attention_rs::TurboquantMode::Turbo4,
            KvCacheDtype::Turbo3 => attention_rs::TurboquantMode::Turbo3,
            _ => return Ok(None),
        };

        if num_cpu_blocks == 0 {
            return Ok(None);
        }

        let mut cpu_tq = Vec::new();
        for layer_idx in 0..self.num_kv_layers {
            let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
            let bs = self.block_size;

            let v_absmax = Tensor::zeros(
                (num_cpu_blocks, bs, kv_heads),
                candle_core::DType::F32,
                &Device::Cpu,
            )?;
            let v_quant = Tensor::zeros(
                (num_cpu_blocks, bs, kv_heads, hd / 2),
                candle_core::DType::U8,
                &Device::Cpu,
            )?;

            let (k_absmax, k_quant) = match tq_mode {
                attention_rs::TurboquantMode::Turbo4 => {
                    let ka = Tensor::zeros(
                        (num_cpu_blocks, bs, kv_heads),
                        candle_core::DType::F32,
                        &Device::Cpu,
                    )?;
                    let kq = Tensor::zeros(
                        (num_cpu_blocks, bs, kv_heads, hd / 2),
                        candle_core::DType::U8,
                        &Device::Cpu,
                    )?;
                    (Some(ka), Some(kq))
                }
                attention_rs::TurboquantMode::Turbo3 => {
                    let ka = Tensor::zeros(
                        (num_cpu_blocks, bs, kv_heads),
                        candle_core::DType::F32,
                        &Device::Cpu,
                    )?;
                    let k_bytes_per_head = (hd * 3 + 7) / 8;
                    let kq = Tensor::zeros(
                        (num_cpu_blocks, bs, kv_heads, k_bytes_per_head),
                        candle_core::DType::U8,
                        &Device::Cpu,
                    )?;
                    (Some(ka), Some(kq))
                }
                _ => (None, None),
            };

            cpu_tq.push(crate::core::runner::CpuTqLayerCache {
                k_absmax,
                k_quant,
                v_absmax,
                v_quant,
            });
        }

        crate::log_warn!(
            "Initialized CPU TurboQuant {} swap cache: {} layers, {} blocks",
            self.kvcache_dtype,
            self.num_kv_layers,
            num_cpu_blocks,
        );
        Ok(Some(cpu_tq))
    }
}
