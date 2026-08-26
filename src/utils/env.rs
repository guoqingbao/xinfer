use std::env;
use std::sync::OnceLock;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "XINFER_STREAM_AS_REASONING_CONTENT";

pub const SM90_LOWER_PRECISION_GDN_PREFILL_ENV: &str = "SM90_LOWER_PRECISION_GDN_PREFILL";

static STREAM_AS_REASONING_CONTENT: OnceLock<bool> = OnceLock::new();
static SM90_LOWER_PRECISION_GDN_PREFILL: OnceLock<bool> = OnceLock::new();

pub fn sm90_lower_precision_gdn_prefill() -> bool {
    *SM90_LOWER_PRECISION_GDN_PREFILL.get_or_init(|| {
        env::var(SM90_LOWER_PRECISION_GDN_PREFILL_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "TRUE" | "YES"))
            .unwrap_or(false)
    })
}

pub fn stream_as_reasoning_content() -> bool {
    *STREAM_AS_REASONING_CONTENT.get_or_init(|| {
        env::var(STREAM_AS_REASONING_CONTENT_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

pub fn mamba_snapshot_block_stride_blocks(default: usize) -> usize {
    let default = default.max(1);
    let Ok(raw) = env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            crate::log_warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            crate::log_warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                raw,
                default
            );
            default
        }
    }
}

pub const DEFAULT_REASONING_MAX_TOKENS_ENV: &str = "XINFER_DEFAULT_REASONING_MAX_TOKENS";
pub const DEFAULT_REASONING_MAX_TOKENS_VALUE: usize = 512;

static DEFAULT_REASONING_MAX_TOKENS: OnceLock<usize> = OnceLock::new();

pub fn default_reasoning_max_tokens() -> usize {
    *DEFAULT_REASONING_MAX_TOKENS.get_or_init(|| {
        env::var(DEFAULT_REASONING_MAX_TOKENS_ENV)
            .map(|raw| {
                raw.trim()
                    .parse::<usize>()
                    .map(|n| {
                        if n == 0 {
                            DEFAULT_REASONING_MAX_TOKENS_VALUE
                        } else {
                            n
                        }
                    })
                    .unwrap_or(DEFAULT_REASONING_MAX_TOKENS_VALUE)
            })
            .unwrap_or(DEFAULT_REASONING_MAX_TOKENS_VALUE)
    })
}

/// Environment variable to disable soft masking for gradient smoothing.
/// When NOT set: soft masking is ENABLED (default behavior).
/// When set to "1", "true", or "yes": soft masking is DISABLED (hard -inf masking).
/// When set to "0", "false", or "no": soft masking is ENABLED.
pub const SOFT_MASK_DISABLED_ENV: &str = "XINFER_SOFT_MASK_DISABLED";

static SOFT_MASK_DISABLED: OnceLock<bool> = OnceLock::new();

pub fn soft_mask_disabled() -> bool {
    *SOFT_MASK_DISABLED.get_or_init(|| {
        env::var(SOFT_MASK_DISABLED_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(false)
    })
}

/// Debug: skip FF-token speculation (treat the grammar's forced run as empty) so DFlash/MTP can be
/// compared with and without the ff prefix on the same rev.
pub const SPEC_NO_FF_ENV: &str = "XINFER_SPEC_NO_FF";

static SPEC_NO_FF: OnceLock<bool> = OnceLock::new();

pub fn spec_no_ff() -> bool {
    *SPEC_NO_FF.get_or_init(|| {
        env::var(SPEC_NO_FF_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Debug: use the granular (per-position FSM-walk) draft mask instead of the batched single-VOB
/// mask (3a). For precise gating when the mask changes across the draft run.
pub const SPEC_GRANULAR_MASK_ENV: &str = "XINFER_SPEC_GRANULAR_MASK";

static SPEC_GRANULAR_MASK: OnceLock<bool> = OnceLock::new();

pub fn spec_granular_mask() -> bool {
    *SPEC_GRANULAR_MASK.get_or_init(|| {
        env::var(SPEC_GRANULAR_MASK_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// DFlash draft backend selector (`XINFER_DFLASH_BACKEND`).
/// - `auto` (default): use the fused attention-rs CUDA kernels when the draft checkpoint
///   carries DFlash2 components (candidate selector / grouped convs) AND the `cuda` feature
///   is built; otherwise the portable candle path.
/// - `v2`: force the fused kernels (no-op on non-CUDA builds, which fall back to candle).
/// - `v1`: force the portable candle implementation.
pub const DFLASH_BACKEND_ENV: &str = "XINFER_DFLASH_BACKEND";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashBackend {
    Auto,
    V1,
    V2,
}

static DFLASH_BACKEND: OnceLock<DflashBackend> = OnceLock::new();

pub fn dflash_backend() -> DflashBackend {
    *DFLASH_BACKEND.get_or_init(|| {
        match env::var(DFLASH_BACKEND_ENV).ok().as_deref() {
            Some(v) if v.trim().eq_ignore_ascii_case("v1") || v.trim() == "1" => DflashBackend::V1,
            Some(v) if v.trim().eq_ignore_ascii_case("v2") || v.trim() == "2" => DflashBackend::V2,
            _ => DflashBackend::Auto,
        }
    })
}

/// Whether the fused DFlash2 CUDA-kernel backend is active. `auto` (the default) enables the
/// kernels on CUDA builds; DFlash1 checkpoints are unaffected because they carry no DFlash2
/// components (selector / convs), so the kernel paths simply never run for them.
pub fn dflash_use_kernels() -> bool {
    match dflash_backend() {
        DflashBackend::V1 => false,
        DflashBackend::V2 => cfg!(feature = "cuda"),
        DflashBackend::Auto => cfg!(feature = "cuda"),
    }
}
