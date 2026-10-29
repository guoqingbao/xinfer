use std::env;
use std::sync::OnceLock;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "XINFER_STREAM_AS_REASONING_CONTENT";

pub const SM90_LOWER_PRECISION_GDN_PREFILL_ENV: &str = "SM90_LOWER_PRECISION_GDN_PREFILL";
/// Select the GDN prefill algorithm: `auto`/`fast` uses the FlashInfer SM90
/// chunked kernel when the build and device support it, `exact` forces the
/// reference sequential recurrence.
pub const GDN_PREFILL_ENV: &str = "XINFER_GDN_PREFILL";

pub const MOE_OFFLOAD_ENV: &str = "XINFER_MOE_OFFLOAD";

static STREAM_AS_REASONING_CONTENT: OnceLock<bool> = OnceLock::new();
static MOE_OFFLOAD: OnceLock<Option<f32>> = OnceLock::new();

pub const MOE_OFFLOAD_PROFILE_ENV: &str = "XINFER_MOE_OFFLOAD_PROFILE";
static MOE_OFFLOAD_PROFILE: OnceLock<bool> = OnceLock::new();

/// Per-call timing breakdown for MoE expert offload (adds device syncs).
pub fn moe_offload_profile() -> bool {
    *MOE_OFFLOAD_PROFILE.get_or_init(|| {
        env::var(MOE_OFFLOAD_PROFILE_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Fraction (0, 1) of each MoE layer's experts kept on GPU; the rest stay in host RAM.
/// `auto` returns 0.0: layers start with the minimum slots and grow to fill free VRAM
/// after loading (see `moe_offload::budget`).
pub fn moe_offload_rate() -> Option<f32> {
    *MOE_OFFLOAD.get_or_init(|| {
        let raw = env::var(MOE_OFFLOAD_ENV).ok()?;
        match raw.trim().parse::<f32>() {
            _ if raw.trim().eq_ignore_ascii_case("auto") => Some(0.0),
            Ok(r) if r > 0.0 && r < 1.0 => Some(r),
            _ => {
                crate::log_warn!(
                    "Invalid {MOE_OFFLOAD_ENV}='{raw}', expected auto or 0 < value < 1. Offload disabled."
                );
                None
            }
        }
    })
}

pub fn moe_offload_auto() -> bool {
    moe_offload_rate() == Some(0.0)
}

pub const MOE_OFFLOAD_RESERVE_ENV: &str = "XINFER_MOE_OFFLOAD_RESERVE_GB";

/// VRAM (bytes) that `XINFER_MOE_OFFLOAD=auto` leaves free for the KV cache and workspaces.
pub fn moe_offload_reserve_bytes() -> usize {
    let gb = env::var(MOE_OFFLOAD_RESERVE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v >= 0.0)
        .unwrap_or(2.5);
    (gb * (1u64 << 30) as f64) as usize
}
static SM90_LOWER_PRECISION_GDN_PREFILL: OnceLock<bool> = OnceLock::new();

pub fn sm90_lower_precision_gdn_prefill() -> bool {
    *SM90_LOWER_PRECISION_GDN_PREFILL.get_or_init(|| {
        // New, explicit control wins. `auto`/`fast` enable the chunked
        // FlashInfer kernel (the launcher itself falls back when the device
        // or build lacks SM90 support). `exact` keeps the reference
        // recurrence. The legacy variable still overrides the default, so
        // existing deployments that set it keep their behavior.
        if let Ok(v) = env::var(GDN_PREFILL_ENV) {
            let v = v.trim().to_ascii_lowercase();
            return matches!(v.as_str(), "1" | "true" | "yes" | "fast" | "chunked");
        }
        // The chunked FlashInfer kernel is SM90-only and uses lower-precision
        // accumulation than the reference recurrence, so it stays opt-in.
        // It matches the reference to BF16 rounding on typical inputs (p50
        // diff 0, p99 ~0.3% of range) but the differential probe shows
        // non-exact states for some input distributions.
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
