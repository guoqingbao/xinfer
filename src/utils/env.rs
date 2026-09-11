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

/// `XINFER_STATE_COMPRESS=1` enables zstd compression of the inference state.
pub const STATE_COMPRESS_ENV: &str = "XINFER_STATE_COMPRESS";

static STATE_COMPRESS: OnceLock<bool> = OnceLock::new();

pub fn state_compress() -> bool {
    *STATE_COMPRESS.get_or_init(|| {
        env::var(STATE_COMPRESS_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_STATE_KEY` (a raw key) enables AES-256-GCM encryption of persisted
/// inference state. Returns the raw key bytes when set, else None.
pub const STATE_KEY_ENV: &str = "XINFER_STATE_KEY";

pub fn state_key() -> Option<Vec<u8>> {
    env::var(STATE_KEY_ENV).ok().filter(|v| !v.trim().is_empty()).map(|v| v.into_bytes())
}

/// `XINFER_STATESTORE_STRICT=1` rejects a warm-loaded state whose version stamp
/// (runtime version + model ID + dtype + block size) does not match the current
/// runtime. When off (default), a mismatched state is loaded with a warning.
pub const STATESTORE_STRICT_ENV: &str = "XINFER_STATESTORE_STRICT";

static STATESTORE_STRICT: OnceLock<bool> = OnceLock::new();

pub fn statestore_strict() -> bool {
    *STATESTORE_STRICT.get_or_init(|| {
        env::var(STATESTORE_STRICT_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_STATE_TTL_MS` (default 7 days): the max age of a persisted state before
/// it is pruned.
pub const STATE_TTL_MS_ENV: &str = "XINFER_STATE_TTL_MS";
pub const DEFAULT_STATE_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1000;

pub fn state_ttl_ms() -> u64 {
    env::var(STATE_TTL_MS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_STATE_TTL_MS)
}

/// `XINFER_STATE_MAX_BYTES` (default 8 GiB): the space watermark for the state
/// store. When the total size exceeds it, the oldest states are pruned.
pub const STATE_MAX_BYTES_ENV: &str = "XINFER_STATE_MAX_BYTES";
pub const DEFAULT_STATE_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024;

pub fn state_max_bytes() -> usize {
    env::var(STATE_MAX_BYTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_STATE_MAX_BYTES)
}
