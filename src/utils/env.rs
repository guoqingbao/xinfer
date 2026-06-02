use std::env;
use std::sync::OnceLock;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "XINFER_STREAM_AS_REASONING_CONTENT";

static STREAM_AS_REASONING_CONTENT: OnceLock<bool> = OnceLock::new();

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
