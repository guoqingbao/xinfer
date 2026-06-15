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

pub const ADAPTIVE_TUNING_ENV: &str = "VLLM_RS_ADAPTIVE_TUNING";

// Use thread_local for testability - can be reset between tests
thread_local! {
    static ADAPTIVE_TUNING_ENABLED: std::cell::RefCell<Option<bool>> = std::cell::RefCell::new(None);
}

pub fn is_adaptive_tuning_enabled() -> bool {
    ADAPTIVE_TUNING_ENABLED.with(|cell| {
        let mut value = cell.borrow_mut();
        if let Some(v) = *value {
            return v;
        }
        
        // Read from environment and cache
        let result = env::var(ADAPTIVE_TUNING_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(false);
        
        *value = Some(result);
        result
    })
}

/// Reset the adaptive tuning cache (for testing)
pub fn reset_adaptive_tuning_cache() {
    ADAPTIVE_TUNING_ENABLED.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Convergence threshold environment variable
pub const BINARY_SEARCH_CONVERGENCE_THRESHOLD_ENV: &str = "VLLM_RS_BINARY_SEARCH_CONVERGENCE_THRESHOLD";
pub const DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD: f64 = 0.10; // 10% default

/// Get the binary search convergence threshold from environment
pub fn get_binary_search_convergence_threshold() -> f64 {
    let Ok(raw) = env::var(BINARY_SEARCH_CONVERGENCE_THRESHOLD_ENV) else {
        return DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD;
    };
    match raw.trim().parse::<f64>() {
        Ok(v) if v > 0.0 && v <= 1.0 => v,
        Ok(v) => {
            crate::log_warn!(
                "{} must be between 0.0 and 1.0, got {}. Using default {}.",
                BINARY_SEARCH_CONVERGENCE_THRESHOLD_ENV,
                v,
                DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD
            );
            DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD
        }
        Err(_) => {
            crate::log_warn!(
                "Invalid {}='{}'. Using default {}.",
                BINARY_SEARCH_CONVERGENCE_THRESHOLD_ENV,
                raw,
                DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD
            );
            DEFAULT_BINARY_SEARCH_CONVERGENCE_THRESHOLD
        }
    }
}
