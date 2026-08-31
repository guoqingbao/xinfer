//! Speculative decoding: built-in MTP heads and external DFlash2 draft models.

pub mod dflash;
pub mod mamba;
pub mod metadata;
pub mod mtp;
pub mod verify;

pub use dflash::DFlashDrafter;
pub use metadata::SpecSeqInfo;
pub use verify::{
    dflash_stats_reset, dflash_stats_summary, mtp_stats_acceptance_rate,
    mtp_stats_avg_tokens_per_step, mtp_stats_reset, mtp_stats_summary, mtp_stats_update,
    verify_draft_greedy, MtpVerifyResult, SPEC_STATS_WINDOW_STEPS,
};

use std::path::Path;

use crate::utils::config::EngineConfig;

/// Whether speculative decoding uses an external DFlash2 draft model.
pub fn uses_dflash(econfig: &EngineConfig) -> bool {
    econfig
        .draft_model
        .as_ref()
        .is_some_and(|model| !model.is_empty())
}

/// Whether speculative decoding uses the target model's built-in MTP head.
pub fn uses_builtin_mtp(econfig: &EngineConfig) -> bool {
    econfig.num_speculative_tokens.unwrap_or(0) > 0 && !uses_dflash(econfig)
}

/// Resolve `--draft-model` into HuggingFace id vs local directory, mirroring main model args.
pub fn resolve_draft_model(draft_model: &str) -> (Option<String>, Option<String>) {
    if Path::new(draft_model).exists() {
        (None, Some(draft_model.to_string()))
    } else {
        (Some(draft_model.to_string()), None)
    }
}
