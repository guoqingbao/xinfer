// src/core/mtp.rs
// Multi-Token Prediction (MTP) speculative decoding support.
//
// MTP uses lightweight prediction heads built into the model (e.g. Qwen3.5, DeepSeek-V3)
// to draft future tokens using the backbone's hidden states and KV cache.
// Accepted draft tokens are verified in a single target-model forward pass.

use candle_core::{Result, Tensor, D};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for MTP speculative decoding at the engine level.
#[derive(Debug, Clone)]
pub struct MtpEngineConfig {
    /// Number of speculative tokens to propose per step.
    pub num_speculative_tokens: usize,
}

impl MtpEngineConfig {
    pub fn new(num_speculative_tokens: usize) -> Self {
        Self {
            num_speculative_tokens: num_speculative_tokens.max(1),
        }
    }
}

/// Outcome of MTP verification for a single sequence.
#[derive(Debug, Clone)]
pub struct MtpVerifyResult {
    /// All accepted tokens (draft tokens that matched the target model).
    pub accepted_tokens: Vec<u32>,
    /// The continuation token sampled from the first rejection point.
    pub continuation_token: u32,
    /// How many of the proposed drafts were accepted.
    pub num_accepted: usize,
    /// Total number proposed.
    pub num_proposed: usize,
}

/// Verify draft tokens against target model logits (greedy / argmax).
///
/// Uses a single batched argmax over all rows + vectorized comparison on GPU,
/// then transfers results to CPU in one shot.
///
/// `verify_logits`: shape [N+1, vocab_size] where N = len(draft_tokens).
///   - Position 0 predicts draft_tokens[0]
///   - Position i predicts draft_tokens[i] (for i < N)
///   - Position N provides the continuation token after last accepted draft
pub fn verify_draft_greedy(
    verify_logits: &Tensor,
    draft_tokens: &[u32],
) -> Result<MtpVerifyResult> {
    let num_positions = verify_logits.dim(0)?;
    let num_proposed = draft_tokens.len();

    if num_positions == 0 || num_proposed == 0 {
        let first_token = if num_positions > 0 {
            verify_logits
                .get(0)?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?
        } else {
            0
        };
        return Ok(MtpVerifyResult {
            accepted_tokens: vec![],
            continuation_token: first_token,
            num_accepted: 0,
            num_proposed,
        });
    }

    // Keep verifier argmax aligned with the normal sampler path, which promotes
    // logits to F32 before selecting tokens.
    let verify_logits = verify_logits.to_dtype(candle_core::DType::F32)?;
    let all_target_tokens = verify_logits.argmax(D::Minus1)?;
    let target_vec: Vec<u32> = all_target_tokens.to_vec1()?;

    let compare_len = num_proposed.min(num_positions);
    let mut num_accepted = 0;
    for i in 0..compare_len {
        if target_vec[i] == draft_tokens[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    let accepted_tokens = draft_tokens[..num_accepted].to_vec();
    let continuation_token = if num_accepted < num_positions {
        target_vec[num_accepted]
    } else {
        target_vec[num_positions - 1]
    };

    Ok(MtpVerifyResult {
        accepted_tokens,
        continuation_token,
        num_accepted,
        num_proposed,
    })
}

/// Global MTP statistics tracker.
pub static MTP_TOTAL_PROPOSED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_ACCEPTED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_STEPS: AtomicUsize = AtomicUsize::new(0);

pub fn mtp_stats_update(proposed: usize, accepted: usize) {
    MTP_TOTAL_PROPOSED.fetch_add(proposed, Ordering::Relaxed);
    MTP_TOTAL_ACCEPTED.fetch_add(accepted, Ordering::Relaxed);
    MTP_TOTAL_STEPS.fetch_add(1, Ordering::Relaxed);
}

pub fn mtp_stats_acceptance_rate() -> f64 {
    let proposed = MTP_TOTAL_PROPOSED.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    }
}

pub fn mtp_stats_avg_tokens_per_step() -> f64 {
    let steps = MTP_TOTAL_STEPS.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    if steps == 0 {
        1.0
    } else {
        // Each step produces: 1 anchor + accepted drafts + 1 continuation
        (accepted + 2 * steps) as f64 / steps as f64
    }
}

pub fn mtp_stats_summary() -> String {
    let proposed = MTP_TOTAL_PROPOSED.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    let steps = MTP_TOTAL_STEPS.load(Ordering::Relaxed);
    format!(
        "MTP Stats: proposed={}, accepted={}, acceptance_rate={:.2}%, avg_tokens/step={:.2}",
        proposed,
        accepted,
        if proposed > 0 {
            accepted as f64 / proposed as f64 * 100.0
        } else {
            0.0
        },
        if steps > 0 {
            (accepted + 2 * steps) as f64 / steps as f64
        } else {
            1.0
        },
    )
}

pub fn mtp_stats_reset() {
    MTP_TOTAL_PROPOSED.store(0, Ordering::Relaxed);
    MTP_TOTAL_ACCEPTED.store(0, Ordering::Relaxed);
    MTP_TOTAL_STEPS.store(0, Ordering::Relaxed);
}
