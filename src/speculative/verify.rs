use candle_core::{Result, Tensor, D};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

/// Rolling window size for speculative acceptance stats (recent decode steps).
pub const SPEC_STATS_WINDOW_STEPS: usize = 256;

/// Outcome of speculative verification for a single sequence.
#[derive(Debug, Clone)]
pub struct MtpVerifyResult {
    pub accepted_tokens: Vec<u32>,
    pub continuation_token: u32,
    pub num_accepted: usize,
    pub num_proposed: usize,
    /// The drafts rejected by the target model (the num_proposed - the num_accepted).
    pub target_bound: usize,
    /// The drafts rejected by the grammar firewall (the PDA/grammar constraint).
    /// 0 for the plain verify (the grammar check is in the spec-decode path).
    pub grammar_bound: usize,
}

/// Validate draft tokens against a PDA. Returns the number of leading tokens
/// that are grammar-legal (stops at the first illegal token). If `pda` is None,
/// returns `tokens.len()` (no constraint). This is the CPU reference walk used
/// for transient recovery when a draft runs ahead of the grammar state.
pub fn pda_validate_draft(
    pda: Option<&pushdown_rs::machine::PdaMachine>,
    start_ctrl: u32,
    tokens: &[u32],
) -> usize {
    let Some(pda) = pda else { return tokens.len() };
    let mut ctrl = start_ctrl;
    let mut stack = vec![pda.start_stack];
    let mut count = 0;
    for &token in tokens {
        let top = stack.last().copied().unwrap_or(pda.start_stack);
        match pda.lookup(ctrl, Some(token), top).as_slice() {
            [t] => {
                // Advance: pop top, push the new symbols.
                stack.pop();
                for &p in t.push.iter().rev() {
                    stack.push(p);
                }
                ctrl = t.next_q;
                count += 1;
            }
            _ => break, // No transition: token is illegal.
        }
    }
    count
}

/// Verify draft tokens against target model logits (greedy / argmax).
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
            target_bound: num_proposed,
            grammar_bound: 0,
        });
    }

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
        target_bound: num_proposed - num_accepted,
        grammar_bound: 0,
    })
}

/// Verify draft tokens via LOSSLESS rejection sampling (the Leviathan algorithm,
/// the `XINFER_SPEC_REJECTION_SAMPLING` path).
///
/// Unlike `verify_draft_greedy` (the argmax accept, which BIASES the output
/// distribution when the target samples with temperature > 0), this honors the
/// target's sampled distribution:
/// - At each draft position `i`, accept the draft token `x` with probability
///   `min(1, p_target(x) / p_draft(x))`.
/// - On rejection, sample the continuation token from the RESIDUAL distribution
///   `max(0, p_target - p_draft)` (normalized) -- this makes the overall output
///   distribution EXACTLY the target's (the lossless property).
///
/// The `verify_probs` + `draft_probs` are the per-position Distributions (the
/// `[num_positions, vocab]` F32, the already-normalized probs -- the caller
/// computes `verify_probs` via the sampler's post-sampling transform + the
/// `draft_probs` via the drafter's logits softmax). The `seed` makes the Gumbel
/// draws reproducible.
pub fn verify_draft_rejection(
    verify_probs: &Tensor,
    draft_probs: &Tensor,
    draft_tokens: &[u32],
    seed: u64,
) -> Result<MtpVerifyResult> {
    let num_positions = verify_probs.dim(0)?;
    let num_proposed = draft_tokens.len();
    let compare_len = num_proposed.min(num_positions);

    let vflat: Vec<f32> = verify_probs
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let vocab = if num_positions > 0 {
        vflat.len() / num_positions
    } else {
        0
    };
    let dflat: Vec<f32> = draft_probs
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let drows = if vocab > 0 { dflat.len() / vocab } else { 0 };

    // The RNG for the residual Gumbel draws (the seed makes it reproducible).
    let mut rng_state = seed;
    let next_f32 = |state: &mut u64| -> f32 {
        // The splitmix64 (the no external dep, the deterministic per-seed stream).
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        (z as f32) / (u64::MAX as f32)
    };

    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(compare_len);
    let mut num_accepted = 0usize;
    let mut continuation_token = 0u32;

    for i in 0..compare_len {
        let draft_tok = draft_tokens[i] as usize;
        let v_row = &vflat[i * vocab..(i + 1) * vocab];
        let d_row = if i < drows {
            &dflat[i * vocab..(i + 1) * vocab]
        } else {
            &[]
        };
        let p_target = v_row.get(draft_tok).copied().unwrap_or(0.0);
        let p_draft = d_row.get(draft_tok).copied().unwrap_or(0.0);

        // The Leviathan accept: min(1, p_target / p_draft). The p_draft == 0 edge
        // (the drafter never proposed this token) -> the accept prob is 0 (the
        // reject, the residual draw).
        let accept_prob = if p_draft > 0.0 {
            (p_target / p_draft).min(1.0)
        } else {
            0.0
        };
        let u = next_f32(&mut rng_state);
        if u < accept_prob {
            accepted_tokens.push(draft_tokens[i]);
            num_accepted += 1;
        } else {
            // The reject: sample the continuation from the residual
            // max(0, p_target - p_draft), normalized. The Gumbel-max trick.
            let residual: Vec<f32> = (0..vocab)
                .map(|t| {
                    let pt = v_row.get(t).copied().unwrap_or(0.0);
                    let pd = d_row.get(t).copied().unwrap_or(0.0);
                    (pt - pd).max(0.0)
                })
                .collect();
            let total: f32 = residual.iter().sum();
            let mut cont = draft_tokens[i]; // the fallback (the no residual mass -> keep the draft)
            if total > 0.0 {
                let best = residual
                    .iter()
                    .enumerate()
                    .filter(|(_, &r)| r > 0.0)
                    .map(|(t, &r)| {
                        let g = r.ln() - (-next_f32(&mut rng_state).max(f32::EPSILON).ln());
                        (t, g)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((t, _)) = best {
                    cont = t as u32;
                }
            }
            continuation_token = cont;
            break;
        }
    }

    // The all-accepted case: the bonus token is the target's sample at the last
    // position (the p_target draw, the no residual).
    if num_accepted == compare_len && num_positions > 0 {
        let last_row = &vflat[(num_positions - 1) * vocab..num_positions * vocab];
        let total: f32 = last_row.iter().sum();
        let mut bonus = 0u32;
        if total > 0.0 {
            let best = last_row
                .iter()
                .enumerate()
                .filter(|(_, &p)| p > 0.0)
                .map(|(t, &p)| {
                    let g = p.ln() - (-next_f32(&mut rng_state).max(f32::EPSILON).ln());
                    (t, g)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((t, _)) = best {
                bonus = t as u32;
            }
        }
        continuation_token = bonus;
    }

    Ok(MtpVerifyResult {
        accepted_tokens,
        continuation_token,
        num_accepted,
        num_proposed,
        target_bound: num_proposed - num_accepted,
        grammar_bound: 0,
    })
}

struct WindowedAcceptanceStats {
    label: &'static str,
    window_capacity: usize,
    log_every: usize,
    ring: VecDeque<(usize, usize)>,
    total_steps: usize,
}

impl WindowedAcceptanceStats {
    fn new(label: &'static str, window_capacity: usize, log_every: usize) -> Self {
        Self {
            label,
            window_capacity,
            log_every,
            ring: VecDeque::with_capacity(window_capacity),
            total_steps: 0,
        }
    }

    fn record(&mut self, proposed: usize, accepted: usize) -> bool {
        if self.ring.len() >= self.window_capacity {
            self.ring.pop_front();
        }
        self.ring.push_back((proposed, accepted));
        self.total_steps += 1;
        self.total_steps % self.log_every == 0
    }

    fn filled_window_steps(&self) -> usize {
        self.ring.len()
    }

    fn window_proposed(&self) -> usize {
        self.ring.iter().map(|(proposed, _)| *proposed).sum()
    }

    fn window_accepted(&self) -> usize {
        self.ring.iter().map(|(_, accepted)| *accepted).sum()
    }

    fn acceptance_rate(&self) -> f64 {
        let proposed = self.window_proposed();
        if proposed == 0 {
            0.0
        } else {
            self.window_accepted() as f64 / proposed as f64
        }
    }

    fn avg_tokens_per_step(&self) -> f64 {
        let steps = self.filled_window_steps();
        if steps == 0 {
            1.0
        } else {
            let accepted = self.window_accepted();
            (accepted + 2 * steps) as f64 / steps as f64
        }
    }

    fn format_summary(&self) -> String {
        let filled = self.filled_window_steps();
        let proposed = self.window_proposed();
        let accepted = self.window_accepted();
        format!(
            "{} Stats: total_steps={}, window={}/{} steps, window_proposed={}, window_accepted={}, acceptance_rate={:.2}%, avg_tokens/step={:.2}",
            self.label,
            self.total_steps,
            filled,
            self.window_capacity,
            proposed,
            accepted,
            self.acceptance_rate() * 100.0,
            self.avg_tokens_per_step(),
        )
    }

    fn reset(&mut self) {
        self.ring.clear();
        self.total_steps = 0;
    }
}

fn mtp_stats_state() -> &'static Mutex<WindowedAcceptanceStats> {
    static STATS: OnceLock<Mutex<WindowedAcceptanceStats>> = OnceLock::new();
    STATS.get_or_init(|| {
        Mutex::new(WindowedAcceptanceStats::new(
            "MTP",
            SPEC_STATS_WINDOW_STEPS,
            SPEC_STATS_WINDOW_STEPS,
        ))
    })
}

fn dflash_stats_state() -> &'static Mutex<WindowedAcceptanceStats> {
    static STATS: OnceLock<Mutex<WindowedAcceptanceStats>> = OnceLock::new();
    STATS.get_or_init(|| {
        Mutex::new(WindowedAcceptanceStats::new(
            "DFlash2",
            SPEC_STATS_WINDOW_STEPS,
            SPEC_STATS_WINDOW_STEPS,
        ))
    })
}

/// Record one MTP verify step. Returns true when a windowed summary should be logged.
pub fn mtp_stats_update(proposed: usize, accepted: usize) -> bool {
    mtp_stats_state()
        .lock()
        .expect("MTP stats lock poisoned")
        .record(proposed, accepted)
}

pub fn mtp_stats_acceptance_rate() -> f64 {
    mtp_stats_state()
        .lock()
        .expect("MTP stats lock poisoned")
        .acceptance_rate()
}

pub fn mtp_stats_avg_tokens_per_step() -> f64 {
    mtp_stats_state()
        .lock()
        .expect("MTP stats lock poisoned")
        .avg_tokens_per_step()
}

pub fn mtp_stats_summary() -> String {
    mtp_stats_state()
        .lock()
        .expect("MTP stats lock poisoned")
        .format_summary()
}

pub fn mtp_stats_reset() {
    mtp_stats_state()
        .lock()
        .expect("MTP stats lock poisoned")
        .reset();
}

/// Record one DFlash2 verify step. Returns true when a windowed summary should be logged.
pub fn dflash_stats_update(proposed: usize, accepted: usize) -> bool {
    dflash_stats_state()
        .lock()
        .expect("DFlash2 stats lock poisoned")
        .record(proposed, accepted)
}

pub fn dflash_stats_summary() -> String {
    dflash_stats_state()
        .lock()
        .expect("DFlash2 stats lock poisoned")
        .format_summary()
}

pub fn dflash_stats_reset() {
    dflash_stats_state()
        .lock()
        .expect("DFlash2 stats lock poisoned")
        .reset();
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    /// The rejection-sampling accept math (the Leviathan `min(1, p/q)` + the
    /// residual draw). The inclusive: when p_target >= p_draft the accept prob is
    /// 1 (the draft is always kept). The exclusive: when p_target < p_draft the
    /// accept prob is the ratio (the draft is kept with that probability, else the
    /// residual supplies the token).
    #[test]
    fn rejection_accept_prob_is_min_one_ratio() {
        // The p_target == p_draft (the drafter matches the target): the accept prob
        // is 1 (the draft is always kept, the no residual).
        let p_target: f32 = 0.5;
        let p_draft: f32 = 0.5;
        let accept = (p_target / p_draft).min(1.0);
        assert_eq!(accept, 1.0, "the equal distributions accept with prob 1");
        // The p_target < p_draft (the target less likely than the drafter): the
        // accept prob is the ratio (the < 1, the residual covers the gap).
        let p_target: f32 = 0.2;
        let p_draft: f32 = 0.5;
        let accept = (p_target / p_draft).min(1.0);
        assert!((accept - 0.4).abs() < 1e-6, "the accept prob is the p/q ratio");
        // The p_draft == 0 (the drafter never proposed the token): the accept prob
        // is 0 (the always-reject, the residual draw).
        let p_draft: f32 = 0.0;
        let accept = if p_draft > 0.0 {
            (0.5f32 / p_draft).min(1.0)
        } else {
            0.0f32
        };
        assert_eq!(accept, 0.0, "the no-draft-prob -> the no accept (the residual)");
    }

    /// The end-to-end rejection verify: a target that ALWAYS prefers the draft token
    // (the p_target(draft) = 1.0, the p_draft(draft) = 1.0) accepts the full draft +
    // the bonus is the target's argmax. The no residual draw (the distributions
    // agree).
    #[test]
    fn rejection_verify_full_accept_when_distributions_agree() {
        let vocab = 4;
        // The target probs: each position puts all mass on the draft token (the
        // agreement). The draft probs: the same.
        let draft_tokens = vec![1u32, 2, 3];
        let num_positions = 4; // the 3 drafts + the 1 bonus
        let mut target = vec![0.0f32; num_positions * vocab];
        let mut draft = vec![0.0f32; num_positions * vocab];
        for (i, &tok) in draft_tokens.iter().enumerate() {
            target[i * vocab + tok as usize] = 1.0;
            draft[i * vocab + tok as usize] = 1.0;
        }
        // The bonus position (the num_positions - 1): the target's mass on a token.
        target[(num_positions - 1) * vocab + 0] = 1.0;
        let target_t = Tensor::from_vec(target, (num_positions, vocab), &candle_core::Device::Cpu).unwrap();
        let draft_t = Tensor::from_vec(draft, (num_positions, vocab), &candle_core::Device::Cpu).unwrap();
        let res = verify_draft_rejection(&target_t, &draft_t, &draft_tokens, 42).unwrap();
        // The all-3 drafts accepted (the p_target == p_draft == 1.0 at each).
        assert_eq!(res.num_accepted, 3, "the agreeing distributions accept all drafts");
        assert_eq!(res.accepted_tokens, vec![1u32, 2, 3]);
        // The bonus is the target's argmax at the last position (the token 0).
        assert_eq!(res.continuation_token, 0, "the bonus is the target's top token");
    }
}
