// Per-sequence speculative-decoding statistics. Replaces the global-atomic
// "DFlash Stats" / "MTP Stats" per-step spam with a clean per-sequence report
// that is fetched across the process boundary at sequence end.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::runner::SpecSeqStatsData;
use crate::speculative::verify::MtpVerifyResult;

/// The rolling window size for the per-seq prefix-acceptance curve (the dynamic-K
/// signal). The last W verify steps feed the curve; W is a few seconds of decode
/// (the no whole-seq cumulative, the no slow global EMA).
const SPEC_K_WINDOW: usize = 32;

#[derive(Default, Clone, Debug)]
pub struct SpecCounters {
    pub mechanism: String,
    pub steps: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub grammar_bound: usize,
    pub target_bound: usize,
    pub ff_continuations: usize,
    /// The per-position acceptance vector (the K-th position's acceptance count
    /// across steps). A K=4 draft with 2 accepted contributes `[1,1,0,0]`.
    pub accepted_per_pos: Vec<usize>,
    /// The rolling draft-acceptance rate (the `accepted / proposed` over the
    /// window). The MTP is stable across batch size; the DFlash varies.
    pub acceptance_rate: f64,
    /// The per-batch acceptance tracking (the MTP-vs-DFlash distinction: the
    /// acceptance rate at each observed batch size). The MTP is flat; the DFlash
    /// varies.
    pub per_batch: Vec<(usize, f64)>, // The (batch_size, acceptance_rate) pairs.
    /// The rolling ring: the last `SPEC_K_WINDOW` verify steps, each `(k_used,
    /// accepted_count)`. Feeds the `rolling_prefix_curve` (the per-position
    /// acceptance over the recent window, the fast per-seq signal the dynamic-K
    /// law reads -- the no global-EMA lag, the no whole-seq cumulative).
    rolling: Vec<(usize, usize)>,
}

impl SpecCounters {
    fn add(&mut self, res: &MtpVerifyResult) {
        self.steps += 1;
        let k = res.num_proposed;
        self.proposed += k;
        self.accepted += res.num_accepted;
        self.rejected += res.num_proposed.saturating_sub(res.num_accepted);
        // The rejection breakdown: the target_bound (the target model rejected the
        // draft) + the grammar_bound (the grammar firewall rejected the draft).
        self.target_bound += res.target_bound;
        self.grammar_bound += res.grammar_bound;
        // The per-position acceptance vector: the K-th position's acceptance. A
        // K=4 draft with 2 accepted contributes `[1,1,0,0]`.
        if self.accepted_per_pos.len() < k {
            self.accepted_per_pos.resize(k, 0);
        }
        for pos in 0..k {
            if pos < res.num_accepted {
                self.accepted_per_pos[pos] += 1;
            }
        }
        // The rolling draft-acceptance rate (the break-even signal).
        self.acceptance_rate = if self.proposed > 0 {
            self.accepted as f64 / self.proposed as f64
        } else {
            0.0
        };
        // The rolling ring: push this step's (k_used, accepted), bounded to W.
        self.rolling.push((k, res.num_accepted));
        if self.rolling.len() > SPEC_K_WINDOW {
            self.rolling.remove(0);
        }
    }

    /// The per-position acceptance curve over the recent window: for each draft
    /// position `k` (1-indexed), the fraction of the last `SPEC_K_WINDOW` steps where
    /// the draft was accepted through position `k` (i.e. `num_accepted >= k`). The
    /// curve is monotonically non-increasing (you can't accept position k without
    /// accepting 0..k-1). This is the fast per-seq signal the dynamic-K law reads
    /// (the no global-EMA lag). Returns `Vec` of length `k_max` (the index `i` is
    /// position `i+1`).
    pub fn rolling_prefix_curve(&self, k_max: usize) -> Vec<f64> {
        let n = self.rolling.len();
        if n == 0 || k_max == 0 {
            // The no verify history yet (the empty ring): return an EMPTY curve (the
            // the law reads this as "no data" -> the ceiling, the no premature shrink).
            // Distinct from a real all-zero curve (the history exists but every
            // position is unreliable -> the knee at 1).
            return Vec::new();
        }
        // For each position k in 1..=k_max, count the steps where accepted >= k.
        let mut curve = vec![0.0; k_max];
        for &(_k_used, accepted) in &self.rolling {
            for k in 1..=k_max.min(accepted) {
                curve[k - 1] += 1.0;
            }
        }
        for v in curve.iter_mut() {
            *v /= n as f64;
        }
        curve
    }

    /// Record the per-batch acceptance (the MTP-vs-DFlash distinction: the
    /// `per_batch` field tracks the acceptance rate at each observed batch size).
    pub fn record_batch(&mut self, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        // The update the last entry for this batch size (the rolling rate).
        if let Some(last) = self.per_batch.iter_mut().rev().find(|(b, _)| *b == batch_size) {
            last.1 = self.acceptance_rate;
        } else {
            self.per_batch.push((batch_size, self.acceptance_rate));
        }
    }
}

/// Per-sequence window, reported (and dropped) when the sequence finishes.
pub static SPEC_SEQ_STATS: LazyLock<Mutex<HashMap<usize, SpecCounters>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record one speculative step into the per-seq window.
pub fn spec_stats_update(name: &str, seq_id: usize, res: &MtpVerifyResult) {
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    let c = map.entry(seq_id).or_default();
    if c.mechanism.is_empty() {
        c.mechanism = name.to_string();
    }
    c.add(res);
}

/// Record a speculative fast-forward continuation (the grammar-forced ff run) into the
/// per-seq window. Unlike MTP/DFlash (which propose + verify drafts), the spec-ff path
/// appends deterministic grammar-forced tokens, so it increments `steps` and
/// `ff_continuations` (the number of ff tokens appended this step).
pub fn spec_stats_update_ff(name: &str, seq_id: usize, ff_count: usize) {
    if ff_count == 0 {
        return;
    }
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    let c = map.entry(seq_id).or_default();
    if c.mechanism.is_empty() {
        c.mechanism = name.to_string();
    }
    c.steps += 1;
    c.ff_continuations += ff_count;
}

/// Drop the per-sequence window at the sequence's end (runner-side cleanup).
pub fn spec_seq_report(seq_id: usize) {
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    map.remove(&seq_id);
}

/// Look up a sequence's speculative stats (without removing them) for cross-process reporting.
pub fn spec_seq_stats_data(seq_id: usize) -> SpecSeqStatsData {
    let map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    map.get(&seq_id)
        .map(|c| SpecSeqStatsData {
            mechanism: c.mechanism.clone(),
            steps: c.steps,
            proposed: c.proposed,
            accepted: c.accepted,
            rejected: c.rejected,
            grammar_bound: c.grammar_bound,
            target_bound: c.target_bound,
            ff_continuations: c.ff_continuations,
            accepted_per_pos: c.accepted_per_pos.clone(),
            acceptance_rate: c.acceptance_rate,
            per_batch: c.per_batch.clone(),
            rolling_curve: c.rolling_prefix_curve(32),
        })
        .unwrap_or_default()
}

/// The per-seq rolling prefix-acceptance curve (the dynamic-K signal): for each draft
/// position `k` (1-indexed), the fraction of the last `SPEC_K_WINDOW` verify steps where
/// the draft was accepted through position `k`. The controller reads this to compute the
/// per-seq optimal K (the knee of the curve). Returns an empty `Vec` when the seq has no
/// recent verify history (the no stats yet -> the K stays at the ceiling, the no shrink).
pub fn spec_seq_prefix_curve(seq_id: usize, k_max: usize) -> Vec<f64> {
    if k_max == 0 {
        return Vec::new();
    }
    let map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    match map.get(&seq_id) {
        Some(c) => c.rolling_prefix_curve(k_max),
        // The no history: an EMPTY curve (the no all-zeros, which the law would read as
        // "every position unreliable" -> K=1). The empty curve means "no data yet" ->
        // the law keeps the ceiling (the no premature shrink before any verify ran).
        None => Vec::new(),
    }
}

/// The rolling window size (the last W verify steps feeding the prefix curve).
pub fn spec_k_window() -> usize {
    SPEC_K_WINDOW
}

/// The per-seq next-step K (the runner-side, the no the scheduler guessing from an
/// empty cross-process global). Reads the seq's rolling prefix curve (the populated in
/// the runner process) + returns the knee: the largest draft position whose per-position
/// acceptance is ≥ the break-even threshold. The `ceiling` is the `spec_draft_max` (the
/// no K above the configured max). When the curve is empty (the no history yet), the
/// ceiling is returned (the no premature shrink before data exists). This is what the
/// runner reports in `SpecStepOut.next_k` (the no the scheduler's `spec_draft_len_for_seq`
/// reading the empty main-process `SPEC_SEQ_STATS`).
pub fn spec_seq_next_k(seq_id: usize, ceiling: usize, break_even: f64) -> usize {
    if ceiling == 0 {
        return 0; // the SD off
    }
    let curve = spec_seq_prefix_curve(seq_id, ceiling);
    if curve.is_empty() {
        return ceiling; // the no history yet: the ceiling (the no premature shrink)
    }
    // The knee: the largest k (1-indexed) whose per-position acceptance is ≥ the
    // break-even threshold. The curve is non-increasing, so the knee is a prefix.
    let mut k = 0;
    for (i, &acc) in curve.iter().enumerate() {
        if acc >= break_even {
            k = i + 1;
        } else {
            break;
        }
    }
    k.max(1) // the floor: at least 1 draft token (the no K=0 while SD is enabled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::verify::MtpVerifyResult;

    fn res(k: usize, accepted: usize) -> MtpVerifyResult {
        MtpVerifyResult {
            accepted_tokens: vec![],
            continuation_token: 0,
            num_accepted: accepted,
            num_proposed: k,
            target_bound: k.saturating_sub(accepted),
            grammar_bound: 0,
        }
    }

    #[test]
    fn per_position_acceptance_vector() {
        // The golden: a K=4 draft with 2 accepted → the `[1,1,0,0]` per-step.
        let mut c = SpecCounters::default();
        c.add(&res(4, 2));
        assert_eq!(c.accepted_per_pos, vec![1, 1, 0, 0]);
        // The second step: a K=4 draft with 3 accepted → the `[2,2,1,0]`.
        c.add(&res(4, 3));
        assert_eq!(c.accepted_per_pos, vec![2, 2, 1, 0]);
    }

    #[test]
    fn rolling_acceptance_rate() {
        let mut c = SpecCounters::default();
        c.add(&res(4, 2)); // The 2/4 = 0.5.
        assert!((c.acceptance_rate - 0.5).abs() < 1e-6);
        c.add(&res(4, 4)); // The (2+4)/(4+4) = 0.75.
        assert!((c.acceptance_rate - 0.75).abs() < 1e-6);
    }

    #[test]
    fn per_batch_tracking() {
        let mut c = SpecCounters::default();
        c.add(&res(4, 2));
        c.record_batch(8); // The batch 8, the rate 0.5.
        c.add(&res(4, 4));
        c.record_batch(16); // The batch 16, the rate 0.75.
        assert_eq!(c.per_batch.len(), 2);
        assert_eq!(c.per_batch[0].0, 8);
        assert!((c.per_batch[0].1 - 0.5).abs() < 1e-6);
        assert_eq!(c.per_batch[1].0, 16);
        assert!((c.per_batch[1].1 - 0.75).abs() < 1e-6);
    }

    // The runner-side dynamic-K (the spec_seq_next_k, the no the scheduler reading
    // the empty cross-process global). The tests populate the per-seq rolling curve via
    // spec_stats_update (the real signal path) + assert the knee.
    #[test]
    fn next_k_empty_curve_holds_ceiling() {
        // The no history (the seq never drafted): the curve is empty -> the ceiling
        // (the no premature shrink before data exists).
        assert_eq!(spec_seq_next_k(999, 8, 0.3), 8);
    }

    #[test]
    fn next_k_collapsing_curve_finds_knee() {
        // A weak drafter: every step drafts 8, accepts only 2 (the curve collapses at
        // position 3). The knee is 2 (the no drafting positions 3-8 that never accept).
        let seq = 42;
        for _ in 0..40 {
            spec_stats_update("MTP", seq, &res(8, 2));
        }
        assert_eq!(spec_seq_next_k(seq, 8, 0.3), 2, "the collapsing curve finds the knee at 2");
        spec_seq_report(seq); // the cleanup (the no cross-test contamination)
    }

    #[test]
    fn next_k_strong_curve_holds_ceiling() {
        // A strong drafter: every step drafts 8, accepts all 8 (the curve stays high).
        // The knee is the ceiling (8, the no shrink when the drafter is reliable).
        let seq = 43;
        for _ in 0..40 {
            spec_stats_update("MTP", seq, &res(8, 8));
        }
        assert_eq!(spec_seq_next_k(seq, 8, 0.3), 8, "the strong curve holds the ceiling");
        spec_seq_report(seq);
    }

    #[test]
    fn next_k_ceiling_zero_means_sd_off() {
        // The ceiling is 0 (the SD not configured): the K is 0 (the no draft, the plain
        // decode), regardless of the curve.
        assert_eq!(spec_seq_next_k(44, 0, 0.3), 0);
    }
}