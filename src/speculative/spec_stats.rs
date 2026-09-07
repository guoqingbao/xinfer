// Per-sequence speculative-decoding statistics. Replaces the global-atomic
// "DFlash Stats" / "MTP Stats" per-step spam with a clean per-sequence report
// that is fetched across the process boundary at sequence end.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::speculative::verify::MtpVerifyResult;
use crate::runner::SpecSeqStatsData;

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
    /// Adaptive-K distribution (drafts proposed per step = the active K).
    k_min: Option<usize>,
    k_max: Option<usize>,
    k_moves: usize,
    prev_k: Option<usize>,
}

impl SpecCounters {
    fn add(&mut self, res: &MtpVerifyResult) {
        self.steps += 1;
        let k = res.num_proposed;
        self.proposed += k;
        self.accepted += res.num_accepted;
        self.rejected += res.num_proposed.saturating_sub(res.num_accepted);
        // Track the K (drafts-proposed) distribution for adaptive-K observability.
        self.k_min = Some(self.k_min.map(|m| m.min(k)).unwrap_or(k));
        self.k_max = Some(self.k_max.map(|m| m.max(k)).unwrap_or(k));
        if self.prev_k.is_some() && self.prev_k != Some(k) {
            self.k_moves += 1;
        }
        self.prev_k = Some(k);
        // grammar_bound / target_bound / ff_continuations are populated by the
        // grammar firewall (ported separately); 0 for now.
    }

    // Runner-side dynamic-K / spec-stats display is commented out; only the
    // per-sequence output (server.rs) is shown. Retained for reference.
    /*
    pub fn summary(&self, label: &str) -> String {
        let rate = if self.proposed > 0 {
            self.accepted as f64 / self.proposed as f64 * 100.0
        } else {
            0.0
        };
        let avg = if self.steps > 0 {
            (self.accepted + 2 * self.steps) as f64 / self.steps as f64
        } else {
            1.0
        };
        format!(
            "{} steps={} proposed={} accepted={} rejected={} rate={:.1}% avg_tok/step={:.2} grammar_bound={} target_bound={} ff_continuations={}",
            label,
            self.steps,
            self.proposed,
            self.accepted,
            self.rejected,
            rate,
            avg,
            self.grammar_bound,
            self.target_bound,
            self.ff_continuations
        )
    }
    */
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

/// Report + drop the per-sequence window (at the sequence's end). None if empty.
pub fn spec_seq_report(seq_id: usize) -> Option<String> {
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    let c = map.remove(&seq_id)?;
    if c.steps == 0 {
        return None;
    }
    // Runner-side display is commented out; only the per-sequence output
    // (server.rs) is shown. The map cleanup above is retained.
    // Some(c.summary(&format!("seq {}", seq_id)))
    None
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
            k_min: c.k_min.unwrap_or(0),
            k_max: c.k_max.unwrap_or(0),
            k_moves: c.k_moves,
        })
        .unwrap_or_default()
}
