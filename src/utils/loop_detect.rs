//! The in-flight closed-loop detector (the anti-loop).
//!
//! Detects a PERIODIC repetition (a closed loop: the tail is S S S ... for some
//! span S), NOT mere word frequency. The detector is always active when the
//! XINFER_ANTI_LOOP env var is set; it is not region-gated.
//!
//! The `content_fingerprint` replicates the prefix-cache hasher pattern (the
//! DefaultHasher over the token slice with the 0xDEADBEEF seed) rather than
//! reaching across modules (the prefix_cache's hasher is private).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

/// The fingerprint of a token slice (the prefix-cache hasher pattern, the
/// DefaultHasher over the slice with the 0xDEADBEEF seed). O(len) to compute,
/// O(1) to compare.
fn content_fingerprint(tokens: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    0xDEADBEEFu64.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// A confirmed closed loop: the period P (tokens), the repeating unit S (the last P
/// tokens), and the seed (the first ~20% of S, the kick target).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSpan {
    pub period: u32,
    pub unit: Vec<u32>,
    pub seed: Vec<u32>,
}

/// The per-sequence closed-loop detector (the bounded tail + the periodicity check).
/// The tail is the recent generated history (fed by the accepted tokens at the
/// OBSERVE points).
pub struct LoopDetector {
    /// seq_id -> the bounded tail (the last `window` generated tokens).
    tails: HashMap<usize, Vec<u32>>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self {
            tails: HashMap::new(),
        }
    }

    /// Append the accepted tokens to the seq's tail and check for a confirmed closed
    /// loop. `window` is the confirmation horizon (the XINFER_REPETITION_PROBE_DEPTH).
    /// Returns the LoopSpan (the period + the unit + the seed) if a loop is confirmed
    /// over the window, else None.
    pub fn observe(&mut self, seq_id: usize, accepted: &[u32], window: usize) -> Option<LoopSpan> {
        let tail = self.tails.entry(seq_id).or_default();
        tail.extend_from_slice(accepted);
        // cap the tail to the window (the bounded look-back).
        if tail.len() > window {
            let excess = tail.len() - window;
            tail.drain(..excess);
        }
        Self::detect(tail, window)
    }

    /// Drop the seq's tail (the sequence finished).
    pub fn clear(&mut self, seq_id: usize) {
        self.tails.remove(&seq_id);
    }

    /// The periodicity check over the tail (the last `window` tokens).
    fn detect(tail: &[u32], window: usize) -> Option<LoopSpan> {
        let len = tail.len();
        // need at least 3 reps of the smallest period p=2 (the 6 tokens).
        if len < 6 {
            return None;
        }
        let conf_span = window.min(len);
        // candidate fundamental period p, small -> large; the loop is confirmed when
        // the last conf_span tokens are p-periodic with >= 3 full reps (the fluke
        // guard). The max detectable period is conf_span/3 (the window bounds it).
        let max_p = conf_span / 3;
        for p in 2..=max_p {
            if Self::is_periodic(tail, conf_span, p) {
                let unit = &tail[len - p..len];
                // the seed: the first ~20% of the unit (the kick target).
                let seed_len = (p / 5).max(1).min(p);
                return Some(LoopSpan {
                    period: p as u32,
                    unit: unit.to_vec(),
                    seed: unit[..seed_len].to_vec(),
                });
            }
        }
        None
    }

    /// True if the last `span` tokens of `tail` are p-periodic (the >= 3 identical
    /// p-blocks, compared via the content_fingerprint).
    fn is_periodic(tail: &[u32], span: usize, p: usize) -> bool {
        let len = tail.len();
        let start = len - span;
        let n_blocks = span / p;
        if n_blocks < 3 {
            return false; // the fluke guard (the >= 3 repetitions)
        }
        let first = content_fingerprint(&tail[start..start + p]);
        for b in 1..n_blocks {
            if content_fingerprint(&tail[start + b * p..start + (b + 1) * p]) != first {
                return false;
            }
        }
        true
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// The anti-loop kick: when a closed loop of period P is confirmed, forbid the
/// loop's trigger token so the model "says something else." The `forbid` is the
/// first token of the repeating unit (the seed[0] that restarts the loop); the
/// full `seed` (the first ~20% of the unit) is carried for logging and for the
/// scaling-by-caught-length (the longer the loop, the harder the kick).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntiLoopKick {
    /// The token to forbid at the next step (the unit[0], the loop trigger).
    pub forbid: u32,
    /// The full seed (the first ~20% of the unit, the scaling target).
    pub seed: Vec<u32>,
    /// The loop period P (the caught length, the scaling key).
    pub period: u32,
}

/// The per-sequence kick store: a kick is held until the periodicity clears (the
/// next observe returns None), then dropped. Self-limiting, no magic N.
#[derive(Debug, Default)]
pub struct KickStore {
    kicks: HashMap<usize, AntiLoopKick>,
}

impl KickStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate + store the kick for a confirmed loop.
    pub fn apply(&mut self, seq_id: usize, span: &LoopSpan) {
        let forbid = span.unit.first().copied().unwrap_or(0);
        self.kicks.insert(
            seq_id,
            AntiLoopKick {
                forbid,
                seed: span.seed.clone(),
                period: span.period,
            },
        );
    }

    /// The active kick for a seq (the None when no loop is caught).
    pub fn get(&self, seq_id: usize) -> Option<&AntiLoopKick> {
        self.kicks.get(&seq_id)
    }

    /// Drop the kick (the periodicity cleared, the lifecycle end).
    pub fn remove(&mut self, seq_id: usize) {
        self.kicks.remove(&seq_id);
    }

    pub fn is_active(&self, seq_id: usize) -> bool {
        self.kicks.contains_key(&seq_id)
    }
}

/// The per-sequence loop-defense stats (the guards + the durations). Captured at
/// sequence end and reported to the admins (the no grammar-digging). Modeled on
/// the SchedSeqStats (the one-line report).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopSeqStats {
    /// The number of loop guards (kicks) triggered for this sequence.
    pub guards: usize,
    /// The cumulative steps a guard was held (the mitigation duration).
    pub guard_steps: usize,
    /// The longest single guard duration (the peak).
    pub max_guard_steps: usize,
    /// The distinct loop periods caught (the P values).
    pub periods: Vec<u32>,
}

impl LoopSeqStats {
    /// One-line body for the per-sequence report (the caller adds the `[Seq N]` prefix).
    pub fn report(&self) -> String {
        if self.guards == 0 {
            return "loop-defense: no guards".to_string();
        }
        format!(
            "loop-defense: guards={} held={} steps (peak {}) periods={:?}",
            self.guards, self.guard_steps, self.max_guard_steps, self.periods
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a closed loop: the unit `u` repeated `reps` times (the S S S ...).
    /// Script-generated (the no hand-written repetitive literals).
    fn loop_seq(unit: &[u32], reps: usize) -> Vec<u32> {
        (0..reps).flat_map(|_| unit.iter().copied()).collect()
    }

    #[test]
    fn detects_three_reps() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4
        let seq = loop_seq(&unit, 3); // 12 tokens, 3 reps
        let span = det.observe(1, &seq, 512).expect("3 reps of P=4 must be detected");
        assert_eq!(span.period, 4, "the fundamental period is 4");
        assert_eq!(span.unit, unit, "the repeating unit");
        assert_eq!(span.seed, &unit[..1], "the seed is the first 20% (1 token of 4)");
    }

    #[test]
    fn fluke_two_reps_not_detected() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4
        let seq = loop_seq(&unit, 2); // 8 tokens, 2 reps (the fluke)
        assert!(
            det.observe(1, &seq, 512).is_none(),
            "2 reps is a fluke, not a confirmed loop (the >= 3 reps guard)"
        );
    }

    #[test]
    fn non_periodic_not_detected() {
        let mut det = LoopDetector::new();
        let seq: Vec<u32> = (0..200).collect(); // all distinct
        assert!(det.observe(1, &seq, 512).is_none(), "an all-distinct sequence is not a loop");
    }

    #[test]
    fn larger_loop_detected() {
        let mut det = LoopDetector::new();
        let unit: Vec<u32> = (100..120).collect(); // P=20
        let seq = loop_seq(&unit, 6); // 120 tokens, 6 reps
        let span = det.observe(1, &seq, 512).expect("6 reps of P=20 must be detected");
        assert_eq!(span.period, 20);
        assert_eq!(span.seed.len(), 4, "the seed is 20% of P=20 (4 tokens)");
    }

    #[test]
    fn window_bounds_detectable_period() {
        // a period larger than window/3 is NOT detectable (the window bounds it).
        let mut det = LoopDetector::new();
        let unit: Vec<u32> = (0..100).collect(); // P=100
        let seq = loop_seq(&unit, 4); // 400 tokens, 4 reps of P=100
        // window=300 -> max_p = 100, so P=100 is the boundary (3 reps fit in 300).
        assert!(det.observe(1, &seq, 300).is_some(), "P=100 is detectable at window=300");
        // window=200 -> max_p = 66, so P=100 is NOT a candidate.
        let mut det2 = LoopDetector::new();
        assert!(det2.observe(1, &seq, 200).is_none(), "P=100 is not detectable at window=200");
    }

    #[test]
    fn clear_drops_tail() {
        let mut det = LoopDetector::new();
        let unit = vec![1u32, 2, 3];
        let seq = loop_seq(&unit, 10);
        assert!(det.observe(1, &seq, 512).is_some());
        det.clear(1);
        // after clear, the tail is empty; a fresh non-loop must not detect.
        assert!(det.observe(1, &[100u32, 101, 102, 103, 104, 105], 512).is_none());
    }

    #[test]
    fn per_seq_independent() {
        let mut det = LoopDetector::new();
        let unit = vec![5u32, 6];
        let seq = loop_seq(&unit, 5);
        assert!(det.observe(1, &seq, 512).is_some());
        // a different seq with no loop is unaffected.
        assert!(det.observe(2, &[1u32, 2, 3, 4, 5, 6], 512).is_none());
    }

    // The kick generation (proves the kick forbids exactly the unit trigger).
    #[test]
    fn kick_forbids_unit_trigger() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4, the trigger is unit[0]=10
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512).expect("the loop must be detected");
        let mut store = KickStore::new();
        store.apply(1, &span);
        let kick = store.get(1).expect("the kick must be active");
        assert_eq!(kick.forbid, 10, "the kick forbids the unit trigger (the unit[0])");
        assert_eq!(kick.period, 4, "the kick carries the loop period");
        assert_eq!(kick.seed, &unit[..1], "the seed is the first 20% (1 token of 4)");
        assert!(store.is_active(1));
    }

    #[test]
    fn kick_lifecycle_remove() {
        let mut det = LoopDetector::new();
        let unit = vec![1u32, 2, 3];
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512).expect("the loop");
        let mut store = KickStore::new();
        store.apply(1, &span);
        assert!(store.is_active(1));
        // the periodicity cleared (the lifecycle end): drop the kick.
        store.remove(1);
        assert!(!store.is_active(1), "the kick is dropped when the loop breaks");
        assert!(store.get(1).is_none());
    }

    #[test]
    fn kick_per_seq_independent() {
        let mut det = LoopDetector::new();
        let unit = vec![7u32, 9];
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512).expect("the loop");
        let mut store = KickStore::new();
        store.apply(1, &span);
        assert!(store.is_active(1));
        assert!(!store.is_active(2), "the kick is per-sequence (the no cross-contamination)");
    }

    // The per-seq stats report (the admins' one-liner, the no grammar-digging).
    #[test]
    fn loop_stats_report() {
        // the no guards (the clean sequence).
        let empty = LoopSeqStats::default();
        assert_eq!(empty.report(), "loop-defense: no guards");
        // the guards (the looped sequence, the what-looped + the durations).
        let stats = LoopSeqStats {
            guards: 3,
            guard_steps: 42,
            max_guard_steps: 15,
            periods: vec![4, 4, 20],
        };
        let report = stats.report();
        assert!(report.contains("guards=3"), "the report shows the guard count");
        assert!(report.contains("peak 15"), "the report shows the peak duration");
        assert!(report.contains("4"), "the report shows the periods");
    }
}