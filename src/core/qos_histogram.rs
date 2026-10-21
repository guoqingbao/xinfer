//! The bounded-memory workload estim (the §1 collection framework, the core).
//!
//! This module is the CLEAN LOGICAL CORE of the QoS collection framework. It
//! defines the `WorkloadHistogram` (the per-conversation-class bounded
//! estimators) + the estimators themselves (the P² quantile, the log-spaced CDF,
//! the EMA, the ring buffer, the exposed-time matrix, the request-state counts).
//!
//! ** design principle (the professional extensibility): the ` histogram
//! (this module, on the `qos` branch) carries ONLY the stable estim cross-branch
//! signals (the ITL, the output length, the batch size, the prefix hit, the KV
//! pressure, the exposed time, the request states, the preemptions, the turns,
//! the reuse evidence). LATER branches (the `spec-decode`, the `llg`, the
//! `biglittle`, the `state`, the `unified-memory`, the `lattice`, the
//! `multi-model`) EXTEND the histogram with their relevant data via the stack
//! (each branch's `qos.rs` is a superset of the previous). The extension points
//! are the `AdaptivePlan` actuators (the `spec_draft_len` is added by the
//! `spec-decode` branch, the `grammar_active` by the `llg` branch, etc.).
//!
//! Every estimator here a PURE data structure (the no I/O, the no GPU, the
//! CPU-only, the unit-testable). The total histogram is O(1) in the number of
//! conversations (the per-class key is the bounded set {Latency, Throughput} ×
//! {Contended, Uncontended}, the no per-conversation storage).

use crate::core::qos::{QosClass, Regime};

/// The P² streaming quantile estimator (the Jain & Chaudhary 1985 result: the
/// 5-cell reservoir, the no full CDF storage). Tracks the p-quantile (the p95 for
/// the ITL) of a stream of values.
///
/// The state:
/// - `q[5]` = the marker heights (the quantile estimates)
/// - `n[5]` = the actual marker positions (the counts)
/// - `n'[5]` = the desired marker positions
/// - `d[5]` = the desired position increments (the target quantile fractions)
///
/// The update per observation `x`:
/// 1. Find the cell `i` where `q[i] <= x < q[i+1]`.
/// 2. Shift the positions: `n[j] += 1` for `j > i`, `n[j] -= 1` for `j < i`.
/// 3. Advance the desired positions: `n'[j] += d[j]` for all `j`.
/// 4. For each interior marker `j in 1..3`: if `n'[j] - n[j] >= 1` or
///    `n[j] - n'[j] >= 1`, adjust `n[j]` by ±1, then update `q[j]` via the
///    parabolic formula (the fallback to the linear formula when the parabolic
///    gives an out-of-order result).
#[derive(Debug, Clone)]
pub struct P2Quantile {
    /// The 5 marker heights (the quantile estimates).
    q: [f64; 5],
    /// The 5 actual marker positions (the counts).
    n: [i64; 5],
    /// The 5 desired marker positions.
    n_prime: [f64; 5],
    /// The 5 desired position increments (the target fractionsile fractions).
    d: [f64; 5],
    /// The number of observations seen (the fill phase: the first 5).
    count: u64,
    /// Whether the fill phase is complete (the 5 markers initialized).
    filled: bool,
}

impl P2Quantile {
    /// Create a new P² estimator for the target quantile `p` (e.g. 0.95).
    pub fn new(p: f64) -> Self {
        Self {
            q: [0.0; 5],
            n: [0; 5],
            n_prime: [0.0; 5],
            d: [0.0, p / 4.0, p / 2.0, (3.0 * p) / 4.0, 1.0],
            count: 0,
            filled: false,
        }
    }

    /// Feed one observation. The 4-step update (the fill phase for the first 5,
    /// the P update thereafter).
    pub fn observe(&mut self, x: f64) {
        self.count += 1;
        if !self.filled {
            // The fill phase: accumulate the first 5 observations.
            self.q[(self.count - 1) as usize] = x;
            if self.count == 5 {
                // The sort the 5 initial values.
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // The initialize the positions (the 0, 1, 2, 3, 4).
                for i in 0..5 {
                    self.n[i] = i as i64;
                    self.n_prime[i] = self.d[i] * 4.0; // the desired positions at n=5.
                }
                self.filled = true;
            }
            return;
        }

        // The find the cell `i` where `q[i] <= x < q[i+1]` (the i in 0..3).
        let i: i32 = if x < self.q[0] {
            // The x is below the min: the shift the positions down.
            self.n[1..].iter_mut().for_each(|n| *n += 1);
            self.q[0] = x; // The update the min.
            -1 // The sentinel: the no interior marker adjustment.
        } else if x >= self.q[4] {
            // The x is above the max: the shift the positions up.
            self.n[..4].iter_mut().for_each(|n| *n -= 1);
            self.q[4] = x; // The update the max.
            5 // The sentinel: the no interior marker adjustment.
        } else {
            // The x is in a cell: the find the cell index.
            (0..4)
                .find(|&k| x >= self.q[k] && x < self.q[k + 1])
                .map(|k| k as i32)
                .unwrap_or(4)
        };

        if i < 0 || i > 4 {
            return; // The sentinel: the min/max update, the no interior adjustment.
        }

        // The shift the positions (the step 2).
        for j in (i as usize + 1)..5 {
            self.n[j] += 1;
        }
        for j in 0..(i as usize) {
            self.n[j] -= 1;
        }

        // The advance the desired positions (the step 3).
        for j in 0..5 {
            self.n_prime[j] += self.d[j];
        }

        // The adjust the interior markers (the step 4).
        for j in 1..4 {
            let diff = self.n_prime[j] - self.n[j] as f64;
            if diff >= 1.0 || diff <= -1.0 {
                let sign = if diff >= 1.0 { 1 } else { -1 };
                self.n[j] += sign;
                // The parabolic formula (the preferred).
                let q_new = self.parabolic(j, sign);
                if self.q[j - 1] < q_new && q_new < self.q[j + 1] {
                    self.q[j] = q_new;
                } else {
                    // The linear fallback (the out-of-order parabolic).
                    self.q[j] = self.linear(j, sign);
                }
            }
        }
    }

    /// The parabolic marker update (the Jain & Chaudhary formula).
    fn parabolic(&self, j: usize, sign: i64) -> f64 {
        let nj = self.n[j] as f64;
        let nj1 = self.n[j - 1] as f64;
        let nj2 = self.n[j + 1] as f64;
        let qj = self.q[j];
        let qj1 = self.q[j - 1];
        let qj2 = self.q[j + 1];
        // The parabolic interpolation.
        let term1 = (nj2 - nj) * (qj2 - qj) / (nj2 - nj1);
        let term2 = (nj - nj1) * (qj - qj1) / (nj2 - nj1);
        qj + sign as f64 * (term1 + term2) / (nj2 - nj1)
    }

    /// The linear marker update (the fallback when the parabolic is out-of-order).
    fn linear(&self, j: usize, sign: i64) -> f64 {
        let nj = self.n[j] as f64;
        let nj1 = self.n[j - 1] as f64;
        let nj2 = self.n[j + 1] as f64;
        let qj = self.q[j];
        let qj1 = self.q[j - 1];
        let qj2 = self.q[j + 1];
        if sign > 0 {
            qj + (qj2 - qj) / (nj2 - nj)
        } else {
            qj - (qj - qj1) / (nj - nj1)
        }
    }

    /// The p p-quantile estimate (the `q[4]` for the p95, the top marker).
    pub fn quantile(&self) -> f64 {
        if !self.filled {
            // The fill phase: the return the max of the seen values.
            let seen = (self.count as usize).min(5);
            return self.q[..seen].iter().cloned().fold(0.0f64, f64::max);
        }
        self.q[4]
    }

    /// The number of observations seen.
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// The 32-bucket log-spaced output-length CDF (the bounded memory, the no full
/// CDF storage). The bucket index is the `log2(len)` (the clamped to 0..31).
#[derive(Debug, Clone, Default)]
pub struct LogCdf {
    /// The 32 log-spaced buckets (the counts).
    buckets: [u64; 32],
    /// The total count.
    total: u64,
}

impl LogCdf {
    /// Feed one output length (the tokens). The log-spaced bucket index.
    pub fn observe(&mut self, len: usize) {
        if len == 0 {
            return;
        }
        self.total += 1;
        let idx = (len as f64).log2().floor() as usize;
        let b = idx.min(31);
        self.buckets[b] += 1;
    }

    /// The median output length (the class-inference signal). The 2^bucket index.
    pub fn median(&self) -> usize {
        if self.total == 0 {
            return 0;
        }
        let half = self.total / 2;
        let mut acc = 0;
        for (i, &c) in self.buckets.iter().enumerate() {
            acc += c;
            if acc >= half {
                return 1usize << i; // The bucket's log bound.
            }
        }
        0
    }

    /// The total count.
    pub fn total(&self) -> u64 {
        self.total
    }
}

/// The EMA + counter (the bounded-memory estimator for the batch-size / the
/// prefix-hit / the kv-pressure signals). The 0.9/0.1 decay (matching the
/// existing `QosController` ITL/output EMAs).
#[derive(Debug, Clone, Default)]
pub struct Ema {
    /// The EMA value.
    value: f64,
    /// The count of observations.
    count: u64,
}

impl Ema {
    /// Feed one sample. The EMA (the 0.9/0.1 decay).
    pub fn observe(&mut self, x: f64) {
        self.count += 1;
        self.value = if self.count == 1 { x } else { 0.9 * self.value + 0.1 * x };
    }

    /// The current EMA value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The count of observations.
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// The 3×2 exposed-time matrix (the NInfer `RequestHostTiming`: the per-work-class
/// `host_ns` + the `device_wait_ns`). The work-classes are the Decode / the
/// Prefill / the Control.
#[derive(Debug, Clone, Default)]
pub struct ExposedTime {
    /// The [work_class][metric] matrix: the 3 work-classes (the decode / the
    /// prefill / the control) × the 2 metrics (the `host_ns` + the
    /// `device_wait_ns`).
    pub matrix: [[u64; 2]; 3],
}

impl ExposedTime {
    /// Add one exposed-time sample (the work-class index + the `host_ns` + the
    /// `device_wait_ns`).
    pub fn add(&mut self, class: usize, host_ns: u64, device_wait_ns: u64) {
        if class < 3 {
            self.matrix[class][0] += host_ns;
            self.matrix[class][1] += device_wait_ns;
        }
    }
}

/// The 8-count request lifecycle (the NInfer `EngineRequestState`: the
/// Waiting / the Materializing / the Prefill / the Replay / the Pausing / the
/// Paused / the DecodeReady / the ControlReady).
#[derive(Debug, Clone, Copy, Default)]
pub struct RequestStates {
    /// The 8 lifecycle counts.
    pub counts: [u64; 8],
}

impl RequestStates {
    /// Increment the count for a lifecycle state (the 0..7 index).
    pub fn observe(&mut self, state: usize) {
        if state < 8 {
            self.counts[state] += 1;
        }
    }
}

/// The bounded 256-entry reuse-evidence digest (the NInfer real-demand signal:
/// the `proven_frontier` + the `demand_digest` + the `adopted_count`). The ring
/// buffer of the 256 most-recent demand digests (the no predictedbounded growth).
#[derive(Debug, Clone, Default)]
pub struct ReuseEvidence {
    /// The ring buffer of the 256 most-recent demand digests.
    digests: std::collections::VecDeque<u64>,
    /// The adopted count (the actual cross-request adoption, the no predicted
    /// hit-rate).
    pub adopted_count: u64,
    /// The proven frontier (the last-verified prefix length).
    pub proven_frontier: usize,
}

impl ReuseEvidence {
    const CAP: usize = 256;

    /// Record one demand digest (the bounded ring).
    pub fn observe(&mut self, digest: u64) {
        if self.digests.len() == Self::CAP {
            self.digests.pop_front();
        }
        self.digests.push_back(digest);
    }

    /// The current demand digest (the last observed).
    pub fn demand_digest(&self) -> u64 {
        self.digests.back().copied().unwrap_or(0)
    }
}

/// The per-drafter acceptance profile (the §spec-decode` extension of the
/// `WorkloadHistogram`). The per-drafter (the MTP / the DFlash / the chained)
/// acceptance-rate EMA + the per-batch curve. Populated only when speculative
/// decoding is active (the `uses_builtin_mtp` / the `uses_dflash`); when spec
/// decode is off the profile stays at its default (the ` data, the `select`
/// returns the MTP default). The coding-agent workload (the high DFlash
/// acceptance at the high batch) → the DFlash; the chatbot workload (the high
/// MTP acceptance stable) → the MTP; the mixed workload (the both high) → the
/// chained.
#[derive(Debug, Clone, Default)]
pub struct DrafterProfile {
    /// The per-drafter acceptance-rate EMA (the 0..1 fraction). The index is the
    /// drafter kind (the 0 = the MTP, the 1 = the DFlash, the 2 = the chained).
    pub accept_ema: [f64; 3],
    /// The per-drafter per-batch curve (the (batch_size, acceptance_rate) pairs).
    pub per_batch: [Vec<(usize, f64)>; 3],
}

impl DrafterProfile {
    /// Record one acceptance observation for a drafter kind (the 0 = the MTP,
    /// the 1 = the DFlash, the 2 = the chained).
    pub fn observe(&mut self, kind: usize, batch_size: usize, accept_frac: f64) {
        if kind >= 3 {
            return;
        }
        // The EMA update (the 0.9/0.1 decay, matching the existing EMAs).
        self.accept_ema[kind] = if self.accept_ema[kind] == 0.0 && self.per_batch[kind].is_empty() {
            accept_frac
        } else {
            0.9 * self.accept_ema[kind] + 0.1 * accept_frac
        };
        // The per-batch curve (the rolling rate at this batch size).
        if let Some(last) = self.per_batch[kind].iter_mut().rev().find(|(b, _)| *b == batch_size) {
            last.1 = self.accept_ema[kind];
        } else {
            self.per_batch[kind].push((batch_size, self.accept_ema[kind]));
        }
    }

    /// The best drafter for the workload class (the highest acceptance EMA).
    pub fn select(&self) -> usize {
        let mtp = self.accept_ema[0];
        let dflash = self.accept_ema[1];
        if mtp > 0.0 && dflash > 0.0 && (mtp + dflash) / 2.0 > mtp.max(dflash) * 0.9 {
            2 // The mixed: the chained (the both high).
        } else if dflash > mtp {
            1 // The coding-agent: the DFlash (the higher acceptance).
        } else {
            0 // The chatbot: the MTP (the higher acceptance).
        }
    }
}

/// The per-class bounded histogram (the §1 core, the bounded-memory streaming
/// estimators). The matrix is keyed by the (regime × class) — the bounded class
/// set {Latency, Throughput} × {Contended, Uncontended}.
///
/// The CORE fields (the `qos` branch): the `itl` (the P² p95), the
/// `output_len` (the CDF), the `batch_size` (the EMA), the `prefix_hit` (the
/// EMA), the `kv_pressure` (the EMA), the `exposed_time` (the 3×2), the
/// `request_states` (the 8-count), the `preemptions` (the 2-count), the
/// `turns`, the `reuse_evidence` (the 256-ring).
///
/// The EXTENSION fields (the later branches, via the stack): the
/// `drafter_profile` (the `spec-decode` branch, the per-drafter acceptance
/// tracking, the populated only when spec decode is active), the
/// `grammar_active` (the `llg` branch), etc.
#[derive(Debug, Clone)]
pub struct WorkloadHistogram {
    /// The ITL P² estimator (the [ × class: the [Contended|Uncontended][Latency|Throughput]).
    pub itl: [[P2Quantile; 2]; 2],
    /// The output-length CDF (the class: the [Latency|Throughput]).
    pub output_len: [LogCdf; 2],
    /// The batch-size EMA (the regime: the [Contended|Uncontended]).
    pub batch_size: [Ema; 2],
    /// The prefix-hit EMA (the class: the [Latency|Throughput]).
    pub prefix_hit: [Ema; 2],
    /// The KV-pressure EMA (the regime: the [Contended|Uncontended]).
    pub kv_pressure: [Ema; 2],
    /// The exposed-time 3×2 matrix (the work-class × the class: the [Decode|Prefill|Control][Latency|Throughput]).
    pub exposed_time: [[ExposedTime; 2]; 3],
    /// The request-state 8-count (the class: the [Latency|Throughput]).
    pub request_states: [RequestStates; 2],
    /// The preemption 2-count (the class: the [Latency|Throughput]).
    pub preemptions: [u64; 2],
    /// The conversation warmth (the turns counter, the L1 capture_seq).
    pub turns: u64,
    /// The reuse-evidence 256-ring (the class: the [Latency|Throughput]).
    pub reuse_evidence: [ReuseEvidence; 2],
    /// The per-drafter acceptance profile (the `spec-decode` extension, the
    /// populated only when spec decode is active).
    pub drafter_profile: DrafterProfile,
}

impl WorkloadHistogram {
    /// Create a new histogram with the P² estimators initialized for the p95.
    pub fn new() -> Self {
        Self {
            itl: [
                [P2Quantile::new(0.95), P2Quantile::new(0.95)],
                [P2Quantile::new(0.95), P2Quantile::new(0.95)],
            ],
            output_len: [LogCdf::default(), LogCdf::default()],
            batch_size: [Ema::default(), Ema::default()],
            prefix_hit: [Ema::default(), Ema::default()],
            kv_pressure: [Ema::default(), Ema::default()],
            exposed_time: [
                [ExposedTime::default(), ExposedTime::default()],
                [ExposedTime::default(), ExposedTime::default()],
                [ExposedTime::default(), ExposedTime::default()],
            ],
            request_states: [RequestStates::default(), RequestStates::default()],
            preemptions: [0, 0],
            turns: 0,
            reuse_evidence: [ReuseEvidence::default(), ReuseEvidence::default()],
            drafter_profile: DrafterProfile::default(),
        }
    }

    /// The regime index (the 0 = the Contended, the 1 = the Uncontended).
    fn regime_idx(regime: Regime) -> usize {
        match regime {
            Regime::Contended => 0,
            Regime::Uncontended => 1,
        }
    }

    /// The class index (the 0 = the Latency, the 1 = the Throughput).
    fn class_idx(class: QosClass) -> usize {
        match class {
            QosClass::Latency => 0,
            QosClass::Throughput => 1,
        }
    }

    /// Feed one ITL sample (the ms) for the (regime, class). The P² update.
    pub fn record_itl(&mut self, regime: Regime, class: QosClass, step_ms: f64) {
        self.itl[Self::regime_idx(regime)][Self::class_idx(class)].observe(step_ms);
    }

    /// Feed one output length (the tokens) for the class. The CDF update.
    pub fn record_output_len(&mut self, class: QosClass, len: usize) {
        self.output_len[Self::class_idx(class)].observe(len);
    }

    /// Feed one batch size (the count count) for the regime. The EMA update.
    pub fn record_batch_size(&mut self, regime: Regime, batch: usize) {
        self.batch_size[Self::regime_idx(regime)].observe(batch as f64);
    }

    /// Feed one prefix-hit ratio (the 0..1) for the class. The EMA update.
    pub fn record_prefix_hit(&mut self, class: QosClass, hit_ratio: f64) {
        self.prefix_hit[Self::class_idx(class)].observe(hit_ratio);
    }

    /// Feed one KV-pressure ratio (the 0..1) for the regime. The EMA update.
    pub fn record_kv_pressure(&mut self, regime: Regime, pressure: f64) {
        self.kv_pressure[Self::regime_idx(regime)].observe(pressure);
    }

    /// Feed one exposed-time sample (the work-class index the host_ns + the
    /// device_wait_ns) for the class. The 3×2 matrix update.
    pub fn record_exposed_time(
        &mut self,
        class: QosClass,
        work_class: usize,
        host_ns: u64,
        device_wait_ns: u64,
    ) {
        self.exposed_time[work_class][Self::class_idx(class)].add(work_class, host_ns, device_wait_ns);
    }

    /// Feed one request-state transition (the 0..7 lifecycle index) for the class.
    pub fn record_request_state(&mut self, class: QosClass, state: usize) {
        self.request_states[Self::class_idx(class)].observe(state);
    }

    /// Feed one preemption for the class. The 2-count update.
    pub fn record_preemption(&mut self, class: QosClass) {
        self.preemptions[Self::class_idx(class)] += 1;
    }

    /// Feed one turn (the conversation warmth, the L1 capture_seq).
    pub fn record_turn(&mut self) {
        self.turns += 1;
    }

    /// Feed one reuse-evidence digest for the class. The 256-ring update.
    pub fn record_reuse_evidence(&mut self, class: QosClass, digest: u64) {
        self.reuse_evidence[Self::class_idx(class)].observe(digest);
    }

    /// The ITL p95 for the (regime, class). The P² estimate.
    pub fn itl_p95(&self, regime: Regime, class: QosClass) -> f64 {
        self.itl[Self::regime_idx(regime)][Self::class_idx(class)].quantile()
    }

    /// The output-length median for the class. The CDF estimate.
    pub fn output_len_median(&self, class: QosClass) -> usize {
        self.output_len[Self::class_idx(class)].median()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p2_p95_converges() {
        let mut p2 = P2Quantile::new(0.95);
        // The feed a sustained high-ITL stream (the 200ms).
        for _ in 0..100 {
            p2.observe(200.0);
        }
        // The p95 should be near the 200ms (the bounded-memory estimator).
        assert!(p2.quantile() > 150.0, "p95 should track the high ITL, got {}", p2.quantile());
    }

    #[test]
    fn p2_p95_mixed_stream() {
        let mut p2 = P2Quantile::new(0.95);
        // The feed a mixed stream: the 90% low (10ms) + the 10% high (1000ms).
        for _ in 0..90 {
            p2.observe(10.0);
        }
        for _ in 0..10 {
            p2.observe(1000.0);
        }
        // The p95 should be near the high value (the 10% tail).
        assert!(p2.quantile() > 500.0, "p95 should be in the high tail, got {}", p2.quantile());
    }

    #[test]
    fn log_cdf_bucketing() {
        let mut cdf = LogCdf::default();
        // The feed a mix of short + long outputs.
        for _ in 0..50 {
            cdf.observe(10); // The short (the log2 ~ 3).
        }
        for _ in 0..50 {
            cdf.observe(1000); // The long (the log2 ~ 10).
        }
        // The median should be between the two (the bounded CDF).
        let med = cdf.median();
        assert!(med >= 8 && med <= 2048, "median out of range: {med}");
    }

    #[test]
    fn regime_classification() {
        // The uncontended: the low kv-pressure, the low decode (below the ).
        assert_eq!(Regime::classify(0.1, 0.1, 0.5, 0.8), Regime::Uncontended);
        // The contended: the high kv-pressure (above the threshold).
        assert_eq!(Regime::classify(0.9, 0.9, 0.5, 0.8), Regime::Contended);
        // The contended: the high decode pressure (above the threshold).
        assert_eq!(Regime::classify(0.9, 0.1, 0.5, 0.8), Regime::Contended);
    }

    #[test]
    fn reuse_evidence_bounded() {
        let mut h = WorkloadHistogram::new();
        // The feed more than the 256-cap (the bounded ring).
        for i in 0..300 {
            h.record_reuse_evidence(QosClass::Throughput, i as u64);
        }
        // The demand digest is the last observed (the 299).
        assert_eq!(h.reuse_evidence[1].demand_digest(), 299);
    }

    #[test]
    fn exposed_time_matrix() {
        let mut h = WorkloadHistogram::new();
        h.record_exposed_time(QosClass::Latency, 0, 100, 50); // The decode host_ns=100, device_wait_ns=50.
        assert_eq!(h.exposed_time[0][0].matrix[0][0], 100);
        assert_eq!(h.exposed_time[0][0].matrix[0][1], 50);
    }
}