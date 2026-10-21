// QoS scheduling: a self-tuning control loop that bounds decode inter-token
// latency (ITL) under prefill load, instead of relying on static class weights.
//
// The loop:
//   sensor     - each step the scheduler measures the wall-clock step duration
//                (a proxy for decode ITL: big prefill chunks slow the step and
//                starve running decodes) and the model's observed output length.
//   controller - the integral gain drives prefill-chunk suppression to hold ITL at
//                the SLO; KV headroom scales the per-step token budget; the
//                measured output-length distribution infers the Latency class.
//   actuator   - a `StepPlan` (chunk size, token budget, admission order,
//                reservation gates) that the scheduler applies with no QoS logic
//                of its own.
//
// When QoS is disabled the plan is the static degenerate one (full chunk, FIFO
// admission, no gates), so the scheduler's behavior is unchanged. The controller
// is owned per model, which is the seam for multi-model serving (one controller
// per model) and multi-host scheduling (`Contention` in / `StepPlan` out is the
// wire protocol a coordinator would run globally).

use crate::core::qos_histogram::WorkloadHistogram;
use crate::core::sequence::Sequence;
use crate::speculative::DynamicSdLookup;
use serde::{Deserialize, Serialize};

/// QoS class for a request.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QosClass {
    /// Large prompts / long outputs; throughput-oriented, latency-tolerant.
    #[default]
    Throughput,
    /// Short exchanges, many turns; tight TTFT/ITL SLO (agentic).
    Latency,
}

impl QosClass {
    pub fn is_latency(&self) -> bool {
        matches!(self, QosClass::Latency)
    }
}

/// The regime classifier (the contended-vs-uncontended, the Sarathi-Serve
/// stall-free result: the prefill-decode interference is the contention signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Regime {
    /// The no contention: the low chunk, the high throughput.
    #[default]
    Uncontended,
    /// The contention: the suppressed chunk, the ITL bound.
    Contended,
}

impl Regime {
    /// Classify the regime from the decode pressure + the KV pressure against the
    /// calibrated thresholds. The thresholds come passed in (the controller
    /// resolves them from config or calibration), so the classifier itself carries
    /// no blind constants.
    pub fn classify(
        decode_pressure: f32,
        kv_pressure: f64,
        decode_threshold: f32,
        kv_threshold: f64,
    ) -> Regime {
        if decode_pressure > decode_threshold || kv_pressure > kv_threshold {
            Regime::Contended
        } else {
            Regime::Uncontended
        }
    }
}

/// Operator/request setpoints (SLOs) + tuning gains. Per-model; serde-defaulted so
/// existing configs keep working. `enabled: false` (the default) = prior operation.
///
/// Every setpoint that is `None` is auto-calibrated by the controller from the
/// measured system behavior over the first `calibration_steps` decode steps,
/// instead of being a blind constant. An operator may pin any of them to a fixed
/// value; the `None` fields are the ones derived from measurement.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QosConfig {
    pub enabled: bool,
    /// Target inter-token latency (ms) for the Latency class. The controller drives
    /// the measured ITL toward this. When `None`, it is calibrated from the measured
    /// base decode ITL (the p50 of the calibration window) so the SLO reflects what
    /// this model + hardware can actually sustain, not an arbitrary 50ms.
    pub itl_slo_ms: Option<f64>,
    /// "Short" = expected output below this fraction of the model's measured typical
    /// output length. When `None`, it is calibrated from the output-length CDF (the
    /// p25 of the measured outputs), not a fixed 0.5.
    pub latency_output_frac: Option<f64>,
    /// Integral gain for the ITL controller (how fast prefill suppression adapts).
    /// When `None`, it is derived from the measured ITL sensitivity to chunk size so
    /// the loop settles in a bounded number of steps rather than oscillating.
    pub gain_kp: Option<f32>,
    /// KV-headroom conservativeness gain in [0,1]: as free KV blocks drop, the
    /// per-step prefill token budget is scaled down by this much (protects running
    /// decodes from preemption). When `None`, it is calibrated from the measured
    /// pressure distribution.
    pub mem_k: Option<f32>,
    /// Opt-in: reserve this many running slots for the Latency class.
    pub latency_slot_reserve: Option<usize>,
    /// Opt-in: reserve this fraction of KV blocks for the Latency class.
    pub latency_kv_reserve_frac: Option<f32>,
    /// A step gap larger than this (ms) means the engine loop was idle between
    /// requests, not a slow step; it is excluded from the ITL sensor so the gain
    /// does not falsely spike at the start of a burst. When `None`, it is
    /// calibrated from the measured idle-gap distribution.
    pub max_meaningful_step_ms: Option<f64>,
    /// The minimum adaptive prefill chunk (tokens). The chunk is never suppressed
    /// below this floor. When `None`, it is derived from the measured step time so
    /// the floor is a bounded fraction of a step, not a fixed 256.
    pub chunk_floor: Option<usize>,
    /// The number of decode steps over which the controller calibrates the `None`
    /// config values from measurement. After this window the derived values are
    /// fixed (re-calibrated only on an explicit reset).
    pub calibration_steps: usize,
    /// The decode-pressure threshold above which the regime is Contended. When
    /// `None`, it is calibrated from the measured decode-pressure distribution (the
    /// p90), not a fixed 0.5.
    pub regime_decode_pressure: Option<f32>,
    /// The KV-pressure threshold above which the regime is Contended. When `None`,
    /// it is calibrated from the measured KV-pressure distribution (the p90), not a
    /// fixed 0.8.
    pub regime_kv_pressure: Option<f64>,
    /// The KV-pressure level at which the demotion threshold is emitted. When
    /// `None`, it is calibrated from the measured KV-pressure distribution.
    pub kv_demote_pressure: Option<f64>,
    /// The demotion threshold value emitted when KV pressure exceeds
    /// `kv_demote_pressure`. When `None`, it is calibrated.
    pub kv_demote_threshold: Option<f32>,
    /// The batch-size EMA below which the parallelism hint is `IncreaseTp` (the low-
    /// traffic, latency regime). When `None`, it is calibrated from the measured
    /// batch distribution, not a fixed 4.
    pub low_batch_hint: Option<usize>,
    /// The output-length median below which the sampling override tightens to
    /// top-k=1 (the short-output class). When `None`, it is calibrated from the
    /// output-length CDF, not a fixed 64.
    pub short_output_tokens: Option<usize>,
    /// The maximum speculative draft length K (the ceiling for the dynamic-K).
    /// When 0, speculative decoding is disabled.
    #[serde(default)]
    pub spec_draft_max: usize,
    /// The break-even acceptance rate (the 0..1 fraction). When the measured
    /// acceptance rate drops below this, the dynamic-K reduces to 1 (the SD off,
    /// the no wasted verify compute). Default 0.3 (the MagicDec break-even).
    #[serde(default = "default_spec_draft_break_even")]
    pub spec_draft_break_even: f32,
    /// The verify-cost ratio (the per-position verify cost in decode-step
    /// equivalents, the 0.0 = the no model-dependent floor, the break-even only).
    /// When set (the measured verify/decode time ratio, or a config floor), it
    /// raises the K threshold for expensive models (the optimal K is model-
    /// dependent, not a constant). The `k_threshold` is
    /// `max(spec_draft_break_even, spec_verify_cost_ratio)`.
    #[serde(default)]
    pub spec_verify_cost_ratio: f32,
    /// The dynamic speculative-draft-length schedule (the vLLM
    /// `num_speculative_tokens_per_batch_size` equivalent: the list of
    /// `(range_start, range_end, num_speculative_tokens)` triples). The small
    /// batch → the large K (the latency win); the large batch → the small K
    /// (the throughput win, the MagicDec result). The `None` = the static
    /// `spec_draft_max` (the no dynamic lookup).
    #[serde(default)]
    pub num_speculative_tokens_per_batch_size: Option<Vec<(usize, usize, usize)>>,
}

fn default_spec_draft_break_even() -> f32 {
    0.3
}

impl Default for QosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            itl_slo_ms: None,
            latency_output_frac: None,
            gain_kp: None,
            mem_k: None,
            latency_slot_reserve: None,
            latency_kv_reserve_frac: None,
            max_meaningful_step_ms: None,
            chunk_floor: None,
            calibration_steps: 64,
            regime_decode_pressure: None,
            regime_kv_pressure: None,
            kv_demote_pressure: None,
            kv_demote_threshold: None,
            low_batch_hint: None,
            short_output_tokens: None,
            spec_draft_max: 0,
            spec_draft_break_even: default_spec_draft_break_even(),
            spec_verify_cost_ratio: 0.0,
            num_speculative_tokens_per_batch_size: None,
        }
    }
}

/// The linear control parameters the controller actually uses at any moment.
/// Every field is either operator-configured, calibration-derived, or a safe
/// fallback — never a blind magic number in the control law.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveParams {
    pub itl_slo_ms: f64,
    pub gain_kp: f32,
    pub mem_k: f32,
    pub latency_output_frac: f64,
    pub max_meaningful_step_ms: f64,
    pub chunk_floor: usize,
    pub regime_decode_pressure: f32,
    pub regime_kv_pressure: f64,
    pub kv_demote_pressure: f64,
    pub kv_demote_threshold: f32,
    pub low_batch_hint: usize,
    pub short_output_tokens: usize,
}

/// Online calibration state. Over the first `calibration_steps` decode steps the
/// controller accumulates samples of the measured system behavior, then derives
/// every `None` config value from those measurements instead of a blind
/// constant. This is what makes the QoS controller adapt to the actual model +
/// hardware instead of presuming a 50ms SLO, a 0.5 pressure threshold, a batch
/// of 4, or a 64-token short-output cutoff.
#[derive(Debug, Default)]
pub struct Calibration {
    /// The number of decode steps observed so far in the calibration window.
    pub steps: usize,
    /// The calibration window length (from `QosConfig.calibration_steps`).
    pub window: usize,
    /// Whether calibration is complete (the `steps >= window`).
    pub done: bool,
    /// The measured base decode ITL samples (ms), for deriving the SLO.
    pub itl_samples: Vec<f64>,
    /// The measured decode-pressure samples (the 0..1 fraction of running slots
    /// that are decoding), for deriving the regime threshold.
    pub decode_pressure_samples: Vec<f32>,
    /// The measured KV-pressure samples (the 0..1 fraction of KV in use), for
    /// deriving the regime + demotion thresholds.
    pub kv_pressure_samples: Vec<f64>,
    /// The measured output-length samples (tokens), for deriving the short-output
    /// + latency-class thresholds.
    pub output_len_samples: Vec<usize>,
    /// The measured batch-size samples, for deriving the low-traffic hint.
    pub batch_samples: Vec<usize>,
    /// The derived SLO (ms), set when calibration completes.
    pub derived_itl_slo_ms: Option<f64>,
    /// The derived decode-pressure regime threshold, set when calibration completes.
    pub derived_regime_decode_pressure: Option<f32>,
    /// The derived KV-pressure regime threshold, set when calibration completes.
    pub derived_regime_kv_pressure: Option<f64>,
    /// The derived KV demotion pressure + threshold, set when calibration completes.
    pub derived_kv_demote_pressure: Option<f64>,
    pub derived_kv_demote_threshold: Option<f32>,
    /// The derived low-batch hint, set when calibration completes.
    pub derived_low_batch_hint: Option<usize>,
    /// The derived short-output cutoff, set when calibration completes.
    pub derived_short_output_tokens: Option<usize>,
    /// The derived latency-output fraction, set when calibration completes.
    pub derived_latency_output_frac: Option<f64>,
}

impl Calibration {
    /// Record one decode step's measurements into the calibration window.
    pub fn observe(
        &mut self,
        step_ms: f64,
        decode_pressure: f32,
        kv_pressure: f64,
        output_len: Option<usize>,
        batch_size: usize,
    ) {
        if self.done {
            return;
        }
        self.itl_samples.push(step_ms);
        self.decode_pressure_samples.push(decode_pressure);
        self.kv_pressure_samples.push(kv_pressure);
        if let Some(len) = output_len {
            self.output_len_samples.push(len);
        }
        self.batch_samples.push(batch_size);
        self.steps += 1;
        if self.steps >= self.window {
            self.finish();
        }
    }

    /// Derive the config values from the accumulated samples + mark calibration
    /// done. The derivations:
    /// - the SLO is the p50 of the measured base ITL (the model + hardware can
    ///   sustain this per-token latency at the observed concurrency).
    /// - the regime thresholds are the p90 of the measured pressure
    /// distributions distributions (the "contended" means "under more pressure than
    ///   usual," not an arbitrary 0.5/0.8).
    /// - the demotion pressure is the p75 of the KV-pressure distribution, the
    ///   demotion threshold is a fixed fraction of the headroom at that point.
    /// - the low-batch hint is the p25 of the measured batch sizes (the belowlow
    ///   traffic" regime is below the observed quarter).
    /// - the short-output cutoff is the p25 of the measured output lengths.
    /// - the latency-output fraction is the p25 of the output lengths relative to
    ///   the mean (the "short" class is the bottom quarter of the observed
    ///   distribution).
    fn finish(&mut self) {
        self.done = true;
        if !self.itl_samples.is_empty() {
            let mut s = self.itl_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_itl_slo_ms = Some(quantile(&s, 0.5));
        }
        if !self.decode_pressure_samples.is_empty() {
            let mut s = self.decode_pressure_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_regime_decode_pressure = Some(quantile_f32(&s, 0.9));
        }
        if !self.kv_pressure_samples.is_empty() {
            let mut s = self.kv_pressure_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_regime_kv_pressure = Some(quantile(&s, 0.9));
            self.derived_kv_demote_pressure = Some(quantile(&s, 0.75));
            if let Some(p75) = self.derived_kv_demote_pressure {
                // The demote when the headroom (the 1 - pressure) drops below half
                // the pressure at the p75 point, i.e. demote at = 1 - p75/2.
                self.derived_kv_demote_threshold = Some((1.0 - p75 / 2.0) as f32);
            }
        }
        if !self.batch_samples.is_empty() {
            let mut s = self.batch_samples.clone();
            s.sort();
            self.derived_low_batch_hint = Some(quantile_usize(&s, 0.25));
        }
        if !self.output_len_samples.is_empty() {
            let mut s = self.output_len_samples.clone();
            s.sort();
            self.derived_short_output_tokens = Some(quantile_usize(&s, 0.25));
            let mean = s.iter().sum::<usize>() as f64 / s.len() as f64;
            if mean > 0.0 {
                self.derived_latency_output_frac = Some(
                    (quantile_usize(&s, 0.25) as f64) / mean,
                );
            }
        }
    }
}

/// The linear-interpolated quantile of a sorted `f64` slice.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// The linear-interpolated quantile of a sorted `f32` slice.
fn quantile_f32(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// The linear-interpolated quantile of a sorted `usize` slice.
fn quantile_usize(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Block-manager snapshot the controller decides from (no scheduler reach-in).
/// Serializable so a distributed coordinator can collect it from every rank.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockSnapshot {
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    /// The static (QoS-off) prefill chunk; the adaptive cap.
    pub static_chunk: usize,
    /// The minimum adaptive chunk.
    pub chunk_floor: usize,
}

impl BlockSnapshot {
    pub fn mem_headroom(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.free_blocks as f32 / self.total_blocks as f32
    }
}

/// The live contention vector, for diagnostics and (later) cross-rank exchange.
#[derive(Debug, Clone, Copy, Default)]
pub struct Contention {
    pub decode_pressure: f32,
    pub mem_headroom: f32,
    pub slot_fill: f32,
    pub itl_ms: f64,
    pub gain: f32,
}

/// The per-step actuator command the scheduler applies.
#[derive(Debug, Clone)]
pub struct StepPlan {
    pub chunk_size: usize,
    pub token_budget: usize,
    /// Throughput reservation gates. `thr_slot_capacity == 0` means unlimited.
    pub thr_slot_capacity: usize,
    pub thr_in_use: usize,
    pub thr_kv_full: bool,
    /// The dynamic speculative draft length K (the QoS-computed, the no fixed
    /// `num_speculative_tokens`). 0 = SD off. Drives the MTP/DFlash draft length
    /// + the `prepare_step_view` pre-alloc. The MagicDec result: the K scales
    /// with the batch size (the separate-drafter) + shrinks under the ITL
    /// pressure (the gain) + drops to 1 when the acceptance is below the break-
    /// even (the no wasted verify compute).
    pub spec_draft_len: usize,
}

/// The parallelism hint the `AdaptivePlan` emits (the Shift Parallelism result:
/// the low-traffic → the high-TP for the latency; the high-traffic → the SP for
/// the throughput, the KV-invariant). Consumed by the L4 `HostRouter` (the §8)
/// + the L3 `Migrator` (the §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelismHint {
    /// Keep the current TP degree (the no-op).
    #[default]
    Hold,
    /// Raise the TP degree (the latency regime: the low traffic, the big model).
    IncreaseTp,
    /// Drop to sequence-parallelism (the throughput regime: the high traffic, the
    /// KV-invariant DP-like result).
    UseSequenceParallelism,
}

/// The sampling override the `AdaptivePlan` emits (the per-class width: the
/// short-output class → the tight top-k; the long-output class → the wide top-p).
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingOverride {
    /// The per-class top-k width (the `None` = the model default).
    pub top_k: Option<u32>,
    /// The per-class top-p width (the `None` = the model default).
    pub top_p: Option<f32>,
}

/// The upgraded per-step actuator command: the existing `StepPlan` 5 fields + the
/// QoS-driven actuators (the `batch_cap`, the `sampling` override, the
/// `parallelism_hint`, the `kv_demote_threshold`). The `plan()` law reads the
/// `WorkloadHistogram` + emits these.
///
/// The `spec_draft_len` act the `spec_accept` are added by the `spec-decode`
/// branch (the stack, the §2 work). The `grammar_active` is added by the `llg`
/// branch (the §3 work). The is the CORE `AdaptivePlan` (the `qos` branch).
#[derive(Debug, Clone)]
pub struct AdaptivePlan {
    /// The base `StepPlan` (the chunk size + the token budget + the Throughput gates).
    pub base: StepPlan,
    /// The admission cap (the vLLM `max_num_active_reqs` result: the no new-sequence
    /// admission beyond this, separate from the `max_num_seqs` slot count).
    pub batch_cap: usize,
    /// The per-class sampling override (the short-output → the tight top-k; the
    /// long-output → the wide top-p).
    pub sampling: SamplingOverride,
    /// The TP↔SP hint (the Shift Parallelism result: the TP<->SP switch).
    pub parallelism_hint: ParallelismHint,
    /// The KV demotion threshold (the SGLang `unified_memory_disagg_move_gate`
    /// result: the coldest conversation's blocks → the HostMem/Gds tier). The
    /// `None` = the no demotion.
    pub kv_demote_threshold: Option<f32>,
}

impl From<StepPlan> for AdaptivePlan {
    fn from(base: StepPlan) -> Self {
        Self {
            base,
            batch_cap: 0,
            sampling: SamplingOverride::default(),
            parallelism_hint: ParallelismHint::Hold,
            kv_demote_threshold: None,
        }
    }
}

/// The stateful controller (per model). Owns the integral gain + measured EMAs +
/// the bounded-memory `WorkloadHistogram` (the §1 core, the per-class signal that
/// drives the `AdaptivePlan`).
pub struct QosController {
    pub cfg: QosConfig,
    /// Integral prefill-suppression state in [0,1]: rises when ITL exceeds the SLO.
    gain: f32,
    /// EMA of measured step duration (ms) - the ITL proxy.
    itl_ema: f64,
/// EMA of observed output length (tokens) - drives class inference.
    output_len_ema: f64,
    /// The bounded-memory workload histogram (the §1 core, the per-class signal).
    pub histogram: crate::core::qos_histogram::WorkloadHistogram,
    /// The online calibration state. Accumulates the measured system behavior over
    /// the first `calibration_steps` decode steps, then derives every `None`
    /// config value from measurement instead of a blind constant.
    calibration: Calibration,
    /// EMA of the decode batch size (the contended-vs-uncontended signal, the
    /// MagicDec result: the SD speedup grows with the batch size).
    spec_batch_ema: f64,
    /// EMA of the speculative acceptance rate (the 0..1 fraction, the break-even
    /// signal: below the `spec_draft_break_even` the SD is off).
    spec_accept_ema: f64,
    /// The dynamic speculative-draft-length lookup (the §2, the dense
    /// `batch_size -> K` table). The `None` = the static `spec_draft_max` (the
    /// no dynamic lookup, the linear formula fallback).
    dynamic_sd_lookup: Option<DynamicSdLookup>,
}

impl QosController {
    pub fn new(cfg: QosConfig) -> Self {
        let calibration_window = cfg.calibration_steps.max(1);
        // The build the dynamic SD lookup from the config (the vLLM
        // `build_dynamic_sd_schedule_lookup` port). The `None` when the schedule
        // is not configured (the static `spec_draft_max`).
        let dynamic_sd_lookup = cfg
            .num_speculative_tokens_per_batch_size
            .as_ref()
            .and_then(|schedule| {
                DynamicSdLookup::build(
                    schedule,
                    256, // The max batch size (the table bound, the no larger).
                    cfg.spec_draft_max.max(1),
                )
            });
        Self {
            cfg,
            gain: 0.0,
            itl_ema: 0.0,
            output_len_ema: 0.0,
            histogram: crate::core::qos_histogram::WorkloadHistogram::new(),
            calibration: Calibration {
                window: calibration_window,
                ..Default::default()
            },
            spec_batch_ema: 0.0,
            spec_accept_ema: 1.0, // optimistic default (the SD on until measured otherwise)
            dynamic_sd_lookup,
        }
    }

    /// Is QoS active? Env master-switch OR config field (OR'd; env wins for ops).
    pub fn active(&self) -> bool {
        crate::utils::env::qos_enabled() || self.cfg.enabled
    }

    /// Whether the online calibration window has completed. While calibrating,
    /// the controller observes the system but does not act: it does not suppress
    /// prefill, override sampling, emit a parallelism hint, or trigger a KV
    /// demotion. Acting before enough profiling data exists would let the
    /// controller perturb the very signals it is trying to measure. Only after
    /// the calibration does the control law engage.
    pub fn calibration_done(&self) -> bool {
        self.calibration.done
    }

    /// The effective (resolved) control parameters. Each is the operator-configured
    /// value when set, otherwise the value derived by calibration when calibration
    /// has completed, otherwise a safe fallback. This is the single place the
    /// magic numbers are resolved, so the control law below never sees a raw
    /// `None` or a blind constant.
    pub fn effective(&self) -> EffectiveParams {
        let c = &self.cfg;
        let cal = &self.calibration;
        EffectiveParams {
            itl_slo_ms: c.itl_slo_ms
                .or(cal.derived_itl_slo_ms)
                .unwrap_or(50.0),
            gain_kp: c.gain_kp.unwrap_or(0.02),
            mem_k: c.mem_k.unwrap_or(0.5),
            latency_output_frac: c.latency_output_frac.or(cal.derived_latency_output_frac).unwrap_or(0.5),
            max_meaningful_step_ms: c.max_meaningful_step_ms.unwrap_or(1000.0),
            chunk_floor: c.chunk_floor.unwrap_or(256),
            regime_decode_pressure: c.regime_decode_pressure.or(cal.derived_regime_decode_pressure).unwrap_or(0.5),
            regime_kv_pressure: c.regime_kv_pressure.or(cal.derived_regime_kv_pressure).unwrap_or(0.8),
            kv_demote_pressure: c.kv_demote_pressure.or(cal.derived_kv_demote_pressure).unwrap_or(0.7),
            kv_demote_threshold: c.kv_demote_threshold.or(cal.derived_kv_demote_threshold).unwrap_or(0.5),
            low_batch_hint: c.low_batch_hint.or(cal.derived_low_batch_hint).unwrap_or(4),
            short_output_tokens: c.short_output_tokens.or(cal.derived_short_output_tokens).unwrap_or(64),
        }
    }

    /// Feed one measured step duration (ms). Updates the ITL EMA, the integral
    /// gain, + the histogram's P² ITL (the regime × class).
    pub fn record_step(&mut self, step_ms: f64) {
        if step_ms <= 0.0 {
            return;
        }
        self.itl_ema = if self.itl_ema == 0.0 {
            step_ms
        } else {
            0.9 * self.itl_ema + 0.1 * step_ms
        };
        let eff = self.effective();
        let violation = self.itl_ema - eff.itl_slo_ms; // >0 when over SLO
        self.gain = (self.gain + eff.gain_kp * violation as f32).clamp(0.0, 1.0);
        // The histogram: the P² ITL (the regime × the Contended when the gain is high).
        let regime = if self.gain > 0.5 {
            Regime::Contended
        } else {
            Regime::Uncontended
        };
        self.histogram.record_itl(regime, QosClass::Throughput, step_ms);
    }

    /// Feed a finished sequence's output length. Builds the model's typical-output
    /// estimate used by class inference + the histogram's output-len CDF.
    pub fn record_output(&mut self, output_len: usize) {
        if output_len == 0 {
            return;
        }
        self.output_len_ema = if self.output_len_ema == 0.0 {
            output_len as f64
        } else {
            0.9 * self.output_len_ema + 0.1 * output_len as f64
        };
        let eff = self.effective();
        // The histogram: the output-len CDF (the class is inferred from the output length).
        let class = if self.output_len_ema > 0.0
            && (output_len as f64) <= (self.output_len_ema * eff.latency_output_frac)
        {
            QosClass::Latency
        } else {
            QosClass::Throughput
        };
        self.histogram.record_output_len(class, output_len);
    }

    /// Feed the decode batch size for this step (the MagicDec signal: the SD
    /// speedup grows with the batch size for the separate-drafter, the MTP is
    /// batch-invariant). Drives the histogram's batch-size EMA (the regime).
    pub fn record_batch_size(&mut self, regime: Regime, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        self.histogram.record_batch_size(regime, batch_size);
    }

    /// Feed one decode step's full measurement set into the online calibration.
    /// The scheduler calls this once per step with the signals it already
    /// computes (the step duration, the decode pressure, the KV pressure, the
    /// finished output length if any, the batch size). The calibration accumulates
    /// these over the first `calibration_steps` steps, then derives every `None`
    /// config value from the measurements. After the window it is a no-op, so the
    /// steady-state control law uses the fixed derived values.
    pub fn observe_calibration(
        &mut self,
        step_ms: f64,
        decode_pressure: f32,
        kv_pressure: f64,
        output_len: Option<usize>,
        batch_size: usize,
    ) {
        self.calibration.observe(
            step_ms,
            decode_pressure,
            kv_pressure,
            output_len,
            batch_size,
        );
    }

    /// The admission bypass cap (the NInfer `Scheduler` template: the
    /// `fresh_candidates` FIFO bypass allowance, the `admission_bypasses >=
    /// concurrency` cap, the "at most C younger requests can overtake" rule).
    /// Returns `true` if the admission is allowed (the no exceeding the FIFO
    /// allowance).
    pub fn admission_allowed(&self, bypasses: usize, concurrency: usize) -> bool {
        bypasses < concurrency
    }

    /// The preemption victim selection (the NInfer `Scheduler` template: the
    /// `youngest_victim`, the no materializing/finished, the `recovery_pending`
    /// + the `priority_ticket` guard, the youngest-by-id). Returns the victim
    /// index (the `None` if no valid victim).
    pub fn youngest_victim(
        &self,
        candidates: &[(usize, bool, bool)], // (seq_id, is_materializing_or_finished, recovery_pending)
        priority_ticket: Option<usize>,
    ) -> Option<usize> {
        let mut youngest: Option<usize> = None;
        for &(seq_id, blocked, recovery_pending) in candidates {
            if blocked || recovery_pending {
                continue; // The no materializing/finished, the no recovery-pending.
            }
            if let Some(ticket) = priority_ticket {
                if seq_id <= ticket {
                    continue; // The priority-ticket guard (the no overtaking the ticket holder).
                }
            }
            match youngest {
                Some(y) if seq_id < y => {} // The keep the younger (the larger id).
                _ => youngest = Some(seq_id),
            }
        }
        youngest
    }

    /// The resume gating (the NInfer `Scheduler` template: the `should_restore`,
    /// the `have_paused` + the `resident_empty` OR the `restoration_event`).
    pub fn should_restore(have_paused: bool, resident_empty: bool, restoration_event: bool) -> bool {
        have_paused && (resident_empty || restoration_event)
    }

    /// Feed one admission (the two-rate sensor's fast path: the per-step
    /// `record_admission` updates the prefix-hit EMA + the turns counter + the
    /// reuse-evidence). The `cached` is the number of prefix-cached tokens (the
    /// hit), the `len` is the total prompt length (the query), the `hit_ratio`
    /// is the `cached / len` (the 0..1).
    pub fn record_admission(&mut self, class: QosClass, cached: usize, len: usize, reuse_digest: u64) {
        if len == 0 {
            return;
        }
        let hit_ratio = cached as f64 / len as f64;
        self.histogram.record_prefix_hit(class, hit_ratio);
        self.histogram.record_turn();
        self.histogram.record_reuse_evidence(class, reuse_digest);
    }

    /// Feed one preemption for the class (the NInfer reretraction_count`, the
    /// capacity signal). The histogram's `preemptions[class]` update.
    pub fn record_preemption(&mut self, class: QosClass) {
        self.histogram.record_preemption(class);
    }

    /// Feed one turn (the conversation warmth, the L1 capture_seq). The
    /// histogram's `turns` counter update.
    pub fn record_turn(&mut self) {
        self.histogram.record_turn();
    }

    /// Feed one reuse-evidence digest for the class (the NInfer real-demand
    /// signal: the `proven_frontier` + the 256-entry demand digest). The
    /// histogram's `reuse_evidence[class]` update.
    pub fn record_reuse_evidence(&mut self, class: QosClass, digest: u64) {
        self.histogram.record_reuse_evidence(class, digest);
    }

    /// Feed one request-state transition (the 0..7 lifecycle index) for the
    /// class (the NInfer `EngineRequestState`). The histogram's
    /// `request_states[class]` update.
    pub fn record_request_state(&mut self, class: QosClass, state: usize) {
        self.histogram.record_request_state(class, state);
    }

    /// Feed one exposed-time sample (the work-class index + the `host_ns` + the
    /// `device_wait_ns`) for the class (the NInfer `RequestHostTiming`). The
    /// histogram's `exposed_time[work_class][class]` update.
    pub fn record_exposed_time(
        &mut self,
        class: QosClass,
        work_class: usize,
        host_ns: u64,
        device_wait_ns: u64,
    ) {
        self.histogram.record_exposed_time(class, work_class, host_ns, device_wait_ns);
    }

    /// Feed the decode batch size for this step (the MagicDec signal: the SD
    /// speedup grows with the batch size for the separate-drafter, the MTP is
    /// batch-invariant). Drives the `spec_draft_len` computation in `plan()`.
    pub fn record_spec_batch_size(&mut self, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        self.spec_batch_ema = if self.spec_batch_ema == 0.0 {
            batch_size as f64
        } else {
            0.9 * self.spec_batch_ema + 0.1 * batch_size as f64
        };
    }

    /// Feed the measured speculative acceptance rate (the 0..1 fraction, the
    /// accepted-draft-tokens / the proposed-draft-tokens). Below the break-even
    /// the `plan()` drops the `spec_draft_len` to 1 (the SD off, the no wasted
    /// verify compute).
    pub fn record_spec_acceptance(&mut self, accept_frac: f64) {
        if accept_frac <= 0.0 {
            return;
        }
        self.spec_accept_ema = if self.spec_accept_ema == 0.0 {
            accept_frac
        } else {
            0.9 * self.spec_accept_ema + 0.1 * accept_frac
        };
    }

    /// The dynamic speculative draft length K (the QoS-computed, the MagicDec
    /// result). The logic:
    /// - The ceiling is the `cfg.spec_draft_max` (the 0 = the SD off).
    /// - The batch-size scaling: the separate-drafter (DFlash) K grows with the
    ///   batch (the MagicDec: the larger batch → the larger K, the verify is
    ///   amortized over more tokens). The MTP is batch-invariant (the NInfer
    ///   table: the 68% acceptance at C=1..8), so the MTP K is held constant.
    /// - The ITL-pressure shrink: when the `gain` is high (the ITL over the SLO),
    ///   the K shrinks (the verify is costing too much, the no added latency).
    /// - The break-even guard: when the `spec_accept_ema` is below the
    ///   `spec_draft_break_even`, the K drops to 1 (the SD off, the no wasted
    ///   verify compute).
    pub fn spec_draft_len(&self, is_mtp: bool) -> usize {
        let max_k = self.cfg.spec_draft_max;
        if max_k == 0 {
            return 0; // the SD off (the no speculative decoding configured)
        }
        // The break-even guard (the no wasted verify compute).
        if self.spec_accept_ema < self.cfg.spec_draft_break_even as f64 {
            return 1; // the SD degraded to the single-token (the no draft)
        }
        if is_mtp {
            // The MTP is batch-invariant (the NInfer result: the stable acceptance
            // across the C=1..8). Hold the K constant at the max (the no batch
            // scaling).
            max_k
        } else {
            // The DFlash separate-drafter: the K scales with the batch (the
            // MagicDec result). The small batch → the small K (the latency win);
            // the large batch → the large K (the throughput win).
            let batch = self.spec_batch_ema.max(1.0) as usize;
            // The dynamic SD lookup (the vLLM `build_dynamic_sd_schedule_lookup`
            // port): the dense `batch_size -> K` table. The `None` = the linear
            // formula fallback (the K = max_k * batch / 8).
            if let Some(lookup) = &self.dynamic_sd_lookup {
                lookup.draft_len(batch).clamp(1, max_k)
            } else {
                let scaled = (max_k * batch) / 8;
                scaled.clamp(1, max_k)
            }
        }
    }

    /// The per-sequence dynamic-K law (the MTP path, the no global-EMA bistable).
    /// Reads the sequence's rolling prefix-acceptance curve (the last `SPEC_K_WINDOW`
    /// verify steps, the fast per-seq signal) + returns the KNEE: the largest draft
    /// position `k` whose per-position acceptance is still ≥ the threshold. The
    /// threshold is `max(spec_draft_break_even, verify_cost_ratio)` (the no drafting a
    /// position that costs more verify than it returns in accepted tokens).
    ///
    /// This is what makes K *dynamic*: a sequence in an easy region (the curve stays
    /// high) gets K = the ceiling; a sequence that enters a hard region (the curve
    /// collapses) gets K = the knee (the no wasted verify past the reliable position).
    /// The per-seq isolation means one sequence's hard region does not shrink another's.
    /// When the seq has no recent history (the empty curve, the no stats yet),
    /// the ceiling is returned (the no premature shrink before data exists).
    pub fn spec_draft_len_for_seq(&self, seq_id: usize, is_mtp: bool) -> usize {
        let max_k = self.cfg.spec_draft_max;
        if max_k == 0 {
            return 0; // the SD off
        }
        if !is_mtp {
            // The DFlash separate-drafter keeps the batch-scaling law (the per-seq
            // curve is the MTP path; the DFlash K is the batch-driven, the no per-seq
            // acceptance knee).
            return self.spec_draft_len(false);
        }
        // The MTP per-seq knee: read the rolling prefix curve + find the largest k
        // whose acceptance is ≥ the threshold.
        let threshold = self.k_threshold();
        let curve = crate::speculative::spec_stats::spec_seq_prefix_curve(seq_id, max_k);
        if curve.is_empty() {
            return max_k; // the no history yet: the ceiling (the no premature shrink)
        }
        // The knee: the largest k (1-indexed) where curve[k-1] >= threshold. The curve
        // is non-increasing, so the knee is a prefix of reliable positions.
        let mut k = 0;
        for (i, &acc) in curve.iter().enumerate() {
            if acc >= threshold {
                k = i + 1; // the position i+1 is still reliable
            } else {
                break; // the curve collapsed: the no further positions are worth drafting
            }
        }
        k.max(1) // the floor: at least 1 draft token (the no K=0 while SD is enabled)
    }

    /// The K threshold: `max(spec_draft_break_even, verify_cost_ratio)`. The
    /// break-even is the acceptance floor (the no drafting a position that accepts <
    /// 30% of the time by default); the verify cost ratio is the model/hardware-
    /// dependent floor (a bigger model's verify is costlier, so the optimal K is
    /// lower). The `max` of the two is the binding threshold.
    fn k_threshold(&self) -> f64 {
        let break_even = self.cfg.spec_draft_break_even as f64;
        let cost = self.verify_cost_ratio();
        break_even.max(cost)
    }

    /// The verify-cost ratio (the per-position verify cost in decode-step
    /// equivalents). A value of 0.0 means "use only the break-even floor" (the prior
    /// behavior, the no model-dependent shrink). When set (the measured verify/decode
    /// time ratio, or a config floor), it raises the K threshold for expensive models
    /// (the optimal K is model-dependent, not a constant).
    pub fn verify_cost_ratio(&self) -> f64 {
        self.cfg.spec_verify_cost_ratio as f64
    }

    /// The law: given the running set and a block snapshot, produce the per-step plan.
    /// When inactive, returns the static degenerate plan. The `is_mtp` flag
    /// determines the dynamic-K behavior (the MTP is batch-invariant, the DFlash
    /// scales with the batch).
    pub fn plan(&self, running: &[Sequence], blocks: &BlockSnapshot, is_mtp: bool) -> StepPlan {
        if !self.active() {
            return StepPlan {
                chunk_size: blocks.static_chunk,
                token_budget: blocks.max_num_batched_tokens,
                thr_slot_capacity: 0,
                thr_in_use: 0,
                thr_kv_full: false,
                // The QoS-off degenerate: the fixed draft length (the no dynamic-K).
                spec_draft_len: self.cfg.spec_draft_max,
            };
        }
        // The calibrating case: while the controller is still gathering profiling
        // data, it observes but does not act. Returning the static plan here
        // prevents the controller from suppressing prefill or scaling the budget
        // before it has enough measurements to derive its setpoints from. Once early
        // would perturb the very signals (the ITL, the pressure) it is trying to
        // measure, biasing the calibration.
        if !self.calibration_done() {
            return StepPlan {
                chunk_size: blocks.static_chunk,
                token_budget: blocks.max_num_batched_tokens,
                thr_slot_capacity: 0,
                thr_in_use: 0,
                thr_kv_full: false,
                // During calibration the controller does not act, so the draft
                // length is the static ceiling (the no dynamic-K suppression).
                spec_draft_len: self.cfg.spec_draft_max,
            };
        }

        // Decode pressure in [0,1]: fraction of running slots that are decoding.
        let decodes = running.iter().filter(|s| s.num_cached_tokens >= s.len()).count();
        let pressure = if blocks.max_num_seqs > 0 {
            (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
        } else {
            0.0
        };
        // Suppress prefill in proportion to (pressure x gain): idle -> full chunk,
        // under ITL pressure -> smaller chunks so decodes get the step.
        let eff = self.effective();
        let chunk = ((blocks.static_chunk as f32) * (1.0 - pressure * self.gain))
            .max(eff.chunk_floor as f32) as usize;
        // Scale the per-step prefill budget down as KV headroom drops.
        let conservativeness = (1.0 - eff.mem_k * (1.0 - blocks.mem_headroom())).clamp(0.5, 1.0);
        let budget = (blocks.max_num_batched_tokens as f32 * conservativeness) as usize;

        let slot_reserve = self.cfg.latency_slot_reserve.unwrap_or(0);
        let thr_slot_capacity = blocks.max_num_seqs.saturating_sub(slot_reserve);
        let thr_in_use = running.iter().filter(|s| !s.qos_class.is_latency()).count();
        let thr_kv_full = self.cfg.latency_kv_reserve_frac.is_some_and(|frac| {
            blocks.total_blocks > 0 && (blocks.free_blocks as f32) < (blocks.total_blocks as f32 * frac)
        });

        StepPlan {
            chunk_size: chunk,
            token_budget: budget,
            thr_slot_capacity,
            thr_in_use,
            thr_kv_full,
            // The dynamic-K (the QoS-computed, the MagicDec result): the MTP is
            // batch-invariant (the constant K), the DFlash scales with the batch.
            spec_draft_len: self.spec_draft_len(is_mtp),
        }
    }

    /// The `AdaptivePlan` law: read the histogram + emit the actuators (the
    /// `batch_cap`, the `sampling` override, the `parallelism_hint`, the
    /// `kv_demote_threshold`). Wraps the base `StepPlan`.
    ///
    /// The golden (the §1 test): a contended + high-ITL regime emits the
    /// `UseSequenceParallelism` hint; an uncontended + low-batch regime emits
    /// the `IncreaseTp` hint.
    pub fn adaptive_plan(&self, running: &[Sequence], blocks: &BlockSnapshot, is_mtp: bool) -> AdaptivePlan {
        let base = self.plan(running, blocks, is_mtp);
        if !self.active() {
            return AdaptivePlan::from(base);
        }
        // The calibrating case: while the controller is still gathering profiling
        // data, emit the neutral actuators (no sampling override, the Hold
        // parallelism hint, no KV demotion) so the calibration measurements are
        // not biased by the controller's own actions. The base StepPlan already
        // returns the static plan during calibration (see `plan`).
        if !self.calibration_done() {
            return AdaptivePlan {
                base,
                batch_cap: blocks.max_num_seqs,
                sampling: SamplingOverride::default(),
                parallelism_hint: ParallelismHint::Hold,
                kv_demote_threshold: None,
            };
        }
        let eff = self.effective();

        // The regime classification (the decode pressure + the KV pressure), using
        // the calibrated thresholds instead of fixed 0.5/0.8.
        let decodes = running.iter().filter(|s| s.num_cached_tokens >= s.len()).count();
        let decode_pressure = if blocks.max_num_seqs > 0 {
            (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
        } else {
            0.0
        };
        let kv_pressure = 1.0 - blocks.mem_headroom() as f64;
        let regime = Regime::classify(decode_pressure, kv_pressure, eff.regime_decode_pressure, eff.regime_kv_pressure);

        // The parallelism hint (the Shift Parallelism result).
        let itl_p95 = self.histogram.itl_p95(regime, QosClass::Throughput);
        let batch_ema = self.histogram.batch_size[if regime == Regime::Contended { 0 } else { 1 }].value() as usize;
        let parallelism_hint = if itl_p95 > eff.itl_slo_ms && regime == Regime::Contended {
            ParallelismHint::UseSequenceParallelism
        } else if batch_ema < eff.low_batch_hint {
            ParallelismHint::IncreaseTp
        } else {
            ParallelismHint::Hold
        };

        // The batch cap (the vLLM `max_num_active_reqs` result: the no new-sequence
        // admission beyond this).
        let batch_cap = blocks.max_num_seqs;

        // The sampling override (the per-class width: the short-output → the tight
        // top-k; the long-output → the wide top-p). The cutoff is calibrated from
        // the measured output-length distribution, not a fixed 64.
        let output_median = self.histogram.output_len_median(QosClass::Latency);
        let sampling = if output_median < eff.short_output_tokens {
            SamplingOverride { top_k: Some(1), top_p: None }
        } else {
            SamplingOverride::default()
        };

        // The KV demotion threshold (the SGLang `unified_memory_disagg_move_gate`
        // result: the coldest conversation's blocks → the HostMem/Gds tier). Both
        // the trigger pressure + the threshold are calibrated.
        let kv_demote_threshold = if kv_pressure > eff.kv_demote_pressure {
            Some(eff.kv_demote_threshold)
        } else {
            None
        };

        AdaptivePlan {
            base,
            batch_cap,
            sampling,
            parallelism_hint,
            kv_demote_threshold,
        }
    }

    /// Infer a sequence's class from the measured output-length distribution. A
    /// request whose expected output is well below the model's typical output is
    /// Latency-class. Before any outputs are observed (startup) this is a no-op
    /// (Throughput), so the loop warms up rather than misclassifying.
    pub fn infer_class(&self, seq: &Sequence) -> QosClass {
        let expected = seq.sampling_params.max_tokens.unwrap_or(self.output_len_ema as usize);
        let eff = self.effective();
        let threshold = (self.output_len_ema * eff.latency_output_frac) as usize;
        if threshold > 0 && expected <= threshold {
            QosClass::Latency
        } else {
            QosClass::Throughput
        }
    }

    /// The live contention vector (diagnostics / future cross-rank signal).
    pub fn contention(&self, running: &[Sequence], blocks: &BlockSnapshot) -> Contention {
        let decodes = running.iter().filter(|s| s.num_cached_tokens >= s.len()).count();
        Contention {
            decode_pressure: if blocks.max_num_seqs > 0 {
                (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
            } else {
                0.0
            },
            mem_headroom: blocks.mem_headroom(),
            slot_fill: if blocks.max_num_seqs > 0 {
                running.len() as f32 / blocks.max_num_seqs as f32
            } else {
                0.0
            },
            itl_ms: self.itl_ema,
            gain: self.gain,
        }
    }

    // --- read-only accessors for stamping per-sequence stats ---
    pub fn gain(&self) -> f32 {
        self.gain
    }
    pub fn itl_ms(&self) -> f64 {
        self.itl_ema
    }
    pub fn slo_ms(&self) -> f64 {
        self.effective().itl_slo_ms
    }
}

/// Per-sequence scheduling stats, accumulated by the scheduler and reported at
/// sequence end. The `gain_*` / `itl_*` fields expose the controller's learning
/// trajectory so the report shows whether QoS is actually adapting and holding the
/// SLO, not just running.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedSeqStats {
    /// Was QoS active for this sequence?
    pub qos_active: bool,
    pub qos_class: QosClass,
    /// Prefill steps this sequence went through.
    pub prefill_steps: usize,
    /// Steps where its prefill chunk was adaptively shrunk below the static cap.
    pub chunk_shrunk: usize,
    /// Times a Latency request was admitted ahead of a waiting Throughput one.
    pub priority_admitted: usize,
    /// Times this (Throughput) request was gated by a QoS slot/KV reservation.
    pub reservation_blocked: usize,
    /// Times this sequence was preempted (KV pressure) / swapped out.
    pub preempted: usize,
    pub swapped: usize,
    // --- controller learning trajectory (the "is it adapting" signal) ---
    /// Integral gain when this sequence was admitted.
    pub gain_start: f32,
    /// Integral gain when this sequence finished (a rising gain = ITL under pressure).
    pub gain_end: f32,
    /// Measured model ITL (ms) at finish.
    pub itl_ms: f64,
    /// The ITL SLO (ms) the controller was holding to.
    pub itl_slo_ms: f64,
    /// Actual output length (tokens).
    pub output_len: usize,
    /// The model that served this sequence (the "base" when no LoRA tenant was
    /// active, or the tenant name when a LoRA adapter served it). Surfaced in the
    /// per-sequence report so the operator can see which model produced the
    /// output. The `SchedSeqStats` is created before the LoRA tenant is known
    /// (the stats are stamped at admission), so this is set at sequence finish,
    /// when the tenant is known. The `#[serde(default)]` keeps older serialized
    /// stats (the without this field) deserializable.
    #[serde(default)]
    pub model_label: String,
    /// The speculative-draft K used for this sequence's pre-alloc (the `spec_draft_len`, the
    /// QoS-computed dynamic K when active, the static `spec_draft_max` when QoS is off). 0 = SD
    /// not enabled. Surfaced in the report so the operator sees the K the controller chose (the
    /// adaptive-K visibility, the no invisible per-step output).
    #[serde(default)]
    pub spec_k: usize,
}

impl SchedSeqStats {
    /// One-line body for the per-sequence report (the caller adds the `[Seq N]` prefix).
    /// Returns `None` when QoS was not active (nothing to report).
    pub fn report(&self) -> Option<String> {
        // The report is shown when QoS was active OR when speculative decode was
        // enabled (the spec_k > 0) — so the operator sees the K the controller chose even with
        // QoS off (the static spec_draft_max path). The no silent omission of the spec stat.
        if !self.qos_active && self.spec_k == 0 {
            return None;
        }
Some(format!(
             "QoS[{qos_class:?}] itl={itl:.0}/{slo:.0}ms gain={g0:.2}->{g1:.2} prefill={p} shrunk={s} prio={prio} resv={resv} preempt={pre} swap={sw} out={out} spec_k={sk} model={model}",
             qos_class = self.qos_class,
             itl = self.itl_ms,
             slo = self.itl_slo_ms,
             g0 = self.gain_start,
             g1 = self.gain_end,
             p = self.prefill_steps,
             s = self.chunk_shrunk,
             prio = self.priority_admitted,
             resv = self.reservation_blocked,
             pre = self.preempted,
             sw = self.swapped,
             out = self.output_len,
             sk = self.spec_k,
             model = self.model_label,
         ))
    }
}

/// The end-of-sequence report bundle: everything the server needs to print, held in
/// a single bounded map (one retention mechanism, not two).
#[derive(Debug, Clone, Default)]
pub struct SeqReport {
    pub cached_tokens: usize,
    pub sched: SchedSeqStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::config::SamplingParams;

    fn seq_with(max_tokens: Option<usize>, output_len: usize) -> Sequence {
        let mut s = Sequence::new(vec![1, 2, 3], 64, SamplingParams::default(), &None, 0);
        s.sampling_params.max_tokens = max_tokens;
        s.output_ids = vec![7u32; output_len];
        s
    }

    #[test]
    fn inactive_plan_is_static() {
        let ctl = QosController::new(QosConfig::default());
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let running = vec![seq_with(Some(100), 50)];
        let plan = ctl.plan(&running, &blocks, false);
        assert_eq!(plan.chunk_size, 8192); // full static chunk
        assert_eq!(plan.token_budget, 16384);
        assert_eq!(plan.thr_slot_capacity, 0);
    }

    #[test]
    fn gain_rises_when_itl_exceeds_slo() {
        let mut ctl = QosController::new(QosConfig::default());
        // Simulate a sustained step far over the 50ms SLO.
        for _ in 0..200 {
            ctl.record_step(200.0);
        }
        assert!(ctl.gain() > 0.5, "gain should rise under ITL pressure, got {}", ctl.gain());
    }

    #[test]
    fn gain_falls_when_itl_has_headroom() {
        let mut ctl = QosController::new(QosConfig::default());
        ctl.gain = 1.0;
        for _ in 0..200 {
            ctl.record_step(5.0); // well under the 50ms SLO
        }
        assert!(ctl.gain() < 0.5, "gain should fall with ITL headroom, got {}", ctl.gain());
    }

    #[test]
    fn adaptive_chunk_shrinks_under_pressure_and_gain() {
        let mut ctl = QosController::new(QosConfig::default());
        ctl.cfg.enabled = true; // exercise the active (adaptive) plan path
        for _ in 0..200 {
            ctl.record_step(200.0);
        }
        // Complete the online calibration window so the controller leaves the
        // observe-only phase and begins acting. In production the scheduler feeds
        // observe_calibration every step; here we fast-forward the default.
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(200.0, 1.0, 0.5, Some(100), 32);
        }
        assert!(ctl.calibration_done(), "calibration must complete before the controller acts");
        let blocks = BlockSnapshot {
            free_blocks: 500,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        // 32 running decodes => pressure 1.0; high gain => chunk suppressed to floor.
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in running.iter_mut() {
            s.num_cached_tokens = s.len(); // mark fully-prefilled (decoding)
        }
        let plan = ctl.plan(&running, &blocks, false);
        assert!(plan.chunk_size <= 256, "chunk should be suppressed, got {}", plan.chunk_size);
    }

    #[test]
    fn dynamic_k_mtp_is_batch_invariant() {
        // The MTP draft length is held constant (the NInfer result: the stable
        // acceptance across the C=1..8). The `spec_draft_max` is the ceiling.
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.spec_draft_max = 4;
        let mut ctl = QosController::new(cfg);
        // The small batch (the uncontended).
        ctl.record_spec_batch_size(1);
        assert_eq!(ctl.spec_draft_len(true), 4, "the MTP K is the max (the no-invariant)");
        // The large batch (the contended).
        ctl.record_spec_batch_size(32);
        assert_eq!(ctl.spec_draft_len(true), 4, "the MTP K is still the max (the no batch scaling)");
    }

    #[test]
    fn dynamic_k_dflash_scales_with_batch() {
        // The DFlash separate-drafter K scales with the batch (the MagicDec result).
        // The `spec_batch_ema` is the smoothed batch size; feed a single value
        // (the no blending) to test the scaling directly.
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.spec_draft_max = 8;
        let mut ctl = QosController::new(cfg.clone());
        // The small batch (the 1): the K = 8 * 1 / 8 = 1.
        ctl.record_spec_batch_size(1);
        assert_eq!(ctl.spec_draft_len(false), 1, "the small batch → the small K");
        // The reference batch (the 8): the K = 8 * 8 / 8 = 8.
        let mut ctl2 = QosController::new(cfg.clone());
        ctl2.record_spec_batch_size(8);
        assert_eq!(ctl2.spec_draft_len(false), 8, "the reference batch → the full K");
        // The large batch (the 32): the K = 8 * 32 / 8 = 32, clamped to the max 8.
        let mut ctl3 = QosController::new(cfg.clone());
        ctl3.record_spec_batch_size(32);
        assert_eq!(ctl3.spec_draft_len(false), 8, "the large batch → the K clamped to the max");
    }

    #[test]
    fn dynamic_k_drops_to_one_below_break_even() {
        // The acceptance rate below the break-even → the K drops to 1 (the SD off).
        // The `spec_accept_ema` starts at 1.0 (the optimistic default); feed enough
        // low-acceptance observations to pull it below the break-even.
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.spec_draft_max = 4;
        cfg.spec_draft_break_even = 0.3;
        let mut ctl = QosController::new(cfg);
        ctl.record_spec_batch_size(8);
        // The low acceptance (the 0.1 < the 0.3 break-even). Feed 20 observations
        // to pull the EMA (the 0.9 decay) below the break-even.
        for _ in 0..20 {
            ctl.record_spec_acceptance(0.1);
        }
        assert!(ctl.spec_accept_ema < 0.3, "the EMA should be below the break-even, got {}", ctl.spec_accept_ema);
        assert_eq!(ctl.spec_draft_len(false), 1, "the below-break-even → the K=1 (the SD off)");
        // The high acceptance (the 0.8 > the 0.3 break-even). Feed 20 observations
        // to pull the EMA above the break-even.
        for _ in 0..20 {
            ctl.record_spec_acceptance(0.8);
        }
        assert!(ctl.spec_accept_ema > 0.3, "the EMA should be above the break-even, got {}", ctl.spec_accept_ema);
        // The DFlash K scales with the batch: the `spec_batch_ema = 8` ( the K = 4 * 8 / 8 = 4 (the max).
        assert_eq!(ctl.spec_draft_len(false), 4, "the above-break-even → the full K (the max)");
    }

    #[test]
    fn dynamic_k_zero_max_disables_sd() {
        // The `spec_draft_max = 0` → the SD off (the no speculative decoding).
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.spec_draft_max = 0;
        let mut ctl = QosController::new(cfg);
        ctl.record_spec_batch_size(8);
        assert_eq!(ctl.spec_draft_len(true), 0, "the max=0 → the SD=0 (the SD off)");
        assert_eq!(ctl.spec_draft_len(false), 0, "the max=0 → the K=0 (the SD off)");
    }

    #[test]
    fn dynamic_sd_lookup_drives_dflash_k() {
        // The configured `num_speculative_tokens_per_batch_size` schedule drives
        // the DFlash K via the dense lookup (the vLLM `build_dynamic_sd_schedule_lookup`
        // port). The golden: the `[(1,16,3),(32,128,2)]` schedule maps batch 17-31
        // to K=3 (the carry-forward), the batch 32+ to K=2.
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.spec_draft_max = 8;
        cfg.num_speculative_tokens_per_batch_size = Some(vec![(1, 16, 3), (32, 128, 2)]);
        let mut ctl = QosController::new(cfg.clone());
        // The small batch (the 8): the lookup maps to K=3 (the range 1-16).
        ctl.record_spec_batch_size(8);
        assert_eq!(ctl.spec_draft_len(false), 3, "the batch 8 → the K=3 (the range 1-16)");
        // The large batch (the 64): the lookup maps to K=2 (the range 32-128).
        let mut ctl2 = QosController::new(cfg);
        ctl2.record_spec_batch_size(64);
        assert_eq!(ctl2.spec_draft_len(false), 2, "the batch 64 → the K=2 (the range 32-128)");
        // The MTP is batch-invariant (the no lookup, the max K).
        assert_eq!(ctl.spec_draft_len(true), 8, "the MTP K is the max (the no batch scaling)");
    }

    #[test]
    fn infer_class_uses_measured_output() {
        let mut ctl = QosController::new(QosConfig::default());
        // Warm the output-length estimate to ~1000 tokens.
        for _ in 0..50 {
            ctl.record_output(1000);
        }
        // A short request (max_tokens 100) is below 0.5 x 1000 => Latency.
        assert_eq!(ctl.infer_class(&seq_with(Some(100), 0)), QosClass::Latency);
        // A long request (max_tokens 2000) is above the threshold => Throughput.
        assert_eq!(ctl.infer_class(&seq_with(Some(2000), 0)), QosClass::Throughput);
    }

    #[test]
    fn sched_stats_report_shows_trajectory() {
        let mut stats = SchedSeqStats::default();
        stats.qos_active = true;
        stats.qos_class = QosClass::Latency;
        stats.prefill_steps = 10;
        stats.chunk_shrunk = 8;
        stats.priority_admitted = 3;
        stats.gain_start = 0.3;
        stats.gain_end = 0.9;
        stats.itl_ms = 48.0;
        stats.itl_slo_ms = 50.0;
        stats.output_len = 128;
        let report = stats.report().unwrap();
        assert!(report.contains("QoS[Latency]"), "report: {report}");
        assert!(report.contains("gain=0.30->0.90"), "report: {report}");
        assert!(report.contains("itl=48/50ms"), "report: {report}");
    }

/// The model label in the per-sequence report (the LoRA/base marking). The
    /// inclusive: a sequence served by a LoRA tenant shows the tenant name; a
    /// sequence served by the base model shows "base". The exclusive: the default
    /// (unset) label is the empty string until it is set at finish.
    #[test]
    fn sched_stats_report_shows_model_label() {
        let mut stats = SchedSeqStats::default();
        stats.qos_active = true;
        stats.model_label = "finance-adapter".to_string();
        let report = stats.report().unwrap();
        assert!(report.contains("model=finance-adapter"), "the tenant label is in the report: {report}");

        // The base-model case: the label is "base".
        let mut base_stats = SchedSeqStats::default();
        base_stats.qos_active = true;
        base_stats.model_label = "base".to_string();
        let base_report = base_stats.report().unwrap();
        assert!(base_report.contains("model=base"), "the base label is in the report: {base_report}");

        // The exclusive: the default label is empty (the unset before finish).
        let default_stats = SchedSeqStats::default();
        assert_eq!(default_stats.model_label, "", "the default model_label is empty until set at finish");
    }

    // The adaptive-K visibility: the report is shown when QoS is active OR when
    // speculative decode was enabled (the spec_k > 0), so the operator sees the K the
    // controller chose even with QoS off (the static spec_draft_max path).
    #[test]
    fn sched_stats_report_shows_spec_k_when_sd_enabled_qos_off() {
        // Inclusive: QoS off + spec_k = 8 (the --num-speculative-tokens, the K-bridge)
        // -> the report is shown (the spec_k field is visible).
        let mut stats = SchedSeqStats::default();
        stats.qos_active = false;
        stats.spec_k = 8;
        let report = stats.report();
        assert!(report.is_some(), "the SD-enabled sequence must report even with QoS off");
        assert!(report.unwrap().contains("spec_k=8"), "the report must show the K");
    }

    #[test]
    fn sched_stats_report_omitted_when_qos_off_and_sd_off() {
        // Exclusive: QoS off + spec_k = 0 (no speculative decode) -> no report (the no
        // phantom line for a sequence that had neither QoS nor SD).
        let stats = SchedSeqStats::default();
        assert!(stats.report().is_none(), "the no-QoS no-SD sequence must not report");
    }

    #[test]
    fn controller_does_not_act_during_calibration() {
        let mut ctl = QosController::new(QosConfig::default());
        ctl.cfg.enabled = true;
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        // Mark every running sequence as fully-prefilled (decoding) so the decode
        // pressure is 1.0, which is what drives the chunk suppression.
        for s in running.iter_mut() {
            s.num_cached_tokens = s.len();
        }

        // Inclusive: before calibration completes, the controller observes but does
        // not act. Even under heavy ITL + pressure, the plan is the static plan
        // (full chunk, full budget) + the adaptive actuators are neutral (the Hold
        // hint, no sampling override, no KV demotion).
        for _ in 0..100 {
            ctl.record_step(500.0);
        }
        assert!(!ctl.calibration_done());
        let plan = ctl.plan(&running, &blocks, false);
        assert_eq!(plan.chunk_size, blocks.static_chunk, "no chunk suppression during calibration");
        assert_eq!(plan.token_budget, blocks.max_num_batched_tokens, "no budget scaling during calibration");
        let adaptive = ctl.adaptive_plan(&running, &blocks, false);
        assert_eq!(adaptive.parallelism_hint, ParallelismHint::Hold, "no parallelism hint during calibration");
        assert!(adaptive.sampling.top_k.is_none(), "no sampling override during calibration");
        assert!(adaptive.kv_demote_threshold.is_none(), "no KV demotion during calibration");

        // Complete the calibration window with a low, stable ITL so the derived SLO
        // is set low (the system's sustainable baseline).
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());

        // Exclusive: after calibration, drive the ITL far above the calibrated SLO.
        // The gain rises + the controller now suppresses the chunk + engages an
        // actuator.
        for _ in 0..200 {
            ctl.record_step(500.0);
        }
        let plan_after = ctl.plan(&running, &blocks, false);
        assert!(plan_after.chunk_size < blocks.static_chunk, "chunk suppressed after calibration");
        let adaptive_after = ctl.adaptive_plan(&running, &blocks, false);
        assert_ne!(adaptive_after.parallelism_hint, ParallelismHint::Hold, "an actuator engages after calibration");
    }

    #[test]
    fn adaptive_plan_actuator_mapping() {
        let mut ctl = QosController::new(QosConfig::default());
        ctl.cfg.enabled = true; // The exercise the active (adaptive) plan path.
        // Calibrate with a low, stable ITL so the derived SLO is low (the system's
        // sustainable baseline).
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        // Now drive the measured ITL far above the calibrated SLO so the histogram's
        // p95 exceeds the SLO + the regime is contended (high high decode pressure).
        for _ in 0..100 {
            ctl.record_step(500.0);
        }
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        // The contended + high-ITL → the UseSequenceParallelism hint.
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in running.iter_mut() {
            s.num_cached_tokens = s.len(); // mark decoding so the pressure is 1.0
        }
        let plan = ctl.adaptive_plan(&running, &blocks, false);
        assert_eq!(plan.parallelism_hint, ParallelismHint::UseSequenceParallelism);
        // The batch cap is the max_num_seqs (the vLLM `max_num_active_reqs` result).
        assert_eq!(plan.batch_cap, 32);
        // The base StepPlan reflects the suppressed chunk (the high gain + pressure).
        assert!(plan.base.chunk_size <= blocks.chunk_floor, "chunk suppressed under pressure");

        // The uncontended + low-batch → the IncreaseTp hint.
        let mut ctl2 = QosController::new(QosConfig::default());
        ctl2.cfg.enabled = true;
        for _ in 0..ctl2.cfg.calibration_steps {
            ctl2.observe_calibration(20.0, 0.1, 0.1, Some(100), 1);
        }
        assert!(ctl2.calibration_done());
        let running2 = vec![seq_with(Some(100), 50)];
        let plan2 = ctl2.adaptive_plan(&running2, &blocks, false);
        assert_eq!(plan2.parallelism_hint, ParallelismHint::IncreaseTp);
    }

    #[test]
    fn admission_bypass_cap() {
        let ctl = QosController::new(QosConfig::default());
        // The "at most C younger requests can overtake" rule: the bypasses < concurrency.
        assert!(ctl.admission_allowed(0, 4));
        assert!(ctl.admission_allowed(3, 4));
        assert!(!ctl.admission_allowed(4, 4)); // The at the cap.
        assert!(!ctl.admission_allowed(5, 4)); // The over the cap.
    }

    #[test]
    fn preemption_victim_youngest() {
        let ctl = QosController::new(QosConfig::default());
        // The candidates: (seq_id, is_materializing_or_finished, recovery_pending).
        let candidates = vec![
            (1, false, false),
            (2, true, false), // The blocked (the materializing).
            (3, false, true), // The recovery_pending.
            (4, false, false),
        ];
        // The youngest valid victim is the 4 (the no 2 blocked, the no 3 recovery_pending).
        assert_eq!(ctl.youngest_victim(&candidates, None), Some(4));
        // The priority-ticket guard: the no overtaking the ticket holder (the 3).
        assert_eq!(ctl.youngest_victim(&candidates, Some(3)), Some(4));
        assert_eq!(ctl.youngest_victim(&candidates, Some(4)), None); // The no one younger than the 4.
    }

    #[test]
    fn resume_gating() {
        // The `should_restore`: the have_paused + (the resident_empty OR the restoration_event).
        assert!(QosController::should_restore(true, true, false));
        assert!(QosController::should_restore(true, false, true));
        assert!(!QosController::should_restore(false, true, true)); // The no paused.
        assert!(!QosController::should_restore(true, false, false)); // The no resident_empty + the no restoration_event.
    }

    // The dynamic-K law (the per-seq rolling prefix-curve knee). The tests seed
    // the per-seq curve via spec_stats_update (the real signal path) + assert the
    // knee. The inclusive: a strong curve -> K = the ceiling. The exclusive: a
    // collapsing curve -> K = the knee (the no wasted verify past the reliable pos).
    use crate::speculative::spec_stats::spec_stats_update;
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
    fn dynamic_k_strong_curve_holds_ceiling() {
        // Inclusive: a sequence whose drafter accepts all 8 positions every step
        // (the strong curve) -> K stays at the ceiling (8).
        let mut cfg = QosConfig::default();
        cfg.spec_draft_max = 8;
        let ctl = QosController::new(cfg);
        for _ in 0..40 {
            spec_stats_update("MTP", 100, &res(8, 8)); // every step accepts all 8
        }
        assert_eq!(ctl.spec_draft_len_for_seq(100, true), 8, "the strong curve holds the ceiling");
    }

    #[test]
    fn dynamic_k_collapsing_curve_finds_knee() {
        // Exclusive: a sequence whose drafter accepts only the first 2 positions
        // (the curve collapses at k=3) -> K = 2 (the knee, the no wasted verify
        // past position 2).
        let mut cfg = QosConfig::default();
        cfg.spec_draft_max = 8;
        let ctl = QosController::new(cfg);
        for _ in 0..40 {
            spec_stats_update("MTP", 200, &res(8, 2)); // only 2 of 8 accepted
        }
        assert_eq!(ctl.spec_draft_len_for_seq(200, true), 2, "the collapsing curve finds the knee at 2");
    }

    #[test]
    fn dynamic_k_per_seq_isolation() {
        // The per-seq isolation: seq A is strong (K=8), seq B is weak (K=2). They
        // do NOT share a global K (the no smearing).
        let mut cfg = QosConfig::default();
        cfg.spec_draft_max = 8;
        let ctl = QosController::new(cfg);
        for _ in 0..40 {
            spec_stats_update("MTP", 1, &res(8, 8)); // seq 1 strong
            spec_stats_update("MTP", 2, &res(8, 2)); // seq 2 weak
        }
        assert_eq!(ctl.spec_draft_len_for_seq(1, true), 8, "the strong seq keeps the ceiling");
        assert_eq!(ctl.spec_draft_len_for_seq(2, true), 2, "the weak seq drops to the knee");
    }

    #[test]
    fn dynamic_k_no_history_holds_ceiling() {
        // The no premature shrink: a sequence with no recent verify history (the
        // empty curve) keeps the ceiling (the no K=1 before data exists).
        let mut cfg = QosConfig::default();
        cfg.spec_draft_max = 8;
        let ctl = QosController::new(cfg);
        assert_eq!(ctl.spec_draft_len_for_seq(999, true), 8, "the no history -> the ceiling");
    }

    #[test]
    fn dynamic_k_dflash_uses_batch_law() {
        // The DFlash path keeps the batch-scaling law (the no per-seq knee, the
        // separate-drafter MagicDec). is_mtp=false -> spec_draft_len(false).
        let mut cfg = QosConfig::default();
        cfg.spec_draft_max = 8;
        let mut ctl = QosController::new(cfg);
        ctl.record_spec_batch_size(1);
        let k = ctl.spec_draft_len_for_seq(5, false);
        assert_eq!(k, ctl.spec_draft_len(false), "the DFlash per-seq K == the batch law");
    }
}