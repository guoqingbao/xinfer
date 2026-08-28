// QoS (quality-of-service) scheduling for the colocated (non-disaggregated)
// engine. Distinguishes latency-sensitive requests (agentic, short, many-turn)
// from throughput-oriented requests (large prompts, long outputs) and drives
// the scheduler's adaptive decisions from a live contention vector:
//
//   1. chunk sizing   - class-weighted decode load shrinks prefill chunks
//   2. admission order- Latency-class requests are served ahead of Throughput
//   3. reservations    - opt-in slot / KV reserves guarantee Latency resources
//   4. conservativeness- SGLang-style budget multiplier under load
//
// Grounded in vLLM (request priority + preemption, num_preemptions tie-break)
// and SGLang (schedule_conservativeness, PrefillAdder token budgeting).

use crate::core::sequence::Sequence;
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

/// QoS configuration (the knobs). Serde-defaulted so existing configs keep
/// working; `Default` enables the safe, high-value axes (class-weighted chunk
/// sizing + priority admission) and leaves reservations off.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QosConfig {
    pub enabled: bool,
    /// A Latency decode counts this much toward prefill-chunk pressure.
    pub latency_weight: f32,
    /// A Throughput decode counts this much.
    pub throughput_weight: f32,
    /// SGLang-style conservativeness in (0, 1]; 1.0 = current admission,
    /// <1.0 admits fewer tokens per step under load (protects running decodes).
    pub conservativeness: f32,
    /// Opt-in: reserve this many running slots for the Latency class.
    pub latency_slot_reserve: Option<usize>,
    /// Opt-in: reserve this fraction of KV blocks for the Latency class.
    pub latency_kv_reserve_frac: Option<f32>,
    /// Inference threshold: a request with max_tokens <= this is Latency-class.
    pub latency_max_tokens: usize,
}

impl Default for QosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            latency_weight: 2.0,
            throughput_weight: 1.0,
            conservativeness: 1.0,
            latency_slot_reserve: None,
            latency_kv_reserve_frac: None,
            latency_max_tokens: 1024,
        }
    }
}

impl QosConfig {
    /// Infer a class from a request's max output tokens (short output => Latency).
    pub fn infer_class(&self, max_tokens: Option<usize>) -> QosClass {
        match max_tokens {
            Some(mt) if mt <= self.latency_max_tokens => QosClass::Latency,
            _ => QosClass::Throughput,
        }
    }

    /// Class-weighted decode load over the running set (drives chunk sizing).
    /// Only fully-prefilled (decoding) sequences contribute; a Latency decode weighs
    /// more so prefill chunks shrink harder to protect its ITL.
    pub fn weighted_decode_load(&self, running: &[Sequence]) -> f32 {
        running
            .iter()
            .filter(|s| s.num_cached_tokens >= s.len())
            .fold(0.0f32, |acc, s| {
                acc + if s.qos_class.is_latency() {
                    self.latency_weight
                } else {
                    self.throughput_weight
                }
            })
    }

    /// Adaptive prefill chunk size: shrink as (class-weighted) decode load rises.
    /// `load <= 0` returns `cap` (today's behavior when nothing is decoding).
    pub fn adaptive_chunk(&self, cap: usize, floor: usize, load: f32) -> usize {
        if load <= 0.0 {
            return cap;
        }
        ((cap as f32 / (1.0 + load)) as usize).max(floor)
    }

    /// Effective per-step prefill token budget after conservativeness.
    pub fn effective_budget(&self, token_budget: usize) -> usize {
        if self.conservativeness >= 1.0 {
            return token_budget;
        }
        (token_budget as f32 * self.conservativeness.max(0.0)) as usize
    }
}

/// A snapshot of the live contention vector, for diagnostics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Contention {
    pub decode_load: f32,
    pub latency_waiting: usize,
    pub throughput_waiting: usize,
    pub slot_fill: f32,
    pub mem_headroom: f32,
}

/// Per-sequence scheduling/contention stats, accumulated by the scheduler and
/// reported at sequence end (conditional: only shown if an adaptive adjustment
/// fired for this sequence). Complements the per-sequence spec (drafting) report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedSeqStats {
    pub qos_class: QosClass,
    /// Prefill steps this sequence went through.
    pub prefill_steps: usize,
    /// Steps where its prefill chunk was shrunk below cap (decode load > 0).
    pub chunk_shrunk: usize,
    /// Times a Latency request was admitted ahead of a waiting Throughput one.
    pub priority_admitted: usize,
    /// Times this (Throughput) request was gated by a QoS slot/KV reservation.
    pub reservation_blocked: usize,
    /// Times this sequence was preempted (KV pressure) / swapped out.
    pub preempted: usize,
    pub swapped: usize,
    /// Peak contention observed while this sequence was active.
    pub peak_decode_load: f32,
    pub peak_slot_fill: f32,
    pub peak_mem_pressure: f32,
}

impl SchedSeqStats {
    /// True if any adaptive adjustment fired for this sequence (gates the display).
    pub fn had_adjustment(&self) -> bool {
        self.chunk_shrunk > 0
            || self.priority_admitted > 0
            || self.reservation_blocked > 0
            || self.preempted > 0
            || self.swapped > 0
    }

    /// One-line body for the per-sequence report (the caller adds the `[Seq N]` prefix).
    pub fn report(&self) -> String {
        format!(
            "{:?}: prefill={} shrunk={} prio={} reserve_block={} preempt={} swap={} | peak load/slot/mem={:.1}/{:.2}/{:.2}",
            self.qos_class,
            self.prefill_steps,
            self.chunk_shrunk,
            self.priority_admitted,
            self.reservation_blocked,
            self.preempted,
            self.swapped,
            self.peak_decode_load,
            self.peak_slot_fill,
            self.peak_mem_pressure
        )
    }
}