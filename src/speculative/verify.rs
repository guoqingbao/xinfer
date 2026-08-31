use candle_core::{Result, Tensor, D};
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
    })
}

struct WindowedAcceptanceStats {
    label: &'static str,
    window_capacity: usize,
    log_every: usize,
    ring: Vec<(usize, usize)>,
    total_steps: usize,
}

impl WindowedAcceptanceStats {
    fn new(label: &'static str, window_capacity: usize, log_every: usize) -> Self {
        Self {
            label,
            window_capacity,
            log_every,
            ring: Vec::with_capacity(window_capacity),
            total_steps: 0,
        }
    }

    fn record(&mut self, proposed: usize, accepted: usize) -> bool {
        if self.ring.len() >= self.window_capacity {
            self.ring.remove(0);
        }
        self.ring.push((proposed, accepted));
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
