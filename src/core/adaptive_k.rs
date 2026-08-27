// SGLang-style tiered adaptive draft-length controller.
//
// Picks the draft count K from a set of candidate tiers based on the EMA of
// observed accepted-draft lengths, with hysteresis so K only scales UP while
// acceptance justifies it (and only DOWN when it clearly doesn't). Ported from
// sglang/srt/speculative/adaptive_spec_params.py (AdaptiveStepSlot).

/// The active tier index for advanced one step at a time; a hysteresis deadband
/// (down_hysteresis < up_hysteresis) prevents tieration between adjacent tiers.
pub struct AdaptiveSpecController {
    enabled: bool,
    candidate_steps: Vec<usize>,
    current_idx: usize,
    ema_accept_len: f64,
    batch_count: usize,
    ema_alpha: f64,
    update_interval: usize,
    warmup_batches: usize,
    down_hysteresis: f64,
    up_hysteresis: f64,
}

/// The tier (capture) set for adaptive K: `XINFER_SPEC_ADAPTIVE_TIERS` clamped to
/// `1..=max_k`, deduped, sorted, with `max_k` always included (the controller starts
/// at the full tier). Default: `[1, 3, max_k]` (SGLang-shaped low/mid/full). Every
/// tier in this set has a pre-captured verify graph, so a tier move is a
/// graph->graph swap, never a graph->eager flip.
pub fn adaptive_tiers(max_k: usize) -> Vec<usize> {
    let max_k = max_k.max(1);
    let mut tiers = crate::utils::env::spec_adaptive_tiers()
        .map(|v| v.into_iter().filter(|t| *t >= 1 && *t <= max_k).collect::<Vec<_>>())
        .unwrap_or_default();
    if tiers.is_empty() {
        tiers = if max_k >= 3 {
            vec![1, 3, max_k]
        } else {
            (1..=max_k).collect()
        };
    }
    tiers.push(max_k);
    tiers.sort_unstable();
    tiers.dedup();
    tiers
}

impl AdaptiveSpecController {
    /// `max_k` is the configured maximum draft count (the CLI `#`); `tiers` is the
    /// candidate set (see `adaptive_tiers`). Adaptive behavior is gated by
    /// `XINFER_SPEC_ADAPTIVE_K` (default off, in which case `current_k()` always
    /// returns `max_k` and `update()` is a no-op).
    pub fn new(max_k: usize, tiers: &[usize]) -> Self {
        let max_k = max_k.max(1);
        let mut candidate_steps: Vec<usize> = if tiers.is_empty() {
            (1..=max_k).collect()
        } else {
            tiers.iter().filter(|t| **t >= 1 && **t <= max_k).cloned().collect()
        };
        if candidate_steps.is_empty() || *candidate_steps.last().unwrap() != max_k {
            candidate_steps.push(max_k);
            candidate_steps.sort_unstable();
            candidate_steps.dedup();
        }
        let current_idx = candidate_steps.len() - 1;
        Self {
            enabled: crate::utils::env::spec_adaptive_k(),
            // Neutral start: one below the current tier (SGLang initializes EMA at steps-1).
            ema_accept_len: (candidate_steps[current_idx] - 1) as f64,
            batch_count: 0,
            candidate_steps,
            current_idx,
            ema_alpha: 0.2,
            update_interval: 5,
            warmup_batches: 10,
            down_hysteresis: -0.25,
            up_hysteresis: 0.0,
        }
    }

    /// The active draft count K. When adaptive is disabled, this is always the full `max_k`
    /// (the pre-adaptive behavior); when enabled, a candidate tier in `1..=max_k`.
    pub fn current_k(&self) -> usize {
        if !self.enabled {
            return self.candidate_steps.last().copied().unwrap_or(1);
        }
        self.candidate_steps[self.current_idx]
    }

    /// Feed the accepted-draft lengths from one decode step (one entry sequence).
    pub fn update(&mut self, accepted_lengths: &[usize]) {
        if !self.enabled {
            return;
        }
        if accepted_lengths.is_empty() {
            return;
        }
        let current = self.candidate_steps[self.current_idx];
        if current > 0 {
            let batch_avg = accepted_lengths.iter().map(|&a| a as f64).sum::<f64>()
                / accepted_lengths.len() as f64;
            self.ema_accept_len = (1.0 - self.ema_alpha) * self.ema_accept_len
                + self.ema_alpha * batch_avg;
        }
        self.batch_count += 1;
        if self.batch_count <= self.warmup_batches {
            return;
        }
        if (self.batch_count - self.warmup_batches) % self.update_interval != 0 {
            return;
        }
        self.recompute();
    }

    /// Recomcompute the active tier from the EMA (SGLang `_recompute_params`).
    fn recompute(&mut self) {
        let old_idx = self.current_idx;
        let mut idx = old_idx;

        // Move down while the EMA is at/below the lower tier's threshold.
        while idx > 0 {
            let prev_step = self.candidate_steps[idx - 1];
            let mut drop_threshold = if prev_step == 0 {
                0.5
            } else {
                prev_step as f64 - 0.5
            };
            drop_threshold += self.down_hysteresis;
            if self.ema_accept_len <= drop_threshold {
                idx -= 1;
            } else {
                break;
            }
        }

        // Move up (only if we did not just move down) while the EMA exceeds the
        // current tier's threshold.
        if idx == old_idx {
            while idx < self.candidate_steps.len() - 1 {
                let cur_step = self.candidate_steps[idx];
                let rise_threshold = cur_step as f64 - 0.5 + self.up_hysteresis;
                if self.ema_accept_len > rise_threshold {
                    idx += 1;
                } else {
                    break;
                }
            }
        }

        self.current_idx = idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_full_k() {
        // env defaults off -> current_k() is always the full count, update() is a no-op.
        let mut c = AdaptiveSpecController::new(4, &adaptive_tiers(4));
        c.enabled = false;
        for _ in 0..40 {
            c.update(&[0]);
        }
        assert_eq!(c.current_k(), 4);
    }

    #[test]
    fn starts_at_max_and_stays_while_acceptance_is_high() {
        let mut c = AdaptiveSpecController::new(4, &adaptive_tiers(4));
        c.enabled = true;
        assert_eq!(c.current_k(), 4);
        // High acceptance (full 4) should keep the max tier.
        for _ in 0..40 {
            c.update(&[4]);
        }
        assert_eq!(c.current_k(), 4);
    }

    #[test]
    fn drops_when_acceptance_is_low() {
        let mut c = AdaptiveSpecController::new(4, &adaptive_tiers(4));
        c.enabled = true;
        for _ in 0..40 {
            c.update(&[0]); // nothing accepted
        }
        assert!(c.current_k() < 4);
    }

    #[test]
    fn floors_at_one() {
        let mut c = AdaptiveSpecController::new(4, &adaptive_tiers(4));
        c.enabled = true;
        for _ in 0..200 {
            c.update(&[0]);
        }
        assert_eq!(c.current_k(), 1);
    }

    #[test]
    fn default_tiers_are_sglang_shaped() {
        // XINFER_SPEC_ADAPTIVE_TIERS unset -> [1, 3, max_k] (low/mid/full).
        assert_eq!(adaptive_tiers(8), vec![1, 3, 8]);
        assert_eq!(adaptive_tiers(4), vec![1, 3, 4]);
        assert_eq!(adaptive_tiers(2), vec![1, 2]);
        assert_eq!(adaptive_tiers(1), vec![1]);
    }

    #[test]
    fn sparse_tiers_move_in_steps() {
        // Tiers [1, 3, 8]: a re-eval steps one tier down while the EMA is still
        // above the lower threshold, floors at 1, and climbs back on acceptance.
        let mut c = AdaptiveSpecController::new(8, &[1, 3, 8]);
        c.enabled = true;
        assert_eq!(c.current_k(), 8);
        for _ in 0..14 {
            c.update(&[0]);
        }
        c.update(&[8]); // bc=15: first re-eval (EMA ~1.85) -> one tier down
        assert_eq!(c.current_k(), 3);
        for _ in 0..10 {
            c.update(&[0]); // bc=25: EMA ~0.2 -> floor
        }
        assert_eq!(c.current_k(), 1);
        for _ in 0..40 {
            c.update(&[8]);
        }
        assert_eq!(c.current_k(), 8);
    }
}
