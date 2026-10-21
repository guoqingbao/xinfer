//! The dynamic speculative-draft-length lookup (the vLLM
//! `build_dynamic_sd_schedule_lookup` port: the dense `batch_size -> K` table,
//! the `num_speculative_tokens_per_batch_size` config, the carry-forward gap
//! fill, the tail fill, the clamp to the max speculative tokens).
//!
//! The MagicDec result: the small batch → the large K (the latency win); the
//! large batch → the small K (the throughput win). The QoS `AdaptivePlan`'s
//! `spec_draft_len` feeds this table's index (the runtime batch size); the
//! table's value is the K for that batch.
//!
//! The config is a list of `(range_start, range_end, num_speculative_tokens)`
//! triples. The lookup expands this into a dense 1-indexed table where
//! `table[batch_size]` is the K to use for that runtime batch size. Gaps
//! between ranges are filled by carrying forward the previous K; the tail after
//! the last range is filled by carrying forward the last K. All K values are
//! clamped to `max_speculative_tokens` (the static `num_speculative_tokens`).

/// The dense `batch_size -> K` lookup table (the 1-indexed, the carry-forward
/// gap fill). Built from the `num_speculative_tokens_per_batch_size` config.
#[derive(Debug, Clone)]
pub struct DynamicSdLookup {
    /// The dense table: the index is the batch size (the 1-indexed), the value
    /// is the K. The 0 is unused.
    table: Vec<usize>,
    /// The max batch size the table covers.
    max_batch: usize,
}

impl DynamicSdLookup {
    /// Build the dense lookup from the config triples (the
    /// `(range_start, range_end, k)` list) + the max batch size + the max
    /// speculative tokens (the static `num_speculative_tokens` clamp).
    /// Returns `None` when the config is empty or invalid (the no dynamic SD,
    /// the static draft length).
    ///
    /// The validation (the `validate_and_normalize_dynamic_sd_schedule` port):
    /// - The ranges must be positive (the `range_start >= 1`, the `range_end >= 1`).
    /// - The `range_start <= range_end`.
    /// - The `k >= 0`.
    /// - The ranges must be non-overlapping + sorted.
    /// - The first range must start at 1 (the every runtime batch size has a
    ///   defined schedule).
    pub fn build(
        schedule: &[(usize, usize, usize)],
        max_batch: usize,
        max_speculative_tokens: usize,
    ) -> Option<Self> {
        if schedule.is_empty() || max_batch == 0 || max_speculative_tokens == 0 {
            return None;
        }

        // The validation + the sort.
        let mut sorted: Vec<(usize, usize, usize)> = schedule.to_vec();
        sorted.sort_by_key(|&(start, _, _)| start);

        let mut previous_end = 0usize;
        for &(range_start, range_end, k) in &sorted {
            if range_start == 0 || range_end == 0 {
                return None; // The positive ranges.
            }
            if range_start > range_end {
                return None; // The start <= end.
            }
            if k > max_speculative_tokens {
                // The clamp (the vLLM `min(vllm_num_speculative_tokens, k)`).
                // The no error, just the clamp.
            }
            if range_start <= previous_end {
                return None; // The non-overlapping.
            }
            previous_end = range_end;
        }
        if sorted[0].0 != 1 {
            return None; // The first range must start at 1.
        }

        // The dense table build (the `build_dynamic_sd_schedule_lookup` port).
        let mut table = vec![0usize; max_batch + 1]; // The 1-indexed.
        let mut next_batch = 1usize;
        let mut last_k: Option<usize> = None;

        for &(range_start, range_end, k) in &sorted {
            let clamped_k = k.min(max_speculative_tokens);
            // The fill the gap before this range (the carry-forward).
            if range_start > next_batch {
                if let Some(prev_k) = last_k {
                    let fill_k = prev_k.min(max_speculative_tokens);
                    for b in next_batch..range_start.min(max_batch + 1) {
                        table[b] = fill_k;
                    }
                }
            }
            // The fill the current range.
            for b in range_start.max(next_batch)..=range_end.min(max_batch) {
                table[b] = clamped_k;
            }
            next_batch = next_batch.max(range_end + 1);
            last_k = Some(clamped_k);
        }

        // The fill the tail (the carry-forward the last K through max_batch).
        if let Some(last_k) = last_k {
            for b in next_batch..=max_batch {
                table[b] = last_k;
            }
        }

        Some(Self { table, max_batch })
    }

    /// Look up the draft length K for a runtime batch size (the QoS
    /// `AdaptivePlan.spec_draft_len` index). The batch size is clamped to the
    /// table's range.
    pub fn draft_len(&self, batch_size: usize) -> usize {
        if batch_size == 0 {
            return self.table[1];
        }
        let idx = batch_size.min(self.max_batch);
        self.table[idx]
    }

    /// The max batch size the table covers.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carry_forward_gap_fill() {
        // The golden (the vLLM doc example): the `[(1,16,3),(32,128,2)]` schedule
        // maps batch 17-31 to K=3 (the carry-forward), the batch 32-128 to K=2.
        let schedule = vec![(1, 16, 3), (32, 128, 2)];
        let lookup = DynamicSdLookup::build(&schedule, 128, 8).unwrap();
        assert_eq!(lookup.draft_len(1), 3);
        assert_eq!(lookup.draft_len(16), 3);
        assert_eq!(lookup.draft_len(17), 3); // The carry-forward gap.
        assert_eq!(lookup.draft_len(31), 3); // The carry-forward gap.
        assert_eq!(lookup.draft_len(32), 2);
        assert_eq!(lookup.draft_len(128), 2);
    }

    #[test]
    fn small_batch_large_k() {
        // The MagicDec result: the small batch → the large K (the latency win).
        let schedule = vec![(1, 4, 8), (5, 16, 4), (17, 64, 2)];
        let lookup = DynamicSdLookup::build(&schedule, 64, 8).unwrap();
        assert_eq!(lookup.draft_len(1), 8);
        assert_eq!(lookup.draft_len(4), 8);
        assert_eq!(lookup.draft_len(5), 4);
        assert_eq!(lookup.draft_len(16), 4);
        assert_eq!(lookup.draft_len(17), 2);
        assert_eq!(lookup.draft_len(64), 2);
    }

    #[test]
    fn k_clamped_to_max() {
        // The K values above the max_speculative_tokens are clamped.
        let schedule = vec![(1, 16, 32)];
        let lookup = DynamicSdLookup::build(&schedule, 16, 8).unwrap();
        assert_eq!(lookup.draft_len(1), 8); // The clamped to the max.
    }

    #[test]
    fn invalid_config_returns_none() {
        // The empty schedule.
        assert!(DynamicSdLookup::build(&[], 16, 8).is_none());
        // The first range doesn't start at 1.
        assert!(DynamicSdLookup::build(&[(2, 16, 3)], 16, 8).is_none());
        // The overlapping ranges.
        assert!(DynamicSdLookup::build(&[(1, 16, 3), (8, 32, 2)], 32, 8).is_none());
        // The zero max_batch.
        assert!(DynamicSdLookup::build(&[(1, 16, 3)], 0, 8).is_none());
        // The zero max_speculative_tokens.
        assert!(DynamicSdLookup::build(&[(1, 16, 3)], 16, 0).is_none());
    }

    #[test]
    fn zero_batch_clamps_to_one() {
        let schedule = vec![(1, 16, 3)];
        let lookup = DynamicSdLookup::build(&schedule, 16, 8).unwrap();
        assert_eq!(lookup.draft_len(0), 3);
    }

    #[test]
    fn tail_carry_forward() {
        // The batches above the last range use the last K.
        let schedule = vec![(1, 4, 8)];
        let lookup = DynamicSdLookup::build(&schedule, 64, 8).unwrap();
        assert_eq!(lookup.draft_len(5), 8); // The tail carry-forward.
        assert_eq!(lookup.draft_len(64), 8);
    }
}