use crate::utils::env::soft_mask_disabled;
use crate::utils::guidance::{GuidanceState, ParserFactory};
use candle_core::{DType, Result, Tensor};
use llguidance::api::TopLevelGrammar;
use parking_lot::RwLock;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;
use toktrie::SimpleVob;

pub struct GuidedDecodingRequest<'a> {
    pub seq_id: usize,
    pub grammar: Option<&'a TopLevelGrammar>,
    pub reasoning_end_ids: &'a [u32],
}

pub struct GuidedDecodingStep {
    guided_seq_ids: Option<HashSet<usize>>,
}

impl GuidedDecodingStep {
    fn none() -> Self {
        Self {
            guided_seq_ids: None,
        }
    }

    fn new(guided_seq_ids: HashSet<usize>) -> Self {
        Self {
            guided_seq_ids: Some(guided_seq_ids),
        }
    }
}

/// Soft masking configuration for guided decoding.
/// Instead of hard masking to -inf, disallowed logits are shifted by a large value.
#[derive(Clone, Debug)]
pub struct SoftMaskConfig {
    pub mask_shift: f32,
    pub min_logit: f32,
    pub enabled: bool,
}

impl Default for SoftMaskConfig {
    fn default() -> Self {
        Self {
            mask_shift: -1000.0,
            min_logit: -1e9,
            enabled: !soft_mask_disabled(),
        }
    }
}

pub struct GuidedDecoding {
    factory: Option<Arc<ParserFactory>>,
    states: RwLock<HashMap<usize, GuidanceState>>,
    failed: RwLock<HashSet<usize>>,
    mismatch: RwLock<HashSet<usize>>,
    soft_mask: SoftMaskConfig,
}

impl GuidedDecoding {
    pub fn new(factory: Option<Arc<ParserFactory>>) -> Self {
        Self {
            factory,
            states: RwLock::new(HashMap::new()),
            failed: RwLock::new(HashSet::new()),
            mismatch: RwLock::new(HashSet::new()),
            soft_mask: SoftMaskConfig::default(),
        }
    }

    pub fn apply(
        &self,
        logits: &Tensor,
        requests: &[GuidedDecodingRequest<'_>],
    ) -> Result<(Tensor, GuidedDecodingStep)> {
        if requests.iter().all(|request| request.grammar.is_none()) {
            return Ok((logits.clone(), GuidedDecodingStep::none()));
        }

        let Some(factory) = &self.factory else {
            return Ok((logits.clone(), GuidedDecodingStep::none()));
        };

        let mut states = self.states.write();
        let mut failed = self.failed.write();
        let mut mismatch = self.mismatch.write();
        let mut modified = false;
        let batch_size = logits.dim(0)?;
        let vocab_size = logits.dim(1)?;

        let mut masks: Vec<(usize, usize, SimpleVob)> = Vec::new();
        let mut failed_seq_ids = Vec::new();
        let mut guided_seq_ids = HashSet::new();

        for request in requests {
            if request.grammar.is_none() {
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                let _ = mismatch.remove(&request.seq_id);
            }
        }

        for (batch_index, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                continue;
            };

            let seq_id = request.seq_id;
            if failed.contains(&seq_id) {
                continue;
            }

            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match GuidanceState::new_from_grammar_with_reasoning(
                    factory.clone(),
                    grammar,
                    request.reasoning_end_ids.to_vec(),
                ) {
                    Ok(state) => entry.insert(state),
                    Err(err) => {
                        failed.insert(seq_id);
                        crate::log_warn!(
                            "[Seq {}] Failed to create guidance state: {}. Disabling constraints for this sequence.",
                            seq_id,
                            err
                        );
                        continue;
                    }
                },
            };

            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    let mask_len = mask.len();
                    if mask_len == 0 {
                        if failed.insert(seq_id) {
                            crate::log_warn!(
                                "[Seq {}] Guidance mask length is 0. Disabling constraints for this sequence.",
                                seq_id
                            );
                        }
                        failed_seq_ids.push(seq_id);
                        continue;
                    }

                    if mask_len != vocab_size && mismatch.insert(seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Guidance mask size {} does not match vocab size {}. Clamping mask application.",
                            seq_id,
                            mask_len,
                            vocab_size
                        );
                    }

                    if !mask_allows_all(&mask, vocab_size) {
                        modified = true;
                    }
                    masks.push((batch_index, seq_id, mask));
                    guided_seq_ids.insert(seq_id);
                }
                Err(err) => {
                    if failed.insert(seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to compute guidance mask: {}. Disabling constraints for this sequence.",
                            seq_id,
                            err
                        );
                    }
                    failed_seq_ids.push(seq_id);
                }
            }
        }

        for seq_id in &failed_seq_ids {
            let _ = states.remove(seq_id);
        }

        let step = GuidedDecodingStep::new(guided_seq_ids);
        if !modified {
            return Ok((logits.clone(), step));
        }

        let mut allow_mask = vec![1u8; batch_size * vocab_size];
        for (seq_idx, _, mask) in masks {
            if mask_allows_all(&mask, vocab_size) {
                continue;
            }
            let start = seq_idx * vocab_size;
            write_allow_row(
                &mut allow_mask[start..start + vocab_size],
                &mask,
                vocab_size,
            );
        }

        let logits = logits.to_dtype(DType::F32)?;
        let allow_mask = Tensor::from_vec(allow_mask, logits.shape().clone(), logits.device())?;
        let masked_logits = if self.soft_mask.enabled {
            let disallowed = logits
                .affine(1.0, self.soft_mask.mask_shift as f64)?
                .clamp(self.soft_mask.min_logit, f32::INFINITY)?;
            allow_mask.where_cond(&logits, &disallowed)?
        } else {
            let disallowed =
                Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
            allow_mask.where_cond(&logits, &disallowed)?
        };

        Ok((masked_logits, step))
    }

    pub fn apply_fast_forward(&self, seq_ids: &[usize], tokens: &mut [u32]) {
        if self.factory.is_none() {
            return;
        }

        let mut states = self.states.write();
        for (i, seq_id) in seq_ids.iter().enumerate() {
            if let Some(state) = states.get_mut(seq_id) {
                let ff_tokens = state.compute_ff_tokens();
                if !ff_tokens.is_empty() && ff_tokens[0] != tokens[i] {
                    tokens[i] = ff_tokens[0];
                }
            }
        }
    }

    pub fn commit(&self, seq_ids: &[usize], tokens: &[u32], step: GuidedDecodingStep) {
        let Some(guided_seq_ids) = step.guided_seq_ids else {
            return;
        };

        let mut states = self.states.write();
        let mut failed = self.failed.write();
        for (seq_idx, seq_id) in seq_ids.iter().enumerate() {
            if !guided_seq_ids.contains(seq_id) || failed.contains(seq_id) {
                continue;
            }

            if let Some(state) = states.get_mut(seq_id) {
                if state.is_finished() {
                    continue;
                }

                let token = tokens[seq_idx];
                if let Err(err) = state.commit_token(token) {
                    if failed.insert(*seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to commit guided token {}: {}. Disabling constraints for this sequence.",
                            seq_id,
                            token,
                            err
                        );
                    }
                    let _ = states.remove(seq_id);
                }
            }
        }
    }

    pub fn finish(&self, seq_id: usize) {
        let mut states = self.states.write();
        let _ = states.remove(&seq_id);
        let mut failed = self.failed.write();
        let _ = failed.remove(&seq_id);
        let mut mismatch = self.mismatch.write();
        let _ = mismatch.remove(&seq_id);
    }
}

fn mask_allows_all(mask: &SimpleVob, vocab_size: usize) -> bool {
    if mask.len() < vocab_size {
        return false;
    }

    let words = mask.as_slice();
    let full_words = vocab_size / 32;
    if words.len() < full_words {
        return false;
    }
    if words[..full_words].iter().any(|word| *word != u32::MAX) {
        return false;
    }

    (full_words * 32..vocab_size).all(|tok| mask.is_allowed(tok as u32))
}

fn write_allow_row(row: &mut [u8], mask: &SimpleVob, vocab_size: usize) {
    row.fill(0);
    let apply_len = std::cmp::min(vocab_size, mask.len());
    mask.iter_set_entries(|idx| {
        if idx < apply_len {
            row[idx] = 1;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{mask_allows_all, write_allow_row};
    use toktrie::SimpleVob;

    #[test]
    fn test_mask_allows_all_respects_vocab_size() {
        let short = SimpleVob::alloc_ones(3);
        assert!(!mask_allows_all(&short, 4));

        let exact = SimpleVob::alloc_ones(4);
        assert!(mask_allows_all(&exact, 4));

        let mut partial = SimpleVob::alloc_ones(64);
        partial.disallow_token(63);
        assert!(!mask_allows_all(&partial, 64));
    }

    #[test]
    fn test_write_allow_row_clamps_to_vocab() {
        let mut mask = SimpleVob::alloc(6);
        mask.allow_token(1);
        mask.allow_token(5);
        let mut row = vec![1u8; 4];

        write_allow_row(&mut row, &mask, 4);

        assert_eq!(row, vec![0, 1, 0, 0]);
    }
}
