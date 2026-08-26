use crate::utils::env::soft_mask_disabled;
use crate::utils::guidance::{GuidanceState, ParserFactory};
use candle_core::{Result, Tensor};
use llguidance::api::TopLevelGrammar;
use parking_lot::RwLock;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;
use toktrie::SimpleVob;

#[derive(Clone, Copy)]
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

    /// True if `seq_id` has an active (non-failed) grammar FSM state.
    pub fn is_guided(&self, seq_id: usize) -> bool {
        let states = self.states.read();
        let failed = self.failed.read();
        states.contains_key(&seq_id) && !failed.contains(&seq_id)
    }

    /// Non-mutating: grammar-legal prefix length of `tokens` from the seq's current state.
    pub fn validate_tokens(&self, seq_id: usize, tokens: &[u32]) -> Result<usize> {
        if tokens.is_empty() {
            return Ok(0);
        }
        let mut states = self.states.write();
        match states.get_mut(&seq_id) {
            Some(state) => state
                .validate_tokens(tokens)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
            None => Ok(tokens.len()),
        }
    }

    /// Commit a single token to the seq's FSM (advances state; tracks reasoning).
    pub fn commit_token(&self, seq_id: usize, token: u32) {
        let mut states = self.states.write();
        let mut failed = self.failed.write();
        if let Some(state) = states.get_mut(&seq_id) {
            if state.is_finished() {
                return;
            }
            if let Err(err) = state.commit_token(token) {
                if failed.insert(seq_id) {
                    crate::log_warn!(
                        "[Seq {}] Failed to commit guided token {}: {}. Disabling constraints.",
                        seq_id,
                        token,
                        err
                    );
                }
                let _ = states.remove(&seq_id);
            }
        }
    }

    /// Grammar-forced token(s) at the seq's current state (empty if none).
    pub fn ff_tokens(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        states.get_mut(&seq_id).map(|s| s.compute_ff_tokens()).unwrap_or_default()
    }

/// Apply the seq's current grammar VOB to a single logit row; returns the masked row.
    /// No-op (returns the row) if the seq is not guided.
    pub fn mask_row(&self, seq_id: usize, row: &Tensor) -> Result<Tensor> {
        let mut states = self.states.write();
        let state = match states.get_mut(&seq_id) {
            Some(s) => s,
            None => return Ok(row.clone()),
        };
        let mask = match state.compute_mask_or_eos() {
            Ok(m) => m,
            Err(e) => return Err(candle_core::Error::Msg(e.to_string())),
        };
        drop(states);
        apply_vob_to_row(row, &mask)
    }

    /// Apply the seq's current grammar VOB to every row of `logits` [n, vocab] (batched). For
    /// simple deny-set grammars the mask is ~identical across positions, so one VOB suffices.
    pub fn mask_rows(&self, seq_id: usize, logits: &Tensor) -> Result<Tensor> {
        let mut states = self.states.write();
        let state = match states.get_mut(&seq_id) {
            Some(s) => s,
            None => return Ok(logits.clone()),
        };
        let mask = match state.compute_mask_or_eos() {
            Ok(m) => m,
            Err(e) => return Err(candle_core::Error::Msg(e.to_string())),
        };
        drop(states);
        let vocab_size = logits.dims().last().copied().unwrap_or(0) as usize;
        if mask_allows_all(&mask, vocab_size) {
            return Ok(logits.clone());
        }
        let n = logits.dim(0)?;
        let mut allow = vec![0u8; vocab_size];
        write_allow_row(&mut allow, &mask, vocab_size);
        let allow = Tensor::from_vec(allow, (vocab_size,), logits.device())?;
        let allow_2d = allow.expand((n, vocab_size))?;
        let disallowed = Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
Ok(allow_2d.where_cond(logits, &disallowed)?)
    }

    /// Sequentially mask `draft_logits` [n, vocab] with a CLONE of the seq's FSM (precise
    /// per-position gating). Returns the n grammar-biased draft tokens. Live FSM untouched.
    pub fn masked_drafts(&self, seq_id: usize, draft_logits: &Tensor) -> Result<Vec<u32>> {
        let mut state = {
            let states = self.states.read();
            match states.get(&seq_id) {
                Some(s) => s.deep_clone(),
                None => {
                    return draft_logits
                        .to_dtype(candle_core::DType::F32)?
                        .argmax(candle_core::D::Minus1)?
                        .to_vec1::<u32>();
                }
            }
        };
        let n = draft_logits.dim(0)?;
        let mut tokens = Vec::with_capacity(n);
        for i in 0..n {
            let row = draft_logits.get(i)?;
            let mask = state
                .compute_mask_or_eos()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let masked = apply_vob_to_row(&row, &mask)?;
            let tok = masked.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            state.commit_token(tok).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            tokens.push(tok);
        }
        Ok(tokens)
    }

    /// Static grammar gate for the fused DFlash2 selector: repeat the sequence's *current*
    /// VOB across all `n` draft positions -> `[n, vocab]` u8 allow matrix (1 legal /
    /// 0 illegal) on `device`. Returns `None` when the seq is unguided/finished or the
    /// current VOB allows the whole vocab (no gate needed). Approximate (single VOB); the
    /// verify-time firewall (`verify_draft_masked`) keeps it sound.
    pub fn draft_allow_repeated(
        &self,
        seq_id: usize,
        n: usize,
        vocab: usize,
        device: &candle_core::Device,
    ) -> Result<Option<Tensor>> {
        if n == 0 || vocab == 0 {
            return Ok(None);
        }
        let mask = {
            let mut states = self.states.write();
            let state = match states.get_mut(&seq_id) {
                Some(s) => s,
                None => return Ok(None),
            };
            if state.is_finished() {
                return Ok(None);
            }
            state
                .compute_mask_or_eos()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        };
        if mask_allows_all(&mask, vocab) {
            // crate::log_info!("[dflash-debug] allow_repeated: seq={} VOB allows-all -> None (no gate)", seq_id);
            return Ok(None);
        }
        let mut row = vec![0u8; vocab];
        write_allow_row(&mut row, &mask, vocab);
        let row = Tensor::from_vec(row, (vocab,), device)?;
        // crate::log_info!("[dflash-debug] allow_repeated: seq={} n={} -> Some({}x{})", seq_id, n, n, vocab);
        Ok(Some(row.unsqueeze(0)?.expand((n, vocab))?))
    }

    /// Exact per-position grammar gate for the fused DFlash2 selector: walk a *clone* of the
    /// sequence's FSM over the draft `logits` (the same argmax chain as `masked_drafts`),
    /// recording each position's VOB into a `[n, vocab]` u8 allow matrix. Returns `None` if
    /// unguided/finished or every position allows the full vocab.
    pub fn draft_allow_walk(
        &self,
        seq_id: usize,
        logits: &Tensor,
        vocab: usize,
    ) -> Result<Option<Tensor>> {
        let n = logits.dim(0)?;
        if n == 0 || vocab == 0 {
            return Ok(None);
        }
        let mut state = {
            let states = self.states.read();
            match states.get(&seq_id) {
                Some(s) => s.deep_clone(),
                None => return Ok(None),
            }
        };
        if state.is_finished() {
            return Ok(None);
        }
        let device = logits.device();
        let mut flat = vec![0u8; n * vocab];
        let mut any_gate = false;
        // let mut walk_tokens: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            let mask = state
                .compute_mask_or_eos()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if mask_allows_all(&mask, vocab) {
                flat[i * vocab..(i + 1) * vocab].fill(1);
            } else {
                any_gate = true;
                write_allow_row(&mut flat[i * vocab..(i + 1) * vocab], &mask, vocab);
            }
            // Advance the clone exactly as masked_drafts does: argmax of the masked row.
            let row = logits.get(i)?;
            let masked = apply_vob_to_row(&row, &mask)?;
            let tok = masked
                .to_dtype(candle_core::DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .to_scalar::<u32>()?;
            // walk_tokens.push(tok);
            state
                .commit_token(tok)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
        if !any_gate {
            // crate::log_info!("[dflash-debug] allow_walk: seq={} all-VOB-allows -> None (no gate)", seq_id);
            return Ok(None);
        }
        // crate::log_info!("[dflash-debug] allow_walk: seq={} n={} -> Some({}x{})", seq_id, n, n, vocab);
        Ok(Some(Tensor::from_vec(flat, (n, vocab), device)?))
    }
}

/// Apply a grammar VOB to a single logit row (disallowed -> -inf). No-op if the mask allows all.
fn apply_vob_to_row(row: &Tensor, mask: &SimpleVob) -> Result<Tensor> {
    let vocab_size = row.dims().last().copied().unwrap_or(0) as usize;
    if mask_allows_all(mask, vocab_size) {
        return Ok(row.clone());
    }
    let mut allow = vec![0u8; vocab_size];
    write_allow_row(&mut allow, mask, vocab_size);
    let allow = Tensor::from_vec(allow, row.shape().clone(), row.device())?;
    let disallowed = Tensor::full(f32::NEG_INFINITY, row.shape().clone(), row.device())?;
    Ok(allow.where_cond(row, &disallowed)?)
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
