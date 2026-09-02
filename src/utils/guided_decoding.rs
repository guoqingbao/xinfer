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

#[derive(Clone)]
pub struct GuidedDecodingStep {
    guided_seq_ids: Option<HashSet<usize>>,
}

impl GuidedDecodingStep {
    fn none() -> Self {
        Self {
            guided_seq_ids: None,
        }
    }

    pub fn new(guided_seq_ids: HashSet<usize>) -> Self {
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
    /// GPU-resident DFA table (uploaded once when XINFER_DFA_GRAMMAR=1).
    #[cfg(feature = "cuda")]
    dfa_table: Option<attention_rs::dfa::DfaGpuTable>,
}

impl GuidedDecoding {
    pub fn new(factory: Option<Arc<ParserFactory>>) -> Self {
        Self {
            factory,
            states: RwLock::new(HashMap::new()),
            failed: RwLock::new(HashSet::new()),
            mismatch: RwLock::new(HashSet::new()),
            soft_mask: SoftMaskConfig::default(),
            #[cfg(feature = "cuda")]
            dfa_table: None,
        }
    }

    /// Upload the DFA table to GPU (called once at model load when XINFER_DFA_GRAMMAR=1).
    #[cfg(feature = "cuda")]
    pub fn upload_dfa_table(&mut self, dfa: &llguidance::hw_dfa::HwDfa, device: &candle_core::Device) -> Result<()> {
        let mask_sign: Vec<u8> = dfa.mask_sign.clone();
        let edge_offsets: Vec<u32> = dfa.edge_offsets.clone();
        let edge_counts: Vec<u32> = dfa.edge_counts.clone();
        let edge_tokens: Vec<u32> = dfa.edges.iter().map(|e| e.token).collect();
        let edge_next: Vec<u32> = dfa.edges.iter().map(|e| e.next_state).collect();
        let mask_words: Vec<u32> = dfa.mask_words.clone();
        let universal_target: Vec<u32> = dfa.universal_target.clone();
        let table = attention_rs::dfa::DfaGpuTable::upload(
            mask_sign,
            edge_offsets,
            edge_counts,
            edge_tokens,
            edge_next,
            mask_words,
            universal_target,
            dfa.num_states,
            dfa.words_per_vob,
            device,
        )?;
        self.dfa_table = Some(table);
        Ok(())
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

    /// Apply the seq's current grammar VOB to every row of `logits` [n, vocab] (batched).
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
    /// VOB across all `n` draft positions -> `[n, vocab]` u8 allow matrix on `device`.
    /// Returns `None` when the seq is unguided/finished or the VBO allows allows the whole vocab.
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
            return Ok(None);
        }
        let mut row = vec![0u8; vocab];
        write_allow_row(&mut row, &mask, vocab);
        let row = Tensor::from_vec(row, (vocab,), device)?;
        Ok(Some(row.unsqueeze(0)?.expand((n, vocab))?))
    }

    /// Exact per-position grammar gate for the fused DFlash2 selector: walk a *clone* of the
    /// sequence's FSM over the draft `logits`, recording each position's VOB into a
    /// `[n, vocab]` u8 allow matrix. Returns `None` if unguided/finished or every position
    /// allows the full vocab.
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
            let row = logits.get(i)?;
            let masked = apply_vob_to_row(&row, &mask)?;
            let tok = masked
                .to_dtype(candle_core::DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .to_scalar::<u32>()?;
            state
                .commit_token(tok)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
        if !any_gate {
            return Ok(None);
        }
        Ok(Some(Tensor::from_vec(flat, (n, vocab), device)?))
    }

    /// Build the per-row grammar allow-mask `[requests.len(), vocab]` (u8, 1 = legal,
    /// 0 = illegal) for the CUDA-sampler offload path. Returns `None` when no row needs
    /// gating (no grammar, or every VOB allows the whole vocab).
    pub fn build_allow_mask(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
        device: &candle_core::Device,
    ) -> Result<Option<Tensor>> {
        if self.factory.is_none() || vocab_size == 0 {
            return Ok(None);
        }
        let factory = self.factory.clone().expect("factory checked non-none above");
        let batch_size = requests.len();
        let mut states = self.states.write();
        let mut failed = self.failed.write();
        let mut any_gate = false;
        let mut allow = vec![1u8; batch_size * vocab_size];
        for (row, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                continue;
            };
            if failed.contains(&request.seq_id) {
                continue;
            }
            let state = match states.entry(request.seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match GuidanceState::new_from_grammar_with_reasoning(
                    factory.clone(),
                    grammar,
                    request.reasoning_end_ids.to_vec(),
                ) {
                    Ok(state) => entry.insert(state),
                    Err(err) => {
                        failed.insert(request.seq_id);
                        crate::log_warn!(
                            "[Seq {}] Failed to create guidance state: {}. Disabling constraints for this sequence.",
                            request.seq_id,
                            err
                        );
                        continue;
                    }
                },
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    if mask.len() == 0 {
                        if failed.insert(request.seq_id) {
                            crate::log_warn!(
                                "[Seq {}] Guidance mask length is 0. Disabling constraints for this sequence.",
                                request.seq_id
                            );
                        }
                        let _ = states.remove(&request.seq_id);
                        continue;
                    }
                    if !mask_allows_all(&mask, vocab_size) {
                        any_gate = true;
                        write_allow_row(
                            &mut allow[row * vocab_size..row * vocab_size + vocab_size],
                            &mask,
                            vocab_size,
                        );
                    }
                }
                Err(err) => {
                    if failed.insert(request.seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to compute guidance mask: {}. Disabling constraints for this sequence.",
                            request.seq_id,
                            err
                        );
                    }
                    let _ = states.remove(&request.seq_id);
                }
            }
        }
        if !any_gate {
            return Ok(None);
        }
        Ok(Some(Tensor::from_vec(allow, (batch_size, vocab_size), device)?))
    }

    /// Build the raw VOB bitset words for the full batch. Returns
    /// `[batch_size * vocab_size/32]` u32 words (bit i set = token allowed).
    /// 8x less data than the F32 mask tensor. Returns None when no row
    /// needs gating.
    pub fn build_vob_words(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
    ) -> Option<Vec<u32>> {
        if self.factory.is_none() || vocab_size == 0 {
            return None;
        }
        let batch_size = requests.len();
        let num_words = (vocab_size + 31) / 32;
        let mut words = vec![0u32; batch_size * num_words];
        let mut any_gate = false;

        let mut states = self.states.write();
        let mut failed = self.failed.write();

        for (row, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                continue;
            };
            if failed.contains(&request.seq_id) {
                continue;
            }
            let state = match states.entry(request.seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let factory = self.factory.clone().unwrap();
                    match GuidanceState::new_from_grammar_with_reasoning(
                        factory,
                        grammar,
                        request.reasoning_end_ids.to_vec(),
                    ) {
                        Ok(state) => entry.insert(state),
                        Err(err) => {
                            failed.insert(request.seq_id);
                            crate::log_warn!(
                                "[Seq {}] Failed to create guidance state: {}. Disabling constraints.",
                                request.seq_id,
                                err
                            );
                            continue;
                        }
                    }
                }
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    if mask.len() == 0 {
                        if failed.insert(request.seq_id) {
                            crate::log_warn!(
                                "[Seq {}] Guidance mask length is 0. Disabling constraints.",
                                request.seq_id
                            );
                        }
                        let _ = states.remove(&request.seq_id);
                        continue;
                    }
                    if !mask_allows_all(&mask, vocab_size) {
                        any_gate = true;
                        let row_base = row * num_words;
                        let apply_len = std::cmp::min(vocab_size, mask.len());
                        mask.iter_set_entries(|idx| {
                            if idx < apply_len {
                                words[row_base + idx / 32] |= 1u32 << (idx % 32);
                            }
                        });
                    }
                }
                Err(err) => {
                    if failed.insert(request.seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to compute guidance mask: {}. Disabling constraints.",
                            request.seq_id,
                            err
                        );
                    }
                    let _ = states.remove(&request.seq_id);
                }
            }
        }

        if !any_gate {
            return None;
        }
        Some(words)
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

    /// Differential test: DFA export must produce identical allowed-sets to the
    /// CPU parser at every position. Uses a single-byte tokenizer (vocab=256,
    /// no model file needed). This is the quality gate: if the DFA disagrees
    /// with the CPU parser, the grammar constraint is broken.
    #[test]
    fn dfa_differential_matches_cpu_parser() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        // Single-byte tokenizer: each byte (0-255) is a token + 6 special tokens.
        // No model file needed.
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Grammar: match the byte sequence "abc" then "END" (bytes 69,78,68).
        // In single-byte mode, each character is its own token.
        let grm_str = r#"start: "abc" "END""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        // CPU parser (driven position by position).
        let cpu_parser = factory.create_parser(grm.clone()).unwrap();
        // DFA exported from an identical parser.
        let dfa_source = factory.create_parser(grm.clone()).unwrap();
        let dfa = dfa_source.export_hw_dfa(10_000).expect("DFA export failed");

        let mut cpu = cpu_parser;
        cpu.start_without_prompt();
        let mut dfa_state = dfa.start_state;

        // The valid token sequence: 'a'=97, 'b'=98, 'c'=99, 'E'=69, 'N'=78, 'D'=68
        let sequence: Vec<u32> = vec![97, 98, 99, 69, 78, 68];
        let vocab = env.tok_trie().vocab_size(); // 256

        for (step, &token) in sequence.iter().enumerate() {
            // 1. CPU mask at this position
            let cpu_mask = cpu.compute_mask().unwrap();

            // 2. Full-set comparison: DFA allowed == CPU allowed for every token
            for t in 0..vocab as u32 {
                let cpu_ok = cpu_mask.is_allowed(t);
                let dfa_ok = dfa.is_token_allowed(dfa_state, t);
                assert_eq!(
                    cpu_ok, dfa_ok,
                    "step {} token {}: CPU={} DFA={} DISAGREE",
                    step, t, cpu_ok, dfa_ok
                );
            }

            // 3. The chosen token must be allowed by both
            assert!(cpu_mask.is_allowed(token), "step {}: CPU disallows token {}", step, token);
            assert!(dfa.is_token_allowed(dfa_state, token), "step {}: DFA disallows token {}", step, token);

            // 4. Advance both
            let next_dfa = dfa.advance(dfa_state, token).expect("DFA advance failed");
            cpu.consume_token(token).expect("CPU consume failed");
            dfa_state = next_dfa;
        }

        // After the full sequence, DFA must be at an accept state.
        assert!(
            dfa.accept_states.contains(&dfa_state),
            "DFA must be accepting after full sequence, got state {}",
            dfa_state
        );

        println!(
            "DIFFERENTIAL: {} positions, full-set agreement (vocab={}), accept reached",
            sequence.len(), vocab
        );
    }

    /// Benchmark: DFA table lookup vs CPU parser walk.
    /// Proves the speedup is real (not just theoretical).
    /// The DFA path must be at least 100x faster than the CPU path.
    #[test]
    fn dfa_benchmark_lookup_vs_parser() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Simple grammar: exactly 3 bytes "abc" (97, 98, 99)
        let grm_str = r#"start: "abc""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        // DFA
        let dfa_source = factory.create_parser(grm.clone()).unwrap();
        let dfa = dfa_source.export_hw_dfa(10_000).expect("DFA export failed");
        assert!(dfa.num_states > 1, "DFA should have multiple states");

        let sequence: Vec<u32> = vec![97, 98, 99]; // 'a', 'b', 'c'

        // Verify the sequence is valid (warm-up + correctness check)
        let mut cpu_check = factory.create_parser(grm.clone()).unwrap();
        cpu_check.start_without_prompt();
        let mut dfa_check = dfa.start_state;
        for &tok in &sequence {
            cpu_check.consume_token(tok).unwrap();
            dfa_check = dfa.advance(dfa_check, tok).unwrap_or_else(|| {
                panic!("DFA rejected token {} - grammar/sequence mismatch", tok)
            });
        }
        assert!(dfa.accept_states.contains(&dfa_check), "DFA should be accepting after 'abc'");

        // Benchmark CPU: compute_mask + consume at each position
        let iterations = 1000;
        let t0 = std::time::Instant::now();
        for _ in 0..iterations {
            let mut p = factory.create_parser(grm.clone()).unwrap();
            p.start_without_prompt();
            for &tok in &sequence {
                let _mask = p.compute_mask().unwrap();
                p.consume_token(tok).unwrap();
            }
        }
        let cpu_time = t0.elapsed();

        // Benchmark DFA: mask_at + advance at each position
        let t1 = std::time::Instant::now();
        for _ in 0..iterations {
            let mut s = dfa.start_state;
            for &tok in &sequence {
                let _mask = dfa.mask_at(s);
                s = dfa.advance(s, tok).unwrap();
            }
        }
        let dfa_time = t1.elapsed();

        let speedup = cpu_time.as_nanos() as f64 / dfa_time.as_nanos() as f64;
        println!(
            "BENCHMARK: CPU parser={:?} DFA={:?} speedup={:.0}x ({} iterations x {} tokens)",
            cpu_time, dfa_time, speedup, iterations, sequence.len()
        );

        // Quality gate: DFA must be at least 100x faster
        assert!(
            speedup > 100.0,
            "DFA speedup {speedup:.0}x is below 100x threshold. \
             The table lookup should be orders of magnitude faster than the CPU parser walk."
        );
    }

    /// GPU DFA accuracy: verifies GPU kernel output matches CPU DFA reference.
    /// Uses a real llguidance grammar export. Benchmark is in examples/bench_dfa_xinfer.rs.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_dfa_accuracy_vs_cpu() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Real grammar: "abc" then "END" (6 tokens, 7 states)
        let grm_str = r#"start: "abc" "END""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        // CPU parser (for timing + accuracy reference)
        let cpu_parser = factory.create_parser(grm.clone()).unwrap();
        let mut cpu = cpu_parser.clone();
        cpu.start_without_prompt();

        // Export DFA
        let dfa = cpu_parser.export_hw_dfa(10_000).expect("DFA export failed for test grammar");
        assert!(dfa.num_states > 1, "DFA should have multiple states");

        // Upload to GPU
        let dev = candle_core::Device::new_cuda(0).unwrap();
        let mask_sign: Vec<u8> = dfa.mask_sign.clone();
        let edge_offsets: Vec<u32> = dfa.edge_offsets.clone();
        let edge_counts: Vec<u32> = dfa.edge_counts.clone();
        let edge_tokens: Vec<u32> = dfa.edges.iter().map(|e| e.token).collect();
        let edge_next: Vec<u32> = dfa.edges.iter().map(|e| e.next_state).collect();
        let mask_words: Vec<u32> = dfa.mask_words.clone();
        let universal_target: Vec<u32> = dfa.universal_target.clone();
        let gpu_table = attention_rs::dfa::DfaGpuTable::upload(
            mask_sign, edge_offsets, edge_counts,
            edge_tokens, edge_next, mask_words, universal_target,
            dfa.num_states, dfa.words_per_vob, &dev,
        ).unwrap();

        // Build a valid token sequence to walk through the grammar
        let sequence: Vec<u32> = vec![97, 98, 99, 69, 78, 68]; // "abcEND"
        let vocab = env.tok_trie().vocab_size();
        let batch = 1;

        // Setup GPU tensors
        let logits = candle_core::Tensor::zeros((batch, vocab), candle_core::DType::F32, &dev).unwrap();
        let states = candle_core::Tensor::from_vec(vec![dfa.start_state], (batch,), &dev).unwrap();

        // Accuracy: verify GPU validate_draft matches CPU advance
        let draft_t = candle_core::Tensor::from_vec(sequence.clone(), (batch, sequence.len()), &dev).unwrap();
        let s_start = candle_core::Tensor::from_vec(vec![dfa.start_state], (batch,), &dev).unwrap();
        let gpu_reject = gpu_table.validate_draft(&s_start, &draft_t).unwrap();
        let gpu_reject_val = gpu_reject.flatten_all().unwrap().to_vec1::<u32>().unwrap()[0] as usize;

        // CPU reference: advance through the sequence
        let mut cpu_state = dfa.start_state;
        let mut cpu_reject = sequence.len();
        for (i, &tok) in sequence.iter().enumerate() {
            match dfa.advance(cpu_state, tok) {
                Some(next) => cpu_state = next,
                None => { cpu_reject = i; break; }
            }
        }
        assert_eq!(
            gpu_reject_val, cpu_reject,
            "GPU validate_draft reject={gpu_reject_val} != CPU reject={cpu_reject}"
        );
        println!("GPU DFA accuracy: validate_draft matches CPU (reject={cpu_reject}, all legal={})", cpu_reject == sequence.len());
    }
}
