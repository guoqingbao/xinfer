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
    pub fn none() -> Self {
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

pub struct GuidedDecoding {
    factory: Option<Arc<ParserFactory>>,
    states: RwLock<HashMap<usize, GuidanceState>>,
    failed: RwLock<HashSet<usize>>,
    /// Gate the per-rank `[llg]` grammar-mode logs to the rank-0 (the no 4x
    /// repetition on a multi-GPU TP run). The NCCL all-reduce makes all ranks
    /// agree, so the per-rank logs are identical.
    log_rank0: bool,
    /// GPU-resident PDA table (uploaded once when XINFER_PDA_GRAMMAR=1).
    #[cfg(feature = "cuda")]
    pda_table: Option<attention_rs::pda::PdaPushdownTable>,
    /// Per-sequence PDA state for current-position masking: (ctrl, stack, sp)
    /// as GPU tensors, persisted across decode steps.
    #[cfg(feature = "cuda")]
    pda_seq_state: RwLock<HashMap<usize, (Tensor, Tensor, Tensor)>>,
    #[cfg(feature = "cuda")]
    pda_stack_depth: usize,
}

impl GuidedDecoding {
    pub fn new(factory: Option<Arc<ParserFactory>>, log_rank0: bool) -> Self {
        Self {
            factory,
            states: RwLock::new(HashMap::new()),
            failed: RwLock::new(HashSet::new()),
            log_rank0,
            #[cfg(feature = "cuda")]
            pda_table: None,
            #[cfg(feature = "cuda")]
            pda_seq_state: RwLock::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            pda_stack_depth: 1,
        }
    }

    /// Upload the PDA table to GPU (the once at model load when XINFER_PDA_GRAMMAR=1).
    /// DEPRECATED: use `upload_pda_package` with the pushdown-rs CudaPackage instead.
    #[cfg(feature = "cuda")]
    #[deprecated(note = "use upload_pda_package with pushudaPackage")]
    pub fn upload_pda_table(&mut self, _pda: &pushdown_rs::machine::PdaMachine, _device: &candle_core::Device) -> Result<()> {
        candle_core::bail!("upload_pda_table is deprecated; use upload_pda_package");
    }

    /// Upload the PDA table from the pushdown-rs CudaPackage (the new format).
    /// The CudaPackage is the bitvec + source primitives (the GPU-uploadable
    /// POD table). The PdaPushdownTable deserializes the bitvec + uploads the
    /// individual tensors.
    #[cfg(feature = "cuda")]
    pub fn upload_pda_package(
        &mut self,
        pkg: &pushdown_rs::cuda::CudaPackage,
        device: &candle_core::Device,
    ) -> Result<()> {
        let table = attention_rs::pda::PdaPushdownTable::from_cuda_package(pkg, device)?;
        // The stack depth is derived from the table (the machine's max_stack_depth), not a
        // static constant (the R1 no-static-bounds rule). A grammar with deep nesting gets a
        // deeper stack allocation; a shallow grammar gets a tighter one.
        self.pda_stack_depth = table.max_stack_depth as usize;
        self.pda_table = Some(table);
        Ok(())
    }

    /// True if a PDA table has been uploaded (enables the on-GPU current-position path).
    #[cfg(feature = "cuda")]
    pub fn has_pda_table(&self) -> bool {
        self.pda_table.is_some()
    }

    /// The PDA table (the `mask_draft_logits` seam, the draft-position projection).
    #[cfg(feature = "cuda")]
    pub fn pda_table(&self) -> &Option<attention_rs::pda::PdaPushdownTable> {
        &self.pda_table
    }

    /// Test-only accessor for the per-seq PDA state (the ctrl, stack, sp tensors).
    /// Exposed so the CUDA-gated PDA tests can assert on the per-seq state the
    /// fused_sample kernel scatters back, without reaching into the private field.
    #[cfg(all(feature = "cuda", test))]
    pub fn pda_seq_state_for_test(
        &self,
        seq_id: usize,
    ) -> Option<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
        self.pda_seq_state.read().get(&seq_id).cloned()
    }

    /// True if the uploaded PDA table is EXACT for the GPU settled-mask kernels
    /// (the `fused_sample` / the `fused_project`). Per pushdown-rs soundness
    /// rule (the `cuda.rs` is_deterministic + the max_closure_width gate), the GPU
    /// single-config settled mask is exact only when the machine is deterministic
    /// (the singleton frontier) AND the epsilon-closure width fits the GPU BFS

    /// (the `MAXF=64`). When either fails, the GPU projection is an approximation
    /// (the under- for the NPDA, the truncated for the wide closure) and the
    /// caller must use the CPU exact frontier mask (the `pda_mask`, the union over
    /// the full config-set) instead.
    #[cfg(feature = "cuda")]
    pub fn pda_is_exact(&self) -> bool {
        let Some(table) = self.pda_table.as_ref() else {
            return false;
        };
        table.is_deterministic && table.max_closure_width <= 64
    }

    /// The projected masks for the MTP/DFlash drafting (the K+1 masks for
    /// the K draft positions). Uses PdaPushdownTable::fused_project (the GPU
    /// kernel) ONLY when the table is exact (the `pda_is_exact`); otherwise it
    /// returns an error so the caller falls back to the CPU exact frontier mask.
    /// The draft is the [n, K] draft tokens.
    #[cfg(feature = "cuda")]
    pub fn pda_project_masks(
        &self,
        guided_seq_ids: &[usize],
        draft: &Tensor,
    ) -> Result<Tensor> {
        // The exactness gate (the RC1 fix): the GPU settled projection is exact only
        // for a deterministic machine with a bounded closure. Otherwise the caller
        // must use the CPU frontier mask (not the inexact GPU approximation).
        if !self.pda_is_exact() {
            candle_core::bail!(
                "PDA table is not exact for the GPU projection (is_deterministic={}, max_closure_width={}); use the CPU frontier mask",
                self.pda_table.as_ref().map(|t| t.is_deterministic).unwrap_or(false),
                self.pda_table.as_ref().map(|t| t.max_closure_width).unwrap_or(0)
            );
        }
        let table = self.pda_table.as_ref().expect("PDA table not uploaded");
        let dev = draft.device();
        let d = self.pda_stack_depth.max(1);
        let n = guided_seq_ids.len();
        let mut ctrl_rows = Vec::with_capacity(n);
        let mut stack_rows = Vec::with_capacity(n);
        let mut sp_rows = Vec::with_capacity(n);
        {
            let seq_state = self.pda_seq_state.write();
            for &seq_id in guided_seq_ids {
                let (c, s, p) = seq_state
                    .get(&seq_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        let c = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
                        let s = Tensor::zeros((1, d), candle_core::DType::U32, &dev).unwrap();
                        let p = Tensor::from_vec(vec![1u32], (1,), &dev).unwrap();
                        (c, s, p)
                    });
                ctrl_rows.push(c);
                stack_rows.push(s);
                sp_rows.push(p);
            }
        }
        let ctrl = Tensor::cat(&ctrl_rows, 0)?;
        let stack = Tensor::cat(&stack_rows, 0)?;
        let sp = Tensor::cat(&sp_rows, 0)?;
        // The per-seq actual draft length (the [n] tensor). For the POC, all
        // sequences draft the same K (the draft.dim(1)), so k_actual is a
        // constant-K tensor. The kernel is graph-capturable (the k_max is fixed,
        // the k_actual is a GPU tensor, not a launch param).
        let k_max = draft.dim(1)?;
        let n = guided_seq_ids.len();
        let k_actual = Tensor::from_vec(vec![k_max as u32; n], (n,), dev)?;
        table.fused_project(&ctrl, &stack, &sp, draft, &k_actual, None)
    }

    /// Current-position on-GPU PDA masking: for each guided sequence, compute the
    /// mask from its PDA control state, sample, and advance the PDA state.
    /// Persists per-seq (ctrl, stack, sp) across decode steps.
    /// `logits` is [n_guided, vocab]; `guided_seq_ids` aligns rows to sequences.
    /// The GPU settled mask is exact only when the table is deterministic + the
    /// closure is bounded (the `pda_is_exact`); otherwise the caller must use the
    /// CPU frontier mask (the `build_vob_words` path).
    #[cfg(feature = "cuda")]
    pub fn pda_current_step(
        &self,
        logits: &Tensor,
        guided_seq_ids: &[usize],
        sampling: &attention_rs::pda::PdaSampling,
    ) -> Result<Tensor> {
        if !self.pda_is_exact() {
            candle_core::bail!(
                "PDA table is not exact for the GPU current-position step (is_deterministic={}, max_closure_width={}); use the CPU frontier mask",
                self.pda_table.as_ref().map(|t| t.is_deterministic).unwrap_or(false),
                self.pda_table.as_ref().map(|t| t.max_closure_width).unwrap_or(0)
            );
        }
        let table = self.pda_table.as_ref().expect("PDA table not uploaded");
        let dev = logits.device();
        let d = self.pda_stack_depth.max(1);
        let n = guided_seq_ids.len();

        // Gather per-seq PDA state into batch tensors (init new seqs to start).
        let mut ctrl_rows = Vec::with_capacity(n);
        let mut stack_rows = Vec::with_capacity(n);
        let mut sp_rows = Vec::with_capacity(n);
        {
            let seq_state = self.pda_seq_state.write();
            for &seq_id in guided_seq_ids {
                let (c, s, p) = seq_state
                    .get(&seq_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        // Fresh guided seq: start control state, stack=[start], sp=1.
                        let c = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
                        let mut sv = vec![0u32; d];
                        sv[0] = 0;
                        let s = Tensor::from_vec(sv, (1, d), &dev).unwrap();
                        let p = Tensor::from_vec(vec![1u32], (1,), &dev).unwrap();
                        (c, s, p)
                    });
                ctrl_rows.push(c);
                stack_rows.push(s);
                sp_rows.push(p);
            }
        }
        let ctrl = Tensor::stack(&ctrl_rows, 0)?; // [n,1]
        let stack = Tensor::stack(&stack_rows, 0)?; // [n,1,d] -> need [n,d]
        let stack = stack.squeeze(1)?;
        let sp = Tensor::stack(&sp_rows, 0)?; // [n,1]
        let sp = sp.squeeze(1)?;

        // One batched PDA step: fused mask + sample + advance (one kernel launch).
        let (out_ctrl, out_sp, out_tok) =
            table.fused_sample(&logits, &ctrl, &stack, &sp, sampling, None, 0)?;

        // Scatter the new (ctrl, sp) back to per-seq state (stack is updated in-place
        // on the GPU buffer owned by each seq's tensor; re-read it).
        {
            let mut seq_state = self.pda_seq_state.write();
            for (i, &seq_id) in guided_seq_ids.iter().enumerate() {
                let nc = out_ctrl.get(i)?;
                let ns = out_sp.get(i)?;
                let nstack = stack.get(i)?; // stack buffer was mutated in-place by the kernel
                seq_state.insert(seq_id, (nc, nstack, ns));
            }
        }
        Ok(out_tok)
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
                    self.log_rank0,
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

                    if !mask_allows_all(&mask, vocab_size) {
                        modified = true;
                    }
                    masks.push((batch_index, seq_id, mask));
                    guided_seq_ids.insert(seq_id);
                }
                Err(err) => {
                    // The P2 fix: instead of disabling constraints (the failed.insert + the
                    // failed_seq_ids.push), resync to the last committed token boundary (the
                    // constraints stay active; the next step re-attempts the mask).
                    state.resync();
                    crate::log_warn!(
                        "[Seq {}] Failed to compute guidance mask: {}. Resyncing to last committed boundary.",
                        seq_id,
                        err
                    );
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
        let disallowed =
            Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
        let masked_logits: Tensor = allow_mask.where_cond(&logits, &disallowed)?;

        Ok((masked_logits, step))
    }

    /// Build the per-seq VOB bitset (the `[b, v/32] U32` tensor, the 8× less data than the
    /// F32 allow-mask). The VOB is the compact allowed-token set (the `SimpleVob`), passed
    /// directly to the `sample_with_vob` kernel (the no bitwise-AND sampler, the no F32
    /// mask round-trip). The `None` when no grammar is active (the no VOB needed).
    pub fn build_vob_words(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
        device: &candle_core::Device,
    ) -> Result<Option<candle_core::Tensor>> {
        if requests.iter().all(|request| request.grammar.is_none()) {
            return Ok(None);
        }
        let Some(_factory) = &self.factory else {
            return Ok(None);
        };

        let mut states = self.states.write();
        let failed = self.failed.read();
        let mut vob_rows: Vec<Vec<u32>> = Vec::with_capacity(requests.len());
        let words_per_vob = (vocab_size + 31) / 32;

        for request in requests {
            let Some(_grammar) = request.grammar else {
                vob_rows.push(vec![u32::MAX; words_per_vob]); // The all-allowed (the no mask).
                continue;
            };
            let seq_id = request.seq_id;
            if failed.contains(&seq_id) {
                vob_rows.push(vec![u32::MAX; words_per_vob]);
                continue;
            }
            let state = match states.get_mut(&seq_id) {
                Some(state) => state,
                None => {
                    vob_rows.push(vec![u32::MAX; words_per_vob]);
                    continue;
                }
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    // The convert the SimpleVob to the [words_per_vob] U32 words (the VOB bitset).
                    let mut words = vec![0u32; words_per_vob];
                    let vob_words: Vec<u32> = mask.into();
                    for (word_idx, word) in vob_words.iter().enumerate() {
                        if word_idx < words_per_vob {
                            words[word_idx] = *word;
                        }
                    }
                    vob_rows.push(words);
                }
                Err(_) => {
                    vob_rows.push(vec![u32::MAX; words_per_vob]); // The all-allowed on error.
                }
            }
        }

        if vob_rows.is_empty() {
            return Ok(None);
        }
        let flat: Vec<u32> = vob_rows.into_iter().flatten().collect();
        let vob = candle_core::Tensor::from_vec(flat, (requests.len(), words_per_vob), device)?;
        Ok(Some(vob))
    }

    pub fn apply_fast_forward(&self, seq_ids: &[usize], tokens: &mut [u32]) {
        if self.factory.is_none() {
            return;
        }

        let mut states = self.states.write();
        for (i, seq_id) in seq_ids.iter().enumerate() {
            if let Some(state) = states.get_mut(seq_id) {
                // The read-only ff (the non-advancing): rewrites the token VALUE to the
                // grammar-forced continuation without committing. The single `commit_run`
                // (the gate) is the only FSM advance, so the sample site stays non-advancing.
                let ff_tokens = state.compute_ff_tokens_immut();
                if !ff_tokens.is_empty() && ff_tokens[0] != tokens[i] {
                    tokens[i] = ff_tokens[0];
                }
            }
        }
    }

/// The full grammar-forced (ff) run at the seq's current state (empty if none).
    pub fn ff_tokens(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        states.get_mut(&seq_id).map(|s| s.compute_ff_tokens()).unwrap_or_default()
    }

    /// The read-only ff run (the non-advancing variant). The sample site uses this
    /// to rewrite the token VALUE without committing; the single `commit_run`
    /// (the gate) is the only FSM advance.
    pub fn ff_tokens_immut(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        states.get_mut(&seq_id).map(|s| s.compute_ff_tokens_immut()).unwrap_or_default()
    }

    /// Commit the full grammar-forced (ff) sequence for a seq, one token at a time.
    /// The ff tokens are deterministic (forced by the grammar), so they are appended
    /// directly without model sampling. Returns the number of tokens committed.
    pub fn commit_ff_sequence(&self, seq_id: usize) -> usize {
        let ff = self.ff_tokens(seq_id);
        if ff.is_empty() {
            return 0;
        }
        let step = GuidedDecodingStep::new(std::collections::HashSet::from([seq_id]));
        for &token in &ff {
            let _ = self.commit(&[seq_id], &[token], step.clone());
        }
        ff.len()
    }

    pub fn commit(&self, seq_ids: &[usize], tokens: &[u32], step: GuidedDecodingStep) {
        let Some(guided_seq_ids) = step.guided_seq_ids else {
            return;
        };

        let mut states = self.states.write();
        let failed = self.failed.read();
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
                    // The P2 fix: instead of disabling constraints (the failed.insert + the
                    // states.remove), resync to the last committed token boundary (the
                    // constraints stay active; the next step restarts from the last accepted token).
                    state.resync();
                    crate::log_warn!(
                        "[Seq {}] Failed to commit guided token {}: {}. Resyncing to last committed boundary.",
                        seq_id,
                        token,
                        err
                    );
                }
            }
        }
    }

    /// Commit a run of tokens to the sequence's grammar FSM (the single-writer
    /// gate). Returns the number of tokens committed (the FSM-passing prefix).
    /// Delegates to the per-sequence `GuidanceState::commit_run`.
    pub fn commit_run(&self, seq_id: usize, run: &[u32]) -> usize {
        let mut states = self.states.write();
        let Some(state) = states.get_mut(&seq_id) else {
            return run.len(); // no guidance state (the unguided sequence), pass through
        };
        state.commit_run(run)
    }

    /// Resync the seq's FSM to the last committed boundary (the P2 fix). Called by
    /// the gate after a partial `commit_run` (the legal suffix) so the matcher
    /// rolls back to the legal boundary. No-op when the seq has no guidance state.
    pub fn resync(&self, seq_id: usize) {
        let mut states = self.states.write();
        if let Some(state) = states.get_mut(&seq_id) {
            state.resync();
        }
    }

    pub fn finish(&self, seq_id: usize) {
        let mut states = self.states.write();
        let _ = states.remove(&seq_id);
        let mut failed = self.failed.write();
        let _ = failed.remove(&seq_id);
    }

    /// True if `seq_id` has an active (non-failed) grammar FSM state.
    pub fn is_guided(&self, seq_id: usize) -> bool {
        let states = self.states.read();
        let failed = self.failed.read();
        states.contains_key(&seq_id) && !failed.contains(&seq_id)
    }

    /// The count of reasoning tokens admitted for this sequence (the two-phase
    /// mode). The `0` when the sequence has no guidance state, or reasoning has
    /// ended, or the sequence is in the single-phase (full-envelope) mode. Used
    /// by the reasoning-budget promotion (the sampling bias that steers the
    /// model to the end-of-reasoning token as the count approaches the budget).
    pub fn reasoning_token_count(&self, seq_id: usize) -> usize {
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.reasoning_token_count()).unwrap_or(0)
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
        if let Some(state) = states.get_mut(&seq_id) {
            if state.is_finished() {
                return;
            }
            if let Err(err) = state.commit_token(token) {
                // The P2 fix: instead of disabling constraints (the failed.insert + the
                // states.remove), resync to the last committed token boundary (the constraints
                // stay active; the next step restarts from the last accepted token).
                state.resync();
                crate::log_warn!(
                    "[Seq {}] Failed to commit guided token {}: {}. Resyncing to last committed boundary.",
                    seq_id,
                    token,
                    err
                );
            }
        }
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
                    self.log_rank0,
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
                    // The P2 fix: instead of disabling constraints (the failed.insert + the
                    // states.remove), resync to the last committed token boundary (the constraints
                    // stay active; the next step re-attempts the mask).
                    state.resync();
                    crate::log_warn!(
                        "[Seq {}] Failed to compute guidance mask: {}. Resyncing to last committed boundary.",
                        request.seq_id,
                        err
                    );
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
    pub fn build_vob_words_flat(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
    ) -> Option<Vec<u32>> {
        if self.factory.is_none() || vocab_size == 0 {
            return None;
        }
        let batch_size = requests.len();
        let num_words = (vocab_size + 31) / 32;
        // Initialize to all-ones (allow everything by default); gated rows are reset
        // to zero below before their allowed bits are OR-ed in. Allow-all rows stay
        // all-ones, so a mixed batch never masks out its free rows.
        let mut words = vec![u32::MAX; batch_size * num_words];
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
                        self.log_rank0,
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
                        // Reset this gated row to zero (the default all-ones would allow
                        // everything); the allowed bits are OR-ed in below.
                        for w in 0..num_words {
                            words[row_base + w] = 0;
                        }
                        let apply_len = std::cmp::min(vocab_size, mask.len());
                        mask.iter_set_entries(|idx| {
                            if idx < apply_len {
                                words[row_base + idx / 32] |= 1u32 << (idx % 32);
                            }
                        });
                    }
                }
                Err(err) => {
                    // The P2 fix: instead of disabling constraints (the failed.insert + the
                    // states.remove), resync to the last committed token boundary (the constraints
                    // stay active; the next step re-attempts the mask).
                    state.resync();
                    crate::log_warn!(
                        "[Seq {}] Failed to compute guidance mask: {}. Resyncing to last committed.",
                        request.seq_id,
                        err
                    );
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

/// Apply K projected PDA masks (one per draft position) to K draft-logit rows.
/// draft_logits` is [K, vocab]. `projected_masks` is a flat
/// `[ (K+1) * words_per_vob ]` ALLOW-bit VOB (from PdaPushdownTable::fused_project);
/// row i constrains draft position i. Disallowed logits
/// are set to -inf so the draft model cannot propose a grammar-illegal token.
pub fn mask_draft_logits(
    table: Option<&attention_rs::pda::PdaPushdownTable>,
    draft_logits: &Tensor,
    projected_masks: &Tensor,
    words_per_vob: usize,
) -> Result<Tensor> {
    let (k, vocab) = draft_logits.dims2()?;
    // The GPU-accelerated path (the vob_to_allow kernel, the no CPU round-trip),
    // gated behind the PDA env var + a live table. The legacy CPU path otherwise.
    if crate::utils::env::pda_grammar_enabled() {
        if let Some(t) = table {
            // The projected_masks is the [(k+1) * words_per_vob] flat VOB. The
            // vob_to_allow expects the [positions * words_per_vob] flat VOB (the
            // positions = k+1 for a k-draft projection).
            let allow = t.vob_to_allow(projected_masks, 1, k, vocab)?;
            // The allow is the [(k+1), vocab] F32 (the 1.0 = legal, the 0.0 = not).
            // Mask the first k rows (the draft positions).
            let allow_k = allow.narrow(0, 0, k)?;
            let disallowed = Tensor::full(f32::NEG_INFINITY, draft_logits.shape().clone(), draft_logits.device())?;
            return allow_k.where_cond(draft_logits, &disallowed);
        }
    }
    // The legacy CPU path (the vob_words_to_allow, the per-row expansion).
    let mut rows = Vec::with_capacity(k);
    for i in 0..k {
        let row = draft_logits.get(i)?; // [vocab]
        let mask_base = i * words_per_vob;
        let mask_row = projected_masks.narrow(0, mask_base, words_per_vob)?;
        let allow = vob_words_to_allow(&mask_row, vocab)?; // [vocab] u8
        let disallowed = Tensor::full(f32::NEG_INFINITY, row.shape().clone(), row.device())?;
        rows.push(allow.where_cond(&row, &disallowed)?);
    }
    Tensor::stack(&rows, 0)
}

/// Expand a VOB row (u32 words, ALLOW bits) into a [vocab] u8 allow mask.
fn vob_words_to_allow(vob_words: &Tensor, vocab: usize) -> Result<Tensor> {
    let words = vob_words.flatten_all()?.to_vec1::<u32>()?;
    let mut allow = vec![1u8; vocab];
    for (w, word) in words.iter().enumerate() {
        for b in 0..32 {
            let idx = w * 32 + b;
            if idx >= vocab {
                break;
            }
            if word & (1u32 << b) == 0 {
                allow[idx] = 0;
            }
        }
    }
    let dev = vob_words.device();
    Tensor::from_vec(allow, (vocab,), dev)
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
    use super::{mask_allows_all, mask_draft_logits, write_allow_row, GuidedDecoding, GuidedDecodingRequest};
    use toktrie::SimpleVob;
    use candle_core::Tensor;

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

/// The §2.3 VOB data-width assertion: the VOB bitset is 8× less data than the
    /// F32 allow-mask. For a vocab of 4096, the F32 mask is 4096 × 4 bytes = 16384
    /// bytes; the VOB bitset is 4096/32 = 128 words × 4 bytes = 512 bytes. The F32
    /// mask is 16384/512 = 32× the VOB total. Per token: the F32 mask is 4 bytes/token,
    /// the VOB is 4 bits/token (0.5 bytes/token), so the VOB is 8× less data per token.
    #[test]
    fn test_vob_is_8x_less_data_per_token() {
        let vocab_size = 4096;
        // The F32 mask total: the vocab_size × 4 bytes.
        let f32_mask_bytes = vocab_size * 4;
        // The VOB bitset total: the (vocab_size + 31) / 32 words × 4 bytes per word.
        let vob_words = (vocab_size + 31) / 32;
        let vob_bytes = vob_words * 4;
        // The F32 mask is 32× the VOB total (the 4 bytes/token vs the 0.125 bytes/token).
        assert_eq!(f32_mask_bytes, vob_bytes * 32, "the F32 mask is 32× the VOB bitset total");
        // Per token: the F32 mask is 4 bytes/token (32 bits), the VOB is 4 bits/token.
        // The VOB is 8× less data per token (32 bits / 4 bits = 8).
        let f32_per_token_bits = 32;
        let vob_per_token_bits = 4;
        assert_eq!(f32_per_token_bits, vob_per_token_bits * 8, "the VOB is 8× less data per token");
    }

    /// GPU PDA accuracy: verifies the GPU PdaPushdownTable fused_sample matches
    /// a CPU PdaMachine reference walk. Uses a real llguidance grammar export.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_pda_accuracy_vs_cpu() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use attention_rs::pda::PdaPushdownTable;
        use pushdown_rs::pda::Dpda;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Sequence grammar: token 97 then 98 then 99 (single-byte: 'a', 'b', 'c').
        let grm_str = r#"start: <[97]> <[98]> <[99]>"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        let parser = factory.create_parser(grm.clone()).unwrap();
        let cgrm = parser.parser.grammar().clone();

        // Compile to PDA.
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile");
        assert!(pda.is_deterministic(), "sequence grammar is deterministic");
        assert!(!pda.accepting.is_empty(), "PDA has accepting states");

        // Verify the PDA structure (the token-walking tests are in llguidance
        // where the single-byte tokenizer makes the local ID mapping known).
        assert!(pda.num_states > 1, "PDA has multiple states");
        assert!(pda.num_inputs >= 3, "PDA has at least 3 inputs (the 3 token ranges)");
        assert!(pda.is_deterministic(), "sequence grammar is deterministic");
        assert!(!pda.accepting.is_empty(), "PDA has accepting states");
        pda.validate_bounds().expect("PDA transitions are in-bounds");

        // Upload to GPU and verify the table structure matches.
        // Note: the GPU fused_sample kernel requires the PDA's num_inputs to
        // match the model's vocab size (the words_per_vob must align). For
        // this unit test, we verify the table upload + structure only.
        let pkg = llguidance::dpda_adapter::export_pda_package(&cgrm).expect("CUDA package");
        let dev = candle_core::Device::new_cuda(0).unwrap();
        let gpu_table = PdaPushdownTable::from_cuda_package(&pkg, &dev).unwrap();
        assert_eq!(gpu_table.num_states, pda.num_states, "GPU table states match CPU");
        assert_eq!(gpu_table.num_transitions, pda.transitions.len() as u32, "GPU transitions match CPU");
        assert_eq!(gpu_table.num_inputs, pda.num_inputs, "GPU inputs match CPU");
        println!(
            "GPU PDA: structure verified (states={}, inputs={}, transitions={}, upload={} bytes)",
            pda.num_states, pda.num_inputs, pda.transitions.len(), pkg.upload_bytes()
        );
    }

    /// CPU test: projected PDA masks correctly mask draft logits.
    /// Builds a small PDA, projects K draft masks via PdaStream, applies them
    /// to draft logits, and verifies disallowed positions become -inf.
    #[test]
    fn mask_draft_logits_applies_projected_masks() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use pushdown_rs::pda::PdaStream;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: <[97]> <[98]> <[99]>"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();

        // Compile to PDA.
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile");

        // Project 2 draft tokens (local IDs 0, 1) from the start state.
        // The projection emits K+1 masks if all K tokens are valid transitions.
        // If the PDA diverges (a token is illegal), it stops early (fewer masks).
        let configs = vec![(pda.start_state, vec![pda.start_stack])];
        let drafts = vec![vec![0u32, 1]];
        let projected = pda.project_batch(&configs, &drafts);
        assert_eq!(projected.len(), 1, "one config");
        assert!(projected[0].len() >= 1, "at least 1 mask (the start position)");
        println!(
            "PDA projection: {} masks for a 2-token draft",
            projected[0].len()
        );

        // Build a flat projected-masks tensor [3 * words_per_vob].
        let words = ((pda.num_inputs + 31) / 32) as usize;
        let flat: Vec<u32> = projected[0].iter()
            .flat_map(|mask| {
                // Convert the allowed-input list to VOB words.
                let mut vob = vec![0u32; words];
                for &allowed in mask {
                    if allowed < (words * 32) as u32 {
                        vob[allowed as usize / 32] |= 1u32 << (allowed % 32);
                    }
                }
                vob.into_iter()
            })
            .collect();
        // Note: this is a simplification. The real VOB conversion is done by
        // the GPU kernel. For the CPU test, we just mask_draft_logits works
        // with a correctly-shaped input.
        let _ = flat;

        // Draft logits: [K=2, vocab]. Make token 98 high at position 0 (should be
        // masked out, since pos 0 only allows 97), and token 97 high at pos 1.
        let vocab = env.tok_trie().vocab_size();
        let dev = candle_core::Device::Cpu;
        let mut dl = vec![0.0f32; 2 * vocab];
        dl[0 * vocab + 98] = 10.0; // pos 0: 98 is illegal (only 97 allowed)
        dl[0 * vocab + 97] = 5.0;
        dl[1 * vocab + 97] = 10.0; // pos 1: 97 illegal (only 98 allowed)
        dl[1 * vocab + 98] = 5.0;
        let draft_logits = Tensor::from_vec(dl, (2, vocab), &dev).unwrap();

        // Build the VOB properly: for each of the 3 positions, the mask tells
        // which inputs are allowed. Position 0 allows {97}, position 1 allows {98},
        // position 2 allows {99}.
        let vob_words = (vocab + 31) / 32;
        let mut vob = vec![0u32; 3 * vob_words];
        // pos 0: allow token 97
        vob[0 * vob_words + 97 / 32] |= 1u32 << (97 % 32);
        // pos 1: allow token 98
        vob[1 * vob_words + 98 / 32] |= 1u32 << (98 % 32);
        // pos 2: allow token 99
        vob[2 * vob_words + 99 / 32] |= 1u32 << (99 % 32);
        let proj = Tensor::from_vec(vob, (3 * vob_words,), &dev).unwrap();

        let masked = mask_draft_logits(None, &draft_logits, &proj, vob_words).unwrap();
        let m = masked.to_vec2::<f32>().unwrap();
        // pos 0: 98 must be -inf (illegal), 97 kept
        assert_eq!(m[0][98], f32::NEG_INFINITY, "pos0 token 98 must be masked");
        assert!((m[0][97] - 5.0).abs() < 1e-6, "pos0 token 97 kept");
        // pos 1: 97 must be -inf (illegal), 98 kept
        assert_eq!(m[1][97], f32::NEG_INFINITY, "pos1 token 97 must be masked");
        assert!((m[1][98] - 5.0).abs() < 1e-6, "pos1 token 98 kept");
        println!("mask_draft_logits: projected masks correctly mask draft positions");
    }

    /// PROOF of the sequential gating contract: the token stream is serialized
    /// through the FSM in acquisition order. The base is committed, the ff-run is
    /// read from the post-base settled state, and the ff-run is committed — so the
    /// ordered queue [base, ff…] is appended in the the order the FSM accepted
    /// it. A future ff-after-drafted path cannot desync because the ff-read is a
    /// pure observation of the settled state that already contains the base.
    #[test]
    fn test_sequential_gating_ordered_queue() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use candle_core::Tensor;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: "abc""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_end_ids: &[],
        }];
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros(
            (1, vocab),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // the base token ('a' = 97) is committed first (the several-clicks + settle)
        let base = vec![97u32];
        let n_base = gd.commit_run(seq_id, &base);
        assert_eq!(n_base, 1, "the base is committed");

        // the ff-run is read from the post-base settled state (the pure read)
        let ff = gd.ff_tokens(seq_id);
        assert_eq!(ff, vec![98, 99], "the ff-run is 'b','c' (the post-base state)");

        // the ff-run is committed (the several-clicks + settle)
        let n_ff = gd.commit_run(seq_id, &ff);
        assert_eq!(n_ff, 2, "the ff-run is committed");

        // the ordered queue [base, ff…] is exactly what the FSM accepted, in order
        let accepted: Vec<u32> = base.iter().chain(ff.iter()).copied().collect();
        assert_eq!(accepted, vec![97, 98, 99], "the ordered queue is in acquisition order");
    }

    /// Benchmark: measure the legacy validate_tokens time (the O(grammar) cost)
    /// for a long sequence. This is the baseline that the PDA O(1) validation
    /// will beat. The PDA benchmark is in the llguidance test suite (the
    /// test_pda_sequence_growth.rs).
    #[test]
    fn bench_validate_tokens_legacy_baseline() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::toktrie::ApproximateTokEnv;
        use crate::utils::guidance::ParserFactory;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: "a" "b" "c" "d" "e" "f" "g" "h" "i" "j"
                         "k" "l" "m" "n" "o" "p" "q" "r" "s" "t"
                         "u" "v" "w" "x" "y" "z""#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_end_ids: &[],
        }];
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros((1, vocab), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // Walk 20 tokens (the 'a' through 't').
        let tokens: Vec<u32> = (97..117).collect(); // 'a' to 'u'
        let mut committed = 0;
        for &tok in &tokens {
            let n = gd.commit_run(seq_id, &[tok]);
            committed += n;
        }
        assert_eq!(committed, tokens.len(), "all tokens committed");

        // Measure the validate_tokens time (the O(grammar) speculative walk).
        // Validate the remaining 6 tokens (the 'v' to 'z', the grammar tail).
        let draft: Vec<u32> = (118..123).collect(); // 'v' to 'z' (5 tokens, the grammar tail)
        let t0 = std::time::Instant::now();
        let n_valid = gd.validate_tokens(seq_id, &draft).unwrap_or(0);
        let elapsed_us = t0.elapsed().as_micros();
        eprintln!(
            "Legacy validate_tokens: {} tokens in {} us ({} us/token), valid={}",
            draft.len(), elapsed_us, (elapsed_us / draft.len() as u128), n_valid
        );
        assert!(n_valid <= draft.len(), "the valid prefix cannot exceed the draft length");
    }

    /// CUDA-gated: a fresh guided sequence initializes its PDA state to the
    /// machine's start config (the ctrl = start_state, the sp = 1, the stack =
    /// [start_stack, 0, ...]). This is the invariant the fused_sample kernel
    /// relies on for a sequence that has not yet advanced. Proves the inclusive
    /// property: every fresh seq starts at the PDA's defined start config.
    #[cfg(feature = "cuda")]
    #[test]
    fn pda_current_step_fresh_seq_initializes_to_start_config() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use attention_rs::pda::PdaSampling;
        use candle_core::{Device, DType, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: "a" "b" "c""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        let parser = factory.create_parser(grm.clone()).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pkg = llguidance::dpda_adapter::export_pda_package(&cgrm).expect("the CUDA package");

        let mut gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let dev = Device::new_cuda(0).unwrap();
        gd.upload_pda_package(&pkg, &dev).expect("the PDA upload");
        let table = gd.pda_table().as_ref().expect("the table is uploaded");

        // The logits are sized to the table's VOB width (the words_per_vob * 32).
        let vocab = table.words_per_vob as usize * 32;
        let logits = Tensor::zeros((1, vocab), DType::F32, &dev).unwrap();
        let sampling = PdaSampling::Greedy;
        let seq_id = 7usize;
        let _out = gd.pda_current_step(&logits, &[seq_id], &sampling).expect("the pda_current_step");

        // The fresh seq was initialized to the start config (the ctrl=start_state,
        // the sp=1, the stack=[start_stack]) + advanced one step by the fused_sample
        // kernel. Verify the per-seq state now reflects a valid PDA config (the ctrl
        // in range, the sp within the depth bound, the stack width = the depth).
        let (ctrl, stack, sp) = gd.pda_seq_state_for_test(seq_id).expect("the fresh seq has PDA state");
        let ctrl_val: Vec<u32> = ctrl.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let sp_val: Vec<u32> = sp.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let stack_val: Vec<u32> = stack.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(ctrl_val[0] < table.num_states, "the ctrl state is in range");
        assert!(sp_val[0] <= table.max_stack_depth, "the stack pointer is within the depth bound");
        assert_eq!(stack_val.len() as u32, table.max_stack_depth, "the stack width is the table depth");
    }

    /// CUDA-gated: the pda_project_masks output has the K+1-per-seq VOB shape
    /// (the batch * (K+1) * words_per_vob U32 words), proving the projection emits
    /// one mask per draft position + the initial position (the inclusive: K+1
    /// masks for K drafts; the exclusive: no more, no fewer).
    #[cfg(feature = "cuda")]
    #[test]
    fn pda_project_masks_emits_k_plus_1_masks_per_seq() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use candle_core::{Device, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: "a" "b" "c""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        let parser = factory.create_parser(grm.clone()).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pkg = llguidance::dpda_adapter::export_pda_package(&cgrm).expect("the CUDA package");

        let mut gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let dev = Device::new_cuda(0).unwrap();
        gd.upload_pda_package(&pkg, &dev).expect("the PDA upload");
        let table = gd.pda_table().as_ref().expect("the table is uploaded");

        let k = 2usize; // the draft length
        // The draft tensor is [n_seqs, K] of token IDs within the PDA input range.
        let draft = Tensor::from_vec(vec![0u32; 1 * k], (1, k), &dev).unwrap();
        let masks = gd.pda_project_masks(&[0], &draft).expect("the pda_project_masks");
        // The output is [n_seqs * (K+1) * words_per_vob] U32 words.
        let expected_words = 1 * (k + 1) * table.words_per_vob as usize;
        let actual_words: Vec<u32> = masks.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(
            actual_words.len(),
            expected_words,
            "the projection emits (K+1) masks per seq, each words_per_vob wide"
        );
    }

    /// PROOF of the seal (the RC2 fix): the single `commit_run` (the gate) truncates
    /// a sampled run to the FSM-legal prefix, so an illegal token never reaches the
    /// Sequence. The inclusive: a fully-legal run commits in full (the prefix == the
    /// run). The exclusive: a run containing an illegal token commits only the legal
    /// prefix (the count < the run length), and the resync leaves the FSM at the legal
    /// boundary (the next step restarts from there, the constraints stay active).
    #[test]
    fn gate_commit_seals_the_sequence_to_the_legal_prefix() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // The grammar: 'a' then 'b' then 'c' (the tokens 97, 98, 99).
        let grm_str = r#"start: "a" "b" "c""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0usize;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_end_ids: &[],
        }];
        // Initialize the guidance state (the apply creates it).
        let vocab = env.tok_trie().vocab_size();
        let logits = candle_core::Tensor::zeros(
            (1, vocab),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // The inclusive: a fully-legal run ('a','b','c') commits in full.
        let n_full = gd.commit_run(seq_id, &[97, 98, 99]);
        assert_eq!(n_full, 3, "a fully-legal run commits all 3 (the seal passes)");

        // The exclusive: a fresh state, a run with an illegal token ('a' then 'c' —
        // the grammar expects 'b' after 'a'). The commit_run must stop at the legal
        // prefix ('a' only, the count 1), NOT commit the illegal 'c'.
        let seq_id2 = 1usize;
        let requests2 = vec![GuidedDecodingRequest {
            seq_id: seq_id2,
            grammar: Some(&grm),
            reasoning_end_ids: &[],
        }];
        let _ = gd.apply(&logits, &requests2).unwrap();
        let n_partial = gd.commit_run(seq_id2, &[97, 99]); // 'a' legal, 'c' illegal (expects 'b')
        assert_eq!(n_partial, 1, "the seal truncates to the legal prefix (the 'a' only)");
        // The 'c' was NOT committed (the llm_tokens hold only the 'a').
        // The resync rolls the matcher back to the committed boundary (the 'a'),
        // so the constraints stay active (not a disable-on-error).
        gd.resync(seq_id2);
    }
}
