use crate::models::dflash::{DFlashDraftModel, DFlashModelConfig};
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;

pub struct DFlashDrafter {
    pub draft_model: DFlashDraftModel,
    pub target_layer_ids: Vec<usize>,
    pub num_speculative_tokens: usize,
    pub mask_token_id: u32,
    device: Device,
    _dtype: DType,
    cached_target_hidden: Mutex<HashMap<usize, Tensor>>,
    context_window: usize,
}

pub struct SpecDecodeOutput {
    pub accepted_tokens: Vec<Vec<u32>>,
    pub accepted_counts: Vec<usize>,
    pub logits: Tensor,
    pub hidden_states: Vec<Tensor>,
}

impl DFlashDrafter {
    pub fn new(
        draft_config: &DFlashModelConfig,
        draft_weight_files: &[PathBuf],
        comm: Rc<Comm>,
        dtype: DType,
        device: &Device,
        num_speculative_tokens: Option<usize>,
        yarn_factor: Option<f64>,
    ) -> Result<Self> {
        let draft_vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                draft_weight_files,
                dtype,
                device,
            )?
        };
        let draft_vb = VarBuilderX(
            either::Either::Left(draft_vb),
            String::new(),
            None,
            None,
            Some(draft_weight_files.to_vec()),
        );

        let draft_model =
            DFlashDraftModel::new(&draft_vb, comm, draft_config, dtype, device, yarn_factor)?;

        let target_layer_ids = draft_config.target_layer_ids();
        // DFlash config.block_size is the verification block width:
        // [known first token] + N draft tokens. The user-facing speculative
        // token count is N.
        let block_size =
            num_speculative_tokens.unwrap_or_else(|| draft_config.block_size().saturating_sub(1));
        let mask_token_id = draft_config.mask_token_id().unwrap_or(0);

        crate::log_info!(
            "DFlash drafter initialized: {} layers, num_speculative_tokens={}, target_layer_ids={:?}, mask_token_id={}",
            draft_config.num_hidden_layers,
            block_size,
            target_layer_ids,
            mask_token_id,
        );

        Ok(Self {
            draft_model,
            target_layer_ids,
            num_speculative_tokens: block_size,
            mask_token_id,
            device: device.clone(),
            _dtype: dtype,
            cached_target_hidden: Mutex::new(HashMap::new()),
            context_window: crate::utils::env::spec_context_window(),
        })
    }

    pub fn extract_and_concat_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        self.draft_model
            .extract_and_project_hidden(all_hidden_states)
    }

    pub fn project_layer_hiddens(&self, layer_hiddens: &[Tensor]) -> Result<Tensor> {
        self.draft_model.project_layer_hiddens(layer_hiddens)
    }

    /// Build the DFlash draft block inputs (eager): the cast target context, the noise
    /// embeddings (`[anchor, MASK x n]` via the target embed table), and the 0-based positions.
    pub fn build_draft_inputs(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        anchor: u32,
        n_mask: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dtype = self.draft_model.dtype();
        let mut block_ids = Vec::with_capacity(1 + n_mask);
        block_ids.push(anchor);
        block_ids.extend(std::iter::repeat(self.mask_token_id).take(n_mask));
        let block_len = block_ids.len();
        let block_tensor = Tensor::from_vec(
            block_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (block_len,),
            &self.device,
        )?;
        let noise_embedding = embed_fn(&block_tensor)?.to_dtype(dtype)?;
        let target_hidden_2d = if target_hidden.rank() == 3 {
            let (_, ctx, h) = target_hidden.dims3()?;
            target_hidden.reshape((ctx, h))?
        } else {
            target_hidden.clone()
        };
        let th_cast = target_hidden_2d.to_dtype(dtype)?;
        let noise_2d = if noise_embedding.rank() == 3 {
            let (_, s, h) = noise_embedding.dims3()?;
            noise_embedding.reshape((s, h))?
        } else {
            noise_embedding
        };
        let total_len = th_cast.dim(0)? + block_len;
        let positions = Tensor::from_vec(
            (0..total_len as i64).collect::<Vec<_>>(),
            (total_len,),
            &self.device,
        )?;
        Ok((th_cast, noise_2d, positions))
    }

    /// Run the draft transformer (graphable). Returns draft_hidden `[ctx + block, hidden]`.
    pub fn draft_forward(
        &self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        self.draft_model.forward(target_hidden, noise_embedding, positions)
    }

    /// The target lm_head logits over the trailing `n_mask` draft positions (eager).
    pub fn lm_head_logits(
        &self,
        draft_hidden_full: &Tensor,
        n_mask: usize,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let total_out = draft_hidden_full.dim(0)?;
        let hidden_n = draft_hidden_full.narrow(0, total_out - n_mask, n_mask)?;
        let logits = lm_head_fn(&hidden_n)?;
        Ok((logits, hidden_n))
    }

    /// Select draft tokens from pre-computed logits (DFlash2 selector, else argmax), applying
    /// the grammar mask (projection by default; the exact per-position FSM walk when
    /// XINFER_SPEC_GRANULAR_MASK is set). Unguided -> no gate.
    pub fn select_draft_tokens(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        anchor: u32,
        guided: &crate::utils::guided_decoding::GuidedDecoding,
        seq_id: usize,
    ) -> Result<Vec<u32>> {
        // The number of draft positions is the logits row count (adaptive K in the
        // single-seq path, the full count in the batch path).
        let n = logits.dim(0)?;
        let logits = if guided.is_guided(seq_id) {
            let vocab = logits.dim(1)?;
            let allow = if crate::utils::env::spec_granular_mask() {
                guided.draft_allow_walk(seq_id, logits, vocab)?
            } else {
                guided.draft_allow_repeated(seq_id, n, vocab, &self.device)?
            };
            match allow {
                Some(a) => {
                    let neg_inf = Tensor::full(
                        f32::NEG_INFINITY,
                        logits.shape().clone(),
                        &self.device,
                    )?
                    .to_dtype(logits.dtype())?;
                    a.where_cond(logits, &neg_inf)?
                }
                None => logits.clone(),
            }
        } else {
            logits.clone()
        };
        // DFlash2: candidate selector walks a top-k lattice. Keep greedy argmax as fallback.
        if self.draft_model.is_dflash2() {
            return self.draft_model.select_candidates(hidden_n, &logits, anchor);
        }
        // Single GPU argmax over all rows (one op, one D2H) instead of n per-row argmax + n D2H.
        let draft_token_ids = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        Ok(draft_token_ids)
    }

    /// Grammar-gated draft-token selection, GPU-resident: returns the K draft token ids as a
    /// `[K]` u32 GPU tensor (no D2H), so the verify block can be built on-device. The v1 path
    /// is a single argmax; the v2 selector still returns CPU ids (one small H2D) until a
    /// GPU-native selector lands in attention-rs.
    pub fn select_draft_tokens_gpu(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        anchor: u32,
        guided: &crate::utils::guided_decoding::GuidedDecoding,
        seq_id: usize,
    ) -> Result<Tensor> {
        let n = logits.dim(0)?;
        let logits = if guided.is_guided(seq_id) {
            let vocab = logits.dim(1)?;
            let allow = if crate::utils::env::spec_granular_mask() {
                guided.draft_allow_walk(seq_id, logits, vocab)?
            } else {
                guided.draft_allow_repeated(seq_id, n, vocab, &self.device)?
            };
            match allow {
                Some(a) => {
                    let neg_inf = Tensor::full(
                        f32::NEG_INFINITY,
                        logits.shape().clone(),
                        &self.device,
                    )?
                    .to_dtype(logits.dtype())?;
                    a.where_cond(logits, &neg_inf)?
                }
                None => logits.clone(),
            }
        } else {
            logits.clone()
        };
        if self.draft_model.is_dflash2() {
            let tokens = self.draft_model.select_candidates(hidden_n, &logits, anchor)?;
            return Tensor::from_vec(
                tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                (tokens.len(),),
                &self.device,
            )?
            .to_dtype(DType::U32);
        }
        logits.argmax(D::Minus1)?.to_dtype(DType::U32)
    }

    /// Full draft step (eager, no graph): build inputs -> draft transformer -> lm_head -> select.
    pub fn draft_tokens(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        last_tokens: &[u32],
        guided: &crate::utils::guided_decoding::GuidedDecoding,
        seq_id: usize,
    ) -> Result<Vec<u32>> {
        assert_eq!(
            last_tokens.len(),
            1,
            "DFlash currently supports batch_size=1 for drafting"
        );
        let n = self.num_speculative_tokens;
        let (th_cast, noise_2d, positions) =
            self.build_draft_inputs(target_hidden, embed_fn, last_tokens[0], n)?;
        let draft_hidden_full = self.draft_forward(&th_cast, &noise_2d, &positions)?;
        let (logits, hidden_n) = self.lm_head_logits(&draft_hidden_full, n, lm_head_fn)?;
        self.select_draft_tokens(&logits, &hidden_n, last_tokens[0], guided, seq_id)
    }

    pub fn verify_tokens(
        draft_tokens: &[u32],
        target_logits: &Tensor,
        _temperature: f32,
    ) -> Result<(Vec<u32>, usize)> {
        let n_draft = draft_tokens.len();
        let mut accepted = Vec::new();

        for i in 0..n_draft {
            let argmax_res = target_logits.i(i)?.argmax(D::Minus1)?;
            let target_token = if argmax_res.rank() > 0 {
                argmax_res.flatten_all()?.i(0)?.to_vec0::<u32>()?
            } else {
                argmax_res.to_vec0::<u32>()?
            };

            if i < n_draft && target_token == draft_tokens[i] {
                accepted.push(target_token);
            } else {
                accepted.push(target_token);
                let len = accepted.len();
                return Ok((accepted, len));
            }
        }

        let bonus_argmax = target_logits.i(n_draft)?.argmax(D::Minus1)?;
        let bonus_token = if bonus_argmax.rank() > 0 {
            bonus_argmax.flatten_all()?.i(0)?.to_vec0::<u32>()?
        } else {
            bonus_argmax.to_vec0::<u32>()?
        };
        accepted.push(bonus_token);

        let len = accepted.len();
        Ok((accepted, len))
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub fn clear_cached_hidden(&self) {
        self.cached_target_hidden.lock().unwrap().clear();
    }

    pub fn clear_seq_hidden(&self, seq_id: usize) {
        self.cached_target_hidden.lock().unwrap().remove(&seq_id);
    }

    /// Return the accumulated hidden state context for drafting, or None if empty.
    pub fn build_draft_context(&self, seq_id: usize) -> Result<Option<Tensor>> {
        let cached = self.cached_target_hidden.lock().unwrap();
        Ok(cached.get(&seq_id).cloned())
    }

    /// After verification, append hidden states for the verified input tokens.
    /// verify_hidden covers [first_token, d0, ..., d_{n-1}] at rows [0, 1, ..., n].
    /// The recovered/bonus token is produced by logits and is not part of this forward input,
    /// so the correct number of rows to keep is accepted_count.
    pub fn replace_with_verified_hidden(
        &self,
        verify_hidden: &Tensor,
        accepted_count: usize,
        seq_id: usize,
    ) -> Result<()> {
        let mut cached = self.cached_target_hidden.lock().unwrap();

        let vdim = verify_hidden.dim(0)?;
        if accepted_count == 0 || vdim == 0 {
            return Ok(());
        }

        let keep = std::cmp::min(accepted_count, vdim);
        let verified_inputs = verify_hidden.narrow(0, 0, keep)?;

        let new_ctx = if let Some(prev) = cached.get(&seq_id).cloned() {
            Tensor::cat(&[prev, verified_inputs], 0)?
        } else {
            verified_inputs
        };
        // Cap the per-seq context to the last `context_window` rows (0 = unbounded).
        let new_ctx = if self.context_window > 0 {
            let total = new_ctx.dim(0)?;
            let keep = std::cmp::min(total, self.context_window);
            if keep < total {
                new_ctx.narrow(0, total - keep, keep)?
            } else {
                new_ctx
            }
        } else {
            new_ctx
        };
        cached.insert(seq_id, new_ctx);
        Ok(())
    }

    /// Store hidden states from a forward pass (prefill or decode).
    /// Appends to the accumulated context.
    pub fn store_decode_hidden(&self, hidden: &Tensor, seq_id: usize) -> Result<()> {
        let mut cached = self.cached_target_hidden.lock().unwrap();
        let new_ctx = if let Some(prev) = cached.get(&seq_id).cloned() {
            Tensor::cat(&[prev, hidden.clone()], 0)?
        } else {
            hidden.clone()
        };
        // Cap the per-seq context to the last `context_window` rows (0 = unbounded).
        let new_ctx = if self.context_window > 0 {
            let total = new_ctx.dim(0)?;
            let w = std::cmp::min(total, self.context_window);
            if w < total {
                new_ctx.narrow(0, total - w, w)?
            } else {
                new_ctx
            }
        } else {
            new_ctx
        };
        cached.insert(seq_id, new_ctx);
        Ok(())
    }
}

use crate::core::mtp::{verify_draft_greedy, MtpSeqInfo};
use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;

impl ModelRunner {
    /// Run one external DFlash speculative step.
    ///
    /// DFlash shares the MTP verification metadata and Mamba rollback path,
    /// while its draft model consumes projected intermediate target states.
    pub fn run_dflash_decode(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        let batch_size = match seqs {
            Seqs::SeqRefs(ref refs) => refs.len(),
            Seqs::DecodeVec(ref decoded) => decoded.len(),
        };
        if batch_size > 1 {
            // Parallel DFlash: run each sequence's single-seq DFlash step separately.
            // The batched verify (run_dflash_decode_batch) crashes in the target's
            // GDN layer on multi-seq input; per-seq single-seq verifies are GDN-safe.
            // Cap the drafting slots; sequences beyond the cap plain-decode.
            let slots = crate::utils::env::dflash_parallel_slots();
            let mut results: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
            match seqs {
                Seqs::SeqRefs(refs) => {
                    for (i, &seq) in refs.iter().enumerate() {
                        let single = Seqs::SeqRefs(&[seq]);
                        if i < slots {
                            results.extend(self.run_dflash_decode(single)?);
                        } else {
                            results.push(self.run(single, false)?);
                        }
                    }
                }
                Seqs::DecodeVec(decoded) => {
                    for (i, dseq) in decoded.iter().enumerate() {
                        let single = vec![dseq.clone()];
                        let s = Seqs::DecodeVec(&single);
                        if i < slots {
                            results.extend(self.run_dflash_decode(s)?);
                        } else {
                            results.push(self.run(s, false)?);
                        }
                    }
                }
            }
            return Ok(results);
        }
        let Some(drafter) = self.dflash_drafter.as_ref() else {
            let output = self.run(seqs, false)?;
            return Ok(output.into_iter().map(|token| vec![token]).collect());
        };

        let seq_info = match seqs {
            Seqs::SeqRefs(ref refs) if refs.len() == 1 => MtpSeqInfo {
                id: refs[0].id,
                len: refs[0].len(),
                block_table: refs[0].block_table.clone(),
            },
            Seqs::DecodeVec(ref decoded) if decoded.len() == 1 => MtpSeqInfo {
                id: decoded[0].id,
                len: decoded[0].len,
                block_table: decoded[0].block_tables.clone(),
            },
            _ => {
                let output = self.run(seqs, false)?;
                return Ok(output.into_iter().map(|token| vec![token]).collect());
            }
        };

        let (input_ids, positions, mut input_metadata) = match seqs {
            Seqs::SeqRefs(ref refs) => self.prepare_decode(refs.iter())?,
            Seqs::DecodeVec(ref decoded) => self.prepare_decode(decoded.iter())?,
        };
        #[cfg(feature = "flashinfer")]
        if let Some(flashinfer_metadata) = input_metadata.flashinfer_metadata.as_mut() {
            if flashinfer_metadata.decode_plan_info.is_none() {
                if let Some(params) = self.flashinfer_kv_params() {
                    flashinfer_metadata.decode_plan_info =
                        Some(attention_rs::flashinfer::decode_plan(
                            self.device(),
                            params.kv_dtype,
                            params.out_dtype,
                            &flashinfer_metadata.indptr_host,
                            flashinfer_metadata.last_len_host.as_deref(),
                            flashinfer_metadata.kv_len_arr_host.as_deref(),
                            input_ids.dim(0)?,
                            params.num_qo_heads,
                            params.num_kv_heads,
                            params.head_dim,
                            params.page_size,
                            flashinfer_metadata.use_cuda_graph,
                        )?);
                }
            }
        }
        let _decode_guard = set_linear_is_prefill(false);
        let (anchor_logits, anchor_hidden_states) = {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let result = match self.model() {
                Model::Qwen3_5(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3_5MoE(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3VL(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }?;
            drop(kv_cache);
            result
        };
        let anchor_token = self.sample(&anchor_logits, seqs, false)?[0];

        let projected_step_hidden = drafter.extract_and_concat_hidden(&anchor_hidden_states)?;
        let cached_context = drafter.build_draft_context(seq_info.id)?;
        if cached_context
            .as_ref()
            .map_or(true, |context| context.dim(0).unwrap_or(0) < seq_info.len)
        {
            drafter.store_decode_hidden(&projected_step_hidden, seq_info.id)?;
        }
        let target_hidden = drafter
            .build_draft_context(seq_info.id)?
            .ok_or_else(|| candle_core::Error::Msg("DFlash target hidden cache is empty".into()))?;

        let embed_fn = |tokens: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.embed_forward(tokens),
                Model::Qwen3_5MoE(model) => model.embed_forward(tokens),
                Model::Qwen3VL(model) => model.embed_forward(tokens),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.forward_lm_head(hidden),
                Model::Qwen3_5MoE(model) => model.forward_lm_head(hidden),
                Model::Qwen3VL(model) => model.forward_lm_head(hidden),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }
        };
        let n = if self.adaptive_enabled {
            self.adaptive_spec.lock().unwrap().current_k()
        } else {
            drafter.num_speculative_tokens
        };
        // Pad the draft block to the captured size so the draft graph replays at
        // every tier (graph->graph, no flip). The draft attention is causal, so
        // the first `n` rows of the padded block are exact; the logits are
        // narrowed back to `n` below.
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let use_draft_graph = {
            let ctx_rows = if target_hidden.rank() == 3 {
                target_hidden.dims3()?.1
            } else {
                target_hidden.dim(0)?
            };
            self.dflash_draft_graph
                .as_ref()
                .map_or(false, |g| g.is_captured() && g.cap() == ctx_rows)
        };
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let use_draft_graph = false;
        let n_pad = if use_draft_graph {
            drafter.num_speculative_tokens.max(n)
        } else {
            n
        };
        let (th_cast, noise_2d, positions) =
            drafter.build_draft_inputs(&target_hidden, &embed_fn, anchor_token, n_pad)?;
        let draft_hidden_full = {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            {
                if use_draft_graph {
                    self.dflash_draft_graph
                        .as_ref()
                        .unwrap()
                        .replay(&th_cast, &noise_2d, &positions)?
                } else {
                    drafter.draft_forward(&th_cast, &noise_2d, &positions)?
                }
            }
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            {
                drafter.draft_forward(&th_cast, &noise_2d, &positions)?
            }
        };
        let (draft_logits, draft_hidden_n) =
            drafter.lm_head_logits(&draft_hidden_full, n_pad, &lm_head_fn)?;
        let (draft_logits, draft_hidden_n) = if n < n_pad {
            (draft_logits.narrow(0, 0, n)?, draft_hidden_n.narrow(0, 0, n)?)
        } else {
            (draft_logits, draft_hidden_n)
        };
        let draft_tokens_gpu = drafter.select_draft_tokens_gpu(
            &draft_logits,
            &draft_hidden_n,
            anchor_token,
            &self.guided_decoding,
            seq_info.id,
        )?;
        let draft_count = draft_tokens_gpu.dim(0)?;
        if draft_count == 0 {
            return Ok(vec![vec![anchor_token]]);
        }

        // DFA grammar validation: truncate draft at first illegal token.
        let draft_cpu: Vec<u32> = draft_tokens_gpu.flatten_all()?.to_vec1::<u32>()?;
        let dfa_legal = self.guided_decoding.validate_tokens(seq_info.id, &draft_cpu)?;
        let draft_tokens_gpu = if dfa_legal < draft_cpu.len() {
            crate::log_info!(
                "[DFlash] DFA truncated draft: {} -> {} legal tokens (seq {})",
                draft_cpu.len(), dfa_legal, seq_info.id
            );
            if dfa_legal == 0 {
                return Ok(vec![vec![anchor_token]]);
            }
            draft_tokens_gpu.narrow(0, 0, dfa_legal)?
        } else {
            draft_tokens_gpu
        };
        let draft_count = draft_tokens_gpu.dim(0)?;

        let verify_len = 1 + draft_count;
        let slot_mappings =
            self.compute_slot_mappings(&seq_info, verify_len, self.block_size(), "DFlash verify")?;
        // Build the verify block on-device: [anchor, drafts] (no D2H of the draft tokens, no
        // H2D of the block). The  is a single-token H2D (negligible).
        let anchor_gpu = Tensor::from_vec(vec![anchor_token as i64], (1,), self.device())?
            .to_dtype(DType::U32)?;
        let verify_input_ids = Tensor::cat(&[&anchor_gpu, &draft_tokens_gpu], 0)?;
        let verify_positions = Tensor::from_vec(
            (seq_info.len..seq_info.len + verify_len)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            (verify_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata(&seq_info, &slot_mappings, verify_len)?;
        let _verify_guard = set_linear_is_prefill(true);

        // MTP-style: use CUDA-graph verify when captured. build_mtp_metadata sets
        // flashinfer use_cuda_graph / prefill_plan_info based on the same condition,
        // so eager and graph paths must stay in sync.
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let use_verify_graph = self
            .mtp_capturer
            .as_ref()
            .map_or(false, |c| c.is_mtp_captured(verify_len));
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let use_verify_graph = false;

        let (verify_logits, projected_verify_hidden) = if use_verify_graph {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            {
                // Hold mamba cache across verify-graph replay (same as decode graphs):
                // GDN gather/scatter mutates cache storage in-place under the graph.
                let logits = match self.model() {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions,
                            &verify_metadata,
                        )?
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions,
                            &verify_metadata,
                        )?
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions,
                                &verify_metadata,
                            )?
                        } else {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions,
                                &verify_metadata,
                            )?
                        }
                    }
                    _ => self.mtp_capturer.as_ref().unwrap().replay_mtp(
                        &verify_input_ids,
                        &verify_positions,
                        &verify_metadata,
                    )?,
                };
                let layer_hiddens = match self.model() {
                    Model::Qwen3_5(model) => model.take_dflash_verify_hiddens(verify_len),
                    Model::Qwen3_5MoE(model) => model.take_dflash_verify_hiddens(verify_len),
                    Model::Qwen3VL(model) => model.take_dflash_verify_hiddens(verify_len),
                    _ => None,
                }
                .ok_or_else(|| {
                    candle_core::Error::Msg(
                        "DFlash verify missing layer-hidden buffers after graph replay".into(),
                    )
                })?;
                (logits, drafter.project_layer_hiddens(&layer_hiddens)?)
            }
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            {
                unreachable!()
            }
        } else {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let (logits, hidden_states) = match self.model() {
                Model::Qwen3_5(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                )?,
                Model::Qwen3_5MoE(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                )?,
                Model::Qwen3VL(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                )?,
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            };
            drop(kv_cache);
            (logits, drafter.extract_and_concat_hidden(&hidden_states)?)
        };

        let verify_result = if self.guided_decoding.is_guided(seq_info.id) {
            // guided: D2H the draft tokens (small) for the FSM firewall
            let draft_tokens_cpu = draft_tokens_gpu.to_vec1::<u32>()?;
            crate::core::mtp::verify_draft_masked(
                &verify_logits,
                &draft_tokens_cpu,
                &self.guided_decoding,
                seq_info.id,
            )?
        } else {
            // Copy the sampling out and release the guard before the (GPU) verify.
            let sampling = {
                let cached = self.cached_sampling.read();
                cached
                    .as_ref()
                    .map(|c| c.sampling.clone())
                    .unwrap_or(crate::utils::logits_processor::Sampling::ArgMax)
            };
            if matches!(
                sampling,
                crate::utils::logits_processor::Sampling::ArgMax
            ) || !crate::utils::env::spec_rejection_sampling()
            {
                // greedy: GPU-resident verify (no D2H of the draft tokens)
                crate::core::mtp::verify_draft_greedy_gpu(&verify_logits, &draft_tokens_gpu)?
            } else {
                // rejection: D2H the draft tokens (small) for the CPU rejection sampling
                let draft_tokens_cpu = draft_tokens_gpu.to_vec1::<u32>()?;
                crate::core::mtp::verify_draft_rejection(&verify_logits, &draft_tokens_cpu, &sampling)?
            }
        };
        if self.adaptive_enabled {
            self.adaptive_spec.lock().unwrap().update(&[verify_result.num_accepted]);
        }
        let commit_len = 1 + verify_result.num_accepted;
        if verify_result.num_accepted < verify_result.num_proposed {
            if !self.mtp_rollback_mamba(seq_info.id, commit_len)? {
                candle_core::bail!(
                    "DFlash failed to roll back mamba state for sequence {}",
                    seq_info.id
                );
            }
        }

        drafter.replace_with_verified_hidden(&projected_verify_hidden, commit_len, seq_info.id)?;

        let mut result_tokens = Vec::with_capacity(commit_len + 1);
        result_tokens.push(anchor_token);
        result_tokens.extend_from_slice(&verify_result.accepted_tokens);
        result_tokens.push(verify_result.continuation_token);
        let name = if drafter.draft_model.is_dflash2() { "DFlash2" } else { "DFlash1" };
        crate::core::spec_stats::spec_stats_update(name, seq_info.id, &verify_result);
        Ok(vec![result_tokens])
    }

    #[allow(dead_code)]
    // Retained for when the target's GDN layer supports a batched verify; the
    // parallel path currently runs per-seq single-seq steps instead.
    fn run_dflash_decode_batch(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        let Some(drafter) = self.dflash_drafter.as_ref() else {
            let output = self.run(seqs, false)?;
            return Ok(output.into_iter().map(|token| vec![token]).collect());
        };
        let seq_infos = match seqs {
            Seqs::SeqRefs(ref refs) => refs
                .iter()
                .map(|seq| MtpSeqInfo {
                    id: seq.id,
                    len: seq.len(),
                    block_table: seq.block_table.clone(),
                })
                .collect::<Vec<_>>(),
            Seqs::DecodeVec(ref decoded) => decoded
                .iter()
                .map(|seq| MtpSeqInfo {
                    id: seq.id,
                    len: seq.len,
                    block_table: seq.block_tables.clone(),
                })
                .collect::<Vec<_>>(),
        };
        let (input_ids, positions, mut input_metadata) = match seqs {
            Seqs::SeqRefs(ref refs) => self.prepare_decode(refs.iter())?,
            Seqs::DecodeVec(ref decoded) => self.prepare_decode(decoded.iter())?,
        };
        #[cfg(feature = "flashinfer")]
        if let Some(flashinfer_metadata) = input_metadata.flashinfer_metadata.as_mut() {
            if flashinfer_metadata.decode_plan_info.is_none() {
                if let Some(params) = self.flashinfer_kv_params() {
                    flashinfer_metadata.decode_plan_info =
                        Some(attention_rs::flashinfer::decode_plan(
                            self.device(),
                            params.kv_dtype,
                            params.out_dtype,
                            &flashinfer_metadata.indptr_host,
                            flashinfer_metadata.last_len_host.as_deref(),
                            flashinfer_metadata.kv_len_arr_host.as_deref(),
                            input_ids.dim(0)?,
                            params.num_qo_heads,
                            params.num_kv_heads,
                            params.head_dim,
                            params.page_size,
                            flashinfer_metadata.use_cuda_graph,
                        )?);
                }
            }
        }
        let _decode_guard = set_linear_is_prefill(false);
        let (anchor_logits, anchor_hidden_states) = {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let result = match self.model() {
                Model::Qwen3_5(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3_5MoE(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3VL(model) => model.forward_with_hidden_states(
                    &input_ids,
                    &positions,
                    kv_pairs,
                    &input_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }?;
            drop(kv_cache);
            result
        };
        let anchor_tokens = self.sample(&anchor_logits, seqs, false)?;
        let projected_step_hidden = drafter.extract_and_concat_hidden(&anchor_hidden_states)?;
        for (index, seq_info) in seq_infos.iter().enumerate() {
            if drafter
                .build_draft_context(seq_info.id)?
                .map_or(true, |context| context.dim(0).unwrap_or(0) < seq_info.len)
            {
                drafter.store_decode_hidden(
                    &projected_step_hidden.narrow(0, index, 1)?,
                    seq_info.id,
                )?;
            }
        }

        let embed_fn = |tokens: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.embed_forward(tokens),
                Model::Qwen3_5MoE(model) => model.embed_forward(tokens),
                Model::Qwen3VL(model) => model.embed_forward(tokens),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.forward_lm_head(hidden),
                Model::Qwen3_5MoE(model) => model.forward_lm_head(hidden),
                Model::Qwen3VL(model) => model.forward_lm_head(hidden),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }
        };

        let mut draft_tokens = Vec::with_capacity(seq_infos.len());
        for (seq_info, &anchor_token) in seq_infos.iter().zip(&anchor_tokens) {
            let target_hidden = drafter.build_draft_context(seq_info.id)?.ok_or_else(|| {
                candle_core::Error::Msg("DFlash target hidden cache is empty".into())
            })?;
            draft_tokens.push(drafter.draft_tokens(
                &target_hidden,
                &embed_fn,
                &lm_head_fn,
                &[anchor_token],
                &self.guided_decoding,
                seq_info.id,
            )?);
        }
        let verify_len = 1 + drafter.num_speculative_tokens;
        if draft_tokens
            .iter()
            .any(|tokens| tokens.len() != drafter.num_speculative_tokens)
        {
            candle_core::bail!("DFlash2 batch produced an unexpected draft length");
        }
        let mut verify_tokens = Vec::with_capacity(seq_infos.len() * verify_len);
        for (anchor, drafts) in anchor_tokens.iter().zip(&draft_tokens) {
            verify_tokens.push(*anchor);
            verify_tokens.extend_from_slice(drafts);
        }
        let q_lens = vec![verify_len; seq_infos.len()];
        let slot_mappings = seq_infos
            .iter()
            .map(|seq_info| {
                self.compute_slot_mappings(
                    seq_info,
                    verify_len,
                    self.block_size(),
                    "DFlash batch verify",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let verify_input_ids = Tensor::from_vec(
            verify_tokens,
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let mut verify_position_ids = Vec::with_capacity(seq_infos.len() * verify_len);
        for seq_info in &seq_infos {
            verify_position_ids
                .extend((seq_info.len..seq_info.len + verify_len).map(|position| position as i64));
        }
        let verify_positions = Tensor::from_vec(
            verify_position_ids,
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata_batch(&seq_infos, &slot_mappings, &q_lens)?;
        let _verify_guard = set_linear_is_prefill(true);
        // Batch verify stays on the same forward_with_hidden_states path; single-seq
        // uses the MTP-style CUDA-graphable forward + layer buffers above.
        let (verify_logits, verify_hidden_states) = {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let result = match self.model() {
                Model::Qwen3_5(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3_5MoE(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                Model::Qwen3VL(model) => model.forward_with_hidden_states(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                    drafter.target_layer_ids(),
                ),
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            }?;
            drop(kv_cache);
            result
        };
        let projected_verify_hidden = drafter.extract_and_concat_hidden(&verify_hidden_states)?;
        let mut result = Vec::with_capacity(seq_infos.len());
        for (index, ((seq_info, drafts), &anchor_token)) in seq_infos
            .iter()
            .zip(&draft_tokens)
            .zip(&anchor_tokens)
            .enumerate()
        {
            let offset = index * verify_len;
            let per_seq_logits = verify_logits.narrow(0, offset, verify_len)?;
            let verify_result = verify_draft_greedy(&per_seq_logits, drafts)?;
            let commit_len = 1 + verify_result.num_accepted;
            if verify_result.num_accepted < verify_result.num_proposed
                && !self.mtp_rollback_mamba_at(seq_info.id, commit_len, offset)?
            {
                candle_core::bail!(
                    "DFlash failed to roll back mamba state for sequence {}",
                    seq_info.id
                );
            }
            drafter.replace_with_verified_hidden(
                &projected_verify_hidden.narrow(0, offset, verify_len)?,
                commit_len,
                seq_info.id,
            )?;
            let mut output = Vec::with_capacity(commit_len + 1);
            output.push(anchor_token);
            output.extend_from_slice(&verify_result.accepted_tokens);
            output.push(verify_result.continuation_token);
            let name = if drafter.draft_model.is_dflash2() { "DFlash2" } else { "DFlash1" };
            crate::core::spec_stats::spec_stats_update(name, seq_info.id, &verify_result);
            result.push(output);
        }
        Ok(result)
    }
}
