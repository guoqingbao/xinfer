use crate::models::dflash::{DFlashDraftModel, DFlashModelConfig};
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static DFLASH_TOTAL_PROPOSED: AtomicUsize = AtomicUsize::new(0);
static DFLASH_TOTAL_ACCEPTED: AtomicUsize = AtomicUsize::new(0);
static DFLASH_TOTAL_STEPS: AtomicUsize = AtomicUsize::new(0);

fn dflash_stats_update(proposed: usize, accepted: usize) {
    DFLASH_TOTAL_PROPOSED.fetch_add(proposed, Ordering::Relaxed);
    DFLASH_TOTAL_ACCEPTED.fetch_add(accepted, Ordering::Relaxed);
    let steps = DFLASH_TOTAL_STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if steps % 64 == 0 {
        let total_proposed = DFLASH_TOTAL_PROPOSED.load(Ordering::Relaxed);
        let total_accepted = DFLASH_TOTAL_ACCEPTED.load(Ordering::Relaxed);
        crate::log_info!(
            "DFlash Stats: steps={}, proposed={}, accepted={}, average_acceptance_rate={:.2}%",
            steps,
            total_proposed,
            total_accepted,
            if total_proposed == 0 {
                0.0
            } else {
                total_accepted as f64 / total_proposed as f64 * 100.0
            }
        );
    }
}

pub struct DFlashDrafter {
    pub draft_model: DFlashDraftModel,
    pub target_layer_ids: Vec<usize>,
    pub num_speculative_tokens: usize,
    pub mask_token_id: u32,
    device: Device,
    _dtype: DType,
    cached_target_hidden: Mutex<HashMap<usize, Tensor>>,
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
    ) -> Result<Self> {
        let draft_vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                draft_weight_files,
                DType::BF16,
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
            DFlashDraftModel::new(&draft_vb, comm, draft_config, DType::BF16, device)?;

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
        })
    }

    pub fn extract_and_concat_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        self.draft_model
            .extract_and_project_hidden(all_hidden_states)
    }

    pub fn project_layer_hiddens(&self, layer_hiddens: &[Tensor]) -> Result<Tensor> {
        self.draft_model.project_layer_hiddens(layer_hiddens)
    }

    pub fn draft_tokens(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        last_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let batch_size = last_tokens.len();
        assert_eq!(
            batch_size, 1,
            "DFlash currently supports batch_size=1 for drafting"
        );

        let n = self.num_speculative_tokens;
        let mut draft_token_ids: Vec<u32> = Vec::with_capacity(n);

        let mut block_ids = vec![self.mask_token_id; n + 1];
        block_ids[0] = last_tokens[0];

        let block_tensor = Tensor::from_vec(
            block_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (n + 1,),
            &self.device,
        )?;

        let noise_embedding = embed_fn(&block_tensor)?;
        let noise_embedding = noise_embedding.to_dtype(DType::BF16)?;

        let target_hidden_2d = if target_hidden.rank() == 3 {
            let (_, ctx, h) = target_hidden.dims3()?;
            target_hidden.reshape((ctx, h))?
        } else {
            target_hidden.clone()
        };
        let target_hidden_bf16 = target_hidden_2d.to_dtype(DType::BF16)?;

        let ctx_len = target_hidden_bf16.dim(0)?;
        let noise_2d = if noise_embedding.rank() == 3 {
            let (_, s, h) = noise_embedding.dims3()?;
            noise_embedding.reshape((s, h))?
        } else {
            noise_embedding
        };

        let total_len = ctx_len + n + 1;
        let positions: Vec<i64> = (0..total_len as i64).collect();
        let positions_tensor = Tensor::from_vec(positions, (total_len,), &self.device)?;

        let draft_hidden =
            self.draft_model
                .forward(&target_hidden_bf16, &noise_2d, &positions_tensor)?;

        let total_out = draft_hidden.dim(0)?;
        let draft_hidden = draft_hidden.narrow(0, total_out - n, n)?;
        if self.draft_model.is_dflash2() {
            return self.draft_model.select_candidates(
                &draft_hidden,
                &lm_head_fn(&draft_hidden)?,
                last_tokens[0],
            );
        }
        let draft_logits = lm_head_fn(&draft_hidden)?;

        for i in 0..n {
            let logit_slice = draft_logits.i(i)?;
            let argmax_result = logit_slice.argmax(D::Minus1)?;
            let token_id = if argmax_result.rank() > 0 {
                argmax_result.flatten_all()?.i(0)?.to_vec0::<u32>()?
            } else {
                argmax_result.to_vec0::<u32>()?
            };
            draft_token_ids.push(token_id);
        }

        Ok(draft_token_ids)
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

        if let Some(prev) = cached.get(&seq_id).cloned() {
            cached.insert(seq_id, Tensor::cat(&[prev, verified_inputs], 0)?);
        } else {
            cached.insert(seq_id, verified_inputs);
        }
        Ok(())
    }

    /// Store hidden states from a forward pass (prefill or decode).
    /// Appends to the accumulated context.
    pub fn store_decode_hidden(&self, hidden: &Tensor, seq_id: usize) -> Result<()> {
        let mut cached = self.cached_target_hidden.lock().unwrap();
        if let Some(prev) = cached.get(&seq_id).cloned() {
            cached.insert(seq_id, Tensor::cat(&[prev, hidden.clone()], 0)?);
        } else {
            cached.insert(seq_id, hidden.clone());
        }
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
            return self.run_dflash_decode_batch(seqs);
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
        let draft_tokens =
            drafter.draft_tokens(&target_hidden, &embed_fn, &lm_head_fn, &[anchor_token])?;
        if draft_tokens.is_empty() {
            return Ok(vec![vec![anchor_token]]);
        }

        let mut verify_tokens = vec![anchor_token];
        verify_tokens.extend_from_slice(&draft_tokens);
        let verify_len = verify_tokens.len();
        let slot_mappings =
            self.compute_slot_mappings(&seq_info, verify_len, self.block_size(), "DFlash verify")?;
        let verify_input_ids = Tensor::from_vec(verify_tokens, (verify_len,), self.device())?;
        let verify_positions = Tensor::from_vec(
            (seq_info.len..seq_info.len + verify_len)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            (verify_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata(&seq_info, &slot_mappings, verify_len)?;
        let _verify_guard = set_linear_is_prefill(true);

        // Same pattern as MTP: one verify forward. Prefer CUDA-graph replay when
        // captured; otherwise eager forward. Both paths set is_mtp_verify and write
        // layer hiddens into the same preallocated buffers.
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let verify_logits = if self
            .mtp_capturer
            .as_ref()
            .map_or(false, |c| c.is_mtp_captured(verify_len))
        {
            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                &verify_input_ids,
                &verify_positions,
                &verify_metadata,
            )?
        } else {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let logits = match self.model() {
                Model::Qwen3_5(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                )?,
                Model::Qwen3_5MoE(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                )?,
                Model::Qwen3VL(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    None,
                )?,
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            };
            drop(kv_cache);
            logits
        };
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let verify_logits = {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let logits = match self.model() {
                Model::Qwen3_5(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                )?,
                Model::Qwen3_5MoE(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    false,
                )?,
                Model::Qwen3VL(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions,
                    kv_pairs,
                    &verify_metadata,
                    None,
                )?,
                _ => candle_core::bail!("DFlash currently supports Qwen3.5 target models"),
            };
            drop(kv_cache);
            logits
        };
        let layer_hiddens = match self.model() {
            Model::Qwen3_5(model) => model.take_dflash_verify_hiddens(verify_len),
            Model::Qwen3_5MoE(model) => model.take_dflash_verify_hiddens(verify_len),
            Model::Qwen3VL(model) => model.take_dflash_verify_hiddens(verify_len),
            _ => None,
        }
        .ok_or_else(|| {
            candle_core::Error::Msg(
                "DFlash verify missing layer-hidden buffers (was preallocate_dflash_verify_buffers called?)"
                    .into(),
            )
        })?;
        let projected_verify_hidden = drafter.project_layer_hiddens(&layer_hiddens)?;

        let verify_result = verify_draft_greedy(&verify_logits, &draft_tokens)?;
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
        dflash_stats_update(verify_result.num_proposed, verify_result.num_accepted);
        Ok(vec![result_tokens])
    }

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
                && !self.mtp_rollback_mamba(seq_info.id, commit_len)?
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
            dflash_stats_update(verify_result.num_proposed, verify_result.num_accepted);
            result.push(output);
        }
        Ok(result)
    }
}
