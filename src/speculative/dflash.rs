use crate::models::dflash::{DFlashDraftModel, DFlashModelConfig};
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;

/// DFlash2 drafts attend to a bounded projected-context window (matches reference training).
const DEFAULT_CONTEXT_WINDOW: usize = 512;

pub struct DFlashDrafter {
    pub draft_model: DFlashDraftModel,
    pub target_layer_ids: Vec<usize>,
    pub num_speculative_tokens: usize,
    pub mask_token_id: u32,
    context_window: usize,
    device: Device,
    _dtype: DType,
    cached_target_hidden: Mutex<HashMap<usize, Tensor>>,
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
        if !draft_config.is_dflash2() {
            candle_core::bail!(
                "Only DFlash2 draft models are supported (architecture DFlash2* or dflash_config.selector_top_k). \
                 For Qwen3.5 built-in speculative decoding, use --num-speculative-tokens without --draft-model."
            );
        }

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
        // block_size = [anchor] + N mask slots; user-facing count is N.
        let block_size =
            num_speculative_tokens.unwrap_or_else(|| draft_config.block_size().saturating_sub(1));
        let mask_token_id = draft_config.mask_token_id().unwrap_or(0);
        let context_window = std::cmp::min(
            DEFAULT_CONTEXT_WINDOW,
            draft_config.max_position_embeddings.max(1),
        );

        crate::log_info!(
            "DFlash2 drafter initialized: {} layers, num_speculative_tokens={}, target_layer_ids={:?}, mask_token_id={}, context_window={}",
            draft_config.num_hidden_layers,
            block_size,
            target_layer_ids,
            mask_token_id,
            context_window,
        );

        Ok(Self {
            draft_model,
            target_layer_ids,
            num_speculative_tokens: block_size,
            mask_token_id,
            context_window,
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

    /// Run the DFlash2 draft block and select tokens via the top-k candidate lattice.
    pub fn draft_tokens(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        let n = self.num_speculative_tokens;
        let mut block_ids = vec![self.mask_token_id; n + 1];
        block_ids[0] = anchor_token;

        let block_tensor = Tensor::from_vec(
            block_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (n + 1,),
            &self.device,
        )?;

        let noise_embedding = embed_fn(&block_tensor)?.to_dtype(DType::BF16)?;

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
        let draft_logits = lm_head_fn(&draft_hidden)?;
        self.draft_model
            .select_candidates(&draft_hidden, &draft_logits, anchor_token)
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

    pub fn build_draft_context(&self, seq_id: usize) -> Result<Option<Tensor>> {
        let cached = self.cached_target_hidden.lock().unwrap();
        Ok(cached.get(&seq_id).cloned())
    }

    pub fn append_context(&self, seq_id: usize, projected: &Tensor) -> Result<()> {
        let rows = projected.dim(0)?;
        if rows == 0 {
            return Ok(());
        }
        let mut cached = self.cached_target_hidden.lock().unwrap();
        let updated = match cached.get(&seq_id).cloned() {
            Some(prev) => Tensor::cat(&[prev, projected.clone()], 0)?,
            None => projected.clone(),
        };
        let total = updated.dim(0)?;
        let keep = std::cmp::min(total, self.context_window);
        cached.insert(seq_id, updated.narrow(0, total - keep, keep)?);
        Ok(())
    }

    pub fn append_verified_context(
        &self,
        verify_hidden: &Tensor,
        accepted_count: usize,
        seq_id: usize,
    ) -> Result<()> {
        let vdim = verify_hidden.dim(0)?;
        if accepted_count == 0 || vdim == 0 {
            return Ok(());
        }
        let keep = std::cmp::min(accepted_count + 1, vdim);
        self.append_context(seq_id, &verify_hidden.narrow(0, 0, keep)?)
    }

    pub fn store_decode_hidden(&self, hidden: &Tensor, seq_id: usize) -> Result<()> {
        self.append_context(seq_id, hidden)
    }
}

use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;
use crate::speculative::metadata::SpecSeqInfo;
use crate::speculative::verify::{dflash_stats_summary, dflash_stats_update, verify_draft_greedy};
use crate::utils::config::EngineConfig;
use attention_rs::InputMetadata;

/// Collect target-layer hidden states for DFlash2 context / verify refresh.
fn forward_collecting_target(
    model: &Model,
    input_ids: &Tensor,
    positions: &Tensor,
    kv_pairs: Option<&Vec<(Tensor, Tensor)>>,
    input_metadata: &InputMetadata,
    target_layer_ids: &[usize],
) -> Result<(Tensor, Vec<Tensor>)> {
    model.forward_collecting_layers(
        input_ids,
        positions,
        kv_pairs,
        input_metadata,
        false,
        target_layer_ids,
    )
}

impl ModelRunner {
    /// One DFlash2 speculative decode step (anchor + draft + verify).
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
            Seqs::SeqRefs(ref refs) if refs.len() == 1 => SpecSeqInfo {
                id: refs[0].id,
                len: refs[0].len(),
                block_table: refs[0].block_table.clone(),
            },
            Seqs::DecodeVec(ref decoded) if decoded.len() == 1 => SpecSeqInfo {
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
            let result = forward_collecting_target(
                self.model(),
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                drafter.target_layer_ids(),
            )?;
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
            drafter.append_context(seq_info.id, &projected_step_hidden)?;
        }
        let target_hidden = drafter
            .build_draft_context(seq_info.id)?
            .ok_or_else(|| candle_core::Error::Msg("DFlash2 context cache is empty".into()))?;

        let embed_fn = |tokens: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.embed_forward(tokens),
                Model::Qwen3_5MoE(model) => model.embed_forward(tokens),
                Model::Qwen3VL(model) => model.embed_forward(tokens),
                _ => candle_core::bail!("DFlash2 supports Qwen3.5 family targets"),
            }
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.forward_lm_head(hidden),
                Model::Qwen3_5MoE(model) => model.forward_lm_head(hidden),
                Model::Qwen3VL(model) => model.forward_lm_head(hidden),
                _ => candle_core::bail!("DFlash2 supports Qwen3.5 family targets"),
            }
        };
        let draft_tokens =
            drafter.draft_tokens(&target_hidden, &embed_fn, &lm_head_fn, anchor_token)?;
        if draft_tokens.is_empty() {
            return Ok(vec![vec![anchor_token]]);
        }

        let mut verify_tokens = vec![anchor_token];
        verify_tokens.extend_from_slice(&draft_tokens);
        let verify_len = verify_tokens.len();
        let slot_mappings =
            self.compute_slot_mappings(&seq_info, verify_len, self.block_size(), "DFlash2 verify")?;
        let verify_input_ids = Tensor::from_vec(verify_tokens, (verify_len,), self.device())?;
        let verify_positions = Tensor::from_vec(
            (seq_info.len..seq_info.len + verify_len)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            (verify_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_verify_metadata(&seq_info, &slot_mappings, verify_len)?;
        let _verify_guard = set_linear_is_prefill(true);

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
                        "DFlash2 verify missing layer-hidden buffers after graph replay".into(),
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
            let (verify_logits, hidden_states) = forward_collecting_target(
                self.model(),
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                drafter.target_layer_ids(),
            )?;
            drop(kv_cache);
            (
                verify_logits,
                drafter.extract_and_concat_hidden(&hidden_states)?,
            )
        };

        let verify_result = verify_draft_greedy(&verify_logits, &draft_tokens)?;
        let commit_len = 1 + verify_result.num_accepted;
        if verify_result.num_accepted < verify_result.num_proposed
            && !self.rollback_mamba(seq_info.id, commit_len)?
        {
            candle_core::bail!(
                "DFlash2 failed to roll back mamba state for sequence {}",
                seq_info.id
            );
        }

        drafter.append_verified_context(
            &projected_verify_hidden,
            verify_result.num_accepted,
            seq_info.id,
        )?;

        let mut result_tokens = Vec::with_capacity(commit_len + 1);
        result_tokens.push(anchor_token);
        result_tokens.extend_from_slice(&verify_result.accepted_tokens);
        result_tokens.push(verify_result.continuation_token);
        if dflash_stats_update(verify_result.num_proposed, verify_result.num_accepted) {
            crate::log_info!("{}", dflash_stats_summary());
        }
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
                .map(|seq| SpecSeqInfo {
                    id: seq.id,
                    len: seq.len(),
                    block_table: seq.block_table.clone(),
                })
                .collect::<Vec<_>>(),
            Seqs::DecodeVec(ref decoded) => decoded
                .iter()
                .map(|seq| SpecSeqInfo {
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
            let result = forward_collecting_target(
                self.model(),
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                drafter.target_layer_ids(),
            )?;
            drop(kv_cache);
            result
        };
        let anchor_tokens = self.sample(&anchor_logits, seqs, false)?;
        let projected_step_hidden = drafter.extract_and_concat_hidden(&anchor_hidden_states)?;
        for (index, seq_info) in seq_infos.iter().enumerate() {
            drafter.append_context(seq_info.id, &projected_step_hidden.narrow(0, index, 1)?)?;
        }

        let embed_fn = |tokens: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.embed_forward(tokens),
                Model::Qwen3_5MoE(model) => model.embed_forward(tokens),
                Model::Qwen3VL(model) => model.embed_forward(tokens),
                _ => candle_core::bail!("DFlash2 supports Qwen3.5 family targets"),
            }
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(model) => model.forward_lm_head(hidden),
                Model::Qwen3_5MoE(model) => model.forward_lm_head(hidden),
                Model::Qwen3VL(model) => model.forward_lm_head(hidden),
                _ => candle_core::bail!("DFlash2 supports Qwen3.5 family targets"),
            }
        };

        let mut draft_tokens = Vec::with_capacity(seq_infos.len());
        for (seq_info, &anchor_token) in seq_infos.iter().zip(&anchor_tokens) {
            let target_hidden = drafter
                .build_draft_context(seq_info.id)?
                .ok_or_else(|| candle_core::Error::Msg("DFlash2 context cache is empty".into()))?;
            draft_tokens.push(drafter.draft_tokens(
                &target_hidden,
                &embed_fn,
                &lm_head_fn,
                anchor_token,
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
                    "DFlash2 batch verify",
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
        let verify_metadata =
            self.build_verify_metadata_batch(&seq_infos, &slot_mappings, &q_lens)?;
        let _verify_guard = set_linear_is_prefill(true);
        let (verify_logits, verify_hidden_states) = {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let result = forward_collecting_target(
                self.model(),
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                drafter.target_layer_ids(),
            )?;
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
                && !self.rollback_mamba_at(seq_info.id, commit_len, offset)?
            {
                candle_core::bail!(
                    "DFlash2 failed to roll back mamba state for sequence {}",
                    seq_info.id
                );
            }
            drafter.append_verified_context(
                &projected_verify_hidden.narrow(0, offset, verify_len)?,
                verify_result.num_accepted,
                seq_info.id,
            )?;
            let mut output = Vec::with_capacity(commit_len + 1);
            output.push(anchor_token);
            output.extend_from_slice(&verify_result.accepted_tokens);
            output.push(verify_result.continuation_token);
            if dflash_stats_update(verify_result.num_proposed, verify_result.num_accepted) {
                crate::log_info!("{}", dflash_stats_summary());
            }
            result.push(output);
        }
        Ok(result)
    }
}

pub fn init_dflash_drafter(
    econfig: &EngineConfig,
    comm: Rc<Comm>,
    device: &Device,
) -> Result<Option<DFlashDrafter>> {
    let Some(draft_model) = econfig.draft_model.as_ref() else {
        return Ok(None);
    };
    if draft_model.is_empty() {
        return Ok(None);
    }

    crate::log_info!("Loading external DFlash2 draft model...");
    let (model_id, weight_path) = crate::speculative::resolve_draft_model(draft_model);
    let loader = crate::utils::downloader::Downloader::new(model_id, weight_path, None);
    let (draft_paths, is_gguf) = loader
        .prepare_draft_model_weights(econfig.hf_token.clone(), econfig.hf_token_path.clone())?;
    if is_gguf {
        candle_core::bail!("DFlash2 draft models must use safetensors weights");
    }

    let config_data = std::fs::read(draft_paths.get_config_filename())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read DFlash2 config: {e}")))?;
    let draft_config: DFlashModelConfig = serde_json::from_slice(&config_data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse DFlash2 config: {e}")))?;
    let drafter = DFlashDrafter::new(
        &draft_config,
        &draft_paths.get_weight_filenames(),
        comm,
        DType::BF16,
        device,
        econfig.num_speculative_tokens,
    )?;
    crate::log_info!("External DFlash2 draft model loaded successfully");
    Ok(Some(drafter))
}
