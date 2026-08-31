use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;
use crate::models::qwen3_5_mtp::Qwen3_5MtpHead;
use crate::speculative::metadata::SpecSeqInfo;
use crate::speculative::verify::{
    mtp_stats_summary, mtp_stats_update, verify_draft_greedy, MTP_TOTAL_STEPS,
};
use candle_core::{Result, Tensor};
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl ModelRunner {
    /// MTP Step 1: single-token decode to get anchor token + hidden state.
    /// Tries CUDA graph replay first (the graph's internal buffer for the
    /// post-norm hidden state is accessible via take_last_hidden_for_mtp),
    /// falling back to eager forward_with_hidden.
    #[allow(unused)]
    fn mtp_decode_step1(&self, seqs: Seqs, _seq_info: &SpecSeqInfo) -> Result<(u32, Tensor)> {
        let (input_ids, positions, mut input_metadata) = match &seqs {
            Seqs::SeqRefs(seqs_ref) => self.prepare_decode(*seqs_ref)?,
            Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
        };

        let _decode_guard = set_linear_is_prefill(false);

        // Try CUDA graph replay for the decode forward. The model's forward()
        // stores hidden states in last_hidden_for_mtp during both capture and
        // replay (the cached tensor shares GPU storage with the graph output,
        // so it's updated in-place on replay).
        #[cfg(all(feature = "cuda", feature = "graph"))]
        {
            let input_batch = input_ids.dim(0)?;
            let require_exact_graph = input_metadata.mamba_slot_mapping.is_some();
            let can_replay = if require_exact_graph {
                self.decode_capturer.is_exact_captured(input_batch)
            } else {
                self.decode_capturer.is_captured(input_batch)
            };
            if can_replay {
                let logits = match self.model() {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.decode_capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.decode_capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.decode_capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        } else {
                            self.decode_capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        }
                    }
                    _ => self
                        .decode_capturer
                        .replay(&input_ids, &positions, &input_metadata)?,
                };

                let hidden_states = match self.model() {
                    Model::Qwen3_5(model) => model.take_last_hidden_for_mtp(),
                    Model::Qwen3_5MoE(model) => model.take_last_hidden_for_mtp(),
                    Model::Qwen3VL(model) => model.take_last_hidden_for_mtp(),
                    _ => None,
                };

                if let Some(hidden_states) = hidden_states {
                    let anchor_token = self.sample(&logits, seqs, false)?[0];
                    let seq_hidden = if hidden_states.dims().len() == 2 && hidden_states.dim(0)? > 1
                    {
                        hidden_states.get(hidden_states.dim(0)? - 1)?
                    } else if hidden_states.dims().len() == 2 {
                        hidden_states.get(0)?
                    } else {
                        hidden_states
                    };
                    return Ok((anchor_token, seq_hidden));
                }
            }
        }

        // Fallback: eager forward_with_hidden (no graph available or hidden state extraction failed)
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
            if input_metadata.is_mla {
                if fm.mla_decode_plan_info.is_none() {
                    if let Some(params) = self.flashinfer_kv_params() {
                        fm.mla_decode_plan_info = Some(attention_rs::mla::mla_decode_plan(
                            self.device(),
                            params.kv_dtype,
                            &fm.indptr_host,
                            input_ids.dim(0)?,
                            params.num_qo_heads,
                            params.page_size,
                            fm.use_cuda_graph,
                        )?);
                    }
                }
            } else if fm.decode_plan_info.is_none() {
                if let Some(params) = self.flashinfer_kv_params() {
                    fm.decode_plan_info = Some(attention_rs::flashinfer::decode_plan(
                        self.device(),
                        params.kv_dtype,
                        params.out_dtype,
                        &fm.indptr_host,
                        fm.last_len_host.as_deref(),
                        fm.kv_len_arr_host.as_deref(),
                        input_ids.dim(0)?,
                        params.num_qo_heads,
                        params.num_kv_heads,
                        params.head_dim,
                        params.page_size,
                        fm.use_cuda_graph,
                    )?);
                }
            }
        }

        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (logits, hidden_states) = match self.model() {
            Model::Qwen3_5(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            Model::Qwen3_5MoE(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            Model::Qwen3VL(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("MTP Step 1 requires Qwen3.5 model");
            }
        };
        drop(kv_cache);

        let anchor_token = self.sample(&logits, seqs, false)?[0];

        let seq_hidden = if hidden_states.dims().len() == 2 && hidden_states.dim(0)? > 1 {
            hidden_states.get(hidden_states.dim(0)? - 1)?
        } else if hidden_states.dims().len() == 2 {
            hidden_states.get(0)?
        } else {
            hidden_states.clone()
        };

        Ok((anchor_token, seq_hidden))
    }

    /// Run MTP speculative decode for a batch of sequences.
    /// Returns Vec<Vec<u32>> where each inner vec contains all accepted tokens for that sequence
    /// (anchor + accepted drafts + bonus token).
    ///
    /// Optimized flow:
    ///   1. Run main model decode via CUDA graph replay (when available) + extract hidden state
    ///   2. Sample anchor token from logits
    ///   3. MTP head drafts K tokens autoregressively (no KV cache)
    ///   4. Verify: run main model on [anchor, draft_0, ..., draft_{K-1}] using native flash
    ///   5. On partial rejection: roll back GDN state to the accepted token boundary
    ///   6. Greedy-accept matching prefix; take bonus token at first mismatch
    pub fn run_mtp_decode(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        let mtp_head = match &self.mtp_head {
            Some(h) => h.clone(),
            None => {
                let output = self.run(seqs, false)?;
                return Ok(output.into_iter().map(|t| vec![t]).collect());
            }
        };

        let (batch_size, seq_infos) = match &seqs {
            Seqs::SeqRefs(s) => {
                let infos: Vec<SpecSeqInfo> = s
                    .iter()
                    .map(|seq| SpecSeqInfo {
                        id: seq.id,
                        len: seq.len(),
                        block_table: seq.block_table.clone(),
                    })
                    .collect();
                (s.len(), infos)
            }
            Seqs::DecodeVec(d) => {
                let infos: Vec<SpecSeqInfo> = d
                    .iter()
                    .map(|ds| SpecSeqInfo {
                        id: ds.id,
                        len: ds.len,
                        block_table: ds.block_tables.clone(),
                    })
                    .collect();
                (d.len(), infos)
            }
        };

        if batch_size != 1 {
            return self.run_mtp_decode_batch(seqs, &seq_infos, mtp_head);
        }

        let seq_info = &seq_infos[0];
        let num_draft = self.mtp_num_speculative;

        // Step 1: Main model decode for logits + hidden state.
        let (anchor_token, seq_hidden) = self.mtp_decode_step1(seqs, seq_info)?;

        // Step 2: Draft K tokens using MTP head (GPU-resident, no per-step CPU sync)
        let embed_weight = match self.model() {
            Model::Qwen3_5(m) => m.embed_weight().clone(),
            Model::Qwen3_5MoE(m) => m.embed_weight().clone(),
            Model::Qwen3VL(m) => m
                .embed_weight()
                .expect("Qwen3VL MTP requires Qwen3.5 text backbone")
                .clone(),
            _ => unreachable!(),
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(hidden),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(hidden),
                Model::Qwen3VL(m) => m.forward_lm_head(hidden),
                _ => unreachable!(),
            }
        };

        let base_position = seq_info.len.saturating_sub(1);
        let anchor_token_tensor = Tensor::from_vec(vec![anchor_token], (1,), self.device())?;
        let (draft_tokens, _last_hidden) = mtp_head.draft_tokens_gpu(
            &seq_hidden,
            &anchor_token_tensor,
            num_draft,
            &embed_weight,
            lm_head_fn,
            base_position,
        )?;

        if draft_tokens.is_empty() {
            return Ok(vec![vec![anchor_token]]);
        }

        // Step 3: Verify draft tokens via prefill-style forward on [anchor, draft_0..K-1].
        let mut verify_tokens = vec![anchor_token];
        verify_tokens.extend_from_slice(&draft_tokens);
        let verify_len = verify_tokens.len();

        let block_size = self.block_size();
        let slot_mappings =
            self.compute_slot_mappings(seq_info, verify_len, block_size, "verify")?;

        let verify_input_ids = Tensor::from_vec(verify_tokens, (verify_len,), self.device())?;
        let verify_positions_tensor = Tensor::from_vec(
            (0..verify_len)
                .map(|i| (seq_info.len + i) as i64)
                .collect::<Vec<_>>(),
            (verify_len,),
            self.device(),
        )?;

        let verify_metadata =
            self.build_verify_metadata(seq_info, &slot_mappings[..verify_len], verify_len)?;

        let _prefill_guard = set_linear_is_prefill(true);

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let use_mtp_graph = self
            .mtp_capturer
            .as_ref()
            .map_or(false, |c| c.is_mtp_captured(verify_len));
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let use_mtp_graph = false;

        let all_logits_result = if use_mtp_graph {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            {
                // Hold mamba cache across verify-graph replay (same as decode graphs).
                match self.model() {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions_tensor,
                            &verify_metadata,
                        )
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions_tensor,
                            &verify_metadata,
                        )
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions_tensor,
                                &verify_metadata,
                            )
                        } else {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions_tensor,
                                &verify_metadata,
                            )
                        }
                    }
                    _ => self.mtp_capturer.as_ref().unwrap().replay_mtp(
                        &verify_input_ids,
                        &verify_positions_tensor,
                        &verify_metadata,
                    ),
                }
            }
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            {
                unreachable!()
            }
        } else {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let res = match self.model() {
                Model::Qwen3_5(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3_5MoE(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3VL(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    None,
                ),
                _ => unreachable!(),
            };
            drop(kv_cache);
            res
        };
        let all_logits = all_logits_result?;

        let verify_result = verify_draft_greedy(&all_logits, &draft_tokens)?;

        if verify_result.num_accepted < verify_result.num_proposed {
            let commit_len = 1 + verify_result.num_accepted;
            // Full-attention KV cache does not need explicit rollback: the next cycle's
            // verify will overwrite rejected positions via append_kv_cache before the
            // attention kernel reads them, and FlashInfer uses kCausal masking.
            // GDN/Mamba state, however, is mutated in-place and must be rolled back.
            let restored = self.rollback_mamba(seq_info.id, commit_len)?;
            if !restored {
                candle_core::bail!(
                    "MTP failed to roll back mamba-state snapshot for seq {} to {} verified token(s)",
                    seq_info.id,
                    commit_len
                );
            }
        }

        let mut result_tokens = Vec::with_capacity(2 + verify_result.num_accepted);
        result_tokens.push(anchor_token);
        result_tokens.extend_from_slice(&verify_result.accepted_tokens);
        result_tokens.push(verify_result.continuation_token);

        mtp_stats_update(verify_result.num_proposed, verify_result.num_accepted);
        if MTP_TOTAL_STEPS.load(Ordering::Relaxed) % 256 == 0 {
            crate::log_info!("{}", mtp_stats_summary());
        }

        Ok(vec![result_tokens])
    }

    fn run_mtp_decode_batch(
        &self,
        seqs: Seqs,
        seq_infos: &[SpecSeqInfo],
        mtp_head: Arc<Qwen3_5MtpHead>,
    ) -> Result<Vec<Vec<u32>>> {
        let embed_weight = match self.model() {
            Model::Qwen3_5(m) => m.embed_weight().clone(),
            Model::Qwen3_5MoE(m) => m.embed_weight().clone(),
            Model::Qwen3VL(m) => m
                .embed_weight()
                .expect("Qwen3VL MTP requires Qwen3.5 text backbone")
                .clone(),
            _ => unreachable!(),
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(hidden),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(hidden),
                Model::Qwen3VL(m) => m.forward_lm_head(hidden),
                _ => unreachable!(),
            }
        };

        let mut anchors = Vec::with_capacity(seq_infos.len());
        let mut draft_tokens = Vec::with_capacity(seq_infos.len());
        for (index, seq_info) in seq_infos.iter().enumerate() {
            let (anchor, seq_hidden) = match &seqs {
                Seqs::SeqRefs(sequences) => {
                    self.mtp_decode_step1(Seqs::SeqRefs(&sequences[index..index + 1]), seq_info)?
                }
                Seqs::DecodeVec(sequences) => {
                    let single_sequence = vec![sequences[index].clone()];
                    self.mtp_decode_step1(Seqs::DecodeVec(&single_sequence), seq_info)?
                }
            };
            let anchor_tensor = Tensor::from_vec(vec![anchor], (1,), self.device())?;
            let base_position = seq_info.len.saturating_sub(1);
            let (draft, _) = mtp_head.draft_tokens_gpu(
                &seq_hidden,
                &anchor_tensor,
                self.mtp_num_speculative,
                &embed_weight,
                &lm_head_fn,
                base_position,
            )?;
            anchors.push(anchor);
            draft_tokens.push(draft);
        }

        if draft_tokens.iter().any(|draft| draft.is_empty()) {
            return Ok(anchors.into_iter().map(|anchor| vec![anchor]).collect());
        }

        let verify_len = self.mtp_num_speculative + 1;
        let mut verify_tokens = Vec::with_capacity(seq_infos.len() * verify_len);
        let mut slot_mappings = Vec::with_capacity(seq_infos.len());
        for (seq_info, (anchor, draft)) in seq_infos.iter().zip(anchors.iter().zip(&draft_tokens)) {
            verify_tokens.push(*anchor);
            verify_tokens.extend_from_slice(draft);
            slot_mappings.push(self.compute_slot_mappings(
                seq_info,
                verify_len,
                self.block_size(),
                "MTP batch verify",
            )?);
        }
        let q_lens = vec![verify_len; seq_infos.len()];
        let verify_metadata =
            self.build_verify_metadata_batch(seq_infos, &slot_mappings, &q_lens)?;
        let verify_input_ids = Tensor::from_vec(
            verify_tokens,
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let verify_positions = Tensor::from_vec(
            seq_infos
                .iter()
                .flat_map(|seq| seq.len..seq.len + verify_len)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;

        let _prefill_guard = set_linear_is_prefill(true);
        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let all_logits = match self.model() {
            Model::Qwen3_5(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
            ),
            Model::Qwen3_5MoE(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
            ),
            Model::Qwen3VL(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                None,
            ),
            _ => unreachable!(),
        }?;
        drop(kv_cache);

        let mut outputs = Vec::with_capacity(seq_infos.len());
        for (index, (seq_info, draft)) in seq_infos.iter().zip(&draft_tokens).enumerate() {
            let offset = index * verify_len;
            let logits = all_logits.narrow(0, offset, verify_len)?;
            let verify_result = verify_draft_greedy(&logits, draft)?;
            if verify_result.num_accepted < verify_result.num_proposed {
                let keep_tokens = 1 + verify_result.num_accepted;
                if !self.rollback_mamba_at(seq_info.id, keep_tokens, offset)? {
                    candle_core::bail!(
                        "MTP failed to roll back mamba-state for batch sequence {}",
                        seq_info.id
                    );
                }
            }
            let mut result = Vec::with_capacity(verify_result.num_accepted + 2);
            result.push(anchors[index]);
            result.extend_from_slice(&verify_result.accepted_tokens);
            result.push(verify_result.continuation_token);
            mtp_stats_update(verify_result.num_proposed, verify_result.num_accepted);
            outputs.push(result);
        }
        Ok(outputs)
    }
}
