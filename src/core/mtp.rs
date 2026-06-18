// src/core/mtp.rs
// Multi-Token Prediction (MTP) speculative decoding support.
//
// MTP uses lightweight prediction heads built into the model (e.g. Qwen3.5, DeepSeek-V3)
// to draft future tokens using the backbone's hidden states and KV cache.
// Accepted draft tokens are verified in a single target-model forward pass.
//
// The speculative decode pipeline (step1 anchor, step2 draft, step3 verify) lives here,
// keeping runner.rs and engine.rs focused on the standard inference path.

use candle_core::{Result, Tensor, D};
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Verification & stats (pure functions, no model dependencies)
// ---------------------------------------------------------------------------

/// Outcome of MTP verification for a single sequence.
#[derive(Debug, Clone)]
pub struct MtpVerifyResult {
    /// All accepted tokens (draft tokens that matched the target model).
    pub accepted_tokens: Vec<u32>,
    /// The continuation token sampled from the first rejection point.
    pub continuation_token: u32,
    /// How many of the proposed drafts were accepted.
    pub num_accepted: usize,
    /// Total number proposed.
    pub num_proposed: usize,
}

/// Verify draft tokens against target model logits (greedy / argmax).
///
/// Uses a single batched argmax over all rows + vectorized comparison on GPU,
/// then transfers results to CPU in one shot.
///
/// `verify_logits`: shape [N+1, vocab_size] where N = len(draft_tokens).
///   - Position 0 predicts draft_tokens[0]
///   - Position i predicts draft_tokens[i] (for i < N)
///   - Position N provides the continuation token after last accepted draft
pub fn verify_draft_greedy(
    verify_logits: &Tensor,
    draft_tokens: &[u32],
) -> Result<MtpVerifyResult> {
    let num_positions = verify_logits.dim(0)?;
    let num_proposed = draft_tokens.len();

    if num_positions == 0 || num_proposed == 0 {
        let first_token = if num_positions > 0 {
            verify_logits
                .get(0)?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?
        } else {
            0
        };
        return Ok(MtpVerifyResult {
            accepted_tokens: vec![],
            continuation_token: first_token,
            num_accepted: 0,
            num_proposed,
        });
    }

    // Keep verifier argmax aligned with the normal sampler path, which promotes
    // logits to F32 before selecting tokens.
    let verify_logits = verify_logits.to_dtype(candle_core::DType::F32)?;
    let all_target_tokens = verify_logits.argmax(D::Minus1)?;
    let target_vec: Vec<u32> = all_target_tokens.to_vec1()?;

    let compare_len = num_proposed.min(num_positions);
    let mut num_accepted = 0;
    for i in 0..compare_len {
        if target_vec[i] == draft_tokens[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    let accepted_tokens = draft_tokens[..num_accepted].to_vec();
    let continuation_token = if num_accepted < num_positions {
        target_vec[num_accepted]
    } else {
        target_vec[num_positions - 1]
    };

    Ok(MtpVerifyResult {
        accepted_tokens,
        continuation_token,
        num_accepted,
        num_proposed,
    })
}

/// Global MTP statistics tracker.
pub static MTP_TOTAL_PROPOSED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_ACCEPTED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_STEPS: AtomicUsize = AtomicUsize::new(0);

pub fn mtp_stats_update(proposed: usize, accepted: usize) {
    MTP_TOTAL_PROPOSED.fetch_add(proposed, Ordering::Relaxed);
    MTP_TOTAL_ACCEPTED.fetch_add(accepted, Ordering::Relaxed);
    MTP_TOTAL_STEPS.fetch_add(1, Ordering::Relaxed);
}

pub fn mtp_stats_acceptance_rate() -> f64 {
    let proposed = MTP_TOTAL_PROPOSED.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    }
}

pub fn mtp_stats_avg_tokens_per_step() -> f64 {
    let steps = MTP_TOTAL_STEPS.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    if steps == 0 {
        1.0
    } else {
        // Each step produces: 1 anchor + accepted drafts + 1 continuation
        (accepted + 2 * steps) as f64 / steps as f64
    }
}

pub fn mtp_stats_summary() -> String {
    let proposed = MTP_TOTAL_PROPOSED.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    let steps = MTP_TOTAL_STEPS.load(Ordering::Relaxed);
    format!(
        "MTP Stats: proposed={}, accepted={}, acceptance_rate={:.2}%, avg_tokens/step={:.2}",
        proposed,
        accepted,
        if proposed > 0 {
            accepted as f64 / proposed as f64 * 100.0
        } else {
            0.0
        },
        if steps > 0 {
            (accepted + 2 * steps) as f64 / steps as f64
        } else {
            1.0
        },
    )
}

pub fn mtp_stats_reset() {
    MTP_TOTAL_PROPOSED.store(0, Ordering::Relaxed);
    MTP_TOTAL_ACCEPTED.store(0, Ordering::Relaxed);
    MTP_TOTAL_STEPS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// MTP speculative decode pipeline (impl ModelRunner)
// ---------------------------------------------------------------------------

use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;
use attention_rs::InputMetadata;

pub(crate) struct MtpSeqInfo {
    pub id: usize,
    pub len: usize,
    pub block_table: Vec<u32>,
}

impl ModelRunner {
    pub(crate) fn compute_slot_mappings(
        &self,
        seq_info: &MtpSeqInfo,
        num_tokens: usize,
        block_size: usize,
        ctx: &str,
    ) -> Result<Vec<i64>> {
        let mut slots = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let pos = seq_info.len + i;
            let block_idx = pos / block_size;
            let block_offset = pos % block_size;
            if block_idx < seq_info.block_table.len() {
                let physical_block = seq_info.block_table[block_idx] as i64;
                slots.push(physical_block * block_size as i64 + block_offset as i64);
            } else {
                candle_core::bail!(
                    "MTP {} missing KV block: block_idx {} >= block_table.len() {}. \
                     Blocks must be pre-allocated before MTP.",
                    ctx,
                    block_idx,
                    seq_info.block_table.len()
                );
            }
        }
        Ok(slots)
    }

    pub(crate) fn build_mtp_metadata(
        &self,
        seq_info: &MtpSeqInfo,
        slot_mappings: &[i64],
        q_len: usize,
    ) -> Result<InputMetadata> {
        let total_kv_len = (seq_info.len + q_len) as u32;
        let mamba_slot_mapping = self.prepare_mamba_slot_mapping(&[seq_info.id], false)?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if let Some(params) = self.flashinfer_kv_params() {
            let num_pages = (total_kv_len as usize).div_ceil(params.page_size);
            if num_pages > seq_info.block_table.len() {
                candle_core::bail!(
                    "MTP verify needs {} KV pages for {} tokens, but only {} pages are allocated",
                    num_pages,
                    total_kv_len,
                    seq_info.block_table.len()
                );
            }
            let indptr_host = vec![0u32, num_pages as u32];
            let indices_vec = seq_info.block_table[..num_pages].to_vec();
            let last_page_tokens = if total_kv_len == 0 {
                0
            } else {
                (total_kv_len as usize - 1) % params.page_size + 1
            };
            let last_len_host = vec![last_page_tokens as u32];
            let kv_len_arr_host = vec![total_kv_len];
            let q_cu_seqlens_host = vec![0u32, q_len as u32];
            let batch_indices = Tensor::zeros((q_len,), candle_core::DType::U32, self.device())?;
            let append_positions = Tensor::from_vec(
                (seq_info.len as u32..total_kv_len).collect::<Vec<_>>(),
                (q_len,),
                self.device(),
            )?;

            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_graph = self
                .mtp_capturer
                .as_ref()
                .map_or(false, |c| c.is_mtp_captured(q_len));
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_graph = false;

            let prefill_plan_info = if use_graph {
                None
            } else {
                Some(attention_rs::flashinfer::prefill_plan(
                    self.device(),
                    &q_cu_seqlens_host,
                    &indptr_host,
                    &kv_len_arr_host,
                    q_len as u32,
                    1,
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                    false,
                )?)
            };

            Some(attention_rs::FlashInferMetadata {
                indptr: Tensor::from_vec(indptr_host.clone(), (2,), self.device())?,
                indptr_host,
                indices: Tensor::from_vec(indices_vec, (num_pages,), self.device())?,
                last_len: Tensor::from_vec(last_len_host.clone(), (1,), self.device())?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: Some(q_len as u32),
                // FlashInfer's multi-token append path is selected only when both
                // tensors are present. Without them it falls back to decode append,
                // which writes one KV row per sequence instead of all verify rows.
                batch_indices: Some(batch_indices),
                positions: Some(append_positions),
                use_cuda_graph: use_graph,
                decode_plan_info: None,
                prefill_plan_info,
                mla_decode_plan_info: None,
                mla_prefill_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        Ok(InputMetadata {
            is_prefill: true,
            is_mla: self.is_mla_model(),
            sequence_ids: Some(vec![seq_info.id]),
            mamba_slot_mapping,
            slot_mapping: Tensor::from_vec(slot_mappings.to_vec(), (q_len,), self.device())?,
            context_lens: Some(Tensor::from_vec(vec![total_kv_len], (1,), self.device())?),
            block_tables: Some(Tensor::from_vec(
                seq_info.block_table.clone(),
                (1, seq_info.block_table.len()),
                self.device(),
            )?),
            seqlens: None,
            cu_seqlens_q: Some(Tensor::from_vec(
                vec![0u32, q_len as u32],
                (2,),
                self.device(),
            )?),
            cu_seqlens_k: Some(Tensor::from_vec(
                vec![0u32, total_kv_len],
                (2,),
                self.device(),
            )?),
            max_seqlen_q: q_len,
            max_seqlen_k: seq_info.len + q_len,
            max_context_len: seq_info.len + q_len,
            flashinfer_metadata,
            is_mtp_verify: true,
        })
    }

    fn mtp_rollback_mamba(&self, seq_id: usize, keep_tokens: usize) -> Result<bool> {
        match self.model() {
            Model::Qwen3_5(m) => m.mtp_rollback_mamba(seq_id, keep_tokens),
            Model::Qwen3_5MoE(m) => m.mtp_rollback_mamba(seq_id, keep_tokens),
            Model::Qwen3VL(m) => m.mtp_rollback_mamba(seq_id, keep_tokens),
            _ => Ok(false),
        }
    }

    /// MTP Step 1: single-token decode to get anchor token + hidden state.
    /// Tries CUDA graph replay first (the graph's internal buffer for the
    /// post-norm hidden state is accessible via take_last_hidden_for_mtp),
    /// falling back to eager forward_with_hidden.
    fn mtp_decode_step1(&self, seqs: Seqs, _seq_info: &MtpSeqInfo) -> Result<(u32, Tensor)> {
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
        let (logits, hidden_states) = match self.model() {
            Model::Qwen3_5(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                Some(&kv_cache),
                &input_metadata,
                false,
            )?,
            Model::Qwen3_5MoE(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                Some(&kv_cache),
                &input_metadata,
                false,
            )?,
            Model::Qwen3VL(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                Some(&kv_cache),
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
                let infos: Vec<MtpSeqInfo> = s
                    .iter()
                    .map(|seq| MtpSeqInfo {
                        id: seq.id,
                        len: seq.len(),
                        block_table: seq.block_table.clone(),
                    })
                    .collect();
                (s.len(), infos)
            }
            Seqs::DecodeVec(d) => {
                let infos: Vec<MtpSeqInfo> = d
                    .iter()
                    .map(|ds| MtpSeqInfo {
                        id: ds.id,
                        len: ds.len,
                        block_table: ds.block_tables.clone(),
                    })
                    .collect();
                (d.len(), infos)
            }
        };

        if batch_size != 1 {
            let output = self.run(seqs, false)?;
            return Ok(output.into_iter().map(|t| vec![t]).collect());
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
            self.build_mtp_metadata(seq_info, &slot_mappings[..verify_len], verify_len)?;

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
                self.mtp_capturer.as_ref().unwrap().replay_mtp(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    &verify_metadata,
                )
            }
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            {
                unreachable!()
            }
        } else {
            let kv_cache = self.get_kv_cache();
            let res = match self.model() {
                Model::Qwen3_5(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    Some(&kv_cache),
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3_5MoE(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    Some(&kv_cache),
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3VL(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    Some(&kv_cache),
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
            let restored = self.mtp_rollback_mamba(seq_info.id, commit_len)?;
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
}
