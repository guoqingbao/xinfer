// DFlash speculative decoding pipeline (sibling of MTP).
//
// DFlash drafts future tokens with a *separate* small draft model that reads the target model's
// projected hidden states, then verifies the whole draft block in ONE prefill-style target forward.
// It reuses MTP's metadata / verify / rollback helpers so it stays correct against the current
// attention-rs API and hybrid (Mamba/GDN) state handling.

use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::core::dflash_drafter::DFlashDrafter;
use crate::core::mtp::{verify_draft_greedy, MtpSeqInfo};
use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;

impl ModelRunner {
    /// DFlash speculative decode for a batch of sequences.
    ///
    /// Returns `Vec<Vec<u32>>` where each inner vec is `[anchor, accepted..., continuation]`
    /// (consumed by `finish_step` -> `scheduler.postprocess`, exactly like MTP).
    ///
    /// Falls back to a plain decode (`run`) when the drafter is absent or the batch size > 1
    /// (DFlash is single-sequence only).
    pub fn run_dflash_decode(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        let drafter: Arc<DFlashDrafter> = match &self.dflash_drafter {
            Some(d) => d.clone(),
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
        let seq_id = seq_info.id;
        let target_layer_ids = drafter.target_layer_ids();

        // Target-model embedding + lm_head accessors (draft reuses the target's tables).
        let embed_fn = |ids: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3(m) => m.embed_forward(ids),
                Model::Qwen3MoE(m) => m.embed_forward(ids),
                Model::Qwen3_5(m) => m.embed_forward(ids),
                Model::Qwen3_5MoE(m) => m.embed_forward(ids),
                Model::Qwen3VL(m) => m.embed_forward(ids),
                _ => candle_core::bail!("DFlash not supported for this model type"),
            }
        };
        let lm_head_fn = |h: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3(m) => m.forward_lm_head(h),
                Model::Qwen3MoE(m) => m.forward_lm_head(h),
                Model::Qwen3_5(m) => m.forward_lm_head(h),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(h),
                Model::Qwen3VL(m) => m.forward_lm_head(h),
                _ => candle_core::bail!("DFlash lm_head not accessible"),
            }
        };

        // ---- Step 1: anchor decode + update the projected-hidden context window. ----
        let (input_ids, positions, mut input_metadata) = match &seqs {
            Seqs::SeqRefs(seqs_ref) => self.prepare_decode(*seqs_ref)?,
            Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
        };
        let _decode_guard = set_linear_is_prefill(false);
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
            if input_metadata.is_mla {
                if fm.mla_decode_plan_info.is_none() {
                    if let Some(params) = self.flashinfer_kv_params() {
                        fm.mla_decode_plan_info =
                            Some(attention_rs::mla::mla_decode_plan(
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
        let (logits, hidden_collector) = match self.model() {
            Model::Qwen3(m) => m.forward_with_hidden_states(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3MoE(m) => m.forward_with_hidden_states(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3_5(m) => m.forward_with_hidden_states(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3_5MoE(m) => m.forward_with_hidden_states(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3VL(m) => m.forward_with_hidden_states(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
                target_layer_ids,
            )?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("DFlash requires a supported model type");
            }
        };
        drop(kv_cache);
        drop(_decode_guard);

        let anchor_token = self.sample(&logits, seqs, false)?[0];
        let step1_proj = drafter.extract_and_project_hidden(&hidden_collector)?;
        drafter.append_context(seq_id, &step1_proj)?;

        // ---- Step 2: draft N tokens with the DFlash model. ----
        let ctx = match drafter.context(seq_id)? {
            Some(c) => c,
            None => return Ok(vec![vec![anchor_token]]),
        };
        let drafts = drafter.draft_tokens(&ctx, &embed_fn, &lm_head_fn, &[anchor_token])?;
        if drafts.is_empty() {
            return Ok(vec![vec![anchor_token]]);
        }

        // Guard: the verify block must fit in the pre-allocated KV blocks.
        let block_size = self.block_size();
        let q_len = drafts.len() + 1; // [anchor, d0..d_{N-1}]
        let needed_pages = (seq_info.len + q_len).div_ceil(block_size);
        if needed_pages > seq_info.block_table.len() {
            return Ok(vec![vec![anchor_token]]);
        }

        // ---- Step 3: verify the whole block in ONE prefill-style target forward. ----
        let verify_tokens: Vec<u32> = std::iter::once(anchor_token).chain(drafts.iter().copied()).collect();
        let slot_mappings = self.compute_slot_mappings(seq_info, q_len, block_size, "dflash")?;
        let verify_ids = Tensor::from_vec(verify_tokens.clone(), (q_len,), self.device())?;
        let verify_positions = Tensor::from_vec(
            (0..q_len).map(|i| (seq_info.len + i) as i64).collect::<Vec<_>>(),
            (q_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata(seq_info, &slot_mappings[..q_len], q_len)?;

        let _prefill_guard = set_linear_is_prefill(true);
        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (vlogits, vhidden) = match self.model() {
            Model::Qwen3(m) => m.forward_with_hidden_states(
                &verify_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3MoE(m) => m.forward_with_hidden_states(
                &verify_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3_5(m) => m.forward_with_hidden_states(
                &verify_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3_5MoE(m) => m.forward_with_hidden_states(
                &verify_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
                target_layer_ids,
            )?,
            Model::Qwen3VL(m) => m.forward_with_hidden_states(
                &verify_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
                target_layer_ids,
            )?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("DFlash requires a supported model type");
            }
        };
        drop(kv_cache);
        drop(_prefill_guard);

        // ---- Accept / reject (reuses MTP's greedy verifier). ----
        let res = verify_draft_greedy(&vlogits, &drafts)?;

        // Update the context window with the verify block's accepted rows
        // (row 0 = anchor, rows 1..=num_accepted = accepted drafts).
        if !vhidden.is_empty() && res.num_accepted > 0 {
            let vproj = drafter.extract_and_project_hidden(&vhidden)?;
            let keep = std::cmp::min(res.num_accepted + 1, vproj.dim(0)?);
            if keep > 0 {
                drafter.append_context(seq_id, &vproj.narrow(0, 0, keep)?)?;
            }
        }

        // Hybrid (Mamba/GDN) models mutate recurrent state in-place; roll back to the accepted
        // boundary on partial rejection. Full-attention models return false (no-op).
        if res.num_accepted < res.num_proposed {
            let keep_tokens = 1 + res.num_accepted;
            self.mtp_rollback_mamba(seq_id, keep_tokens)?;
        }

        let mut result_tokens = Vec::with_capacity(2 + res.num_accepted);
        result_tokens.push(anchor_token);
        result_tokens.extend_from_slice(&res.accepted_tokens);
        result_tokens.push(res.continuation_token);

        Ok(vec![result_tokens])
    }
}