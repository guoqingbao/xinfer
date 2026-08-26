// DFlash speculative decoding pipeline (sibling of MTP).
//
// DFlash drafts future tokens with a *separate* small draft model that reads the target model's
// projected hidden states, then verifies the whole draft block in ONE prefill-style target forward.
// The mechanism-specific propose (anchor decode + context + draft) lives here; the shared
// verify/accept/rollback/emit/stats core lives in `speculative.rs`.

use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::core::dflash_drafter::DFlashDrafter;
use crate::core::mtp::MtpSeqInfo;
use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::core::speculative::{Drafter, Proposal};
use crate::models::layers::linear::set_linear_is_prefill;

/// Wraps the DFlash drafter (model + context window) as a `Drafter`: `propose` runs the anchor
/// decode + context update + draft (steps 1-2); `on_verified` refreshes the context window from
/// the verify block's hidden states (step 3).
pub struct DflashDrafter {
    inner: Arc<DFlashDrafter>,
}

impl Drafter for DflashDrafter {
    fn name(&self) -> &'static str {
        if self.inner.draft_model.config.has_v2_components() {
            "dflash2"
        } else {
            "dflash1"
        }
    }

    fn verify_target_layers(&self) -> &[usize] {
        self.inner.target_layer_ids()
    }

    fn anchor(&self, runner: &ModelRunner, seqs: Seqs, seq: &MtpSeqInfo) -> Result<(u32, Option<Tensor>)> {
        let seq_id = seq.id;
        let target_layer_ids = self.inner.target_layer_ids();

        // ---- Step 1: anchor decode + update the projected-hidden context window. ----
        #[allow(unused_mut)] // `mut` only needed under the flashinfer feature
        let (input_ids, positions, mut input_metadata) = match &seqs {
            Seqs::SeqRefs(seqs_ref) => runner.prepare_decode(*seqs_ref)?,
            Seqs::DecodeVec(decode_seqs) => runner.prepare_decode(decode_seqs.iter())?,
        };
        let _decode_guard = set_linear_is_prefill(false);
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
            if input_metadata.is_mla {
                if fm.mla_decode_plan_info.is_none() {
                    if let Some(params) = runner.flashinfer_kv_params() {
                        fm.mla_decode_plan_info =
                            Some(attention_rs::mla::mla_decode_plan(
                                runner.device(),
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
                if let Some(params) = runner.flashinfer_kv_params() {
                    fm.decode_plan_info = Some(attention_rs::flashinfer::decode_plan(
                        runner.device(),
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
        let kv_cache = runner.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (logits, hidden_collector) = match runner.model() {
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

        let anchor_token = runner.sample(&logits, seqs, false)?[0];
        let step1_proj = self.inner.extract_and_project_hidden(&hidden_collector)?;
        // crate::log_info!(
        //     "[dflash-debug] anchor: seq={} anchor_tok={} step1_proj_rows={} collector_len={}",
        //     seq_id, anchor_token, step1_proj.dim(0)?, hidden_collector.len()
        // );
        self.inner.append_context(seq_id, &step1_proj)?;

        Ok((anchor_token, None))
    }

    fn draft(
        &self,
        runner: &ModelRunner,
        seq: &MtpSeqInfo,
        anchor: u32,
        _hidden: &Option<Tensor>,
    ) -> Result<Vec<u32>> {
        let seq_id = seq.id;
        // Target-model embedding + lm_head accessors (draft reuses the target's tables).
        let embed_fn = |ids: &Tensor| -> Result<Tensor> {
            match runner.model() {
                Model::Qwen3(m) => m.embed_forward(ids),
                Model::Qwen3MoE(m) => m.embed_forward(ids),
                Model::Qwen3_5(m) => m.embed_forward(ids),
                Model::Qwen3_5MoE(m) => m.embed_forward(ids),
                Model::Qwen3VL(m) => m.embed_forward(ids),
                _ => candle_core::bail!("DFlash not supported for this model type"),
            }
        };
        let lm_head_fn = |h: &Tensor| -> Result<Tensor> {
            match runner.model() {
                Model::Qwen3(m) => m.forward_lm_head(h),
                Model::Qwen3MoE(m) => m.forward_lm_head(h),
                Model::Qwen3_5(m) => m.forward_lm_head(h),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(h),
                Model::Qwen3VL(m) => m.forward_lm_head(h),
                _ => candle_core::bail!("DFlash lm_head not accessible"),
            }
        };

        // ---- Step 2: draft N tokens (block = [anchor, MASK x N]). ----
        let ctx = match self.inner.context(seq_id)? {
            Some(c) => {
                // crate::log_info!("[dflash-debug] draft: seq={} ctx_rows={}", seq_id, c.dim(0)?);
                c
            }
            None => {
                // crate::log_info!("[dflash-debug] draft: seq={} CONTEXT EMPTY -> no drafts", seq_id);
                return Ok(vec![]);
            }
        };
        let n_mask = self.inner.num_speculative();
        if n_mask == 0 {
            // crate::log_info!("[dflash-debug] draft: seq={} n_mask=0 -> no drafts", seq_id);
            return Ok(vec![]);
        }
        let (logits, hidden_n) =
            self.inner.draft_logits(&ctx, &embed_fn, &lm_head_fn, anchor, n_mask)?;

        // v2 (fused CUDA kernels): grammar gating is applied *inside* the candidate-walk
        // kernel via a per-position allow matrix. Static repeated VOB by default; the exact
        // per-position FSM walk when XINFER_SPEC_GRANULAR_MASK is set. Unguided -> no gate.
        // [dflash-debug] commented out
        // crate::log_info!(
        //     "[dflash-debug] draft: seq={} n_mask={} logits={}x{} uses_kernels={} is_guided={} granular={}",
        //     seq_id, n_mask, logits.dim(0)?, logits.dim(1)?, self.inner.uses_kernels(),
        //     runner.guided_decoding.is_guided(seq_id), crate::utils::env::spec_granular_mask()
        // );
        if self.inner.uses_kernels() {
            let vocab = logits.dim(1)?;
            let allow = if runner.guided_decoding.is_guided(seq_id) {
                if crate::utils::env::spec_granular_mask() {
                    runner.guided_decoding.draft_allow_walk(seq_id, &logits, vocab)?
                } else {
                    runner
                        .guided_decoding
                        .draft_allow_repeated(seq_id, n_mask, vocab, logits.device())?
                }
            } else {
                None
            };
            // [dflash-debug] commented out
            // let allow_shape = match &allow {
            //     Some(a) => format!("Some({}x{})", a.dim(0)?, a.dim(1)?),
            //     None => "None".to_string(),
            // };
            // crate::log_info!("[dflash-debug] v2-path: allow={}", allow_shape);
            let drafts = self
                .inner
                .select_tokens_masked(&logits, &hidden_n, anchor, allow.as_ref())?;
            // crate::log_info!("[dflash-debug] v2-path drafts={} tokens={:?}", drafts.len(), &drafts);
            return Ok(drafts);
        }

        // v1 (portable candle) path.
        // Grammar-aware drafting: batched single-VOB mask (3a) by default; the granular
        // per-position FSM walk when XINFER_SPEC_GRANULAR_MASK is set.
        if runner.guided_decoding.is_guided(seq_id) {
            if crate::utils::env::spec_granular_mask() {
                let d = runner.guided_decoding.masked_drafts(seq_id, &logits)?;
                // crate::log_info!("[dflash-debug] v1-path guided granular: drafts={} tokens={:?}", d.len(), &d);
                return Ok(d);
            }
            let masked = runner.guided_decoding.mask_rows(seq_id, &logits)?;
            let d = masked
                .to_dtype(candle_core::DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .to_vec1::<u32>()?;
            // crate::log_info!("[dflash-debug] v1-path guided static: drafts={} tokens={:?}", d.len(), &d);
            return Ok(d);
        }
        let d = self.inner.select_from_logits(&logits, &hidden_n, anchor)?;
        // crate::log_info!("[dflash-debug] v1-path unguided: drafts={} tokens={:?}", d.len(), &d);
        Ok(d)
    }

    fn on_verified(
        &self,
        _runner: &ModelRunner,
        seq: &MtpSeqInfo,
        _proposal: &Proposal,
        vhidden: &[Tensor],
        accepted: usize,
    ) -> Result<()> {
        // Refresh the context window with the verify block's accepted rows. `vhidden` is the
        // per-layer hiddens (graph path) or the stripped collector (eager path).
        if !vhidden.is_empty() && accepted > 0 {
            let vproj = self.inner.project_layer_hiddens(vhidden)?;
            let keep = std::cmp::min(accepted + 1, vproj.dim(0)?);
            if keep > 0 {
                self.inner.append_context(seq.id, &vproj.narrow(0, 0, keep)?)?;
            }
        }
        Ok(())
    }

    /// Verify forward: CUDA-graph replay when captured (hybrid models), else eager.
    ///
    /// The graph-safe per-layer hidden buffers are written by the `forward_inner` copy block
    /// DURING the replayed graph, so the order is: `replay_mtp` (writes buffers) ->
    /// `take_dflash_verify_hiddens` (reads buffers). Both DFlash v1 and v2 use the identical
    /// target-verify graph (the drafter version only affects the eager draft step), so this is
    /// variety-agnostic.
    fn verify_forward(
        &self,
        runner: &ModelRunner,
        verify_ids: &Tensor,
        verify_positions: &Tensor,
        kv_pairs: Option<&Vec<(Tensor, Tensor)>>,
        metadata: &attention_rs::InputMetadata,
        verify_len: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        #[cfg(all(feature = "cuda", feature = "graph"))]
        if matches!(
            runner.model(),
            Model::Qwen3_5(_) | Model::Qwen3_5MoE(_) | Model::Qwen3VL(_)
        ) && runner
            .mtp_capturer
            .as_ref()
            .is_some_and(|c| c.is_mtp_captured(verify_len))
        {
            let logits = runner
                .mtp_capturer
                .as_ref()
                .unwrap()
                .replay_mtp(verify_ids, verify_positions, metadata)?;
            let layer_hiddens = match runner.model() {
                Model::Qwen3_5(m) => m.take_dflash_verify_hiddens(verify_len),
                Model::Qwen3_5MoE(m) => m.take_dflash_verify_hiddens(verify_len),
                Model::Qwen3VL(m) => m.take_dflash_verify_hiddens(verify_len),
                _ => None,
            };
            if let Some(layer_hiddens) = layer_hiddens {
                return Ok((logits, layer_hiddens));
            }
            // Graph replayed but no DFlash buffers for this model: fall through to eager.
        }
        // Eager fallback: collect hiddens at the target layers, drop the embedding row.
        let (logits, collector) = runner.spec_verify_forward(
            verify_ids,
            verify_positions,
            kv_pairs,
            metadata,
            self.verify_target_layers(),
        )?;
        let layer_hiddens: Vec<Tensor> = collector.into_iter().skip(1).collect();
        Ok((logits, layer_hiddens))
    }
}

impl ModelRunner {
    /// DFlash speculative decode: route through the shared core with the DFlash drafter.
    /// Unguided batches take the batched verify path; everything else (single seq, or any
    /// guided seq) takes the single-seq grammar-firewall core.
    pub fn run_dflash_decode(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        match &self.dflash_drafter {
            Some(inner) => {
                let drafter = DflashDrafter {
                    inner: inner.clone(),
                };
                if self.dflash_batch_eligible(&seqs) {
                    return self.run_dflash_decode_batch(seqs, &drafter);
                }
                self.run_spec_decode(seqs, &drafter)
            }
            None => {
                let output = self.run(seqs, false)?;
                Ok(output.into_iter().map(|t| vec![t]).collect())
            }
        }
    }

    /// True when a batched (unguided) DFlash verify is worthwhile: more than one sequence and
    /// none of them is grammar-guided (guided sequences stay on the single-seq firewall path).
    fn dflash_batch_eligible(&self, seqs: &Seqs) -> bool {
        let ids: Vec<usize> = match seqs {
            Seqs::SeqRefs(refs) => refs.iter().map(|s| s.id).collect(),
            Seqs::DecodeVec(d) => d.iter().map(|s| s.id).collect(),
        };
        ids.len() > 1 && !ids.iter().any(|id| self.guided_decoding.is_guided(*id))
    }

    /// Batched DFlash speculative verify: one prefill-style target forward over all sequences'
    /// `[anchor, drafts...]` blocks. Unguided only. Mirrors the owner's `run_dflash_decode_batch`.
    fn run_dflash_decode_batch(
        &self,
        seqs: Seqs,
        drafter: &DflashDrafter,
    ) -> Result<Vec<Vec<u32>>> {
        let inner = &drafter.inner;
        let seq_infos: Vec<MtpSeqInfo> = match &seqs {
            Seqs::SeqRefs(refs) => refs
                .iter()
                .map(|s| MtpSeqInfo {
                    id: s.id,
                    len: s.len(),
                    block_table: s.block_table.clone(),
                })
                .collect(),
            Seqs::DecodeVec(d) => d
                .iter()
                .map(|s| MtpSeqInfo {
                    id: s.id,
                    len: s.len,
                    block_table: s.block_tables.clone(),
                })
                .collect(),
        };
        let n = inner.num_speculative();
        let verify_len = 1 + n;

        // 1. Batched anchor forward -> per-seq context update.
        #[allow(unused_mut)]
        let (input_ids, positions, mut input_metadata) = match &seqs {
            Seqs::SeqRefs(seqs_ref) => self.prepare_decode(*seqs_ref)?,
            Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
        };
        let _decode_guard = set_linear_is_prefill(false);
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
            if fm.decode_plan_info.is_none() {
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
        let (anchor_logits, anchor_collector) = match self.model() {
            Model::Qwen3_5(m) => m.forward_with_hidden_states(&input_ids, &positions, kv_pairs, &input_metadata, false, inner.target_layer_ids())?,
            Model::Qwen3_5MoE(m) => m.forward_with_hidden_states(&input_ids, &positions, kv_pairs, &input_metadata, false, inner.target_layer_ids())?,
            Model::Qwen3VL(m) => m.forward_with_hidden_states(&input_ids, &positions, kv_pairs, &input_metadata, false, inner.target_layer_ids())?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("DFlash batch requires a supported model type");
            }
        };
        drop(kv_cache);
        drop(_decode_guard);
        let anchor_tokens = self.sample(&anchor_logits, seqs, false)?;
        let projected = inner.extract_and_project_hidden(&anchor_collector)?;
        for (i, si) in seq_infos.iter().enumerate() {
            inner.append_context(si.id, &projected.narrow(0, i, 1)?)?;
        }

        // 2. Per-seq draft (block build + run + select, unguided).
        let embed_fn = |ids: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.embed_forward(ids),
                Model::Qwen3_5MoE(m) => m.embed_forward(ids),
                Model::Qwen3VL(m) => m.embed_forward(ids),
                _ => candle_core::bail!("DFlash batch not supported for this model type"),
            }
        };
        let lm_head_fn = |h: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(h),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(h),
                Model::Qwen3VL(m) => m.forward_lm_head(h),
                _ => candle_core::bail!("DFlash batch lm_head not accessible"),
            }
        };
        let mut drafts = Vec::with_capacity(seq_infos.len());
        for (si, &anchor) in seq_infos.iter().zip(&anchor_tokens) {
            let ctx = inner.context(si.id)?.ok_or_else(|| {
                candle_core::Error::Msg("DFlash target hidden cache is empty".into())
            })?;
            let (logits, hidden_n) = inner.draft_logits(&ctx, &embed_fn, &lm_head_fn, anchor, n)?;
            drafts.push(inner.select_from_logits(&logits, &hidden_n, anchor)?);
        }

        // 3. One batched verify.
        let mut verify_tokens = Vec::with_capacity(seq_infos.len() * verify_len);
        for (anchor, d) in anchor_tokens.iter().zip(&drafts) {
            verify_tokens.push(*anchor);
            verify_tokens.extend_from_slice(d);
        }
        let q_lens = vec![verify_len; seq_infos.len()];
        let slot_mappings = seq_infos
            .iter()
            .map(|si| {
                self.compute_slot_mappings(si, verify_len, self.block_size(), "DFlash batch verify")
            })
            .collect::<Result<Vec<_>>>()?;
        let verify_ids = Tensor::from_vec(
            verify_tokens,
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let verify_positions = Tensor::from_vec(
            seq_infos
                .iter()
                .flat_map(|si| si.len..si.len + verify_len)
                .map(|p| p as i64)
                .collect::<Vec<_>>(),
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata_batch(&seq_infos, &slot_mappings, &q_lens)?;
        let _verify_guard = set_linear_is_prefill(true);
        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (verify_logits, verify_collector) = match self.model() {
            Model::Qwen3_5(m) => m.forward_with_hidden_states(&verify_ids, &verify_positions, kv_pairs, &verify_metadata, false, inner.target_layer_ids())?,
            Model::Qwen3_5MoE(m) => m.forward_with_hidden_states(&verify_ids, &verify_positions, kv_pairs, &verify_metadata, false, inner.target_layer_ids())?,
            Model::Qwen3VL(m) => m.forward_with_hidden_states(&verify_ids, &verify_positions, kv_pairs, &verify_metadata, false, inner.target_layer_ids())?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("DFlash batch requires a supported model type");
            }
        };
        drop(kv_cache);
        drop(_verify_guard);
        let verify_projected = inner.extract_and_project_hidden(&verify_collector)?;

        // 4. Per-seq accept + rollback + context update + stats.
        let name = drafter.name();
        let mut result = Vec::with_capacity(seq_infos.len());
        for (i, ((si, d), &anchor)) in seq_infos.iter().zip(&drafts).zip(&anchor_tokens).enumerate() {
            let offset = i * verify_len;
            let per_seq_logits = verify_logits.narrow(0, offset, verify_len)?;
            let res = crate::core::mtp::verify_draft_greedy(&per_seq_logits, d)?;
            let commit_len = 1 + res.num_accepted;
            if res.num_accepted < res.num_proposed {
                self.mtp_rollback_mamba(si.id, commit_len)?;
            }
            let vproj_row = verify_projected.narrow(0, offset, verify_len)?;
            let keep = std::cmp::min(res.num_accepted + 1, vproj_row.dim(0)?);
            if keep > 0 {
                inner.append_context(si.id, &vproj_row.narrow(0, 0, keep)?)?;
            }
            let mut row = Vec::with_capacity(commit_len + 1);
            row.push(anchor);
            row.extend_from_slice(&res.accepted_tokens);
            row.push(res.continuation_token);
            crate::core::speculative::spec_stats_update(name, si.id, &res);
            result.push(row);
        }
        Ok(result)
    }
}