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
        "dflash"
    }

    fn verify_target_layers(&self) -> &[usize] {
        self.inner.target_layer_ids()
    }

    fn anchor(&self, runner: &ModelRunner, seqs: Seqs, seq: &MtpSeqInfo) -> Result<(u32, Option<Tensor>)> {
        let seq_id = seq.id;
        let target_layer_ids = self.inner.target_layer_ids();

        // ---- Step 1: anchor decode + update the projected-hidden context window. ----
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
            Some(c) => c,
            None => return Ok(vec![]),
        };
        let n_mask = self.inner.num_speculative();
        if n_mask == 0 {
            return Ok(vec![]);
        }
        let (logits, hidden_n) =
            self.inner.draft_logits(&ctx, &embed_fn, &lm_head_fn, anchor, n_mask)?;
        // Grammar-aware drafting: batched single-VOB mask (3a) by default; the granular
        // per-position FSM walk when XINFER_SPEC_GRANULAR_MASK is set.
        if runner.guided_decoding.is_guided(seq_id) {
            if crate::utils::env::spec_granular_mask() {
                return runner.guided_decoding.masked_drafts(seq_id, &logits);
            }
            let masked = runner.guided_decoding.mask_rows(seq_id, &logits)?;
            return masked
                .to_dtype(candle_core::DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .to_vec1::<u32>();
        }
        self.inner.select_from_logits(&logits, &hidden_n, anchor)
    }

    fn on_verified(
        &self,
        _runner: &ModelRunner,
        seq: &MtpSeqInfo,
        _proposal: &Proposal,
        vhidden: &[Tensor],
        accepted: usize,
    ) -> Result<()> {
        // Refresh the context window with the verify block's accepted rows.
        if !vhidden.is_empty() && accepted > 0 {
            let vproj = self.inner.extract_and_project_hidden(vhidden)?;
            let keep = std::cmp::min(accepted + 1, vproj.dim(0)?);
            if keep > 0 {
                self.inner.append_context(seq.id, &vproj.narrow(0, 0, keep)?)?;
            }
        }
        Ok(())
    }
}

impl ModelRunner {
    /// DFlash speculative decode: route through the shared core with the DFlash drafter.
    pub fn run_dflash_decode(&self, seqs: Seqs) -> Result<Vec<Vec<u32>>> {
        match &self.dflash_drafter {
            Some(inner) => {
                let drafter = DflashDrafter {
                    inner: inner.clone(),
                };
                self.run_spec_decode(seqs, &drafter)
            }
            None => {
                let output = self.run(seqs, false)?;
                Ok(output.into_iter().map(|t| vec![t]).collect())
            }
        }
    }
}