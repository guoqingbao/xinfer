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
        let draft_vb = VarBuilderX(either::Either::Left(draft_vb), String::new(), None);

        let draft_model =
            DFlashDraftModel::new(&draft_vb, comm, draft_config, DType::BF16, device)?;

        let target_layer_ids = draft_config.target_layer_ids();
        // DFlash config.block_size is the verification block width:
        // [known first token] + N draft tokens. The user-facing speculative
        // token count is N.
        let block_size =
            num_speculative_tokens.unwrap_or_else(|| draft_config.block_size.saturating_sub(1));
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
        let draft_logits = lm_head_fn(&draft_hidden.narrow(0, total_out - n, n)?)?;

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
