// DFlash drafter: owns the (replicated) DFlash draft model and a bounded per-sequence window of
// projected target hidden states that feeds the draft model's cross-attention.

use crate::models::dflash::{DFlashDraftModel, DFlashModelConfig};
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::Mutex;

/// Default cap on the projected-context window kept per sequence (bounds memory for long gens).
const DEFAULT_CONTEXT_WINDOW: usize = 512;

pub struct DFlashDrafter {
    pub draft_model: DFlashDraftModel,
    pub target_layer_ids: Vec<usize>,
    /// Number of speculative (draft) tokens per step (N = block_size - 1).
    pub num_speculative_tokens: usize,
    pub mask_token_id: u32,
    device: Device,
    context_window: usize,
    cached_target_hidden: Mutex<HashMap<usize, Tensor>>,
}

impl DFlashDrafter {
    pub fn new(
        draft_config: &DFlashModelConfig,
        draft_vb: &VarBuilderX,
        dtype: DType,
        device: &Device,
        num_speculative_tokens: Option<usize>,
    ) -> Result<Self> {
        let draft_model = DFlashDraftModel::new(draft_vb, draft_config, dtype, device)?;

        let target_layer_ids = draft_config.target_layer_ids();
        // DFlash config.block_size is the verification block width:
        // [known first token] + N draft tokens. The user-facing count is N.
        let block_size =
            num_speculative_tokens.unwrap_or_else(|| draft_config.block_size.saturating_sub(1));
        let mask_token_id = draft_config.mask_token_id().unwrap_or(0);
        let context_window = std::cmp::min(
            DEFAULT_CONTEXT_WINDOW,
            std::cmp::max(1, draft_config.max_position_embeddings),
        );

        crate::log_info!(
            "DFlash drafter initialized: {} layers, num_speculative_tokens={}, target_layer_ids={:?}, mask_token_id={}, context_window={}",
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
            device: device.clone(),
            context_window,
            cached_target_hidden: Mutex::new(HashMap::new()),
        })
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub fn extract_and_project_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        self.draft_model
            .extract_and_project_hidden(all_hidden_states)
    }

    /// Draft `num_speculative_tokens` ids: embed `[last_token, MASK..MASK]` with the target's
    /// embedding table, run the draft model cross-attending to `target_hidden`, and argmax the
    /// last N positions through the target's lm_head.
    pub fn draft_tokens(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        last_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        assert_eq!(
            last_tokens.len(),
            1,
            "DFlash currently supports batch_size=1 for drafting"
        );

        let n = self.num_speculative_tokens;
        if n == 0 {
            return Ok(Vec::new());
        }
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

    /// Append `projected` (one or more projected context rows) to the per-sequence window,
    /// keeping only the last `context_window` rows.
    pub fn append_context(&self, seq_id: usize, projected: &Tensor) -> Result<()> {
        let mut cached = self.cached_target_hidden.lock().unwrap();
        let rows = projected.dim(0)?;
        if rows == 0 {
            return Ok(());
        }
        let updated = match cached.get(&seq_id).cloned() {
            Some(prev) => Tensor::cat(&[prev, projected.clone()], 0)?,
            None => projected.clone(),
        };
        let total = updated.dim(0)?;
        let keep = std::cmp::min(total, self.context_window);
        let windowed = updated.narrow(0, total - keep, keep)?;
        cached.insert(seq_id, windowed);
        Ok(())
    }

    /// The current projected-context window for a sequence (or None if empty).
    pub fn context(&self, seq_id: usize) -> Result<Option<Tensor>> {
        let cached = self.cached_target_hidden.lock().unwrap();
        Ok(cached.get(&seq_id).cloned())
    }

    /// Drop a finished sequence's window.
    pub fn clear(&self, seq_id: usize) {
        self.cached_target_hidden.lock().unwrap().remove(&seq_id);
    }
}