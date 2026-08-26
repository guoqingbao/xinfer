// DFlash drafter: owns the (replicated) DFlash draft model and a bounded per-sequence window of
// projected target hidden states that feeds the draft model's cross-attention.

use crate::models::dflash::{DFlashDraftModel, DFlashModelConfig};
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

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
        yarn_factor: Option<f64>,
    ) -> Result<Self> {
        let draft_model = DFlashDraftModel::new(draft_vb, draft_config, dtype, device, yarn_factor)?;

        let target_layer_ids = draft_config.target_layer_ids();
        // DFlash config.block_size is the verification block width:
        // [known first token] + N draft tokens. The user-facing count is N.
        let block_size = num_speculative_tokens
            .or_else(|| draft_config.effective_block_size().map(|w| w.saturating_sub(1)))
            .unwrap_or(0);
        let mask_token_id = draft_config.mask_token_id().unwrap_or(0);
        let context_window = crate::utils::env::dflash_context_window();
        // 0 = unbounded full history (matches the original DFlash branch); a positive value
        // caps the window to the last N projected rows to bound memory on long generations.
        // [dflash-debug] commented out (only used by the debug init log above)
        // let has_conv = draft_config
        //     .dflash_config
        //     .as_ref()
        //     .is_some_and(|dc| dc.conv_kernel_size.is_some() && dc.conv_group_size.is_some());
        // let has_selector = draft_config
        //     .dflash_config
        //     .as_ref()
        //     .is_some_and(|dc| dc.selector_rank.is_some() && dc.selector_top_k.is_some());

        // [dflash-debug] commented out
        // crate::log_info!(
        //     "DFlash drafter initialized: {} layers, version={}, num_speculative_tokens={}, target_layer_ids={:?}, mask_token_id={}, context_window={}, yarn_scaling_factor={:?}, dflash2_conv={}, dflash2_selector={}, backend={:?}, kernels={}",
        //     draft_config.num_hidden_layers,
        //     if draft_config.has_v2_components() { "dflash2" } else { "dflash1" },
        //     block_size,
        //     target_layer_ids,
        //     mask_token_id,
        //     context_window,
        //     yarn_factor,
        //     has_conv,
        //     has_selector,
        //     crate::utils::env::dflash_backend(),
        //     crate::utils::env::dflash_use_kernels(),
        // );
        crate::log_info!(
            "DFlash drafter initialized: {} layers, num_speculative_tokens={}, target_layer_ids={:?}, mask_token_id={}, context_window={}, yarn_scaling_factor={:?}",
            draft_config.num_hidden_layers,
            block_size,
            target_layer_ids,
            mask_token_id,
            context_window,
            yarn_factor,
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
    /// The number of speculative (MASK) tokens per step.
    pub fn num_speculative(&self) -> usize {
        self.num_speculative_tokens
    }

    /// True when the fused DFlash2 CUDA-kernel backend is active (`XINFER_DFLASH_BACKEND`
    /// auto/v2 + cuda built). The draft model's conv + candidate selector dispatch on this.
    pub fn uses_kernels(&self) -> bool {
        crate::utils::env::dflash_use_kernels()
    }

    /// Grammar-gated draft-token selection (delegates to the draft model). `allow = None`
    /// preserves the original unmasked interface; `allow = Some` applies the per-position
    /// grammar gate (in-kernel on the v2 backend, pre-masked on v1).
    pub fn select_tokens_masked(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        anchor: u32,
        allow: Option<&Tensor>,
    ) -> Result<Vec<u32>> {
        // crate::log_info!(
        //     "[dflash-debug] DFlashDrafter.select_tokens_masked: allow={} has_v2_components={}",
        //     allow.is_some(),
        //     self.draft_model.config.has_v2_components()
        // );
        self.draft_model.select_tokens_masked(logits, hidden_n, anchor, allow)
    }

    /// Build the DFlash2 block [anchor, MASK x n], run the draft model, and return the
    /// target lm_head logits + hiddens over the n MASK positions.
    pub fn draft_logits(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        anchor: u32,
        n_mask: usize,
    ) -> Result<(Tensor, Tensor)> {
        let dtype = self.draft_model.dtype();
        // Block = [anchor, MASK x n_mask].
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
        let target_hidden_cast = target_hidden_2d.to_dtype(dtype)?;

        let ctx_len = target_hidden_cast.dim(0)?;
        let noise_2d = if noise_embedding.rank() == 3 {
            let (_, s, h) = noise_embedding.dims3()?;
            noise_embedding.reshape((s, h))?
        } else {
            noise_embedding
        };

        let total_len = ctx_len + block_len;
        let positions: Vec<i64> = (0..total_len as i64).collect();
        let positions_tensor = Tensor::from_vec(positions, (total_len,), &self.device)?;

        let draft_hidden =
            self.draft_model
                .forward(&target_hidden_cast, &noise_2d, &positions_tensor)?;
        self.draft_model.draft_logits(&draft_hidden, n_mask, lm_head_fn)
    }

    /// Select draft tokens from pre-computed logits (DFlash2 selector, else argmax).
    pub fn select_from_logits(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        sel_anchor: u32,
    ) -> Result<Vec<u32>> {
        self.draft_model.select_from_logits(logits, hidden_n, sel_anchor)
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
        // context_window == 0 means unbounded full history (the original DFlash behavior).
        if self.context_window == 0 {
            cached.insert(seq_id, updated);
            return Ok(());
        }
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