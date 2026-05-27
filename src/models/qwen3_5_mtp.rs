// src/models/qwen3_5_mtp.rs
// Qwen3.5 MTP (Multi-Token Prediction) Head
//
// The MTP head is a lightweight transformer layer that predicts future tokens
// using the backbone model's hidden states and KV cache.
//
// Weight prefix: "mtp."
// Structure:
//   - pre_fc_norm_hidden: RMS norm on backbone hidden states
//   - pre_fc_norm_embedding: RMS norm on token embeddings
//   - fc: Linear projection [2*hidden_size -> hidden_size]
//   - layers.0: Single full-attention decoder layer (same arch as backbone)
//   - norm: Final RMS norm before lm_head

use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{DType, Device, Result, Tensor, D};
use std::rc::Rc;
use std::sync::Arc;

pub struct Qwen3_5MtpHead {
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    fc: ReplicatedLinear,
    layer: Qwen3_5MtpDecoderLayer,
    norm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    device: Device,
    dtype: DType,
}

struct Qwen3_5MtpDecoderLayer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Qwen3_5MtpDecoderLayer {
    /// Forward for MTP head - single token, no KV cache needed.
    /// For seq_len=1, self-attention is trivially identity on the value,
    /// so we compute: output = O_proj(V_proj(norm(x))) after RoPE on Q/K.
    /// This avoids going through PagedAttention/FlashInfer backends entirely.
    fn forward_single_token(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        rotary_emb: &Arc<ScalingRotaryEmbedding>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let rope: Arc<dyn ApplyRotaryEmbedding> = rotary_emb.clone();
        let attn_output = self
            .attn
            .forward_single_token_no_cache(&xs, &rope, positions)?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

impl Qwen3_5MtpHead {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let is_qvar_builder = vb.is_qvar_builder();
        let prefix = "mtp.";

        let pre_fc_norm_hidden = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}pre_fc_norm_hidden", prefix)),
            DType::F32,
            !is_qvar_builder,
        )?;

        let pre_fc_norm_embedding = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}pre_fc_norm_embedding", prefix)),
            DType::F32,
            !is_qvar_builder,
        )?;

        let fc = ReplicatedLinear::load_no_bias(
            hidden_size * 2,
            hidden_size,
            vb.pp(&format!("{}fc", prefix)),
            &None,
            &None,
            dtype,
        )?;

        let norm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}norm", prefix)),
            DType::F32,
            !is_qvar_builder,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            config,
            device,
            is_rope_i,
            config.rope_theta,
        )?);

        let layer_prefix = format!("{}layers.0", prefix);
        let attn = Attention::new(
            if is_qvar_builder {
                vb.pp(&layer_prefix)
            } else {
                vb.pp(&format!("{}.self_attn", layer_prefix))
            },
            comm.clone(),
            config,
            None,
            config.sliding_window,
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.pp(&layer_prefix)
            } else {
                vb.pp(&format!("{}.mlp", layer_prefix))
            },
            comm.clone(),
            hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            &config.quantization_config,
            &config.quant,
            false,
            dtype,
            "",
        )?;

        let input_layernorm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}.input_layernorm", layer_prefix)),
            DType::F32,
            !is_qvar_builder,
        )?;

        let post_attention_layernorm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}.post_attention_layernorm", layer_prefix)),
            DType::F32,
            !is_qvar_builder,
        )?;

        let layer = Qwen3_5MtpDecoderLayer {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        };

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layer,
            norm,
            rotary_emb,
            device: device.clone(),
            dtype,
        })
    }

    /// Single MTP draft step.
    ///
    /// Given the backbone's last hidden states and the embedding of the current token,
    /// produces the next hidden state for the MTP head.
    /// The caller should apply lm_head to get logits.
    ///
    /// `backbone_hidden`: [batch, hidden_size] - last hidden from the backbone
    /// `token_embedding`: [batch, hidden_size] - embedding of the last sampled/draft token
    /// `positions`: position IDs for this step
    /// `kv_cache`: MTP head's own KV cache (separate from backbone)
    /// `input_metadata`: attention metadata for this step
    pub fn forward_step(
        &self,
        backbone_hidden: &Tensor,
        token_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let norm_hidden = self.pre_fc_norm_hidden.forward(backbone_hidden)?;
        let norm_embed = self.pre_fc_norm_embedding.forward(token_embedding)?;

        // Concat order: [embedding, hidden] — matches vLLM/HuggingFace weight layout
        // The FC weight's first half corresponds to embedding columns
        let norm_embed = norm_embed.to_dtype(norm_hidden.dtype())?;
        let fused = Tensor::cat(&[norm_embed, norm_hidden], D::Minus1)?;
        let fused = fused.to_dtype(self.dtype)?;
        let xs = self.fc.forward(&fused)?;

        // MTP head uses single-token attention without KV cache
        let xs = self
            .layer
            .forward_single_token(&xs, positions, &self.rotary_emb)?;

        self.norm.forward(&xs)
    }

    /// Draft K tokens given the initial backbone hidden state and the anchor token.
    ///
    /// Uses the backbone's shared lm_head (passed as a closure) and embed_tokens
    /// to autoregressively predict K tokens.
    ///
    /// Returns (draft_token_ids, last_hidden_state_for_each_step).
    pub fn draft_tokens(
        &self,
        initial_hidden: &Tensor,
        anchor_token: u32,
        num_tokens: usize,
        embed_fn: impl Fn(u32) -> Result<Tensor>,
        lm_head_fn: impl Fn(&Tensor) -> Result<Tensor>,
        positions_base: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        let mut draft_tokens = Vec::with_capacity(num_tokens);
        let mut current_hidden = initial_hidden.clone();
        let mut current_token = anchor_token;

        for step in 0..num_tokens {
            let token_embed = embed_fn(current_token)?;
            let token_embed = match token_embed.dims().len() {
                1 => token_embed.unsqueeze(0)?,
                _ => token_embed,
            };

            let current_hidden_2d = match current_hidden.dims().len() {
                1 => current_hidden.unsqueeze(0)?,
                _ => current_hidden.clone(),
            };

            let pos = (positions_base + step) as i64;
            let positions = Tensor::from_vec(vec![pos], (1,), &self.device)?;

            let hidden_out = self.forward_step(&current_hidden_2d, &token_embed, &positions)?;

            let logits = lm_head_fn(&hidden_out.to_dtype(self.dtype)?)?;
            let logits = logits.to_dtype(DType::F32)?;
            let logits_last = if logits.dims().len() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };
            let next_token = logits_last.argmax(D::Minus1)?.to_scalar::<u32>()?;

            draft_tokens.push(next_token);
            current_hidden = if hidden_out.dims().len() == 2 {
                hidden_out.get(hidden_out.dim(0)? - 1)?
            } else {
                hidden_out
            };
            current_token = next_token;
        }

        Ok((draft_tokens, current_hidden))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
