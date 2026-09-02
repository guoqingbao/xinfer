use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{shard, Comm, TensorParallelColumnLinear};
use crate::models::layers::others::rms_norm;
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::Qwen4Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor};
use std::rc::Rc;
use std::sync::Arc;

/// Qwen4 QSA attention: gated full attention + block-level sparse indexer mask.
pub struct Qwen4QSAAttention {
    attention: Attention,
    index_qk_proj: TensorParallelColumnLinear,
    q_index_norm_weight: Tensor,
    k_index_norm_weight: Tensor,
    index_n_heads: usize,
    index_head_dim: usize,
    compress_ratio: usize,
    block_topk: usize,
    rotary_dim: usize,
    rms_norm_eps: f64,
    cos_table: Tensor,
    sin_table: Tensor,
}

impl Qwen4QSAAttention {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        qwen4: &Qwen4Config,
        dtype: DType,
        cos_table: Tensor,
        sin_table: Tensor,
    ) -> Result<Self> {
        let index_n_heads = qwen4.indexer_n_heads;
        let index_head_dim = qwen4.indexer_head_dim;
        let index_kv_heads = qwen4.indexer_kv_heads;
        let index_qk_out = (index_n_heads + index_kv_heads) * index_head_dim;
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let partial = config.partial_rotary_factor.unwrap_or(1.0) as f64;
        let rotary_dim = (head_dim as f64 * partial) as usize;

        let index_qk_proj = TensorParallelColumnLinear::load_with_hints(
            config.hidden_size,
            index_qk_out,
            false,
            vb.pp("indexer.index_qk_proj"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let _q_index_norm = rms_norm(
            index_head_dim,
            config.rms_norm_eps,
            vb.pp("indexer.q_layernorm"),
            dtype,
            false,
        )?;
        let q_index_norm_weight = vb
            .get((index_head_dim,), "indexer.q_layernorm.weight")?
            .to_dtype(dtype)?;
        let k_index_norm_weight = vb
            .get((index_head_dim,), "indexer.k_layernorm.weight")?
            .to_dtype(dtype)?;

        let mut attn_config = config.clone();
        attn_config.attn_output_gate = Some(true);
        let attention = Attention::new(
            vb.clone(),
            comm,
            &attn_config,
            None,
            config.sliding_window,
            dtype,
        )?;

        Ok(Self {
            attention,
            index_qk_proj,
            q_index_norm_weight,
            k_index_norm_weight,
            index_n_heads,
            index_head_dim,
            compress_ratio: qwen4.indexer_compress_ratio,
            block_topk: qwen4.indexer_budget / qwen4.indexer_compress_ratio,
            rotary_dim,
            rms_norm_eps: config.rms_norm_eps,
            cos_table,
            sin_table,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Arc<dyn ApplyRotaryEmbedding>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let index_qk = self.index_qk_proj.forward(xs)?;
        let q_index_size = self.index_n_heads * self.index_head_dim;
        let q_index = index_qk.narrow(1, 0, q_index_size)?;
        let k_index = index_qk.narrow(1, q_index_size, self.index_head_dim)?;

        let q_index = q_index.reshape((seq_len, self.index_n_heads, self.index_head_dim))?;
        let q_index_flat = q_index.reshape((seq_len * self.index_n_heads, self.index_head_dim))?;
        let q_index_normed = self.rms_norm(&q_index_flat, &self.q_index_norm_weight)?;
        let q_index = q_index_normed.reshape((seq_len, self.index_n_heads, self.index_head_dim))?;

        let q_for_rope = q_index.clone();
        let k_for_rope = k_index.reshape((seq_len, 1, self.index_head_dim))?;
        let q_index_rope =
            match rotary_emb.apply_rotary_emb_qkv(&q_for_rope, &k_for_rope, positions)? {
                Some((q_rope, _)) => q_rope,
                None => q_for_rope,
            };

        let k_index_flat = k_for_rope.reshape((seq_len, self.index_head_dim))?;
        let kv_len = seq_len;
        let cos = self.cos_table.narrow(0, 0, kv_len)?;
        let sin = self.sin_table.narrow(0, 0, kv_len)?;

        let q_flat = q_index_rope.reshape((seq_len, self.index_n_heads * self.index_head_dim))?;
        let _qsa_mask = attention_rs::qwen4::qsa_indexer_mask(
            &q_flat,
            &k_index_flat,
            &self.q_index_norm_weight,
            &self.k_index_norm_weight,
            &cos,
            &sin,
            self.index_n_heads,
            self.index_head_dim,
            self.rotary_dim,
            self.compress_ratio,
            self.block_topk,
            self.rms_norm_eps as f32,
        )?;

        self.attention.forward(
            xs,
            &Some(rotary_emb.clone()),
            attention_mask,
            positions,
            cache,
            input_metadata,
        )
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
        let w = (weight.to_dtype(x.dtype())? + 1.0)?;
        x.broadcast_mul(&w)
    }
}
