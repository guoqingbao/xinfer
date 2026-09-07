use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::Comm;
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::Qwen4Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};
use std::rc::Rc;
use std::sync::Arc;

/// Qwen4 QSA attention: gated full attention + block-level sparse indexer mask.
///
/// Reference: `Qwen4ExpTextQSAIndexer` + `Qwen4ExpTextAttention` in HF
/// transformers `models/qwen4_exp/modeling_qwen4_exp.py`.
pub struct Qwen4QSAAttention {
    attention: Attention,
    // Replicated (not TP-sharded): the projection packs 4 q heads + 1 shared k
    // head, so column sharding would strand the k head on a single rank. The
    // indexer is tiny (640 x hidden) and every rank needs the full mask anyway.
    index_qk_proj: Linear,
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
        if rotary_dim > index_head_dim {
            candle_core::bail!(
                "Qwen4 QSA: rotary_dim ({rotary_dim}) exceeds indexer head_dim ({index_head_dim}); check partial_rotary_factor"
            );
        }

        let index_qk_weight = vb
            .get(
                (index_qk_out, config.hidden_size),
                "indexer.index_qk_proj.weight",
            )?
            .to_dtype(dtype)?;
        let index_qk_proj = Linear::new(index_qk_weight, None);
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
        // Per the HF reference, the indexer q is RMS-normed then RoPE'd at the
        // current positions, while keys are pooled RAW (no norm, no rope),
        // then normed and RoPE'd at block-start positions. The CUDA kernel
        // performs all of these steps, so pass raw q/k here.
        let index_qk = self.index_qk_proj.forward(xs)?;
        let q_index_size = self.index_n_heads * self.index_head_dim;
        let q_index = index_qk.narrow(1, 0, q_index_size)?.contiguous()?;
        let k_index = index_qk
            .narrow(1, q_index_size, self.index_head_dim)?
            .contiguous()?;

        // TODO: the indexer needs its own raw-key cache to score against the
        // full history during decode; for now it scores within the current
        // forward pass (exact for prefill from position 0).
        let kv_len = seq_len;
        let cos = self.cos_table.narrow(0, 0, kv_len)?;
        let sin = self.sin_table.narrow(0, 0, kv_len)?;

        let _qsa_mask = attention_rs::qwen4::qsa_indexer_mask(
            &q_index,
            &k_index,
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

        // TODO: apply the sparse mask to the attention computation. For
        // sequences within the indexer budget (2048 tokens) QSA selects all
        // visible tokens, i.e. exact full attention — which is what we run.
        self.attention.forward(
            xs,
            &Some(rotary_emb.clone()),
            attention_mask,
            positions,
            cache,
            input_metadata,
        )
    }
}
