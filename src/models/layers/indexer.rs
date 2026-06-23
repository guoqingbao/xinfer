use crate::models::layers::distributed::ReplicatedLinear;
use crate::models::layers::others::{layer_norm, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{DType, Result, Tensor, D};
use std::sync::Arc;

pub struct IndexerConfig {
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub index_skip_topk_offset: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub hidden_size: usize,
}

/// DSA (DeepSeek Sparse Attention) lightning indexer (prefill-only).
///
/// Selects the top-k most relevant tokens for each query position,
/// producing sparse indices used to mask the main MLA attention during prefill.
/// All operations are GPU-only — no CPU↔GPU sync.
///
/// For decode, dense MLA with FlashInfer is used (faster than sparse scoring
/// at all practical context lengths due to per-layer indexer overhead).
///
/// Reference: HuggingFace transformers `DeepseekV32Indexer`
pub struct DsaIndexer {
    wq_b: ReplicatedLinear,
    wk: ReplicatedLinear,
    k_norm: NormX,
    weights_proj: ReplicatedLinear,
    cfg: IndexerConfig,
    score_scale: f32,
}

impl DsaIndexer {
    pub fn new(vb: VarBuilderX, config: &Config, cfg: IndexerConfig, dtype: DType) -> Result<Self> {
        let is_gguf = vb.is_qvar_builder();
        let wq_b = ReplicatedLinear::load_no_bias(
            cfg.q_lora_rank,
            cfg.index_n_heads * cfg.index_head_dim,
            vb.pp(if is_gguf { "attn_q_b" } else { "wq_b" }),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let wk = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.index_head_dim,
            vb.pp(if is_gguf { "attn_k" } else { "wk" }),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let k_norm = layer_norm(cfg.index_head_dim, 1e-6, true, vb.pp("k_norm"), dtype)?;
        let weights_proj = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.index_n_heads,
            vb.pp(if is_gguf { "proj" } else { "weights_proj" }),
            &None,
            &config.quant,
            dtype,
        )?;

        // Combine softmax_scale (head_dim^-0.5) and head_scale (n_heads^-0.5)
        // into a single constant, eliminating a per-token tensor multiplication.
        let softmax_scale = 1.0 / (cfg.index_head_dim as f32).sqrt();
        let head_scale = (cfg.index_n_heads as f32).powf(-0.5);
        let score_scale = softmax_scale * head_scale;

        Ok(Self {
            wq_b,
            wk,
            k_norm,
            weights_proj,
            cfg,
            score_scale,
        })
    }

    pub fn index_topk(&self) -> usize {
        self.cfg.index_topk
    }

    /// Run the indexer to produce top-k token indices for sparse attention (prefill).
    ///
    /// Returns `[seq_len, topk]` U32 indices, or None when seq_len <= topk
    /// (dense attention is equivalent in that case).
    #[cfg(feature = "cuda")]
    pub fn forward(
        &self,
        xs: &Tensor,
        q_resid: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        positions: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (seq_len, _) = xs.dims2()?;
        if seq_len <= self.cfg.index_topk {
            return Ok(None);
        }

        // Indexer Q: wq_b(q_resid) -> [seq_len, n_heads, head_dim]
        let idx_q = self.wq_b.forward(q_resid)?;
        let idx_q = idx_q.reshape((seq_len, self.cfg.index_n_heads, self.cfg.index_head_dim))?;

        let idx_q_rope = idx_q.narrow(D::Minus1, 0, self.cfg.qk_rope_head_dim)?;
        let idx_q_pass = idx_q.narrow(
            D::Minus1,
            self.cfg.qk_rope_head_dim,
            self.cfg.index_head_dim - self.cfg.qk_rope_head_dim,
        )?;

        // Indexer K: k_norm(wk(hidden_states)) -> [seq_len, 1, head_dim]
        let idx_k = self.wk.forward(xs)?;
        let idx_k = self.k_norm.forward(&idx_k)?;
        let idx_k = idx_k.unsqueeze(1)?;

        let idx_k_rope = idx_k.narrow(D::Minus1, 0, self.cfg.qk_rope_head_dim)?;
        let idx_k_pass = idx_k.narrow(
            D::Minus1,
            self.cfg.qk_rope_head_dim,
            self.cfg.index_head_dim - self.cfg.qk_rope_head_dim,
        )?;

        // Apply RoPE
        let (idx_q_rope, idx_k_rope) = if let Some(re) = rotary_emb {
            let idx_q_rope_c = idx_q_rope.contiguous()?;
            let idx_k_rope_c = idx_k_rope.contiguous()?;
            match re.apply_rotary_emb_qkv(&idx_q_rope_c, &idx_k_rope_c, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (idx_q_rope_c, idx_k_rope_c),
            }
        } else {
            (idx_q_rope.contiguous()?, idx_k_rope.contiguous()?)
        };

        let idx_q = Tensor::cat(&[&idx_q_rope, &idx_q_pass.contiguous()?], D::Minus1)?;
        let idx_k = Tensor::cat(&[&idx_k_rope, &idx_k_pass.contiguous()?], D::Minus1)?;
        let idx_k = idx_k.squeeze(1)?;

        // Ensure BF16 for the fused CUDA kernel
        let idx_q = idx_q.to_dtype(DType::BF16)?.contiguous()?;
        let idx_k = idx_k.to_dtype(DType::BF16)?.contiguous()?;

        // Per-head weights: [seq_len, n_heads] F32
        let weights = self.weights_proj.forward(xs)?;
        let weights = weights.to_dtype(DType::F32)?.contiguous()?;

        // Fused CUDA kernel: score computation + causal mask + top-k selection
        // score_scale already includes both softmax_scale and head_scale
        let topk = self.cfg.index_topk.min(seq_len);
        let topk_indices = attention_rs::mla::dsa_lightning_indexer_prefill(
            &idx_q,
            &idx_k,
            &weights,
            topk,
            self.score_scale,
        )?;
        Ok(Some(topk_indices))
    }
}
