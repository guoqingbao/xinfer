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

/// DSA (DeepSeek Sparse Attention) lightning indexer.
///
/// Selects the top-k most relevant tokens for each query position,
/// producing sparse indices used to mask the main MLA attention.
/// All operations are GPU-only — no CPU↔GPU sync — compatible with CUDA graph capture.
///
/// Reference: HuggingFace transformers `DeepseekV32Indexer`
pub struct DsaIndexer {
    wq_b: ReplicatedLinear,
    wk: ReplicatedLinear,
    k_norm: NormX,
    weights_proj: ReplicatedLinear,
    cfg: IndexerConfig,
    softmax_scale: f32,
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

        let softmax_scale = 1.0 / (cfg.index_head_dim as f32).sqrt();

        Ok(Self {
            wq_b,
            wk,
            k_norm,
            weights_proj,
            cfg,
            softmax_scale,
        })
    }

    pub fn index_topk(&self) -> usize {
        self.cfg.index_topk
    }

    /// Run the indexer to produce top-k token indices for sparse attention.
    /// All operations are GPU-only — no CPU↔GPU sync. Compatible with CUDA graph capture.
    ///
    /// Returns `[seq_len, topk]` U32 indices, or None when seq_len <= topk
    /// (dense attention is equivalent in that case).
    ///
    /// # Arguments
    /// * `xs` - hidden states `[seq_len, hidden_size]`
    /// * `q_resid` - query latent from `q_a_layernorm(q_a_proj(x))`, shape `[seq_len, q_lora_rank]`
    /// * `rotary_emb` - rotary embedding (shared with main attention)
    /// * `positions` - position tensor
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

        let idx_q_f32 = idx_q.to_dtype(DType::F32)?;
        let idx_k_f32 = idx_k.to_dtype(DType::F32)?;

        // [n_heads, seq_len, head_dim] @ [head_dim, seq_len] -> [n_heads, seq_len, seq_len]
        let idx_q_t = idx_q_f32.transpose(0, 1)?.contiguous()?;
        let idx_k_t = idx_k_f32.t()?.contiguous()?;
        let idx_k_t = idx_k_t
            .unsqueeze(0)?
            .broadcast_as((self.cfg.index_n_heads, self.cfg.index_head_dim, seq_len))?
            .contiguous()?;
        let scores = idx_q_t.matmul(&idx_k_t)?;
        let scores = (scores * (self.softmax_scale as f64))?;
        let scores = scores.relu()?;

        // Per-head weights
        let weights = self.weights_proj.forward(xs)?;
        let weights = weights.to_dtype(DType::F32)?;
        let head_scale = (self.cfg.index_n_heads as f32).powf(-0.5);
        let weights = (weights * (head_scale as f64))?;

        // Weighted sum: [seq_len, 1, n_heads] @ [n_heads, seq_len, seq_len] -> [seq_len, seq_len]
        let weights_t = weights.unsqueeze(1)?;
        let scores_t = scores.permute((1, 0, 2))?;
        let index_scores = weights_t.matmul(&scores_t)?.squeeze(1)?;

        // GPU causal mask: create zeros then apply causal_mask kernel (all GPU)
        let causal = Tensor::zeros((seq_len, seq_len), DType::F32, xs.device())?;
        attention_rs::mask::causal_mask(&causal, None)?;
        let index_scores = (index_scores + causal)?;

        // Top-k selection (GPU-only, returns U32 indices)
        let topk = self.cfg.index_topk.min(seq_len);
        let index_scores = index_scores.contiguous()?;
        let (_topk_values, topk_indices) = attention_rs::topk::topk_select(&index_scores, topk)?;
        Ok(Some(topk_indices))
    }
}
