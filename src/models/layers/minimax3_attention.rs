use crate::models::layers::distributed::{
    kv_head_shard, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Result, Tensor};
use std::rc::Rc;
use std::sync::Arc;

/// MiniMax M3's sparse GQA attention.
///
/// M3 is not MLA/DSA: its index branch is a four-head, 128-wide block scorer
/// attached to ordinary 64-head/4-KV-head attention.  The prefill path uses
/// the local CUDA block-indexer and sparse GQA kernels; decode uses the shared
/// dense paged-attention implementation until an index-K side cache is added
/// to the scheduler's cache allocation.
pub struct MiniMax3SparseAttention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    index_q_proj: TensorParallelColumnLinear,
    index_k_proj: ReplicatedLinear,
    q_norm: NormX,
    k_norm: NormX,
    index_q_norm: NormX,
    index_k_norm: NormX,
    o_proj: TensorParallelRowLinear,
    attn: PagedAttention,
    num_heads: usize,
    num_kv_heads: usize,
    index_heads: usize,
    head_dim: usize,
    index_dim: usize,
    topk_blocks: usize,
    block_size: usize,
    scale: f32,
    dtype: DType,
}

impl MiniMax3SparseAttention {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        index_dim: usize,
        index_heads: usize,
        topk_blocks: usize,
        block_size: usize,
        use_gemma_norm: bool,
        dtype: DType,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim.unwrap_or(128);
        let total_heads = config.num_attention_heads;
        let num_heads = total_heads / comm.world_size();
        let (num_kv_heads, kv_shard) =
            kv_head_shard(config.num_key_value_heads, comm.rank(), comm.world_size())?;

        if index_heads != config.num_key_value_heads {
            candle_core::bail!(
                "MiniMax M3 index heads ({index_heads}) must equal KV heads ({})",
                config.num_key_value_heads
            );
        }

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden,
            total_heads * head_dim,
            false,
            vb.pp("q_proj"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_shard(
            hidden,
            config.num_key_value_heads * head_dim,
            false,
            vb.pp("k_proj"),
            kv_shard,
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let v_proj = TensorParallelColumnLinear::load_with_shard(
            hidden,
            config.num_key_value_heads * head_dim,
            false,
            vb.pp("v_proj"),
            kv_shard,
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let index_q_proj = TensorParallelColumnLinear::load_with_shard(
            hidden,
            index_heads * index_dim,
            false,
            vb.pp("index_q_proj"),
            kv_shard,
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let index_k_proj = ReplicatedLinear::load_no_bias(
            hidden,
            index_dim,
            vb.pp("index_k_proj"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("q_norm"),
            DType::F32,
            use_gemma_norm,
        )?;
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("k_norm"),
            DType::F32,
            use_gemma_norm,
        )?;
        let index_q_norm = rms_norm(
            index_dim,
            config.rms_norm_eps,
            vb.pp("index_q_norm"),
            DType::F32,
            use_gemma_norm,
        )?;
        let index_k_norm = rms_norm(
            index_dim,
            config.rms_norm_eps,
            vb.pp("index_k_norm"),
            DType::F32,
            use_gemma_norm,
        )?;
        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden,
            vb.pp("o_proj"),
            comm,
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let attn = PagedAttention::new(
            num_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            Some(num_kv_heads),
            config.sliding_window,
            vb.device().clone(),
            None,
            config.kvcache_dtype.is_fp8_keys(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            index_q_proj,
            index_k_proj,
            q_norm,
            k_norm,
            index_q_norm,
            index_k_norm,
            o_proj,
            attn,
            num_heads,
            num_kv_heads,
            index_heads: num_kv_heads,
            head_dim,
            index_dim,
            topk_blocks,
            block_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
            dtype,
        })
    }

    fn norm_heads(&self, x: Tensor, norm: &NormX, heads: usize, dim: usize) -> Result<Tensor> {
        norm.forward(&x.reshape(((), dim))?)?
            .reshape(((), heads, dim))
    }

    fn project_and_rotate(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let q = self.norm_heads(
            self.q_proj.forward(xs)?,
            &self.q_norm,
            self.num_heads,
            self.head_dim,
        )?;
        let k = self.norm_heads(
            self.k_proj.forward(xs)?,
            &self.k_norm,
            self.num_kv_heads,
            self.head_dim,
        )?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape(((), self.num_kv_heads, self.head_dim))?;

        let iq = self.norm_heads(
            self.index_q_proj.forward(xs)?,
            &self.index_q_norm,
            self.index_heads,
            self.index_dim,
        )?;
        let ik = self.index_k_norm.forward(&self.index_k_proj.forward(xs)?)?;

        let (q, k) = if let Some(rope) = rotary_emb {
            match rope.apply_rotary_emb_qkv(&q, &k, positions)? {
                Some((q, k)) => (q, k),
                None => (q, k),
            }
        } else {
            (q, k)
        };
        let (iq, ik_3d) = if let Some(rope) = rotary_emb {
            let ik_3d = ik.unsqueeze(1)?;
            match rope.apply_rotary_emb_qkv(&iq, &ik_3d, positions)? {
                Some((q, k)) => (q, k.squeeze(1)?),
                None => (iq, ik),
            }
        } else {
            (iq, ik)
        };
        Ok((
            q.contiguous()?.to_dtype(self.dtype)?,
            k.contiguous()?.to_dtype(self.dtype)?,
            v.contiguous()?.to_dtype(self.dtype)?,
            iq.contiguous()?.to_dtype(self.dtype)?,
            ik_3d.contiguous()?.to_dtype(self.dtype)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (tokens, _) = xs.dims2()?;
        let (q, k, v, index_q, index_k) = self.project_and_rotate(xs, rotary_emb, positions)?;

        // The packed CUDA kernels handle multiple uncached sequences using
        // cu_seqlens_q. Prefix-cache/chunked-prefill and decode still use the
        // established paged path until the scheduler allocates index-K cache.
        let full_uncached_prefill = input_metadata.is_prefill
            && input_metadata.seqlens.as_ref().is_some_and(|q| {
                q.last().copied().map(|x| x as usize) == Some(tokens)
                    && input_metadata.kv_seqlens.as_ref().is_some_and(|k| k == q)
            });
        if full_uncached_prefill {
            // Cache K/V for subsequent decode before selecting sparse blocks.
            if let Some((key_cache, value_cache)) = cache {
                attention_rs::reshape_and_cache(
                    &k,
                    &v,
                    key_cache,
                    value_cache,
                    None,
                    None,
                    &input_metadata.slot_mapping.flatten_all()?,
                )?;
            }
            let cu_seqlens_q = input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("MiniMax M3 prefill needs cu_seqlens_q".into())
            })?;
            let topk = attention_rs::minimax_m3_indexer_prefill(
                &index_q,
                &index_k,
                cu_seqlens_q,
                self.topk_blocks,
                self.block_size,
                input_metadata.max_seqlen_q,
                1.0 / (self.index_dim as f32).sqrt(),
            )?;
            let attn = attention_rs::minimax_m3_sparse_attention_prefill(
                &q,
                &k,
                &v,
                &topk,
                cu_seqlens_q,
                input_metadata.max_seqlen_q,
                self.scale,
                self.block_size,
            )?;
            let y = attn.reshape((tokens, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&y);
        }

        let y = self.attn.forward(
            &q,
            &k,
            &v,
            None,
            cache.map(|(k, _)| k.clone()),
            cache.map(|(_, v)| v.clone()),
            input_metadata,
            None,
        )?;
        self.o_proj
            .forward(&y.reshape((tokens, self.num_heads * self.head_dim))?)
    }
}
