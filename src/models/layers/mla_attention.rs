use crate::models::layers::distributed::{shard, Comm, ReplicatedLinear};
use crate::models::layers::indexer::{DsaIndexer, IndexerConfig};
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor, D};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

pub struct MlaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub index_head_dim: Option<usize>,
    pub index_n_heads: Option<usize>,
    pub index_topk: Option<usize>,
    pub index_skip_topk_offset: Option<usize>,
}

impl MlaConfig {
    pub fn from_config(config: &Config) -> Self {
        let extra: serde_json::Value = config
            .extra_config_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);

        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            q_lora_rank: extra
                .get("q_lora_rank")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            kv_lora_rank: extra
                .get("kv_lora_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            qk_nope_head_dim: extra
                .get("qk_nope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            qk_rope_head_dim: extra
                .get("qk_rope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize,
            v_head_dim: extra
                .get("v_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            rms_norm_eps: config.rms_norm_eps,
            attention_bias: config.attention_bias.unwrap_or(false),
            index_head_dim: extra
                .get("index_head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_n_heads: extra
                .get("index_n_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_topk: extra
                .get("index_topk")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_skip_topk_offset: extra
                .get("index_skip_topk_offset")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}

#[allow(unused)]
pub struct MlaAttention {
    q_a_proj: Option<ReplicatedLinear>,
    q_a_layernorm: Option<NormX>,
    q_b_proj: Option<ReplicatedLinear>,
    q_proj: Option<ReplicatedLinear>,
    kv_a_proj_with_mqa: ReplicatedLinear,
    kv_a_layernorm: NormX,
    #[allow(dead_code)]
    kv_b_proj: Option<ReplicatedLinear>,
    o_proj: ReplicatedLinear,
    w_uk: Tensor,
    w_uv_t: Tensor,
    num_heads: usize,
    q_head_dim: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    sm_scale: f32,
    rope_scale: f32,
    rope_theta: f32,
    promote_qk_to_f32: bool,
    dtype: DType,
    indexer: Option<DsaIndexer>,
}

impl MlaAttention {
    pub fn new(
        vb: VarBuilderX,
        _comm: Rc<Comm>,
        mla_cfg: &MlaConfig,
        config: &Config,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let hidden_size = mla_cfg.hidden_size;
        let num_heads = mla_cfg.num_attention_heads;
        let kv_lora_rank = mla_cfg.kv_lora_rank;
        let qk_nope_head_dim = mla_cfg.qk_nope_head_dim;
        let qk_rope_head_dim = mla_cfg.qk_rope_head_dim;
        let v_head_dim = mla_cfg.v_head_dim;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let is_qvar_builder = vb.is_qvar_builder();
        let norm_dtype = if is_qvar_builder || config.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };

        let key_map: HashMap<&str, &str> = [
            ("q_a_proj", "attn_q_a"),
            ("q_a_layernorm", "attn_q_a_norm"),
            ("q_b_proj", "attn_q_b"),
            ("kv_a_proj_with_mqa", "attn_kv_a_mqa"),
            ("kv_a_layernorm", "attn_kv_a_norm"),
            ("kv_b_proj", "attn_k_b"),
            ("o_proj", "attn_output"),
        ]
        .iter()
        .cloned()
        .collect();

        let pp = |name: &str| -> VarBuilderX {
            if is_qvar_builder {
                if let Some(gguf_name) = key_map.get(name) {
                    vb.pp(gguf_name)
                } else {
                    vb.pp(name)
                }
            } else {
                vb.pp(name)
            }
        };

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = mla_cfg.q_lora_rank {
                let q_a = ReplicatedLinear::load_b(
                    hidden_size,
                    q_lora_rank,
                    mla_cfg.attention_bias,
                    pp("q_a_proj"),
                    &config.quantization_config,
                    &config.quant,
                    dtype,
                )?;
                let q_a_ln = rms_norm(
                    q_lora_rank,
                    mla_cfg.rms_norm_eps,
                    pp("q_a_layernorm"),
                    norm_dtype,
                    false,
                )?;
                let q_b = ReplicatedLinear::load_b(
                    q_lora_rank,
                    num_heads * q_head_dim,
                    false,
                    pp("q_b_proj"),
                    &config.quantization_config,
                    &config.quant,
                    dtype,
                )?;
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = ReplicatedLinear::load_b(
                    hidden_size,
                    num_heads * q_head_dim,
                    mla_cfg.attention_bias,
                    vb.pp("q_proj"),
                    &config.quantization_config,
                    &config.quant,
                    dtype,
                )?;
                (None, None, None, Some(q))
            };

        let kv_a_proj_with_mqa = ReplicatedLinear::load_b(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            mla_cfg.attention_bias,
            pp("kv_a_proj_with_mqa"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let kv_a_layernorm = rms_norm(
            kv_lora_rank,
            mla_cfg.rms_norm_eps,
            pp("kv_a_layernorm"),
            norm_dtype,
            false,
        )?;

        let has_separate_kv_b = is_qvar_builder && vb.pp("attn_k_b").has_key("weight");

        let kv_b_proj = if !has_separate_kv_b {
            Some(ReplicatedLinear::load_b(
                kv_lora_rank,
                num_heads * (qk_nope_head_dim + v_head_dim),
                false,
                pp("kv_b_proj"),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?)
        } else {
            None
        };

        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * v_head_dim,
            hidden_size,
            pp("o_proj"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let (w_uk, w_uv_t) = if has_separate_kv_b {
            // attn_k_b layout varies between GGUF converters:
            //   Variant A: [qk_nope_head_dim, kv_lora_rank, num_heads] (transposed by converter)
            //   Variant B: [num_heads, kv_lora_rank, qk_nope_head_dim] (natural order)
            // Target: w_uk = [num_heads, qk_nope_head_dim, kv_lora_rank]
            let k_b_vb = vb.pp("attn_k_b");
            let k_b_shape = k_b_vb.tensor_shape("weight");
            let k_b_heads_first = k_b_shape
                .as_ref()
                .is_some_and(|s| s.len() == 3 && s[0] == num_heads && s[2] != num_heads);
            let k_b_weight = if k_b_heads_first {
                k_b_vb.get_with_hints_dtype(
                    (num_heads, kv_lora_rank, qk_nope_head_dim),
                    "weight",
                    shard(0, 0, 1),
                    dtype,
                )?
            } else {
                k_b_vb.get_with_hints_dtype(
                    (qk_nope_head_dim, kv_lora_rank, num_heads),
                    "weight",
                    shard(0, 0, 1),
                    dtype,
                )?
            };

            // attn_v_b layout varies between GGUF converters:
            //   Variant A: [v_head_dim, kv_lora_rank, num_heads] (transposed by converter)
            //   Variant B: [num_heads, v_head_dim, kv_lora_rank] (natural order)
            // Target: w_uv_t = [num_heads, kv_lora_rank, v_head_dim]
            let v_b_vb = vb.pp("attn_v_b");
            let v_b_shape = v_b_vb.tensor_shape("weight");
            let v_b_kv_last = v_b_shape
                .as_ref()
                .is_some_and(|s| s.len() == 3 && s[2] == kv_lora_rank && s[1] != kv_lora_rank);
            let v_b_weight = if v_b_kv_last {
                v_b_vb.get_with_hints_dtype(
                    (num_heads, v_head_dim, kv_lora_rank),
                    "weight",
                    shard(0, 0, 1),
                    dtype,
                )?
            } else {
                v_b_vb.get_with_hints_dtype(
                    (v_head_dim, kv_lora_rank, num_heads),
                    "weight",
                    shard(0, 0, 1),
                    dtype,
                )?
            };

            // w_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]
            let w_uk = if k_b_heads_first {
                // (num_heads, kv_lora_rank, qk_nope_head_dim) → transpose(1,2)
                k_b_weight.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?
            } else {
                // (qk_nope_head_dim, kv_lora_rank, num_heads) → permute(2,0,1)
                k_b_weight
                    .permute((2, 0, 1))?
                    .contiguous()?
                    .to_dtype(dtype)?
            };
            // w_uv_t target: [num_heads, kv_lora_rank, v_head_dim]
            let w_uv_t = if v_b_kv_last {
                // (num_heads, v_head_dim, kv_lora_rank) → transpose(1,2)
                v_b_weight.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?
            } else {
                // (v_head_dim, kv_lora_rank, num_heads) → permute(2,0,1)
                // → (num_heads, v_head_dim, kv_lora_rank) → transpose(1,2)
                let w_uv = v_b_weight.permute((2, 0, 1))?.contiguous()?;
                w_uv.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?
            };
            (w_uk, w_uv_t)
        } else {
            // For FP8 models, kv_b_proj.weight is stored as F8_E4M3 with block-wise
            // weight_scale_inv. We must dequantize before splitting into W_UK / W_UV.
            // Use the already-loaded kv_b_proj (which properly handles FP8 via LnFp8)
            // to dequantize by applying it to an identity matrix.
            let kv_b_out_dim = num_heads * (qk_nope_head_dim + v_head_dim);
            let is_fp8 = config
                .quantization_config
                .as_ref()
                .is_some_and(|c| c.quant_method == "fp8");
            let kv_b_weight = if is_fp8 {
                if let Some(ref kv_b) = kv_b_proj {
                    let identity = Tensor::eye(kv_lora_rank, dtype, &vb.device())?;
                    let dequantized = kv_b.forward(&identity)?;
                    // dequantized: [kv_lora_rank, kv_b_out_dim] = I @ W^T
                    dequantized.t()?.contiguous()?
                } else {
                    vb.pp("kv_b_proj").get_with_hints_dtype(
                        (kv_b_out_dim, kv_lora_rank),
                        "weight",
                        shard(0, 0, 1),
                        dtype,
                    )?
                }
            } else {
                vb.pp("kv_b_proj").get_with_hints_dtype(
                    (kv_b_out_dim, kv_lora_rank),
                    "weight",
                    shard(0, 0, 1),
                    dtype,
                )?
            };
            let w =
                kv_b_weight.reshape((num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank))?;
            let w_uk = w
                .narrow(1, 0, qk_nope_head_dim)?
                .contiguous()?
                .to_dtype(dtype)?;
            let w_uv = w.narrow(1, qk_nope_head_dim, v_head_dim)?.contiguous()?;
            let w_uv_t = w_uv.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?;
            (w_uk, w_uv_t)
        };

        let mut sm_scale = 1.0 / (q_head_dim as f32).sqrt();
        let mut rope_scale = 1.0f32;

        if let Some(ref rope_scaling) = config.rope_scaling {
            use crate::utils::config::RopeScalingValue;
            let is_yarn = rope_scaling.get("type").and_then(|v| {
                if let RopeScalingValue::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            }) == Some("yarn");
            if is_yarn {
                let factor = rope_scaling
                    .get("factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                let mscale_all_dim = rope_scaling
                    .get("mscale_all_dim")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                if mscale_all_dim > 0.0 && factor > 1.0 {
                    let mscale = 0.1 * mscale_all_dim * factor.ln() + 1.0;
                    sm_scale *= mscale * mscale;
                }
                rope_scale = 1.0;
            }
        }

        let skip_offset = mla_cfg.index_skip_topk_offset.unwrap_or(1);
        let has_indexer = mla_cfg.index_head_dim.is_some()
            && layer_idx >= skip_offset
            && (vb.pp("indexer").has_key("wq_b.weight")
                || vb.pp("indexer").has_key("attn_q_b.weight"));
        let indexer = if has_indexer {
            let idx_cfg = IndexerConfig {
                index_head_dim: mla_cfg.index_head_dim.unwrap(),
                index_n_heads: mla_cfg.index_n_heads.unwrap_or(4),
                index_topk: mla_cfg.index_topk.unwrap_or(2048),
                index_skip_topk_offset: mla_cfg.index_skip_topk_offset.unwrap_or(1),
                qk_rope_head_dim,
                q_lora_rank: mla_cfg.q_lora_rank.unwrap_or(256),
                hidden_size,
            };
            Some(DsaIndexer::new(vb.pp("indexer"), config, idx_cfg, dtype)?)
        } else {
            None
        };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            w_uk,
            w_uv_t,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            sm_scale,
            rope_scale,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            promote_qk_to_f32: is_qvar_builder || config.higher_precision_required(),
            dtype,
            indexer,
        })
    }

    /// Project the fused kernel output through w_uv_t and reshape for o_proj.
    #[allow(unused)]
    fn project_mla_output(
        &self,
        attn_out: &Tensor,
        seq_len: usize,
        xs_dtype: DType,
    ) -> Result<Tensor> {
        // attn_out: [seq_len, num_heads, kv_lora_rank]
        // w_uv_t: [num_heads, kv_lora_rank, v_head_dim]
        let attn_t = attn_out.transpose(0, 1)?.contiguous()?;
        let y = attn_t.matmul(&self.w_uv_t)?;
        let y = y.transpose(0, 1)?.contiguous()?;
        let y = y.reshape((seq_len, self.num_heads * self.v_head_dim))?;
        let y = y.to_dtype(xs_dtype)?;
        self.o_proj.forward(&y)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        // Q projection — keep q_resid for DSA indexer
        let (q, q_resid) = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(xs)?;
            let q_a_normed = q_a_ln.forward(&q_a_out)?;
            let q = q_b.forward(&q_a_normed)?;
            (q, Some(q_a_normed))
        } else {
            (self.q_proj.as_ref().unwrap().forward(xs)?, None)
        };

        let q = q.reshape((seq_len, self.num_heads, self.q_head_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection
        let kv_a = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv = kv_a.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_a.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        let ckv = self.kv_a_layernorm.forward(&ckv)?;

        // RoPE on q_pe and k_pe
        let k_pe = k_pe_raw.reshape((seq_len, 1, self.qk_rope_head_dim))?;
        let q_pe_for_rope = q_pe.contiguous()?;

        let (q_pe_for_rope, k_pe) = if self.promote_qk_to_f32 {
            (
                q_pe_for_rope.to_dtype(DType::F32)?,
                k_pe.to_dtype(DType::F32)?,
            )
        } else {
            (q_pe_for_rope, k_pe)
        };
        let (q_pe, k_pe) = if let Some(rotary_emb) = &rotary_emb {
            match rotary_emb.apply_rotary_emb_qkv(&q_pe_for_rope, &k_pe, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q_pe_for_rope, k_pe),
            }
        } else {
            (q_pe_for_rope, k_pe)
        };
        let k_pe = k_pe.squeeze(1)?;

        let q_pe = q_pe.to_dtype(self.dtype)?;
        let q_nope = q_nope.contiguous()?.to_dtype(self.dtype)?;
        let ckv = ckv.to_dtype(self.dtype)?;
        let k_pe = k_pe.to_dtype(self.dtype)?;

        // FlashInfer MLA path
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_ref() {
            if let Some((ckv_cache, kpe_cache)) = cache {
                attention_rs::mla::concat_and_cache_mla(
                    &ckv,
                    &k_pe,
                    ckv_cache,
                    kpe_cache,
                    &input_metadata.slot_mapping,
                )?;

                // Absorb w_uk into q_nope: q_nope_absorbed = q_nope @ w_uk
                // q_nope: [seq_len, num_heads, qk_nope_head_dim]
                // w_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]
                let q_nope_t = q_nope.transpose(0, 1)?.contiguous()?;
                let q_nope_absorbed = q_nope_t
                    .matmul(&self.w_uk)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .to_dtype(self.dtype)?;
                // q_nope_absorbed: [seq_len, num_heads, kv_lora_rank]
                let q_pe = q_pe.to_dtype(self.dtype)?;

                let page_size = ckv_cache.dim(1)?;

                // Returns None when seq_len <= topk (dense is equivalent).
                if input_metadata.is_prefill {
                    if let (Some(indexer), Some(q_res)) = (&self.indexer, &q_resid) {
                        if let Some(block_tables) = &input_metadata.block_tables {
                            if let Some(context_lens) = &input_metadata.context_lens {
                                if let Some(topk_idxs) =
                                    indexer.forward(xs, q_res, rotary_emb, positions)?
                                {
                                    let ckv_cache_3d = ckv_cache.squeeze(2)?;
                                    let kpe_cache_3d = kpe_cache.squeeze(2)?;

                                    let cu_seqlens_q =
                                        input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "MLA sparse prefill requires cu_seqlens_q",
                                            )
                                        })?;

                                    // All tensors are already U32 from the scheduler —
                                    // no dtype conversion needed.
                                    let attn_out = attention_rs::mla::mla_sparse_paged_prefill(
                                        &q_nope_absorbed,
                                        &q_pe,
                                        &ckv_cache_3d,
                                        &kpe_cache_3d,
                                        block_tables,
                                        context_lens,
                                        cu_seqlens_q,
                                        &topk_idxs,
                                        self.sm_scale,
                                    )?;

                                    return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                                }
                            }
                        }
                    }
                }

                // DSA is prefill-only: sparse decode adds per-layer scoring overhead that exceeds
                // the attention savings. Dense MLA decode with FlashInfer is faster at all
                // practical context lengths. Indexer K cache is populated during prefill.

                let attn_out = if input_metadata.is_prefill {
                    let plan_info = fm.mla_prefill_plan_info.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA prefill requires mla_prefill_plan_info")
                    })?;
                    attention_rs::mla::mla_prefill_run(
                        &q_nope_absorbed,
                        &q_pe,
                        ckv_cache,
                        kpe_cache,
                        &fm.indices,
                        self.num_heads,
                        page_size,
                        self.sm_scale,
                        plan_info,
                        true, // causal
                    )?
                } else {
                    let plan_info = fm.mla_decode_plan_info.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA decode requires mla_decode_plan_info")
                    })?;
                    attention_rs::mla::mla_decode_run(
                        &q_nope_absorbed,
                        &q_pe,
                        ckv_cache,
                        kpe_cache,
                        &fm.indptr,
                        &fm.indices,
                        &fm.last_len,
                        seq_len,
                        self.num_heads,
                        page_size,
                        self.sm_scale,
                        self.rope_scale,
                        self.rope_theta,
                        plan_info,
                        fm.use_cuda_graph,
                    )?
                };

                // attn_out: [seq_len, num_heads, kv_lora_rank]
                // Project via w_uv_t: [num_heads, kv_lora_rank, v_head_dim]
                let attn_out_t = attn_out.transpose(0, 1)?.contiguous()?;
                let y = attn_out_t
                    .matmul(&self.w_uv_t)?
                    .transpose(0, 1)?
                    .contiguous()?;
                // y: [seq_len, num_heads, v_head_dim]
                let y = y.reshape((seq_len, self.num_heads * self.v_head_dim))?;
                let y = y.to_dtype(xs.dtype())?;
                return self.o_proj.forward(&y);
            }
        }

        // Non-FlashInfer path: fused MLA CUDA kernels with paged KV cache
        #[cfg(feature = "cuda")]
        if let Some((ckv_cache, kpe_cache)) = cache {
            attention_rs::mla::concat_and_cache_mla(
                &ckv,
                &k_pe,
                ckv_cache,
                kpe_cache,
                &input_metadata.slot_mapping,
            )?;

            // Compute absorbed query: q_absorbed = q_nope @ w_uk
            // q_nope: [seq_len, num_heads, qk_nope_head_dim]
            // w_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]
            // q_absorbed: [seq_len, num_heads, kv_lora_rank]
            let q_nope_t = q_nope.transpose(0, 1)?.contiguous()?;
            let q_absorbed = q_nope_t.matmul(&self.w_uk)?.transpose(0, 1)?.contiguous()?;

            // The fused kernels need block_tables and context_lens as i32 on GPU,
            // and ckv_cache/kpe_cache in 3D [num_blocks, block_size, dim] layout.
            // The allocator gives us [num_blocks, block_size, 1, dim] so squeeze dim 2.
            let ckv_cache_3d = ckv_cache.squeeze(2)?;
            let kpe_cache_3d = kpe_cache.squeeze(2)?;

            if let (Some(block_tables), Some(context_lens)) =
                (&input_metadata.block_tables, &input_metadata.context_lens)
            {
                // block_tables, context_lens, cu_seqlens_q are already U32 from
                // the scheduler — pass directly to CUDA kernels without conversion.

                if input_metadata.is_prefill {
                    let cu_seqlens_q = input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA fused prefill requires cu_seqlens_q")
                    })?;

                    if let (Some(indexer), Some(q_res)) = (&self.indexer, &q_resid) {
                        if let Some(topk_idxs) =
                            indexer.forward(xs, q_res, rotary_emb, positions)?
                        {
                            let attn_out = attention_rs::mla::mla_sparse_paged_prefill(
                                &q_absorbed,
                                &q_pe,
                                &ckv_cache_3d,
                                &kpe_cache_3d,
                                block_tables,
                                context_lens,
                                cu_seqlens_q,
                                &topk_idxs,
                                self.sm_scale,
                            )?;
                            return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                        }
                    }

                    let attn_out = attention_rs::mla::mla_paged_prefill(
                        &q_absorbed,
                        &q_pe,
                        &ckv_cache_3d,
                        &kpe_cache_3d,
                        block_tables,
                        context_lens,
                        cu_seqlens_q,
                        self.sm_scale,
                    )?;
                    return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                }

                // DSA is prefill-only: dense MLA decode is faster at all practical context lengths.
                let attn_out = attention_rs::mla::mla_paged_decode(
                    &q_absorbed,
                    &q_pe,
                    &ckv_cache_3d,
                    &kpe_cache_3d,
                    block_tables,
                    context_lens,
                    self.sm_scale,
                )?;
                return self.project_mla_output(&attn_out, seq_len, xs.dtype());
            }
        }
        candle_core::bail!("MLA attention requires CUDA platform!")
    }
}
