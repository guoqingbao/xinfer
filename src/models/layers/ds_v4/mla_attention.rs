use super::rope_cache::V4RopeTables;
use crate::models::layers::distributed::{
    shard, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::linear::{linear_is_prefill, linear_no_bias_x, Linear, LinearX};
use crate::models::layers::others::{rms_norm_v4, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor, D};
use std::rc::Rc;

pub struct MlaV4Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
}

impl MlaV4Config {
    pub fn from_config(config: &Config) -> Self {
        let extra: serde_json::Value = config
            .extra_config_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);

        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: extra
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            q_lora_rank: extra
                .get("q_lora_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024) as usize,
            qk_rope_head_dim: extra
                .get("qk_rope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize,
            o_groups: extra.get("o_groups").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
            o_lora_rank: extra
                .get("o_lora_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024) as usize,
            rms_norm_eps: config.rms_norm_eps,
            attention_bias: config.attention_bias.unwrap_or(false),
        }
    }
}

#[allow(unused)]
pub struct MlaV4Attention {
    layer_idx: usize,
    rank: usize,
    wq_a: ReplicatedLinear,
    q_norm: NormX,
    wq_b: TensorParallelColumnLinear,
    wkv: ReplicatedLinear,
    kv_norm: NormX,
    wo_a: LinearX,
    wo_a_group_weights: Vec<Tensor>,
    wo_a_group_scales: Vec<Tensor>,
    wo_a_group_scales_cutlass: Vec<Option<Tensor>>,
    wo_a_block_size: Vec<usize>,
    wo_a_stacked_weights: Option<Tensor>,
    wo_a_stacked_scales: Option<Tensor>,
    wo_b: TensorParallelRowLinear,
    attn_sink: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    q_lora_rank: usize,
    o_groups: usize,
    o_lora_rank: usize,
    per_group_dim: usize,
    sm_scale: f32,
    dtype: DType,
    #[cfg(feature = "cuda")]
    sm_version: usize,
}

impl MlaV4Attention {
    /// DeepSeek V4 stores `wo_a` as FP8, but the reference implementation
    /// intentionally materializes this grouped projection as BF16.  Match
    /// `convert.py`: dequant in F32 (`weight.float() * scale.float()`), then
    /// cast to BF16 — never multiply UE8M0/FP8 scales in BF16/F8 space.
    fn materialize_wo_a_bf16(wo_a: LinearX) -> Result<LinearX> {
        let LinearX::LnFp8(fp8) = wo_a else {
            return Ok(wo_a);
        };
        let (out_dim, in_dim) = fp8.weight.dims2()?;
        let block_size = fp8.weight_block_size.as_slice();
        if block_size != [128, 128] || out_dim % block_size[0] != 0 || in_dim % block_size[1] != 0 {
            candle_core::bail!(
                "DeepSeek V4 wo_a BF16 materialization requires 128x128-aligned FP8 weight, got weight=({out_dim},{in_dim}) block_size={:?}",
                fp8.weight_block_size
            );
        }
        let blocks_y = out_dim / block_size[0];
        let blocks_x = in_dim / block_size[1];
        // F8E4M3 → F32 via Candle's native convert (correct path).
        let weight = fp8.weight.to_dtype(DType::F32)?.reshape((
            blocks_y,
            block_size[0],
            blocks_x,
            block_size[1],
        ))?;
        // UE8M0 / F32 scales → F32 via Candle (uses f8e8m0_to_dtype on CUDA).
        let scales = fp8
            .weight_scale
            .to_dtype(DType::F32)?
            .reshape((blocks_y, 1, blocks_x, 1))?;
        let weight = weight
            .broadcast_mul(&scales)?
            .reshape((out_dim, in_dim))?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        Ok(LinearX::Linear(Linear::new(weight, None)))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        mla_cfg: &MlaV4Config,
        config: &Config,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let hidden_size = mla_cfg.hidden_size;
        let global_num_heads = mla_cfg.num_attention_heads;
        if global_num_heads % comm.world_size() != 0 {
            candle_core::bail!(
                "DeepSeek V4 attention heads {} not divisible by TP world size {}",
                global_num_heads,
                comm.world_size()
            );
        }
        if mla_cfg.o_groups % comm.world_size() != 0 {
            candle_core::bail!(
                "DeepSeek V4 output groups {} not divisible by TP world size {}",
                mla_cfg.o_groups,
                comm.world_size()
            );
        }
        let num_heads = global_num_heads / comm.world_size();
        let head_dim = mla_cfg.head_dim;
        let q_lora_rank = mla_cfg.q_lora_rank;
        let qk_rope_head_dim = mla_cfg.qk_rope_head_dim;
        let qk_nope_head_dim = head_dim - qk_rope_head_dim;
        let o_groups = mla_cfg.o_groups / comm.world_size();
        let o_lora_rank = mla_cfg.o_lora_rank;
        let is_qvar_builder = vb.is_qvar_builder();
        let norm_dtype = if is_qvar_builder || config.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };

        // Q path: wq_a [q_lora_rank, hidden_size] -> q_norm -> wq_b [num_heads*head_dim, q_lora_rank]
        let wq_a = ReplicatedLinear::load_b(
            hidden_size,
            q_lora_rank,
            mla_cfg.attention_bias,
            vb.pp("wq_a"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let q_norm = rms_norm_v4(
            q_lora_rank,
            mla_cfg.rms_norm_eps,
            vb.pp("q_norm"),
            norm_dtype,
        )?;

        let wq_b = TensorParallelColumnLinear::load_with_hints(
            q_lora_rank,
            global_num_heads * head_dim,
            false,
            vb.pp("wq_b"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // KV path: single wkv [head_dim, hidden_size] -> kv_norm
        let wkv = ReplicatedLinear::load_b(
            hidden_size,
            head_dim,
            false,
            vb.pp("wkv"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let kv_norm = rms_norm_v4(head_dim, mla_cfg.rms_norm_eps, vb.pp("kv_norm"), norm_dtype)?;

        // Output path: wo_a (grouped FP8/BF16) -> wo_b (FP8)
        // wo_a is a block-diagonal grouped linear: weight shape [o_groups * o_lora_rank, per_group_dim]
        let per_group_dim = global_num_heads * head_dim / mla_cfg.o_groups;
        let wo_a = Self::materialize_wo_a_bf16(linear_no_bias_x(
            per_group_dim,
            mla_cfg.o_groups * o_lora_rank,
            vb.pp("wo_a"),
            shard(0, comm.rank(), comm.world_size()),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?)?;

        // Pre-slice grouped weight/scale tensors for zero-copy forward pass
        let (wo_a_group_weights, wo_a_group_scales, wo_a_group_scales_cutlass, wo_a_block_size) =
            match &wo_a {
                LinearX::LnFp8(fp8) => {
                    let by = fp8.weight_block_size[0];
                    let scale_rows_per_group = (o_lora_rank + by - 1) / by;
                    let mut weights = Vec::with_capacity(o_groups);
                    let mut scales = Vec::with_capacity(o_groups);
                    let mut scales_c = Vec::with_capacity(o_groups);
                    for g in 0..o_groups {
                        weights.push(
                            fp8.weight
                                .narrow(0, g * o_lora_rank, o_lora_rank)?
                                .contiguous()?,
                        );
                        scales.push(
                            fp8.weight_scale
                                .narrow(0, g * scale_rows_per_group, scale_rows_per_group)?
                                .contiguous()?,
                        );
                        scales_c.push(fp8.weight_scale_cutlass.as_ref().map(|s| {
                            s.narrow(1, g * scale_rows_per_group, scale_rows_per_group)
                                .unwrap()
                                .contiguous()
                                .unwrap()
                        }));
                    }
                    (weights, scales, scales_c, fp8.weight_block_size.clone())
                }
                _ => (vec![], vec![], vec![], vec![128, 128]),
            };

        // Pre-stack FP8 weights and scales for fused grouped GEMM (CUDA graph safe).
        // Used by FlashInfer strided batch GEMM or CUTLASS fused grouped GEMM.
        // wo_a_stacked_weights: [o_groups, o_lora_rank, per_group_dim] U8 (FP8_E4M3)
        // wo_a_stacked_scales: [o_groups, scale_n, scale_k] F32
        #[cfg(any(feature = "flashinfer", feature = "cutlass"))]
        let (wo_a_stacked_weights, wo_a_stacked_scales) = if !wo_a_group_weights.is_empty() {
            let stacked_w = Tensor::stack(&wo_a_group_weights, 0)?.contiguous()?;
            let stacked_s = Tensor::stack(&wo_a_group_scales, 0)?.contiguous()?;
            (Some(stacked_w), Some(stacked_s))
        } else {
            (None, None)
        };
        #[cfg(not(any(feature = "flashinfer", feature = "cutlass")))]
        let (wo_a_stacked_weights, wo_a_stacked_scales): (Option<Tensor>, Option<Tensor>) =
            (None, None);

        let wo_b = TensorParallelRowLinear::load_with_hints(
            mla_cfg.o_groups * o_lora_rank,
            hidden_size,
            vb.pp("wo_b"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // Learnable attention sink bias per head
        let attn_sink = match vb.get_with_hints_dtype(
            (global_num_heads,),
            "attn_sink",
            shard(0, comm.rank(), comm.world_size()),
            DType::F32,
        ) {
            Ok(t) => Some(t),
            Err(_) => None,
        };

        // DeepSeek V4 sparse attention always uses plain 1/sqrt(head_dim).
        // Do not apply YaRN mscale² — openinfer/vLLM keep the unscaled form.
        let sm_scale = 1.0 / (head_dim as f32).sqrt();

        #[cfg(feature = "cuda")]
        let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
            .unwrap_or(0) as usize;

        Ok(Self {
            layer_idx,
            rank: comm.rank(),
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            wo_a,
            wo_a_group_weights,
            wo_a_group_scales,
            wo_a_group_scales_cutlass,
            wo_a_block_size,
            wo_a_stacked_weights,
            wo_a_stacked_scales,
            wo_b,
            attn_sink,
            num_heads,
            head_dim,
            qk_rope_head_dim,
            qk_nope_head_dim,
            q_lora_rank,
            o_groups,
            o_lora_rank,
            per_group_dim,
            sm_scale,
            dtype,
            #[cfg(feature = "cuda")]
            sm_version,
        })
    }

    /// Project attention output through grouped wo_a -> wo_b
    /// wo_a is block-diagonal: input is split into o_groups, each group processed independently
    #[allow(unused)]
    fn project_output(&self, attn_out: &Tensor, seq_len: usize, xs_dtype: DType) -> Result<Tensor> {
        // attn_out: [seq_len, num_heads * head_dim]
        let y = attn_out.to_dtype(xs_dtype)?;

        // Grouped linear: reshape to [seq_len, o_groups, per_group_dim]
        let grouped = y.reshape((seq_len, self.o_groups, self.per_group_dim))?;
        // -> [o_groups, seq_len, per_group_dim]
        let grouped = grouped.transpose(0, 1)?.contiguous()?;

        let low_rank = if !self.wo_a_group_weights.is_empty() {
            // FP8 model path
            #[cfg(feature = "flashinfer")]
            {
                // FlashInfer FP8 strided GEMM requires SM90+ (Hopper);
                // fall back to per-group fp8_grouped_matmul on SM80 (Ampere).
                #[cfg(feature = "cuda")]
                let use_flashinfer_strided = self.sm_version >= 90;
                #[cfg(not(feature = "cuda"))]
                let use_flashinfer_strided = false;

                if use_flashinfer_strided {
                    if let (Some(sw), Some(ss)) =
                        (&self.wo_a_stacked_weights, &self.wo_a_stacked_scales)
                    {
                        #[cfg(feature = "cutlass")]
                        let result = attention_rs::fp8_linear::fp8_grouped_gemm_fused(
                            &grouped,
                            sw,
                            ss,
                            self.o_groups,
                            seq_len,
                            self.o_lora_rank,
                            self.per_group_dim,
                        )?;
                        #[cfg(not(feature = "cutlass"))]
                        let result = attention_rs::fp8_linear::fp8_grouped_matmul_strided(
                            &grouped,
                            sw,
                            ss,
                            self.o_groups,
                            seq_len,
                            self.o_lora_rank,
                            self.per_group_dim,
                        )?;
                        result
                            .transpose(0, 1)?
                            .contiguous()?
                            .reshape((seq_len, self.o_groups * self.o_lora_rank))?
                    } else {
                        attention_rs::fp8_linear::fp8_grouped_matmul(
                            &grouped,
                            &self.wo_a_group_weights,
                            &self.wo_a_group_scales,
                            &self.wo_a_group_scales_cutlass,
                            &self.wo_a_block_size,
                            linear_is_prefill(),
                        )?
                    }
                } else {
                    attention_rs::fp8_linear::fp8_grouped_matmul(
                        &grouped,
                        &self.wo_a_group_weights,
                        &self.wo_a_group_scales,
                        &self.wo_a_group_scales_cutlass,
                        &self.wo_a_block_size,
                        linear_is_prefill(),
                    )?
                }
            }
            #[cfg(not(feature = "flashinfer"))]
            {
                #[cfg(feature = "cutlass")]
                {
                    // CUTLASS fused grouped GEMM also requires SM90+
                    #[cfg(feature = "cuda")]
                    let use_cutlass_fused = self.sm_version >= 90;
                    #[cfg(not(feature = "cuda"))]
                    let use_cutlass_fused = false;

                    if use_cutlass_fused {
                        if let (Some(sw), Some(ss)) =
                            (&self.wo_a_stacked_weights, &self.wo_a_stacked_scales)
                        {
                            let result = attention_rs::fp8_linear::fp8_grouped_gemm_fused(
                                &grouped,
                                sw,
                                ss,
                                self.o_groups,
                                seq_len,
                                self.o_lora_rank,
                                self.per_group_dim,
                            )?;
                            result
                                .transpose(0, 1)?
                                .contiguous()?
                                .reshape((seq_len, self.o_groups * self.o_lora_rank))?
                        } else {
                            attention_rs::fp8_linear::fp8_grouped_matmul(
                                &grouped,
                                &self.wo_a_group_weights,
                                &self.wo_a_group_scales,
                                &self.wo_a_group_scales_cutlass,
                                &self.wo_a_block_size,
                                linear_is_prefill(),
                            )?
                        }
                    } else {
                        attention_rs::fp8_linear::fp8_grouped_matmul(
                            &grouped,
                            &self.wo_a_group_weights,
                            &self.wo_a_group_scales,
                            &self.wo_a_group_scales_cutlass,
                            &self.wo_a_block_size,
                            linear_is_prefill(),
                        )?
                    }
                }
                #[cfg(not(feature = "cutlass"))]
                {
                    attention_rs::fp8_linear::fp8_grouped_matmul(
                        &grouped,
                        &self.wo_a_group_weights,
                        &self.wo_a_group_scales,
                        &self.wo_a_group_scales_cutlass,
                        &self.wo_a_block_size,
                        linear_is_prefill(),
                    )?
                }
            }
        } else {
            // BF16 model: single batched matmul via cuBLAS strided batched GEMM
            match &self.wo_a {
                LinearX::Linear(ln) => {
                    let weight_3d = ln.weight().reshape((
                        self.o_groups,
                        self.o_lora_rank,
                        self.per_group_dim,
                    ))?;
                    let result = grouped.matmul(&weight_3d.transpose(1, 2)?)?;
                    result
                        .transpose(0, 1)?
                        .contiguous()?
                        .reshape((seq_len, self.o_groups * self.o_lora_rank))?
                }
                _ => {
                    candle_core::bail!(
                        "MlaV4Attention: grouped output projection not supported for this quantization type"
                    );
                }
            }
        };

        // Row-parallel all-reduce in FP32 and retain FP32 for hc_post_f32_branch.
        // Casting back to BF16 here reused intermediate storage that later looked
        // like NaN/garbage by the time hc_post ran on compressed layers.
        let local = self.wo_b.forward_local(&low_rank)?;
        let output = self.wo_b.reduce_local_f32(local, DType::F32)?;
        output.device().synchronize()?;
        Ok(output)
    }

    /// Per-head RMSNorm on Q output via CUDA kernel.
    fn per_head_rms_norm(&self, q: &Tensor, eps: f32) -> Result<Tensor> {
        // q: [seq_len, num_heads, head_dim]
        attention_rs::deepseek_v4::head_rms_norm(q, self.num_heads, self.head_dim, eps)
    }

    /// Compute the shared Q bottleneck (wq_a -> q_norm) for indexer and sparse attention.
    pub fn compute_qr(&self, xs: &Tensor) -> Result<Tensor> {
        let q_a_out = self.wq_a.forward(xs)?;
        self.q_norm.forward(&q_a_out)
    }

    /// Sparse attention forward for DeepSeek V4 prefill and decode.
    ///
    /// This implements the V4 attention mechanism where all layer types (SWA, CSA, HCA)
    /// use sparse_attn with topk_idxs selecting which KV positions each query attends to.
    ///
    /// - SWA (ratio=0): Only window indices into raw KV
    /// - CSA (ratio=4): Window + indexer-selected compressed KV indices
    /// - HCA (ratio=128): Window + all compressed KV indices
    ///
    /// kv_combined: raw KV [seq_len, head_dim] concatenated with compressed KV if applicable
    /// topk_idxs: [seq_len, topk] indices into kv_combined
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        xs: &Tensor,
        qr: &Tensor,
        rope: &V4RopeTables,
        start_pos: usize,
        kv_combined: &Tensor,
        topk_idxs: &Tensor,
        kv_len: usize,
        topk: usize,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        // Q: wq_b -> reshape -> per-head RMSNorm -> partial RoPE
        let q_raw = self.wq_b.forward(qr)?;
        let q = q_raw.reshape((seq_len, self.num_heads, self.head_dim))?;
        let q = self.per_head_rms_norm(&q, 1e-6_f32)?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;
        let q_pe = q_pe.contiguous()?;

        let q_pe = q_pe.to_dtype(DType::BF16)?;
        rope.apply_inplace(&q_pe, start_pos, false)?;

        // Reconstruct full Q: [nope | pe]
        let q_pe = q_pe.to_dtype(self.dtype)?;
        let q_nope = q_nope.contiguous()?.to_dtype(self.dtype)?;
        let q_full = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;

        let attn_sink = self
            .attn_sink
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("V4 sparse attention requires attn_sink"))?;

        let out = attention_rs::deepseek_v4::sparse_attention(
            &q_full.contiguous()?,
            &kv_combined.contiguous()?,
            attn_sink,
            &topk_idxs.contiguous()?,
            seq_len,
            self.num_heads,
            self.head_dim,
            kv_len,
            topk,
            self.sm_scale,
        )?;
        out.device().synchronize()?;

        // Inverse RoPE on the rope dimensions of the output (conjugate / -sin).
        let out_nope = out.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let out_pe = out.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;
        let out_pe = out_pe.contiguous()?;
        let out_pe_inv = out_pe.to_dtype(DType::BF16)?;
        rope.apply_inplace(&out_pe_inv, start_pos, true)?;

        let out_full = Tensor::cat(
            &[&out_nope.contiguous()?, &out_pe_inv.to_dtype(self.dtype)?],
            D::Minus1,
        )?;

        let y = out_full
            .reshape((seq_len, self.num_heads * self.head_dim))?
            .to_dtype(xs.dtype())?;
        self.project_output(&y, seq_len, xs.dtype())
    }

    /// Compute raw KV: wkv(x) -> kv_norm. Returns [seq_len, head_dim] BF16.
    pub fn wkv_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let kv_raw = self.wkv.forward(xs)?;
        let kv = self.kv_norm.forward(&kv_raw)?;
        kv.to_dtype(self.dtype)
    }

    /// Populate the normal MLA cache after sparse compressed-layer prefill.
    pub fn cache_prefill_kv(
        &self,
        kv: &Tensor,
        rope: &V4RopeTables,
        cache: (&Tensor, &Tensor),
        input_metadata: &InputMetadata,
    ) -> Result<()> {
        let seq_len = kv.dim(0)?;
        let ckv = kv
            .narrow(D::Minus1, 0, self.qk_nope_head_dim)?
            .to_dtype(self.dtype)?;
        let k_pe = kv
            .narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?
            .reshape((seq_len, 1, self.qk_rope_head_dim))?
            .contiguous()?;
        rope.apply_inplace(&k_pe, 0, false)?;
        let k_pe = k_pe
            .reshape((seq_len, self.qk_rope_head_dim))?
            .to_dtype(self.dtype)?;
        attention_rs::mla::concat_and_cache_mla(
            &ckv,
            &k_pe,
            cache.0,
            cache.1,
            &input_metadata.slot_mapping,
        )
    }

    pub fn get_num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn get_qk_rope_head_dim(&self) -> usize {
        self.qk_rope_head_dim
    }
}
