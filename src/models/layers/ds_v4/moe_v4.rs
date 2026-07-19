//! DeepSeek V4 MoE: MXFP4 experts + fused hash-gate (tid2eid) router.
//!
//! Kept separate from the shared `layers::moe` module so other models keep a
//! clean general MoE path. Hash routing matches openinfer's
//! `deepseek_hash_gate_cuda` (partial gate dots + sqrt-softplus + normalize).

use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::LinearX as Linear;
use crate::models::layers::moe::{
    gated_activation, resolve_expert_proj_prefix, sort_expert_assignments,
    try_load_e_score_correction_bias,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::moe;
use attention_rs::sort::ArgSortOp;
use candle_core::Module;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::var_builder::Shard;
use candle_nn::Activation;
use either::Either;
use std::rc::Rc;

/// Numerically stable softplus for V4 sqrt-softplus score routing.
fn stable_softplus(xs: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros_like(xs)?;
    let positive = xs.broadcast_maximum(&zero)?;
    let neg_abs = xs.broadcast_minimum(&xs.neg()?)?;
    let tail = (neg_abs.exp()? + 1.0)?.log()?;
    positive.broadcast_add(&tail)
}

fn select_topk_indices(scores: &Tensor, topk: usize, is_prefill: bool) -> Result<Tensor> {
    let sorted_idx = if is_prefill {
        scores.contiguous()?.arg_sort(false)?
    } else {
        scores.arg_sort_last_dim(false)?
    };
    sorted_idx.narrow(D::Minus1, 0, topk)?.contiguous()
}

/// V4 score-gate: sqrt(softplus(logits)), bias only for expert selection,
/// then L1-normalize selected scores × route_scale (openinfer / official).
fn score_route(
    router_logits: &Tensor,
    bias: Option<&Tensor>,
    topk: usize,
    route_scale: f32,
    is_prefill: bool,
) -> Result<(Tensor, Tensor)> {
    let scores = stable_softplus(&router_logits.to_dtype(DType::F32)?)?.sqrt()?;
    let scores_for_choice = if let Some(bias) = bias {
        scores.broadcast_add(&bias.to_dtype(DType::F32)?)?
    } else {
        scores.clone()
    };
    let topk_indices = select_topk_indices(&scores_for_choice, topk, is_prefill)?;
    let mut topk_weights = scores.gather(&topk_indices, D::Minus1)?;
    topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
    if route_scale != 1.0 {
        topk_weights = (topk_weights * route_scale as f64)?;
    }
    Ok((topk_weights, topk_indices.to_dtype(DType::U32)?))
}

/// V4 router: hash layers use fused CUDA tid2eid path; later layers use
/// learned sqrt-softplus score/top-k (kept out of general `moe.rs`).
pub enum V4Router {
    Hash {
        /// BF16 `[n_experts, hidden]` — same layout as Linear weight.
        gate_weight: Tensor,
        /// I64 `[vocab, topk]` — keep I64 like openinfer / checkpoint.
        tid2eid: Tensor,
        n_experts: usize,
        topk: usize,
        route_scale: f32,
    },
    Score {
        gate: Linear,
        /// Selection bias only (`gate.bias`); weights stay unbiased.
        bias: Option<Tensor>,
        topk: usize,
        route_scale: f32,
    },
}

impl V4Router {
    pub fn route(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Hash {
                gate_weight,
                tid2eid,
                n_experts,
                topk,
                route_scale,
            } => {
                let ids = input_ids.ok_or_else(|| {
                    candle_core::Error::Msg("DeepSeek V4 hash-gate requires input_ids".into())
                })?;
                let x_bf16 = if xs.dtype() != DType::BF16 {
                    xs.to_dtype(DType::BF16)?
                } else {
                    xs.clone()
                };
                attention_rs::deepseek_v4::hash_gate_route(
                    &x_bf16,
                    gate_weight,
                    tid2eid,
                    ids,
                    *n_experts,
                    *topk,
                    *route_scale,
                )
            }
            Self::Score {
                gate,
                bias,
                topk,
                route_scale,
            } => {
                // Score gate: F32 weights + F32 activations.
                let gate_input = if xs.dtype() != DType::F32 {
                    xs.to_dtype(DType::F32)?
                } else {
                    xs.clone()
                };
                let router_logits = gate.forward(&gate_input)?.to_dtype(DType::F32)?;
                score_route(
                    &router_logits,
                    bias.as_ref(),
                    *topk,
                    *route_scale,
                    is_prefill,
                )
            }
        }
    }

    pub fn num_experts_per_tok(&self) -> usize {
        match self {
            Self::Hash { topk, .. } | Self::Score { topk, .. } => *topk,
        }
    }
}

/// DeepSeek-V4 MXFP4 MoE (hash or score router + FP4 experts).
pub struct FusedMoeMxfp4 {
    pub(crate) router: V4Router,
    pub(crate) gate_up_blocks: Tensor,
    pub(crate) gate_up_scales: Tensor,
    pub(crate) down_blocks: Tensor,
    pub(crate) down_scales: Tensor,
    pub(crate) w_size_n: usize,
    pub(crate) act: Activation,
    pub(crate) all_reduce: AllReduce,
    pub(crate) world_size: usize,
    pub(crate) dtype: DType,
    /// Asymmetric SwiGLU clamp (gate ≤ limit, up ∈ [-limit, limit]).
    pub(crate) swiglu_limit: f32,
}

impl FusedMoeMxfp4 {
    fn mxfp4_tensor_name_packed(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_packed") {
            "weight_packed"
        } else if vb.contains_tensor("weight") {
            "weight"
        } else {
            "blocks"
        }
    }

    fn mxfp4_tensor_name_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_scale") {
            "weight_scale"
        } else if vb.contains_tensor("scale") {
            "scale"
        } else {
            "scales"
        }
    }

    fn load_mxfp4_packed(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        shape: (usize, usize),
        name: &str,
        shard_hint: Shard,
    ) -> Result<Tensor> {
        // Packed FP4/INT8 tensors are byte payloads.  Candle maps safetensors
        // I8/F4 to U8 without changing the payload, which is what the MXFP4
        // kernels expect.
        vb.get_with_hints_dtype(shape, name, shard_hint, DType::U8)
    }

    fn load_mxfp4_scale(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        shape: (usize, usize),
        name: &str,
        shard_hint: Shard,
    ) -> Result<Tensor> {
        // MXFP4 scales are normally safetensors F8_E8M0.  Load them with the
        // logical FP8 dtype so their U8-backed exponent bytes are preserved.
        // Requesting U8 first would invoke Candle's numeric F8 -> U8 cast and
        // corrupt the scale bytes.  The F8E4M3 fallback covers older exports;
        // U8 is retained for checkpoints that store the same bytes as I8/U8.
        vb.get_with_hints_dtype(shape, name, shard_hint, DType::F8E8M0)
            .or_else(|_| vb.get_with_hints_dtype(shape, name, shard_hint, DType::F8E4M3))
            .or_else(|_| vb.get_with_hints_dtype(shape, name, shard_hint, DType::U8))
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        Self::new_with_gate(cfg, vb.pp("gate"), vb.pp("experts"), &vb, comm, dtype)
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilderX,
        experts_vb: VarBuilderX,
        bias_vb: &VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        // Prefer I64 tid2eid (checkpoint / openinfer). Hash layers have no bias.
        let tid2eid = if gate_vb.has_key("tid2eid") {
            let vocab = cfg.vocab_size.unwrap_or(0);
            if vocab == 0 {
                None
            } else {
                let shape = (vocab, moe_cfg.num_experts_per_tok);
                Some(
                    gate_vb
                        .get_with_hints_dtype(shape, "tid2eid", Default::default(), DType::I64)?
                        .contiguous()?,
                )
            }
        } else {
            None
        };

        let router = if let Some(tid2eid) = tid2eid {
            // Hash gate: load BF16 weight only — fused kernel does expert-row dots.
            let gate_weight = gate_vb
                .get_with_hints_dtype(
                    (num_experts, cfg.hidden_size),
                    "weight",
                    Default::default(),
                    DType::BF16,
                )?
                .contiguous()?;
            let route_scale = moe_cfg.routed_scaling_factor.unwrap_or(1.0) as f32;
            V4Router::Hash {
                gate_weight,
                tid2eid,
                n_experts: num_experts,
                topk: moe_cfg.num_experts_per_tok,
                route_scale,
            }
        } else {
            // Score layers: plain dense F32 gate (openinfer). Bypass LinearX
            // quant dispatch so weights cannot stay BF16/FP8.
            let gate = match &gate_vb.0 {
                Either::Left(vb) => Linear::Linear(crate::models::layers::linear::linear_no_bias(
                    cfg.hidden_size,
                    num_experts,
                    vb.clone(),
                    Shard::default(),
                    DType::F32,
                )?),
                _ => candle_core::bail!("DeepSeek V4 score gate requires safetensors weights"),
            };
            let bias = try_load_e_score_correction_bias(bias_vb, num_experts);
            let route_scale = moe_cfg.routed_scaling_factor.unwrap_or(1.0) as f32;
            V4Router::Score {
                gate,
                bias,
                topk: moe_cfg.num_experts_per_tok,
                route_scale,
            }
        };

        // Build the packed expert tensors directly in their final buffers.  Keeping
        // every per-expert tensor alive, stacking gate/up separately, and then
        // concatenating them needs roughly three copies of an MoE layer at peak.
        // V4 only has enough headroom for its final weights on two H800s, so that
        // transient allocation can OOM near the end of model loading.
        let local_intermediate = moe_cfg.moe_intermediate_size / comm.world_size();
        let packed_hidden = cfg.hidden_size / 2;
        let hidden_scales = cfg.hidden_size / 32;
        let packed_intermediate = local_intermediate / 2;
        let intermediate_scales = local_intermediate / 32;
        let device = match &experts_vb.0 {
            Either::Left(vb) => vb.device(),
            _ => candle_core::bail!("FusedMoeMxfp4: GGUF loading not supported for MXFP4"),
        };
        let gate_up_blocks = Tensor::zeros(
            (num_experts, 2 * local_intermediate, packed_hidden),
            DType::U8,
            device,
        )?;
        let gate_up_scales = Tensor::zeros(
            (num_experts, 2 * local_intermediate, hidden_scales),
            DType::F8E8M0,
            device,
        )?;
        let down_blocks = Tensor::zeros(
            (num_experts, cfg.hidden_size, packed_intermediate),
            DType::U8,
            device,
        )?;
        let down_scales = Tensor::zeros(
            (num_experts, cfg.hidden_size, intermediate_scales),
            DType::F8E8M0,
            device,
        )?;

        let gate_up_block_expert_elems = 2 * local_intermediate * packed_hidden;
        let gate_up_scale_expert_elems = 2 * local_intermediate * hidden_scales;
        let one_proj_block_elems = local_intermediate * packed_hidden;
        let one_proj_scale_elems = local_intermediate * hidden_scales;
        let down_block_expert_elems = cfg.hidden_size * packed_intermediate;
        let down_scale_expert_elems = cfg.hidden_size * intermediate_scales;

        match &experts_vb.0 {
            Either::Left(vb) => {
                for i in 0..num_experts {
                    let expert_vb = vb.pp(i.to_string());
                    let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);

                    let gate_proj_vb = expert_vb.pp(gate_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&gate_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&gate_proj_vb);

                    let gate_b = Self::load_mxfp4_packed(
                        &gate_proj_vb,
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    let gate_s = Self::load_mxfp4_scale(
                        &gate_proj_vb,
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                        scale_name,
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &gate_up_blocks,
                        &gate_b,
                        i * gate_up_block_expert_elems,
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &gate_up_scales,
                        &gate_s,
                        i * gate_up_scale_expert_elems,
                    )?;

                    let up_proj_vb = expert_vb.pp(up_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&up_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&up_proj_vb);

                    let up_b = Self::load_mxfp4_packed(
                        &up_proj_vb,
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    let up_s = Self::load_mxfp4_scale(
                        &up_proj_vb,
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                        scale_name,
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &gate_up_blocks,
                        &up_b,
                        i * gate_up_block_expert_elems + one_proj_block_elems,
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &gate_up_scales,
                        &up_s,
                        i * gate_up_scale_expert_elems + one_proj_scale_elems,
                    )?;

                    let down_proj_vb = expert_vb.pp(down_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&down_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&down_proj_vb);

                    let down_b = Self::load_mxfp4_packed(
                        &down_proj_vb,
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                        packed_name,
                        shard(1, comm.rank(), comm.world_size()),
                    )?;
                    let down_s = Self::load_mxfp4_scale(
                        &down_proj_vb,
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 32),
                        scale_name,
                        shard(1, comm.rank(), comm.world_size()),
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &down_blocks,
                        &down_b,
                        i * down_block_expert_elems,
                    )?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &down_scales,
                        &down_s,
                        i * down_scale_expert_elems,
                    )?;
                }
            }
            _ => candle_core::bail!("FusedMoeMxfp4: GGUF loading not supported for MXFP4"),
        }
        let w_size_n = local_intermediate;

        let mut swiglu_limit = 0.0f32;
        if let Some(extra) = &cfg.extra_config_json {
            if let Ok(extra) = serde_json::from_str::<serde_json::Value>(extra) {
                swiglu_limit = extra
                    .get("swiglu_limit")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
            }
        }

        Ok(Self {
            router,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
            w_size_n,
            act: cfg.hidden_act,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            swiglu_limit,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        self.forward_with_ids(xs, None, is_prefill)
    }

    pub fn forward_with_ids(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        self.forward_with_ids_f32(xs, input_ids, is_prefill)?
            .to_dtype(self.dtype)
    }

    /// Routed-expert output with the FP32 route reduction and collective kept
    /// intact. DeepSeek V4 adds the BF16 shared expert to this tensor in FP32
    /// and rounds only once at the FFN/HC boundary.
    pub fn forward_with_ids_f32(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let topk = self.router.num_experts_per_tok();
        let (topk_weights, topk_ids) = self.router.route(xs, input_ids, is_prefill)?;

        let moe_dtype = if self.dtype == DType::F32 {
            DType::BF16
        } else {
            self.dtype
        };
        let xs = if xs.dtype() != moe_dtype {
            xs.to_dtype(moe_dtype)?
        } else {
            xs.clone()
        }
        .contiguous()?;

        // DeepSeek V4's routed experts are not ordinary weight-only MXFP4
        // GEMMs.  The checkpoint uses FP4 weights with the official FP8
        // E4M3/UE8M0 activation quantizer (128 values per activation scale).
        // Keep the quantize/dequantize result in BF16 here: the generic
        // MXFP4 GEMM can then consume it without a host round-trip, while
        // preserving the model's activation quantization semantics on SM90.
        let xs = if moe_dtype == DType::BF16 {
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16(&xs, 1, hidden_dim, 0, 128)?
        } else {
            xs
        };

        let gate_up = moe::moe_gemm_mxfp4(
            &xs,
            &self.gate_up_blocks,
            &self.gate_up_scales,
            None,
            &topk_ids,
            is_prefill,
            None,
        )?;

        let down_inputs = if matches!(self.act, Activation::Silu) && self.swiglu_limit > 0.0 {
            let gate_up_f32 = gate_up.to_dtype(DType::F32)?;
            let flat_gate_up = gate_up_f32.reshape((num_tokens * topk, self.w_size_n * 2))?;
            let gate = flat_gate_up
                .narrow(D::Minus1, 0, self.w_size_n)?
                .minimum(self.swiglu_limit)?;
            let up = flat_gate_up
                .narrow(D::Minus1, self.w_size_n, self.w_size_n)?
                .clamp(-self.swiglu_limit, self.swiglu_limit)?;
            let flat_down = (candle_nn::ops::silu(&gate)? * up)?;
            flat_down.reshape((num_tokens, topk, self.w_size_n))?
        } else {
            gated_activation(&gate_up, self.w_size_n, &self.act)?
        };
        let down_inputs = if down_inputs.dtype() != moe_dtype {
            down_inputs.to_dtype(moe_dtype)?
        } else {
            down_inputs
        }
        .contiguous()?;

        // W2 sees the unweighted SwiGLU output.  Route weights are applied
        // after W2; applying them before this quantizer changes the FP8
        // activation scale per route and is not algebraically equivalent.
        // The same official FP8 activation quantizer is used for W2, over the
        // local tensor-parallel slice (which is 128-aligned).
        let down_inputs = if moe_dtype == DType::BF16 {
            let flat_down = down_inputs.reshape((num_tokens * topk, self.w_size_n))?;
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16(
                &flat_down,
                1,
                self.w_size_n,
                0,
                128,
            )?
            .reshape((num_tokens, topk, self.w_size_n))?
        } else {
            down_inputs
        };

        let mut ys = moe::moe_gemm_mxfp4(
            &down_inputs,
            &self.down_blocks,
            &self.down_scales,
            None,
            &topk_ids,
            is_prefill,
            None,
        )?
        .reshape((num_tokens, topk, hidden_dim))?
        .to_dtype(DType::F32)?
        .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

/// DeepSeek-V4 2-bit MoE (optional FP4 restore). Uses the same V4Router.
pub struct FusedMoeW2 {
    router: V4Router,
    /// [E, N*K/4] U8 row-major 2-bit planes for fused gate+up (N = 2*inter)
    gate_up_planes: Tensor,
    /// [E, N*(K/32)] U8 UE8M0 scales
    gate_up_scales: Tensor,
    /// [E, H*(I/4)] U8 planes for down
    down_planes: Tensor,
    down_scales: Tensor,
    gate_up_n: usize,
    gate_up_k: usize,
    down_n: usize,
    down_k: usize,
    act: Activation,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    swiglu_limit: f32,
    restore_mxfp4: Option<FusedMoeMxfp4>,
    restore_gate_enabled: bool,
    restore_gate_tau: f32,
}

impl FusedMoeW2 {
    /// Build W2 experts by GPU-side re-quant of MXFP4/e2m1 expert tensors.
    /// No host D2H pack — uses `attention_rs::moe_w2::moe_w2_pack_from_mxfp4`.
    pub fn from_mxfp4(mx: FusedMoeMxfp4, _device: &Device) -> Result<Self> {
        let restore_gate_enabled = std::env::var("XINFER_MOE_W2_GATE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let restore_gate_tau = std::env::var("XINFER_MOE_W2_GATE_TAU")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.60);
        let restore_mxfp4 = FusedMoeMxfp4 {
            router: match &mx.router {
                V4Router::Hash {
                    gate_weight,
                    tid2eid,
                    n_experts,
                    topk,
                    route_scale,
                } => V4Router::Hash {
                    gate_weight: gate_weight.clone(),
                    tid2eid: tid2eid.clone(),
                    n_experts: *n_experts,
                    topk: *topk,
                    route_scale: *route_scale,
                },
                V4Router::Score {
                    gate,
                    bias,
                    topk,
                    route_scale,
                } => V4Router::Score {
                    gate: gate.clone(),
                    bias: bias.clone(),
                    topk: *topk,
                    route_scale: *route_scale,
                },
            },
            gate_up_blocks: mx.gate_up_blocks.clone(),
            gate_up_scales: mx.gate_up_scales.clone(),
            down_blocks: mx.down_blocks.clone(),
            down_scales: mx.down_scales.clone(),
            w_size_n: mx.w_size_n,
            act: mx.act.clone(),
            all_reduce: mx.all_reduce.clone(),
            world_size: mx.world_size,
            dtype: mx.dtype,
            swiglu_limit: mx.swiglu_limit,
        };

        let e = mx.gate_up_blocks.dim(0)?;
        let n = mx.gate_up_blocks.dim(1)?;
        let k_packed = mx.gate_up_blocks.dim(2)?; // K/2
        let k = k_packed * 2;

        let gu_blocks = mx.gate_up_blocks.contiguous()?;
        let gu_scales = mx.gate_up_scales.contiguous()?;
        // Accept [E,N,K/32] or [E, N*(K/32)]
        let gu_scales = match gu_scales.dims().len() {
            3 => gu_scales,
            2 => gu_scales.reshape((e, n, k / 32))?,
            _ => candle_core::bail!(
                "FusedMoeW2: unexpected gate_up_scales dims {:?}",
                gu_scales.dims()
            ),
        };

        let (gate_up_planes, gate_up_scales) =
            attention_rs::moe_w2::moe_w2_pack_from_mxfp4(&gu_blocks, &gu_scales, n, k)?;

        let dn_n = mx.down_blocks.dim(1)?;
        let dn_k_packed = mx.down_blocks.dim(2)?;
        let dn_k = dn_k_packed * 2;
        let dn_blocks = mx.down_blocks.contiguous()?;
        let dn_scales = mx.down_scales.contiguous()?;
        let dn_scales = match dn_scales.dims().len() {
            3 => dn_scales,
            2 => dn_scales.reshape((e, dn_n, dn_k / 32))?,
            _ => candle_core::bail!(
                "FusedMoeW2: unexpected down_scales dims {:?}",
                dn_scales.dims()
            ),
        };

        let (down_planes, down_scales) =
            attention_rs::moe_w2::moe_w2_pack_from_mxfp4(&dn_blocks, &dn_scales, dn_n, dn_k)?;

        Ok(Self {
            router: mx.router,
            gate_up_planes,
            gate_up_scales: gate_up_scales.reshape((e, n * (k / 32)))?,
            down_planes,
            down_scales: down_scales.reshape((e, dn_n * (dn_k / 32)))?,
            gate_up_n: n,
            gate_up_k: k,
            down_n: dn_n,
            down_k: dn_k,
            act: mx.act,
            all_reduce: mx.all_reduce,
            world_size: mx.world_size,
            dtype: mx.dtype,
            swiglu_limit: if mx.swiglu_limit > 0.0 {
                mx.swiglu_limit
            } else {
                10.0
            },
            restore_mxfp4: restore_gate_enabled.then_some(restore_mxfp4),
            restore_gate_enabled,
            restore_gate_tau,
        })
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let mx = FusedMoeMxfp4::new(cfg, vb, comm, dtype)?;
        let device = mx.gate_up_blocks.device().clone();
        let mut out = Self::from_mxfp4(mx, &device)?;
        if let Some(extra) = &cfg.extra_config_json {
            if let Ok(extra) = serde_json::from_str::<serde_json::Value>(extra) {
                out.swiglu_limit = extra
                    .get("swiglu_limit")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(out.swiglu_limit as f64) as f32;
            }
        }
        Ok(out)
    }

    fn forward_fp4_restore(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let Some(restore) = self.restore_mxfp4.as_ref() else {
            return self.forward_with_ids_2bit(xs, input_ids, is_prefill);
        };
        restore.forward_with_ids(xs, input_ids, is_prefill)
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        self.forward_with_ids(xs, None, is_prefill)
    }

    pub fn forward_with_ids(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        if self.restore_gate_enabled && !is_prefill && self.restore_mxfp4.is_some() {
            let (topk_weights, _) = self.router.route(xs, input_ids, is_prefill)?;
            let max_prob = topk_weights.max(1)?.max(0)?.to_vec0::<f32>()?;
            if max_prob <= self.restore_gate_tau {
                return self.forward_fp4_restore(xs, input_ids, is_prefill);
            }
        }
        self.forward_with_ids_2bit(xs, input_ids, is_prefill)
    }

    fn forward_with_ids_2bit(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let topk = self.router.num_experts_per_tok();
        let (topk_weights, topk_ids) = self.router.route(xs, input_ids, is_prefill)?;

        let moe_dtype = if self.dtype == DType::F32 {
            DType::BF16
        } else {
            self.dtype
        };
        let xs = if xs.dtype() != moe_dtype {
            xs.to_dtype(moe_dtype)?
        } else {
            xs.clone()
        };

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let gate_up = moe::moe_gemm_w2(
            &xs,
            &self.gate_up_planes,
            &self.gate_up_scales,
            &None,
            &sorted_token_ids,
            &expert_ids,
            topk,
            self.gate_up_n,
            self.gate_up_k,
            is_prefill,
        )?;

        let down_inputs = if matches!(self.act, Activation::Silu) {
            attention_rs::moe_w2::moe_w2_swiglu_clamp_bf16(
                &gate_up,
                self.gate_up_n / 2,
                self.swiglu_limit,
            )?
        } else {
            gated_activation(&gate_up, self.gate_up_n / 2, &self.act)?
        };
        let down_inputs = if down_inputs.dtype() != moe_dtype {
            down_inputs.to_dtype(moe_dtype)?
        } else {
            down_inputs
        };

        let mut ys = moe::moe_gemm_w2(
            &down_inputs,
            &self.down_planes,
            &self.down_scales,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            topk,
            self.down_n,
            self.down_k,
            is_prefill,
        )?
        .reshape((num_tokens, topk, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }

    /// Max sqrt-softplus router score — used by confidence gate for FP4 restore.
    pub fn max_router_prob(router_logits: &Tensor) -> Result<f32> {
        let scores = stable_softplus(router_logits)?.sqrt()?;
        let mx = scores.max_all()?.to_scalar::<f32>()?;
        Ok(mx)
    }
}
