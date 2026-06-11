// src/models/layers/moe.rs
use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::config::QuantConfig;
use attention_rs::moe;
use attention_rs::moe::moe_gemm_fp8;
use attention_rs::silu_and_mul::silu_and_mul;
use attention_rs::sort::ArgSortOp;
use candle_core::Module;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Tensor, D,
};
use candle_nn::var_builder::Shard;
use candle_nn::Activation;
use either::Either;
use std::rc::Rc;
use std::sync::Arc;

/// Apply gated activation on fused gate_up tensor.
/// Uses optimized `silu_and_mul` kernel for SiLU activation, falls back to
/// generic tensor operations for other activations (e.g., GeluPytorchTanh).
fn gated_activation(gate_up: &Tensor, half_dim: usize, act: &Activation) -> Result<Tensor> {
    if matches!(act, Activation::Silu) {
        silu_and_mul(gate_up, half_dim)
    } else {
        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, half_dim)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, half_dim, half_dim)?
            .contiguous()?;
        (up * gate.apply(act)?)?.contiguous()
    }
}

/// Shared MoE routing config extracted from MoEConfig at construction time.
#[derive(Clone, Debug)]
pub struct MoeRouting {
    pub e_score_correction_bias: Option<Tensor>,
    pub use_sigmoid_scoring: bool,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: Option<f64>,
    pub num_experts_per_tok: usize,
}

impl MoeRouting {
    /// Build routing config from the MoE config section.
    pub fn from_moe_cfg(cfg: &crate::utils::config::MoEConfig, bias: Option<Tensor>) -> Self {
        let use_sigmoid = cfg.topk_method.as_deref().is_some_and(|m| m == "noaux_tc")
            || cfg.scoring_func.as_deref().is_some_and(|s| s == "sigmoid");
        Self {
            e_score_correction_bias: bias,
            use_sigmoid_scoring: use_sigmoid,
            n_group: cfg.n_group.unwrap_or(1),
            topk_group: cfg.topk_group.unwrap_or(1),
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            num_experts_per_tok: cfg.num_experts_per_tok,
        }
    }

    /// Route tokens to experts, returning `(topk_weights, topk_ids)`.
    /// `router_logits` must be F32 with shape `[num_tokens, num_experts]`.
    pub fn route(&self, router_logits: &Tensor, is_prefill: bool) -> Result<(Tensor, Tensor)> {
        let (mut topk_weights, topk_ids) = if self.use_sigmoid_scoring {
            let scores = candle_nn::ops::sigmoid(router_logits)?;

            let scores_for_choice = if let Some(bias) = &self.e_score_correction_bias {
                scores.broadcast_add(&bias.to_dtype(DType::F32)?)?
            } else {
                scores.clone()
            };

            let topk_indices = if self.n_group > 1 {
                let num_tokens = scores_for_choice.dim(0)?;
                let num_experts = scores_for_choice.dim(1)?;
                if num_experts % self.n_group != 0 {
                    candle_core::bail!(
                        "MoE routing requires num_experts ({num_experts}) divisible by n_group ({})",
                        self.n_group
                    );
                }
                if self.topk_group > self.n_group {
                    candle_core::bail!(
                        "MoE routing requires topk_group ({}) <= n_group ({})",
                        self.topk_group,
                        self.n_group
                    );
                }
                let experts_per_group = num_experts / self.n_group;
                if experts_per_group * self.topk_group < self.num_experts_per_tok {
                    candle_core::bail!(
                        "MoE routing selected-group capacity ({}) is smaller than num_experts_per_tok ({})",
                        experts_per_group * self.topk_group,
                        self.num_experts_per_tok
                    );
                }
                // [num_tokens, n_group, experts_per_group]
                let grouped =
                    scores_for_choice.reshape((num_tokens, self.n_group, experts_per_group))?;
                // top-2 per group summed -> [num_tokens, n_group]
                let top2_idx = select_topk_indices(&grouped, experts_per_group.min(2), is_prefill)?;
                let top2_vals = grouped.gather(&top2_idx, D::Minus1)?;
                let group_scores = top2_vals.sum(D::Minus1)?;
                // top topk_group groups -> [num_tokens, topk_group]
                let group_idx = select_topk_indices(&group_scores, self.topk_group, is_prefill)?;
                // build group mask [num_tokens, n_group]
                let group_mask = group_scores.zeros_like()?.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(DType::F32)?,
                    1,
                )?;
                // expand to per-expert mask [num_tokens, num_experts]
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as((num_tokens, self.n_group, experts_per_group))?
                    .reshape((num_tokens, num_experts))?;
                let masked = scores_for_choice.broadcast_mul(&score_mask)?;
                select_topk_indices(&masked, self.num_experts_per_tok, is_prefill)?
            } else {
                select_topk_indices(&scores_for_choice, self.num_experts_per_tok, is_prefill)?
            };

            let topk_weights = scores.gather(&topk_indices, D::Minus1)?;
            let topk_ids = topk_indices.to_dtype(DType::U32)?;
            (topk_weights, topk_ids)
        } else {
            let mut logits = router_logits.clone();
            if let Some(bias) = &self.e_score_correction_bias {
                logits = logits.broadcast_add(&bias.to_dtype(DType::F32)?)?;
            }
            attention_rs::topk::topk_softmax(&logits, self.num_experts_per_tok)?
        };

        if self.norm_topk_prob {
            // let denom = (topk_weights.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * factor)?;
        }

        Ok((topk_weights, topk_ids))
    }
}

fn select_topk_indices(scores: &Tensor, topk: usize, is_prefill: bool) -> Result<Tensor> {
    let sorted_idx = if is_prefill {
        scores.contiguous()?.arg_sort(false)?
    } else {
        scores.arg_sort_last_dim(false)?
    };
    sorted_idx.narrow(D::Minus1, 0, topk)?.contiguous()
}

fn sort_expert_assignments(topk_ids: &Tensor, is_prefill: bool) -> Result<(Tensor, Tensor)> {
    let flat = topk_ids.flatten_all()?;
    if is_prefill {
        flat.sort(true)
    } else {
        flat.sort_last_dim(true)
    }
}

fn presorted_expert_assignments(
    topk_ids: &Tensor,
    is_prefill: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    if !is_prefill {
        return Ok(None);
    }
    let (expert_ids, sorted_token_ids) = sort_expert_assignments(topk_ids, true)?;
    Ok(Some((sorted_token_ids, expert_ids)))
}

/// Try to load `e_score_correction_bias` from the MoE var-builder.
/// First tries `gate.e_score_correction_bias` (Qwen style), then falls back
/// to `e_score_correction_bias` directly (MiniMax style).
fn try_load_e_score_correction_bias(vb: &VarBuilderX, num_experts: usize) -> Option<Tensor> {
    let vb_gate = vb.pp("gate");
    vb_gate
        .get_with_hints_dtype(
            num_experts,
            "e_score_correction_bias",
            shard(0, 0, 1),
            DType::F32,
        )
        .ok()
        .or_else(|| {
            vb.get_with_hints_dtype(
                num_experts,
                "e_score_correction_bias",
                shard(0, 0, 1),
                DType::F32,
            )
            .ok()
        })
}

#[derive(Clone, Copy, Debug)]
enum PackedGateUpLayout {
    // [experts, hidden, 2*intermediate]
    HiddenPacked,
    // [experts, 2*intermediate, hidden]
    InterPacked,
}

#[derive(Clone, Copy, Debug)]
enum PackedDownLayout {
    // [experts, intermediate, hidden] -> transpose to [experts, hidden, intermediate]
    InterHidden,
    // [experts, hidden, intermediate] -> already in expected GEMM layout.
    HiddenInter,
}

/// Resolve per-expert projection sub-prefix. Falls back from standard
/// `gate_proj`/`up_proj`/`down_proj` to MiniMax-style `w1`/`w3`/`w2`.
fn resolve_expert_proj_prefix(
    expert_vb: &candle_nn::var_builder::ShardedVarBuilder,
) -> (&'static str, &'static str, &'static str) {
    if expert_vb.contains_tensor("gate_proj.weight")
        || expert_vb.contains_tensor("gate_proj.weight_packed")
        || expert_vb.contains_tensor("gate_proj.blocks")
    {
        ("gate_proj", "up_proj", "down_proj")
    } else {
        ("w1", "w3", "w2")
    }
}

fn resolve_expert_proj_prefix_x(
    expert_vb: &VarBuilderX,
) -> (&'static str, &'static str, &'static str) {
    if expert_vb.has_key("gate_proj.weight")
        || expert_vb.has_key("gate_proj.weight_packed")
        || expert_vb.has_key("gate_proj.blocks")
    {
        ("gate_proj", "up_proj", "down_proj")
    } else {
        ("w1", "w3", "w2")
    }
}

fn resolve_packed_gate_up_layout(cfg: &Config) -> Result<PackedGateUpLayout> {
    let arch = cfg
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("");

    // Qwen3.5 MoE / Qwen3Next checkpoints store gate_up as [experts, 2*intermediate, hidden].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
            | "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
    ) {
        return Ok(PackedGateUpLayout::InterPacked);
    }

    let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
    if cfg.hidden_size == moe_cfg.moe_intermediate_size * 2 {
        candle_core::bail!(
            "Ambiguous packed gate_up_proj layout for arch {:?}: hidden_size ({}) == 2 * moe_intermediate_size ({}). \
Please add an architecture-specific layout mapping.",
            arch,
            cfg.hidden_size,
            moe_cfg.moe_intermediate_size
        );
    }

    Ok(PackedGateUpLayout::HiddenPacked)
}

fn resolve_packed_down_layout(cfg: &Config) -> PackedDownLayout {
    let arch = cfg
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("");

    // Qwen3.5 MoE / Qwen3Next checkpoints store down_proj as [experts, hidden, intermediate].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
            | "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
    ) {
        PackedDownLayout::HiddenInter
    } else {
        PackedDownLayout::InterHidden
    }
}

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    gate_dtype: DType,
}

impl FusedMoe {
    pub fn load_packed(
        cfg: &Config,
        experts_vb: VarBuilderX,
        comm: Rc<Comm>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        let (gate_experts, up_experts, down_experts) = if experts_vb.has_key("gate_up_proj") {
            match &experts_vb.0 {
                Either::Left(vb) => {
                    let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
                    let (gate_expert, up_expert) = match gate_up_layout {
                        // [experts, hidden, 2*intermediate]
                        PackedGateUpLayout::HiddenPacked => {
                            let gate = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        cfg.hidden_size,
                                        moe_cfg.moe_intermediate_size * 2,
                                    ),
                                    "gate_up_proj",
                                    shard(2, comm.rank(), comm.world_size() * 2),
                                )?
                                .t()?
                                .contiguous()?;

                            let up = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        cfg.hidden_size,
                                        moe_cfg.moe_intermediate_size * 2,
                                    ),
                                    "gate_up_proj",
                                    shard(
                                        2,
                                        comm.rank() + comm.world_size(),
                                        comm.world_size() * 2,
                                    ),
                                )?
                                .t()?
                                .contiguous()?;
                            (gate, up)
                        }
                        // [experts, 2*intermediate, hidden]
                        PackedGateUpLayout::InterPacked => {
                            let gate = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        moe_cfg.moe_intermediate_size * 2,
                                        cfg.hidden_size,
                                    ),
                                    "gate_up_proj",
                                    shard(1, comm.rank(), comm.world_size() * 2),
                                )?
                                .contiguous()?;
                            let up = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        moe_cfg.moe_intermediate_size * 2,
                                        cfg.hidden_size,
                                    ),
                                    "gate_up_proj",
                                    shard(
                                        1,
                                        comm.rank() + comm.world_size(),
                                        comm.world_size() * 2,
                                    ),
                                )?
                                .contiguous()?;
                            (gate, up)
                        }
                    };
                    let down_layout = resolve_packed_down_layout(cfg);
                    let down_expert = match down_layout {
                        PackedDownLayout::InterHidden => vb
                            .get_with_hints(
                                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "down_proj",
                                shard(1, comm.rank(), comm.world_size()),
                            )?
                            .t()?
                            .contiguous()?,
                        PackedDownLayout::HiddenInter => vb
                            .get_with_hints(
                                (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "down_proj",
                                shard(2, comm.rank(), comm.world_size()),
                            )?
                            .contiguous()?,
                    };
                    let (_, gate_n, gate_k) = gate_expert.dims3()?;
                    let (_, up_n, up_k) = up_expert.dims3()?;
                    let (_, down_n, down_k) = down_expert.dims3()?;
                    if gate_n != up_n
                        || gate_k != up_k
                        || gate_k != cfg.hidden_size
                        || down_n != cfg.hidden_size
                        || down_k != gate_n
                    {
                        candle_core::bail!(
                            "Invalid packed MoE tensor shapes after loading: gate={:?}, up={:?}, down={:?}, hidden_size={}, arch={:?}. \
This usually means packed down_proj / gate_up_proj layout was interpreted incorrectly.",
                            gate_expert.shape(),
                            up_expert.shape(),
                            down_expert.shape(),
                            cfg.hidden_size,
                            cfg.architectures
                        );
                    }
                    (gate_expert, up_expert, down_expert)
                }
                _ => candle_core::bail!("invalid varbuild or quant config!"),
            }
        } else {
            for i in 0..num_experts {
                let experts_vb = experts_vb.pp(format!("{}", i).as_str());
                match &experts_vb.0 {
                    Either::Left(vb) => {
                        let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(vb);
                        // n x k format
                        let gate_expert = vb.pp(gate_name).get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let up_expert = vb.pp(up_name).get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let down_expert = vb.pp(down_name).get_with_hints(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                            shard(1, comm.rank(), comm.world_size()),
                        )?;
                        gate_experts.push(gate_expert);
                        up_experts.push(up_expert);
                        down_experts.push(down_expert);
                    }
                    _ => candle_core::bail!("invalid varbuild or quant config!"),
                }
            }
            (
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&down_experts, 0)?,
            )
        };
        Ok((gate_experts, up_experts, down_experts))
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

        assert!(
            cfg.quantization_config.is_none(),
            "Invalid quantization format!"
        );
        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            gate_dtype,
        )?;

        let (gate_w, up_w, down_w) = Self::load_packed(cfg, experts_vb, comm.clone())?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 1)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(
                moe_cfg,
                try_load_e_score_correction_bias(bias_vb, num_experts),
            ),
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
            gate_dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            std::borrow::Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let topk = self.routing.num_experts_per_tok;
        let gate_up = moe::moe_gemm(
            &xs,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            topk,
            is_prefill,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let mut ys = moe::moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            topk,
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

pub struct FusedMoeGGUF {
    gate: Linear,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    act: candle_nn::Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeGGUF {
    pub fn new_repack(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let gate_ws = match &vb.pp("ffn_gate_inp").0 {
            Either::Right(v) => v
                .get((num_experts, cfg.hidden_size), "weight")?
                .dequantize(v.device())?,
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let gate = Linear::new(gate_ws, None, &None)?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => {
                let packed_result = v.pp("ffn_gate_up_exps").get(
                    (
                        num_experts,
                        moe_cfg.moe_intermediate_size * 2,
                        cfg.hidden_size,
                    ),
                    "weight",
                );
                if let Ok(packed) = packed_result {
                    let deq = packed.dequantize_f16(&v.device())?;
                    let gate_exp = deq
                        .narrow(1, 0, moe_cfg.moe_intermediate_size)?
                        .contiguous()?;
                    let up_exp = deq
                        .narrow(
                            1,
                            moe_cfg.moe_intermediate_size,
                            moe_cfg.moe_intermediate_size,
                        )?
                        .contiguous()?;
                    let down = v
                        .pp("ffn_down_exps")
                        .get(
                            (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                        )?
                        .dequantize_f16(&v.device())?;
                    (gate_exp, up_exp, down)
                } else {
                    (
                        v.pp("ffn_gate_exps")
                            .get(
                                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                            )?
                            .dequantize_f16(&v.device())?,
                        v.pp("ffn_up_exps")
                            .get(
                                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                            )?
                            .dequantize_f16(&v.device())?,
                        v.pp("ffn_down_exps")
                            .get(
                                (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "weight",
                            )?
                            .dequantize_f16(&v.device())?,
                    )
                }
            }
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let (ggml_dtype, block_size) = (GgmlDType::Q4K, GgmlDType::Q4K.block_size());

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

        let cur_chunk_size = if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
            < moe_cfg.moe_intermediate_size
        {
            moe_intermediate_chunk
        } else {
            moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk
        };

        assert!(cur_chunk_size > 0 && cur_chunk_size % block_size == 0,
            "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
            \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
            moe_cfg.moe_intermediate_size,
            comm.world_size(),
            block_size
        );

        let (gate_experts, up_experts, down_experts) = (
            gate_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            up_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            down_experts
                .narrow(2, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
        );
        let gate_experts = Arc::new(QTensor::quantize(&gate_experts, ggml_dtype)?);
        let up_experts = Arc::new(QTensor::quantize(&up_experts, ggml_dtype)?);
        let down_experts = Arc::new(QTensor::quantize(&down_experts, GgmlDType::Q8_0)?);

        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, None),
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        if comm.world_size() > 1 {
            return Self::new_repack(cfg, vb, comm.clone(), dtype);
        }
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let gate_ws = match &vb.pp("ffn_gate_inp").0 {
            Either::Right(v) => v
                .get((num_experts, cfg.hidden_size), "weight")?
                .dequantize(v.device())?,
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let gate = Linear::new(gate_ws, None, &None)?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => {
                let packed_result = v.pp("ffn_gate_up_exps").get(
                    (
                        num_experts,
                        moe_cfg.moe_intermediate_size * 2,
                        cfg.hidden_size,
                    ),
                    "weight",
                );
                if let Ok(packed) = packed_result {
                    let orig_dtype = packed.dtype();
                    let deq = packed.dequantize_f16(&v.device())?;
                    let gate_exp = deq
                        .narrow(1, 0, moe_cfg.moe_intermediate_size)?
                        .contiguous()?;
                    let up_exp = deq
                        .narrow(
                            1,
                            moe_cfg.moe_intermediate_size,
                            moe_cfg.moe_intermediate_size,
                        )?
                        .contiguous()?;
                    let down = v.pp("ffn_down_exps").get(
                        (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        "weight",
                    )?;
                    (
                        Arc::new(QTensor::quantize(&gate_exp, orig_dtype)?),
                        Arc::new(QTensor::quantize(&up_exp, orig_dtype)?),
                        down,
                    )
                } else {
                    (
                        v.pp("ffn_gate_exps").get(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                        )?,
                        v.pp("ffn_up_exps").get(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                        )?,
                        v.pp("ffn_down_exps").get(
                            (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                        )?,
                    )
                }
            }
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, None),
            all_reduce: AllReduce::new(comm),
            world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;
        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;

            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeISQ {
    gate: Linear,
    gate_experts: QTensor,
    up_experts: QTensor,
    down_experts: QTensor,
    act: candle_nn::Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeISQ {
    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let mut quant_type = match cfg.quant.as_ref().unwrap().as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let get_moe_intermediate_chunk = |blk_size: usize| -> usize {
            let base = moe_cfg.moe_intermediate_size / comm.world_size();
            if base % blk_size != 0 {
                ((base + blk_size - 1) / blk_size) * blk_size
            } else {
                base
            }
        };

        let mut block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            //in case of the experts unable to be split under qkk format,
            //and asymetric split also not workable, switch to q8_0
            let chunk = get_moe_intermediate_chunk(block_size);
            if (moe_cfg.moe_intermediate_size - chunk) % (comm.world_size() - 1) != 0 {
                quant_type = GgmlDType::Q8_0;
                block_size = quant_type.block_size();
            }
        }
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let (gate_experts, up_experts, down_experts) = if moe_cfg.moe_intermediate_size
            / comm.world_size()
            % block_size
            == 0
        {
            FusedMoe::load_packed(cfg, vb.pp("experts"), comm.clone())?
        } else {
            let experts_vb = vb.pp("experts");
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);

            let moe_intermediate_chunk = get_moe_intermediate_chunk(block_size);

            let (gate_experts, up_experts, down_experts) = if experts_vb.has_key("gate_up_proj") {
                match &experts_vb.0 {
                    Either::Left(vb) => {
                        let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
                        let (gate_expert, up_expert) = match gate_up_layout {
                            // [experts, hidden, 2*intermediate]
                            PackedGateUpLayout::HiddenPacked => {
                                let gate = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            cfg.hidden_size,
                                            moe_cfg.moe_intermediate_size * 2,
                                        ),
                                        "gate_up_proj",
                                        shard(2, 0, 2),
                                    )?
                                    .t()?
                                    .contiguous()?;
                                let up = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            cfg.hidden_size,
                                            moe_cfg.moe_intermediate_size * 2,
                                        ),
                                        "gate_up_proj",
                                        shard(2, 1, 2),
                                    )?
                                    .t()?
                                    .contiguous()?;
                                (gate, up)
                            }
                            // [experts, 2*intermediate, hidden]
                            PackedGateUpLayout::InterPacked => {
                                let gate = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            moe_cfg.moe_intermediate_size * 2,
                                            cfg.hidden_size,
                                        ),
                                        "gate_up_proj",
                                        shard(1, 0, 2),
                                    )?
                                    .contiguous()?;
                                let up = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            moe_cfg.moe_intermediate_size * 2,
                                            cfg.hidden_size,
                                        ),
                                        "gate_up_proj",
                                        shard(1, 1, 2),
                                    )?
                                    .contiguous()?;
                                (gate, up)
                            }
                        };

                        let down_expert = match vb.get_with_hints(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "down_proj",
                            Shard::default(),
                        ) {
                            // Layout A: [experts, intermediate, hidden] -> transpose to [experts, hidden, intermediate]
                            Ok(w) => w.t()?.contiguous()?,
                            // Layout B: [experts, hidden, intermediate] -> already in expected GEMM layout.
                            Err(_) => vb
                                .get_with_hints(
                                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                    "down_proj",
                                    Shard::default(),
                                )?
                                .contiguous()?,
                        };
                        (gate_expert, up_expert, down_expert)
                    }
                    Either::Right(_) => panic!("invalid varbuild!"),
                }
            } else {
                //pack experts
                for i in 0..num_experts {
                    let experts_vb = experts_vb.pp(format!("{}", i).as_str());
                    match &experts_vb.0 {
                        Either::Left(vb) => {
                            let gate_expert = vb.pp("gate_proj").get_with_hints(
                                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                                Shard::default(),
                            )?;
                            let up_expert = vb.pp("up_proj").get_with_hints(
                                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                                Shard::default(),
                            )?;
                            let down_expert = vb.pp("down_proj").get_with_hints(
                                (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "weight",
                                Shard::default(),
                            )?;
                            gate_experts.push(gate_expert);
                            up_experts.push(up_expert);
                            down_experts.push(down_expert);
                        }
                        Either::Right(_) => panic!("invalid varbuild!"),
                    }
                }

                (
                    Tensor::stack(&gate_experts, 0)?,
                    Tensor::stack(&up_experts, 0)?,
                    Tensor::stack(&down_experts, 0)?,
                )
            };

            let mut last_remain_size = moe_intermediate_chunk;
            if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
                < moe_cfg.moe_intermediate_size
            {
            } else {
                last_remain_size =
                    moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk;
                assert!(last_remain_size > 0 && last_remain_size % block_size == 0,
                    "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
                    \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
                    moe_cfg.moe_intermediate_size,
                    comm.world_size(),
                    block_size
                );
            };

            let gate_experts =
                gate_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let up_experts =
                up_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let down_experts =
                down_experts.narrow(2, comm.rank() * moe_intermediate_chunk, last_remain_size)?;

            (gate_experts, up_experts, down_experts)
        };

        let gate_experts = QTensor::quantize(&gate_experts, quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(
                moe_cfg,
                try_load_e_score_correction_bias(&vb, num_experts),
            ),
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
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

        let mut quant_type = match cfg.quant.as_ref().unwrap().as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            quant_type = GgmlDType::Q8_0;
        }

        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let (gate_experts, up_experts, down_experts) =
            FusedMoe::load_packed(cfg, experts_vb, comm.clone())?;

        let gate_experts = QTensor::quantize(&gate_experts, quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(
                moe_cfg,
                try_load_e_score_correction_bias(bias_vb, num_experts),
            ),
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;
        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.routing.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

#[allow(dead_code)]
pub struct FusedMoeFp8 {
    gate: Linear,
    gate_up_experts: Tensor,
    gate_up_experts_scale: Tensor,
    down_experts: Tensor,
    down_experts_scale: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    block_size: Vec<usize>,
    gate_dtype: DType,
}

impl FusedMoeFp8 {
    pub fn new(
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        Self::new_with_gate(
            cfg,
            vb.pp("gate"),
            vb.pp("experts"),
            &vb,
            comm,
            dtype,
            quant_cfg,
        )
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilderX,
        experts_vb: VarBuilderX,
        bias_vb: &VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("FusedMoeFp8: weight_block_size must have 2 elements");
        }
        let by = block_size[0]; // for scale_n
        let bx = block_size[1]; // for scale_k

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            gate_dtype,
        )?;

        let (
            gate_experts,
            gate_experts_scale,
            up_experts,
            up_experts_scale,
            down_experts,
            down_experts_scale,
        ) = if experts_vb.has_key("gate_up_proj") {
            // Qwen3 VL approach.
            match &experts_vb.0 {
                Either::Left(vb) => {
                    let gate_weight = vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let up_weight = vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank() + comm.world_size(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let scale_n = (cfg.hidden_size + by - 1) / by;
                    let scale_k = (moe_cfg.moe_intermediate_size * 2 + bx - 1) / bx;

                    let gate_up_scale = vb.get_with_hints_dtype(
                        (num_experts, scale_n, scale_k),
                        "gate_up_proj_scale_inv",
                        Default::default(),
                        DType::F32,
                    )?;

                    let inter_blocks = moe_cfg.moe_intermediate_size / bx;
                    let local_inter_blocks = inter_blocks / comm.world_size();
                    let start_blocks = comm.rank() * local_inter_blocks;

                    let gate_s_t = gate_up_scale.narrow(2, 0, inter_blocks)?.contiguous()?;
                    let up_s_t = gate_up_scale
                        .narrow(2, inter_blocks, inter_blocks)?
                        .contiguous()?;

                    let gate_s = gate_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;
                    let up_s = up_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;

                    let down_weight = vb
                        .get_with_hints_dtype(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "down_proj",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let down_s = vb
                        .get_with_hints_dtype(
                            (num_experts, scale_k / 2, scale_n),
                            "down_proj_scale_inv",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    (gate_weight, gate_s, up_weight, up_s, down_weight, down_s)
                }
                _ => candle_core::bail!("FusedMoeFp8: Invalid varbuilder for packed loading"),
            }
        } else {
            // Per-expert loading
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut gate_experts_scale = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut up_experts_scale = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let mut down_experts_scale = Vec::with_capacity(num_experts);
            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(format!("{}", i).as_str());
                let (gate_name, up_name, down_name) = resolve_expert_proj_prefix_x(&expert_vb);
                let gate_weight = expert_vb.pp(gate_name).get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let gate_s = match expert_vb.pp(gate_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(gate_name).get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let up_weight = expert_vb.pp(up_name).get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let up_s = match expert_vb.pp(up_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(up_name).get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let down_weight = expert_vb.pp(down_name).get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (cfg.hidden_size + by - 1) / by;
                let sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                let down_s = match expert_vb.pp(down_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(down_name).get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                gate_experts.push(gate_weight);
                gate_experts_scale.push(gate_s);
                up_experts.push(up_weight);
                up_experts_scale.push(up_s);
                down_experts.push(down_weight);
                down_experts_scale.push(down_s);
            }

            (
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&gate_experts_scale, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&up_experts_scale, 0)?,
                Tensor::stack(&down_experts, 0)?,
                Tensor::stack(&down_experts_scale, 0)?,
            )
        };
        let gate_up_experts = Tensor::cat(&[&gate_experts, &up_experts], 1)?;
        let gate_up_experts_scale = Tensor::cat(&[&gate_experts_scale, &up_experts_scale], 1)?;
        let w_size_n = gate_up_experts.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_experts,
            gate_up_experts_scale,
            down_experts,
            down_experts_scale,
            w_size_n,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(
                moe_cfg,
                try_load_e_score_correction_bias(bias_vb, num_experts),
            ),
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            block_size: vec![by, bx],
            gate_dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let gate_input = if xs.dtype() != self.gate_dtype {
            std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            std::borrow::Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(self.dtype)?
        } else {
            xs.clone()
        };

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let gate_up = moe_gemm_fp8(
            &xs,
            &self.gate_up_experts,
            &self.gate_up_experts_scale,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.routing.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let mut ys = moe_gemm_fp8(
            &down_inputs,
            &self.down_experts,
            &self.down_experts_scale,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.routing.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}

#[allow(dead_code)]
pub struct FusedMoeMxfp4 {
    gate: Linear,
    gate_up_blocks: Tensor,
    gate_up_scales: Tensor,
    down_blocks: Tensor,
    down_scales: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    gate_dtype: DType,
}

impl FusedMoeMxfp4 {
    fn mxfp4_tensor_name_packed(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_packed") {
            "weight_packed"
        } else {
            "blocks"
        }
    }

    fn mxfp4_tensor_name_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_scale") {
            "weight_scale"
        } else {
            "scales"
        }
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

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &cfg.quantization_config,
            &None,
            gate_dtype,
        )?;

        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();

        match &experts_vb.0 {
            Either::Left(vb) => {
                for i in 0..num_experts {
                    let expert_vb = vb.pp(i.to_string());
                    let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);

                    let gate_proj_vb = expert_vb.pp(gate_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&gate_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&gate_proj_vb);

                    let gate_b = gate_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;
                    let gate_s = gate_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                        scale_name,
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;

                    let up_proj_vb = expert_vb.pp(up_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&up_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&up_proj_vb);

                    let up_b = up_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;
                    let up_s = up_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                        scale_name,
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;

                    let down_proj_vb = expert_vb.pp(down_name);
                    let packed_name = Self::mxfp4_tensor_name_packed(&down_proj_vb);
                    let scale_name = Self::mxfp4_tensor_name_scale(&down_proj_vb);

                    let down_b = down_proj_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                        packed_name,
                        shard(1, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;
                    let down_s = down_proj_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 32),
                        scale_name,
                        shard(1, comm.rank(), comm.world_size()),
                        DType::U8,
                    )?;

                    gate_blocks_vec.push(gate_b);
                    gate_scales_vec.push(gate_s);
                    up_blocks_vec.push(up_b);
                    up_scales_vec.push(up_s);
                    down_blocks_vec.push(down_b);
                    down_scales_vec.push(down_s);
                }
            }
            _ => candle_core::bail!("FusedMoeMxfp4: GGUF loading not supported for MXFP4"),
        }

        let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
        let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
        let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
        let up_scales = Tensor::stack(&up_scales_vec, 0)?;

        let gate_up_blocks = Tensor::cat(&[&gate_blocks, &up_blocks], 1)?;
        let gate_up_scales = Tensor::cat(&[&gate_scales, &up_scales], 1)?;
        let w_size_n = gate_up_blocks.dim(1)? / 2;

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;

        Ok(Self {
            gate,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
            w_size_n,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(
                moe_cfg,
                try_load_e_score_correction_bias(bias_vb, num_experts),
            ),
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            gate_dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let gate_input = if xs.dtype() != self.gate_dtype {
            std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            std::borrow::Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;

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

        let gate_up = moe::moe_gemm_mxfp4(
            &xs,
            &self.gate_up_blocks,
            &self.gate_up_scales,
            None,
            &topk_ids,
            is_prefill,
            None,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;
        let down_inputs = if down_inputs.dtype() != moe_dtype {
            down_inputs.to_dtype(moe_dtype)?
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
            Some(&topk_weights),
        )?
        .reshape((num_tokens, self.routing.num_experts_per_tok, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}

pub struct FusedMoeNvfp4 {
    gate: Linear,
    gate_up_blocks: Tensor,
    gate_up_scales: Tensor,
    gate_up_global_scales: Tensor,
    gate_up_input_scales: Tensor,
    gate_up_scales_swizzled: Option<Tensor>,
    down_blocks: Tensor,
    down_scales: Tensor,
    down_global_scales: Tensor,
    down_input_scales: Tensor,
    down_scales_swizzled: Option<Tensor>,
    w_size_n: usize,
    act: Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    apply_router_weight_on_input: bool,
    gate_dtype: DType,
}

impl FusedMoeNvfp4 {
    fn tensor_name_packed(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_packed") {
            "weight_packed"
        } else if vb.contains_tensor("weight") {
            "weight"
        } else {
            "blocks"
        }
    }

    fn tensor_name_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_scale") {
            "weight_scale"
        } else {
            "scales"
        }
    }

    fn load_global_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> f32 {
        let no_shard = Shard::default();
        if vb.contains_tensor("weight_global_scale") {
            // compressed-tensors format: weight_global_scale is a divisor, invert it
            let raw = vb
                .get_with_hints_dtype((1,), "weight_global_scale", no_shard, DType::F32)
                .or_else(|_| {
                    vb.get_with_hints_dtype((), "weight_global_scale", no_shard, DType::F32)
                })
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0);
            if raw != 0.0 {
                1.0 / raw
            } else {
                1.0
            }
        } else if vb.contains_tensor("weight_scale_2") {
            // modelopt format: weight_scale_2 is the direct multiplier
            vb.get_with_hints_dtype((1,), "weight_scale_2", no_shard, DType::F32)
                .or_else(|_| vb.get_with_hints_dtype((), "weight_scale_2", no_shard, DType::F32))
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn load_input_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> f32 {
        let no_shard = Shard::default();
        if vb.contains_tensor("input_scale") {
            vb.get_with_hints_dtype((1,), "input_scale", no_shard, DType::F32)
                .or_else(|_| vb.get_with_hints_dtype((), "input_scale", no_shard, DType::F32))
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0)
        } else if vb.contains_tensor("input_global_scale") {
            let raw = vb
                .get_with_hints_dtype((1,), "input_global_scale", no_shard, DType::F32)
                .or_else(|_| {
                    vb.get_with_hints_dtype((), "input_global_scale", no_shard, DType::F32)
                })
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0);
            if raw != 0.0 {
                1.0 / raw
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &cfg.quantization_config,
            &None,
            gate_dtype,
        )?;

        Self::load_experts(
            cfg,
            moe_cfg,
            num_experts,
            gate,
            vb.pp("experts"),
            comm,
            dtype,
            None,
            gate_dtype,
        )
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilderX,
        experts_vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_gate_and_bias(cfg, gate_vb, experts_vb, None, comm, dtype)
    }

    pub fn new_with_gate_and_bias(
        cfg: &Config,
        gate_vb: VarBuilderX,
        experts_vb: VarBuilderX,
        bias_vb: Option<&VarBuilderX>,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            gate_dtype,
        )?;

        let bias = bias_vb.and_then(|bvb| try_load_e_score_correction_bias(bvb, num_experts));
        Self::load_experts(
            cfg,
            moe_cfg,
            num_experts,
            gate,
            experts_vb,
            comm,
            dtype,
            bias,
            gate_dtype,
        )
    }

    fn load_experts(
        cfg: &Config,
        moe_cfg: &crate::utils::config::MoEConfig,
        num_experts: usize,
        gate: Linear,
        experts_vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        bias: Option<Tensor>,
        gate_dtype: DType,
    ) -> Result<Self> {
        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut gate_gscales_vec: Vec<f32> = Vec::new();
        let mut gate_iscales_vec: Vec<f32> = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut up_gscales_vec: Vec<f32> = Vec::new();
        let mut up_iscales_vec: Vec<f32> = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();
        let mut down_gscales_vec: Vec<f32> = Vec::new();
        let mut down_iscales_vec: Vec<f32> = Vec::new();

        match &experts_vb.0 {
            Either::Left(vb) => {
                let has_packed_gate_up = vb.contains_tensor("gate_up_proj")
                    || vb.contains_tensor("gate_up_proj_weight_scale_2");

                if has_packed_gate_up {
                    // Fused format: gate_up_proj [E, K/2, 2*N], down_proj [E, N/2, K]
                    // Transpose per expert to [2*N, K/2] / [K, N/2], then split
                    // gate_up into gate [N, K/2] + up [N, K/2] to reuse the
                    // standard stack-then-cat assembly below.
                    let inter = moe_cfg.moe_intermediate_size;
                    let hidden = cfg.hidden_size;
                    let sh0 = shard(0, comm.rank(), comm.world_size());
                    let sh1 = shard(1, comm.rank(), comm.world_size());
                    let no_shard = Shard::default();

                    let gu_raw = vb.get_with_hints_dtype(
                        (num_experts, hidden / 2, 2 * inter),
                        "gate_up_proj",
                        sh0,
                        DType::U8,
                    )?;
                    let gu_sc_raw = vb.get_with_hints_dtype(
                        (num_experts, hidden / 16, 2 * inter),
                        "gate_up_proj_weight_scale",
                        sh0,
                        DType::U8,
                    )?;
                    let gate_up_gscale = if vb.contains_tensor("gate_up_proj_weight_scale_2") {
                        vb.get_with_hints_dtype(
                            (),
                            "gate_up_proj_weight_scale_2",
                            no_shard,
                            DType::F32,
                        )
                        .or_else(|_| {
                            vb.get_with_hints_dtype(
                                (1,),
                                "gate_up_proj_weight_scale_2",
                                no_shard,
                                DType::F32,
                            )
                        })
                        .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                        .unwrap_or(1.0)
                    } else if vb.contains_tensor("gate_up_proj_weight_global_scale") {
                        let raw = vb
                            .get_with_hints_dtype(
                                (),
                                "gate_up_proj_weight_global_scale",
                                no_shard,
                                DType::F32,
                            )
                            .or_else(|_| {
                                vb.get_with_hints_dtype(
                                    (1,),
                                    "gate_up_proj_weight_global_scale",
                                    no_shard,
                                    DType::F32,
                                )
                            })
                            .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                            .unwrap_or(1.0);
                        if raw != 0.0 {
                            1.0 / raw
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };

                    let gate_up_iscale = if vb.contains_tensor("gate_up_proj_input_scale") {
                        vb.get_with_hints_dtype(
                            (),
                            "gate_up_proj_input_scale",
                            no_shard,
                            DType::F32,
                        )
                        .or_else(|_| {
                            vb.get_with_hints_dtype(
                                (1,),
                                "gate_up_proj_input_scale",
                                no_shard,
                                DType::F32,
                            )
                        })
                        .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                        .unwrap_or(1.0)
                    } else if vb.contains_tensor("gate_up_proj_input_global_scale") {
                        let raw = vb
                            .get_with_hints_dtype(
                                (),
                                "gate_up_proj_input_global_scale",
                                no_shard,
                                DType::F32,
                            )
                            .or_else(|_| {
                                vb.get_with_hints_dtype(
                                    (1,),
                                    "gate_up_proj_input_global_scale",
                                    no_shard,
                                    DType::F32,
                                )
                            })
                            .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                            .unwrap_or(1.0);
                        if raw != 0.0 {
                            1.0 / raw
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };

                    for i in 0..num_experts {
                        let gu = gu_raw.get(i)?.t()?.contiguous()?;
                        let gs = gu_sc_raw.get(i)?.t()?.contiguous()?;
                        gate_blocks_vec.push(gu.narrow(0, 0, inter)?.contiguous()?);
                        up_blocks_vec.push(gu.narrow(0, inter, inter)?.contiguous()?);
                        gate_scales_vec.push(gs.narrow(0, 0, inter)?.contiguous()?);
                        up_scales_vec.push(gs.narrow(0, inter, inter)?.contiguous()?);
                        gate_gscales_vec.push(gate_up_gscale);
                        up_gscales_vec.push(gate_up_gscale);
                        gate_iscales_vec.push(gate_up_iscale);
                        up_iscales_vec.push(gate_up_iscale);
                    }

                    let d_raw = vb.get_with_hints_dtype(
                        (num_experts, inter / 2, hidden),
                        "down_proj",
                        sh1,
                        DType::U8,
                    )?;
                    let d_sc_raw = vb.get_with_hints_dtype(
                        (num_experts, inter / 16, hidden),
                        "down_proj_weight_scale",
                        sh1,
                        DType::U8,
                    )?;
                    let down_gscale = if vb.contains_tensor("down_proj_weight_scale_2") {
                        vb.get_with_hints_dtype(
                            (),
                            "down_proj_weight_scale_2",
                            no_shard,
                            DType::F32,
                        )
                        .or_else(|_| {
                            vb.get_with_hints_dtype(
                                (1,),
                                "down_proj_weight_scale_2",
                                no_shard,
                                DType::F32,
                            )
                        })
                        .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                        .unwrap_or(1.0)
                    } else if vb.contains_tensor("down_proj_weight_global_scale") {
                        let raw = vb
                            .get_with_hints_dtype(
                                (),
                                "down_proj_weight_global_scale",
                                no_shard,
                                DType::F32,
                            )
                            .or_else(|_| {
                                vb.get_with_hints_dtype(
                                    (1,),
                                    "down_proj_weight_global_scale",
                                    no_shard,
                                    DType::F32,
                                )
                            })
                            .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                            .unwrap_or(1.0);
                        if raw != 0.0 {
                            1.0 / raw
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };

                    let down_iscale = if vb.contains_tensor("down_proj_input_scale") {
                        vb.get_with_hints_dtype((), "down_proj_input_scale", no_shard, DType::F32)
                            .or_else(|_| {
                                vb.get_with_hints_dtype(
                                    (1,),
                                    "down_proj_input_scale",
                                    no_shard,
                                    DType::F32,
                                )
                            })
                            .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                            .unwrap_or(1.0)
                    } else if vb.contains_tensor("down_proj_input_global_scale") {
                        let raw = vb
                            .get_with_hints_dtype(
                                (),
                                "down_proj_input_global_scale",
                                no_shard,
                                DType::F32,
                            )
                            .or_else(|_| {
                                vb.get_with_hints_dtype(
                                    (1,),
                                    "down_proj_input_global_scale",
                                    no_shard,
                                    DType::F32,
                                )
                            })
                            .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                            .unwrap_or(1.0);
                        if raw != 0.0 {
                            1.0 / raw
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };

                    for i in 0..num_experts {
                        down_blocks_vec.push(d_raw.get(i)?.t()?.contiguous()?);
                        down_scales_vec.push(d_sc_raw.get(i)?.t()?.contiguous()?);
                        down_gscales_vec.push(down_gscale);
                        down_iscales_vec.push(down_iscale);
                    }
                } else {
                    // Per-expert: experts.{i}.gate_proj / up_proj / down_proj
                    //            or w1 / w3 / w2 (MiniMax naming)
                    for i in 0..num_experts {
                        let expert_vb = vb.pp(i.to_string());
                        let (gate_name, up_name, down_name) =
                            resolve_expert_proj_prefix(&expert_vb);

                        let gate_proj_vb = expert_vb.pp(gate_name);
                        let packed_name = Self::tensor_name_packed(&gate_proj_vb);
                        let scale_name = Self::tensor_name_scale(&gate_proj_vb);
                        let sh0 = shard(0, comm.rank(), comm.world_size());

                        gate_blocks_vec.push(gate_proj_vb.get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                            packed_name,
                            sh0,
                            DType::U8,
                        )?);
                        gate_scales_vec.push(gate_proj_vb.get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                            scale_name,
                            sh0,
                            DType::U8,
                        )?);
                        gate_gscales_vec.push(Self::load_global_scale(&gate_proj_vb));
                        gate_iscales_vec.push(Self::load_input_scale(&gate_proj_vb));

                        let up_proj_vb = expert_vb.pp(up_name);
                        let packed_name = Self::tensor_name_packed(&up_proj_vb);
                        let scale_name = Self::tensor_name_scale(&up_proj_vb);

                        up_blocks_vec.push(up_proj_vb.get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                            packed_name,
                            sh0,
                            DType::U8,
                        )?);
                        up_scales_vec.push(up_proj_vb.get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                            scale_name,
                            sh0,
                            DType::U8,
                        )?);
                        up_gscales_vec.push(Self::load_global_scale(&up_proj_vb));
                        up_iscales_vec.push(Self::load_input_scale(&up_proj_vb));

                        let down_proj_vb = expert_vb.pp(down_name);
                        let packed_name = Self::tensor_name_packed(&down_proj_vb);
                        let scale_name = Self::tensor_name_scale(&down_proj_vb);
                        let sh1 = shard(1, comm.rank(), comm.world_size());

                        down_blocks_vec.push(down_proj_vb.get_with_hints_dtype(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                            packed_name,
                            sh1,
                            DType::U8,
                        )?);
                        down_scales_vec.push(down_proj_vb.get_with_hints_dtype(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size / 16),
                            scale_name,
                            sh1,
                            DType::U8,
                        )?);
                        down_gscales_vec.push(Self::load_global_scale(&down_proj_vb));
                        down_iscales_vec.push(Self::load_input_scale(&down_proj_vb));
                    }
                }
            }
            _ => candle_core::bail!("FusedMoeNvfp4: GGUF loading not supported for NVFP4"),
        }

        let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
        let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
        let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
        let up_scales = Tensor::stack(&up_scales_vec, 0)?;

        let gate_up_blocks = Tensor::cat(&[&gate_blocks, &up_blocks], 1)?;
        let gate_up_scales = Tensor::cat(&[&gate_scales, &up_scales], 1)?;
        let w_size_n = gate_up_blocks.dim(1)? / 2;

        let dev = gate_up_blocks.device();
        let gate_up_gscales: Vec<f32> = gate_gscales_vec
            .iter()
            .zip(up_gscales_vec.iter())
            .map(|(g, u)| {
                if (g - u).abs() > f32::EPSILON {
                    crate::log_warn!(
                        "NVFP4 MoE: gate/up global scales differ ({g} vs {u}), using gate scale"
                    );
                }
                *g
            })
            .collect();
        let gate_up_global_scales = Tensor::from_vec(gate_up_gscales, (num_experts,), dev)?;

        let gate_up_iscales: Vec<f32> = gate_iscales_vec
            .iter()
            .zip(up_iscales_vec.iter())
            .map(|(g, u)| {
                if (g - u).abs() > f32::EPSILON {
                    crate::log_warn!(
                        "NVFP4 MoE: gate/up input scales differ ({g} vs {u}), using gate scale"
                    );
                }
                *g
            })
            .collect();
        let gate_up_input_scales = Tensor::from_vec(gate_up_iscales, (num_experts,), dev)?;

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;
        let down_global_scales = Tensor::from_vec(down_gscales_vec, (num_experts,), dev)?;
        let down_input_scales = Tensor::from_vec(down_iscales_vec, (num_experts,), dev)?;

        let (gate_up_scales_swizzled, down_scales_swizzled) = {
            #[cfg(feature = "cuda")]
            {
                let sm = attention_rs::cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0)
                    as usize;
                if sm >= 100 {
                    let gu_sw =
                        attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(&gate_up_scales)?;
                    let d_sw =
                        attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(&down_scales)?;
                    (Some(gu_sw), Some(d_sw))
                } else {
                    (None, None)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                (None, None)
            }
        };

        Ok(Self {
            gate,
            gate_up_blocks,
            gate_up_scales,
            gate_up_global_scales,
            gate_up_input_scales,
            gate_up_scales_swizzled,
            down_blocks,
            down_scales,
            down_global_scales,
            down_input_scales,
            down_scales_swizzled,
            w_size_n,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, bias),
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            apply_router_weight_on_input: false,
            gate_dtype,
        })
    }

    pub fn set_sigmoid_routing(&mut self) {
        self.routing.use_sigmoid_scoring = true;
    }

    pub fn set_apply_router_weight_on_input(&mut self, v: bool) {
        self.apply_router_weight_on_input = v;
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            std::borrow::Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
        let (topk_weights, topk_ids) = self.routing.route(&router_logits, is_prefill)?;

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(self.dtype)?
        } else {
            xs.clone()
        };

        let xs = if self.apply_router_weight_on_input {
            let w = topk_weights.to_dtype(xs.dtype())?;
            xs.broadcast_mul(&w)?
        } else {
            xs
        };

        let pre_sorted = presorted_expert_assignments(&topk_ids, is_prefill)?;

        let pre_sorted_refs = pre_sorted.as_ref().map(|(a, b)| (a, b));

        let gate_up = moe::moe_gemm_nvfp4(
            &xs,
            &self.gate_up_blocks,
            &self.gate_up_scales,
            &self.gate_up_global_scales,
            Some(&self.gate_up_input_scales),
            None,
            &topk_ids,
            pre_sorted_refs,
            is_prefill,
            None,
            self.gate_up_scales_swizzled.as_ref(),
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let down_topk_weights = if self.apply_router_weight_on_input {
            None
        } else {
            Some(&topk_weights)
        };

        let mut ys = moe::moe_gemm_nvfp4(
            &down_inputs,
            &self.down_blocks,
            &self.down_scales,
            &self.down_global_scales,
            Some(&self.down_input_scales),
            None,
            &topk_ids,
            pre_sorted_refs,
            is_prefill,
            down_topk_weights,
            self.down_scales_swizzled.as_ref(),
        )?
        .reshape((num_tokens, self.routing.num_experts_per_tok, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}
