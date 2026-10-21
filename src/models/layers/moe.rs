// src/models/layers/moe.rs
use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::{isq_high_precision_dtype, VarBuilderX};
use crate::utils::config::Config;
use crate::utils::config::QuantConfig;
use crate::core::lora::{active_tenant, global_moe_lora, lora_execution, MoeLoraState};
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
pub(crate) fn gated_activation(
    gate_up: &Tensor,
    half_dim: usize,
    act: &Activation,
) -> Result<Tensor> {
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

pub(crate) fn sort_expert_assignments(
    topk_ids: &Tensor,
    is_prefill: bool,
) -> Result<(Tensor, Tensor)> {
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
/// to `e_score_correction_bias` directly (MiniMax style), then
/// `exp_probs_b.bias` (GLM-DSA / GGUF style), then `gate.bias`.
pub(crate) fn try_load_e_score_correction_bias(
    vb: &VarBuilderX,
    num_experts: usize,
) -> Option<Tensor> {
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
        .or_else(|| {
            vb.pp("exp_probs_b")
                .get_with_hints_dtype(num_experts, "bias", shard(0, 0, 1), DType::F32)
                .ok()
        })
        .or_else(|| {
            vb_gate
                .get_with_hints_dtype(num_experts, "bias", shard(0, 0, 1), DType::F32)
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
pub(crate) fn resolve_expert_proj_prefix(
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
            | "Qwen4ExpForCausalLM"
            | "Qwen4ExpForConditionalGeneration"
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
            | "Qwen4ExpForCausalLM"
            | "Qwen4ExpForConditionalGeneration"
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
    /// The MoE expert LoRA state (the #1.8, the per-expert gate_up/down A/B).
    /// The `None` when no MoE LoRA adapter is loaded (the fused `moe_gemm`
    /// kernel path, the prior behavior). When set + a tenant is active, the
    /// per-expert dense path applies the LoRA delta.
    moe_lora: Option<MoeLoraState>,
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
            moe_lora: None,
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

        // The MoE expert LoRA path (the #1.8): when a LoRA tenant is active + the
        // MoE LoRA state is loaded (the struct field OR the global, the set at
        // startup), use the per-expert dense path (the no the fused `moe_gemm`
        // kernel, which cannot take a delta). The per-expert dense path computes
        // the base expert output + the LoRA delta (the `gate_up_delta` + the
        // `down_delta`) for each expert, then combines.
        let global_moe_lora = global_moe_lora();
        let moe_lora_state = self.moe_lora.as_ref().or(global_moe_lora.as_ref());
        // The MoE LoRA path activates when a tenant is active (the single-tenant
        // thread-local) OR the batched execution has exactly one distinct tenant
        // (the common batched case, the all rows the same tenant). The no-op when
        // no tenant is active (the base model). The multi-distinct-tenant MoE case
        // (the each row a different MoE adapter) is the next increment (the per-
        // tenant MooeLoraState bank); for now the single-tenant + the uniform-
        // batch cases are correct.
        let tenant_active = active_tenant().is_some()
            || lora_execution().is_some_and(|exec| exec.distinct_tenants() == 1);
        if moe_lora_state.is_some() && tenant_active {
            return self.forward_with_routing_moe_lora(xs, &topk_weights, &topk_ids, is_prefill);
        }

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

    /// The per-expert dense MoE forward with the LoRA delta (the #1.8). The
    /// `moe_gemm` fused kernel cannot take a delta, so this path computes each
    /// expert's output densely (the `x @ expert_gate_up^T` + the activation +
    /// the `x @ expert_down^T`) + adds the LoRA delta (the `gate_up_delta` +
    /// the `down_delta`). The per-expert loop is the no the fused kernel (the
    /// slower, but correct for the LoRA case).
    fn forward_with_routing_moe_lora(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        // The per-tenant MoE state (the #1.8, the mixed-tenant batch). The
        // `self.moe_lora` is the instance-level state (the single-tenant case);
        // the `global_moe_lora_for(tenant)` resolves the per-tenant state for the
        // batched case. When no tenant-specific state is available, fall back to
        // the instance kernel (the no LoRA, the base model).
        let topk = self.routing.num_experts_per_tok;
        let num_experts = self.gate_up_w.dim(0)?;
        // The gather the per-token expert ids (the `topk_ids` is the
        // `[num_tokens, topk]`).
        let expert_ids: Vec<Vec<i64>> = topk_ids.to_vec2::<i64>()?;
        // The per-token tenant (the `LoraExecution` per-row mapping, the batched
        // multi-tenant case). The `None` when no execution context is set (the
        // single-tenant case, the `self.moe_lora` is used for all tokens).
        let row_tenants: Vec<Option<String>> = match crate::core::lora::lora_execution() {
            Some(exec) if exec.distinct_tenants() > 1 => exec
                .row_tenants
                .iter()
                .take(num_tokens)
                .cloned()
                .collect(),
            _ => vec![None; num_tokens],
        };
        // The top-expert token groups (the which tokens route to each expert).
        let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
        for (t, experts) in expert_ids.iter().enumerate() {
            for &e in experts.iter().take(topk) {
                if (0..num_experts).contains(&(e as usize)) {
                    expert_tokens[e as usize].push(t);
                }
            }
        }
        // The topk weights (the `[num_tokens, topk]` f32), for the per-expert
        // routing weight lookup.
        let topk_weights_f32 = topk_weights.to_vec2::<f32>()?;
        // The per-expert dense compute + the LoRA delta. The delta uses the
        // token's tenant's MoE state (the mixed-tenant batch, the #1.8); when no
        // per-rowtenant state is available (the single-tenant case), the
        // `self.moe_lora` is used for all tokens.
        let mut ys = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;
        for (e, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            // The gather this expert's tokens (the `[num_tokens_for_e, hidden]`).
            let idx = Tensor::from_vec(
                tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                (tokens.len(),),
                xs.device(),
            )?;
            let x_e = xs.index_select(&idx, 0)?;
            // The base expert gate_up (the `x_e @ gate_up_w[e]^T`, the
            // `[num_tokens_for_e, 2*intermediate]`). The `gate_up_w[e]` is the
            // `[hidden, 2*intermediate]`.
            let gate_up_e_w = self.gate_up_w.narrow(0, e, 1)?.squeeze(0)?;
            let gate_up_e = x_e.matmul(&gate_up_e_w.t()?)?;
            // The LoRA gate_up delta, per-token- (the mixed-tenant batch).
            // The group this expert's tokens by tenant; each tenant group uses
            // its own MoMoeLoraState`. The no-tenant tokens (the base model) get
            // no delta.
            let gate_up_delta = self.moe_gate_up_delta_for_tokens(
                &x_e,
                tokens,
                &row_tenants,
                e,
            )?;
            let gate_up = gate_up_e.broadcast_add(&gate_up_delta)?;
            // The activation (the gated, the `silu(gate) * up`).
            let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;
            // The base expert down (the `down_inputs @ down_w[e]^T`, the
            // `[num_tokens_for_e, hidden]`). The `down_w[e]` is the
            // `[intermediate, hidden]`.
            let down_e_w = self.down_w.narrow(0, e, 1)?.squeeze(0)?;
            let down_e = down_inputs.matmul(&down_e_w.t()?)?;
            // The LoRA down delta, per-token tenant.
            let down_delta = self.moe_down_delta_for_tokens(
                &down_inputs,
                tokens,
                &row_tenants,
                e,
            )?;
            let expert_out = down_e.broadcast_add(&down_delta)?;
            // The weight by the topk weight (the routing weight for this expert).
            // The `topk_weights_f32[t][pos]` is the weight for token `t`'s
            // expert at position `pos` (the `pos` where the `topk_ids[t][pos] == e`).
            let weights_e: Vec<f32> = tokens
                .iter()
                .map(|&t| {
                    let topk_ids_row = &expert_ids[t];
                    let pos = topk_ids_row
                        .iter()
                        .take(topk)
                        .position(|&x| x == e as i64)
                        .unwrap_or(0);
                    topk_weights_f32[t][pos]
                })
                .collect();
            let weight_tensor = Tensor::from_vec(weights_e, (tokens.len(),), xs.device())?
                .to_dtype(xs.dtype())?
                .reshape((tokens.len(), 1))?;
            let weighted = expert_out.broadcast_mul(&weight_tensor)?;
            // The scatter back into the output (the `index_add`).
            ys = ys.index_add(&idx, &weighted, 0)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }

    /// The per-token-tenant MoE gate_up delta (the #1.8, the mixed-tenant batch).
    /// The group the expert's tokens by tenant; each tenant group uses its own
    /// `MoeLoraState` (the `global_moe_lora_for(tenant)`). The no-tenant tokens
    /// (the base model) get the zero delta. The result is the `[num_tokens_for_e,
    /// 2*intermediate]` delta to add to the base gate_up.
    fn moe_gate_up_delta_for_tokens(
        &self,
        x_e: &Tensor,
        tokens: &[usize],
        row_tenants: &[Option<String>],
        expert: usize,
    ) -> Result<Tensor> {
        // The group the tokens by tenant (the `None` tenant = the base model, the
        // no delta).
        let mut by_tenant: std::collections::HashMap<Option<String>, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &t) in tokens.iter().enumerate() {
            let tenant = row_tenants.get(t).cloned().flatten();
            by_tenant.entry(tenant).or_default().push(i);
        }
        let mut delta = Tensor::zeros_like(x_e)?;
        for (tenant, locals) in by_tenant.iter() {
            // The resolve this tenant's MoE state (the `self.moe_lora` for the
            // single-tenant case, the `global_moe_lora_for` for the batched case).
            let state = match tenant {
                Some(t) => crate::core::lora::global_moe_lora_for(t).or_else(|| self.moe_lora.clone()),
                None => self.moe_lora.clone(),
            };
            let Some(state) = state else {
                continue; // the no state for this tenant (the no delta, the base model)
            };
            let idx = Tensor::from_vec(
                locals.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                (locals.len(),),
                x_e.device(),
            )?;
            let x_t = x_e.index_select(&idx, 0)?;
            let d_t = state.gate_up_delta(&x_t, expert)?;
            // The scatter this tenant's delta back into the full delta (the
            // `index_add`).
            delta = delta.index_add(&idx, &d_t, 0)?;
        }
        Ok(delta)
    }

    /// The per-token-tenant MoE down delta (the #1.8, the mixed-tenant batch).
    /// The same grouping as the `moe_gate_up_delta_for_tokens`, applied to the
    /// down projection.
    fn moe_down_delta_for_tokens(
        &self,
        down_inputs: &Tensor,
        tokens: &[usize],
        row_tenants: &[Option<String>],
        expert: usize,
    ) -> Result<Tensor> {
        let mut by_tenant: std::collections::HashMap<Option<String>, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &t) in tokens.iter().enumerate() {
            by_tenant
                .entry(row_tenants.get(t).cloned().flatten())
                .or_default()
                .push(i);
        }
        let mut delta = Tensor::zeros_like(down_inputs)?;
        for (tenant, idxs) in by_tenant.iter() {
            let state = match tenant {
                Some(t) => crate::core::lora::global_moe_lora_for(t)
                    .or_else(|| self.moe_lora.clone()),
                None => self.moe_lora.clone(),
            };
            let Some(state) = state else {
                continue;
            };
            let idx = Tensor::from_vec(
                idxs.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                (idxs.len(),),
                down_inputs.device(),
            )?;
            let x_t = down_inputs.index_select(&idx, 0)?;
            let d_t = state.down_delta(&x_t, expert)?;
            delta = delta.index_add(&idx, &d_t, 0)?;
        }
        Ok(delta)
    }
}

/// MoE WNA16 loader for compressed-tensors `pack-quantized` and symmetric
/// classic GPTQ checkpoints.
///
/// The checkpoint stores each expert projection independently as
/// `weight_packed` (`[out, in / pack_factor]`) and `weight_scale`
/// (`[out, in / group_size]`).  The packed tensors stay packed on device and
/// are consumed by attention.rs' grouped WNA16 kernel.
pub struct FusedMoeWNA16 {
    gate: Linear,
    gate_up_packed: Tensor,
    gate_up_scales: Tensor,
    down_packed: Tensor,
    down_scales: Tensor,
    w_size_n: usize,
    act: Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    bits: usize,
    group_size: usize,
    gate_dtype: DType,
    legacy_gptq: bool,
}

impl FusedMoeWNA16 {
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
        let bits = quant_cfg.bits;
        let legacy_gptq = quant_cfg.quant_method == "gptq";
        if !matches!(bits, 4 | 8) || quant_cfg.group_size <= 0 {
            candle_core::bail!(
                "WNA16 MoE requires 4/8 bits and a positive group size, got bits={bits}, group_size={}",
                quant_cfg.group_size
            );
        }
        let group_size = quant_cfg.group_size as usize;
        let pack_factor = 32 / bits;
        if cfg.hidden_size % pack_factor != 0
            || moe_cfg.moe_intermediate_size % pack_factor != 0
            || cfg.hidden_size % group_size != 0
            || moe_cfg.moe_intermediate_size % group_size != 0
        {
            candle_core::bail!(
                "WNA16 MoE dimensions must be divisible by pack_factor/group_size: hidden_size={}, moe_intermediate_size={}, pack_factor={}, group_size={group_size}",
                cfg.hidden_size,
                moe_cfg.moe_intermediate_size,
                pack_factor
            );
        }
        if quant_cfg.sym == Some(false) {
            candle_core::bail!("asymmetric WNA16 MoE is not supported yet");
        }
        if legacy_gptq && quant_cfg.desc_act == Some(true) {
            candle_core::bail!("classic GPTQ WNA16 MoE with desc_act/g_idx is not supported yet");
        }

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

        let mut gate_weights = Vec::with_capacity(num_experts);
        let mut gate_scales = Vec::with_capacity(num_experts);
        let mut up_weights = Vec::with_capacity(num_experts);
        let mut up_scales = Vec::with_capacity(num_experts);
        let mut down_weights = Vec::with_capacity(num_experts);
        let mut down_scales = Vec::with_capacity(num_experts);
        for expert_id in 0..num_experts {
            let expert = experts_vb.pp(expert_id.to_string().as_str());
            let (gate_name, up_name, down_name) = resolve_expert_proj_prefix_x(&expert);
            let gate_vb = expert.pp(gate_name);
            let up_vb = expert.pp(up_name);
            let down_vb = expert.pp(down_name);

            let (gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale) =
                if legacy_gptq {
                    // Classic GPTQ stores qweight=[K/pack,N] and scales=[K/group,N].
                    // The grouped WNA16 kernels consume [N,K/pack] and [N,K/group].
                    for (name, vb) in [
                        ("gate_proj", &gate_vb),
                        ("up_proj", &up_vb),
                        ("down_proj", &down_vb),
                    ] {
                        if !vb.has_key("qzeros") {
                            candle_core::bail!(
                                "classic GPTQ MoE projection {name} is missing qzeros"
                            );
                        }
                    }
                    let gate_weight = gate_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / pack_factor, moe_cfg.moe_intermediate_size),
                            "qweight",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let gate_scale = gate_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / group_size, moe_cfg.moe_intermediate_size),
                            "scales",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    let up_weight = up_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / pack_factor, moe_cfg.moe_intermediate_size),
                            "qweight",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let up_scale = up_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / group_size, moe_cfg.moe_intermediate_size),
                            "scales",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    let down_weight = down_vb
                        .get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size / pack_factor, cfg.hidden_size),
                            "qweight",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let down_scale = down_vb
                        .get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size / group_size, cfg.hidden_size),
                            "scales",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    (
                        gate_weight,
                        gate_scale,
                        up_weight,
                        up_scale,
                        down_weight,
                        down_scale,
                    )
                } else {
                    for (name, vb) in [
                        ("gate_proj", &gate_vb),
                        ("up_proj", &up_vb),
                        ("down_proj", &down_vb),
                    ] {
                        if vb.has_key("weight_g_idx") {
                            candle_core::bail!(
                                "compressed-tensors WNA16 MoE projection {name} has unsupported weight_g_idx"
                            );
                        }
                    }
                    let gate_weight = gate_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / pack_factor),
                        "weight_packed",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let gate_scale = gate_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / group_size),
                        "weight_scale",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    let up_weight = up_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / pack_factor),
                        "weight_packed",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let up_scale = up_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / group_size),
                        "weight_scale",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    let down_weight = down_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / pack_factor),
                        "weight_packed",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let down_scale = down_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / group_size),
                        "weight_scale",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    (
                        gate_weight,
                        gate_scale,
                        up_weight,
                        up_scale,
                        down_weight,
                        down_scale,
                    )
                };
            gate_weights.push(gate_weight);
            gate_scales.push(gate_scale);
            up_weights.push(up_weight);
            up_scales.push(up_scale);
            down_weights.push(down_weight);
            down_scales.push(down_scale);
        }

        let gate_up_packed = Tensor::cat(
            &[
                &Tensor::stack(&gate_weights, 0)?,
                &Tensor::stack(&up_weights, 0)?,
            ],
            1,
        )?
        .contiguous()?;
        let gate_up_scales = Tensor::cat(
            &[
                &Tensor::stack(&gate_scales, 0)?,
                &Tensor::stack(&up_scales, 0)?,
            ],
            1,
        )?
        .contiguous()?;
        let down_packed = Tensor::stack(&down_weights, 0)?.contiguous()?;
        let down_scales = Tensor::stack(&down_scales, 0)?.contiguous()?;
        let w_size_n = gate_up_packed.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_packed,
            gate_up_scales,
            down_packed,
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
            bits,
            group_size,
            gate_dtype,
            legacy_gptq,
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

        let gate_up = moe::moe_gemm_wna16(
            xs,
            &self.gate_up_packed,
            &self.gate_up_scales,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.routing.num_experts_per_tok,
            self.bits,
            self.group_size,
            is_prefill,
            self.legacy_gptq,
        )?;
        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;
        let mut ys = moe::moe_gemm_wna16(
            &down_inputs,
            &self.down_packed,
            &self.down_scales,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.routing.num_experts_per_tok,
            self.bits,
            self.group_size,
            is_prefill,
            self.legacy_gptq,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
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

/// Dtypes supported by `moe_gemm_gguf` kernel and `QTensor::quantize`.
fn can_quantize_to(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
    )
}

/// Dtypes supported natively by the CUDA `moe_gemm_gguf` kernel (no requant
/// needed). The IQ formats are decoded directly from their GGUF block layout.
fn moe_gemm_gguf_supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
            | GgmlDType::IQ2_XXS
            | GgmlDType::IQ2_XS
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ1_S
            | GgmlDType::IQ4_NL
            | GgmlDType::IQ3_S
            | GgmlDType::IQ2_S
            | GgmlDType::IQ4_XS
            | GgmlDType::IQ1_M
    )
}

fn closest_quantizable_dtype(orig: GgmlDType) -> GgmlDType {
    if can_quantize_to(orig) {
        return orig;
    }
    let bpw = orig.type_size() as f64 * 8.0 / orig.block_size() as f64;
    let candidates = [
        (GgmlDType::Q2K, 2.625),
        (GgmlDType::Q3K, 3.4375),
        (GgmlDType::Q4K, 4.5),
        (GgmlDType::Q5K, 5.5),
        (GgmlDType::Q6K, 6.5625),
        (GgmlDType::Q8_0, 8.5),
    ];
    candidates
        .iter()
        .min_by(|a, b| (a.1 - bpw).abs().partial_cmp(&(b.1 - bpw).abs()).unwrap())
        .map(|c| c.0)
        .unwrap_or(GgmlDType::Q4K)
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

        let v = match &vb.0 {
            Either::Right(v) => v.clone(),
            _ => panic!("Invalid varbuilder!"),
        };

        let is_packed = v.pp("ffn_gate_up_exps").contains_key("weight");

        // Get original dtypes from tensor metadata without loading data to GPU.
        let (orig_gate_dtype, orig_down_dtype) = if is_packed {
            let gate_dt = v
                .pp("ffn_gate_up_exps")
                .tensor_dtype("weight")
                .unwrap_or(GgmlDType::Q8_0);
            let down_dt = v
                .pp("ffn_down_exps")
                .tensor_dtype("weight")
                .unwrap_or(GgmlDType::Q8_0);
            (gate_dt, down_dt)
        } else {
            let gate_dt = v
                .pp("ffn_gate_exps")
                .tensor_dtype("weight")
                .unwrap_or(GgmlDType::Q8_0);
            let down_dt = v
                .pp("ffn_down_exps")
                .tensor_dtype("weight")
                .unwrap_or(GgmlDType::Q8_0);
            (gate_dt, down_dt)
        };

        // Check if the moe_gemm_gguf kernel supports this dtype directly
        let kernel_supports_gate = moe_gemm_gguf_supports(orig_gate_dtype);
        let kernel_supports_down = moe_gemm_gguf_supports(orig_down_dtype);

        // Determine target requant dtypes (only used when kernel doesn't support native format
        // or block alignment fails for direct sharding)
        let chunk_base = moe_cfg.moe_intermediate_size / comm.world_size();
        let target_gate_dtype = closest_quantizable_dtype(orig_gate_dtype);
        let ggml_dtype = if chunk_base % target_gate_dtype.block_size() == 0 {
            target_gate_dtype
        } else if chunk_base % GgmlDType::Q4K.block_size() == 0 {
            GgmlDType::Q4K
        } else {
            GgmlDType::Q8_0
        };
        let block_size = ggml_dtype.block_size();
        let target_down_dtype = closest_quantizable_dtype(orig_down_dtype);
        let high_precision_dtype = if chunk_base % target_down_dtype.block_size() == 0 {
            target_down_dtype
        } else if chunk_base % isq_high_precision_dtype(cfg.quant.as_deref()).block_size() == 0 {
            isq_high_precision_dtype(cfg.quant.as_deref())
        } else {
            ggml_dtype
        };

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

        let cur_chunk_size = if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
            <= moe_cfg.moe_intermediate_size
        {
            moe_intermediate_chunk
        } else {
            moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk
        };

        let (ggml_dtype, high_precision_dtype, cur_chunk_size) = if cur_chunk_size == 0
            || cur_chunk_size % block_size != 0
        {
            let fb = GgmlDType::Q8_0;
            let fb_bs = fb.block_size();
            let fb_chunk = moe_cfg.moe_intermediate_size / comm.world_size();
            let fb_cur = if comm.rank() * fb_chunk + fb_chunk <= moe_cfg.moe_intermediate_size {
                fb_chunk
            } else {
                moe_cfg.moe_intermediate_size - comm.rank() * fb_chunk
            };
            assert!(
                    fb_cur > 0 && fb_cur % fb_bs == 0,
                    "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
                    \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
                    moe_cfg.moe_intermediate_size,
                    comm.world_size(),
                    fb_bs
                );
            (fb, fb, fb_cur)
        } else {
            (ggml_dtype, high_precision_dtype, cur_chunk_size)
        };

        let shard_offset = comm.rank() * moe_intermediate_chunk;

        let (gate_experts, up_experts, down_experts) = if is_packed {
            // Packed gate_up_exps: shard each half separately using world_size*2 trick.
            // rank -> gate shard, rank+world_size -> up shard
            let packed_shape = (
                num_experts,
                moe_cfg.moe_intermediate_size * 2,
                cfg.hidden_size,
            );
            let ws2 = comm.world_size() * 2;

            let gate_q = if kernel_supports_gate && chunk_base % orig_gate_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_gate_up_exps").get_sharded(
                    packed_shape,
                    "weight",
                    1,
                    comm.rank(),
                    ws2,
                )? {
                    qt
                } else {
                    Self::fallback_shard_gate_up(
                        &v,
                        "ffn_gate_up_exps",
                        packed_shape,
                        0,
                        moe_cfg.moe_intermediate_size,
                        shard_offset,
                        cur_chunk_size,
                        ggml_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_gate_up(
                    &v,
                    "ffn_gate_up_exps",
                    packed_shape,
                    0,
                    moe_cfg.moe_intermediate_size,
                    shard_offset,
                    cur_chunk_size,
                    ggml_dtype,
                )?
            };

            let up_q = if kernel_supports_gate && chunk_base % orig_gate_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_gate_up_exps").get_sharded(
                    packed_shape,
                    "weight",
                    1,
                    comm.rank() + comm.world_size(),
                    ws2,
                )? {
                    qt
                } else {
                    Self::fallback_shard_gate_up(
                        &v,
                        "ffn_gate_up_exps",
                        packed_shape,
                        moe_cfg.moe_intermediate_size,
                        moe_cfg.moe_intermediate_size,
                        shard_offset,
                        cur_chunk_size,
                        ggml_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_gate_up(
                    &v,
                    "ffn_gate_up_exps",
                    packed_shape,
                    moe_cfg.moe_intermediate_size,
                    moe_cfg.moe_intermediate_size,
                    shard_offset,
                    cur_chunk_size,
                    ggml_dtype,
                )?
            };

            // down_exps: direct shard on dim 2
            let down_q = if kernel_supports_down && chunk_base % orig_down_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_down_exps").get_sharded(
                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    2,
                    comm.rank(),
                    comm.world_size(),
                )? {
                    qt
                } else {
                    Self::fallback_shard_down(
                        &v,
                        num_experts,
                        cfg.hidden_size,
                        moe_cfg.moe_intermediate_size,
                        shard_offset,
                        cur_chunk_size,
                        high_precision_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_down(
                    &v,
                    num_experts,
                    cfg.hidden_size,
                    moe_cfg.moe_intermediate_size,
                    shard_offset,
                    cur_chunk_size,
                    high_precision_dtype,
                )?
            };

            (gate_q, up_q, down_q)
        } else {
            // Separate gate/up/down: direct sharded loading for each
            let gate_q = if kernel_supports_gate && chunk_base % orig_gate_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_gate_exps").get_sharded(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    1,
                    comm.rank(),
                    comm.world_size(),
                )? {
                    qt
                } else {
                    Self::fallback_shard_expert(
                        &v,
                        "ffn_gate_exps",
                        num_experts,
                        moe_cfg.moe_intermediate_size,
                        cfg.hidden_size,
                        1,
                        shard_offset,
                        cur_chunk_size,
                        ggml_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_expert(
                    &v,
                    "ffn_gate_exps",
                    num_experts,
                    moe_cfg.moe_intermediate_size,
                    cfg.hidden_size,
                    1,
                    shard_offset,
                    cur_chunk_size,
                    ggml_dtype,
                )?
            };

            let up_q = if kernel_supports_gate && chunk_base % orig_gate_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_up_exps").get_sharded(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    1,
                    comm.rank(),
                    comm.world_size(),
                )? {
                    qt
                } else {
                    Self::fallback_shard_expert(
                        &v,
                        "ffn_up_exps",
                        num_experts,
                        moe_cfg.moe_intermediate_size,
                        cfg.hidden_size,
                        1,
                        shard_offset,
                        cur_chunk_size,
                        ggml_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_expert(
                    &v,
                    "ffn_up_exps",
                    num_experts,
                    moe_cfg.moe_intermediate_size,
                    cfg.hidden_size,
                    1,
                    shard_offset,
                    cur_chunk_size,
                    ggml_dtype,
                )?
            };

            let down_q = if kernel_supports_down && chunk_base % orig_down_dtype.block_size() == 0 {
                if let Some(qt) = v.pp("ffn_down_exps").get_sharded(
                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    2,
                    comm.rank(),
                    comm.world_size(),
                )? {
                    qt
                } else {
                    Self::fallback_shard_down(
                        &v,
                        num_experts,
                        cfg.hidden_size,
                        moe_cfg.moe_intermediate_size,
                        shard_offset,
                        cur_chunk_size,
                        high_precision_dtype,
                    )?
                }
            } else {
                Self::fallback_shard_down(
                    &v,
                    num_experts,
                    cfg.hidden_size,
                    moe_cfg.moe_intermediate_size,
                    shard_offset,
                    cur_chunk_size,
                    high_precision_dtype,
                )?
            };

            (gate_q, up_q, down_q)
        };

        let world_size = comm.world_size();

        let bias = try_load_e_score_correction_bias(&vb, num_experts);

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, bias),
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    fn fallback_shard_expert(
        v: &crate::utils::gguf_varbuilder::VarBuilder,
        prefix: &str,
        num_experts: usize,
        dim0: usize,
        dim1: usize,
        shard_dim: usize,
        shard_offset: usize,
        cur_chunk_size: usize,
        target_dtype: GgmlDType,
    ) -> Result<Arc<QTensor>> {
        let qt = v.pp(prefix).get((num_experts, dim0, dim1), "weight")?;
        let f16 = qt.dequantize_f16(&v.device())?;
        drop(qt);
        v.clear_cache();
        let f16 = f16
            .narrow(shard_dim, shard_offset, cur_chunk_size)?
            .contiguous()?;
        let q = Arc::new(QTensor::quantize(&f16, target_dtype)?);
        drop(f16);
        Ok(q)
    }

    fn fallback_shard_gate_up(
        v: &crate::utils::gguf_varbuilder::VarBuilder,
        prefix: &str,
        packed_shape: (usize, usize, usize),
        narrow_offset: usize,
        intermediate_size: usize,
        shard_offset: usize,
        cur_chunk_size: usize,
        target_dtype: GgmlDType,
    ) -> Result<Arc<QTensor>> {
        let packed = v.pp(prefix).get(packed_shape, "weight")?;
        let deq = packed.dequantize_f16(&v.device())?;
        drop(packed);
        v.clear_cache();
        let f16 = deq
            .narrow(1, narrow_offset, intermediate_size)?
            .narrow(1, shard_offset, cur_chunk_size)?
            .contiguous()?;
        drop(deq);
        let q = Arc::new(QTensor::quantize(&f16, target_dtype)?);
        drop(f16);
        Ok(q)
    }

    fn fallback_shard_down(
        v: &crate::utils::gguf_varbuilder::VarBuilder,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        shard_offset: usize,
        cur_chunk_size: usize,
        target_dtype: GgmlDType,
    ) -> Result<Arc<QTensor>> {
        let qt = v
            .pp("ffn_down_exps")
            .get((num_experts, hidden_size, intermediate_size), "weight")?;
        let f16 = qt.dequantize_f16(&v.device())?;
        drop(qt);
        v.clear_cache();
        let f16 = f16.narrow(2, shard_offset, cur_chunk_size)?.contiguous()?;
        let q = Arc::new(QTensor::quantize(&f16, target_dtype)?);
        drop(f16);
        Ok(q)
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
                    if moe_gemm_gguf_supports(orig_dtype) {
                        // Kernel supports this dtype; split packed into gate/up without requant.
                        // Use get_sharded with world_size=2 to read each half directly.
                        let half_shape = (
                            num_experts,
                            moe_cfg.moe_intermediate_size * 2,
                            cfg.hidden_size,
                        );
                        let gate_exp = v
                            .pp("ffn_gate_up_exps")
                            .get_sharded(half_shape, "weight", 1, 0, 2)?;
                        let up_exp = v
                            .pp("ffn_gate_up_exps")
                            .get_sharded(half_shape, "weight", 1, 1, 2)?;
                        if let (Some(g), Some(u)) = (gate_exp, up_exp) {
                            let down = v.pp("ffn_down_exps").get(
                                (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "weight",
                            )?;
                            (g, u, down)
                        } else {
                            // Fallback: dequant + requant
                            let requant_dtype = if can_quantize_to(orig_dtype) {
                                orig_dtype
                            } else {
                                closest_quantizable_dtype(orig_dtype)
                            };
                            let deq = packed.dequantize_f16(&v.device())?;
                            let gate_f16 = deq
                                .narrow(1, 0, moe_cfg.moe_intermediate_size)?
                                .contiguous()?;
                            let up_f16 = deq
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
                                Arc::new(QTensor::quantize(&gate_f16, requant_dtype)?),
                                Arc::new(QTensor::quantize(&up_f16, requant_dtype)?),
                                down,
                            )
                        }
                    } else {
                        let requant_dtype = if can_quantize_to(orig_dtype) {
                            orig_dtype
                        } else {
                            closest_quantizable_dtype(orig_dtype)
                        };
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
                            Arc::new(QTensor::quantize(&gate_exp, requant_dtype)?),
                            Arc::new(QTensor::quantize(&up_exp, requant_dtype)?),
                            down,
                        )
                    }
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

        let bias = try_load_e_score_correction_bias(&vb, num_experts);

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            routing: MoeRouting::from_moe_cfg(moe_cfg, bias),
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
        if self.world_size > 1 {
            // AllReduce in the current dtype (F32 from moe_gemm_gguf) to preserve
            // precision for low-bit quantized weights (IQ2_XS, Q2_K) where partial
            // sums have high variance and BF16 truncation causes catastrophic
            // cancellation.
            ys = self.all_reduce.apply(&ys)?;
        }
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
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

        let gate_last_dim = gate_experts.dim(candle_core::D::Minus1)?;
        let gate_up_quant_type = if gate_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if gate_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle_core::bail!(
                "ISQ MoE gate/up last dim {} incompatible with any GGUF block size",
                gate_last_dim
            );
        };
        let down_last_dim = down_experts.dim(candle_core::D::Minus1)?;
        let hp_dtype = isq_high_precision_dtype(cfg.quant.as_deref());
        let down_quant_type = if down_last_dim % hp_dtype.block_size() == 0 {
            hp_dtype
        } else if down_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if down_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle_core::bail!(
                "ISQ MoE down_experts last dim {} incompatible with any GGUF block size",
                down_last_dim
            );
        };
        let gate_experts = QTensor::quantize(&gate_experts, gate_up_quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, gate_up_quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, down_quant_type)?;
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

        let gate_last_dim = gate_experts.dim(candle_core::D::Minus1)?;
        let gate_up_quant_type = if gate_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if gate_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle_core::bail!(
                "ISQ MoE gate/up last dim {} incompatible with any GGUF block size",
                gate_last_dim
            );
        };
        let down_last_dim = down_experts.dim(candle_core::D::Minus1)?;
        let hp_dtype = isq_high_precision_dtype(cfg.quant.as_deref());
        let down_quant_type = if down_last_dim % hp_dtype.block_size() == 0 {
            hp_dtype
        } else if down_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if down_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle_core::bail!(
                "ISQ MoE down_experts last dim {} incompatible with any GGUF block size",
                down_last_dim
            );
        };
        let gate_experts = QTensor::quantize(&gate_experts, gate_up_quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, gate_up_quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, down_quant_type)?;
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
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
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
        // Preserve logical F8 scale tensors as U8-backed FP8.  Requesting U8
        // first would numerically convert F8_E8M0/F8_E4M3 and corrupt the
        // exponent bytes consumed by the MXFP4 CUDA kernels.
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

struct Nvfp4Projection {
    blocks: Tensor,
    scales: Tensor,
    global_scales: Tensor,
    input_scales: Tensor,
    scales_swizzled: Option<Tensor>,
    /// Optional per-half (gate/up) weight global scales, shape [E, 2]. When
    /// present, the GEMM applies `half_scales[e, 0]` to the gate rows and
    /// `half_scales[e, 1]` to the up rows, so independently-quantized gate/up
    /// projections can be merged into one fused GEMM without re-quantization.
    half_scales: Option<Tensor>,
}

enum Nvfp4GateUpWeights {
    Fused(Nvfp4Projection),
    // Kept as a defensive fallback for gate/up projections that cannot be fused
    // (e.g. mismatched shapes). All supported NVFP4 load paths fuse at load
    // time, so this variant is currently never constructed.
    #[allow(dead_code)]
    Separate {
        gate: Nvfp4Projection,
        up: Nvfp4Projection,
    },
}

fn maybe_swizzle_nvfp4_scales(_scales: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        let sm = attention_rs::cuda_utils::sm_version(_scales.device().as_cuda_device()?)
            .unwrap_or(0) as usize;
        if sm >= 100 {
            return Ok(Some(
                attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(_scales)?,
            ));
        }
    }
    Ok(None)
}

impl Nvfp4Projection {
    fn new(
        blocks: Tensor,
        scales: Tensor,
        global_scales: Vec<f32>,
        input_scales: Vec<f32>,
    ) -> Result<Self> {
        let dev = blocks.device().clone();
        let num_experts = blocks.dim(0)?;
        if global_scales.len() != num_experts || input_scales.len() != num_experts {
            candle_core::bail!(
                "NVFP4 MoE projection scale count mismatch: experts={}, global={}, input={}",
                num_experts,
                global_scales.len(),
                input_scales.len()
            );
        }
        let scales_swizzled = maybe_swizzle_nvfp4_scales(&scales)?;
        Ok(Self {
            blocks,
            scales,
            global_scales: Tensor::from_vec(global_scales, (num_experts,), &dev)?,
            input_scales: Tensor::from_vec(input_scales, (num_experts,), &dev)?,
            scales_swizzled,
            half_scales: None,
        })
    }
}

pub struct FusedMoeNvfp4 {
    gate: Linear,
    gate_up: Nvfp4GateUpWeights,
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
        let mut gate_up: Option<Nvfp4GateUpWeights> = None;

        match &experts_vb.0 {
            Either::Left(vb) => {
                let has_packed_gate_up = vb.contains_tensor("gate_up_proj")
                    || vb.contains_tensor("gate_up_proj_weight_scale_2");

                if has_packed_gate_up {
                    // Fused format: gate_up_proj [E, K/2, 2*N], down_proj [E, N/2, K]
                    // Mirror BF16/FP8 packed MoE: shard the intermediate (dim 2 /
                    // dim 1 after layout) — never dim 0 (experts).
                    let inter = moe_cfg.moe_intermediate_size;
                    let hidden = cfg.hidden_size;
                    let world = comm.world_size().max(1);
                    let rank = comm.rank();
                    let no_shard = Shard::default();
                    let sh_gate = shard(2, rank, world * 2);
                    let sh_up = shard(2, rank + world, world * 2);
                    let sh_down = shard(1, rank, world);

                    let gu_gate = vb.get_with_hints_dtype(
                        (num_experts, hidden / 2, 2 * inter),
                        "gate_up_proj",
                        sh_gate,
                        DType::U8,
                    )?;
                    let gu_up = vb.get_with_hints_dtype(
                        (num_experts, hidden / 2, 2 * inter),
                        "gate_up_proj",
                        sh_up,
                        DType::U8,
                    )?;
                    let gu_sc_gate = vb.get_with_hints_dtype(
                        (num_experts, hidden / 16, 2 * inter),
                        "gate_up_proj_weight_scale",
                        sh_gate,
                        DType::U8,
                    )?;
                    let gu_sc_up = vb.get_with_hints_dtype(
                        (num_experts, hidden / 16, 2 * inter),
                        "gate_up_proj_weight_scale",
                        sh_up,
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

                    let mut fused_blocks = Vec::with_capacity(num_experts);
                    let mut fused_scales = Vec::with_capacity(num_experts);
                    for i in 0..num_experts {
                        // The checkpoint is already gate_up_proj. Rebuild the
                        // local fused tensor after sharding; do not create
                        // separate gate/up runtime projections.
                        let gate_blocks = gu_gate.get(i)?.t()?.contiguous()?;
                        let up_blocks = gu_up.get(i)?.t()?.contiguous()?;
                        fused_blocks.push(Tensor::cat(&[&gate_blocks, &up_blocks], 0)?);

                        let gate_scales = gu_sc_gate.get(i)?.t()?.contiguous()?;
                        let up_scales = gu_sc_up.get(i)?.t()?.contiguous()?;
                        fused_scales.push(Tensor::cat(&[&gate_scales, &up_scales], 0)?);
                    }
                    gate_up = Some(Nvfp4GateUpWeights::Fused(Nvfp4Projection::new(
                        Tensor::stack(&fused_blocks, 0)?,
                        Tensor::stack(&fused_scales, 0)?,
                        vec![gate_up_gscale; num_experts],
                        vec![gate_up_iscale; num_experts],
                    )?));

                    let d_raw = vb.get_with_hints_dtype(
                        (num_experts, inter / 2, hidden),
                        "down_proj",
                        sh_down,
                        DType::U8,
                    )?;
                    let d_sc_raw = vb.get_with_hints_dtype(
                        (num_experts, inter / 16, hidden),
                        "down_proj_weight_scale",
                        sh_down,
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

        let (gate_up, w_size_n) = if let Some(gate_up) = gate_up {
            let w_size_n = match &gate_up {
                Nvfp4GateUpWeights::Fused(projection) => projection.blocks.dim(1)? / 2,
                Nvfp4GateUpWeights::Separate { .. } => unreachable!(),
            };
            (gate_up, w_size_n)
        } else {
            // Merge independently-quantized gate/up projections into a single
            // fused GEMM (sglang-style): byte-concatenate the FP4 blocks and
            // E4M3 block scales, and carry the per-half global scales so the
            // kernel applies the correct scale to each half. No re-quantization,
            // so the original NVFP4 weights are preserved exactly.
            let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
            let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
            let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
            let up_scales = Tensor::stack(&up_scales_vec, 0)?;
            let n_gate = gate_blocks.dim(1)?;
            let n_up = up_blocks.dim(1)?;
            if n_gate != n_up {
                candle_core::bail!(
                    "NVFP4 MoE gate/up output dimensions differ: gate={}, up={}",
                    n_gate,
                    n_up
                );
            }
            let dev = gate_blocks.device().clone();
            let merged_blocks = Tensor::cat(&[&gate_blocks, &up_blocks], 1)?;
            let merged_scales = Tensor::cat(&[&gate_scales, &up_scales], 1)?;
            let mut merged_gscales = Vec::with_capacity(num_experts);
            let mut merged_iscales = Vec::with_capacity(num_experts);
            let mut half_rows = Vec::with_capacity(num_experts * 2);
            for e in 0..num_experts {
                let g = gate_gscales_vec[e];
                let u = up_gscales_vec[e];
                merged_gscales.push(g.max(u));
                merged_iscales.push(gate_iscales_vec[e].max(up_iscales_vec[e]));
                half_rows.push(g);
                half_rows.push(u);
            }
            let half_scales = Tensor::from_vec(half_rows, (num_experts, 2), &dev)?;
            let mut projection =
                Nvfp4Projection::new(merged_blocks, merged_scales, merged_gscales, merged_iscales)?;
            projection.half_scales = Some(half_scales);
            (Nvfp4GateUpWeights::Fused(projection), n_gate)
        };

        let dev = match &gate_up {
            Nvfp4GateUpWeights::Fused(projection) => projection.blocks.device().clone(),
            Nvfp4GateUpWeights::Separate { gate, .. } => gate.blocks.device().clone(),
        };

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;
        let down_global_scales = Tensor::from_vec(down_gscales_vec, (num_experts,), &dev)?;
        let down_input_scales = Tensor::from_vec(down_iscales_vec, (num_experts,), &dev)?;

        let down_scales_swizzled = maybe_swizzle_nvfp4_scales(&down_scales)?;

        Ok(Self {
            gate,
            gate_up,
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

        let gate_up = match &self.gate_up {
            Nvfp4GateUpWeights::Fused(projection) => moe::moe_gemm_nvfp4(
                &xs,
                &projection.blocks,
                &projection.scales,
                &projection.global_scales,
                Some(&projection.input_scales),
                None,
                &topk_ids,
                pre_sorted_refs,
                is_prefill,
                None,
                projection.scales_swizzled.as_ref(),
                None,
            )?,
            Nvfp4GateUpWeights::Separate { gate, up } => {
                let gate_output = moe::moe_gemm_nvfp4(
                    &xs,
                    &gate.blocks,
                    &gate.scales,
                    &gate.global_scales,
                    Some(&gate.input_scales),
                    None,
                    &topk_ids,
                    pre_sorted_refs,
                    is_prefill,
                    None,
                    gate.scales_swizzled.as_ref(),
                    None,
                )?;
                let up_output = moe::moe_gemm_nvfp4(
                    &xs,
                    &up.blocks,
                    &up.scales,
                    &up.global_scales,
                    Some(&up.input_scales),
                    None,
                    &topk_ids,
                    pre_sorted_refs,
                    is_prefill,
                    None,
                    up.scales_swizzled.as_ref(),
                    None,
                )?;
                Tensor::cat(&[&gate_output, &up_output], 2)?
            }
        };

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
            None,
        )?
        .reshape((num_tokens, self.routing.num_experts_per_tok, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}

/// Packed 2-bit MoE expert weights (GPU planes + UE8M0 scales).
///
/// Shared by [`FusedMoeW2`] (standard gate routing) and model-specific routers
/// (e.g. DeepSeek V4 hash/score) that call [`MoeW2ExpertWeights::forward`].
pub struct MoeW2ExpertWeights {
    pub gate_up_planes: Tensor,
    pub gate_up_scales: Tensor,
    pub down_planes: Tensor,
    pub down_scales: Tensor,
    pub gate_up_n: usize,
    pub gate_up_k: usize,
    pub down_n: usize,
    pub down_k: usize,
}

impl MoeW2ExpertWeights {
    /// GPU-side re-quant of MXFP4/e2m1 expert tensors into 2-bit planes.
    pub fn pack_from_mxfp4(
        gate_up_blocks: &Tensor,
        gate_up_scales: &Tensor,
        down_blocks: &Tensor,
        down_scales: &Tensor,
    ) -> Result<Self> {
        let e = gate_up_blocks.dim(0)?;
        let n = gate_up_blocks.dim(1)?;
        let k_packed = gate_up_blocks.dim(2)?; // K/2
        let k = k_packed * 2;

        let gu_blocks = gate_up_blocks.contiguous()?;
        let gu_scales = gate_up_scales.contiguous()?;
        let gu_scales = match gu_scales.dims().len() {
            3 => gu_scales,
            2 => gu_scales.reshape((e, n, k / 32))?,
            _ => candle_core::bail!(
                "MoeW2ExpertWeights: unexpected gate_up_scales dims {:?}",
                gu_scales.dims()
            ),
        };

        let (gate_up_planes, gate_up_scales) =
            attention_rs::moe_w2::moe_w2_pack_from_mxfp4(&gu_blocks, &gu_scales, n, k)?;

        let dn_n = down_blocks.dim(1)?;
        let dn_k_packed = down_blocks.dim(2)?;
        let dn_k = dn_k_packed * 2;
        let dn_blocks = down_blocks.contiguous()?;
        let dn_scales = down_scales.contiguous()?;
        let dn_scales = match dn_scales.dims().len() {
            3 => dn_scales,
            2 => dn_scales.reshape((e, dn_n, dn_k / 32))?,
            _ => candle_core::bail!(
                "MoeW2ExpertWeights: unexpected down_scales dims {:?}",
                dn_scales.dims()
            ),
        };

        let (down_planes, down_scales) =
            attention_rs::moe_w2::moe_w2_pack_from_mxfp4(&dn_blocks, &dn_scales, dn_n, dn_k)?;

        Ok(Self {
            gate_up_planes,
            gate_up_scales: gate_up_scales.reshape((e, n * (k / 32)))?,
            down_planes,
            down_scales: down_scales.reshape((e, dn_n * (dn_k / 32)))?,
            gate_up_n: n,
            gate_up_k: k,
            down_n: dn_n,
            down_k: dn_k,
        })
    }

    /// Expert GEMM path only (caller supplies top-k routing).
    pub fn forward(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: &Activation,
        swiglu_limit: f32,
        dtype: DType,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let topk = topk_ids.dim(D::Minus1)?;

        let moe_dtype = if dtype == DType::F32 {
            DType::BF16
        } else {
            dtype
        };
        let xs = if xs.dtype() != moe_dtype {
            xs.to_dtype(moe_dtype)?
        } else {
            xs.clone()
        };

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(topk_ids, is_prefill)?;

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

        let down_inputs = if matches!(act, Activation::Silu) && swiglu_limit > 0.0 {
            attention_rs::moe_w2::moe_w2_swiglu_clamp_bf16(
                &gate_up,
                self.gate_up_n / 2,
                swiglu_limit,
            )?
        } else {
            gated_activation(&gate_up, self.gate_up_n / 2, act)?
        };
        let down_inputs = if down_inputs.dtype() != moe_dtype {
            down_inputs.to_dtype(moe_dtype)?
        } else {
            down_inputs
        };

        moe::moe_gemm_w2(
            &down_inputs,
            &self.down_planes,
            &self.down_scales,
            &Some(topk_weights.clone()),
            &sorted_token_ids,
            &expert_ids,
            topk,
            self.down_n,
            self.down_k,
            is_prefill,
        )?
        .reshape((num_tokens, topk, hidden_dim))?
        .sum(1)?
        .to_dtype(dtype)
    }
}

/// General 2-bit MoE (`--isq w2` / `--isq moe_w2`): standard gate + [`MoeRouting`].
///
/// Built by GPU-side re-quant of checkpoint MXFP4 experts. FP4 restore is off by
/// default so MXFP4 weights are freed after packing; enable with `XINFER_MOE_W2_GATE=1`.
pub struct FusedMoeW2 {
    gate: Linear,
    experts: MoeW2ExpertWeights,
    act: Activation,
    routing: MoeRouting,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    gate_dtype: DType,
    swiglu_limit: f32,
    restore_mxfp4: Option<FusedMoeMxfp4>,
    restore_gate_enabled: bool,
    restore_gate_tau: f32,
}

impl FusedMoeW2 {
    /// Pack MXFP4 experts into W2 planes, keeping the MXFP4 gate/routing.
    pub fn from_mxfp4(mx: FusedMoeMxfp4) -> Result<Self> {
        let restore_gate_enabled = std::env::var("XINFER_MOE_W2_GATE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let restore_gate_tau = std::env::var("XINFER_MOE_W2_GATE_TAU")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.60);

        let experts = MoeW2ExpertWeights::pack_from_mxfp4(
            &mx.gate_up_blocks,
            &mx.gate_up_scales,
            &mx.down_blocks,
            &mx.down_scales,
        )?;

        let restore_mxfp4 = if restore_gate_enabled {
            Some(FusedMoeMxfp4 {
                gate: mx.gate.clone(),
                gate_up_blocks: mx.gate_up_blocks.clone(),
                gate_up_scales: mx.gate_up_scales.clone(),
                down_blocks: mx.down_blocks.clone(),
                down_scales: mx.down_scales.clone(),
                w_size_n: mx.w_size_n,
                act: mx.act.clone(),
                routing: mx.routing.clone(),
                all_reduce: mx.all_reduce.clone(),
                world_size: mx.world_size,
                dtype: mx.dtype,
                gate_dtype: mx.gate_dtype,
            })
        } else {
            None
        };

        Ok(Self {
            gate: mx.gate,
            experts,
            act: mx.act,
            routing: mx.routing,
            all_reduce: mx.all_reduce,
            world_size: mx.world_size,
            dtype: mx.dtype,
            gate_dtype: mx.gate_dtype,
            swiglu_limit: 0.0,
            restore_mxfp4,
            restore_gate_enabled,
            restore_gate_tau,
        })
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let mx = FusedMoeMxfp4::new(cfg, vb, comm, dtype)?;
        let mut out = Self::from_mxfp4(mx)?;
        if let Some(extra) = &cfg.extra_config_json {
            if let Ok(extra) = serde_json::from_str::<serde_json::Value>(extra) {
                out.swiglu_limit = extra
                    .get("swiglu_limit")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
            }
        }
        Ok(out)
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilderX,
        experts_vb: VarBuilderX,
        bias_vb: &VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let mx = FusedMoeMxfp4::new_with_gate(cfg, gate_vb, experts_vb, bias_vb, comm, dtype)?;
        Self::from_mxfp4(mx)
    }

    pub fn with_swiglu_limit(mut self, limit: f32) -> Self {
        self.swiglu_limit = limit;
        self
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        if self.restore_gate_enabled && !is_prefill {
            if let Some(restore) = self.restore_mxfp4.as_ref() {
                let gate_input = if xs.dtype() != self.gate_dtype {
                    std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
                } else {
                    std::borrow::Cow::Borrowed(xs)
                };
                let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;
                let (topk_weights, _) = self.routing.route(&router_logits, is_prefill)?;
                let max_prob = topk_weights.max(1)?.max(0)?.to_vec0::<f32>()?;
                if max_prob <= self.restore_gate_tau {
                    return restore.forward(xs, is_prefill);
                }
            }
        }
        self.forward_2bit(xs, is_prefill)
    }

    fn forward_2bit(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
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
        let mut ys = self.experts.forward(
            xs,
            &topk_weights,
            &topk_ids,
            &self.act,
            self.swiglu_limit,
            self.dtype,
            is_prefill,
        )?;
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}
