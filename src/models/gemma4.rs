use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear, VocabParallelLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{
    FusedMoe, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4,
};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, RotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

struct Gemma4Router {
    scale: Tensor,
    proj: candle_nn::Linear,
    per_expert_scale: Tensor,
    hidden_size: usize,
    top_k: usize,
    eps: f64,
}

impl Gemma4Router {
    fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        eps: f64,
        vb: &VarBuilderX,
        dtype: DType,
    ) -> Result<Self> {
        let scale = match &vb.0 {
            Either::Left(v) => v.get(hidden_size, "scale")?.to_dtype(dtype)?,
            Either::Right(v) => v
                .get(hidden_size, "scale")?
                .dequantize(v.device())?
                .to_dtype(dtype)?,
        };
        let proj_vb = vb.pp("proj");
        let proj_w = match &proj_vb.0 {
            Either::Left(v) => {
                use candle_nn::var_builder::Shard;
                v.get_with_hints((num_experts, hidden_size), "weight", Shard::default())?
                    .to_dtype(dtype)?
            }
            Either::Right(v) => v
                .get((num_experts, hidden_size), "weight")?
                .dequantize(v.device())?
                .to_dtype(dtype)?,
        };
        let proj = candle_nn::Linear::new(proj_w, None);

        let per_expert_scale = match &vb.0 {
            Either::Left(v) => v
                .get(num_experts, "per_expert_scale")?
                .to_dtype(DType::F32)?,
            Either::Right(v) => v
                .get(num_experts, "per_expert_scale")?
                .dequantize(v.device())?
                .to_dtype(DType::F32)?,
        };

        Ok(Self {
            scale,
            proj,
            per_expert_scale,
            hidden_size,
            top_k,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let rms = (xs_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)? + self.eps)?.sqrt()?;
        let normed = xs_f32.broadcast_div(&rms)?;

        let root_size = (self.hidden_size as f64).powf(-0.5);
        let scaled = (normed * root_size)?;
        let scale_f32 = self.scale.to_dtype(DType::F32)?;
        let scaled = scaled.broadcast_mul(&scale_f32.unsqueeze(0)?)?;

        let logits = scaled
            .to_dtype(self.proj.weight().dtype())?
            .apply(&self.proj)?;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;

        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let topk_indices = sorted_idx.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_weights = probs.contiguous()?.gather(&topk_indices, 1)?;

        let renorm = topk_weights.sum_keepdim(candle_core::D::Minus1)?;
        let topk_weights = topk_weights.broadcast_div(&renorm)?;

        let flat_idx = topk_indices.flatten_all()?.to_dtype(DType::U32)?;
        let scales = self
            .per_expert_scale
            .index_select(&flat_idx, 0)?
            .reshape(topk_indices.shape())?;
        let topk_weights = (topk_weights * scales)?;

        let topk_indices = topk_indices.to_dtype(DType::U32)?;
        Ok((topk_weights, topk_indices))
    }
}

enum Gemma4MoE {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
}

impl Gemma4MoE {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
        }
    }

    fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill),
            Self::FusedMoeNvfp4(m) => {
                m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
            }
            _ => self.forward(xs, is_prefill),
        }
    }
}

pub struct Gemma4DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    moe: Option<Gemma4MoE>,
    gemma4_router: Option<Gemma4Router>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    pre_feedforward_layernorm: NormX,
    post_feedforward_layernorm: NormX,
    post_feedforward_layernorm_1: Option<NormX>,
    post_feedforward_layernorm_2: Option<NormX>,
    pre_feedforward_layernorm_2: Option<NormX>,
    post_per_layer_input_norm: Option<NormX>,
    per_layer_input_gate: Option<ReplicatedLinear>,
    per_layer_projection: Option<ReplicatedLinear>,
    layer_scalar: Tensor,
    is_sliding: bool,
    rotary_emb: Arc<RotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    layer_idx: usize,
    #[allow(dead_code)]
    hidden_size_per_layer_input: Option<usize>,
}

impl Gemma4DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<RotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        config: &Config,
        layer_idx: usize,
        is_sliding: bool,
        enable_moe: bool,
        global_head_dim: usize,
        dtype: DType,
        intermediate_size: usize,
        hidden_size_per_layer_input: Option<usize>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        let swa_head_dim = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("swa_head_dim")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("head_dim")))
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize
        } else {
            256
        };

        let head_dim = if is_sliding {
            swa_head_dim
        } else {
            global_head_dim
        };

        let mut layer_config = config.clone();
        layer_config.head_dim = Some(head_dim);

        if !is_sliding {
            if let Some(extra) = &config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                if let Some(gkv) = v
                    .get("num_global_key_value_heads")
                    .or_else(|| {
                        v.get("text_config")
                            .and_then(|tc| tc.get("num_global_key_value_heads"))
                    })
                    .and_then(|v| v.as_u64())
                {
                    layer_config.num_key_value_heads = gkv as usize;
                }
            }
        }

        let k_eq_v = if !is_sliding {
            if let Some(extra) = &config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                v.get("attention_k_eq_v")
                    .or_else(|| {
                        v.get("text_config")
                            .and_then(|tc| tc.get("attention_k_eq_v"))
                    })
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            } else {
                false
            }
        } else {
            false
        };

        let self_attn = Attention::new_with_options(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            &layer_config,
            Some(1.0),
            sliding_window,
            dtype,
            k_eq_v,
            false,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp").clone()
            },
            comm.clone(),
            config.hidden_size,
            intermediate_size,
            &config.hidden_act,
            &config.quantization_config,
            &config.quant,
            false,
            dtype,
            "",
        )?;

        let moe_cfg_ref = config.moe_cfg.as_ref();
        let (moe, gemma4_router) = if enable_moe && moe_cfg_ref.is_some() {
            let mc = moe_cfg_ref.unwrap();
            let num_experts = mc.num_experts.unwrap();
            let top_k = mc.num_experts_per_tok;

            let m = if is_qvar_builder {
                Gemma4MoE::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if let Some(quant_config) = &config.quantization_config {
                if quant_config.quant_method == "mxfp4" {
                    Gemma4MoE::FusedMoeMxfp4(FusedMoeMxfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else if quant_config.quant_method == "nvfp4" {
                    Gemma4MoE::FusedMoeNvfp4(FusedMoeNvfp4::new_with_gate(
                        config,
                        vb.pp("router").pp("proj").clone(),
                        vb.pp("experts").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else {
                    panic!(
                        "Unsupported quantization for Gemma4 MoE: {}",
                        quant_config.quant_method
                    );
                }
            } else if config.quant.is_some() {
                Gemma4MoE::FusedMoeISQ(FusedMoeISQ::new_with_gate(
                    config,
                    vb.pp("router").pp("proj").clone(),
                    vb.pp("experts").clone(),
                    &vb,
                    comm.clone(),
                    dtype,
                )?)
            } else {
                Gemma4MoE::FusedMoe(FusedMoe::new_with_gate(
                    config,
                    vb.pp("router").pp("proj").clone(),
                    vb.pp("experts").clone(),
                    &vb,
                    comm.clone(),
                    dtype,
                )?)
            };

            let router = Gemma4Router::new(
                config.hidden_size,
                num_experts,
                top_k,
                config.rms_norm_eps,
                &vb.pp("router"),
                dtype,
            )?;

            (Some(m), Some(router))
        } else {
            (None, None)
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;
        let pre_feedforward_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
            DType::F32,
            false,
        )?;
        let post_feedforward_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
            DType::F32,
            false,
        )?;

        let (
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
        ) = if enable_moe && config.moe_cfg.is_some() {
            (
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_1"),
                    DType::F32,
                    false,
                )?),
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_2"),
                    DType::F32,
                    false,
                )?),
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("pre_feedforward_layernorm_2"),
                    DType::F32,
                    false,
                )?),
            )
        } else {
            (None, None, None)
        };

        let (post_per_layer_input_norm, per_layer_input_gate, per_layer_projection) =
            if let Some(pli_dim) = hidden_size_per_layer_input {
                let norm = rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                    DType::F32,
                    false,
                )?;
                let gate = ReplicatedLinear::load_no_bias(
                    config.hidden_size,
                    pli_dim,
                    vb.pp("per_layer_input_gate"),
                    &config.quantization_config,
                    &config.quant,
                    dtype,
                )?;
                let proj = ReplicatedLinear::load_no_bias(
                    pli_dim,
                    config.hidden_size,
                    vb.pp("per_layer_projection"),
                    &config.quantization_config,
                    &config.quant,
                    dtype,
                )?;
                (Some(norm), Some(gate), Some(proj))
            } else {
                (None, None, None)
            };

        let layer_scalar = match &vb.0 {
            Either::Left(v) => v.get(1, "layer_scalar")?.to_dtype(dtype)?,
            Either::Right(v) => v
                .pp("layer_output_scale")
                .get((1,), "weight")?
                .dequantize(&v.device())?
                .to_dtype(dtype)?,
        };

        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
            post_per_layer_input_norm,
            per_layer_input_gate,
            per_layer_projection,
            layer_scalar,
            is_sliding,
            rotary_emb,
            rotary_emb_local,
            layer_idx,
            hidden_size_per_layer_input,
            gemma4_router,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        sliding_mask: Option<&Vec<Tensor>>,
        full_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let xs = xs.clone();

        let residual = xs.clone();
        let normed = self.input_layernorm.forward(&xs)?;

        let mask = if self.is_sliding {
            sliding_mask
        } else {
            full_mask
        };

        let rope: Arc<dyn ApplyRotaryEmbedding> = if self.is_sliding {
            self.rotary_emb_local.clone()
        } else {
            self.rotary_emb.clone()
        };

        let attn_output =
            self.self_attn
                .forward(&normed, &Some(rope), mask, positions, cache, input_metadata)?;

        let mut xs = self.post_attention_layernorm.forward(&attn_output)?;
        xs = (xs + &residual)?;

        let residual = xs.clone();

        let mlp_input = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;

        let combined = if let Some(moe) = &self.moe {
            let mlp_normed = self
                .post_feedforward_layernorm_1
                .as_ref()
                .unwrap()
                .forward(&mlp_output)?;

            let residual_flat = residual.flatten(0, residual.rank() - 2)?;

            let moe_output = if let Some(router) = &self.gemma4_router {
                let (topk_weights, topk_ids) = router.forward(&residual_flat)?;
                let moe_input = self
                    .pre_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&residual_flat)?;
                moe.forward_with_routing(
                    &moe_input,
                    topk_weights,
                    topk_ids,
                    input_metadata.is_prefill,
                )?
            } else {
                let moe_input = self
                    .pre_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&residual_flat)?;
                moe.forward(&moe_input, input_metadata.is_prefill)?
            };
            let moe_output = moe_output.reshape(residual.shape())?;

            let moe_normed = self
                .post_feedforward_layernorm_2
                .as_ref()
                .unwrap()
                .forward(&moe_output)?;

            (mlp_normed + moe_normed)?
        } else {
            mlp_output
        };

        let combined = self.post_feedforward_layernorm.forward(&combined)?;
        let mut xs = (&residual + combined)?;

        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual_ple = xs.clone();
            let gated = gate
                .forward(&xs)?
                .apply(&candle_nn::Activation::GeluPytorchTanh)?;
            let gated = (gated * pli)?;
            let projected = proj.forward(&gated)?;
            xs = (&residual_ple + norm.forward(&projected)?)?;
        }

        xs.broadcast_mul(&self.layer_scalar.to_dtype(xs.dtype())?)
    }
}

pub struct Gemma4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<NormX>,
    layers: Vec<Gemma4DecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    embed_scale: f64,
    is_qvar_builder: bool,
    #[allow(dead_code)]
    layer_types: Vec<String>,
    hidden_size_per_layer_input: Option<usize>,
    num_hidden_layers: usize,
}

impl Gemma4ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();

        let layer_types: Vec<String> = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("layer_types")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("layer_types")))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or("sliding_attention").to_string())
                        .collect()
                })
                .unwrap_or_else(|| {
                    (0..config.num_hidden_layers)
                        .map(|i| {
                            if (i + 1) % 6 == 0 {
                                "full_attention".to_string()
                            } else {
                                "sliding_attention".to_string()
                            }
                        })
                        .collect()
                })
        } else {
            (0..config.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        };

        let enable_moe = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("enable_moe_block")
                .or_else(|| {
                    v.get("text_config")
                        .and_then(|tc| tc.get("enable_moe_block"))
                })
                .and_then(|v| v.as_bool())
                .unwrap_or(config.moe_cfg.is_some())
        } else {
            config.moe_cfg.is_some()
        };

        let global_head_dim = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("global_head_dim")
                .or_else(|| {
                    v.get("text_config")
                        .and_then(|tc| tc.get("global_head_dim"))
                })
                .and_then(|v| v.as_u64())
                .unwrap_or(config.head_dim.unwrap_or(256) as u64) as usize
        } else {
            config.head_dim.unwrap_or(256)
        };

        let rope_local_base_freq = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("rope_local_base_freq")
                .or_else(|| {
                    v.get("text_config")
                        .and_then(|tc| tc.get("rope_parameters"))
                        .and_then(|rp| rp.get("sliding_attention"))
                        .and_then(|sa| sa.get("rope_theta"))
                })
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0)
        } else {
            10000.0
        };

        let hidden_size_per_layer_input: Option<usize> =
            if let Some(extra) = &config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                v.get("hidden_size_per_layer_input")
                    .or_else(|| {
                        v.get("text_config")
                            .and_then(|tc| tc.get("hidden_size_per_layer_input"))
                    })
                    .and_then(|v| v.as_u64())
                    .filter(|&v| v > 0)
                    .map(|v| v as usize)
            } else {
                None
            };

        let num_kv_shared_layers: usize = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("num_kv_shared_layers")
                .or_else(|| {
                    v.get("text_config")
                        .and_then(|tc| tc.get("num_kv_shared_layers"))
                })
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
        } else {
            0
        };

        let use_double_wide_mlp = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("use_double_wide_mlp")
                .or_else(|| {
                    v.get("text_config")
                        .and_then(|tc| tc.get("use_double_wide_mlp"))
                })
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        } else {
            false
        };

        let first_kv_shared_layer = config
            .num_hidden_layers
            .saturating_sub(num_kv_shared_layers);

        let lm_prefix = if is_qvar_builder {
            "model"
        } else {
            "model.language_model"
        };

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp(&format!("{}.embed_tokens", lm_prefix)),
            dtype,
        )?;

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if let Some(pli_dim) = hidden_size_per_layer_input {
                let total_dim = pli_dim * config.num_hidden_layers;
                let (emb, _) = embedding(
                    config.vocab_size,
                    total_dim,
                    vb.pp(&format!("{}.embed_tokens_per_layer", lm_prefix)),
                    dtype,
                )?;

                let proj_vb = vb.pp(&format!("{}.per_layer_model_projection", lm_prefix));
                let proj_linear = match &proj_vb.0 {
                    Either::Left(pvb) => {
                        use candle_nn::var_builder::Shard;
                        let w = pvb.get_with_hints(
                            (total_dim, config.hidden_size),
                            "weight",
                            Shard::default(),
                        )?;
                        let w_f32 = w.to_dtype(DType::F32)?;
                        let w = if let Ok(scale) = pvb
                            .get_with_hints((total_dim, 1), "weight_scale", Shard::default())
                            .or_else(|_| {
                                pvb.get_with_hints(
                                    (total_dim, 1),
                                    "weight_scale_inv",
                                    Shard::default(),
                                )
                            }) {
                            let scale = scale.to_dtype(DType::F32)?;
                            w_f32.broadcast_mul(&scale)?.to_dtype(dtype)?
                        } else {
                            w_f32.to_dtype(dtype)?
                        };
                        Linear::new(w, None)
                    }
                    Either::Right(pvb) => {
                        let w = pvb
                            .get((total_dim, config.hidden_size), "weight")?
                            .dequantize(pvb.device())?
                            .to_dtype(dtype)?;
                        Linear::new(w, None)
                    }
                };

                let norm = rms_norm(
                    pli_dim,
                    config.rms_norm_eps,
                    vb.pp(&format!("{}.per_layer_projection_norm", lm_prefix)),
                    DType::F32,
                    false,
                )?;

                (Some(emb), Some(proj_linear), Some(norm))
            } else {
                (None, None, None)
            };

        let embed_scale = (config.hidden_size as f64).sqrt();

        let rope_dtype = if is_qvar_builder || config.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };

        let (global_rope_theta, partial_rotary_factor) = {
            let mut theta = config.rope_theta.unwrap_or(1_000_000.0);
            let mut prf = config.partial_rotary_factor.unwrap_or(0.25) as f64;
            if let Some(extra) = &config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                let fa = v
                    .get("text_config")
                    .and_then(|tc| tc.get("rope_parameters"))
                    .and_then(|rp| rp.get("full_attention"));
                if let Some(fa) = fa {
                    if let Some(t) = fa.get("rope_theta").and_then(|v| v.as_f64()) {
                        theta = t;
                    }
                    if let Some(p) = fa.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                        prf = p;
                    }
                }
            }
            (theta, prf)
        };
        let rope_angles = (partial_rotary_factor * global_head_dim as f64 / 2.0) as usize;
        let half_dim = global_head_dim / 2;

        let mut inv_freq_vec: Vec<f32> = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(
                1.0f32 / (global_rope_theta as f32).powf((2 * i) as f32 / global_head_dim as f32),
            );
        }
        for _ in rope_angles..half_dim {
            inv_freq_vec.push(0.0f32);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), &vb.device())?;
        let t = Tensor::arange(0u32, config.max_position_embeddings as u32, &vb.device())?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let rotary_emb = Arc::new(RotaryEmbedding {
            cos: freqs.cos()?.to_dtype(rope_dtype)?,
            sin: freqs.sin()?.to_dtype(rope_dtype)?,
            is_rope_i,
            rotary_dim: None,
            original_max_position_embeddings: None,
            llama_4_scaling_beta: None,
        });

        let swa_head_dim_for_rope = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("swa_head_dim")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("head_dim")))
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize
        } else {
            256
        };

        let mut local_config = config.clone();
        local_config.head_dim = Some(swa_head_dim_for_rope);
        local_config.partial_rotary_factor = None;

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            &local_config,
            &vb.device(),
            is_rope_i,
            Some(rope_local_base_freq),
            None,
            None,
        )?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let is_sliding = layer_types
                .get(i)
                .map(|t| t == "sliding_attention")
                .unwrap_or(true);
            let layer_prefix = format!("{}.layers.{}", lm_prefix, i);

            let layer_intermediate =
                if use_double_wide_mlp && num_kv_shared_layers > 0 && i >= first_kv_shared_layer {
                    config.intermediate_size * 2
                } else {
                    config.intermediate_size
                };

            let layer = Gemma4DecoderLayer::new(
                vb.pp(&layer_prefix),
                comm.clone(),
                rotary_emb.clone(),
                rotary_emb_local.clone(),
                config,
                i,
                is_sliding,
                enable_moe,
                global_head_dim,
                dtype,
                layer_intermediate,
                hidden_size_per_layer_input,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp(&format!("{}.norm", lm_prefix)),
            DType::F32,
            false,
        )?;

        let tie_word_embeddings = config.tie_word_embeddings.unwrap_or(true);
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings {
                vb.pp(&format!("{}.embed_tokens", lm_prefix))
            } else if is_qvar_builder {
                vb.pp("model.output")
            } else {
                vb.pp("lm_head")
            },
            comm.clone(),
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            embed_scale,
            is_qvar_builder,
            layer_types,
            hidden_size_per_layer_input,
            num_hidden_layers: config.num_hidden_layers,
        })
    }

    fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs =
            if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
                xs.to_dtype(DType::F32)?
            } else {
                xs
            };
        xs * self.embed_scale
    }

    fn get_per_layer_embeddings(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        let (emb_per_layer, pli_dim, proj, norm) = match (
            &self.embed_tokens_per_layer,
            self.hidden_size_per_layer_input,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(e), Some(d), Some(p), Some(n)) => (e, d, p, n),
            _ => return Ok(None),
        };

        let embedded = emb_per_layer.forward(input_ids)?;
        let embedded = (embedded * (pli_dim as f64).sqrt())?;

        let projected = inputs_embeds.apply(proj)?;
        let projected = (projected * (self.config.hidden_size as f64).powf(-0.5))?;

        let seq_len = input_ids.dim(0)?;
        let projected = projected.reshape((seq_len, self.num_hidden_layers, pli_dim))?;
        let projected = norm.forward(&projected)?;

        let embedded = embedded.reshape((seq_len, self.num_hidden_layers, pli_dim))?;
        let combined = ((projected + embedded)? * std::f64::consts::FRAC_1_SQRT_2)?;

        let mut per_layer = Vec::with_capacity(self.num_hidden_layers);
        for i in 0..self.num_hidden_layers {
            per_layer.push(combined.narrow(1, i, 1)?.squeeze(1)?);
        }
        Ok(Some(per_layer))
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let mut xs = self.embed_forward(input_ids)?;
        let per_layer_inputs = self.get_per_layer_embeddings(input_ids, &xs)?;

        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let full_mask = if input_metadata.is_prefill && !seqlens.is_empty() {
            let mut masks = Vec::new();
            let mut start = 0u32;
            for seq_offset in &seqlens {
                let seq_len = (seq_offset - start) as usize;
                if seq_len > 1 {
                    let mask = Tensor::zeros((seq_len, seq_len), self.dtype, &self.device)?;
                    attention_rs::mask::causal_mask(&mask, None)?;
                    masks.push(mask.unsqueeze(0)?.unsqueeze(0)?);
                } else {
                    masks.push(Tensor::zeros((1, 1, 1, 1), self.dtype, &self.device)?);
                }
                start = *seq_offset;
            }
            Some(masks)
        } else {
            None
        };

        let sliding_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), (i, layer)) in
                zip(kv_caches.iter(), self.layers.iter().enumerate())
            {
                let pli = per_layer_inputs.as_ref().map(|v| &v[i]);
                xs = layer.forward(
                    &xs,
                    pli,
                    sliding_mask.as_ref(),
                    full_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            xs.to_dtype(DType::F32)
        } else {
            let logits = if self.is_qvar_builder {
                self.lm_head.forward(&xs)?
            } else {
                self.lm_head
                    .forward(&xs.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)?
            };

            let final_logit_softcapping = if let Some(extra) = &self.config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                v.get("final_logit_softcapping")
                    .or_else(|| {
                        v.get("text_config")
                            .and_then(|tc| tc.get("final_logit_softcapping"))
                    })
                    .and_then(|v| v.as_f64())
                    .or(self.config.final_logit_softcapping)
            } else {
                self.config.final_logit_softcapping
            };

            let logits = if let Some(cap) = final_logit_softcapping {
                let scaled = (logits / cap)?;
                let tanh = scaled.tanh()?;
                (tanh * cap)?
            } else {
                logits
            };

            Ok(logits.to_dtype(DType::F32)?)
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        _embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        _embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, positions, kv_caches, input_metadata, true)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
