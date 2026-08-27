use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{DType, Device, Result, Tensor, D};
use std::rc::Rc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashConfig {
    #[serde(default)]
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub selector_rank: Option<usize>,
    #[serde(default)]
    pub selector_top_k: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashModelConfig {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub block_size: Option<usize>,
    pub num_target_layers: usize,
    #[serde(default)]
    pub dflash_config: Option<DFlashConfig>,
    pub hidden_act: Option<String>,
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub is_causal: Option<bool>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

impl DFlashModelConfig {
    pub fn is_dflash2(&self) -> bool {
        self.architectures
            .as_ref()
            .and_then(|architectures| architectures.first())
            .is_some_and(|architecture| architecture.contains("DFlash2"))
            || self
                .dflash_config
                .as_ref()
                .is_some_and(|config| config.selector_top_k.is_some())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
            .or_else(|| self.dflash_config.as_ref().and_then(|c| c.block_size))
            .unwrap_or(1)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|parameters| parameters.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
            })
            .unwrap_or(10000.0)
    }

    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.target_layer_ids.clone())
            .unwrap_or_else(|| {
                build_target_layer_ids(self.num_target_layers, self.num_hidden_layers)
            })
    }

    pub fn mask_token_id(&self) -> Option<u32> {
        self.dflash_config.as_ref().and_then(|c| c.mask_token_id)
    }

    pub fn to_config(&self) -> Config {
        Config {
            architectures: None,
            head_dim: self.head_dim,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            max_model_len: None,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: Some(self.vocab_size),
            rope_theta: self.rope_theta,
            attention_bias: self.attention_bias,
            qkv_bias: None,
            attn_output_gate: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            tie_word_embeddings: None,
            bos_token_id: None,
            eos_token_id: None,
            use_sliding_window: None,
            sliding_window: None,
            max_window_layers: None,
            partial_rotary_factor: None,
            hidden_act: candle_nn::Activation::Silu,
            rope_scaling: None,
            quant: None,
            moe_cfg: None,
            kvcache_dtype: crate::utils::config::KvCacheDtype::Auto,
            quantization_config: None,
            is_multi_model: None,
            extra_config_json: None,
            is_f16_mode: false,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            mtp_enabled: false,
            mtp_max_verify_tokens: 0,
            expert_dtype: None,
        }
    }
}

fn build_target_layer_ids(num_target_layers: usize, num_draft_layers: usize) -> Vec<usize> {
    if num_draft_layers == 1 {
        return vec![num_target_layers / 2];
    }
    let start = 1usize;
    let end = num_target_layers.saturating_sub(3);
    let span = end - start;
    (0..num_draft_layers)
        .map(|i| start + (i * span) / (num_draft_layers - 1))
        .collect()
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let half = last_dim / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let q_len = q.dim(2)?;
    let cos_len = cos.dim(2)?;

    let cos_q = if cos_len > q_len {
        cos.narrow(2, cos_len - q_len, q_len)?
    } else {
        cos.clone()
    };
    let sin_q = if cos_len > q_len {
        sin.narrow(2, cos_len - q_len, q_len)?
    } else {
        sin.clone()
    };

    let q_embed = (q.broadcast_mul(&cos_q)? + rotate_half(q)?.broadcast_mul(&sin_q)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}

/// Optional sliding-window bias for draft queries over [ctx | noise].
/// DFlash2 checkpoints set `is_causal=false` (block diffusion / encoder-only), so we
/// do NOT apply a causal triangle — only a local window when configured.
fn build_dflash_attn_bias(
    ctx_len: usize,
    q_len: usize,
    sliding_window: Option<usize>,
    is_causal: bool,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let kv_len = ctx_len + q_len;
    let window = sliding_window.unwrap_or(usize::MAX);
    // Full attention when the whole sequence fits in the window and we are non-causal.
    if !is_causal && kv_len <= window {
        return Ok(None);
    }
    let mut bias = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let abs_q = ctx_len + i;
        let oldest = abs_q.saturating_add(1).saturating_sub(window);
        let newest = if is_causal {
            abs_q
        } else {
            (abs_q + window.saturating_sub(1)).min(kv_len.saturating_sub(1))
        };
        let row = i * kv_len;
        for j in 0..kv_len {
            if j < oldest || j > newest || (is_causal && j > abs_q) {
                bias[row + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Some(
        Tensor::from_vec(bias, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)?,
    ))
}

pub struct DFlashGroupedConv {
    base_kernel: Tensor,
    kernel_projection: ReplicatedLinear,
    block_size: usize,
    taps: usize,
    num_groups: usize,
}

impl DFlashGroupedConv {
    pub fn new(
        vb: VarBuilderX,
        hidden_size: usize,
        group_size: usize,
        taps: usize,
        block_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        if group_size == 0 || hidden_size % group_size != 0 {
            candle_core::bail!(
                "DFlash2 convolution group size {} must divide hidden size {}",
                group_size,
                hidden_size
            );
        }
        let base_kernel = vb.get((2, taps, hidden_size), "base_kernel")?;
        let num_groups = hidden_size / group_size;
        let kernel_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            2 * taps * num_groups,
            vb.pp("kernel_projection"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self {
            base_kernel,
            kernel_projection,
            block_size,
            taps,
            num_groups,
        })
    }

    fn convolve(&self, hidden_states: &Tensor, delta: &Tensor, side: usize) -> Result<Tensor> {
        attention_rs::topk::dflash_grouped_conv(
            hidden_states,
            delta,
            &self.base_kernel,
            self.block_size,
            side,
        )
    }

    pub fn prepare(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let coefficients = self.kernel_projection.forward(hidden_states)?.reshape((
            hidden_states.dim(0)?,
            2,
            self.taps,
            self.num_groups,
        ))?;
        Ok((
            self.convolve(hidden_states, &coefficients.narrow(1, 0, 1)?.squeeze(1)?, 0)?,
            coefficients.narrow(1, 1, 1)?.squeeze(1)?,
        ))
    }

    pub fn finish(&self, hidden_states: &Tensor, coefficients: &Tensor) -> Result<Tensor> {
        self.convolve(hidden_states, coefficients, 1)
    }
}

pub struct DFlashCandidateSelector {
    predecessor_codebook: Tensor,
    successor_codebook: Tensor,
    hidden_projection: ReplicatedLinear,
    top_k: usize,
}

impl DFlashCandidateSelector {
    pub fn new(
        vb: VarBuilderX,
        hidden_size: usize,
        vocab_size: usize,
        rank: usize,
        top_k: usize,
        dtype: DType,
    ) -> Result<Self> {
        let predecessor_codebook = vb.get((vocab_size, rank), "predecessor_codebook")?;
        let successor_codebook = vb.get((vocab_size, rank), "successor_codebook")?;
        let hidden_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            rank,
            vb.pp("hidden_projection"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self {
            predecessor_codebook,
            successor_codebook,
            hidden_projection,
            top_k,
        })
    }

    pub fn select(
        &self,
        hidden_states: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        let logits = logits.contiguous()?.to_dtype(DType::F32)?;
        let (unary_logits, candidate_ids) = attention_rs::topk::topk_select(&logits, self.top_k)?;
        let hidden = self
            .hidden_projection
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let selected = attention_rs::topk::dflash_select_candidates(
            &hidden,
            &unary_logits,
            &candidate_ids,
            &self.predecessor_codebook,
            &self.successor_codebook,
            &Tensor::from_vec(vec![anchor_token], (1,), hidden_states.device())?,
        )?;
        selected.to_vec1::<u32>()
    }
}

pub struct DFlashAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    q_norm: NormX,
    k_norm: NormX,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    sliding_window: Option<usize>,
    is_causal: bool,
    dtype: DType,
    device: Device,
}

impl DFlashAttention {
    pub fn new(vb: VarBuilderX, config: &DFlashModelConfig, dtype: DType) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
            &None,
            &None,
            dtype,
        )?;
        let k_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            &None,
            &None,
            dtype,
        )?;
        let v_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            &None,
            &None,
            dtype,
        )?;
        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
            &None,
            &None,
            dtype,
        )?;

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("q_norm"),
            DType::F32,
            false,
        )?;
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("k_norm"),
            DType::F32,
            false,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            sliding_window: config.sliding_window,
            is_causal: config.is_causal.unwrap_or(false),
            dtype,
            device: vb.device(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let q_len = hidden_states.dim(0)?;
        let ctx_len = target_hidden.dim(0)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((1, q_len, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = q.transpose(1, 2)?;

        let k_ctx = self.k_proj.forward(target_hidden)?;
        let k_noise = self.k_proj.forward(hidden_states)?;
        let v_ctx = self.v_proj.forward(target_hidden)?;
        let v_noise = self.v_proj.forward(hidden_states)?;

        let k = Tensor::cat(&[&k_ctx, &k_noise], 0)?;
        let v = Tensor::cat(&[&v_ctx, &v_noise], 0)?;
        let kv_len = ctx_len + q_len;
        let k = k.reshape((1, kv_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = k.transpose(1, 2)?;
        let v = v
            .reshape((1, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        let k = if num_kv_groups > 1 {
            k.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            k
        };
        let v = if num_kv_groups > 1 {
            v.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            v
        };

        let mut attn_weights = (q.matmul(&k.t()?)? * self.scaling)?;
        if let Some(attn_bias) = build_dflash_attn_bias(
            ctx_len,
            q_len,
            self.sliding_window,
            self.is_causal,
            self.dtype,
            &self.device,
        )? {
            attn_weights = attn_weights.broadcast_add(&attn_bias)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct DFlashDecoderLayer {
    self_attn: DFlashAttention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    attention_conv: Option<DFlashGroupedConv>,
    mlp_conv: Option<DFlashGroupedConv>,
}

impl DFlashDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &DFlashModelConfig,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = DFlashAttention::new(vb.pp("self_attn"), config, dtype)?;
        let mlp = MLP::new(
            vb.pp("mlp"),
            comm,
            config.hidden_size,
            config.intermediate_size,
            &candle_nn::Activation::Silu,
            &None,
            &None,
            false,
            dtype,
            "",
        )?;
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
        let (attention_conv, mlp_conv) = if config.is_dflash2() {
            let dflash_config = config.dflash_config.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing dflash_config".into())
            })?;
            let group_size = dflash_config.conv_group_size.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing conv_group_size".into())
            })?;
            let taps = dflash_config.conv_kernel_size.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing conv_kernel_size".into())
            })?;
            let block_size = config.block_size();
            (
                Some(DFlashGroupedConv::new(
                    vb.pp("attention_conv"),
                    config.hidden_size,
                    group_size,
                    taps,
                    block_size,
                    dtype,
                )?),
                Some(DFlashGroupedConv::new(
                    vb.pp("mlp_conv"),
                    config.hidden_size,
                    group_size,
                    taps,
                    block_size,
                    dtype,
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attention_conv,
            mlp_conv,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let (hidden_states, attention_coefficients) = if let Some(conv) = &self.attention_conv {
            let (hidden_states, coefficients) = conv.prepare(&hidden_states)?;
            (hidden_states, Some(coefficients))
        } else {
            (hidden_states, None)
        };
        let attn_output = self
            .self_attn
            .forward(&hidden_states, target_hidden, cos, sin)?;
        let attn_output = if let (Some(conv), Some(coefficients)) =
            (&self.attention_conv, &attention_coefficients)
        {
            conv.finish(&attn_output, coefficients)?
        } else {
            attn_output
        };
        let hidden_states = (attn_output + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let (hidden_states, mlp_coefficients) = if let Some(conv) = &self.mlp_conv {
            let (hidden_states, coefficients) = conv.prepare(&hidden_states)?;
            (hidden_states, Some(coefficients))
        } else {
            (hidden_states, None)
        };
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let mlp_output =
            if let (Some(conv), Some(coefficients)) = (&self.mlp_conv, &mlp_coefficients) {
                conv.finish(&mlp_output, coefficients)?
            } else {
                mlp_output
            };
        residual + mlp_output
    }
}

pub struct DFlashRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl DFlashRotaryEmbedding {
    pub fn new(config: &DFlashModelConfig, dtype: DType, device: &Device, yarn_factor: Option<f64>) -> Result<Self> {
        let head_dim = config.head_dim();
        let rope_theta = config.rope_theta();
        let max_pos = config.max_position_embeddings;

        // When the backbone uses dynamic YARN scaling, the draft model's RoPE must be
        // scaled by the same factor so its positional encoding stays consistent with the
        // target model at extended context lengths.
        if let Some(factor) = yarn_factor.filter(|f| *f > 1.0) {
            let (beta_fast, beta_slow, extrapolation_factor, attn_factor) =
                crate::utils::derive_yarn_parameters(factor);
            let yarn = crate::models::layers::rotary_emb::YarnRotaryEmbedding::new_yarn(
                dtype,
                device,
                rope_theta as f32,
                head_dim,
                max_pos,
                max_pos,
                beta_fast as f32,
                beta_slow as f32,
                attn_factor as f32,
                extrapolation_factor as f32,
                factor as f32,
            )?;
            // new_yarn returns half-width (table_len, head_dim/2) tables; DFlash uses the
            // doubled (interleaved) layout (table_len, head_dim).
            return Ok(Self {
                cos: Tensor::cat(&[&yarn.cos, &yarn.cos], D::Minus1)?,
                sin: Tensor::cat(&[&yarn.sin, &yarn.sin], D::Minus1)?,
            });
        }

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos_half = freqs.cos()?.to_dtype(dtype)?;
        let sin_half = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos: Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?,
            sin: Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?,
        })
    }

    pub fn get_cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        Ok((cos.unsqueeze(0)?, sin.unsqueeze(0)?))
    }
}

pub struct DFlashDraftModel {
    fc: ReplicatedLinear,
    hidden_norm: NormX,
    layers: Vec<DFlashDecoderLayer>,
    norm: NormX,
    rotary_emb: DFlashRotaryEmbedding,
    pub config: DFlashModelConfig,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: Option<u32>,
    device: Device,
    dtype: DType,
    candidate_selector: Option<DFlashCandidateSelector>,
}

impl DFlashDraftModel {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &DFlashModelConfig,
        dtype: DType,
        device: &Device,
        yarn_factor: Option<f64>,
    ) -> Result<Self> {
        let target_layer_ids = config.target_layer_ids();
        let fc_in_dim = target_layer_ids.len() * config.hidden_size;

        let fc = ReplicatedLinear::load_no_bias(
            fc_in_dim,
            config.hidden_size,
            vb.pp("fc"),
            &None,
            &None,
            dtype,
        )?;

        let hidden_norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("hidden_norm"),
            DType::F32,
            false,
        )?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = DFlashDecoderLayer::new(
                vb.pp(&format!("layers.{}", i)),
                comm.clone(),
                config,
                dtype,
            )?;
            layers.push(layer);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm"),
            DType::F32,
            false,
        )?;

        let rotary_emb = DFlashRotaryEmbedding::new(config, dtype, device, yarn_factor)?;
        let candidate_selector = if config.is_dflash2() {
            let dflash_config = config.dflash_config.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing dflash_config".into())
            })?;
            Some(DFlashCandidateSelector::new(
                vb.pp("candidate_selector"),
                config.hidden_size,
                config.vocab_size,
                dflash_config.selector_rank.ok_or_else(|| {
                    candle_core::Error::Msg("DFlash2 config is missing selector_rank".into())
                })?,
                dflash_config.selector_top_k.ok_or_else(|| {
                    candle_core::Error::Msg("DFlash2 config is missing selector_top_k".into())
                })?,
                dtype,
            )?)
        } else {
            None
        };

        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm,
            rotary_emb,
            target_layer_ids,
            block_size: config.block_size(),
            mask_token_id: config.mask_token_id(),
            config: config.clone(),
            device: device.clone(),
            dtype,
            candidate_selector,
        })
    }

    pub fn extract_and_project_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        let selected: Vec<Tensor> = (0..self.target_layer_ids.len())
            .map(|i| all_hidden_states[i + 1].clone())
            .collect();
        let concatenated = Tensor::cat(&selected, D::Minus1)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    /// Project target-layer hiddens already extracted into a draft context vector.
    /// Used after graph-safe verify forwards that write layer buffers in-place.
    pub fn project_layer_hiddens(&self, layer_hiddens: &[Tensor]) -> Result<Tensor> {
        if layer_hiddens.len() != self.target_layer_ids.len() {
            candle_core::bail!(
                "DFlash expected {} layer hiddens, got {}",
                self.target_layer_ids.len(),
                layer_hiddens.len()
            );
        }
        let concatenated = Tensor::cat(layer_hiddens, D::Minus1)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    pub fn forward(
        &self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let positions_flat = positions.flatten_all()?;
        let (cos, sin) = self.rotary_emb.get_cos_sin(&positions_flat)?;

        let mut hidden_states = noise_embedding.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, target_hidden, &cos, &sin)?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_dflash2(&self) -> bool {
        self.candidate_selector.is_some()
    }

    pub fn select_candidates(
        &self,
        hidden_states: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        self.candidate_selector
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 candidate selector is unavailable".into())
            })?
            .select(hidden_states, logits, anchor_token)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
