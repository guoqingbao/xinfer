// DFlash draft model: a small, fully-replicated dense transformer that drafts future tokens by
// reading the target model's hidden states (at `target_layer_ids`) instead of re-reading the KV
// cache. Loaded separately from the primary model (its own safetensors/gguf checkpoint) and
// replicated on every tensor-parallel rank (no `comm`, no NCCL in the draft path).

use crate::models::layers::distributed::ReplicatedLinear;
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor, D};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashConfig {
    pub mask_token_id: Option<u32>,
    pub target_layer_ids: Option<Vec<usize>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashModelConfig {
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
    pub block_size: usize,
    pub num_target_layers: usize,
    #[serde(default)]
    pub dflash_config: Option<DFlashConfig>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

impl DFlashModelConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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
}

/// Default target-layer selection for the draft model's cross-attention context.
/// - 1 draft layer -> the middle target layer.
/// - else -> `num_draft_layers` ids evenly spaced in `[1, num_target_layers - 3]`.
pub fn build_target_layer_ids(num_target_layers: usize, num_draft_layers: usize) -> Vec<usize> {
    if num_draft_layers == 1 {
        return vec![num_target_layers / 2];
    }
    let start = 1usize;
    let end = num_target_layers.saturating_sub(3);
    let span = end.saturating_sub(start);
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

/// Replicated (dense, non-TP-sharded) SiLU-Glu MLP. Weight names match a Qwen-style draft
/// checkpoint (`gate_proj`/`up_proj`/`down_proj`); the gguf aliases are resolved by the loader.
pub struct DFlashMLP {
    gate_proj: ReplicatedLinear,
    up_proj: ReplicatedLinear,
    down_proj: ReplicatedLinear,
}

impl DFlashMLP {
    pub fn new(vb: VarBuilderX, hidden_size: usize, intermediate_size: usize, dtype: DType) -> Result<Self> {
        let gate_proj = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            &None,
            &None,
            dtype,
        )?;
        let up_proj = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            &None,
            &None,
            dtype,
        )?;
        let down_proj = ReplicatedLinear::load_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let activated = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&activated)
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
        })
    }

    /// Cross-attention: Q from the draft/noise tokens, K/V from `concat(target_hidden, draft)`.
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

        let attn_weights = (q.matmul(&k.t()?)? * self.scaling)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct DFlashDecoderLayer {
    self_attn: DFlashAttention,
    mlp: DFlashMLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl DFlashDecoderLayer {
    pub fn new(vb: VarBuilderX, config: &DFlashModelConfig, dtype: DType) -> Result<Self> {
        let self_attn = DFlashAttention::new(vb.pp("self_attn"), config, dtype)?;
        let mlp = DFlashMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            dtype,
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

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
        let attn_output = self
            .self_attn
            .forward(&hidden_states, target_hidden, cos, sin)?;
        let hidden_states = (attn_output + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        residual + mlp_output
    }
}

pub struct DFlashRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl DFlashRotaryEmbedding {
    pub fn new(config: &DFlashModelConfig, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let rope_theta = config.rope_theta.unwrap_or(10000.0);
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
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
}

impl DFlashDraftModel {
    /// `vb` is a (separately-loaded) var builder pointing at the DFlash checkpoint root.
    /// The model is fully replicated: no `comm`, no NCCL.
    pub fn new(
        vb: &VarBuilderX,
        config: &DFlashModelConfig,
        dtype: DType,
        device: &Device,
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
            let layer = DFlashDecoderLayer::new(vb.pp(&format!("layers.{}", i)), config, dtype)?;
            layers.push(layer);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm"),
            DType::F32,
            false,
        )?;

        let rotary_emb = DFlashRotaryEmbedding::new(config, dtype, device)?;

        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm,
            rotary_emb,
            target_layer_ids,
            block_size: config.block_size,
            mask_token_id: config.mask_token_id(),
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Select the per-`target_layer_ids` hidden states from the target model's collector
    /// (`collector[0]` = embedding, `collector[1..]` = selected layer outputs), concatenate on
    /// the last dim, project through `fc`, and norm. Result: one projected context vector per
    /// input token.
    pub fn extract_and_project_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        let selected: Vec<Tensor> = (0..self.target_layer_ids.len())
            .map(|i| all_hidden_states[i + 1].clone())
            .collect();
        let concatenated = Tensor::cat(&selected, D::Minus1)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    /// Run the draft transformer over `noise_embedding` (the `[last_token, MASK..MASK]` block)
    /// cross-attending to `target_hidden` (projected context). Returns the final normed hidden
    /// for every position (`ctx_len + noise_len`).
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

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_target: usize, num_draft: usize) -> DFlashModelConfig {
        DFlashModelConfig {
            hidden_size: 64,
            num_hidden_layers: num_draft,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            head_dim: None,
            vocab_size: 1000,
            max_position_embeddings: 2048,
            rope_theta: None,
            attention_bias: None,
            block_size: num_draft + 1,
            num_target_layers: num_target,
            dflash_config: None,
            hidden_act: None,
            layer_types: None,
        }
    }

    #[test]
    fn target_layer_ids_single_draft_layer_is_midpoint() {
        assert_eq!(build_target_layer_ids(12, 1), vec![6]);
        assert_eq!(build_target_layer_ids(1, 1), vec![0]);
    }

    #[test]
    fn target_layer_ids_multi_draft_layer_spread() {
        // start=1, end=num_target-3, span=end-start; id_i = 1 + (i*span)/(n-1)
        assert_eq!(build_target_layer_ids(12, 3), vec![1, 5, 9]);
        assert_eq!(build_target_layer_ids(8, 2), vec![1, 4]);
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads() {
        let c = cfg(12, 1);
        assert_eq!(c.head_dim(), 64 / 4);
    }

    #[test]
    fn target_layer_ids_prefers_explicit_override() {
        let mut c = cfg(12, 1);
        c.dflash_config = Some(DFlashConfig {
            mask_token_id: Some(7),
            target_layer_ids: Some(vec![3, 7, 11]),
        });
        assert_eq!(c.target_layer_ids(), vec![3, 7, 11]);
        assert_eq!(c.mask_token_id(), Some(7));
    }

    #[test]
    fn target_layer_ids_default_uses_formula() {
        let c = cfg(12, 3);
        assert_eq!(c.target_layer_ids(), vec![1, 5, 9]);
        assert_eq!(c.mask_token_id(), None);
    }
}