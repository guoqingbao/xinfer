// src/models/gemma3.rs

use crate::models::layers::attention::{Attention, NaiveAttention};
use crate::models::layers::distributed::{Comm, VocabParallelLinear};
use crate::models::layers::linear::LinearX;
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::NaiveMLP;
use crate::models::layers::others::{conv2d, embedding, layer_norm, rms_norm, AvgPool2d, NormX};
use crate::models::layers::rotary_emb::{
    ApplyRotaryEmbedding, RotaryEmbedding, ScalingRotaryEmbedding,
};
use crate::models::layers::{mlp::MLP as TextMLP, VarBuilderX};
use crate::utils::config::Config;
use crate::utils::image::ImageData;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig};
use either::Either;
use parking_lot::RwLock;
use std::collections::HashMap;
pub mod config;
use attention_rs::ops::NonZeroOp;
use config::{Gemma3Config, TextConfig, VisionConfig};
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

// =========================================================================
//  Vision Components (SigLIP-like)
// =========================================================================

#[allow(dead_code)]
struct Gemma3VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: candle_nn::Embedding,
    num_patches: usize,
    embed_dim: usize,
}

impl Gemma3VisionEmbeddings {
    fn new(vb: VarBuilderX, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let embed_dim = config.vision_config.hidden_size;
        let patch_size = config.vision_config.patch_size;
        let image_size = config.vision_config.image_size;

        let patch_embedding = conv2d(
            config.vision_config.num_channels,
            embed_dim,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
            true,
        )?;

        let num_patches = (image_size / patch_size).pow(2);
        let (position_embedding, _) = embedding(
            Some(num_patches),
            embed_dim,
            vb.pp("position_embedding"),
            dtype,
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches,
            embed_dim,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [Batch, Channels, Height, Width]
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        // Flatten: [Batch, EmbedDim, H', W'] -> [Batch, EmbedDim, NumPatches]
        let (b, _, _, _) = patch_embeds.dims4()?;
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?;

        // Create position ids
        let position_ids = Tensor::arange(0, self.num_patches as u32, &pixel_values.device())?
            .unsqueeze(0)?
            .broadcast_as((b, self.num_patches))?;

        let embeddings = (patch_embeds + self.position_embedding.forward(&position_ids)?)?;
        Ok(embeddings)
    }
}

struct DummyRotaryEmbedding {}

impl ApplyRotaryEmbedding for DummyRotaryEmbedding {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        _: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(Some((q.to_owned(), k.to_owned())))
    }

    fn get_original_max_position_embeddings(&self) -> Option<usize> {
        None
    }

    fn get_llama_4_scaling_beta(&self) -> Option<f64> {
        None
    }
}

unsafe impl Send for DummyRotaryEmbedding {}
unsafe impl Sync for DummyRotaryEmbedding {}

struct Gemma3VisionEncoderLayer {
    self_attn: NaiveAttention,
    mlp: NaiveMLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<DummyRotaryEmbedding>,
}

impl Gemma3VisionEncoderLayer {
    fn new(vb: VarBuilderX, _comm: Rc<Comm>, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        // Vision Attention usually doesn't need rotary embeddings, just absolute pos provided in embeddings
        // For this snippet, assuming Attention can handle "None" for RoPE or we pass a dummy.
        let mut key_mappings = HashMap::<String, String>::new();
        key_mappings.insert("o_proj".into(), "out_proj".into());
        let head_dim = config.vision_config.hidden_size / config.vision_config.num_attention_heads;
        let self_attn = NaiveAttention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            config.vision_config.num_attention_heads,
            config.vision_config.hidden_size,
            head_dim,
            None,
            dtype,
            key_mappings,
        )?;

        let mlp = NaiveMLP::new(
            vb.pp("mlp").clone(),
            config.vision_config.hidden_size,
            config.vision_config.intermediate_size,
            true,
            &["fc1", "fc2"],
            config.vision_config.hidden_act,
            dtype,
        )?;

        let input_layernorm = layer_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            true,
            vb.pp("layer_norm1"),
            dtype,
        )?;

        let post_attention_layernorm = layer_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            true,
            vb.pp("layer_norm2"),
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb: Arc::new(DummyRotaryEmbedding {}),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        // Vision encoder is usually bidirectional, no causal mask
        // We pass None for mask to imply full attention
        let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();
        let attn_output = self.self_attn.forward(&xs, &rope, &None, None);
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

struct Gemma3VisionTransformer {
    embeddings: Gemma3VisionEmbeddings,
    layers: Vec<Gemma3VisionEncoderLayer>,
    norm: NormX,
}

impl Gemma3VisionTransformer {
    fn new(vb: VarBuilderX, comm: Rc<Comm>, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let embeddings = Gemma3VisionEmbeddings::new(vb.pp("embeddings"), config, dtype)?;

        let mut layers = Vec::new();
        for i in 0..config.vision_config.num_hidden_layers {
            layers.push(Gemma3VisionEncoderLayer::new(
                vb.pp(&format!("encoder.layers.{}", i)),
                comm.clone(),
                config,
                dtype,
            )?);
        }

        let norm = layer_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            true,
            vb.pp("post_layernorm"),
            dtype,
        )?;

        Ok(Self {
            embeddings,
            layers,
            norm,
        })
    }

    fn forward(&self, pixel_values: &Tensor, _: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

struct Gemma3MultiModalProjector {
    projector: LinearX,
    norm: NormX,
    avg_pool: AvgPool2d,
    patches: usize,
}

impl Gemma3MultiModalProjector {
    fn new(vb: VarBuilderX, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let ws = match &vb.0 {
            Either::Left(v) => v.get(
                (
                    config.vision_config.hidden_size,
                    config.text_config.hidden_size,
                ),
                "mm_input_projection_weight",
            )?,
            _ => {
                todo!()
            }
        }
        .to_dtype(dtype)?;

        let projector = LinearX::new(ws.t()?, None, &None)?;
        let norm = rms_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
            dtype,
            true,
        )?;

        let patches = config.vision_config.image_size / config.vision_config.patch_size;
        let kernel_size = patches / config.mm_tokens_per_image.isqrt();
        let avg_pool = AvgPool2d::new(kernel_size, kernel_size);

        Ok(Self {
            projector,
            norm,
            avg_pool,
            patches,
        })
    }

    fn forward(&self, xs: &Tensor, _: Vec<(usize, usize)>) -> Result<Tensor> {
        // xs is [Batch, SeqLen (Patches), HiddenDim]
        let (bs, _, hidden_dim) = xs.dims3()?;

        // 1. Transpose to [Batch, Hidden, Patches] to prepare for spatial reshape
        let mut out = xs.transpose(1, 2)?;

        // 2. Reshape to spatial grid [Batch, Hidden, GridH, GridW]
        // self.patches is the grid size (e.g., 64 for 896/14)
        out = out
            .reshape((bs, hidden_dim, self.patches, self.patches))?
            .contiguous()?;

        // 3. Pool [Batch, Hidden, PooledH, PooledW]
        // If mm_tokens=256, kernel will be 4. 64->16.
        out = self.avg_pool.forward(&out)?;

        // 4. Flatten spatial dims back to sequence: [Batch, Hidden, NewSeqLen]
        out = out.flatten_from(2)?;

        // 5. Transpose back to transformer format: [Batch, NewSeqLen, Hidden]
        out = out.transpose(1, 2)?;

        // 6. Project
        out = self.norm.forward(&out)?;
        self.projector.forward(&out)
    }
}

// =========================================================================
//  Text Components (Gemma 3 Specifics)
// =========================================================================

pub struct Gemma3DecoderLayer {
    self_attn: Attention,
    mlp: TextMLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    pre_feedforward_layernorm: NormX,
    post_feedforward_layernorm: NormX,
    sliding_window: Option<usize>,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
}

impl Gemma3DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        config: &Gemma3Config,
        g_cfg: &Config,
        layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let cfg = &config.text_config;

        // Logic for Alternating Sliding Window
        let sliding_window = if (layer_idx + 1) % cfg.sliding_window_pattern != 0 {
            Some(cfg.sliding_window)
        } else {
            None
        };

        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            &g_cfg,
            Some(1. / (config.text_config.query_pre_attn_scalar as f32).sqrt()),
            sliding_window,
            dtype,
        )?;

        let mlp = TextMLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp").clone()
            },
            comm.clone(),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.hidden_activation,
            &g_cfg.quantization_config,
            &g_cfg.quant,
            false,
            dtype,
            "",
        )?;

        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            sliding_window,
            rotary_emb,
            rotary_emb_local,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        sliding_attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        // Select appropriate mask based on layer type
        let mask = if sliding_attention_mask.is_some() {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let rope: Arc<dyn ApplyRotaryEmbedding> = if self.sliding_window.is_some() {
            self.rotary_emb_local.clone()
        } else {
            self.rotary_emb.clone()
        };

        let attn_output =
            self.self_attn
                .forward(&xs, &Some(rope), mask, positions, cache, input_metadata)?;

        let mut xs = self.post_attention_layernorm.forward(&attn_output)?;
        xs = (xs + residual)?;

        let residual = &xs;

        let mlp_input = self.pre_feedforward_layernorm.forward(&xs)?;

        let mut mlp_output = self.mlp.forward(&mlp_input)?;
        mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;

        residual + mlp_output
    }
}

// =========================================================================
//  Main Model
// =========================================================================
#[allow(dead_code)]
pub struct Gemma3ForConditionalGeneration {
    // Vision
    vision_tower: Option<Gemma3VisionTransformer>,
    multi_modal_projector: Option<Gemma3MultiModalProjector>,

    // Text
    embed_tokens: candle_nn::Embedding, // ScaledEmbedding logic needed inside
    layers: Vec<Gemma3DecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,

    // Metadata
    device: Device,
    config: Gemma3Config,
    g_cfg: Config,
    dtype: DType,
    vocab_size: usize,
    embed_scale: f64,
    is_qvar_builder: bool,
}

impl Gemma3ForConditionalGeneration {
    fn synthesize_text_only_config(g_cfg: &Config) -> Gemma3Config {
        Gemma3Config {
            text_config: TextConfig {
                attention_bias: g_cfg.attention_bias.unwrap_or(false),
                head_dim: g_cfg.head_dim.unwrap_or(256),
                hidden_activation: g_cfg.hidden_act,
                hidden_size: g_cfg.hidden_size,
                intermediate_size: g_cfg.intermediate_size,
                num_attention_heads: g_cfg.num_attention_heads,
                num_hidden_layers: g_cfg.num_hidden_layers,
                num_key_value_heads: g_cfg.num_key_value_heads,
                rms_norm_eps: g_cfg.rms_norm_eps,
                rope_theta: g_cfg.rope_theta.unwrap_or(1_000_000.0),
                vocab_size: g_cfg.vocab_size.unwrap_or(262_208),
                sliding_window: g_cfg
                    .sliding_window
                    .unwrap_or(g_cfg.max_position_embeddings),
                attn_logit_softcapping: g_cfg.attn_logit_softcapping,
                final_logit_softcapping: g_cfg.final_logit_softcapping,
                query_pre_attn_scalar: g_cfg.head_dim.unwrap_or(256),
                max_position_embeddings: g_cfg.max_position_embeddings,
                quantization_config: g_cfg.quantization_config.clone(),
                tie_word_embeddings: g_cfg.tie_word_embeddings.unwrap_or(true),
                rope_local_base_freq: 10_000.0,
                sliding_window_pattern: 6,
                rope_scaling: g_cfg.rope_scaling.clone(),
                quant: g_cfg.quant.clone(),
            },
            vision_config: VisionConfig {
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                num_channels: 3,
                image_size: 224,
                patch_size: 16,
                hidden_act: candle_nn::Activation::GeluPytorchTanh,
                layer_norm_eps: 1e-6,
            },
            image_token_index: 0,
            mm_tokens_per_image: 0,
            eos_token_id: g_cfg.eos_token_id.clone(),
            has_vision: false,
        }
    }

    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        g_cfg: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let config: Gemma3Config = if let Some(extra_config_json) = g_cfg.extra_config_json.as_ref()
        {
            serde_json::from_str(extra_config_json).map_err(candle_core::Error::wrap)?
        } else {
            crate::log_error!(
                "Vision tower is disabled because no multimodal vision config was found."
            );
            Self::synthesize_text_only_config(g_cfg)
        };

        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();
        let text_cfg = &config.text_config;

        // 1. Text Embeddings
        // Gemma scales embeddings by sqrt(dim)
        let (embed_tokens, vocab_size) = embedding(
            Some(text_cfg.vocab_size),
            text_cfg.hidden_size,
            if is_qvar_builder {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("language_model.model.embed_tokens")
            },
            dtype,
        )?;

        let embed_scale = (config.text_config.hidden_size as f64).sqrt();
        crate::log_info!("Loading vision tower...");

        // 2. Vision & Projector (Optional load based on config)
        let vision_tower = if config.has_vision {
            Some(Gemma3VisionTransformer::new(
                vb.pp("vision_tower.vision_model"),
                comm.clone(),
                &config,
                dtype,
            )?)
        } else {
            None
        };

        let multi_modal_projector = if config.has_vision {
            Some(Gemma3MultiModalProjector::new(
                vb.pp("multi_modal_projector"),
                &config,
                dtype,
            )?)
        } else {
            None
        };

        crate::log_info!("Loading language model...");

        // 3. RoPE
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            &g_cfg,
            &vb.device(),
            is_rope_i,
            g_cfg.rope_theta,
        )?);

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            &g_cfg,
            &vb.device(),
            is_rope_i,
            Some(config.text_config.rope_local_base_freq),
            None,
            None,
        )?);

        // 4. Layers
        let mut layers = Vec::new();
        for i in 0..text_cfg.num_hidden_layers {
            let layer = Gemma3DecoderLayer::new(
                vb.pp(&format!("language_model.model.layers.{}", i)),
                comm.clone(),
                rotary_emb.clone(),
                rotary_emb_local.clone(),
                &config,
                &g_cfg,
                i,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        // 5. Final Norm & Head
        let norm = rms_norm(
            text_cfg.hidden_size,
            text_cfg.rms_norm_eps,
            vb.pp("language_model.model.norm"),
            if is_qvar_builder || g_cfg.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        let lm_head = VocabParallelLinear::load_no_bias(
            text_cfg.hidden_size,
            vocab_size,
            if text_cfg.tie_word_embeddings {
                vb.pp("language_model.model.embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            comm.clone(),
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            embed_scale,
            is_qvar_builder,
            g_cfg: g_cfg.clone(),
        })
    }

    fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = if (self.is_qvar_builder || self.g_cfg.quant.is_some()) && xs.dtype() != DType::F32
        {
            xs.to_dtype(DType::F32)?
        } else {
            xs
        };
        xs * self.embed_scale
    }

    fn vision_tower(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let image_outputs = self
            .vision_tower
            .as_ref()
            .unwrap()
            .forward(image_features, image_sizes.clone())?;
        let selected_image_feature = image_outputs;
        self.multi_modal_projector
            .as_ref()
            .unwrap()
            .forward(&selected_image_feature, image_sizes)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let text_cfg = &self.config.text_config;
        // 1. Prepare Text Embeddings (Scaled)
        let mut xs = self.embed_forward(input_ids)?;

        // vision projection and embedding
        if let Some(images) = images {
            if !self.config.has_vision {
                crate::log_warn!("Ignoring image inputs because the vision tower is disabled.");
            } else {
                let image_mask = input_ids.eq(self.config.image_token_index as u32)?;
                let image_mask = image_mask
                    .unsqueeze(candle_core::D::Minus1)?
                    .broadcast_as(xs.shape().clone())?
                    .to_dtype(DType::U32)?;

                let mut image_tensor = images.to_tensor_f32(&xs.device())?.to_dtype(self.dtype)?;
                let mut image_sizes = images.patches.clone();
                let num_images = image_tensor.dim(0)?;
                assert!(
                    num_images == image_sizes.len(),
                    "Input image and patch dim mismatch!"
                );
                if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                    image_tensor = image_tensor.narrow(
                        0,
                        images.image_idx as usize,
                        num_images - images.image_idx as usize,
                    )?;
                    image_sizes = image_sizes[images.image_idx as usize..].to_vec();
                    crate::log_warn!(
                        "Slicing images: start idx {} -> {:?}",
                        images.image_idx,
                        image_tensor.shape()
                    );
                }

                let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
                if indices.shape().dim(0)? > 0 {
                    let image_features = self.vision_tower(&image_tensor, image_sizes)?;
                    let hidden = xs.dim(D::Minus1)?;
                    let indices_len = indices.shape().dim(0)?;
                    if indices_len % hidden != 0 {
                        candle_core::bail!(
                            "image indices length {} not divisible by hidden size {}",
                            indices_len,
                            hidden
                        );
                    }
                    let tokens_in_chunk = indices_len / hidden;
                    let total_tokens = image_features.dim(0)?;
                    let start = images.image_token_offset.min(total_tokens);
                    let end = start + tokens_in_chunk;
                    if end > total_tokens {
                        candle_core::bail!(
                            "image token slice out of range: start {}, len {}, total {}",
                            start,
                            tokens_in_chunk,
                            total_tokens
                        );
                    }
                    let image_features = if start > 0 || end < total_tokens {
                        image_features.narrow(0, start, tokens_in_chunk)?
                    } else {
                        image_features
                    };

                    let mut x_flat = xs.flatten_all()?;
                    let image_flat = image_features.flatten_all()?.to_dtype(xs.dtype())?;

                    x_flat = x_flat.scatter_add(
                        &indices,
                        &(image_flat - x_flat.gather(&indices, 0)?)?,
                        0,
                    )?;
                    xs = x_flat.reshape(xs.shape())?;
                } else {
                    crate::log_info!(
                        "Skip image embedding because no image tokens found in this chunk!"
                    );
                }
            }
        }

        // Followings are language model logics
        // 3. Prepare Masks
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        // Full Global Mask
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            None, // No window for global
            input_metadata.is_prefill,
        );

        // Sliding Window Mask
        let sliding_attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            Some(text_cfg.sliding_window),
            input_metadata.is_prefill,
        );

        // 4. Pass through layers
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), (i, layer)) in
                zip(kv_caches.iter(), self.layers.iter().enumerate())
            {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    if i + 1 % self.config.text_config.sliding_window_pattern != 0 {
                        sliding_attention_mask.as_ref()
                    } else {
                        None
                    },
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        }

        // 5. Final Norm & Projection
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
                self.lm_head.forward(&xs.to_dtype(self.dtype)?)?
            };

            // Final Logit Softcapping (Gemma Specific)
            if let Some(cap) = text_cfg.final_logit_softcapping {
                // tanh(logits / cap) * cap
                let scaled = (logits / cap)?;
                let tanh = scaled.tanh()?;
                tanh * cap
            } else {
                Ok(logits)
            }
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
            true,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        todo!()
    }
}
