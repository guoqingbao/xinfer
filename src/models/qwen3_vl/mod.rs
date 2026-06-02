use parking_lot::RwLock;
use parking_lot::RwLockWriteGuard;
use std::rc::Rc;
use std::sync::Arc;
pub mod config;
pub mod input;
pub mod vision;

use crate::models::layers::VarBuilderX;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::models::qwen3_5::Qwen3_5ForCausalLM;
use crate::models::qwen3_5_moe::Qwen3_5MoEForCausalLM;
use crate::models::qwen3_moe::Qwen3MoEForCausalLM;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use crate::{models::layers::distributed::Comm, utils::image::ImageData};
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use config::Qwen3VLConfig;
use vision::Qwen3VLVisionModel;

pub enum Qwen3TextModel {
    Dense(Qwen3ForCausalLM),
    MoE(Qwen3MoEForCausalLM),
    Dense35(Qwen3_5ForCausalLM),
    MoE35(Qwen3_5MoEForCausalLM),
}

#[allow(dead_code)]
pub struct Qwen3VLForConditionalGeneration {
    text_model: Qwen3TextModel,
    vision_model: Option<Qwen3VLVisionModel>,
    spatial_merge_size: usize,
    image_token_id: Option<u32>,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

pub(crate) fn try_parse_multimodal_extra_config(config: &Config) -> Result<Option<Qwen3VLConfig>> {
    let Some(extra_config_json) = config.extra_config_json.as_ref() else {
        return Ok(None);
    };
    let raw: serde_json::Value =
        serde_json::from_str(extra_config_json).map_err(candle_core::Error::wrap)?;
    if raw.get("vision_config").is_none() {
        return Ok(None);
    }
    let mut cfg: Qwen3VLConfig = serde_json::from_value(raw).map_err(candle_core::Error::wrap)?;
    cfg.text_config = config.clone();
    Ok(Some(cfg))
}

impl Qwen3VLForConditionalGeneration {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let mut config_text = config.clone();
        let mut vision_model = None;
        let mut spatial_merge_size = 0;
        let mut image_token_id = None;
        let mut vision_start_token_id = 0;
        let mut vision_end_token_id = 0;

        if let Some(cfg) = try_parse_multimodal_extra_config(config)? {
            if let Some(mut qcfg) = cfg.quantization_config.clone() {
                qcfg.normalize_compressed_tensors();
                config_text.quantization_config = Some(qcfg);
            }

            spatial_merge_size = cfg.vision_config.spatial_merge_size;
            image_token_id = Some(cfg.image_token_id);
            vision_start_token_id = cfg.vision_start_token_id;
            vision_end_token_id = cfg.vision_end_token_id;

            let vision_vb = if vb.is_qvar_builder() {
                vb.aux()
            } else if vb.has_key("vision_tower.patch_embed.proj.weight") {
                // MLX-style prefix: vision_tower.*
                Some(vb.pp("vision_tower"))
            } else {
                Some(vb.pp("model.visual"))
            };
            if let Some(vision_vb) = vision_vb {
                crate::log_info!("Loading vision tower...");
                vision_model = Some(Qwen3VLVisionModel::new(
                    &cfg.vision_config,
                    &vision_vb,
                    dtype,
                    device,
                )?);
            } else {
                crate::log_error!(
                    "Vision tower is disabled because no auxiliary GGUF mmproj file was found."
                );
            }
        } else {
            crate::log_error!(
                "Vision tower is disabled because no multimodal vision config (or weight) was found."
            );
        }

        let arch = config
            .architectures
            .as_ref()
            .and_then(|archs| archs.first())
            .map(|s| s.as_str())
            .unwrap_or("Qwen3VLForConditionalGeneration");
        crate::log_info!("Loading language model...");

        let next_is_moe = config_text
            .moe_cfg
            .as_ref()
            .and_then(|m| m.num_experts)
            .unwrap_or(0)
            > 0;
        let text_prefix = if vb.is_qvar_builder() {
            None
        } else if vb.has_key("language_model.model.embed_tokens.weight")
            || vb.has_key("language_model.model.embed_tokens.scales")
        {
            // MLX-style prefix: language_model.model.*
            Some("language_model.model.".to_string())
        } else {
            Some("model.language_model.".to_string())
        };

        let text_model = match arch {
            "Qwen3VLMoeForConditionalGeneration" => {
                Qwen3TextModel::MoE(Qwen3MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    text_prefix.clone(),
                )?)
            }
            "Qwen3_5MoeForConditionalGeneration" => {
                Qwen3TextModel::MoE35(Qwen3_5MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    text_prefix.clone(),
                )?)
            }
            "Qwen3_5ForConditionalGeneration" => {
                Qwen3TextModel::Dense35(Qwen3_5ForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    text_prefix.clone(),
                )?)
            }
            "Qwen3NextForConditionalGeneration" if next_is_moe => {
                Qwen3TextModel::MoE35(Qwen3_5MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    text_prefix.clone(),
                )?)
            }
            "Qwen3NextForConditionalGeneration" => {
                Qwen3TextModel::Dense35(Qwen3_5ForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    text_prefix.clone(),
                )?)
            }
            _ => Qwen3TextModel::Dense(Qwen3ForCausalLM::new_with_prefix(
                &vb,
                comm.clone(),
                &config_text,
                dtype,
                is_rope_i,
                device,
                progress_reporter,
                text_prefix,
            )?),
        };

        Ok(Self {
            text_model,
            vision_model,
            spatial_merge_size,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        let (mut input_embeds, dtype) = match &self.text_model {
            Qwen3TextModel::Dense(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::Dense35(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE35(m) => (m.embed_forward(input_ids)?, m.dtype()),
        };
        let device = input_embeds.device().clone();
        let mut visual_pos_masks: Option<Tensor> = None;
        let mut deepstack_visual_embeds: Option<Vec<Tensor>> = None;

        if let Some(images) = images {
            let Some(vision_model) = self.vision_model.as_ref() else {
                crate::log_warn!("Ignoring image inputs because the vision tower is disabled.");
                return match &self.text_model {
                    Qwen3TextModel::Dense(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::MoE(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::Dense35(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::MoE35(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                };
            };
            let mut pixel_values = images.to_tensor_f32(&device)?.to_dtype(dtype)?;
            let mut patches = Vec::new();
            for (h, w) in &images.patches {
                patches.extend(vec![1, *h as u32, *w as u32]);
            }
            let mut image_grid_thw = Tensor::from_vec(patches, (images.patches.len(), 3), &device)?;
            let num_images = pixel_values.dim(0)?;
            assert!(
                num_images == image_grid_thw.dim(0)?,
                "Input image and patch dim mismatch!"
            );
            if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                pixel_values = pixel_values.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
                image_grid_thw = image_grid_thw.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
                crate::log_warn!(
                    "Slicing images: start idx {} -> {:?}",
                    images.image_idx,
                    pixel_values.shape()
                );
            }

            let dims = pixel_values.dims();
            if dims.len() == 3 {
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }
            let (image_embeds, deepstack_image_embeds) =
                vision_model.forward(&pixel_values, &image_grid_thw)?;

            let image_embeds = image_embeds
                .to_device(&device)?
                .to_dtype(input_embeds.dtype())?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(input_embeds.dtype()))
                .collect::<Result<Vec<_>>>()?;

            let Some(image_token_id) = self.image_token_id else {
                crate::log_warn!(
                    "Ignoring image inputs because image token metadata is unavailable."
                );
                return match &self.text_model {
                    Qwen3TextModel::Dense(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::MoE(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::Dense35(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                    Qwen3TextModel::MoE35(m) => m.forward_with_deepstack(
                        &input_embeds,
                        &positions,
                        kv_caches,
                        input_metadata,
                        true,
                        &visual_pos_masks,
                        &deepstack_visual_embeds,
                    ),
                };
            };
            let image_mask = input_ids.eq(image_token_id)?;
            visual_pos_masks = Some(image_mask.to_dtype(DType::U8)?);

            let image_mask = image_mask
                .unsqueeze(candle_core::D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            use attention_rs::ops::NonZeroOp;
            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.shape().dim(0)? > 0 {
                let hidden = input_embeds.dim(D::Minus1)?;
                let indices_len = indices.shape().dim(0)?;
                if indices_len % hidden != 0 {
                    candle_core::bail!(
                        "image indices length {} not divisible by hidden size {}",
                        indices_len,
                        hidden
                    );
                }
                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_embeds.dim(0)?;
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
                let image_embeds = if start > 0 || end < total_tokens {
                    image_embeds.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_embeds
                };
                let deepstack_image_embeds = deepstack_image_embeds
                    .into_iter()
                    .map(|t| {
                        if start > 0 || end < total_tokens {
                            t.narrow(0, start, tokens_in_chunk)
                        } else {
                            Ok(t)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut x_flat = input_embeds.flatten_all()?;
                let image_flat = image_embeds.flatten_all()?;

                x_flat = x_flat.scatter_add(
                    &indices,
                    &(image_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
                deepstack_visual_embeds = Some(deepstack_image_embeds);
            } else {
                crate::log_info!(
                    "Skip image embedding because no image tokens found in this chunk!"
                );
            }
        }

        match &self.text_model {
            Qwen3TextModel::Dense(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::Dense35(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE35(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
        }
    }

    pub fn get_vocab_size(&self) -> usize {
        match &self.text_model {
            Qwen3TextModel::Dense(m) => m.get_vocab_size(),
            Qwen3TextModel::MoE(m) => m.get_vocab_size(),
            Qwen3TextModel::Dense35(m) => m.get_vocab_size(),
            Qwen3TextModel::MoE35(m) => m.get_vocab_size(),
        }
    }

    pub fn uses_hybrid_mamba_text_model(&self) -> bool {
        matches!(
            &self.text_model,
            Qwen3TextModel::Dense35(_) | Qwen3TextModel::MoE35(_)
        )
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.release_sequence_state(sequence_id),
            Qwen3TextModel::MoE35(m) => m.release_sequence_state(sequence_id),
            _ => {}
        }
    }

    pub fn ensure_mamba_slots_for_sequences(
        &self,
        sequence_ids: &[usize],
    ) -> Result<Option<Vec<usize>>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => {
                Ok(Some(m.ensure_mamba_slots_for_sequences(sequence_ids)?))
            }
            Qwen3TextModel::MoE35(m) => Ok(Some(m.ensure_mamba_slots_for_sequences(sequence_ids)?)),
            _ => Ok(None),
        }
    }

    pub fn get_mamba_slots_for_sequences(
        &self,
        sequence_ids: &[usize],
    ) -> Result<Option<Vec<usize>>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => Ok(Some(m.get_mamba_slots_for_sequences(sequence_ids)?)),
            Qwen3TextModel::MoE35(m) => Ok(Some(m.get_mamba_slots_for_sequences(sequence_ids)?)),
            _ => Ok(None),
        }
    }

    pub fn lock_mamba_cache_for_graph(&self) -> Option<RwLockWriteGuard<'_, MambaCache>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => Some(m.lock_mamba_cache_for_graph()),
            Qwen3TextModel::MoE35(m) => Some(m.lock_mamba_cache_for_graph()),
            _ => None,
        }
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.preallocate_mamba_cache(max_num_seqs),
            Qwen3TextModel::MoE35(m) => m.preallocate_mamba_cache(max_num_seqs),
            _ => Ok(()),
        }
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.set_mamba_prefix_cache_capacity(capacity),
            Qwen3TextModel::MoE35(m) => m.set_mamba_prefix_cache_capacity(capacity),
            _ => {}
        }
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.capture_mamba_prefix_state(seq_id, hash, preserve),
            Qwen3TextModel::MoE35(m) => m.capture_mamba_prefix_state(seq_id, hash, preserve),
            _ => Ok(true),
        }
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.has_mamba_prefix_state(hash),
            Qwen3TextModel::MoE35(m) => m.has_mamba_prefix_state(hash),
            _ => true,
        }
    }

    pub fn remove_mamba_prefix_state(&self, hash: u64) -> bool {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.remove_mamba_prefix_state(hash),
            Qwen3TextModel::MoE35(m) => m.remove_mamba_prefix_state(hash),
            _ => true,
        }
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.restore_mamba_prefix_state(seq_id, hash),
            Qwen3TextModel::MoE35(m) => m.restore_mamba_prefix_state(seq_id, hash),
            _ => Ok(true),
        }
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.reset_mamba_cache(),
            Qwen3TextModel::MoE35(m) => m.reset_mamba_cache(),
            _ => Ok(()),
        }
    }
}
