use crate::utils::config::ModelType;
use crate::utils::Config;
use candle_core::{DType, Device, Result, Storage, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
pub const IMAGE_PLACEHOLDER: &str = "<|XINFER-IMAGE|>";
pub const PLACEHOLDER: &str = "<|XINFER-PLACEHOLDER|>";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageData {
    pub raw: Vec<u8>,
    pub shape: Vec<usize>,
    pub patches: Vec<(usize, usize)>,
    pub image_idx: i32,
    #[serde(default)]
    pub image_token_offset: usize,
    #[serde(default)]
    pub tokens_per_image: Vec<usize>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
}

impl ImageData {
    pub fn to_tensor_f32(&self, device: &Device) -> Result<Tensor> {
        let floats: &[f32] = bytemuck::cast_slice(&self.raw);
        Tensor::from_slice(floats, self.shape.clone(), device)
    }
}

pub fn compute_tokens_per_image(
    cfg: &ImageProcessConfig,
    image_sizes: &[(usize, usize)],
) -> Vec<usize> {
    if image_sizes.is_empty() {
        return Vec::new();
    }
    match cfg.model_type {
        ModelType::Gemma3 => {
            if let Some(tokens) = cfg.mm_tokens_per_image {
                return vec![tokens; image_sizes.len()];
            }
            let denom = cfg.patch_size * cfg.spatial_merge_size;
            if denom == 0 {
                return vec![0; image_sizes.len()];
            }
            image_sizes
                .iter()
                .map(|&(h, w)| (h / denom) * (w / denom))
                .collect()
        }
        ModelType::Qwen3VL => {
            let merge = cfg.spatial_merge_size;
            let merge_area = merge * merge;
            image_sizes
                .iter()
                .map(|&(h, w)| {
                    if merge_area == 0 {
                        0
                    } else {
                        (h * w) / merge_area
                    }
                })
                .collect()
        }
        ModelType::LLaMa4 => {
            if let Some(tokens) = cfg.mm_tokens_per_image {
                return vec![tokens; image_sizes.len()];
            }
            vec![0; image_sizes.len()]
        }
        _ => {
            let denom = cfg.patch_size * cfg.spatial_merge_size;
            if denom == 0 {
                return vec![0; image_sizes.len()];
            }
            image_sizes
                .iter()
                .map(|&(h, w)| (h / denom) * (w / denom))
                .collect()
        }
    }
}

pub fn compute_image_slice(
    token_ids: &[u32],
    num_cached_tokens: usize,
    images: &ImageData,
) -> Option<(i32, usize)> {
    let base_idx = images.image_idx;
    if base_idx < 0 {
        return None;
    }
    let num_images = if !images.tokens_per_image.is_empty() {
        images.tokens_per_image.len()
    } else {
        images.patches.len()
    };
    if num_images == 0 {
        return None;
    }
    let base_idx = base_idx as usize;
    if num_cached_tokens == 0 {
        return if base_idx < num_images {
            Some((base_idx as i32, 0))
        } else {
            None
        };
    }
    let cached_len = num_cached_tokens.min(token_ids.len());
    if cached_len == 0 {
        return if base_idx < num_images {
            Some((base_idx as i32, 0))
        } else {
            None
        };
    }

    let Some(image_token_id) = images.image_token_id else {
        return if base_idx < num_images {
            Some((base_idx as i32, 0))
        } else {
            None
        };
    };
    if images.tokens_per_image.is_empty() {
        return if base_idx < num_images {
            Some((base_idx as i32, 0))
        } else {
            None
        };
    }

    let cached_tokens = &token_ids[..cached_len];
    let cached_image_tokens = cached_tokens
        .iter()
        .filter(|&&id| id == image_token_id)
        .count();
    let mut remaining = cached_image_tokens;
    let mut prefix_idx = 0usize;
    let mut token_offset = 0usize;
    for &tokens in &images.tokens_per_image {
        if tokens == 0 {
            break;
        }
        if remaining >= tokens {
            remaining -= tokens;
            prefix_idx += 1;
        } else {
            token_offset = remaining;
            break;
        }
    }

    let mut image_idx = prefix_idx;
    if base_idx > image_idx {
        image_idx = base_idx;
        token_offset = 0;
    }
    if image_idx >= num_images {
        return None;
    }
    let image_idx = image_idx.min(i32::MAX as usize) as i32;
    Some((image_idx, token_offset))
}
// load from url
pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    crate::log_info!("Start downloading image from {}", url);
    let bytes = reqwest::blocking::get(url)
        .map_err(candle_core::Error::wrap)?
        .bytes()
        .map_err(candle_core::Error::wrap)?;
    let img = image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)?;
    Ok(img)
}

// load from "data:image/jpeg;base64,XXXXX"
pub fn load_image_from_base64(data: &str) -> Result<DynamicImage> {
    use base64::prelude::{Engine as _, BASE64_STANDARD};
    let base64_part = data.split(",").last().unwrap_or(data);
    let bytes = BASE64_STANDARD
        .decode(base64_part)
        .map_err(candle_core::Error::wrap)?;
    let img = image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)?;
    Ok(img)
}

pub fn get_tensor_raw_data(t: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape = t.dims().to_vec();
    let (storage, _) = t.storage_and_layout();
    let storage = match &*storage {
        Storage::Cpu(p) => p,
        _ => candle_core::bail!("t must be a cpu tensor"),
    };
    let bytes: Vec<u8> = match t.dtype() {
        DType::F32 => {
            let slice = storage.as_slice::<f32>()?;
            bytemuck::cast_slice(slice).to_vec()
        }
        _ => {
            return Err(candle_core::Error::Msg(
                "unsupported dtype for tensor_raw".into(),
            ));
        }
    };

    Ok((bytes, shape))
}

fn image_resize(
    image: &DynamicImage,
    mut height: usize,
    mut width: usize,
    max_height: usize,
    max_width: usize,
    patch_size: usize,
    filter: FilterType,
) -> DynamicImage {
    let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
    if ratio > 1. {
        height = (height as f64 / ratio).floor() as usize;
        width = (width as f64 / ratio).floor() as usize;
    }

    let num_height_tokens = (height - 1) / patch_size + 1;
    let num_width_tokens = (width - 1) / patch_size + 1;

    let resize_height = num_height_tokens * patch_size;
    let resize_width = num_width_tokens * patch_size;

    image.resize_exact(resize_width as u32, resize_height as u32, filter)
}

pub fn to_tensor(
    images: &Vec<DynamicImage>,
    image_mean: Option<[f32; 3]>,
    image_std: Option<[f32; 3]>,
) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let mut image_sizes = Vec::new();
    let mut pixel_values = Vec::new();
    for image in images.iter() {
        let (width, height) = image.dimensions();
        image_sizes.push((height as usize, width as usize));

        let rgb = image.to_rgb32f(); // HWC f32
        let (w, h) = rgb.dimensions();
        let data = rgb.into_raw(); // Vec<f32> in HWC layout, FAST

        // Build tensor from slice
        let t = Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?;

        // Convert HWC → CHW without copying data manually
        let t = t.permute((2, 0, 1))?.contiguous()?; // [H, W, C] -> [C, H, W]

        // Only NOW can we normalize, because we are in Floating Point land
        let t = if let (Some(mean), Some(std)) = (image_mean, image_std) {
            let mean = Tensor::new(&mean[..], &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
            let std = Tensor::new(&std[..], &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
            t.broadcast_sub(&mean)?.broadcast_div(&std)?
        } else {
            t
        };

        // let t = image_to_pixels(image, &Device::Cpu)?;
        pixel_values.push(t.unsqueeze(0)?);
    }
    Ok((Tensor::cat(&pixel_values, 0)?, image_sizes))
}

#[derive(Clone)]
pub struct ImageProcessConfig {
    image_start_token: Option<String>,
    image_token: String,
    image_break_token: Option<String>,
    image_end_token: String,
    pub spatial_merge_size: usize,
    pub mm_tokens_per_image: Option<usize>,
    pub do_normalize: Option<bool>,
    pub do_resize: Option<bool>,
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub max_height: usize,
    pub max_width: usize,
    pub patch_size: usize,
    pub temporal_patch_size: Option<usize>,
    pub abolute_resize: bool,
    pub scale_factor: Option<f32>,
    pub resampling: Option<usize>,
    pub model_type: ModelType,
    pub image_token_id: Option<u32>,
}

impl ImageProcessConfig {
    pub fn default(
        image_start_token: Option<String>,
        image_token: String,
        image_break_token: Option<String>,
        image_end_token: String,
        spatial_merge_size: usize,
        temporal_patch_size: Option<usize>,
        patch_size: usize,
        image_size: usize,
        abolute_resize: bool,
    ) -> Self {
        ImageProcessConfig {
            image_start_token,
            image_token,
            image_break_token,
            image_end_token,
            spatial_merge_size,
            mm_tokens_per_image: None,
            image_mean: None,
            image_std: None,
            do_resize: Some(true),
            do_normalize: Some(true),
            max_height: image_size,
            max_width: image_size,
            patch_size,
            abolute_resize,
            resampling: None,
            scale_factor: None,
            temporal_patch_size: temporal_patch_size,
            model_type: ModelType::Mistral3VL,
            image_token_id: None,
        }
    }

    pub fn prompt_marker_tokens(&self) -> Vec<String> {
        let mut tokens = Vec::new();
        if let Some(start) = &self.image_start_token {
            if !start.is_empty() {
                tokens.push(start.clone());
            }
        }
        if !self.image_token.is_empty() {
            tokens.push(self.image_token.clone());
        }
        if let Some(brk) = &self.image_break_token {
            if !brk.is_empty() {
                tokens.push(brk.clone());
            }
        }
        if !self.image_end_token.is_empty() {
            tokens.push(self.image_end_token.clone());
        }
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        tokens
    }
}

pub trait ImageProcessTrait: Send {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &Vec<DynamicImage>,
    ) -> Result<(Tensor, Vec<(usize, usize)>)>;
}

pub struct ImageProcessor {
    cfg: ImageProcessConfig,
    fixed_width: Option<usize>,
    fixed_height: Option<usize>,
}

impl ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub fn new(cfg: &ImageProcessConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            fixed_width: None,
            fixed_height: None,
        }
    }

    /// Preprocess input DynamicImages:
    /// 1. Resize w/ custom rule (padding to uniform size)
    /// 2. Optional rescale
    /// 3. Optional normalize
    pub fn preprocess_images(
        &mut self,
        images: &Vec<DynamicImage>,
        max_height: usize,
        max_width: usize,
        patch_size: usize,
        abolute_resize: bool,
    ) -> Vec<DynamicImage> {
        let mut resized_imgs: Vec<DynamicImage> = Vec::new();
        let filter = FilterType::Nearest;

        if abolute_resize {
            for img in images.iter() {
                let resized = img.resize_exact(max_width as u32, max_height as u32, filter);
                resized_imgs.push(resized);
            }
        } else {
            for img in images.iter() {
                let resized = if let (Some(h), Some(w)) = (self.fixed_height, self.fixed_width) {
                    img.resize_exact(w as u32, h as u32, filter)
                } else {
                    let (h, w) = img.dimensions();
                    let resized = image_resize(
                        img, h as usize, w as usize, max_height, max_width, patch_size, filter,
                    );
                    self.fixed_height = Some(resized.height() as usize);
                    self.fixed_width = Some(resized.width() as usize);
                    resized
                };
                resized_imgs.push(resized);
            }
        }
        resized_imgs
    }

    fn preprocess(&mut self, images: &Vec<DynamicImage>) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let do_normalize = self.cfg.do_normalize.unwrap_or(false);
        let image_mean = if do_normalize {
            Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN))
        } else {
            None
        };

        let image_std = if do_normalize {
            Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD))
        } else {
            None
        };

        let images = self.preprocess_images(
            images,
            self.cfg.max_height,
            self.cfg.max_width,
            self.cfg.patch_size,
            self.cfg.abolute_resize,
        );
        to_tensor(&images, image_mean, image_std)
    }
}

impl ImageProcessTrait for ImageProcessor {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &Vec<DynamicImage>,
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (pixel_values, image_sizes_all) =
            self.preprocess(images).expect("Preprocessing failed");

        crate::log_info!(
            "pixel_values tensor shape {:?}, image_sizes_all {:?}",
            pixel_values.shape(),
            image_sizes_all
        );
        let mut image_sizes_all_iter = image_sizes_all.clone().into_iter();
        let mut replace_strings = Vec::new();
        while prompt.contains(IMAGE_PLACEHOLDER) {
            let (height, width) = image_sizes_all_iter.next().unwrap();
            let num_height_tokens =
                (height as usize) / (self.cfg.patch_size * self.cfg.spatial_merge_size);
            let num_width_tokens =
                (width as usize) / (self.cfg.patch_size * self.cfg.spatial_merge_size);

            crate::log_info!(
                "num_height_tokens {num_height_tokens}, num_width_tokens {num_width_tokens}"
            );
            let mut replace_tokens = vec![
                [
                    vec![self.cfg.image_token.clone(); num_width_tokens],
                    if self.cfg.image_break_token.is_some() {
                        vec![self.cfg.image_break_token.as_ref().unwrap().clone()]
                    } else {
                        vec![]
                    }
                ]
                .concat();
                num_height_tokens
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

            if self.cfg.image_break_token.is_some() {
                *replace_tokens.last_mut().unwrap() = self.cfg.image_end_token.clone();
            } else {
                replace_tokens.push(self.cfg.image_end_token.clone());
            }

            if let Some(start_token) = &self.cfg.image_start_token {
                let mut vec_final = vec![start_token.clone()];
                vec_final.extend(replace_tokens);
                replace_tokens = vec_final.clone();
            }
            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replace(IMAGE_PLACEHOLDER, PLACEHOLDER);
        }

        while prompt.contains(PLACEHOLDER) {
            let replace_str = replace_strings.pop().unwrap();
            *prompt = prompt.replace(PLACEHOLDER, &replace_str);
        }

        Ok((pixel_values, image_sizes_all))
    }
}

pub fn get_image_config(
    model_type: ModelType,
    config: &Config,
) -> Result<Option<ImageProcessConfig>> {
    let img_cfg = match model_type {
        ModelType::Mistral3VL => {
            use crate::models::mistral3_vl::Mistral3Config;
            let Some(extra_config_json) = config.extra_config_json.as_ref() else {
                return Ok(None);
            };
            let cfg: Mistral3Config =
                serde_json::from_str(extra_config_json).map_err(candle_core::Error::wrap)?;

            let mut img_cfg = ImageProcessConfig::default(
                None,
                "[IMG]".to_string(),
                Some("[IMG_BREAK]".to_string()),
                "[IMG_END]".to_string(),
                cfg.spatial_merge_size,
                None,
                cfg.vision_config.patch_size,
                896,
                false,
            );
            img_cfg.model_type = ModelType::Mistral3VL;
            img_cfg.image_token_id = Some(cfg.image_token_index as u32);
            Some(img_cfg)
        }
        ModelType::Gemma3 => {
            use crate::models::gemma3::config::Gemma3Config;
            let Some(extra_config_json) = config.extra_config_json.as_ref() else {
                return Ok(None);
            };
            let cfg: Gemma3Config =
                serde_json::from_str(extra_config_json).map_err(candle_core::Error::wrap)?;

            let mut img_cfg = ImageProcessConfig::default(
                Some("<start_of_image>".to_string()),
                "<image_soft_token>".to_string(),
                None,
                "<end_of_image>".to_string(),
                4,
                None,
                cfg.vision_config.patch_size,
                896,
                true,
            );
            img_cfg.model_type = ModelType::Gemma3;
            img_cfg.image_token_id = Some(cfg.image_token_index as u32);
            img_cfg.mm_tokens_per_image = Some(cfg.mm_tokens_per_image);
            img_cfg.scale_factor = Some(0.003921567);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Some(img_cfg)
        }
        ModelType::Qwen3VL => {
            let Some(cfg) = crate::models::qwen3_vl::try_parse_multimodal_extra_config(config)?
            else {
                return Ok(None);
            };

            let mut img_cfg = ImageProcessConfig::default(
                Some("<|vision_start|>".to_string()),
                "<|image_pad|>".to_string(),
                None,
                "<|vision_end|>".to_string(),
                cfg.vision_config.spatial_merge_size,
                Some(cfg.vision_config.temporal_patch_size),
                cfg.vision_config.patch_size,
                896,
                false,
            );
            img_cfg.model_type = ModelType::Qwen3VL;
            img_cfg.image_token_id = Some(cfg.image_token_id);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Some(img_cfg)
        }
        ModelType::LLaMa4 => {
            use crate::models::llama4::config::Llama4Config;
            let Some(extra_config_json) = config.extra_config_json.as_ref() else {
                return Ok(None);
            };
            let cfg: Llama4Config =
                serde_json::from_str(extra_config_json).map_err(candle_core::Error::wrap)?;
            let patch_size = cfg.vision_config.patch_size;
            let image_size = cfg.vision_config.image_size;
            let pixel_shuffle_ratio = cfg.vision_config.pixel_shuffle_ratio;
            let num_patches = (image_size / patch_size).pow(2);
            let downsampled = ((num_patches as f32).sqrt() * pixel_shuffle_ratio).powi(2) as usize;

            let mut img_cfg = ImageProcessConfig::default(
                None,
                "<|image|>".to_string(),
                None,
                "".to_string(),
                1,
                None,
                patch_size,
                image_size,
                true,
            );
            img_cfg.model_type = ModelType::LLaMa4;
            img_cfg.image_token_id = Some(cfg.image_token_index as u32);
            img_cfg.mm_tokens_per_image = Some(downsampled);
            img_cfg.scale_factor = Some(1.0 / 255.0);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Some(img_cfg)
        }
        _ => None,
    };
    Ok(img_cfg)
}

#[allow(dead_code)]
pub trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for Option<usize> {
    fn to_filter(self) -> Result<FilterType> {
        match self {
            Some(0) => Ok(FilterType::Nearest),
            Some(1) => Ok(FilterType::Lanczos3),
            Some(2) | None => Ok(FilterType::Triangle), // BiLinear
            Some(3) => Ok(FilterType::CatmullRom),      // BiCubic
            Some(4) => Ok(FilterType::Nearest),
            Some(x) => candle_core::bail!("Filter number {x} not supported"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::get_image_config;
    use crate::utils::config::{Config, EosTokenId, ModelType};

    fn base_config(extra_config_json: Option<String>) -> Config {
        Config {
            architectures: Some(vec!["Qwen3_5ForConditionalGeneration".to_string()]),
            head_dim: Some(128),
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 1024,
            hidden_size: 512,
            num_hidden_layers: 2,
            max_model_len: Some(1024),
            intermediate_size: 1024,
            rms_norm_eps: 1e-6,
            vocab_size: Some(32000),
            rope_theta: Some(10000.0),
            attention_bias: None,
            qkv_bias: None,
            attn_output_gate: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            tie_word_embeddings: Some(false),
            bos_token_id: Some(1),
            eos_token_id: Some(EosTokenId::Single(2)),
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
            extra_config_json,
        }
    }

    #[test]
    fn qwen35_text_only_extra_config_disables_image_config() {
        let config = base_config(Some(
            serde_json::json!({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "full_attention_interval": 4,
            })
            .to_string(),
        ));
        let cfg = get_image_config(ModelType::Qwen3VL, &config).unwrap();
        assert!(cfg.is_none());
    }
}
