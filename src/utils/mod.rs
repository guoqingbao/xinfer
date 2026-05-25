pub mod chat_template;
pub mod command;
pub mod config;
pub mod downloader;
pub mod env;
pub mod gguf_helper;
pub mod gguf_varbuilder;
pub mod gptq;
#[cfg(all(feature = "cuda", feature = "graph"))]
pub mod graph;
pub mod guidance;
pub mod heartbeat;
pub mod image;
pub mod kvcache_allocator;
pub mod logits_processor;
pub mod progress;
pub mod reasoning;
pub mod special_tokens;
use crate::core::GenerationOutput;
use crate::models::gemma3::config::Gemma3Config;
use crate::models::qwen3_vl::config::{
    Qwen3VLConfig as Qwen3VLGgufConfig, VisionConfig as Qwen3VLGgufVisionConfig,
};
use crate::utils::config::MoEConfig;
use crate::utils::config::ModelType;
use crate::utils::config::QuantConfig;
use crate::utils::config::RopeScalingValue;
use crate::utils::downloader::ModelPaths;
use crate::utils::gguf_helper::{load_gguf_info_from_files, GGUFInfo};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result};
use config::{Config, EngineConfig, EosTokenId, GenerationConfig, TokenizerConfig};
pub use reasoning::{
    build_reasoning_grammar, thinking_grammar_with_reasoning_block, ReasoningEffort,
    ThinkingGrammarBuilder,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[cfg(feature = "flashinfer")]
#[derive(Clone, Copy, Debug)]
pub struct FlashInferKvParams {
    pub kv_dtype: DType,
    pub out_dtype: DType,
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_qo_heads: usize,
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

pub fn module_path_matches_not_convert(module_path: &str, item: &str) -> bool {
    crate::utils::config::match_ignore_pattern(module_path, item)
}

pub fn should_skip_fp8_for_module(module_path: &str, cfg: &QuantConfig) -> bool {
    cfg.should_skip_module(module_path)
}

pub fn should_skip_quant_for_module(module_path: &str, cfg: &QuantConfig) -> bool {
    cfg.should_skip_module(module_path)
}

fn parse_fallback_moe_cfg(arch_name: &str, raw_cfg: &[u8]) -> Option<MoEConfig> {
    if arch_name == "MiniMaxM2ForCausalLM" {
        let mut raw_cfg_json: serde_json::Value = serde_json::from_slice(raw_cfg).ok()?;
        let raw_cfg_obj = raw_cfg_json.as_object_mut()?;

        if !raw_cfg_obj.contains_key("moe_intermediate_size") {
            let intermediate_size = raw_cfg_obj.get("intermediate_size")?.clone();
            raw_cfg_obj.insert("moe_intermediate_size".to_string(), intermediate_size);
        }

        serde_json::from_value(raw_cfg_json).ok()
    } else {
        serde_json::from_slice(raw_cfg).ok()
    }
}

pub fn hub_load_local_safetensors(path: &String, json_file: &str) -> Result<Vec<PathBuf>> {
    crate::log_info!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let json: serde_json::Value =
        serde_json::from_reader(&jsfile).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| Path::new(path).join(v))
        .collect();
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> Result<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal).unwrap());
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            crate::log_info!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            crate::log_info!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(Device::Cpu)
    }
}

pub fn config_from_gguf<R: std::io::Seek + std::io::Read>(
    ct: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
) -> Result<Config> {
    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };
    let arch = md_get("general.architecture")?.to_string()?;

    let head_count = md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
    let canonical_arch = match arch.as_str() {
        "qwen35" => "Qwen3_5ForConditionalGeneration".to_string(),
        "qwen35moe" => "Qwen3_5MoeForConditionalGeneration".to_string(),
        "qwen3vl" => "Qwen3VLForConditionalGeneration".to_string(),
        "qwen3vlmoe" => "Qwen3VLMoeForConditionalGeneration".to_string(),
        "gemma3" => "Gemma3ForConditionalGeneration".to_string(),
        "gemma4" => "Gemma4ForConditionalGeneration".to_string(),
        "mistral3" => "Mistral3ForConditionalGeneration".to_string(),
        _ => arch.clone(),
    };

    let head_count_kv =
        md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;

    let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
    let head_dim = if head_dim.is_ok() {
        Some(head_dim.unwrap().to_u32()? as usize)
    } else {
        None
    };
    let embedding_length = md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
    let feed_forward_length = md_get(format!("{arch}.feed_forward_length").as_str())
        .and_then(|v| v.to_u32())
        .map(|v| v as usize)
        .or_else(|_| {
            if arch == "qwen35moe" {
                Ok(0) //Qwen3.5 MoE has no MLP layer
            } else {
                candle_core::bail!("cannot find {arch}.feed_forward_length in metadata")
            }
        })?;
    let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
    let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
    let rms_norm_eps =
        md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
    let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
        .and_then(|m| m.to_f32())
        .unwrap_or(10000f32);
    let vocab_size = md_get(format!("{arch}.vocab_size").as_str());

    let vocab_size = if vocab_size.is_ok() {
        Some(vocab_size.unwrap().to_u32()? as usize)
    } else {
        let vocab_size = md_get("tokenizer.ggml.tokens");
        if vocab_size.is_ok() {
            let size = vocab_size.unwrap().to_vec()?.len();
            crate::log_info!(
                "No vocab_size in metadata, using tokenizer.ggml.tokens with size {}",
                size
            );
            Some(size)
        } else {
            None
        }
    };

    let bos_token_id = md_get("tokenizer.ggml.bos_token_id");

    let bos_token_id = if bos_token_id.is_ok() {
        Some(bos_token_id.unwrap().to_u32()? as usize)
    } else {
        None
    };

    let eos_token_id = md_get("tokenizer.ggml.eos_token_id");

    let eos_token_id = if eos_token_id.is_ok() {
        EosTokenId::Single(eos_token_id.unwrap().to_u32()?)
    } else {
        EosTokenId::Multiple(vec![])
    };

    // ---------------- RoPE scaling --------------------------
    let rope_scaling = md_get(format!("{arch}.rope.scaling.type").as_str())
        .ok()
        .map(|v| {
            let scaling_type = v.to_string()?;
            crate::log_info!("Rope scaling type: {}", scaling_type);

            let mut map = HashMap::<String, RopeScalingValue>::new();

            if let Ok(alpha) = md_get(format!("{arch}.rope.scaling.alpha").as_str()) {
                map.insert(
                    "alpha".into(),
                    RopeScalingValue::Number(alpha.to_f32()? as f64),
                );
            } else if let Ok(factor) = md_get(format!("{arch}.rope.scaling.factor").as_str()) {
                map.insert(
                    "factor".into(),
                    RopeScalingValue::Number(factor.to_f32()? as f64),
                );
            }

            if let Ok(v) = md_get(format!("{arch}.rope.scaling.original_context_length").as_str()) {
                map.insert(
                    "original_max_position_embeddings".into(),
                    RopeScalingValue::Number(v.to_u32()? as f64),
                );
            }

            if scaling_type == "llama3" {
                if let Ok(v) = md_get(format!("{arch}.rope.scaling.low_freq_factor").as_str()) {
                    map.insert(
                        "low_freq_factor".into(),
                        RopeScalingValue::Number(v.to_f32()? as f64),
                    );
                }
                if let Ok(v) = md_get(format!("{arch}.rope.scaling.high_freq_factor").as_str()) {
                    map.insert(
                        "high_freq_factor".into(),
                        RopeScalingValue::Number(v.to_f32()? as f64),
                    );
                }
            }

            if scaling_type == "yarn" {
                for (key, alt) in [
                    ("beta_fast", "yarn_beta_fast"),
                    ("beta_slow", "yarn_beta_slow"),
                ] {
                    if let Ok(v) = md_get(format!("{arch}.rope.scaling.{key}").as_str())
                        .or_else(|_| md_get(format!("{arch}.rope.scaling.{alt}").as_str()))
                    {
                        map.insert(key.into(), RopeScalingValue::Number(v.to_f32()? as f64));
                    }
                }

                for key in ["extrapolation_factor", "attn_factor"] {
                    if let Ok(v) = md_get(format!("{arch}.rope.scaling.{key}").as_str()) {
                        map.insert(key.into(), RopeScalingValue::Number(v.to_f32()? as f64));
                    }
                }

                if let Ok(v) = md_get(format!("{arch}.rope.attention.temperature_scale").as_str()) {
                    map.insert(
                        "llama_4_scaling_beta".into(),
                        RopeScalingValue::Number(v.to_f32()? as f64),
                    );
                }
            }

            // -------- MRoPE support --------

            if let Ok(v) = md_get(format!("{arch}.rope.scaling.mrope_interleaved").as_str()) {
                map.insert(
                    "mrope_interleaved".into(),
                    RopeScalingValue::Bool(v.to_bool()?),
                );
            }

            if let Ok(v) = md_get(format!("{arch}.rope.scaling.mrope_section").as_str()) {
                let section = v
                    .to_vec()?
                    .into_iter()
                    .map(|v| v.to_u32().unwrap() as f64)
                    .collect::<Vec<_>>();
                map.insert(
                    "mrope_section".into(),
                    RopeScalingValue::NumberArray(section),
                );
            }

            map.insert(
                "rope_type".into(),
                RopeScalingValue::String(scaling_type.clone()),
            );
            crate::log_info!("Rope scaling map: {:?}", map);

            Ok::<HashMap<String, RopeScalingValue>, candle_core::Error>(map)
        })
        .transpose()?;
    // --------------------------------------------------------

    let head_dim = head_dim.unwrap_or(embedding_length / head_count);

    let has_output_weight = ct.tensor(reader, "output.weight", &Device::Cpu).is_ok();

    let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
    let partial_rotary_factor = if rope_dim.is_ok() {
        let rope_dim = rope_dim.unwrap().to_u32()? as usize;
        if rope_dim != head_dim {
            Some(rope_dim as f32 / head_dim as f32)
        } else {
            None
        }
    } else {
        None
    };

    let md_opt_usize = |key: &str| {
        md_get(key)
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize)
    };
    let md_opt_f64 = |key: &str| md_get(key).and_then(|v| v.to_f64()).ok();
    let md_opt_bool = |key: &str| md_get(key).and_then(|v| v.to_bool()).ok();
    let md_opt_string = |key: &str| {
        md_get(key)
            .and_then(|v| v.to_string())
            .ok()
            .map(|v| v.to_owned())
    };

    let mod_cfg = if arch == "gemma4" {
        let expert_count = md_get(format!("{arch}.expert_count").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);
        let expert_used_count = md_get(format!("{arch}.expert_used_count").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);
        let expert_ff_length = md_get(format!("{arch}.expert_feed_forward_length").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);
        if let (Some(ec), Some(euc)) = (expert_count, expert_used_count) {
            if ec > 0 {
                Some(MoEConfig {
                    moe_intermediate_size: expert_ff_length.unwrap_or(feed_forward_length),
                    shared_expert_intermediate_size: None,
                    num_experts: Some(ec),
                    mlp_only_layers: Some(Vec::new()),
                    decoder_sparse_step: Some(1),
                    norm_topk_prob: md_opt_bool(format!("{arch}.norm_topk_prob").as_str())
                        .unwrap_or(true),
                    num_experts_per_tok: euc,
                    first_k_dense_replace: None,
                    n_shared_experts: None,
                    routed_scaling_factor: md_opt_f64(
                        format!("{arch}.routed_scaling_factor").as_str(),
                    ),
                    n_group: md_opt_usize(format!("{arch}.n_group").as_str()),
                    topk_group: md_opt_usize(format!("{arch}.topk_group").as_str()),
                    scoring_func: md_opt_string(format!("{arch}.scoring_func").as_str()),
                    topk_method: md_opt_string(format!("{arch}.topk_method").as_str()),
                })
            } else {
                None
            }
        } else {
            None
        }
    } else if matches!(arch.as_str(), "qwen3moe" | "qwen2moe" | "qwen35moe") {
        let expert_feed_forward_length =
            md_get(format!("{arch}.expert_feed_forward_length").as_str())?.to_u32()? as usize;
        let expert_weights_norm = md_get(format!("{arch}.expert_weights_norm").as_str());
        let expert_weights_norm = if expert_weights_norm.is_ok() {
            expert_weights_norm.unwrap().to_bool().ok()
        } else {
            None
        };

        let expert_weights_scale = md_get(format!("{arch}.expert_weights_scale").as_str());
        let expert_weights_scale = if expert_weights_scale.is_ok() {
            expert_weights_scale.unwrap().to_f64().ok()
        } else {
            None
        };

        let leading_dense_block_count =
            md_get(format!("{arch}.leading_dense_block_count").as_str());
        let leading_dense_block_count = if leading_dense_block_count.is_ok() {
            Some(leading_dense_block_count.unwrap().to_u32()? as usize)
        } else {
            None
        };

        let expert_shared_count = md_get(format!("{arch}.expert_shared_count").as_str());
        let expert_shared_count = if expert_shared_count.is_ok() {
            Some(expert_shared_count.unwrap().to_u32()? as usize)
        } else {
            None
        };
        let expert_shared_feed_forward_length =
            md_get(format!("{arch}.expert_shared_feed_forward_length").as_str());
        let expert_shared_feed_forward_length = if expert_shared_feed_forward_length.is_ok() {
            Some(expert_shared_feed_forward_length.unwrap().to_u32()? as usize)
        } else if arch.to_string() == "glm4moe" {
            Some(expert_feed_forward_length)
        } else {
            None
        };

        Some(MoEConfig {
            moe_intermediate_size: expert_feed_forward_length,
            shared_expert_intermediate_size: expert_shared_feed_forward_length,
            num_experts: Some(md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize),
            mlp_only_layers: Some(match md_get(format!("{arch}.moe_layer_pattern").as_str()) {
                Ok(pattern) => pattern
                    .to_vec()?
                    .into_iter()
                    .enumerate()
                    .filter_map(|(idx, v)| match v.to_bool() {
                        Ok(true) => None,
                        Ok(false) => Some(idx),
                        Err(_) => None,
                    })
                    .collect(),
                Err(_) => Vec::new(),
            }),
            decoder_sparse_step: Some(1),
            norm_topk_prob: expert_weights_norm
                .or_else(|| md_opt_bool(format!("{arch}.norm_topk_prob").as_str()))
                .unwrap_or(true),
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
            first_k_dense_replace: leading_dense_block_count,
            n_shared_experts: expert_shared_count,
            routed_scaling_factor: expert_weights_scale
                .or_else(|| md_opt_f64(format!("{arch}.routed_scaling_factor").as_str())),
            n_group: md_opt_usize(format!("{arch}.n_group").as_str()),
            topk_group: md_opt_usize(format!("{arch}.topk_group").as_str()),
            scoring_func: md_opt_string(format!("{arch}.scoring_func").as_str()),
            topk_method: md_opt_string(format!("{arch}.topk_method").as_str()),
        })
    } else {
        None
    };

    let extra_config_json = if arch == "gemma4" {
        let sw = md_get(format!("{arch}.attention.sliding_window").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);

        let sliding_window_pattern: Vec<bool> =
            match md_get(format!("{arch}.attention.sliding_window_pattern").as_str()) {
                Ok(v) => match v.to_vec() {
                    Ok(arr) => arr.iter().map(|v| v.to_bool().unwrap_or(true)).collect(),
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            };

        let layer_types_vec: Vec<&str> = if sliding_window_pattern.is_empty() {
            (0..block_count)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                })
                .collect()
        } else {
            sliding_window_pattern
                .iter()
                .map(|&is_sliding| {
                    if is_sliding {
                        "sliding_attention"
                    } else {
                        "full_attention"
                    }
                })
                .collect()
        };

        let global_head_dim = md_get(format!("{arch}.attention.key_length").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let swa_head_dim = md_get(format!("{arch}.attention.key_length_swa").as_str())
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);

        let rope_freq_base_swa = md_get(format!("{arch}.rope.freq_base_swa").as_str())
            .and_then(|v| v.to_f64())
            .ok()
            .unwrap_or(10000.0);

        let final_logit_softcapping = md_get(format!("{arch}.final_logit_softcapping").as_str())
            .and_then(|v| v.to_f64())
            .ok();

        let enable_moe = mod_cfg.is_some();

        let num_global_kv_heads = md_get(format!("{arch}.attention.head_count_kv").as_str())
            .ok()
            .and_then(|v| {
                v.to_u32().ok().map(|val| val as usize).or_else(|| {
                    v.to_vec().ok().and_then(|arr| {
                        arr.last()
                            .and_then(|val| val.to_u32().ok())
                            .map(|val| val as usize)
                    })
                })
            });

        Some(
            serde_json::json!({
                "architectures": ["Gemma4ForConditionalGeneration"],
                "layer_types": layer_types_vec,
                "sliding_window": sw,
                "global_head_dim": global_head_dim,
                "swa_head_dim": swa_head_dim,
                "rope_local_base_freq": rope_freq_base_swa,
                "final_logit_softcapping": final_logit_softcapping,
                "enable_moe_block": enable_moe,
                "num_global_key_value_heads": num_global_kv_heads,
            })
            .to_string(),
        )
    } else if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        let conv_kernel_size =
            md_get(format!("{arch}.ssm.conv_kernel").as_str())?.to_u32()? as usize;
        let num_k_heads = md_get(format!("{arch}.ssm.group_count").as_str())?.to_u32()? as usize;
        let num_v_heads = md_get(format!("{arch}.ssm.time_step_rank").as_str())?.to_u32()? as usize;
        let key_head_dim = md_get(format!("{arch}.ssm.state_size").as_str())?.to_u32()? as usize;
        let inner_size = md_get(format!("{arch}.ssm.inner_size").as_str())?.to_u32()? as usize;
        let value_head_dim = if num_v_heads > 0 && inner_size % num_v_heads == 0 {
            inner_size / num_v_heads
        } else {
            key_head_dim
        };
        let full_attention_interval =
            md_get(format!("{arch}.full_attention_interval").as_str())?.to_u32()? as usize;
        Some(
            serde_json::json!({
                "architectures": [canonical_arch.clone()],
                "linear_conv_kernel_dim": conv_kernel_size,
                "linear_num_key_heads": num_k_heads,
                "linear_num_value_heads": num_v_heads,
                "linear_key_head_dim": key_head_dim,
                "linear_value_head_dim": value_head_dim,
                "full_attention_interval": full_attention_interval,
            })
            .to_string(),
        )
    } else {
        None
    };

    let _gguf_sliding_window = md_get(format!("{arch}.attention.sliding_window").as_str())
        .and_then(|v| v.to_u32())
        .ok()
        .map(|v| v as usize);

    let mut cfg = Config {
        architectures: Some(vec![canonical_arch.clone()]),
        head_dim: Some(head_dim),
        num_attention_heads: head_count,
        num_key_value_heads: head_count_kv,
        max_position_embeddings: context_length,
        hidden_size: embedding_length,
        num_hidden_layers: block_count,
        max_model_len: Some(context_length),
        intermediate_size: feed_forward_length,
        rms_norm_eps,
        vocab_size,
        rope_theta: Some(rope_freq_base as f64),
        attention_bias: None,
        qkv_bias: None,
        attn_output_gate: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        tie_word_embeddings: Some(!has_output_weight),
        bos_token_id,
        eos_token_id: Some(eos_token_id),
        use_sliding_window: None,
        sliding_window: None,
        max_window_layers: None,
        partial_rotary_factor,
        hidden_act: candle_nn::Activation::Silu,
        rope_scaling,
        quant: None,
        moe_cfg: mod_cfg,
        kvcache_dtype: crate::utils::config::KvCacheDtype::Auto,
        quantization_config: None,
        is_multi_model: None,
        extra_config_json,
    };

    if arch == "gemma4" || arch == "gemma3" {
        cfg.hidden_act = candle_nn::Activation::GeluPytorchTanh;
    }

    Ok(cfg)
}

/// Derives optimal YARN RoPE scaling parameters based on the scaling factor.
///
/// For factors > 4.0, parameters are scaled proportionally to maintain
/// appropriate frequency band transitions. The attn_factor is kept at 1.0
/// as it's a multiplier applied to the YARN mscale calculation.
/// Reference: https://github.com/jquesnelle/yarn
///
/// Returns: (beta_fast, beta_slow, extrapolation_factor, attn_factor)
pub fn derive_yarn_parameters(factor: f64) -> (f64, f64, f64, f64) {
    // Validate factor
    let factor = factor.max(1.0);

    // beta_fast: Controls transition band width between fast and slow frequencies
    // For factor > 4, scale proportionally to sqrt(factor/4)
    let beta_fast = if factor <= 4.0 {
        32.0
    } else {
        32.0 * (factor / 4.0).sqrt()
    };

    // beta_slow: Controls low-frequency attenuation
    // Keep at 1.0 for all scaling factors (standard YARN behavior)
    let beta_slow = 1.0;

    // extrapolation_factor: Adjusts extrapolation behavior beyond original context
    // Slightly increase for factors > 8.0
    let extrapolation_factor = if factor > 8.0 {
        1.0 + 0.05 * (factor - 8.0).sqrt()
    } else {
        1.0
    };

    // attn_factor: Attention scaling multiplier
    // In YARN, this is applied to mscale = (0.1 * ln(factor) + 1) * attn_factor
    // The reference implementation keeps attn_factor = 1.0
    let attn_factor = 1.0;

    (beta_fast, beta_slow, extrapolation_factor, attn_factor)
}

pub fn apply_static_rope_scaling(
    yarn_scaling_factor: Option<f64>,
    max_position_embeddings: usize,
) -> Option<HashMap<String, RopeScalingValue>> {
    if let Some(factor) = yarn_scaling_factor {
        let (beta_fast, beta_slow, extrapolation_factor, attn_factor) =
            derive_yarn_parameters(factor);

        let mut scaling_map = HashMap::new();
        scaling_map.insert("rope_type".into(), RopeScalingValue::String("yarn".into()));
        scaling_map.insert("factor".into(), RopeScalingValue::Number(factor));
        scaling_map.insert(
            "original_max_position_embeddings".into(),
            RopeScalingValue::Number(max_position_embeddings as f64),
        );
        scaling_map.insert("beta_fast".into(), RopeScalingValue::Number(beta_fast));
        scaling_map.insert("beta_slow".into(), RopeScalingValue::Number(beta_slow));
        scaling_map.insert(
            "extrapolation_factor".into(),
            RopeScalingValue::Number(extrapolation_factor),
        );
        scaling_map.insert("attn_factor".into(), RopeScalingValue::Number(attn_factor));
        return Some(scaling_map);
    }
    None
}

fn apply_runtime_rope_overrides(config: &mut Config, yarn_scaling_factor: Option<f64>) {
    if let Some(scaling) =
        apply_static_rope_scaling(yarn_scaling_factor, config.max_position_embeddings)
    {
        config.rope_scaling = Some(scaling);
    }
}

fn effective_max_position_embeddings(config: &Config) -> usize {
    let Some(rope_scaling) = &config.rope_scaling else {
        return config.max_position_embeddings;
    };

    let rope_type = rope_scaling
        .get("rope_type")
        .or_else(|| rope_scaling.get("type"))
        .and_then(|value| value.as_str());
    let factor = rope_scaling.get("factor").and_then(|value| value.as_f64());
    let original_max_position_embeddings = rope_scaling
        .get("original_max_position_embeddings")
        .and_then(|value| value.as_f64());

    match (rope_type, factor, original_max_position_embeddings) {
        (Some("yarn"), Some(factor), Some(original_max_position_embeddings)) if factor > 1.0 => {
            let scaled_max_position_embeddings =
                (original_max_position_embeddings * factor).round() as usize;
            std::cmp::max(
                config.max_position_embeddings,
                scaled_max_position_embeddings,
            )
        }
        _ => config.max_position_embeddings,
    }
}

fn resolve_config_model_len(config: &Config, config_tokenizer: &TokenizerConfig) -> usize {
    let effective_max_position_embeddings = effective_max_position_embeddings(config);
    let (tokenizer_model_len, tokenizer_limit_is_fallback) = match config_tokenizer.model_max_length
    {
        Some(model_max_length) if model_max_length < 10_000_000.0 => {
            (model_max_length as usize, false)
        }
        Some(_) => (262_144, true),
        None => (262_144, true),
    };

    let tokenizer_model_len = if effective_max_position_embeddings > config.max_position_embeddings
        && (tokenizer_limit_is_fallback || tokenizer_model_len <= config.max_position_embeddings)
    {
        effective_max_position_embeddings
    } else {
        tokenizer_model_len
    };

    std::cmp::min(effective_max_position_embeddings, tokenizer_model_len)
}

fn tokenizer_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .get_vocab(true)
        .get(token)
        .copied()
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("missing multimodal token `{token}`").into())
        })
}

fn qwen3_vl_deepstack_indexes(
    ct: &candle_core::quantized::gguf_file::Content,
) -> Result<Vec<usize>> {
    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };
    let values = md_get("clip.vision.is_deepstack_layers")
        .and_then(|v| v.to_vec().cloned())
        .ok();
    Ok(values
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .filter_map(|(idx, v)| match v.to_bool() {
            Ok(true) => Some(idx),
            _ => None,
        })
        .collect())
}

fn build_qwen3_vl_gguf_extra_config(
    text_config: &Config,
    mmproj_ct: &candle_core::quantized::gguf_file::Content,
    tokenizer: &Tokenizer,
) -> Result<String> {
    let md_get = |s: &str| match mmproj_ct.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };
    let vision_cfg = Qwen3VLGgufVisionConfig {
        depth: md_get("clip.vision.block_count")?.to_u32()? as usize,
        hidden_size: md_get("clip.vision.embedding_length")?.to_u32()? as usize,
        out_hidden_size: md_get("clip.vision.projection_dim")?.to_u32()? as usize,
        hidden_act: if md_get("clip.use_gelu")
            .and_then(|v| v.to_bool())
            .unwrap_or(true)
        {
            candle_nn::Activation::Gelu
        } else {
            candle_nn::Activation::Silu
        },
        intermediate_size: md_get("clip.vision.feed_forward_length")?.to_u32()? as usize,
        num_heads: md_get("clip.vision.attention.head_count")?.to_u32()? as usize,
        in_chans: 3,
        patch_size: md_get("clip.vision.patch_size")?.to_u32()? as usize,
        spatial_merge_size: md_get("clip.vision.spatial_merge_size")?.to_u32()? as usize,
        temporal_patch_size: 2,
        num_position_embeddings: {
            let image_size = md_get("clip.vision.image_size")?.to_u32()? as usize;
            let patch_size = md_get("clip.vision.patch_size")?.to_u32()? as usize;
            (image_size / patch_size).pow(2)
        },
        deepstack_visual_indexes: qwen3_vl_deepstack_indexes(mmproj_ct)?,
    };
    let cfg = Qwen3VLGgufConfig {
        architectures: text_config.architectures.clone(),
        text_config: text_config.clone(),
        vision_config: vision_cfg,
        image_token_id: tokenizer_token_id(tokenizer, "<|image_pad|>")?,
        video_token_id: tokenizer_token_id(tokenizer, "<|video_pad|>")?,
        vision_start_token_id: tokenizer_token_id(tokenizer, "<|vision_start|>")?,
        vision_end_token_id: tokenizer_token_id(tokenizer, "<|vision_end|>")?,
        tie_word_embeddings: text_config.tie_word_embeddings.unwrap_or(false),
        quantization_config: None,
    };
    let mut root = serde_json::to_value(&cfg).map_err(candle_core::Error::wrap)?;
    if let Some(raw_text_extra) = &text_config.extra_config_json {
        if let Ok(extra_root) = serde_json::from_str::<serde_json::Value>(raw_text_extra) {
            if let (Some(text_cfg), Some(extra_obj)) = (
                root.get_mut("text_config").and_then(|v| v.as_object_mut()),
                extra_root.as_object(),
            ) {
                for (key, value) in extra_obj {
                    text_cfg.insert(key.clone(), value.clone());
                }
            }
        }
    }
    serde_json::to_string(&root).map_err(candle_core::Error::wrap)
}

#[derive(Debug, serde::Deserialize)]
struct DummyMultiModelConfig {
    architectures: Option<Vec<String>>,
    text_config: Option<serde_json::Value>,
    vision_config: Option<serde_json::Value>,
}

fn is_multi_model(config_path: &PathBuf) -> Result<DummyMultiModelConfig> {
    let config: DummyMultiModelConfig =
        serde_json::from_slice(&std::fs::read(config_path).map_err(candle_core::Error::wrap)?)
            .map_err(candle_core::Error::wrap)?;
    Ok(config)
}

fn merge_multimodal_top_level_config(
    config: &mut Config,
    raw_root: &serde_json::Value,
) -> Result<()> {
    if let Some(qcfg) = raw_root.get("quantization_config") {
        if !qcfg.is_null() {
            let mut parsed = serde_json::from_value::<QuantConfig>(qcfg.clone())
                .map_err(candle_core::Error::wrap)?;
            parsed.normalize_compressed_tensors();
            config.quantization_config = Some(parsed);
        }
    }

    if let Some(v) = raw_root
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
    {
        config.tie_word_embeddings = Some(v);
    }

    if let Some(bos) = raw_root.get("bos_token_id") {
        if !bos.is_null() {
            if let Ok(bos_id) = serde_json::from_value::<usize>(bos.clone()) {
                config.bos_token_id = Some(bos_id);
            }
        }
    }

    if let Some(eos) = raw_root.get("eos_token_id") {
        if !eos.is_null() {
            if let Ok(eos_token_id) = serde_json::from_value::<EosTokenId>(eos.clone()) {
                config.eos_token_id = Some(eos_token_id);
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Qwen3HybridRawConfig {
    #[serde(alias = "layer_types")]
    pub layers_block_type: Option<Vec<String>>,
    #[serde(alias = "linear_conv_kernel_dim")]
    pub conv_kernel_size: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub linear_num_heads: Option<usize>,
    #[serde(alias = "linear_num_key_heads")]
    pub linear_num_key_heads: Option<usize>,
    #[serde(alias = "linear_num_value_heads")]
    pub linear_num_value_heads: Option<usize>,
    pub linear_num_key_value_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Qwen3HybridConfig {
    pub layer_types: Vec<String>,
    pub conv_kernel_size: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

pub fn is_qwen3_hybrid_arch_name(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5ForCausalLM"
            | "Qwen3_5MoeForCausalLM"
            | "Qwen3NextForCausalLM"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForConditionalGeneration"
    )
}

fn is_qwen3_hybrid_arch(config: &Config) -> bool {
    let arch = config.architectures.as_ref().and_then(|a| a.first());
    arch.map(|a| is_qwen3_hybrid_arch_name(a)).unwrap_or(false)
}

fn qwen3_hybrid_raw_from_extra_config(config: &Config) -> Option<Qwen3HybridRawConfig> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root = serde_json::from_str::<serde_json::Value>(extra).ok()?;
    let cfg = root.get("text_config").cloned().unwrap_or(root);
    serde_json::from_value::<Qwen3HybridRawConfig>(cfg).ok()
}

pub fn resolve_qwen3_hybrid_config(config: &Config) -> Qwen3HybridConfig {
    let raw_cfg = qwen3_hybrid_raw_from_extra_config(config).unwrap_or_default();

    let mut layer_types = if let Some(layer_types) = raw_cfg.layers_block_type {
        layer_types
    } else if let Some(interval) = raw_cfg.full_attention_interval {
        if interval > 0 {
            (0..config.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec!["full_attention".to_string(); config.num_hidden_layers]
        }
    } else {
        vec!["full_attention".to_string(); config.num_hidden_layers]
    };

    for layer_type in layer_types.iter_mut() {
        if layer_type == "attention" {
            *layer_type = "full_attention".to_string();
        }
    }
    if layer_types.len() != config.num_hidden_layers {
        crate::log_warn!(
            "Qwen3 hybrid layer_types length {} != num_hidden_layers {}, fallback to full_attention.",
            layer_types.len(),
            config.num_hidden_layers
        );
        layer_types = vec!["full_attention".to_string(); config.num_hidden_layers];
    }

    let num_v_heads = raw_cfg
        .linear_num_value_heads
        .or(raw_cfg.linear_num_heads)
        .unwrap_or(config.num_attention_heads);
    let num_k_heads = raw_cfg
        .linear_num_key_heads
        .or(raw_cfg.linear_num_key_value_heads)
        .unwrap_or(num_v_heads);
    let key_head_dim = raw_cfg.linear_key_head_dim.unwrap_or(
        config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads),
    );
    let value_head_dim = raw_cfg.linear_value_head_dim.unwrap_or(key_head_dim);
    let conv_kernel_size = raw_cfg.conv_kernel_size.unwrap_or(4);

    Qwen3HybridConfig {
        layer_types,
        conv_kernel_size,
        num_v_heads,
        num_k_heads,
        key_head_dim,
        value_head_dim,
    }
}

pub fn qwen3_hybrid_layer_types(config: &Config) -> Option<Vec<String>> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    Some(resolve_qwen3_hybrid_config(config).layer_types)
}

/// For Gemma4 models with heterogeneous head dims (SWA=head_dim, full_attention=global_head_dim),
/// returns per-layer (num_kv_heads, head_dim) for KV cache allocation.
///
/// Handles three config layouts:
/// - HF multimodal (`Gemma4ForConditionalGeneration`): keys nested under `text_config`
/// - HF text-only (`Gemma4ForCausalLM`): keys at top level of `extra_config_json`
/// - GGUF: synthetic JSON with `swa_head_dim`/`global_head_dim` at top level
pub fn gemma4_per_layer_cache_config(config: &Config) -> Option<Vec<(usize, usize)>> {
    let arch = config.architectures.as_ref()?.first()?;
    if !arch.contains("Gemma4") {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root: serde_json::Value = serde_json::from_str(extra).ok()?;
    let cfg = root.get("text_config").unwrap_or(&root);

    let get = |key: &str| -> Option<&serde_json::Value> { cfg.get(key).or_else(|| root.get(key)) };

    let layer_types: Vec<String> =
        get("layer_types").and_then(|value| serde_json::from_value(value.clone()).ok())?;
    if layer_types.len() != config.num_hidden_layers {
        crate::log_warn!(
            "Gemma4 layer_types length {} != num_hidden_layers {}; ignoring heterogeneous KV cache config.",
            layer_types.len(),
            config.num_hidden_layers
        );
        return None;
    }

    let swa_head_dim = get("swa_head_dim")
        .or_else(|| cfg.get("head_dim"))
        .or_else(|| root.get("head_dim"))
        .and_then(|v| v.as_u64())? as usize;
    let global_head_dim = get("global_head_dim").and_then(|v| v.as_u64())? as usize;
    let swa_kv_heads = get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(config.num_key_value_heads as u64) as usize;
    let global_kv_heads = get("num_global_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(swa_kv_heads as u64) as usize;

    let per_layer = layer_types
        .iter()
        .map(|lt| {
            if lt == "full_attention" {
                (global_kv_heads, global_head_dim)
            } else {
                (swa_kv_heads, swa_head_dim)
            }
        })
        .collect::<Vec<_>>();

    if per_layer
        .iter()
        .all(|&(kv_heads, head_dim)| kv_heads == swa_kv_heads && head_dim == swa_head_dim)
    {
        return None;
    }

    Some(per_layer)
}

fn require_model_penalty(arch: String) -> bool {
    matches!(
        arch.as_str(),
        "Glm4ForCausalLM"
            | "Glm4ForConditionalGeneration"
            | "glm4"
            | "Phi3ForCausalLM"
            | "Phi4ForCausalLM"
            | "phi3"
            | "phi4"
            | "Gemma3ForConditionalGeneration"
            | "Gemma3ForCausalLM"
            | "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
    )
}

fn apply_qwen35_next_moe_norm_topk_default(config: &mut Config) {
    let arch = config
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("");
    if !matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
    ) {
        return;
    }

    let Some(moe_cfg) = config.moe_cfg.as_mut() else {
        return;
    };

    let Some(raw) = config.extra_config_json.as_ref() else {
        return;
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(raw) else {
        return;
    };
    let cfg_root = root.get("text_config").unwrap_or(&root);

    if cfg_root.get("norm_topk_prob").is_none() {
        moe_cfg.norm_topk_prob = true;
    }
}

pub fn init_config_tokenizer(
    econfig: &EngineConfig,
) -> Result<(
    ModelPaths,
    bool,
    Config,
    TokenizerConfig,
    Tokenizer,
    Option<GenerationConfig>,
)> {
    let loader = crate::utils::downloader::Downloader::new(
        econfig.model_id.clone(),
        econfig.weight_path.clone(),
        econfig.weight_file.clone(),
    );
    let (model_pathes, is_gguf) =
        loader.prepare_model_weights(econfig.hf_token.clone(), econfig.hf_token_path.clone())?;
    if !is_gguf {
        let config_path = model_pathes.get_config_filename();
        let mut config: Config = if let Ok(cfg) = is_multi_model(&config_path) {
            if cfg.text_config.is_some() && cfg.vision_config.is_some() {
                crate::log_warn!("Multimodel model {:?} detected!", cfg.architectures);
                let raw_config = std::fs::read(&config_path).map_err(candle_core::Error::wrap)?;
                let raw_config_json: serde_json::Value =
                    serde_json::from_slice(&raw_config).map_err(candle_core::Error::wrap)?;
                let Some(mut config_value) = cfg.text_config else {
                    panic!("Not supported model type {:?}", cfg.architectures);
                };

                let mut config: Config = match cfg.architectures.as_ref().unwrap()[0].as_str() {
                    "Gemma3ForConditionalGeneration" => {
                        let gemma3_cfg: Gemma3Config = serde_json::from_slice(&raw_config)
                            .map_err(candle_core::Error::wrap)?;
                        config_value = serde_json::to_value(&gemma3_cfg.text_config)
                            .map_err(candle_core::Error::wrap)?;
                        let mut config: Config = serde_json::from_value(config_value)
                            .map_err(candle_core::Error::wrap)?;
                        config.eos_token_id = gemma3_cfg.eos_token_id;
                        config
                    }
                    "Gemma4ForConditionalGeneration" => {
                        let mut cv = config_value.clone();
                        if let Some(obj) = cv.as_object_mut() {
                            obj.remove("rope_parameters");
                        }
                        let mut config: Config =
                            serde_json::from_value(cv).map_err(candle_core::Error::wrap)?;
                        let tc = &raw_config_json["text_config"];
                        let tc_usize =
                            |key: &str| tc.get(key).and_then(|v| v.as_u64()).map(|v| v as usize);
                        let tc_f64 = |key: &str| tc.get(key).and_then(|v| v.as_f64());
                        let tc_bool = |key: &str| tc.get(key).and_then(|v| v.as_bool());
                        let tc_string =
                            |key: &str| tc.get(key).and_then(|v| v.as_str()).map(str::to_owned);
                        if let Some(num_experts) = tc.get("num_experts").and_then(|v| v.as_u64()) {
                            if num_experts > 0 {
                                let top_k =
                                    tc.get("top_k_experts")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(8) as usize;
                                let moe_intermediate = tc
                                    .get("moe_intermediate_size")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(config.intermediate_size as u64)
                                    as usize;
                                config.moe_cfg = Some(MoEConfig {
                                    moe_intermediate_size: moe_intermediate,
                                    shared_expert_intermediate_size: None,
                                    num_experts: Some(num_experts as usize),
                                    mlp_only_layers: Some(Vec::new()),
                                    decoder_sparse_step: Some(1),
                                    norm_topk_prob: tc_bool("norm_topk_prob").unwrap_or(true),
                                    num_experts_per_tok: top_k,
                                    first_k_dense_replace: None,
                                    n_shared_experts: None,
                                    routed_scaling_factor: tc_f64("routed_scaling_factor"),
                                    n_group: tc_usize("n_group"),
                                    topk_group: tc_usize("topk_group"),
                                    scoring_func: tc_string("scoring_func"),
                                    topk_method: tc_string("topk_method"),
                                });
                            }
                        }
                        if let Some(rp) = tc.get("rope_parameters") {
                            if let Some(fa) = rp.get("full_attention") {
                                if let Some(theta) = fa.get("rope_theta").and_then(|v| v.as_f64()) {
                                    config.rope_theta = Some(theta);
                                }
                                if let Some(prf) =
                                    fa.get("partial_rotary_factor").and_then(|v| v.as_f64())
                                {
                                    config.partial_rotary_factor = Some(prf as f32);
                                }
                            }
                        }
                        if let Some(eos) = raw_config_json.get("eos_token_id") {
                            config.eos_token_id = serde_json::from_value(eos.clone()).ok();
                        }
                        if let Some(ghd) = tc.get("global_head_dim").and_then(|v| v.as_u64()) {
                            config.head_dim = Some(ghd as usize);
                        }
                        config
                    }
                    "Qwen3VLMoeForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
                        let mut config: Config = serde_json::from_value(config_value.clone())
                            .map_err(candle_core::Error::wrap)?;
                        let moe_cfg: MoEConfig = serde_json::from_value(config_value)
                            .map_err(candle_core::Error::wrap)?;
                        config.moe_cfg = Some(moe_cfg);
                        config
                    }
                    _ => serde_json::from_value(config_value).map_err(candle_core::Error::wrap)?,
                };

                config.architectures = cfg.architectures.clone();
                config.is_multi_model = Some(true);
                merge_multimodal_top_level_config(&mut config, &raw_config_json)?;
                config.extra_config_json =
                    Some(String::from_utf8(raw_config).map_err(candle_core::Error::wrap)?);
                // Remap rope_theta in rope_scaling to config file
                if let Some(scaling) = &config.rope_scaling {
                    if let Some(v) = scaling.get("rope_theta").and_then(|v| v.as_f64()) {
                        config.rope_theta = Some(v);
                    }
                    if let Some(v) = scaling
                        .get("partial_rotary_factor")
                        .and_then(|v| v.as_f64())
                    {
                        config.partial_rotary_factor = Some(v as f32);
                    }
                }
                config
            } else {
                serde_json::from_slice(
                    &std::fs::read(&config_path).map_err(candle_core::Error::wrap)?,
                )
                .map_err(candle_core::Error::wrap)?
            }
        } else {
            serde_json::from_slice(&std::fs::read(&config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?
        };

        apply_runtime_rope_overrides(&mut config, econfig.yarn_scaling_factor);

        if config.extra_config_json.is_none() {
            if let Ok(raw) = std::fs::read_to_string(&config_path) {
                config.extra_config_json = Some(raw);
            }
        }

        if let Some(qcfg) = &mut config.quantization_config {
            qcfg.normalize_compressed_tensors();
            if let Some(mode) = &qcfg.mode {
                if mode.eq_ignore_ascii_case("nvfp4") || mode.eq_ignore_ascii_case("mxfp4") {
                    panic!(
                        "MLX-quantized models (mode=\"{}\") are not supported. \
                         MLX uses an incompatible packing format (U32 weights with integer scales). \
                         Please use a modelopt or compressed-tensors quantized model instead \
                         (e.g. AxionML/Qwen3.5-*-NVFP4 or nvidia/*-NVFP4).",
                        mode
                    );
                }
            }
            assert!(
                qcfg.quant_method == "gptq"
                    || qcfg.quant_method == "awq"
                    || qcfg.quant_method == "fp8"
                    || qcfg.quant_method == "mxfp4"
                    || qcfg.quant_method == "nvfp4",
                "Invalid quantization format! Only `gptq`, `awq`, `fp8`, `mxfp4` and `nvfp4` supported, got `{}`",
                qcfg.quant_method
            );
            if qcfg.quant_method == "gptq" || qcfg.quant_method == "awq" {
                assert!(
                    (qcfg.bits == 4 || qcfg.bits == 8),
                    "Only 4-bit and 8-bit gptq or awq models supported!"
                );
                if qcfg.desc_act.unwrap_or(false) {
                    candle_core::bail!("desc_act==true not supported!");
                }
                #[cfg(not(feature = "cuda"))]
                candle_core::bail!("GPTQ/AWQ models are only supported under CUDA platform!");
            }
        }
        let arch_name = config.architectures.as_ref().unwrap()[0].clone();
        if config.moe_cfg.is_none()
            && matches!(
                arch_name.as_str(),
                "Qwen2MoeForCausalLM"
                    | "Qwen3MoeForCausalLM"
                    | "Glm4MoeForCausalLM"
                    | "Glm4MoeLiteForCausalLM"
                    | "DeepseekV3ForCausalLM"
                    | "DeepseekV32ForCausalLM"
                    | "DeepseekForCausalLM"
                    | "Qwen3_5MoeForCausalLM"
                    | "Qwen3_5MoeForConditionalGeneration"
                    | "Qwen3NextForCausalLM"
                    | "Qwen3NextForConditionalGeneration"
                    | "MiniMaxM2ForCausalLM"
            )
        {
            if let Ok(raw_cfg) = std::fs::read(&config_path) {
                if let Some(moe_cfg) = parse_fallback_moe_cfg(&arch_name, &raw_cfg) {
                    if moe_cfg.num_experts.unwrap_or(0) > 0 {
                        config.moe_cfg = Some(moe_cfg);
                    }
                }
            }
        }
        apply_qwen35_next_moe_norm_topk_default(&mut config);

        if let Some(moe_cfg) = config.moe_cfg.as_mut() {
            if moe_cfg.shared_expert_intermediate_size.is_none() {
                if let Some(n_shared) = moe_cfg.n_shared_experts {
                    if n_shared > 0 {
                        moe_cfg.shared_expert_intermediate_size =
                            Some(moe_cfg.moe_intermediate_size);
                    }
                }
            }
            if arch_name == "MiniMaxM2ForCausalLM" {
                moe_cfg.norm_topk_prob = true;
            }
        }

        config.quant = econfig.isq.clone();
        let tokenizer_config_path = model_pathes.get_tokenizer_config_filename();
        let mut config_tokenizer: TokenizerConfig = {
            match std::fs::read(tokenizer_config_path).map_err(candle_core::Error::wrap) {
                Ok(f) => serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?,
                _ => {
                    crate::log_error!(
                        "Missing tokenizer_config.json file, chat template may not correct!"
                    );
                    TokenizerConfig {
                        model_max_length: None,
                        add_bos_token: None,
                        add_eos_token: None,
                        chat_template: None,
                        bos_token: None,
                        eos_token: None,
                    }
                }
            }
        };
        let tokenizer_file = model_pathes.get_tokenizer_filename();

        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_file).map_err(candle_core::Error::wrap)?;
        let _ = tokenizer.with_truncation(None);
        let _ = tokenizer.with_padding(None);

        let generation_config_path = model_pathes.get_generation_config_filename();
        let generation_cfg = if generation_config_path.display().to_string() != ""
            && Path::new(&generation_config_path).exists()
        {
            let str_cfg: Option<String> = std::fs::read_to_string(generation_config_path).ok();
            let cfg: GenerationConfig = serde_json::from_str(str_cfg.unwrap().as_str()).unwrap();
            Some(cfg)
        } else {
            if require_model_penalty(arch_name.clone()) {
                Some(GenerationConfig {
                    temperature: Some(0.7),
                    top_p: Some(0.9),
                    top_k: None,
                    frequency_penalty: Some(1.2),
                    presence_penalty: Some(1.2),
                    bos_token_id: None,
                    eos_token_id: None,
                })
            } else {
                None
            }
        };

        // Handle jinja chat template
        if config_tokenizer.chat_template.is_none() {
            if let Some(dir) = Path::new(&config_path).parent() {
                if dir.join("chat_template.jinja").exists() {
                    crate::log_warn!("Try loading chat template from chat_template.jinja");
                    config_tokenizer.chat_template = Some(
                        std::fs::read_to_string(&dir.join("chat_template.jinja"))
                            .map_err(candle_core::Error::wrap)?,
                    );
                }
            } else if let Some(f) = model_pathes.get_chat_template_filename() {
                crate::log_warn!("Try loading chat template from chat_template.json");
                config_tokenizer.chat_template =
                    Some(std::fs::read_to_string(&f).map_err(candle_core::Error::wrap)?);
            }
        }

        Ok((
            model_pathes,
            is_gguf,
            config,
            config_tokenizer,
            tokenizer,
            generation_cfg,
        ))
    } else if !model_pathes.get_weight_filenames().is_empty()
        && model_pathes.get_weight_filenames()[0].exists()
    {
        assert!(econfig.isq.is_none(), "GGUF model does not support ISQ! \n\t***Tips: use `--w` to specify safetensors model path!***");
        let auxiliary_weight_files = model_pathes.get_auxiliary_filenames();
        let GGUFInfo {
            tokenizer,
            bos,
            eos,
            unk: _,
            context_length,
            chat_template,
        } = load_gguf_info_from_files(&model_pathes.get_weight_filenames()).map_err(|e| {
            candle_core::Error::msg(format!(
                "Unable to read {:?} as a GGUF file: {e}\n\t***Tips: use `--w` to specify safetensor model directory!***",
                model_pathes.get_weight_filenames()[0]
            ))
        })?;

        let config = {
            let mut file = std::fs::File::open(&model_pathes.get_weight_filenames()[0]).unwrap();
            let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
            let mut config = config_from_gguf(&content, &mut file)?;
            let arch_name = config
                .architectures
                .as_ref()
                .and_then(|archs| archs.first())
                .map(|s| s.as_str())
                .unwrap_or("");
            if let Some(aux_path) = auxiliary_weight_files.first() {
                if matches!(
                    arch_name,
                    "Qwen3VLForConditionalGeneration"
                        | "Qwen3VLMoeForConditionalGeneration"
                        | "Qwen3_5ForConditionalGeneration"
                        | "Qwen3_5MoeForConditionalGeneration"
                ) {
                    crate::log_info!("Loading GGUF multimodal config from {}", aux_path.display());
                    let mut aux_file =
                        std::fs::File::open(aux_path).map_err(candle_core::Error::wrap)?;
                    let aux_content =
                        candle_core::quantized::gguf_file::Content::read(&mut aux_file)
                            .map_err(candle_core::Error::wrap)?;
                    config.is_multi_model = Some(true);
                    config.extra_config_json = Some(build_qwen3_vl_gguf_extra_config(
                        &config,
                        &aux_content,
                        &tokenizer,
                    )?);
                } else {
                    crate::log_warn!(
                        "Auxiliary GGUF file(s) detected, but multimodal GGUF loading is not implemented for architecture {}. Loading text-only model.",
                        arch_name
                    );
                }
            } else {
                if matches!(
                    arch_name,
                    "Qwen3_5ForConditionalGeneration"
                        | "Qwen3_5MoeForConditionalGeneration"
                        | "Qwen3VLForConditionalGeneration"
                        | "Qwen3VLMoeForConditionalGeneration"
                        | "Gemma3ForConditionalGeneration"
                        | "Gemma4ForConditionalGeneration"
                        | "Mistral3ForConditionalGeneration"
                ) {
                    crate::log_error!(
                        "No auxiliary GGUF mmproj file found for multimodal GGUF model. Vision is disabled and the model will run in text-only mode."
                    );
                }
            }
            apply_runtime_rope_overrides(&mut config, econfig.yarn_scaling_factor);
            config
        };

        let config_tokenizer = TokenizerConfig {
            model_max_length: Some(context_length.unwrap_or(config.max_position_embeddings) as f64),
            add_bos_token: Some(bos.is_some()),
            add_eos_token: Some(eos.is_some()),
            chat_template,
            bos_token: bos,
            eos_token: eos,
        };
        let archs = config.architectures.as_ref().unwrap();

        let generation_cfg = if require_model_penalty(archs[0].clone()) {
            Some(GenerationConfig {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: None,
                frequency_penalty: Some(1.2),
                presence_penalty: Some(1.2),
                bos_token_id: None,
                eos_token_id: None,
            })
        } else {
            None
        };

        Ok((
            model_pathes,
            is_gguf,
            config,
            config_tokenizer,
            tokenizer,
            generation_cfg,
        ))
    } else {
        candle_core::bail!("Model file(s) not found!");
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

pub fn get_runner_path() -> Result<PathBuf> {
    #[cfg(feature = "python")]
    {
        Python::with_gil(|py| {
            let module = PyModule::import(py, "xinfer").map_err(candle_core::Error::wrap)?;
            let file: String = module
                .getattr("__file__")
                .map_err(candle_core::Error::wrap)?
                .extract()
                .map_err(candle_core::Error::wrap)?;
            let module_path = Path::new(&file).parent().unwrap().join("xinfer");
            Ok(module_path)
        })
    }

    #[cfg(not(feature = "python"))]
    {
        let exe_path = std::env::current_exe()?;
        Ok(exe_path)
    }
}

pub fn spawn_runner(
    #[cfg(feature = "python")] py: Python,
    runner_path: &str,
    sock_name: &str,
    uuid_str: &str,
) -> Result<()> {
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::{PyDict, PyString, PyTuple};
        crate::log_info!("Spawning runner at: {}", runner_path);
        let subprocess = py.import("subprocess").map_err(candle_core::Error::wrap)?;

        let args = PyTuple::new(
            py,
            &[
                PyString::new(py, runner_path),
                PyString::new(py, "runner"),
                PyString::new(py, "--sock"),
                PyString::new(py, sock_name),
                PyString::new(py, "--uuid"),
                PyString::new(py, uuid_str),
            ],
        )
        .map_err(candle_core::Error::wrap)?;

        let kwargs = PyDict::new(py);
        kwargs
            .set_item("shell", false)
            .map_err(candle_core::Error::wrap)?;
        kwargs
            .set_item("text", true)
            .map_err(candle_core::Error::wrap)?;
        let libs_dir = Path::new(runner_path)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("xinfer.libs"));
        if let Some(libs_dir) = libs_dir {
            if libs_dir.exists() {
                let abs_libs_dir = libs_dir.canonicalize().unwrap_or(libs_dir);
                crate::log_warn!(
                    "Runner rpath not set; preparing LD_LIBRARY_PATH for {}",
                    abs_libs_dir.display()
                );
                let env_result: std::result::Result<(), String> = (|| {
                    let os = py.import("os").map_err(|e| e.to_string())?;
                    let env_any = os
                        .getattr("environ")
                        .map_err(|e| e.to_string())?
                        .call_method0("copy")
                        .map_err(|e| e.to_string())?;
                    let env = env_any.downcast::<PyDict>().map_err(|e| e.to_string())?;
                    let mut ld_paths: Vec<String> = Vec::new();
                    #[cfg(target_os = "linux")]
                    {
                        let tmp_dir = std::env::temp_dir().join("xinfer.libs");
                        if std::fs::create_dir_all(&tmp_dir).is_ok() {
                            if let Ok(entries) = std::fs::read_dir(&abs_libs_dir) {
                                for entry in entries {
                                    let entry = match entry {
                                        Ok(entry) => entry,
                                        Err(err) => {
                                            crate::log_warn!("Skipping bundled lib entry: {}", err);
                                            continue;
                                        }
                                    };
                                    let path = entry.path();
                                    if !path.is_file() {
                                        continue;
                                    }
                                    let name = match path.file_name().and_then(|s| s.to_str()) {
                                        Some(name) => name,
                                        None => continue,
                                    };
                                    if !name.starts_with("lib") {
                                        continue;
                                    }
                                    let so_idx = match name.rfind(".so.") {
                                        Some(idx) => idx,
                                        None => continue,
                                    };
                                    let base = &name[..so_idx];
                                    let dash_idx = match base.rfind('-') {
                                        Some(idx) => idx,
                                        None => continue,
                                    };
                                    let unsuffixed =
                                        format!("{}{}", &base[..dash_idx], &name[so_idx..]);
                                    let link_path = tmp_dir.join(&unsuffixed);
                                    if let Ok(existing) = std::fs::read_link(&link_path) {
                                        if existing == path {
                                            continue;
                                        }
                                        let _ = std::fs::remove_file(&link_path);
                                    }
                                    if let Err(err) = std::os::unix::fs::symlink(&path, &link_path)
                                    {
                                        crate::log_warn!(
                                            "Failed to create symlink {}: {}",
                                            link_path.display(),
                                            err
                                        );
                                    }
                                }
                            }
                            ld_paths.push(tmp_dir.to_string_lossy().to_string());
                            crate::log_warn!(
                                "Runner using symlink dir for bundled libs: {}",
                                tmp_dir.display()
                            );
                        } else {
                            crate::log_warn!("Failed to create temp symlink dir for bundled libs");
                        }
                    }
                    ld_paths.push(abs_libs_dir.to_string_lossy().to_string());
                    let ld_prefix = ld_paths.join(":");
                    let new_ld = match env.get_item("LD_LIBRARY_PATH").map_err(|e| e.to_string())? {
                        Some(val) => {
                            let existing: String = val.extract().map_err(|e| e.to_string())?;
                            if existing.is_empty() {
                                ld_prefix
                            } else {
                                format!("{}:{}", ld_prefix, existing)
                            }
                        }
                        None => ld_prefix,
                    };
                    env.set_item("LD_LIBRARY_PATH", new_ld)
                        .map_err(|e| e.to_string())?;
                    kwargs.set_item("env", env).map_err(|e| e.to_string())?;
                    Ok(())
                })();
                if let Err(err) = env_result {
                    crate::log_warn!("Failed to set LD_LIBRARY_PATH fallback: {}", err);
                }
            }
        }

        let result = subprocess
            .call_method("Popen", (args,), Some(&kwargs))
            .map_err(candle_core::Error::wrap)?;
        crate::log_info!("Runner spawned {:?}", result);
        Ok(())
    }
    #[cfg(not(feature = "python"))]
    {
        use std::process::Command;

        Command::new(runner_path)
            .arg("runner")
            .arg("--sock")
            .arg(sock_name)
            .arg("--uuid")
            .arg(uuid_str)
            .spawn()
            .map_err(|e| e.into())
            .map(|_child| ())
    }
}

pub fn is_no_cuda_graph_supprt(architectures: String) -> bool {
    #[allow(unused_mut)]
    let mut black_list = vec!["Phi3ForCausalLM", "Phi4ForCausalLM", "phi3", "phi4"];

    #[cfg(not(feature = "flashinfer"))]
    {
        black_list.extend(vec!["Glm4MoeLiteForCausalLM", "DeepseekV3ForCausalLM"]);
    }

    black_list.contains(&architectures.as_str())
}

pub fn get_arch_rope(
    tokenizer: &Tokenizer,
    architectures: String,
) -> Result<(ModelType, String, bool)> {
    let rope_key_map: HashMap<&str, bool> = [
        ("Qwen2ForCausalLM", false),
        ("Qwen3ForCausalLM", false),
        ("Qwen3ForConditionalGeneration", false),
        ("Qwen3VLForConditionalGeneration", false),
        ("Qwen3VLMoeForConditionalGeneration", false),
        ("Glm4MoeForCausalLM", true),
        ("Glm4MoeLiteForCausalLM", true),
        ("DeepseekV3ForCausalLM", false),
        ("DeepseekV32ForCausalLM", false),
        ("DeepseekForCausalLM", false),
        ("Phi3ForCausalLM", false),
        ("Phi4ForCausalLM", false),
        ("MistralForCausalLM", false),
        ("Mistral3ForConditionalGeneration", false),
        ("LlamaForCausalLM", false),
        ("LlamaForConditionalGeneration", false),
        ("IQuestCoderForCausalLM", false),
        ("Glm4ForCausalLM", true),
        ("glm4", true),
        ("qwen2", false),
        ("qwen3", false),
        ("phi3", false),
        ("phi4", false),
        ("llama", true),
        ("mistral", true),
        ("mistral3", false),
        ("Gemma3ForConditionalGeneration", false),
        ("Gemma3ForCausalLM", false),
        ("Llama4ForConditionalGeneration", true),
        ("llama4", true),
        ("Qwen3_5ForCausalLM", false),
        ("Qwen3_5ForConditionalGeneration", false),
        ("Qwen3_5MoeForCausalLM", false),
        ("Qwen3_5MoeForConditionalGeneration", false),
        ("Qwen3NextForCausalLM", false),
        ("Qwen3NextForConditionalGeneration", false),
        ("qwen35", false),
        ("qwen35moe", false),
        ("qwen3vl", false),
        ("qwen3vlmoe", false),
        ("gemma3", false),
        ("Gemma4ForConditionalGeneration", false),
        ("Gemma4ForCausalLM", false),
        ("gemma4", false),
        ("MiniMaxM2ForCausalLM", false),
        ("minimax_m2", false),
    ]
    .iter()
    .cloned()
    .collect();

    let arch = architectures.as_str();
    let (model_type, default_chat_template) = match arch {
        "Qwen2ForCausalLM"
        | "Qwen2ForConditionalGeneration"
        | "Qwen3ForCausalLM"
        | "Qwen3ForConditionalGeneration"
        | "qwen2"
        | "qwen3" => (
            ModelType::Qwen3,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "qwen2moe" | "Qwen2MoeForCausalLM" | "qwen3moe" | "Qwen3MoeForCausalLM" => (
            ModelType::Qwen3MoE,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "Qwen3_5ForCausalLM" | "qwen35" => (
            ModelType::Qwen3_5,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "Qwen3_5MoeForCausalLM" | "Qwen3NextForCausalLM" | "qwen35moe" => (
            ModelType::Qwen3_5MoE,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "Qwen3VLForConditionalGeneration"
        | "Qwen3VLMoeForConditionalGeneration"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration"
        | "Qwen3NextForConditionalGeneration"
        | "qwen3vl"
        | "qwen3vlmoe" => (
            ModelType::Qwen3VL,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "LlamaForCausalLM"
        | "MistralForCausalLM"
        | "Mistral3ForConditionalGeneration"
        | "LlamaForConditionalGeneration"
        | "IQuestCoderForCausalLM"
        | "llama"
        | "mistral"
        | "mistral3"
        | "llama2"
        | "llama3" => {
            let model_type = if arch == "Mistral3ForConditionalGeneration" {
                ModelType::Mistral3VL
            } else {
                ModelType::LLaMa
            };
            if let Some(_) = tokenizer
                .get_vocab(true)
                .get("<|start_header_id|>")
                .copied()
            {
                //llama3
                (
                    model_type,
                    "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>".to_string(),
                )
            } else {
                //llama2
                (model_type, "[INST] {} [/INST]".to_string())
            }
        }
        "Glm4ForCausalLM" | "Glm4ForConditionalGeneration" | "glm4" => (
            ModelType::GLM4,
            "[gMASK]<sop><|user|>{}<|assistant|>".to_string(),
        ),
        "Glm4MoeForCausalLM" | "glm4moe" => (
            ModelType::GLM4MoE,
            "[gMASK]<sop><|user|>{}<|assistant|>".to_string(),
        ),
        "Glm4MoeLiteForCausalLM" | "glm4moelite" => (
            ModelType::GLM4MoeLite,
            "[gMASK]<sop><|user|>{}<|assistant|>".to_string(),
        ),
        "DeepseekV3ForCausalLM"
        | "DeepseekV32ForCausalLM"
        | "DeepseekForCausalLM"
        | "deepseek3"
        | "deepseek" => (ModelType::DeepSeek, "<|User|>{}<|Assistant|>".to_string()),
        "Phi3ForCausalLM" | "Phi4ForCausalLM" | "phi3" | "phi4" => {
            (ModelType::Phi4, "<|user|>\n{}<|assistant|>".to_string())
        }
        "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM" | "gemma3" => (
            ModelType::Gemma3,
            "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>".to_string(),
        ),
        "Llama4ForConditionalGeneration" | "llama4" => (
            ModelType::LLaMa4,
            "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>".to_string(),
        ),
        "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM" | "gemma4" => (
            ModelType::Gemma4,
            "<|turn>user\n{}<turn|>\n<|turn>model\n".to_string(),
        ),
        "MiniMaxM2ForCausalLM" | "minimax_m2" => (
            ModelType::MiniMax,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        _ => candle_core::bail!("Unsupported architecture: {}", architectures),
    };

    let is_rope_i = if rope_key_map.contains_key(arch) {
        rope_key_map[arch]
    } else {
        false
    };
    Ok((model_type, default_chat_template, is_rope_i))
}

pub fn get_dtype(dtype: Option<String>) -> DType {
    let dtype = match dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    #[cfg(feature = "cuda")]
    let dtype = {
        use candle_core::cuda_backend::cudarc::driver::result::{device, init};
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        match (init(), device::get(0)) {
            (Ok(_), Ok(d)) => {
                let (compute_major, compute_minor) = unsafe {
                    (
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )
                        .unwrap_or(8),
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )
                        .unwrap_or(8),
                    )
                };
                crate::log_info!(
                    "CUDA compute capability: {}.{}",
                    compute_major,
                    compute_minor,
                );
                if dtype != DType::F32 && compute_major < 8 {
                    crate::log_warn!(
                        "CUDA compute capability: {} (<8), switched to F16 cause no BF16 support.",
                        compute_major
                    );
                    DType::F16
                } else {
                    dtype
                }
            }
            _ => dtype,
        }
    };
    dtype
}

pub fn prepare_engine_config(
    econfig: &EngineConfig,
    config: &Config,
    config_tokenizer: &TokenizerConfig,
    generation_cfg: &mut Option<GenerationConfig>,
) -> (EngineConfig, bool) {
    let mut econfig = econfig.clone();
    econfig.prefill_chunk_size =
        crate::utils::config::normalize_prefill_chunk_size(econfig.prefill_chunk_size);

    let config_model_len = resolve_config_model_len(config, config_tokenizer);

    econfig.config_model_len = Some(config_model_len);

    if econfig.max_model_len.is_none() || econfig.max_model_len.unwrap() < config_model_len {
        crate::log_warn!(
            "This model has maximum context {} but the current config is {:?}!",
            config_model_len,
            econfig.max_model_len
        );
    }

    assert!(
        config.architectures.as_ref().unwrap().len() == 1,
        "Only one architecture is supported at the moment!"
    );

    match (&generation_cfg, &mut econfig.generation_cfg) {
        (Some(gen_cfg), None) => {
            econfig.generation_cfg = Some(gen_cfg.clone());
        }
        (Some(gen_cfg), Some(egen_cfg)) => {
            if egen_cfg.frequency_penalty.is_none() {
                egen_cfg.frequency_penalty = gen_cfg.frequency_penalty;
            }
            if egen_cfg.presence_penalty.is_none() {
                egen_cfg.presence_penalty = gen_cfg.presence_penalty;
            }
            if egen_cfg.temperature.is_none() {
                egen_cfg.temperature = gen_cfg.temperature;
                egen_cfg.top_p = gen_cfg.top_p;
                egen_cfg.top_k = gen_cfg.top_k;
            }
        }
        _ => {
            crate::log_warn!("No generation config found for this model!");
        }
    }

    let mut device_ids = econfig.device_ids.clone().unwrap_or_default();
    if device_ids.is_empty() {
        device_ids.push(0);
    }
    let num_shards = device_ids.len();
    econfig.device_ids = Some(device_ids);
    econfig.num_shards = Some(num_shards);

    #[cfg(not(feature = "nccl"))]
    assert!(
        num_shards == 1,
        "Multi-rank inference is only available when `nccl` feature is enabled!"
    );

    #[cfg(feature = "nccl")]
    let use_runner = true;

    #[cfg(not(feature = "nccl"))]
    assert!(
        num_shards == 1,
        "Multi-gpu inference is only available when `cuda` and `nccl` features enabled!"
    );
    #[cfg(not(feature = "nccl"))]
    let use_runner = num_shards > 1;

    crate::log_warn!("Check use_runner {:?}", use_runner);
    (econfig, use_runner)
}

pub fn get_llama4_attn_scale(
    positions: &candle_core::Tensor,
    llama_4_scaling_beta: f64,
    original_max_position_embeddings: f64,
) -> Result<candle_core::Tensor> {
    let div = (positions.to_dtype(DType::F32)? / original_max_position_embeddings)?;
    let floored = div.floor()?;

    let one = floored.ones_like()?; // tensor filled with 1.0
    let log_term = (one + floored)?.log()?;

    let scaling = (1f64 + (llama_4_scaling_beta * &log_term)?)?;
    scaling
        .unsqueeze(candle_core::D::Minus1)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

pub fn contains_gguf(path: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "gguf" {
                    return true;
                }
            }
        }
    }
    false
}

pub fn has_complete_safetensors(path: &Path) -> Result<bool> {
    use regex::Regex;
    use std::collections::HashSet;
    use std::fs;

    // Check for single model.safetensors file (small models without sharding)
    if path.join("model.safetensors").exists() {
        return Ok(true);
    }

    // Check for sharded safetensors (e.g., model-00001-of-00005.safetensors format)
    let re = Regex::new(r"^.+-(\d{5})-of-(\d{5})\.safetensors$").unwrap();

    let mut found_indices = HashSet::new();
    let mut expected_total: Option<u32> = None;

    for entry in fs::read_dir(path).map_err(candle_core::Error::wrap)? {
        let entry = entry.map_err(candle_core::Error::wrap)?;
        let filename = entry.file_name();
        let filename = filename.to_string_lossy();

        if let Some(caps) = re.captures(&filename) {
            let idx: u32 = caps[1].parse().map_err(candle_core::Error::wrap)?;
            let total: u32 = caps[2].parse().map_err(candle_core::Error::wrap)?;

            if let Some(expected) = expected_total {
                if expected != total {
                    return Ok(false); // inconsistent shard count
                }
            } else {
                expected_total = Some(total);
            }

            found_indices.insert(idx);
        }
    }

    let total = match expected_total {
        Some(t) => t,
        None => return Ok(false), // no safetensors found
    };

    crate::log_info!(
        "Local cache expect {total} safetensors, found {:?}",
        found_indices
    );
    // Ensure all shards 1..=total are present
    Ok((1..=total).all(|i| found_indices.contains(&i)))
}

pub fn log_throughput(outputs: &[GenerationOutput]) {
    use colored::Colorize;
    const EPS: f32 = 1e-6;
    if outputs.is_empty() {
        tracing::warn!("No outputs provided; cannot compute throughput.");
        return;
    }

    let mut total_prompt_tokens: usize = 0;
    let mut total_decoded_tokens: usize = 0;

    let mut prompt_time_taken: f32 = 0.0;
    let mut all_decode_time_taken: f32 = 0.0;

    for GenerationOutput {
        prompt_length,
        prompt_start_time,
        decode_start_time,
        decode_finish_time,
        decoded_length,
        ..
    } in outputs.iter()
    {
        total_prompt_tokens += *prompt_length as usize;
        total_decoded_tokens += *decoded_length as usize;

        let duration_prompt = (*decode_start_time - *prompt_start_time) as f32 / 1000.0;
        if duration_prompt > prompt_time_taken {
            prompt_time_taken = duration_prompt;
        }

        let duration_decode = (*decode_finish_time - *decode_start_time) as f32 / 1000.0;
        all_decode_time_taken += duration_decode;
    }

    // Add a very small epsilon to avoid zero / near-zero times
    let prompt_time_taken = prompt_time_taken + EPS;
    let decode_time_taken = (all_decode_time_taken / outputs.len() as f32) + EPS;

    eprintln!("{}", String::from("--- Performance Metrics ---").red());

    eprintln!(
        "{}",
        String::from(format!(
            "⏱️ Prompt tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_prompt_tokens,
            prompt_time_taken,
            total_prompt_tokens as f32 / prompt_time_taken,
        ))
        .yellow()
    );

    eprintln!(
        "{}",
        String::from(format!(
            "⏱️ Decoded tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_decoded_tokens,
            decode_time_taken,
            total_decoded_tokens as f32 / decode_time_taken,
        ))
        .yellow()
    );
}

#[cfg(test)]
mod tests {
    use super::{gemma4_per_layer_cache_config, get_arch_rope, parse_fallback_moe_cfg, ModelType};
    use crate::utils::config::Config;
    use candle_nn::Activation;
    use tokenizers::{models::bpe::BPE, Tokenizer};

    fn empty_tokenizer() -> Tokenizer {
        Tokenizer::new(BPE::default())
    }

    #[test]
    fn gguf_qwen35_arch_maps_to_qwen35_model_type() {
        let tokenizer = empty_tokenizer();
        let (model_type, _, is_rope_i) = get_arch_rope(&tokenizer, "qwen35".to_string()).unwrap();
        assert!(matches!(model_type, ModelType::Qwen3_5));
        assert!(!is_rope_i);
    }

    #[test]
    fn gguf_qwen35moe_arch_maps_to_qwen35_moe_model_type() {
        let tokenizer = empty_tokenizer();
        let (model_type, _, is_rope_i) =
            get_arch_rope(&tokenizer, "qwen35moe".to_string()).unwrap();
        assert!(matches!(model_type, ModelType::Qwen3_5MoE));
        assert!(!is_rope_i);
    }

    #[test]
    fn gguf_qwen3vl_arch_maps_to_multimodal_model_type() {
        let tokenizer = empty_tokenizer();
        let (model_type, _, is_rope_i) = get_arch_rope(&tokenizer, "qwen3vl".to_string()).unwrap();
        assert!(matches!(model_type, ModelType::Qwen3VL));
        assert!(!is_rope_i);
    }

    #[test]
    fn minimax_moe_fallback_uses_intermediate_size() {
        let raw_cfg = serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "intermediate_size": 1536,
            "num_experts_per_tok": 8,
            "num_local_experts": 256,
            "scoring_func": "sigmoid"
        });

        let moe_cfg =
            parse_fallback_moe_cfg("MiniMaxM2ForCausalLM", raw_cfg.to_string().as_bytes())
                .expect("MiniMax fallback MoE config should deserialize");

        assert_eq!(moe_cfg.moe_intermediate_size, 1536);
        assert_eq!(moe_cfg.num_experts, Some(256));
        assert_eq!(moe_cfg.num_experts_per_tok, 8);
        assert_eq!(moe_cfg.scoring_func.as_deref(), Some("sigmoid"));
    }

    fn gemma4_test_config(extra_config_json: serde_json::Value) -> Config {
        Config {
            architectures: Some(vec!["Gemma4ForConditionalGeneration".to_string()]),
            head_dim: Some(512),
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 8192,
            hidden_size: 4096,
            num_hidden_layers: 6,
            max_model_len: None,
            intermediate_size: 14336,
            rms_norm_eps: 1e-6,
            vocab_size: Some(256000),
            rope_theta: None,
            attention_bias: None,
            qkv_bias: None,
            attn_output_gate: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            tie_word_embeddings: None,
            bos_token_id: None,
            eos_token_id: None,
            use_sliding_window: None,
            sliding_window: Some(4096),
            max_window_layers: None,
            partial_rotary_factor: None,
            hidden_act: Activation::GeluPytorchTanh,
            rope_scaling: None,
            quant: None,
            moe_cfg: None,
            kvcache_dtype: crate::utils::config::KvCacheDtype::Auto,
            quantization_config: None,
            is_multi_model: Some(true),
            extra_config_json: Some(extra_config_json.to_string()),
        }
    }

    #[test]
    fn gemma4_per_layer_cache_config_prefers_text_config_swa_head_dim() {
        let config = gemma4_test_config(serde_json::json!({
            "head_dim": 1024,
            "text_config": {
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 4,
                "num_global_key_value_heads": 8
            }
        }));

        let per_layer = gemma4_per_layer_cache_config(&config).unwrap();
        assert_eq!(per_layer.len(), 6);
        assert_eq!(per_layer[0], (4, 256));
        assert_eq!(per_layer[4], (4, 256));
        assert_eq!(per_layer[5], (8, 512));
    }

    #[test]
    fn gemma4_per_layer_cache_config_handles_gguf_extra_config() {
        let config = gemma4_test_config(serde_json::json!({
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ],
            "swa_head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 8
        }));

        let per_layer = gemma4_per_layer_cache_config(&config).unwrap();
        assert_eq!(per_layer[0], (8, 256));
        assert_eq!(per_layer[5], (8, 512));
    }
}

/// Fail fast if `host:port` is already bound, before spending minutes loading
/// model weights.  Prints a user-friendly error and exits the process when the
/// port is occupied.
pub fn ensure_port_free(host: &str, port: u16) {
    let addr = format!("{host}:{port}");
    match std::net::TcpListener::bind(&addr) {
        Ok(_listener) => { /* port is free; drop the listener immediately */ }
        Err(e) => {
            eprintln!(
                "\n❌ Port {port} is already in use ({e}).\n   \
                 Free the port or choose a different one with --port <port>.\n"
            );
            std::process::exit(1);
        }
    }
}
