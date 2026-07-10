use crate::models::layers::activation::GatedActivation;
use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, VocabParallelLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{
    FusedMoe, FusedMoeFp8, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4,
};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct MiniMax3Config {
    dense_intermediate_size: usize,
    shared_intermediate_size: usize,
    first_k_dense_replace: usize,
    use_gemma_norm: bool,
}

impl MiniMax3Config {
    fn from_config(config: &Config) -> Result<Self> {
        let raw = config
            .extra_config_json
            .as_deref()
            .ok_or_else(|| candle_core::Error::Msg("MiniMax M3 raw config is missing".into()))?;
        let root: serde_json::Value =
            serde_json::from_str(raw).map_err(candle_core::Error::wrap)?;
        let cfg = root.get("text_config").unwrap_or(&root);
        let get_usize = |name: &str, default: usize| {
            cfg.get(name)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Ok(Self {
            dense_intermediate_size: get_usize("dense_intermediate_size", config.intermediate_size),
            shared_intermediate_size: get_usize(
                "shared_intermediate_size",
                get_usize("intermediate_size", config.intermediate_size),
            ),
            first_k_dense_replace: get_usize("first_k_dense_replace", 3),
            use_gemma_norm: cfg
                .get("use_gemma_norm")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }
}

fn resolve_input_seqlens(input_metadata: &InputMetadata) -> Result<Vec<u32>> {
    if let Some(seqlens) = input_metadata.seqlens.as_ref() {
        Ok(seqlens.clone())
    } else if let Some(cu_seqlens) = input_metadata.cu_seqlens_q.as_ref() {
        Ok(cu_seqlens.to_vec1::<u32>()?[1..].to_vec())
    } else {
        Ok(Vec::new())
    }
}

enum MoeVariant {
    Mlp(MLP),
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
}

impl MoeVariant {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct MiniMax3DecoderLayer {
    self_attn: Attention,
    moe: MoeVariant,
    shared_expert: Option<MLP>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl MiniMax3DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        dtype: DType,
        _layer_idx: usize,
    ) -> Result<Self> {
        let m3_config = MiniMax3Config::from_config(config)?;
        let activation = GatedActivation::from_model_config(
            config.hidden_act.clone(),
            config.extra_config_json.as_deref(),
        );
        let is_qvar_builder = vb.is_qvar_builder();
        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            config,
            None,
            config.sliding_window,
            dtype,
        )?;

        let moe_vb = vb.pp("block_sparse_moe");

        let moe = if _layer_idx < m3_config.first_k_dense_replace {
            MoeVariant::Mlp(MLP::new_with_gated_activation(
                vb.pp("mlp"),
                comm.clone(),
                config.hidden_size,
                m3_config.dense_intermediate_size,
                &config.quantization_config,
                &config.quant,
                dtype,
                activation.clone(),
            )?)
        } else if is_qvar_builder {
            MoeVariant::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
        } else if let Some(quant_config) = &config.quantization_config {
            if quant_config.quant_method == "fp8" {
                MoeVariant::FusedMoeFp8(FusedMoeFp8::new_with_gate(
                    config,
                    moe_vb.pp("gate"),
                    moe_vb.pp("experts"),
                    &moe_vb,
                    comm.clone(),
                    dtype,
                    quant_config,
                )?)
            } else if quant_config.quant_method == "mxfp4" {
                MoeVariant::FusedMoeMxfp4(FusedMoeMxfp4::new_with_gate(
                    config,
                    moe_vb.pp("gate"),
                    moe_vb.pp("experts"),
                    &moe_vb,
                    comm.clone(),
                    dtype,
                )?)
            } else if quant_config.quant_method == "nvfp4" {
                MoeVariant::FusedMoeNvfp4(FusedMoeNvfp4::new_with_gate_and_bias(
                    config,
                    moe_vb.pp("gate"),
                    moe_vb.pp("experts"),
                    Some(&moe_vb),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                panic!("Unsupported quantization for MiniMax (use unquantized, gguf, fp8, mxfp4 or nvfp4)!");
            }
        } else if config.quant.is_some() {
            MoeVariant::FusedMoeISQ(FusedMoeISQ::new_with_gate(
                config,
                moe_vb.pp("gate"),
                moe_vb.pp("experts"),
                &moe_vb,
                comm.clone(),
                dtype,
            )?)
        } else {
            MoeVariant::FusedMoe(FusedMoe::new_with_gate(
                config,
                moe_vb.pp("gate"),
                moe_vb.pp("experts"),
                &moe_vb,
                comm.clone(),
                dtype,
            )?)
        };

        let shared_expert = if _layer_idx >= m3_config.first_k_dense_replace {
            Some(MLP::new_with_gated_activation(
                moe_vb.pp("shared_experts"),
                comm.clone(),
                config.hidden_size,
                m3_config.shared_intermediate_size,
                &config.quantization_config,
                &config.quant,
                dtype,
                activation,
            )?)
        } else {
            None
        };

        let key_map: HashMap<&str, &str> = [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
        .iter()
        .cloned()
        .collect();

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["input_layernorm"]).clone()
            } else {
                vb.pp("input_layernorm").clone()
            },
            DType::F32,
            m3_config.use_gemma_norm,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["post_attention_layernorm"]).clone()
            } else {
                vb.pp("post_attention_layernorm").clone()
            },
            DType::F32,
            m3_config.use_gemma_norm,
        )?;

        Ok(Self {
            self_attn,
            moe,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();
        let attn_output = self.self_attn.forward(
            &xs,
            &Some(rope),
            attention_mask,
            positions,
            cache,
            input_metadata,
        )?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mut mlp_output = self.moe.forward(&xs, input_metadata.is_prefill)?;
        if let Some(shared_expert) = &self.shared_expert {
            mlp_output = (mlp_output + shared_expert.forward(&xs)?)?;
        }
        residual + mlp_output
    }
}

pub struct MiniMax3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMax3DecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

/// Dispatches the shared MiniMax ModelType to the legacy M2 implementation
/// or the isolated M3 implementation without changing the M2 model module.
pub enum MiniMaxModel {
    M2(crate::models::minimax::MiniMaxForCausalLM),
    M3(MiniMax3ForCausalLM),
}

impl MiniMaxModel {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let is_m3 = config
            .architectures
            .as_ref()
            .and_then(|a| a.first())
            .is_some_and(|a| a.starts_with("MiniMaxM3"));
        if is_m3 {
            Ok(Self::M3(MiniMax3ForCausalLM::new(
                vb,
                comm,
                config,
                dtype,
                is_rope_i,
                device,
                progress_reporter,
            )?))
        } else {
            Ok(Self::M2(crate::models::minimax::MiniMaxForCausalLM::new(
                vb,
                comm,
                config,
                dtype,
                is_rope_i,
                device,
                progress_reporter,
            )?))
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        match self {
            Self::M2(m) => m.forward(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                embeded_inputs,
            ),
            Self::M3(m) => m.forward(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                embeded_inputs,
            ),
        }
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        match self {
            Self::M2(m) => m.forward_embedding(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                embeded_inputs,
            ),
            Self::M3(m) => m.forward_embedding(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                embeded_inputs,
            ),
        }
    }

    pub fn get_vocab_size(&self) -> usize {
        match self {
            Self::M2(m) => m.get_vocab_size(),
            Self::M3(m) => m.get_vocab_size(),
        }
    }
}

impl MiniMax3ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let m3_config = MiniMax3Config::from_config(config)?;
        let prefix = "language_model.model.".to_string();
        let gguf_prefix = "".to_string();
        let key_map: HashMap<&str, &str> = [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("norm", "output_norm"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let reporter = progress_reporter.clone();

        let is_qvar_builder = vb.is_qvar_builder();

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
        )?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            config,
            &vb.device(),
            is_rope_i,
            config.rope_theta,
        )?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = MiniMax3DecoderLayer::new(
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        format!("{}{}", gguf_prefix, key_map["layers"])
                    } else {
                        format!("{}layers", prefix)
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                dtype,
                i,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["norm"]))
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            m3_config.use_gemma_norm,
        )?;

        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("language_model.lm_head")
                }
            },
            comm.clone(),
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)
        } else {
            Ok(xs)
        }
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = resolve_input_seqlens(input_metadata)?;

        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), (_i, layer)) in
                zip(kv_caches.iter(), self.layers.iter().enumerate())
            {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
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
        } else if self.is_qvar_builder {
            self.lm_head.forward(&xs)
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            true,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
