use crate::models::layers::distributed::{Comm, VocabParallelLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mla_attention::{MlaAttention, MlaConfig};
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
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
    Mlp(MLP),
}

impl MoeOrMlp {
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

pub struct GLM4MoeLiteDecoderLayer {
    self_attn: MlaAttention,
    mlp: MoeOrMlp,
    shared_expert: Option<MLP>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl GLM4MoeLiteDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        mla_cfg: &MlaConfig,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let self_attn = MlaAttention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            mla_cfg,
            config,
            dtype,
        )?;

        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("MoE config is not available!");

        let mlp = if layer_idx >= moe_cfg.first_k_dense_replace.unwrap_or(0) {
            if is_qvar_builder {
                MoeOrMlp::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if let Some(quant_config) = &config.quantization_config {
                if quant_config.quant_method == "fp8" {
                    MoeOrMlp::FusedMoeFp8(FusedMoeFp8::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                        quant_config,
                    )?)
                } else if quant_config.quant_method == "mxfp4" {
                    MoeOrMlp::FusedMoeMxfp4(FusedMoeMxfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else if quant_config.quant_method == "nvfp4" {
                    MoeOrMlp::FusedMoeNvfp4(FusedMoeNvfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else {
                    MoeOrMlp::FusedMoe(FusedMoe::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                }
            } else if config.quant.is_some() {
                MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                MoeOrMlp::FusedMoe(FusedMoe::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            }
        } else {
            let mlp = MLP::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("mlp").clone()
                },
                comm.clone(),
                config.hidden_size,
                config.intermediate_size,
                &config.hidden_act,
                &config.quantization_config,
                &config.quant,
                false,
                dtype,
                "",
            )?;
            MoeOrMlp::Mlp(mlp)
        };

        let is_moe_layer = layer_idx >= moe_cfg.first_k_dense_replace.unwrap_or(0);
        let shared_expert = if is_moe_layer {
            if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                if intermediate_size > 0 {
                    let mlp = MLP::new(
                        if is_qvar_builder {
                            vb.clone()
                        } else {
                            vb.pp("mlp.shared_experts").clone()
                        },
                        comm.clone(),
                        config.hidden_size,
                        intermediate_size * moe_cfg.n_shared_experts.unwrap_or(1),
                        &config.hidden_act,
                        &config.quantization_config,
                        &config.quant,
                        false,
                        dtype,
                        if is_qvar_builder { "_shexp" } else { "" },
                    )?;
                    Some(mlp)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("attn_norm").clone()
            } else {
                vb.pp("input_layernorm").clone()
            },
            DType::F32,
            false,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("ffn_norm").clone()
            } else {
                vb.pp("post_attention_layernorm").clone()
            },
            DType::F32,
            false,
        )?;

        Ok(Self {
            self_attn,
            mlp,
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
        _layer_idx: usize,
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

        let shared_output = if let Some(shared_expert) = &self.shared_expert {
            Some(shared_expert.forward(&xs)?)
        } else {
            None
        };
        let mlp_output = self.mlp.forward(&xs, input_metadata.is_prefill)?;
        let out = if let Some(shared_output) = shared_output {
            (residual + (mlp_output + shared_output)?)?
        } else {
            (residual + mlp_output)?
        };
        Ok(out)
    }
}

pub struct GLM4MoeLiteForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<GLM4MoeLiteDecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl GLM4MoeLiteForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let prefix = "model.";
        let mla_cfg = MlaConfig::from_config(config);

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp("token_embd")
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
        )?;

        let rotary_emb = {
            let mut mla_config = config.clone();
            mla_config.head_dim = Some(mla_cfg.qk_rope_head_dim);
            mla_config.partial_rotary_factor = None;
            Arc::new(ScalingRotaryEmbedding::new(
                if is_qvar_builder || config.higher_precision_required() {
                    DType::F32
                } else {
                    dtype
                },
                &mla_config,
                &vb.device(),
                is_rope_i,
                config.rope_theta,
            )?)
        };

        let reporter = progress_reporter.clone();
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = GLM4MoeLiteDecoderLayer::new(
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        "blk".to_string()
                    } else {
                        format!("{}layers", prefix)
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                &mla_cfg,
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
                vb.pp("output_norm")
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            false,
        )?;

        let tie_word_embeddings = config.tie_word_embeddings;
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp("token_embd")
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else if is_qvar_builder {
                vb.pp("output")
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

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();
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
            for (i, ((k_cache, v_cache), layer)) in
                zip(kv_caches.iter(), self.layers.iter()).enumerate()
            {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                    i,
                )?;
            }
        }

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = if self.is_qvar_builder {
            self.lm_head.forward(&xs)?
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)?
        };
        Ok(logits)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
        )
    }

    pub fn forward_with_deepstack(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        _visual_pos_masks: &Option<Tensor>,
        _deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
