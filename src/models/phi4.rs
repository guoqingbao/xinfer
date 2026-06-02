// src/models/phi4.rs
// This implementation is adapted from Phi-3, using the local Candle layers.
use crate::models::layers::distributed::{
    kv_head_shard, shard, Comm, MergedParallelColumnLinear, TensorParallelRowLinear,
    VocabParallelLinear,
};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, RotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::{Config, RopeScalingValue};
use crate::utils::progress::ProgressLike;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Phi4RotaryEmbedding {
    normal_emb: RotaryEmbedding,
    long_emb: Option<RotaryEmbedding>,
    original_max_position_embeddings: Option<usize>,
}

impl Phi4RotaryEmbedding {
    fn effective_max_seq_len(cfg: &Config) -> usize {
        let Some(rope_scaling) = &cfg.rope_scaling else {
            return cfg.max_position_embeddings;
        };

        let factor = rope_scaling.get("factor").and_then(|v| v.as_f64());
        let original_max_position_embeddings = rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_f64());

        match (factor, original_max_position_embeddings) {
            (Some(factor), Some(original_max_position_embeddings)) if factor > 1.0 => {
                std::cmp::max(
                    cfg.max_position_embeddings,
                    (original_max_position_embeddings * factor).round() as usize,
                )
            }
            _ => cfg.max_position_embeddings,
        }
    }

    fn rope_scaling_array(
        value: &RopeScalingValue,
        expected_len: usize,
        name: &str,
    ) -> Result<Vec<f64>> {
        match value {
            RopeScalingValue::NumberArray(v) => {
                if v.len() != expected_len {
                    candle_core::bail!(
                        "{name} length mismatch: expected {expected_len}, got {}",
                        v.len()
                    );
                }
                Ok(v.clone())
            }
            RopeScalingValue::Number(v) => Ok(vec![*v; expected_len]),
            _ => candle_core::bail!("{name} must be a number array"),
        }
    }

    fn new(dtype: DType, cfg: &Config, is_rope_i: bool, dev: &Device) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);
        let max_seq_len = Self::effective_max_seq_len(cfg);
        let rope_theta = cfg.rope_theta.unwrap_or(10000.0);
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        if let Some(rope_scaling) = &cfg.rope_scaling {
            let original_max_position_embeddings = rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_f64())
                .unwrap_or(cfg.max_position_embeddings as f64);
            let rope_type = rope_scaling
                .get("type")
                .or_else(|| rope_scaling.get("rope_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("default");

            let short_factor = rope_scaling
                .get("short_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing short_factor"))?;
            let long_factor = rope_scaling
                .get("long_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing long_factor"))?;
            let short_factor =
                Self::rope_scaling_array(short_factor, inv_freq_len, "short_factor")?;
            let long_factor = Self::rope_scaling_array(long_factor, inv_freq_len, "long_factor")?;

            // Compute scaling factor (same for both short and long embeddings)
            let scale = max_seq_len as f64 / original_max_position_embeddings;
            let scaling_factor = if scale <= 1.0 {
                1.0
            } else {
                match rope_type {
                    "su" | "longrope" => {
                        (1.0 + scale.ln() / original_max_position_embeddings.ln()).sqrt()
                    }
                    "yarn" => 0.1 * scale.ln() + 1.0,
                    _ => 1.0,
                }
            };

            let inv_freq_long = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (long_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64))) as f32
                })
                .collect::<Vec<_>>();
            let inv_freq_short = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (short_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64)))
                        as f32
                })
                .collect::<Vec<_>>();

            let inv_freq_long =
                Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
            let inv_freq_short =
                Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;

            let freqs_long = t.matmul(&inv_freq_long)?;
            let long_sin = (freqs_long.sin()? * scaling_factor)?;
            let long_cos = (freqs_long.cos()? * scaling_factor)?;

            let freqs_short = t.matmul(&inv_freq_short)?;
            let short_sin = (freqs_short.sin()? * scaling_factor)?;
            let short_cos = (freqs_short.cos()? * scaling_factor)?;

            let normal_emb = RotaryEmbedding {
                sin: short_sin.to_dtype(dtype)?,
                cos: short_cos.to_dtype(dtype)?,
                is_rope_i,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
                original_max_position_embeddings: Some(original_max_position_embeddings as usize),
                llama_4_scaling_beta: None,
            };

            let long_emb = RotaryEmbedding {
                sin: long_sin.to_dtype(dtype)?,
                cos: long_cos.to_dtype(dtype)?,
                is_rope_i,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
                original_max_position_embeddings: Some(original_max_position_embeddings as usize),
                llama_4_scaling_beta: None,
            };

            return Ok(Self {
                normal_emb,
                long_emb: Some(long_emb),
                original_max_position_embeddings: Some(
                    original_max_position_embeddings.round() as usize
                ),
            });
        }

        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs = t.matmul(&inv_freq)?;

        let normal_emb = RotaryEmbedding {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            is_rope_i,
            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                Some(rotary_dim)
            } else {
                None
            },
            original_max_position_embeddings: None,
            llama_4_scaling_beta: None,
        };
        Ok(Self {
            normal_emb,
            long_emb: None,
            original_max_position_embeddings: None,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if let (Some(long_emb), Some(original_max_position_embeddings)) =
            (&self.long_emb, self.original_max_position_embeddings)
        {
            let max_position = input_positions
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .max()
                .unwrap_or(0)
                + 1;
            if max_position >= original_max_position_embeddings as i64 {
                long_emb.apply_rotary_emb_qkv(q, k, input_positions)
            } else {
                self.normal_emb.apply_rotary_emb_qkv(q, k, input_positions)
            }
        } else {
            self.normal_emb.apply_rotary_emb_qkv(q, k, input_positions)
        }
    }
}

struct Phi4Attention {
    qkv_proj: MergedParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Phi4RotaryEmbedding>,
    attn: PagedAttention,
    is_quantized: bool,
    dtype: DType,
}

impl Phi4Attention {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let is_qvar_builder = vb.is_qvar_builder();

        let world_size = comm.world_size();
        let attention_heads = num_heads / world_size;
        let (kv_heads, kv_shard) = kv_head_shard(num_kv_heads, comm.rank(), world_size)?;
        let kv_shard_rank = kv_shard.rank;
        let kv_shard_world_size = kv_shard.world_size;

        let qkv_proj = MergedParallelColumnLinear::load_merged_chunks(
            cfg.hidden_size,
            op_size,
            0,
            vec![
                num_heads * head_dim,
                num_kv_heads * head_dim,
                num_kv_heads * head_dim,
            ],
            Some(vec![
                shard(0, comm.rank(), world_size),
                shard(0, kv_shard_rank, kv_shard_world_size),
                shard(0, kv_shard_rank, kv_shard_world_size),
            ]),
            if is_qvar_builder {
                vb.pp("attn_qkv")
            } else {
                vb.pp("qkv_proj")
            },
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;
        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            cfg.hidden_size,
            if is_qvar_builder {
                vb.pp("attn_output")
            } else {
                vb.pp("o_proj")
            },
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(kv_heads),
                cfg.sliding_window,
                vb.device().clone(),
                None,
                cfg.kvcache_dtype.is_fp8_keys(),
            )?,
            is_quantized: is_qvar_builder || cfg.quant.is_some(),
            dtype,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv[0].reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = qkv[1].reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = qkv[2].reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = match self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, input_positions)?
        {
            Some((q_new, k_new)) => (q_new, k_new),
            None => (q, k), // In-place operation, keep originals
        };

        let (q, k, v) = if self.is_quantized {
            (
                q.to_dtype(self.dtype)?,
                k.to_dtype(self.dtype)?,
                v.to_dtype(self.dtype)?,
            )
        } else {
            (q, k, v)
        };

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((seq_len, ()))?;

        self.o_proj.forward(&y)?.to_dtype(xs.dtype())
    }
}

struct Phi4DecoderLayer {
    self_attn: Phi4Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Phi4DecoderLayer {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let self_attn = Phi4Attention::new(
            rotary_emb,
            cfg,
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn")
            },
            comm.clone(),
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp")
            },
            comm.clone(),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.hidden_act,
            &cfg.quantization_config,
            &cfg.quant,
            true,
            dtype,
            "",
        )?;
        let key_map: HashMap<&str, &str> = [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
        .iter()
        .cloned()
        .collect();
        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["input_layernorm"])
            } else {
                vb.pp("input_layernorm")
            },
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["post_attention_layernorm"])
            } else {
                vb.pp("post_attention_layernorm")
            },
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

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

pub struct Phi4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Phi4DecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl Phi4ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("model.embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
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
                vb.pp(key_map["model.embed_tokens"])
            } else {
                vb.pp("model.embed_tokens")
            },
            dtype,
        )?;
        let rotary_emb = Arc::new(Phi4RotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            config,
            is_rope_i,
            &vb.device(),
        )?);
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = Phi4DecoderLayer::new(
                rotary_emb.clone(),
                config,
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        key_map["model.layers"]
                    } else {
                        "model.layers"
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }
        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["model.norm"])
            } else {
                vb.pp("model.norm")
            },
            DType::F32,
            false,
        )?;
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(key_map["model.embed_tokens"])
                } else {
                    vb.pp("model.embed_tokens")
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("lm_head")
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
        let seqlens = if input_metadata.seqlens.is_some() {
            input_metadata.seqlens.as_ref().unwrap().clone()
        } else {
            Vec::new()
        };
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
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
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
