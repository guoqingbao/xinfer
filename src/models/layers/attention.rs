use crate::models::layers::distributed::{
    kv_head_shard, shard, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::others::{rms_norm, rms_norm_sharded, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Result, Tensor, D};
use either::Either;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

enum QkvProjection {
    Separate {
        q_proj: TensorParallelColumnLinear,
        k_proj: TensorParallelColumnLinear,
        v_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
}

pub struct Attention {
    qkv_proj: QkvProjection,
    o_proj: TensorParallelRowLinear,
    q_norm: Option<NormX>,
    k_norm: Option<NormX>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    attn: PagedAttention,
    softcapping: Option<f64>,
    dtype: DType,
    no_per_head_norm: bool,
    full_dim_qk_norm: bool,
    is_qvar_builder: bool,
    qk_l2_norm: bool,
    promote_qk_to_f32: bool,
    v_norm_eps: Option<f64>,
}

impl Attention {
    fn normalize_sharded_2d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            candle_core::bail!("unexpected shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            );
        }

        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name,
            d0,
            d1,
            local,
            global_dim1
        );
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}), expects local {} or global {}",
            name,
            d0,
            local,
            global_dim
        );
    }

    fn load_sharded_bias(
        vb: &VarBuilderX,
        out_dim: usize,
        shard: candle_nn::var_builder::Shard,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let bias = match &vb.0 {
            Either::Left(inner) => {
                inner.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32)
            }
            Either::Right(_) => return Ok(None),
        };
        let Ok(bias) = bias else {
            return Ok(None);
        };
        let bias = Self::normalize_sharded_1d(bias, shard, out_dim, "bias")?;
        if bias.dtype() != dtype {
            Ok(Some(bias.to_dtype(dtype)?))
        } else {
            Ok(Some(bias))
        }
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilderX,
        out_dim: usize,
        in_dim: usize,
        shard: candle_nn::var_builder::Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.has_key("weight_scale") && !vb.has_key("weight_scale_inv") {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8) {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;

        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(scale) => scale,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(scale) => scale,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;

        Ok(Some((weight, weight_scale)))
    }

    fn try_load_packed_qkv(
        vb: &VarBuilderX,
        hidden_size: usize,
        q_out_dim: usize,
        kv_out_dim: usize,
        attention_bias: bool,
        comm: Rc<Comm>,
        kv_shard: candle_nn::var_builder::Shard,
        dtype: DType,
        quant_cfg: &Option<crate::utils::config::QuantConfig>,
        quant: &Option<String>,
        k_eq_v: bool,
    ) -> Result<Option<QkvProjection>> {
        if vb.is_qvar_builder() || quant.is_some() {
            return Ok(None);
        }

        let q_shard = shard(0, comm.rank(), comm.world_size());
        let q_vb = vb.pp("q_proj");
        let k_vb = vb.pp("k_proj");
        let v_vb = if k_eq_v {
            vb.pp("k_proj")
        } else {
            vb.pp("v_proj")
        };

        let is_fp8_quant = quant_cfg
            .as_ref()
            .map(|cfg| cfg.quant_method == "fp8")
            .unwrap_or(false);
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                candle_core::bail!("LnFp8: weight_block_size must be configured for packed qkv");
            };
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }

            let Some((q_weight, q_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &q_vb,
                q_out_dim,
                hidden_size,
                q_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((k_weight, k_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &k_vb,
                kv_out_dim,
                hidden_size,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((v_weight, v_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &v_vb,
                kv_out_dim,
                hidden_size,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };

            let local_q = q_weight.dim(0)?;
            let local_k = k_weight.dim(0)?;
            let local_v = v_weight.dim(0)?;
            let by = block_size[0];
            let q_global_start = q_shard.rank * local_q;
            let k_global_start = q_out_dim + kv_shard.rank * local_k;
            let v_global_start = q_out_dim + kv_out_dim + kv_shard.rank * local_v;
            if q_global_start % by != 0 || k_global_start % by != 0 || v_global_start % by != 0 {
                return Ok(None);
            }

            let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
            let packed_scale = Tensor::cat(&[&q_scale, &k_scale, &v_scale], 0)?;
            let packed_bias = if attention_bias {
                let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
                let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
                let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
                match (q_bias, k_bias, v_bias) {
                    (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                    (None, None, None) => None,
                    _ => return Ok(None),
                }
            } else {
                None
            };

            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;

            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;

            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                packed_bias,
                block_size,
                sm_version,
                vec![local_q, local_k, local_v],
            )?;
            return Ok(Some(QkvProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let q_weight =
            q_vb.get_with_hints_dtype((q_out_dim, hidden_size), "weight", q_shard, dtype)?;
        let k_weight =
            k_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;
        let v_weight =
            v_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;

        let local_q = q_weight.dim(0)?;
        let local_k = k_weight.dim(0)?;
        let local_v = v_weight.dim(0)?;
        let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;

        let packed_bias = if attention_bias {
            let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
            let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
            let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
            match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                (None, None, None) => None,
                _ => return Ok(None),
            }
        } else {
            None
        };

        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            packed_bias,
            vec![local_q, local_k, local_v],
            &None,
        )?;
        Ok(Some(QkvProjection::Packed(merged)))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_options(
            vb,
            comm,
            config,
            attention_scale,
            sliding_window,
            dtype,
            false,
            false,
        )
    }

    pub fn new_with_options(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
        k_eq_v: bool,
        qk_l2_norm: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);
        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
            ("q_norm", "attn_q_norm"),
            ("k_norm", "attn_k_norm"),
        ]
        .iter()
        .cloned()
        .collect();

        let is_qvar_builder = vb.is_qvar_builder();
        let arch = config.architectures.as_ref().unwrap()[0].clone();
        let is_qwen35_or_next = matches!(
            arch.as_str(),
            "Qwen3_5ForCausalLM"
                | "Qwen3_5ForConditionalGeneration"
                | "Qwen3_5MoeForCausalLM"
                | "Qwen3_5MoeForConditionalGeneration"
                | "Qwen3NextForCausalLM"
                | "Qwen3NextForConditionalGeneration"
        );
        let attention_bias = if is_qwen35_or_next {
            config.qkv_bias.or(config.attention_bias).unwrap_or(false)
        } else {
            config.qkv_bias.or(config.attention_bias).unwrap_or(true)
        };
        let attn_output_gate = if is_qwen35_or_next {
            config.attn_output_gate.unwrap_or(true)
        } else {
            config.attn_output_gate.unwrap_or(false)
        };
        let q_out_dim = num_heads * head_dim * if attn_output_gate { 2 } else { 1 };
        let no_per_head_norm_models: Vec<String> = vec![
            "Gemma3ForConditionalGeneration",
            "Gemma3ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "Gemma4ForCausalLM",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3NextForCausalLM",
            "Qwen3NextForConditionalGeneration",
            "MiniMaxM2ForCausalLM",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let is_gemma = arch == "Gemma3ForConditionalGeneration".to_string()
            || arch == "Gemma3ForCausalLM".to_string();
        let is_mlx_nvfp4 = config
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.is_mlx_nvfp4);
        // Qwen3.5/Qwen3Next HF weights use Gemma-style +1 weight semantics for
        // q/k norms. MLX-sanitized NVFP4 checkpoints store the actual weights.
        let qk_norm_add_one = is_gemma || (is_qwen35_or_next && !is_qvar_builder && !is_mlx_nvfp4);

        let world_size = comm.world_size();
        let attention_heads = num_heads / world_size;
        let (kv_heads, kv_shard) = kv_head_shard(num_kv_heads, comm.rank(), world_size)?;

        let qkv_proj = if let Some(packed) = Self::try_load_packed_qkv(
            &vb,
            hidden_size,
            q_out_dim,
            num_kv_heads * head_dim,
            attention_bias,
            comm.clone(),
            kv_shard,
            dtype,
            &config.quantization_config,
            &config.quant,
            k_eq_v,
        )? {
            packed
        } else {
            let q_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                q_out_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map["q_proj"])
                } else {
                    vb.pp("q_proj")
                },
                comm.clone(),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let k_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map["k_proj"])
                } else {
                    vb.pp("k_proj")
                },
                kv_shard,
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let q8_0_qunat = Some("q8_0".to_string());
            let v_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map[if k_eq_v { "k_proj" } else { "v_proj" }])
                } else {
                    vb.pp(if k_eq_v { "k_proj" } else { "v_proj" })
                },
                kv_shard,
                &config.quantization_config,
                if config.quant.is_some() && config.quantization_config.is_none() {
                    &q8_0_qunat
                } else {
                    &None
                },
                dtype,
            )?;
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            }
        };

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                vb.pp("o_proj")
            },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let norm_dtype = if is_qvar_builder
            || config.quant.is_some()
            || config.quantization_config.is_some()
            || config.is_f16_mode
        {
            DType::F32
        } else {
            dtype
        };
        let q_norm_vb = if is_qvar_builder {
            vb.pp(key_map["q_norm"])
        } else {
            vb.pp("q_norm")
        };
        let k_norm_vb = if is_qvar_builder {
            vb.pp(key_map["k_norm"])
        } else {
            vb.pp("k_norm")
        };

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            q_norm_vb.clone(),
            norm_dtype,
            qk_norm_add_one,
        );
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            k_norm_vb.clone(),
            norm_dtype,
            qk_norm_add_one,
        );

        let (q_norm, k_norm, full_dim_qk_norm) = if q_norm.is_ok() && k_norm.is_ok() {
            (Some(q_norm.unwrap()), Some(k_norm.unwrap()), false)
        } else {
            let q_shard = shard(0, comm.rank(), comm.world_size());
            let q_full = rms_norm_sharded(
                num_heads * head_dim,
                config.rms_norm_eps,
                q_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                q_shard,
            );
            let k_full = rms_norm_sharded(
                num_kv_heads * head_dim,
                config.rms_norm_eps,
                k_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                kv_shard,
            );
            if q_full.is_ok() && k_full.is_ok() {
                (Some(q_full.unwrap()), Some(k_full.unwrap()), true)
            } else {
                (None, None, false)
            }
        };

        let is_gemma4 = arch == "Gemma4ForConditionalGeneration" || arch == "Gemma4ForCausalLM";
        let v_norm_eps = if is_gemma4 {
            Some(config.rms_norm_eps)
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn_output_gate,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                attention_scale.unwrap_or(1. / (head_dim as f32).sqrt()),
                Some(kv_heads),
                sliding_window,
                vb.device().clone(),
                None,
                config.kvcache_dtype.is_fp8_keys(),
            )?,
            softcapping: config.attn_logit_softcapping,
            dtype,
            no_per_head_norm: no_per_head_norm_models.contains(&arch),
            full_dim_qk_norm,
            is_qvar_builder,
            qk_l2_norm,
            promote_qk_to_f32: is_qvar_builder || config.higher_precision_required() || qk_l2_norm,
            v_norm_eps,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_ext(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            None,
        )
    }

    pub fn forward_ext(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        q_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (q_raw, k, v) = match &self.qkv_proj {
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
            ),
            QkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                if qkv.len() != 3 {
                    candle_core::bail!(
                        "Expected 3 outputs from packed qkv projection, got {}",
                        qkv.len()
                    );
                }
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
            }
        };

        let local_q_dim = self.num_heads * self.head_dim;
        let (q_linear, gate) = if self.attn_output_gate {
            let q_dim = q_raw.dim(1)?;
            if q_dim != local_q_dim * 2 {
                candle_core::bail!(
                    "q_proj output dim mismatch for gated attention, expected {}, got {}",
                    local_q_dim * 2,
                    q_dim
                );
            }
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2, 0, self.head_dim)?;
            let gate = q_gate.narrow(2, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let q = q_linear.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = if self.promote_qk_to_f32 && q.dtype() != DType::F32 {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
        } else {
            (q, k)
        };

        let (q, k) = if self.q_norm.is_some() && self.k_norm.is_some() {
            if self.full_dim_qk_norm {
                let q_2d = q.reshape((seq_len, self.num_heads * self.head_dim))?;
                let k_2d = k.reshape((seq_len, self.num_kv_heads * self.head_dim))?;
                let q_2d = self.q_norm.as_ref().unwrap().forward(&q_2d)?;
                let k_2d = self.k_norm.as_ref().unwrap().forward(&k_2d)?;
                let q = q_2d.reshape((seq_len, self.num_heads, self.head_dim))?;
                let k = k_2d.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
                (q, k)
            } else if self.no_per_head_norm {
                let q = self.q_norm.as_ref().unwrap().forward(&q)?;
                let k = self.k_norm.as_ref().unwrap().forward(&k)?;
                (q, k)
            } else {
                let q_flat = q.flatten(0, 1)?;
                let k_flat = k.flatten(0, 1)?;
                let q_flat = self.q_norm.as_ref().unwrap().forward(&q_flat)?;
                let k_flat = self.k_norm.as_ref().unwrap().forward(&k_flat)?;
                let q = q_flat.reshape((seq_len, self.num_heads, self.head_dim))?;
                let k = k_flat.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
                (q, k)
            }
        } else {
            (q, k)
        };

        // Apply rotary embeddings
        let (q, k) = if let Some(rotary_emb) = &rotary_emb {
            match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q, k),
            }
        } else {
            (q, k)
        };

        let (q, k) = if self.qk_l2_norm {
            let q_rms = (q.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?;
            let k_rms = (k.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?;
            let q = q.broadcast_div(&q_rms)?;
            let k = k.broadcast_div(&k_rms)?;
            (q, k)
        } else {
            (q, k)
        };

        let (mut q, k) = if q.dtype() != self.dtype {
            let q = q.to_dtype(self.dtype)?;
            let k = k.to_dtype(self.dtype)?;
            (q, k)
        } else {
            (q, k)
        };

        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v
        };

        let v = if let Some(eps) = self.v_norm_eps {
            let orig_dtype = v.dtype();
            let v_f32 = v.to_dtype(DType::F32)?;
            let mean_sq = v_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rms = (mean_sq + eps)?.sqrt()?;
            v_f32.broadcast_div(&rms)?.to_dtype(orig_dtype)?
        } else {
            v
        };

        if let Some(rotary_emb) = &rotary_emb {
            if let (Some(original_max_position_embeddings), Some(llama_4_scaling_beta)) = (
                rotary_emb.get_original_max_position_embeddings(),
                rotary_emb.get_llama_4_scaling_beta(),
            ) {
                use crate::utils::get_llama4_attn_scale;
                let scale = get_llama4_attn_scale(
                    &positions,
                    llama_4_scaling_beta,
                    original_max_position_embeddings as f64,
                )?
                .to_dtype(q.dtype())?;
                let scale = scale.squeeze(0)?.squeeze(0)?.reshape((seq_len, 1, 1))?;
                q = q.broadcast_mul(&scale)?;
            }
        }

        if let Some(scale) = q_scale {
            let scale = scale.reshape((seq_len, 1, 1))?;
            q = q
                .to_dtype(DType::F32)?
                .broadcast_mul(&scale.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
        }

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
                self.softcapping,
            )?
            .reshape((seq_len, ()))?;

        let y = if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            y.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?
        } else {
            y
        };

        let y = if self.is_qvar_builder {
            y
        } else {
            y.to_dtype(xs.dtype())?
        };
        self.o_proj.forward(&y)
    }
}

pub struct NaiveAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
    softcapping: Option<f64>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilderX,
        num_heads: usize,
        hidden_size: usize,
        head_dim: usize,
        softcapping: Option<f64>,
        dtype: DType,
        key_mappings: HashMap<String, String>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let q_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["q_proj"])
            } else {
                vb.pp("q_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let k_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["k_proj"])
            } else {
                vb.pp("k_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let v_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["v_proj"])
            } else {
                vb.pp("v_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let o_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                if key_mappings.contains_key("o_proj") {
                    vb.pp(&key_mappings["o_proj"])
                } else {
                    vb.pp("o_proj")
                }
            },
            &None,
            &None,
            dtype,
        )?;

        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
            softcapping,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let shape = (b, seq_len, self.num_heads, self.head_dim);
        let q_packed = q
            .reshape(((), self.num_heads, self.head_dim))?
            .contiguous()?;
        let k_packed = k
            .reshape(((), self.num_heads, self.head_dim))?
            .contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let (q_packed, k_packed) = if let Some(positions) = positions {
            match emb.apply_rotary_emb_qkv(&q_packed, &k_packed, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q_packed, k_packed), // In-place operation, keep originals
            }
        } else {
            (q_packed, k_packed)
        };

        let q = q_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let chunk_size = 1024;
        let mut attn_chunks = Vec::new();
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        for c in 0..num_chunks {
            let offset = c * chunk_size;
            let len = chunk_size.min(seq_len - offset);
            //chunk at query is correct for the following
            let q_chunk = q.narrow(2, offset, len)?.contiguous()?;
            let mut att = (q_chunk.matmul(&k.t()?)? * f64::from(self.scale))?;

            if let Some(sc) = self.softcapping {
                att = ((att / sc)?.tanh()? * sc)?;
            }
            if let Some(mask) = &mask {
                //mask needs to be chunked
                let q_chunk_mask = mask.narrow(2, offset, len)?; // shape: [1, 1, chunk_len, K_len]
                att = att.broadcast_add(&q_chunk_mask)?;
            }
            att = candle_nn::ops::softmax_last_dim(&att.to_dtype(candle_core::DType::F32)?)?
                .to_dtype(att.dtype())?;

            let att_chunk = att.matmul(&v)?;
            attn_chunks.push(att_chunk);
        }

        let att = Tensor::cat(&attn_chunks, 2)?
            .contiguous()?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((b, seq_len, ()))?;
        self.o_proj.forward(&att)
    }
}
