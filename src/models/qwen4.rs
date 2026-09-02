// src/models/qwen4.rs
// Qwen4-Exp MoE variant with hyper-connections, QSA attention, and GatedDeltaNet layers
use crate::models::layers::deltanet::GatedDeltaNet;
use crate::models::layers::distributed::{Comm, VocabParallelLinear};
use crate::models::layers::linear::LinearX as Linear;
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{
    FusedMoe, FusedMoeFp8, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4, FusedMoeWNA16,
};
use crate::models::layers::others::embedding;
use crate::models::layers::qwen4::{Qwen4HyperConnection, Qwen4QSAAttention};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use crate::utils::{
    resolve_qwen3_hybrid_config, resolve_qwen4_config, resolve_qwen4_layer_types, InputMetadataExt,
};
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// =============================================================================
// MoE dispatch (reused from qwen3_5_moe pattern)
// =============================================================================

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
    FusedMoeWNA16(FusedMoeWNA16),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeWNA16(m) => m.forward(xs, is_prefill),
        }
    }
}

// =============================================================================
// Attention type: QSA sparse attention or GatedDeltaNet
// =============================================================================

enum Qwen4AttnType {
    QSAAttention(Qwen4QSAAttention),
    LinearAttention(GatedDeltaNet),
}

// =============================================================================
// Decoder layer: hyper-connection + hybrid attention + MoE
// =============================================================================

pub struct Qwen4DecoderLayer {
    attn: Qwen4AttnType,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<MLP>,
    attn_hyper_connection: Qwen4HyperConnection,
    mlp_hyper_connection: Qwen4HyperConnection,
}

impl Qwen4DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        qwen4: &crate::utils::Qwen4Config,
        layer_type: &str,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let hc_count = qwen4.hc_count;
        let hidden_size = config.hidden_size;
        let hc_lowrank = qwen4.hc_lowrank;

        let attn_hyper_connection = Qwen4HyperConnection::new(
            vb.clone(),
            comm.clone(),
            hc_count,
            hidden_size,
            hc_lowrank,
            config.rms_norm_eps,
            dtype,
            true,
            "attn_hyper_connection",
        )?;
        let mlp_hyper_connection = Qwen4HyperConnection::new(
            vb.clone(),
            comm.clone(),
            hc_count,
            hidden_size,
            hc_lowrank,
            config.rms_norm_eps,
            dtype,
            true,
            "mlp_hyper_connection",
        )?;

        let cos_table = rotary_emb.0.cos.clone();
        let sin_table = rotary_emb.0.sin.clone();

        let attn = if layer_type == "qwen_sparse_attention" {
            Qwen4AttnType::QSAAttention(Qwen4QSAAttention::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("self_attn").clone()
                },
                comm.clone(),
                config,
                qwen4,
                dtype,
                cos_table,
                sin_table,
            )?)
        } else {
            Qwen4AttnType::LinearAttention(GatedDeltaNet::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("linear_attn").clone()
                },
                comm.clone(),
                config,
                gdn_layer_idx,
                dtype,
            )?)
        };

        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("MoE config is not available!");
        let mlp = if is_qvar_builder {
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
            } else if quant_config.is_compressed_tensors || quant_config.quant_method == "gptq" {
                MoeOrMlp::FusedMoeWNA16(FusedMoeWNA16::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                    quant_config,
                )?)
            } else {
                panic!("Unsupported quant method for MoE (use unquantized, gguf, fp8, mxfp4 or nvfp4)!");
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
        };

        let (shared_gate, shared_expert) = if let Some(intermediate_size) =
            moe_cfg.shared_expert_intermediate_size
        {
            if intermediate_size > 0 {
                let ws = match &vb.0 {
                    Either::Left(vb) => vb
                        .pp("mlp.shared_expert_gate")
                        .get((1, config.hidden_size), "weight")?,
                    Either::Right(vb) => {
                        let ws = vb
                            .pp("ffn_gate_inp_shexp")
                            .get((config.hidden_size,), "weight")?;
                        ws.dequantize(&vb.device())?
                            .reshape((1, config.hidden_size))?
                    }
                }
                .to_dtype(if is_qvar_builder || config.quant.is_some() {
                    DType::F32
                } else {
                    dtype
                })?;
                let shared_gate = Linear::new(ws, None, &None)?;
                let mlp = MLP::new(
                    if is_qvar_builder {
                        vb.clone()
                    } else {
                        vb.pp("mlp.shared_expert").clone()
                    },
                    comm.clone(),
                    config.hidden_size,
                    intermediate_size,
                    &config.hidden_act,
                    &config.quantization_config,
                    &config.quant,
                    false,
                    dtype,
                    if is_qvar_builder { "_shexp" } else { "" },
                )?;
                (Some(shared_gate), Some(mlp))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(Self {
            attn,
            mlp,
            shared_gate,
            shared_expert,
            attn_hyper_connection,
            mlp_hyper_connection,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        mamba_cache: &mut MambaCache,
        seq_slots: &Tensor,
        rotary_emb: &Arc<dyn ApplyRotaryEmbedding>,
    ) -> Result<Tensor> {
        let (attn_input, attn_hc_state) = self.attn_hyper_connection.read(xs)?;
        let attn_output = match &self.attn {
            Qwen4AttnType::QSAAttention(qsa) => qsa.forward(
                &attn_input,
                rotary_emb,
                attention_mask,
                positions,
                cache,
                input_metadata,
            )?,
            Qwen4AttnType::LinearAttention(gdn) => {
                gdn.forward(&attn_input, mamba_cache, input_metadata, seq_slots)?
            }
        };
        let xs = self
            .attn_hyper_connection
            .write(&attn_output, &attn_hc_state)?;

        let (mlp_input, mlp_hc_state) = self.mlp_hyper_connection.read(&xs)?;

        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&mlp_input)?)?;
                let shared_output = shared_expert.forward(&mlp_input)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };

        let mlp_output = self
            .mlp
            .forward(&mlp_input, input_metadata.moe_is_prefill())?;
        let block_out = if let Some(shared_output) = shared_output {
            (mlp_output + shared_output)?
        } else {
            mlp_output
        };

        self.mlp_hyper_connection.write(&block_out, &mlp_hc_state)
    }

    pub fn is_qsa_attention(&self) -> bool {
        matches!(&self.attn, Qwen4AttnType::QSAAttention(_))
    }
}

// =============================================================================
// Qwen4 MoE Causal LM
// =============================================================================

pub struct Qwen4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen4DecoderLayer>,
    hyper_connection_mixer: Qwen4HyperConnection,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    lm_head: VocabParallelLinear,
    mamba_cache: RwLock<MambaCache>,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
    hc_count: usize,
    hc_hidden_size: usize,
    /// Pre-allocated hidden state buffer for MTP speculative decoding (graph-safe).
    pub mtp_hidden_buffer: std::sync::Mutex<Option<Tensor>>,
    pub dflash_verify_hidden_buffers: std::sync::Mutex<Option<Vec<Tensor>>>,
    pub dflash_target_layer_ids: std::sync::Mutex<Vec<usize>>,
}

impl Qwen4ForCausalLM {
    fn expand_hc_hidden(&self, xs: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden) = xs.dims2()?;
        if hidden != self.config.hidden_size {
            candle_core::bail!(
                "Qwen4 expected hidden_size {}, got {}",
                self.config.hidden_size,
                hidden
            );
        }
        let xs = xs.reshape((seq_len, 1, hidden))?;
        xs.broadcast_as((seq_len, self.hc_count, hidden))?
            .flatten(1, 2)
    }

    fn collapse_hc_hidden(&self, xs: &Tensor) -> Result<Tensor> {
        let (collapsed, _) = self.hyper_connection_mixer.read(xs)?;
        Ok(collapsed)
    }

    fn resolve_seq_slots(
        &self,
        input_metadata: &InputMetadata,
        token_count: usize,
    ) -> Result<Tensor> {
        if let Some(slot_mapping) = &input_metadata.mamba_slot_mapping {
            if slot_mapping.dtype() != DType::I64 {
                candle_core::bail!(
                    "Qwen4 MoE expects mamba_slot_mapping dtype I64, got {:?}",
                    slot_mapping.dtype()
                )
            }
            let slot_count = slot_mapping.dim(0)?;
            if slot_count == 0 {
                candle_core::bail!("Qwen4 MoE received empty mamba_slot_mapping")
            }
            if !input_metadata.is_prefill && slot_count != token_count {
                candle_core::bail!(
                    "Qwen4 MoE decode mamba_slot_mapping length mismatch: slots={} tokens={}",
                    slot_count,
                    token_count
                )
            }
            return Ok(slot_mapping.clone());
        }

        let sequence_ids = input_metadata
            .sequence_ids
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Qwen4 MoE requires sequence_ids".into()))?;
        if sequence_ids.is_empty() {
            candle_core::bail!("Qwen4 MoE received empty sequence_ids");
        }

        let slots = if input_metadata.is_prefill {
            self.ensure_mamba_slots_for_sequences(sequence_ids)?
        } else {
            self.get_mamba_slots_for_sequences(sequence_ids)?
        };
        if slots.is_empty() {
            candle_core::bail!("Qwen4 MoE resolved empty mamba slots from sequence_ids");
        }
        if !input_metadata.is_prefill && slots.len() != token_count {
            candle_core::bail!(
                "Qwen4 MoE decode mamba slot count mismatch: slots={} tokens={}",
                slots.len(),
                token_count
            );
        }

        let slots_i64 = slots.into_iter().map(|s| s as i64).collect::<Vec<_>>();
        let len = slots_i64.len();
        Tensor::from_vec(slots_i64, (len,), &self.device)
    }

    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        Self::new_with_prefix(
            vb,
            comm,
            config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
            None,
        )
    }

    pub fn new_with_prefix(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
        prefix: Option<String>,
    ) -> Result<Self> {
        let has_prefix = prefix.is_some();
        let mut prefix = prefix.unwrap_or("model.".to_string());
        let gguf_prefix = if has_prefix {
            prefix.clone()
        } else {
            "".to_string()
        };
        let key_map: HashMap<&str, &str> = [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();
        let reporter = progress_reporter.clone();
        let tie_word_embeddings = if !is_qvar_builder
            && vb.has_key("embed_tokens.weight")
            && !vb.has_key(&format!("{}embed_tokens.weight", prefix))
        {
            crate::log_error!("This model does not support decoding!");
            prefix.clear();
            Some(true)
        } else {
            config.tie_word_embeddings
        };

        let mut model_config = config.clone();
        if model_config
            .architectures
            .as_ref()
            .is_none_or(|archs| !archs.iter().any(|a| crate::utils::is_qwen4_arch_name(a)))
        {
            model_config.architectures = Some(vec!["Qwen4ExpForCausalLM".to_string()]);
        }

        let qwen4 = resolve_qwen4_config(&model_config)?;
        let hc_count = qwen4.hc_count;
        let hc_hidden_size = hc_count * model_config.hidden_size;

        let (embed_tokens, vocab_size) = embedding(
            model_config.vocab_size,
            model_config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || model_config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            &model_config,
            &vb.device(),
            is_rope_i,
            model_config.rope_theta,
        )?);

        let layer_types = resolve_qwen4_layer_types(&model_config);

        let mut layers = Vec::new();
        let mut gdn_layer_idx = 0usize;

        for i in 0..model_config.num_hidden_layers {
            let layer_type = &layer_types[i];
            let current_gdn_idx = if layer_type == "linear_attention" {
                let idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                idx
            } else {
                0
            };

            let layer = Qwen4DecoderLayer::new(
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
                &model_config,
                &qwen4,
                layer_type,
                current_gdn_idx,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let num_gdn_layers = gdn_layer_idx;

        let mixer_prefix = if is_qvar_builder {
            "hyper_connection_mixer".to_string()
        } else {
            format!("{}hyper_connection_mixer", prefix)
        };
        let hyper_connection_mixer = Qwen4HyperConnection::new(
            vb.clone(),
            comm.clone(),
            hc_count,
            model_config.hidden_size,
            qwen4.hc_lowrank,
            model_config.rms_norm_eps,
            dtype,
            false,
            &mixer_prefix,
        )?;

        let is_mlx_nvfp4_tied = tie_word_embeddings.is_some_and(|x| x)
            && model_config
                .quantization_config
                .as_ref()
                .is_some_and(|q| q.is_mlx_nvfp4);
        let lm_head = if is_mlx_nvfp4_tied {
            VocabParallelLinear::from_weight_bias(
                embed_tokens.embeddings().clone(),
                None,
                comm.clone(),
                vocab_size,
                dtype,
            )?
        } else {
            let lm_head_vb = if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else if is_qvar_builder {
                vb.pp(key_map["lm_head"])
            } else {
                vb.pp("lm_head")
            };
            VocabParallelLinear::load_no_bias(
                model_config.hidden_size,
                vocab_size,
                lm_head_vb,
                comm.clone(),
                &model_config.quantization_config,
                &None,
                dtype,
            )?
        };

        let hybrid = resolve_qwen3_hybrid_config(&model_config);
        let world_size = comm.world_size();
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        if num_v_heads % world_size != 0 || num_k_heads % world_size != 0 {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads,
                num_k_heads,
                world_size
            );
        }
        let num_v_heads = num_v_heads / world_size;
        let num_k_heads = num_k_heads / world_size;
        let key_head_dim = hybrid.key_head_dim;
        let value_head_dim = hybrid.value_head_dim;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let d_conv = num_k_heads * key_head_dim * 2 + num_v_heads * value_head_dim;
        let max_batch_size = 1;

        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                max_batch_size,
                d_conv,
                conv_kernel_size,
                num_v_heads,
                key_head_dim,
                value_head_dim,
                DType::F32,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, DType::F32, DType::F32, device)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            hyper_connection_mixer,
            rotary_emb,
            lm_head,
            mamba_cache: RwLock::new(mamba_cache),
            device: device.clone(),
            config: model_config,
            dtype,
            vocab_size,
            is_qvar_builder,
            hc_count,
            hc_hidden_size,
            mtp_hidden_buffer: std::sync::Mutex::new(None),
            dflash_verify_hidden_buffers: std::sync::Mutex::new(None),
            dflash_target_layer_ids: std::sync::Mutex::new(Vec::new()),
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

    pub fn embed_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn take_last_hidden_for_mtp(&self) -> Option<Tensor> {
        let guard = self.mtp_hidden_buffer.lock().ok()?;
        let buf = guard.as_ref()?;
        buf.get(0).ok()
    }

    pub fn preallocate_dflash_verify_buffers(
        &self,
        target_layer_ids: &[usize],
        max_verify_len: usize,
    ) -> Result<()> {
        let buf_dtype = if self.is_qvar_builder
            || self.config.quant.is_some()
            || self.config.higher_precision_required()
        {
            DType::F32
        } else {
            self.dtype
        };
        let mut buffers = Vec::with_capacity(target_layer_ids.len());
        for _ in target_layer_ids {
            buffers.push(Tensor::zeros(
                (max_verify_len, self.hc_hidden_size),
                buf_dtype,
                &self.device,
            )?);
        }
        if let Ok(mut guard) = self.dflash_verify_hidden_buffers.lock() {
            *guard = Some(buffers);
        }
        if let Ok(mut guard) = self.dflash_target_layer_ids.lock() {
            *guard = target_layer_ids.to_vec();
        }
        Ok(())
    }

    pub fn take_dflash_verify_hiddens(&self, num_tokens: usize) -> Option<Vec<Tensor>> {
        let ids = self.dflash_target_layer_ids.lock().ok()?;
        let guard = self.dflash_verify_hidden_buffers.lock().ok()?;
        let buffers = guard.as_ref()?;
        if buffers.len() != ids.len() || num_tokens == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(buffers.len());
        for buf in buffers {
            if num_tokens > buf.dim(0).ok()? {
                return None;
            }
            out.push(buf.narrow(0, 0, num_tokens).ok()?.contiguous().ok()?);
        }
        Some(out)
    }

    pub fn preallocate_mtp_hidden_buffer(&self, max_batch_size: usize) -> Result<()> {
        let buf = Tensor::zeros(
            (max_batch_size, self.hc_hidden_size),
            self.dtype,
            &self.device,
        )?;
        if let Ok(mut guard) = self.mtp_hidden_buffer.lock() {
            *guard = Some(buf);
        }
        Ok(())
    }

    pub fn forward_with_hidden(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut none_layers = None;
        let hidden = self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            true,
            None,
            &mut none_layers,
        )?;
        let logits = if self.is_qvar_builder {
            self.lm_head.forward(&hidden)?
        } else {
            self.lm_head
                .forward(&hidden.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)?
        };
        Ok((logits, hidden))
    }

    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        if self.is_qvar_builder {
            self.lm_head.forward(hidden)
        } else {
            self.lm_head
                .forward(&hidden.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
        return_hidden: bool,
        collect_layer_ids: Option<&[usize]>,
        collected_layers: &mut Option<Vec<Tensor>>,
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

        let embed = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };
        let mut xs = self.expand_hc_hidden(&embed)?;

        if collect_layer_ids.is_some() {
            *collected_layers = Some(vec![xs.clone()]);
        }

        let mut kv_cache_idx = 0usize;
        let seq_slots = self.resolve_seq_slots(input_metadata, xs.dim(0)?)?;
        let mut mamba_cache = self.mamba_cache.write();
        let shared_rotary: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_qsa_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                positions,
                cache,
                input_metadata,
                &mut mamba_cache,
                &seq_slots,
                &shared_rotary,
            )?;

            if let (Some(layer_ids), Some(layers)) = (collect_layer_ids, collected_layers.as_mut())
            {
                if layer_ids.contains(&i) {
                    layers.push(xs.clone());
                }
            }

            if input_metadata.is_mtp_verify {
                if let (Ok(ids), Ok(bufs)) = (
                    self.dflash_target_layer_ids.lock(),
                    self.dflash_verify_hidden_buffers.lock(),
                ) {
                    if let Some(buffers) = bufs.as_ref() {
                        for (buf_idx, &layer_id) in ids.iter().enumerate() {
                            if layer_id == i {
                                if let Some(buf) = buffers.get(buf_idx) {
                                    let n = xs.dim(0)?;
                                    if n <= buf.dim(0).unwrap_or(0) {
                                        if xs.dtype() != buf.dtype() {
                                            buf.narrow(0, 0, n)?
                                                .copy_(&xs.to_dtype(buf.dtype())?, 0)?;
                                        } else {
                                            buf.narrow(0, 0, n)?.copy_(&xs, 0)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let (Some(pos_mask), Some(deepstacks)) = (visual_pos_masks, deepstack_visual_embeds)
            {
                use crate::models::layers::deepstack::ApplyDeepStack;
                if i < deepstacks.len() {
                    xs = xs.apply_deep_stack(pos_mask, &deepstacks[i])?;
                }
            }
        }

        if collect_layer_ids.is_some() {
            let logits_xs = if !seqlens.is_empty() {
                let indices: Vec<_> = seqlens.iter().map(|x| x - 1).collect();
                let batch = indices.len();
                xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?
            } else {
                xs
            };
            let logits_xs = self.collapse_hc_hidden(&logits_xs)?;
            return if self.is_qvar_builder {
                self.lm_head.forward(&logits_xs)
            } else {
                self.lm_head
                    .forward(&logits_xs.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)
            };
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        if return_hidden {
            return self.collapse_hc_hidden(&xs)?.to_dtype(DType::F32);
        }

        if let Ok(guard) = self.mtp_hidden_buffer.lock() {
            if let Some(buf) = guard.as_ref() {
                if xs.elem_count() <= buf.elem_count() {
                    let _ = buf.copy_(&xs, 0);
                }
            }
        }

        let xs = self.collapse_hc_hidden(&xs)?;
        if self.is_qvar_builder {
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
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            false,
            None,
            &mut none_layers,
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
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            true,
            None,
            &mut none_layers,
        )
    }

    pub fn forward_collecting_layers(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        target_layer_ids: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut collected_layers = None;
        let logits = self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            false,
            Some(target_layer_ids),
            &mut collected_layers,
        )?;
        Ok((logits, collected_layers.unwrap_or_default()))
    }

    pub fn forward_with_deepstack(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            visual_pos_masks,
            deepstack_visual_embeds,
            false,
            None,
            &mut none_layers,
        )
    }

    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor> {
        if self.is_qvar_builder {
            self.lm_head.forward(hidden)
        } else {
            self.lm_head
                .forward(&hidden.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .ensure_slots_for_sequences(sequence_ids)
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .get_slots_for_sequences(sequence_ids)
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        self.mamba_cache
            .write()
            .capture_prefix_state(seq_id, hash, preserve)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.write().has_prefix_state(hash)
    }

    pub fn remove_mamba_prefix_state(&self, hash: u64) -> bool {
        let mut cache = self.mamba_cache.write();
        let existed = cache.has_prefix_state(hash);
        cache.remove_prefix_state(hash);
        existed
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn mtp_rollback_mamba(&self, seq_id: usize, keep_tokens: usize) -> Result<bool> {
        self.mtp_rollback_mamba_at(seq_id, keep_tokens, 0)
    }

    pub fn mtp_rollback_mamba_at(
        &self,
        seq_id: usize,
        keep_tokens: usize,
        snapshot_offset: usize,
    ) -> Result<bool> {
        let mut mamba_cache = self.mamba_cache.write();

        let slots = mamba_cache
            .get_slots_for_sequences(&[seq_id])?
            .into_iter()
            .map(|s| s as i64)
            .collect::<Vec<_>>();
        let seq_slots = Tensor::from_vec(slots, (1,), &self.device)?;
        for layer in &self.layers {
            if let Qwen4AttnType::LinearAttention(gdn) = &layer.attn {
                gdn.rollback_mtp_verify_at(
                    &mut mamba_cache,
                    &seq_slots,
                    keep_tokens,
                    snapshot_offset,
                )?;
            }
        }
        Ok(true)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
