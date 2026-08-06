use crate::models::layers::distributed::{AllReduce, Comm, VocabParallelLinear};
use crate::models::layers::ds_v4::{
    hc_expand, hc_head, hc_post, hc_pre_norm, CompressorDecodeState, CompressorWeights,
    FusedMoeMxfp4, HcBlockWeights, HcHeadWeights, HcHiddenStates, IndexerDecodeState,
    IndexerWeights, LayerCompressionType, LayerDecodeBuffers, LayerSparseKvCache, MlaV4Attention,
    MlaV4Config, V4RopeTables, V4Router,
};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::MoeW2ExpertWeights;
use crate::models::layers::others::{embedding, rms_norm_v4, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, Module};
use parking_lot::{Mutex, RwLock};
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

/// DeepSeek V4 specific config fields parsed from extra_config_json
pub struct DeepSeekV4Config {
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub n_hash_layers: usize,
    pub swiglu_limit: f32,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub sliding_window: usize,
}

impl DeepSeekV4Config {
    pub fn from_config(config: &Config) -> Self {
        let extra: serde_json::Value = config
            .extra_config_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);

        let compress_ratios = extra
            .get("compress_ratios")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Self {
            hc_mult: extra.get("hc_mult").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
            hc_eps: extra.get("hc_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
            hc_sinkhorn_iters: extra
                .get("hc_sinkhorn_iters")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize,
            n_hash_layers: extra
                .get("num_hash_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize,
            swiglu_limit: extra
                .get("swiglu_limit")
                .and_then(|v| v.as_f64())
                .unwrap_or(10.0) as f32,
            compress_ratios,
            compress_rope_theta: extra
                .get("compress_rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(160000.0),
            index_head_dim: extra
                .get("index_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            index_n_heads: extra
                .get("index_n_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize,
            index_topk: extra
                .get("index_topk")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            sliding_window: config.sliding_window.unwrap_or(128),
        }
    }

    /// Get the layer compression type for a given layer index.
    pub fn layer_compression(&self, layer_idx: usize) -> LayerCompressionType {
        self.compress_ratios
            .get(layer_idx)
            .copied()
            .map(LayerCompressionType::from_ratio)
            .unwrap_or(LayerCompressionType::Swa)
    }
}

/// DeepSeek V4 W2 MoE: V4 hash/score router + general [`MoeW2ExpertWeights`] from `moe.rs`.
struct V4MoeW2 {
    router: V4Router,
    experts: MoeW2ExpertWeights,
    act: Activation,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    swiglu_limit: f32,
}

impl V4MoeW2 {
    fn from_mxfp4(mx: FusedMoeMxfp4, swiglu_limit: f32) -> Result<Self> {
        let experts = MoeW2ExpertWeights::pack_from_mxfp4(
            &mx.gate_up_blocks,
            &mx.gate_up_scales,
            &mx.down_blocks,
            &mx.down_scales,
        )?;
        Ok(Self {
            router: mx.router,
            experts,
            act: mx.act,
            all_reduce: mx.all_reduce,
            world_size: mx.world_size,
            dtype: mx.dtype,
            swiglu_limit: if swiglu_limit > 0.0 {
                swiglu_limit
            } else if mx.swiglu_limit > 0.0 {
                mx.swiglu_limit
            } else {
                10.0
            },
        })
    }

    fn new(
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        layer_idx: usize,
        swiglu_limit: f32,
    ) -> Result<Self> {
        let mx = FusedMoeMxfp4::new(cfg, vb, comm, dtype, layer_idx)?;
        Self::from_mxfp4(mx, swiglu_limit)
    }

    fn forward_with_ids(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (topk_weights, topk_ids) = self.router.route(xs, input_ids, is_prefill)?;
        let mut ys = self.experts.forward(
            xs,
            &topk_weights,
            &topk_ids,
            &self.act,
            self.swiglu_limit,
            self.dtype,
            is_prefill,
        )?;
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

#[allow(dead_code)]
enum MoeOrMlp {
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeW2(V4MoeW2),
    Mlp(MLP),
}

impl MoeOrMlp {
    fn forward_with_ids(
        &self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoeMxfp4(m) => m.forward_with_ids_f32(xs, input_ids, is_prefill),
            Self::FusedMoeW2(m) => m.forward_with_ids(xs, input_ids, is_prefill),
        }
    }
}

fn seed_compressor_decode_state(
    compressor: &CompressorWeights,
    input: &Tensor,
    state: &mut CompressorDecodeState,
    _rope: &V4RopeTables,
    seq_len: usize,
    _rotate_fp4: bool,
) -> Result<()> {
    // Match official Compressor.forward prefill state copy. Decode-replay
    // seeding leaves dirty overlap second-half slots and diverges on
    // seq_len % ratio == 0; that poisons longer generations.
    compressor.seed_decode_state_after_prefill(input, state, seq_len)
}

pub struct DeepSeekV4DecoderLayer {
    self_attn: MlaV4Attention,
    mlp: MoeOrMlp,
    shared_expert: Option<MLP>,
    attn_norm: NormX,
    ffn_norm: NormX,
    hc_attn: HcBlockWeights,
    hc_ffn: HcBlockWeights,
    rope: Arc<V4RopeTables>,
    sparse_kv: Mutex<Option<LayerSparseKvCache>>,
    compressor_state: Mutex<Option<CompressorDecodeState>>,
    indexer_state: Mutex<Option<IndexerDecodeState>>,
    decode_buffers: Mutex<Option<LayerDecodeBuffers>>,
    max_seq_len: usize,
    qk_rope_head_dim: usize,
    #[allow(dead_code)]
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    compression: LayerCompressionType,
    compressor: Option<CompressorWeights>,
    indexer: Option<IndexerWeights>,
    sliding_window: usize,
}

impl DeepSeekV4DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rope: Arc<V4RopeTables>,
        max_seq_len: usize,
        config: &Config,
        mla_cfg: &MlaV4Config,
        v4_cfg: &DeepSeekV4Config,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let self_attn = MlaV4Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("attn").clone()
            },
            comm.clone(),
            mla_cfg,
            config,
            dtype,
            layer_idx,
        )?;

        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("DeepSeek V4 requires MoE config!");

        // DeepSeek V4 routed experts: MXFP4 safetensors, or optional W2 requant.
        let use_w2 = config
            .quant
            .as_deref()
            .map(|s| matches!(s.to_lowercase().as_str(), "w2" | "moe_w2"))
            .unwrap_or(false);
        let expert_dtype_is_fp4 = config
            .expert_dtype
            .as_deref()
            .is_none_or(|dtype| matches!(dtype.to_ascii_lowercase().as_str(), "fp4" | "mxfp4"));

        let mlp = if is_qvar_builder {
            candle_core::bail!(
                "DeepSeek V4 does not support GGUF MoE; use MXFP4 safetensors or --isq w2"
            );
        } else if use_w2 {
            // Load MXFP4 experts then GPU-repack via general moe::MoeW2ExpertWeights.
            MoeOrMlp::FusedMoeW2(V4MoeW2::new(
                config,
                vb.pp("ffn").clone(),
                comm.clone(),
                dtype,
                layer_idx,
                v4_cfg.swiglu_limit,
            )?)
        } else if config.quant.is_some() {
            candle_core::bail!(
                "DeepSeek V4 only supports MXFP4 MoE or --isq w2/moe_w2 (got quant={:?})",
                config.quant
            );
        } else if !expert_dtype_is_fp4 {
            candle_core::bail!(
                "DeepSeek V4 routed experts require expert_dtype=fp4/mxfp4; got {:?} (global quant_method={:?})",
                config.expert_dtype,
                config
                    .quantization_config
                    .as_ref()
                    .map(|quant| quant.quant_method.as_str())
            );
        } else {
            MoeOrMlp::FusedMoeMxfp4(FusedMoeMxfp4::new(
                config,
                vb.pp("ffn").clone(),
                comm.clone(),
                dtype,
                layer_idx,
            )?)
        };

        // Shared experts
        let shared_expert = if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size
        {
            if intermediate_size > 0 {
                let shared_vb = if is_qvar_builder {
                    vb.clone()
                } else if vb.pp("ffn.shared_experts").has_key("gate_proj.weight")
                    || vb
                        .pp("ffn.shared_experts")
                        .has_key("gate_proj.weight_packed")
                {
                    vb.pp("ffn.shared_experts").clone()
                } else if vb.pp("ffn.shared_experts.w1").has_key("weight") {
                    vb.pp("ffn.shared_experts").clone()
                } else {
                    vb.pp("ffn.shared_expert").clone()
                };
                // W2 only applies to routed experts; shared experts stay at checkpoint dtype.
                let shared_quant = if use_w2 { &None } else { &config.quant };
                let mlp = MLP::new(
                    shared_vb,
                    comm.clone(),
                    config.hidden_size,
                    intermediate_size * moe_cfg.n_shared_experts.unwrap_or(1),
                    &config.hidden_act,
                    &config.quantization_config,
                    shared_quant,
                    false,
                    dtype,
                    if is_qvar_builder { "_shexp" } else { "" },
                )?
                .with_swiglu_limit(v4_cfg.swiglu_limit);
                Some(mlp)
            } else {
                None
            }
        } else {
            None
        };

        // Match official ATen (128,4) mean-reduction order. Candle's generic
        // RMSNorm drifts enough that HC amplification blows up by mid layers.
        let attn_norm = rms_norm_v4(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("attn_norm").clone(),
            DType::F32,
        )?;

        let ffn_norm = rms_norm_v4(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("ffn_norm").clone(),
            DType::F32,
        )?;

        // HC weights for attention and FFN branches
        // Real model: hc_attn_* and hc_ffn_* both at layer root
        let hc_attn = HcBlockWeights::load(&vb, "hc_attn", v4_cfg.hc_mult, config.hidden_size)?;
        let hc_ffn = HcBlockWeights::load(&vb, "hc_ffn", v4_cfg.hc_mult, config.hidden_size)?;

        let compression = v4_cfg.layer_compression(layer_idx);

        let compressor = if compression.has_compressor() {
            let attn_vb = if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("attn")
            };
            CompressorWeights::load(
                &attn_vb,
                "compressor",
                compression.ratio(),
                mla_cfg.head_dim,
                config.hidden_size,
            )?
        } else {
            None
        };

        let indexer = if compression.has_indexer() {
            let attn_vb = if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("attn")
            };
            IndexerWeights::load(
                &attn_vb,
                "",
                config.hidden_size,
                v4_cfg.index_head_dim,
                v4_cfg.index_n_heads,
                v4_cfg.index_topk,
                mla_cfg.q_lora_rank,
                mla_cfg.qk_rope_head_dim,
                comm.world_size(),
                comm.clone(),
                layer_idx,
                v4_cfg.compress_rope_theta,
            )?
        } else {
            None
        };

        Ok(Self {
            self_attn,
            mlp,
            shared_expert,
            attn_norm,
            ffn_norm,
            hc_attn,
            hc_ffn,
            rope,
            sparse_kv: Mutex::new(None),
            compressor_state: Mutex::new(None),
            indexer_state: Mutex::new(None),
            decode_buffers: Mutex::new(None),
            max_seq_len,
            qk_rope_head_dim: mla_cfg.qk_rope_head_dim,
            hc_mult: v4_cfg.hc_mult,
            hc_sinkhorn_iters: v4_cfg.hc_sinkhorn_iters,
            hc_eps: v4_cfg.hc_eps,
            compression,
            compressor,
            indexer,
            sliding_window: v4_cfg.sliding_window,
        })
    }

    /// Allocate sparse/indexer/compressor decode buffers once. Required before CUDA
    /// graph capture and reused across prefill resets so captured kernels keep valid pointers.
    pub fn ensure_decode_buffers(&self, device: &Device) -> Result<()> {
        {
            let mut sparse = self.sparse_kv.lock();
            if sparse.is_none() {
                *sparse = Some(LayerSparseKvCache::new(
                    self.sliding_window,
                    self.compression.ratio(),
                    self.max_seq_len,
                    self.self_attn.get_head_dim(),
                    device,
                )?);
            }
        }
        if self.compressor.is_some() {
            let mut compressor_state = self.compressor_state.lock();
            if compressor_state.is_none() {
                let compressor = self.compressor.as_ref().unwrap();
                *compressor_state = Some(CompressorDecodeState::new(
                    compressor.ratio,
                    compressor.head_dim,
                    device,
                )?);
            }
        }
        if self.indexer.is_some() {
            let mut indexer_state = self.indexer_state.lock();
            if indexer_state.is_none() {
                *indexer_state = Some(IndexerDecodeState::new(
                    self.indexer.as_ref().unwrap().index_head_dim,
                    self.max_seq_len,
                    device,
                )?);
            }
        }
        {
            let mut decode_buffers = self.decode_buffers.lock();
            if decode_buffers.is_none() {
                let compress_topk = if let Some(indexer) = &self.indexer {
                    indexer.index_topk
                } else if self.compressor.is_some() {
                    self.max_seq_len / self.compression.ratio().max(1)
                } else {
                    0
                };
                let compressor_head_dim = self.compressor.as_ref().map(|c| c.head_dim);
                let indexer_head_dim = self.indexer.as_ref().map(|i| i.compressor.head_dim);
                *decode_buffers = Some(LayerDecodeBuffers::new(
                    device,
                    self.self_attn.get_num_heads(),
                    self.self_attn.get_head_dim(),
                    self.sliding_window,
                    compress_topk,
                    self.max_seq_len.div_ceil(self.compression.ratio().max(1)),
                    compressor_head_dim,
                    indexer_head_dim,
                )?);
            }
        }
        Ok(())
    }

    /// Zero sparse/compressor/indexer decode state before CUDA graph capture.
    pub fn reset_decode_state(&self) -> Result<()> {
        if let Some(sparse) = self.sparse_kv.lock().as_mut() {
            sparse.reset()?;
        }
        if let Some(state) = self.compressor_state.lock().as_mut() {
            state.reset()?;
        }
        if let Some(state) = self.indexer_state.lock().as_mut() {
            state.reset()?;
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        hc_hidden: &HcHiddenStates,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        _cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        input_ids: Option<&Tensor>,
    ) -> Result<HcHiddenStates> {
        // Attention branch: fused hc_pre + attn_norm -> attn -> hc_post
        let attn_norm_w = self.attn_norm.v4_weight_f32().ok_or_else(|| {
            candle_core::Error::Msg("DeepSeek V4 attn_norm missing F32 weight".into())
        })?;
        let (attn_normed, attn_hc_state) = hc_pre_norm(
            hc_hidden,
            &self.hc_attn.hc_fn,
            &self.hc_attn.hc_scale,
            &self.hc_attn.hc_base,
            attn_norm_w,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.attn_norm.v4_eps(),
        )?;
        let qr = self.self_attn.compute_qr(&attn_normed)?;
        let seq_len = attn_normed.dims()[0];
        let start_pos = if input_metadata.is_prefill {
            // Prefill currently only supports a fresh start (start_pos=0).
            0usize
        } else {
            // Do not D2H `positions` (I64 GPU→host reads were returning garbage in
            // the runner process). `max_context_len` is authoritative for decode.
            input_metadata.max_context_len.saturating_sub(1)
        };
        let attn_output = if input_metadata.is_prefill {
            if start_pos != 0 {
                candle_core::bail!("DeepSeek V4 sparse prefill currently requires start_pos=0");
            }

            let kv = self.self_attn.wkv_forward(&attn_normed)?.contiguous()?;
            self.rope.apply_inplace(&kv, 0, false)?;
            // Official AttentionMLA: FP8-simulate non-rope KV dims to match QAT
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                &kv,
                1,
                self.self_attn.get_head_dim(),
                self.qk_rope_head_dim,
                64,
            )?;

            let window_topk = self.sliding_window.min(seq_len);
            let mut topk_idxs = attention_rs::deepseek_v4::window_topk_indices(
                seq_len,
                self.sliding_window,
                window_topk,
                attn_normed.device(),
            )?;
            let mut total_topk = window_topk;
            let mut compressed_kv = None;
            let mut kv_combined = kv.clone();

            if let Some(compressor) = &self.compressor {
                if seq_len >= compressor.ratio {
                    let compressed =
                        compressor.prefill(&attn_normed, seq_len, Some(&self.rope), 0, false)?;
                    let compressed_len = compressed.dim(0)?;
                    let offset = seq_len;
                    let compress_idxs = if let Some(indexer) = &self.indexer {
                        let scores = indexer.scores_prefill(
                            &attn_normed,
                            &qr,
                            seq_len,
                            compressed_len,
                            &self.rope,
                        )?;
                        indexer.topk_prefill(&scores, seq_len, compressed_len, offset)?
                    } else {
                        attention_rs::deepseek_v4::compress_topk_indices(
                            seq_len,
                            compressed_len,
                            compressor.ratio,
                            offset,
                            attn_normed.device(),
                        )?
                    };
                    let compress_topk = compress_idxs.dim(1)?;
                    topk_idxs = attention_rs::deepseek_v4::concat_topk_indices(
                        &topk_idxs,
                        &compress_idxs,
                        seq_len,
                        window_topk,
                        compress_topk,
                    )?;
                    total_topk += compress_topk;
                    kv_combined = Tensor::cat(&[&kv, &compressed], 0)?;
                    compressed_kv = Some(compressed);
                }
            }

            let output = self.self_attn.sparse_attn(
                &attn_normed,
                &qr,
                &self.rope,
                0,
                &kv_combined,
                &topk_idxs,
                kv_combined.dim(0)?,
                total_topk,
            )?;
            self.ensure_decode_buffers(attn_normed.device())?;
            {
                let mut sparse = self.sparse_kv.lock();
                let sparse = sparse.as_mut().expect("sparse cache ensured");
                sparse.reset()?;
                sparse.seed_window_from_prefill(&kv)?;
                if let Some(compressed) = &compressed_kv {
                    sparse.seed_compressed_from_prefill(compressed)?;
                }
            }

            if let Some(compressor) = &self.compressor {
                let mut compressor_state = self.compressor_state.lock();
                let state = compressor_state.as_mut().expect("compressor state ensured");
                seed_compressor_decode_state(
                    compressor,
                    &attn_normed,
                    state,
                    &self.rope,
                    seq_len,
                    false,
                )?;
            }

            if let Some(indexer) = &self.indexer {
                let mut indexer_state = self.indexer_state.lock();
                let state = indexer_state.as_mut().expect("indexer state ensured");
                state.reset()?;
                if seq_len >= indexer.compressor.ratio {
                    // Prefill compressor leaves BF16 compressed KV; apply the same
                    // Hadamard+FP4 transform used on decode emits so scores_decode
                    // sees a consistent cache (Q is always Hadamard'd).
                    let indexer_kv = indexer.compressor.prefill(
                        &attn_normed,
                        seq_len,
                        Some(&self.rope),
                        0,
                        false,
                    )?;
                    let indexer_kv = indexer_kv.contiguous()?;
                    attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(
                        &indexer_kv,
                        1,
                        indexer.index_head_dim,
                    )?;
                    state.seed_from_prefill(&indexer_kv)?;
                }
                seed_compressor_decode_state(
                    &indexer.compressor,
                    &attn_normed,
                    &mut state.compressor_state,
                    &self.rope,
                    seq_len,
                    true,
                )?;
            }
            output
        } else {
            if seq_len != 1 {
                candle_core::bail!(
                    "DeepSeek V4 sparse decode currently supports one sequence at a time"
                );
            }
            if start_pos >= self.max_seq_len {
                candle_core::bail!(
                    "DeepSeek V4 decode position {start_pos} exceeds sparse cache capacity {}",
                    self.max_seq_len
                );
            }

            self.ensure_decode_buffers(attn_normed.device())?;
            let bufs = self.decode_buffers.lock();
            let bufs = bufs.as_ref().expect("decode buffers ensured");

            let kv = self.self_attn.wkv_forward(&attn_normed)?.contiguous()?;
            self.rope.apply_from_positions(&kv, positions, 0, false)?;
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                &kv,
                1,
                self.self_attn.get_head_dim(),
                self.qk_rope_head_dim,
                64,
            )?;
            self.sparse_kv
                .lock()
                .as_mut()
                .expect("sparse cache ensured")
                .write_window_token_from_pos(&kv, positions)?;

            if let Some(compressor) = &self.compressor {
                let weighted = bufs
                    .compressor_weighted
                    .as_ref()
                    .expect("compressor weighted buffer");
                let out = bufs.compressor_out.as_ref().expect("compressor out buffer");
                let emitted = {
                    let state = self.compressor_state.lock();
                    let state = state.as_ref().expect("compressor state ensured");
                    compressor.decode_graph(
                        &attn_normed,
                        state,
                        positions,
                        weighted,
                        out,
                        Some(&self.rope),
                        false,
                    )?
                };
                self.sparse_kv
                    .lock()
                    .as_mut()
                    .expect("sparse cache ensured")
                    .write_compressed_row_from_pos(&emitted, positions)?;
            }

            if let Some(indexer) = &self.indexer {
                let weighted = bufs
                    .indexer_compressor_weighted
                    .as_ref()
                    .expect("indexer compressor weighted buffer");
                let out = bufs
                    .indexer_compressor_out
                    .as_ref()
                    .expect("indexer compressor out buffer");
                let emitted = {
                    let state = self.indexer_state.lock();
                    let state = state.as_ref().expect("indexer state ensured");
                    indexer.compressor.decode_graph(
                        &attn_normed,
                        &state.compressor_state,
                        positions,
                        weighted,
                        out,
                        Some(&self.rope),
                        true,
                    )?
                };
                self.indexer_state
                    .lock()
                    .as_mut()
                    .expect("indexer state ensured")
                    .write_compressed_from_pos(&emitted, positions, indexer.compressor.ratio)?;
            }

            let win = self.sliding_window;
            attention_rs::deepseek_v4::window_topk_indices_decode_from_pos_into(
                positions,
                win,
                &bufs.window_topk,
            )?;

            // compress_topk is a per-layer constant (index_topk or compressed slot
            // capacity). Never branch on it for kernel topology — always fill
            // compress_topk (when this layer has a compressor) and always concat
            // into concat_topk so sparse_attn sees a fixed tensor address.
            let compress_topk = if self.indexer.is_some() {
                self.indexer.as_ref().unwrap().index_topk
            } else if self.compressor.is_some() {
                self.sparse_kv
                    .lock()
                    .as_ref()
                    .expect("sparse cache ensured")
                    .compressed_slots
            } else {
                0
            };

            if self.compressor.is_some() {
                if let Some(indexer) = &self.indexer {
                    let state = self.indexer_state.lock();
                    let state = state.as_ref().expect("indexer state ensured");
                    // Score/topk over the live compressed length, not the config
                    // capacity. Capacity can be max_position_embeddings (1M+) and
                    // must not be passed into shared-memory topk kernels.
                    let score_len = state.compressed_len.max(1).min(state.max_compressed_len);
                    let scores = bufs.indexer_scores.as_ref().expect("indexer scores buffer");
                    indexer.scores_decode_from_positions_into(
                        &attn_normed,
                        &qr,
                        &state.kv_cache,
                        score_len,
                        &self.rope,
                        positions,
                        scores,
                    )?;
                    attention_rs::deepseek_v4::indexer_mask_scores_by_position(
                        scores,
                        positions,
                        self.compression.ratio().max(1),
                    )?;
                    let topk_row = bufs.compress_topk.narrow(0, 0, 1)?.squeeze(0)?;
                    indexer.topk_decode_into(scores, score_len, win, &topk_row)?;
                } else {
                    attention_rs::deepseek_v4::compress_topk_indices_decode_from_pos_into(
                        positions,
                        compress_topk,
                        win,
                        self.compression.ratio().max(1),
                        &bufs.compress_topk,
                    )?;
                }
            }

            attention_rs::deepseek_v4::concat_topk_indices_into(
                &bufs.window_topk,
                &bufs.compress_topk,
                1,
                win,
                compress_topk,
                &bufs.concat_topk,
            )?;
            let total_topk = win + compress_topk;

            let sparse_cache = self.sparse_kv.lock();
            let sparse_cache = sparse_cache.as_ref().expect("sparse cache ensured");
            let kv_len = sparse_cache.total_slots().max(1);
            let attention_kv = sparse_cache.kv.clone();
            self.self_attn.sparse_attn_from_positions(
                &attn_normed,
                &qr,
                &self.rope,
                positions,
                &attention_kv,
                &bufs.concat_topk,
                &bufs.attn_out,
                kv_len,
                total_topk,
            )?
        };

        let attn_branch = attn_output.to_dtype(DType::F32)?.contiguous()?;
        let after_attn = hc_post(&attn_branch, hc_hidden, &attn_hc_state)?;
        // FFN branch: fused hc_pre + ffn_norm -> MoE -> hc_post
        let ffn_norm_w = self.ffn_norm.v4_weight_f32().ok_or_else(|| {
            candle_core::Error::Msg("DeepSeek V4 ffn_norm missing F32 weight".into())
        })?;
        let (ffn_normed, ffn_hc_state) = hc_pre_norm(
            &after_attn,
            &self.hc_ffn.hc_fn,
            &self.hc_ffn.hc_scale,
            &self.hc_ffn.hc_base,
            ffn_norm_w,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.ffn_norm.v4_eps(),
        )?;
        let shared_output = if let Some(shared_expert) = &self.shared_expert {
            Some(shared_expert.forward(&ffn_normed)?)
        } else {
            None
        };
        let mlp_output =
            self.mlp
                .forward_with_ids(&ffn_normed, input_ids, input_metadata.is_prefill)?;
        let ffn_output = if let Some(shared_output) = shared_output {
            (mlp_output.to_dtype(DType::F32)? + shared_output.to_dtype(DType::F32)?)?
        } else if mlp_output.dtype() != DType::F32 {
            mlp_output.to_dtype(DType::F32)?
        } else {
            mlp_output
        };
        let ffn_branch = ffn_output.to_dtype(DType::F32)?.contiguous()?;
        hc_post(&ffn_branch, &after_attn, &ffn_hc_state)
    }
}

pub struct DeepSeekV4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Mutex<DeepSeekV4DecoderLayer>>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    hc_head: Option<HcHeadWeights>,
    device: Device,
    config: Config,
    v4_cfg: DeepSeekV4Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl DeepSeekV4ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        _is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let prefix = if !is_qvar_builder && !vb.has_key(&format!("model.layers.0.attn_norm.weight"))
        {
            // DeepSeek V4 uses flat naming: "layers.0.attn.*" instead of "model.layers.0.*"
            ""
        } else {
            "model."
        };
        let mla_cfg = MlaV4Config::from_config(config);
        let v4_cfg = DeepSeekV4Config::from_config(config);

        let (embed_tokens, vocab_size) = if is_qvar_builder {
            embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp("token_embd"),
                dtype,
            )?
        } else {
            // DeepSeek V4 uses "embed.weight" in the weight map, not "model.embed_tokens.weight"
            match embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp(&format!("{}embed_tokens", prefix)),
                dtype,
            ) {
                Ok(r) => r,
                Err(_) => embedding(config.vocab_size, config.hidden_size, vb.pp("embed"), dtype)?,
            }
        };

        // Sparse RoPE / indexer / compressor capacity comes from the model
        // config (`max_position_embeddings`), not engine `max_model_len`.
        // Scheduling length is planned later by the KV allocator like other models.
        let sparse_max_seq_len = config.max_position_embeddings.max(1);
        let rope_scaling = config.rope_scaling.as_ref();
        let yarn_factor = rope_scaling
            .and_then(|m| m.get("factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let beta_fast = rope_scaling
            .and_then(|m| m.get("beta_fast"))
            .and_then(|v| v.as_f64())
            .unwrap_or(32.0) as f32;
        let beta_slow = rope_scaling
            .and_then(|m| m.get("beta_slow"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let original_seq_len = rope_scaling
            .and_then(|m| m.get("original_max_position_embeddings"))
            .and_then(|v| v.as_f64())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if yarn_factor > 1.0 {
                    (config.max_position_embeddings as f64 / yarn_factor) as usize
                } else {
                    config.max_position_embeddings
                }
            });
        let rope_theta = config.rope_theta.unwrap_or(10000.0);
        let rope_compress = Arc::new(V4RopeTables::precompute(
            &vb.device(),
            sparse_max_seq_len,
            mla_cfg.qk_rope_head_dim,
            v4_cfg.compress_rope_theta,
            original_seq_len,
            yarn_factor,
            beta_fast,
            beta_slow,
        )?);
        let rope_swa = Arc::new(V4RopeTables::precompute(
            &vb.device(),
            sparse_max_seq_len,
            mla_cfg.qk_rope_head_dim,
            rope_theta,
            0,
            yarn_factor,
            beta_fast,
            beta_slow,
        )?);

        let reporter = progress_reporter.clone();
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer_vb = if is_qvar_builder {
                vb.pp(format!("blk.{}", i).as_str())
            } else {
                vb.pp(format!("{}layers.{}", prefix, i).as_str())
            };
            let rope = if v4_cfg.layer_compression(i).has_compressor() {
                rope_compress.clone()
            } else {
                rope_swa.clone()
            };
            let layer = DeepSeekV4DecoderLayer::new(
                layer_vb,
                comm.clone(),
                rope,
                sparse_max_seq_len,
                config,
                &mla_cfg,
                &v4_cfg,
                dtype,
                i,
            )?;
            layers.push(Mutex::new(layer));
            reporter.write().set_progress(i + 1);
        }

        // Allocate sparse/indexer buffers during model load so KV planning sees
        // the real free VRAM (these are private caches, not the paged KV pool).
        for layer in &layers {
            layer.lock().ensure_decode_buffers(&vb.device())?;
        }

        let norm = rms_norm_v4(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("output_norm")
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
        )?;

        let tie_word_embeddings = config.tie_word_embeddings;
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp("token_embd")
                } else if vb.has_key(&format!("{}embed_tokens.weight", prefix)) {
                    vb.pp(&format!("{}embed_tokens", prefix))
                } else {
                    vb.pp("embed")
                }
            } else if is_qvar_builder {
                vb.pp("output")
            } else if vb.has_key("lm_head.weight") {
                vb.pp("lm_head")
            } else {
                vb.pp("head")
            },
            comm.clone(),
            &None,
            &None,
            dtype,
        )?;

        // HC head weights (for the final collapse)
        let hc_head = if v4_cfg.hc_mult > 1 {
            let hc_vb = if is_qvar_builder {
                vb.clone()
            } else if prefix.is_empty() {
                vb.clone()
            } else {
                vb.pp(prefix)
            };
            match HcHeadWeights::load(&hc_vb, v4_cfg.hc_mult, config.hidden_size) {
                Ok(w) => Some(w),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hc_head,
            device: device.clone(),
            config: config.clone(),
            v4_cfg,
            dtype,
            vocab_size,
            is_qvar_builder,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        // W2 only requants routed MoE experts; dense/attn stay FP8 and need BF16/F16 acts.
        // Casting embeds to F32 (ISQ/GGUF path) breaks fp8_matmul_cutlass on SM90.
        let is_w2 = self
            .config
            .quant
            .as_deref()
            .map(|s| matches!(s.to_lowercase().as_str(), "w2" | "moe_w2"))
            .unwrap_or(false);
        if !is_w2
            && (self.is_qvar_builder || self.config.quant.is_some())
            && xs.dtype() != DType::F32
        {
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

        let xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        // Expand to HC hidden states
        let mut hc_hidden = hc_expand(&xs, self.v4_cfg.hc_mult)?;

        if let Some(kv_caches) = kv_caches {
            for (layer_idx, ((k_cache, v_cache), layer)) in
                zip(kv_caches.iter(), self.layers.iter()).enumerate()
            {
                let mut layer = layer.lock();
                let layer_input_ids = if layer_idx < self.v4_cfg.n_hash_layers {
                    Some(input_ids)
                } else {
                    None
                };
                hc_hidden = layer.forward(
                    &hc_hidden,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                    layer_input_ids,
                )?;
            }
        }

        // HC head collapse
        let mut xs = if let Some(hc_head_weights) = &self.hc_head {
            hc_head(
                &hc_hidden,
                &hc_head_weights.hc_fn,
                &hc_head_weights.hc_scale,
                &hc_head_weights.hc_base,
            )?
        } else {
            // Fallback: take first HC branch [seq, hc, dim] -> [seq, dim]
            hc_hidden.data.narrow(1, 0, 1)?.squeeze(1)?
        };
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs
                .contiguous()?
                .index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        // Final RMSNorm: owned buffer, single in-place CUDA kernel.
        let xs = if xs.dtype() == DType::BF16 && xs.is_contiguous() {
            xs
        } else {
            xs.to_dtype(DType::BF16)?.contiguous()?
        };
        self.norm.forward_v4_inplace(&xs)?;
        self.lm_head.forward(&xs)
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

    /// Prewarm V4 HC/indexer/compressor scratch pools before CUDA graph capture.
    pub fn prewarm_cuda_graph_scratch(&self) -> Result<()> {
        for layer in &self.layers {
            layer.lock().ensure_decode_buffers(&self.device)?;
        }
        let hc = self.v4_cfg.hc_mult;
        let hidden = self.config.hidden_size;
        let hc_elems = hc * hidden;
        let fp4_elems = self.v4_cfg.index_n_heads * self.v4_cfg.index_head_dim.max(128);
        attention_rs::deepseek_v4::prewarm_decode_scratch(&self.device, hc_elems, fp4_elems)
    }

    /// Clear per-layer decode accumulators before graph capture warmup.
    pub fn reset_decode_state_for_graph(&self) -> Result<()> {
        for layer in &self.layers {
            layer.lock().reset_decode_state()?;
        }
        Ok(())
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
