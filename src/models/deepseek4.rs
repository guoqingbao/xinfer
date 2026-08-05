use crate::models::layers::distributed::{Comm, VocabParallelLinear};
use crate::models::layers::ds_v4::{
    hc_expand, hc_head, hc_post, hc_pre, CompressorDecodeState, CompressorWeights, FusedMoeMxfp4,
    HcBlockWeights, HcHeadWeights, HcHiddenStates, IndexerDecodeState, IndexerWeights,
    LayerCompressionType, LayerSparseKvCache, MlaV4Attention, MlaV4Config, V4RopeTables,
};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm_v4, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
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

#[allow(dead_code)]
enum MoeOrMlp {
    FusedMoeMxfp4(FusedMoeMxfp4),
    // FusedMoeW2(FusedMoeW2),
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
            // Self::FusedMoeW2(m) => m.forward_with_ids(xs, input_ids, is_prefill),
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
                "DeepSeek V4 does not support GGUF MoE; use MXFP4 safetensors or --quant w2"
            );
        } else if config.quant.is_some() {
            candle_core::bail!(
                "DeepSeek V4 only supports MXFP4 MoE or --quant w2/moe_w2 (got quant={:?})",
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

    pub fn forward(
        &mut self,
        hc_hidden: &HcHiddenStates,
        _attention_mask: Option<&Vec<Tensor>>,
        _positions: &Tensor,
        _cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        input_ids: Option<&Tensor>,
    ) -> Result<HcHiddenStates> {
        // Attention branch: hc_pre -> norm -> attn -> hc_post
        let (attn_input, attn_hc_state) = hc_pre(
            hc_hidden,
            &self.hc_attn.hc_fn,
            &self.hc_attn.hc_scale,
            &self.hc_attn.hc_base,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )?;
        let attn_normed = self.attn_norm.forward(&attn_input)?;
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
            let mut sparse_cache = LayerSparseKvCache::new(
                self.sliding_window,
                self.compression.ratio(),
                self.max_seq_len,
                self.self_attn.get_head_dim(),
                attn_normed.device(),
            )?;
            sparse_cache.seed_window_from_prefill(&kv)?;
            if let Some(compressed) = &compressed_kv {
                sparse_cache.seed_compressed_from_prefill(compressed)?;
            }
            *self.sparse_kv.lock() = Some(sparse_cache);

            *self.compressor_state.lock() = if let Some(compressor) = &self.compressor {
                let mut state = CompressorDecodeState::new(
                    compressor.ratio,
                    compressor.head_dim,
                    attn_normed.device(),
                )?;
                seed_compressor_decode_state(
                    compressor,
                    &attn_normed,
                    &mut state,
                    &self.rope,
                    seq_len,
                    false,
                )?;
                Some(state)
            } else {
                None
            };

            *self.indexer_state.lock() = if let Some(indexer) = &self.indexer {
                let mut state = IndexerDecodeState::new(
                    indexer.index_head_dim,
                    self.max_seq_len,
                    attn_normed.device(),
                )?;
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
                Some(state)
            } else {
                None
            };
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

            if self.sparse_kv.lock().is_none() {
                *self.sparse_kv.lock() = Some(LayerSparseKvCache::new(
                    self.sliding_window,
                    self.compression.ratio(),
                    self.max_seq_len,
                    self.self_attn.get_head_dim(),
                    attn_normed.device(),
                )?);
            }

            let kv = self.self_attn.wkv_forward(&attn_normed)?.contiguous()?;
            self.rope.apply_inplace(&kv, start_pos, false)?;
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
                .expect("sparse cache initialized")
                .write_window_token(&kv, start_pos)?;

            if let Some(compressor) = &self.compressor {
                if self.compressor_state.lock().is_none() {
                    *self.compressor_state.lock() = Some(CompressorDecodeState::new(
                        compressor.ratio,
                        compressor.head_dim,
                        attn_normed.device(),
                    )?);
                }
                let emitted = {
                    let state = self.compressor_state.lock();
                    compressor.decode(
                        &attn_normed,
                        state.as_ref().expect("compressor state initialized"),
                        start_pos,
                        Some(&self.rope),
                        false,
                    )?
                };
                if let Some(row) = emitted {
                    self.sparse_kv
                        .lock()
                        .as_mut()
                        .expect("sparse cache initialized")
                        .write_compressed_row(&row, start_pos / compressor.ratio)?;
                }
            }

            if let Some(indexer) = &self.indexer {
                if self.indexer_state.lock().is_none() {
                    *self.indexer_state.lock() = Some(IndexerDecodeState::new(
                        indexer.index_head_dim,
                        self.max_seq_len,
                        attn_normed.device(),
                    )?);
                }
                let emitted = {
                    let state = self.indexer_state.lock();
                    indexer.compressor.decode(
                        &attn_normed,
                        &state
                            .as_ref()
                            .expect("indexer state initialized")
                            .compressor_state,
                        start_pos,
                        Some(&self.rope),
                        true,
                    )?
                };
                if let Some(row) = emitted {
                    let row_idx = start_pos / indexer.compressor.ratio;
                    self.indexer_state
                        .lock()
                        .as_mut()
                        .expect("indexer state initialized")
                        .write_compressed_at(&row, row_idx)?;
                }
            }

            let win = self.sliding_window;
            // Build the ring order and `-1` padding on the device. This runs
            // once per sparse layer and token, so a host Vec + H2D copy was a
            // material decode bottleneck.
            let mut topk_idxs = attention_rs::deepseek_v4::window_topk_indices_decode(
                start_pos,
                win,
                attn_normed.device(),
            )?;
            let mut total_topk = win;

            let compressed_len = self
                .sparse_kv
                .lock()
                .as_ref()
                .expect("sparse cache initialized")
                .compressed_len;
            // Official indexer scores `:end_pos // ratio` (decode: start_pos+1).
            let end_compressed = (start_pos + 1) / self.compression.ratio().max(1);
            if end_compressed > 0 && compressed_len > 0 {
                let compress_idxs = if let Some(indexer) = &self.indexer {
                    let state = self.indexer_state.lock();
                    let state = state.as_ref().expect("indexer state initialized");
                    let score_len = end_compressed
                        .min(compressed_len)
                        .min(state.compressed_len)
                        .max(1);
                    let scores = indexer.scores_decode(
                        &attn_normed,
                        &qr,
                        &state.kv_cache,
                        score_len,
                        &self.rope,
                        start_pos,
                    )?;
                    indexer
                        .topk_decode(&scores, score_len, self.sliding_window)?
                        .reshape((1, indexer.index_topk.min(score_len)))?
                } else {
                    let n = ((start_pos + 1) / self.compression.ratio().max(1)).min(compressed_len);
                    attention_rs::deepseek_v4::compress_topk_indices_decode(
                        n,
                        self.sliding_window,
                        attn_normed.device(),
                    )?
                };
                let compress_topk = compress_idxs.dim(1)?;
                topk_idxs = attention_rs::deepseek_v4::concat_topk_indices(
                    &topk_idxs,
                    &compress_idxs,
                    1,
                    win,
                    compress_topk,
                )?;
                total_topk += compress_topk;
            }

            let sparse_cache = self.sparse_kv.lock();
            let sparse_cache = sparse_cache.as_ref().expect("sparse cache initialized");
            // Official decode attends over the full cache buffer (unused
            // compressed slots stay zero / are skipped via -1 topk).
            let kv_len = sparse_cache.total_slots().max(1);
            let attention_kv = sparse_cache.kv.clone();
            self.self_attn.sparse_attn(
                &attn_normed,
                &qr,
                &self.rope,
                start_pos,
                &attention_kv,
                &topk_idxs.contiguous()?,
                kv_len,
                total_topk,
            )?
        };

        let attn_branch = attn_output.to_dtype(DType::F32)?.contiguous()?;
        let after_attn = hc_post(&attn_branch, hc_hidden, &attn_hc_state)?;
        // FFN branch: hc_pre -> norm -> MoE -> hc_post
        let (ffn_input, ffn_hc_state) = hc_pre(
            &after_attn,
            &self.hc_ffn.hc_fn,
            &self.hc_ffn.hc_scale,
            &self.hc_ffn.hc_base,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )?;
        let ffn_normed = self.ffn_norm.forward(&ffn_input)?;
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

        let max_seq_len = config
            .max_model_len
            // Prefer engine scheduling length; never default to the model's 1M+
            // `max_position_embeddings` (that blows up sparse KV / RoPE tables and
            // breaks decode shared-memory top-k for large compressed_len).
            // When unset, keep a modest default so `--kv-fraction` still has room
            // for the paged KV cache after V4 private sparse buffers are allocated.
            .unwrap_or(8192)
            .min(config.max_position_embeddings);
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
            max_seq_len,
            mla_cfg.qk_rope_head_dim,
            v4_cfg.compress_rope_theta,
            original_seq_len,
            yarn_factor,
            beta_fast,
            beta_slow,
        )?);
        let rope_swa = Arc::new(V4RopeTables::precompute(
            &vb.device(),
            max_seq_len,
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
                max_seq_len,
                config,
                &mla_cfg,
                &v4_cfg,
                dtype,
                i,
            )?;
            layers.push(Mutex::new(layer));
            reporter.write().set_progress(i + 1);
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
        let xs = self.norm.forward(&xs)?;
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

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
