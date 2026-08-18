use crate::models::layers::activation::GatedActivation;
use crate::models::layers::distributed::{AllReduce, Comm, VocabParallelLinear};
use crate::models::layers::ds_v4::{
    hc_expand, hc_head, hc_post, hc_pre_norm, CompressorDecodeState, CompressorWeights,
    FusedMoeMxfp4, HcBlockWeights, HcHeadWeights, HcHiddenStates, IndexerDecodeState,
    IndexerWeights, LayerCompressionType, LayerDecodeBuffers, LayerSparseKvCache, MlaV4Attention,
    MlaV4Config, V4HybridPagePool, V4RopeTables, V4Router, V4_NATIVE_BLOCK_SIZE,
};
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::MoeW2ExpertWeights;
use crate::models::layers::others::{embedding, rms_norm_v4, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
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
    act: GatedActivation,
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
    hybrid_pool: Arc<Mutex<Option<V4HybridPagePool>>>,
    layer_idx: usize,
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

/// Per-request recurrent state owned by the compressor, indexer and sparse ring.
///
/// These are GPU tensor handles, not device copies: switching requests moves the
/// handles between the request table and the layer's active fields, so a request
/// transition never reconstructs recurrent state from quantized page data.
struct LayerStateSnapshot {
    sparse: Option<LayerSparseKvCache>,
    compressor: Option<CompressorDecodeState>,
    indexer: Option<IndexerDecodeState>,
}

/// Parked recurrent state for one request, tagged with the absolute position it
/// was parked at.
///
/// The tag is what makes reuse safe: a snapshot only describes the history up to
/// `next_pos`, so a request that resumes anywhere else (prefix-cache hit,
/// recompute after preemption) must rebuild from its pages instead of silently
/// inheriting a history that does not belong to its current block table.
struct RequestState {
    next_pos: usize,
    /// Physical pages backing `[0, next_pos)`, used to prove that another
    /// request resuming at `next_pos` shares this exact history.
    block_prefix: Vec<u32>,
    layers: Vec<LayerStateSnapshot>,
}

impl RequestState {
    /// True when a request resuming at `start_pos` over `block_table` has the
    /// same token history as this state.
    ///
    /// A prefix-cache hit hands the new request the *same physical pages* for
    /// the matched region, so identical pages at an identical position mean the
    /// recurrent state is already correct — the compressor and indexer never
    /// have to be rebuilt from the page residual.
    fn covers(&self, start_pos: usize, block_table: &[u32]) -> bool {
        // `>=` mirrors the single-sequence path this replaces: a state that has
        // already run past `start_pos` still holds the history up to it, since
        // the ring and the compressed rows are indexed by absolute position.
        if self.next_pos < start_pos {
            return false;
        }
        let pages = start_pos / V4_NATIVE_BLOCK_SIZE;
        if pages > self.block_prefix.len() || pages > block_table.len() {
            return false;
        }
        self.block_prefix[..pages] == block_table[..pages]
    }
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
        hybrid_pool: Arc<Mutex<Option<V4HybridPagePool>>>,
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
            hybrid_pool,
            layer_idx,
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

    /// Detach this layer's recurrent state so it can be parked in the request table.
    fn take_state(&self) -> LayerStateSnapshot {
        LayerStateSnapshot {
            sparse: self.sparse_kv.lock().take(),
            compressor: self.compressor_state.lock().take(),
            indexer: self.indexer_state.lock().take(),
        }
    }

    /// Install the recurrent state of the request that is about to run.
    ///
    /// A request without parked state is either starting from scratch or resuming
    /// from a prefix-cache hit at a frozen native boundary, which is hydrated from
    /// its pages.
    fn restore_state(
        &self,
        snapshot: Option<LayerStateSnapshot>,
        start_pos: usize,
        block_table: &[u32],
        device: &Device,
    ) -> Result<()> {
        let had_persistent_state = snapshot.is_some();
        if let Some(snapshot) = snapshot {
            if let Some(state) = snapshot.sparse {
                *self.sparse_kv.lock() = Some(state);
            }
            if let Some(state) = snapshot.compressor {
                *self.compressor_state.lock() = Some(state);
            }
            if let Some(state) = snapshot.indexer {
                *self.indexer_state.lock() = Some(state);
            }
        }

        // Install persistent handles before ensuring buffers: the reverse order
        // would allocate a throw-away recurrent state on every request switch.
        self.ensure_decode_buffers(device)?;

        if had_persistent_state {
            return Ok(());
        }

        // No parked state: these handles were just allocated, so put them in the
        // same cleared state a finished request used to leave behind before
        // hydrating. Hydration only rewrites the residual window and the rows
        // covered by `start_pos`; anything outside that has to already be
        // cleared, or the compressor scores over the untouched rows are read as
        // real history.
        self.reset_decode_state()?;
        if start_pos == 0 {
            return Ok(());
        }
        if start_pos % V4_NATIVE_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "DeepSeek V4 request state missing for non-boundary position {start_pos}"
            );
        }
        self.load_residual_from_pages(start_pos, block_table)?;
        Ok(())
    }

    /// Fresh single-shot prefill (Phase 0 golden path). Used when `start_pos==0`.
    /// Cold prefill (start_pos=0). Returns `(attn_out, token_kv)` so the caller
    /// can commit the full-sequence SWA into pages (ring alone is not enough).
    fn prefill_fresh(
        &mut self,
        attn_normed: &Tensor,
        qr: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let kv = self.self_attn.wkv_forward(attn_normed)?.contiguous()?;
        self.rope.apply_inplace(&kv, 0, false)?;
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
                    compressor.prefill(attn_normed, seq_len, Some(&self.rope), 0, false)?;
                let compressed_len = compressed.dim(0)?;
                let offset = seq_len;
                let compress_idxs = if let Some(indexer) = &self.indexer {
                    let scores = indexer.scores_prefill(
                        attn_normed,
                        qr,
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
                // Single allocation instead of Tensor::cat (avoids temporary peak).
                let head_dim = self.self_attn.get_head_dim();
                let total = seq_len + compressed_len;
                let combined = Tensor::zeros((total, head_dim), DType::BF16, attn_normed.device())?;
                attention_rs::deepseek_v4::copy_contiguous_into(&combined, &kv, 0)?;
                attention_rs::deepseek_v4::copy_contiguous_into(
                    &combined,
                    &compressed,
                    seq_len * head_dim,
                )?;
                kv_combined = combined;
                compressed_kv = Some(compressed);
            }
        }

        let output = self.self_attn.sparse_attn(
            attn_normed,
            qr,
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
            // Keep Phase-0 golden window layout for fresh prefills.
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
                attn_normed,
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
                let indexer_kv =
                    indexer
                        .compressor
                        .prefill(attn_normed, seq_len, Some(&self.rope), 0, false)?;
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
                attn_normed,
                &mut state.compressor_state,
                &self.rope,
                seq_len,
                true,
            )?;
        }
        Ok((output, kv))
    }

    /// One-token sparse decode step (write KV/compressor then attend).
    fn decode_one_token(
        &mut self,
        attn_normed: &Tensor,
        qr: &Tensor,
        positions: &Tensor,
        abs_pos: usize,
        block_table: &[u32],
    ) -> Result<Tensor> {
        self.ensure_decode_buffers(attn_normed.device())?;

        let kv = self.self_attn.wkv_forward(attn_normed)?.contiguous()?;
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

        let out = {
            let bufs_guard = self.decode_buffers.lock();
            let bufs = bufs_guard.as_ref().expect("decode buffers ensured");

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
                        attn_normed,
                        state,
                        positions,
                        weighted,
                        out,
                        Some(&self.rope),
                        false,
                    )?
                };
                {
                    let mut sparse = self.sparse_kv.lock();
                    let sparse = sparse.as_mut().expect("sparse cache ensured");
                    sparse.write_compressed_row_from_pos(&emitted, positions)?;
                    let ratio = sparse.compress_ratio.max(1);
                    if (abs_pos + 1) % ratio == 0 {
                        let row = abs_pos / ratio;
                        sparse.compressed_len = sparse.compressed_len.max(row + 1);
                    }
                }
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
                        attn_normed,
                        &state.compressor_state,
                        positions,
                        weighted,
                        out,
                        Some(&self.rope),
                        true,
                    )?
                };
                {
                    let mut istate = self.indexer_state.lock();
                    let istate = istate.as_mut().expect("indexer state ensured");
                    istate.write_compressed_from_pos(
                        &emitted,
                        positions,
                        indexer.compressor.ratio,
                    )?;
                    let ratio = indexer.compressor.ratio.max(1);
                    if (abs_pos + 1) % ratio == 0 {
                        let row = abs_pos / ratio;
                        istate.compressed_len = istate.compressed_len.max(row + 1);
                    }
                }
            }

            let win = self.sliding_window;
            attention_rs::deepseek_v4::window_topk_indices_decode_from_pos_into(
                positions,
                win,
                &bufs.window_topk,
            )?;

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
                    let score_len = state.compressed_len.max(1).min(state.max_compressed_len);
                    let scores = bufs.indexer_scores.as_ref().expect("indexer scores buffer");
                    indexer.scores_decode_from_positions_into(
                        attn_normed,
                        qr,
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

            let (attention_kv, kv_len) = {
                let sparse_cache = self.sparse_kv.lock();
                let sparse_cache = sparse_cache.as_ref().expect("sparse cache ensured");
                (sparse_cache.kv.clone(), sparse_cache.total_slots().max(1))
            };
            self.self_attn.sparse_attn_from_positions(
                attn_normed,
                qr,
                &self.rope,
                positions,
                &attention_kv,
                &bufs.concat_topk,
                &bufs.attn_out,
                kv_len,
                total_topk,
                Some(attention_rs::deepseek_v4::FlashInferSm120Layout {
                    window_len: win,
                    compressed_offset: Some(win),
                }),
            )?
        };
        // Write-through into engine pages (pages = source of truth).
        self.commit_pages(abs_pos, 1, block_table, None)?;
        Ok(out)
    }

    /// Decode-style sparse attend against the live `[ring | compressed]` buffer.
    /// Caller must have already written this token's SWA ring slot (and any
    /// compressed rows this position emits).
    fn decode_attend_only(
        &self,
        attn_normed: &Tensor,
        qr: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let bufs_guard = self.decode_buffers.lock();
        let bufs = bufs_guard.as_ref().expect("decode buffers ensured");
        let win = self.sliding_window;
        attention_rs::deepseek_v4::window_topk_indices_decode_from_pos_into(
            positions,
            win,
            &bufs.window_topk,
        )?;

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
                let score_len = state.compressed_len.max(1).min(state.max_compressed_len);
                let scores = bufs.indexer_scores.as_ref().expect("indexer scores buffer");
                indexer.scores_decode_from_positions_into(
                    attn_normed,
                    qr,
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
        let (attention_kv, kv_len) = {
            let sparse_cache = self.sparse_kv.lock();
            let sparse_cache = sparse_cache.as_ref().expect("sparse cache ensured");
            (sparse_cache.kv.clone(), sparse_cache.total_slots().max(1))
        };
        self.self_attn.sparse_attn_from_positions(
            attn_normed,
            qr,
            &self.rope,
            positions,
            &attention_kv,
            &bufs.concat_topk,
            &bufs.attn_out,
            kv_len,
            total_topk,
            Some(attention_rs::deepseek_v4::FlashInferSm120Layout {
                window_len: win,
                compressed_offset: Some(win),
            }),
        )
    }

    /// Load compressor/indexer residual (+ sparse gather) from engine pages at a
    /// native-block boundary — vLLM prefix/chunk handoff (no mamba snapshot).
    fn load_residual_from_pages(&self, start_pos: usize, block_table: &[u32]) -> Result<()> {
        if start_pos == 0 || block_table.is_empty() {
            return Ok(());
        }
        if start_pos % V4_NATIVE_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "DeepSeek V4 page handoff at {} is not native-block aligned ({})",
                start_pos,
                V4_NATIVE_BLOCK_SIZE
            );
        }
        let device = {
            let sparse = self.sparse_kv.lock();
            if let Some(sparse) = sparse.as_ref() {
                sparse.kv.device().clone()
            } else {
                let pool_guard = self.hybrid_pool.lock();
                let Some(pool) = pool_guard.as_ref() else {
                    return Ok(());
                };
                pool.layers
                    .first()
                    .map(|l| l.swa.device().clone())
                    .ok_or_else(|| candle_core::Error::Msg("V4 hybrid pool empty".into()))?
            }
        };
        self.ensure_decode_buffers(&device)?;
        let pool_guard = self.hybrid_pool.lock();
        let Some(pool) = pool_guard.as_ref() else {
            return Ok(());
        };
        if !pool.residual_frozen_at(start_pos, block_table) {
            candle_core::bail!(
                "DeepSeek V4 handoff at {}: residual not frozen at boundary",
                start_pos
            );
        }
        let ratio = self.compression.ratio();
        let compressed_len = if ratio == 0 { 0 } else { start_pos / ratio };
        {
            let mut sparse = self.sparse_kv.lock();
            let sparse = sparse.as_mut().expect("sparse cache ensured");
            sparse.reset()?;
            pool.gather_sparse_into(
                self.layer_idx,
                &sparse.kv,
                sparse.sliding_window,
                compressed_len,
                start_pos,
                block_table,
            )?;
            sparse.compressed_len = compressed_len.min(sparse.compressed_slots);
        }
        if let Some(state) = self.compressor_state.lock().as_mut() {
            pool.load_residual_into_state(
                self.layer_idx,
                &state.kv_state,
                &state.score_state,
                start_pos,
                block_table,
                false,
            )?;
        }
        if let Some(state) = self.indexer_state.lock().as_mut() {
            pool.load_residual_into_state(
                self.layer_idx,
                &state.compressor_state.kv_state,
                &state.compressor_state.score_state,
                start_pos,
                block_table,
                true,
            )?;
            let compressed_len = start_pos / 4;
            if compressed_len > 0 {
                if let Some(layer) = pool.layer(self.layer_idx) {
                    if let Some(ic) = &layer.indexer_compressed {
                        let n = compressed_len.min(state.max_compressed_len);
                        let rows_per_page = V4_NATIVE_BLOCK_SIZE / 4;
                        let hd = state.kv_cache.dim(1)?;
                        let mut abs_row = 0;
                        while abs_row < n {
                            let page_idx = abs_row / rows_per_page;
                            let row = abs_row % rows_per_page;
                            let room = rows_per_page - row;
                            let take = room.min(n - abs_row);
                            let Some(&page) = block_table.get(page_idx) else {
                                abs_row += take;
                                continue;
                            };
                            let page = page as usize;
                            if page >= pool.num_pages {
                                abs_row += take;
                                continue;
                            }
                            let src = ic
                                .narrow(0, page, 1)?
                                .squeeze(0)?
                                .narrow(0, row, take)?
                                .contiguous()?;
                            attention_rs::deepseek_v4::copy_contiguous_into(
                                &state.kv_cache,
                                &src,
                                abs_row * hd,
                            )?;
                            abs_row += take;
                        }
                        state.compressed_len = n;
                    }
                }
            }
        }
        Ok(())
    }

    /// Commit SWA / compressed / residual into engine-owned pages (vLLM write path).
    /// Called from the same forward that produced the values — not a scratch mirror.
    fn commit_pages(
        &self,
        start_pos: usize,
        seq_len: usize,
        block_table: &[u32],
        token_kv: Option<&Tensor>,
    ) -> Result<()> {
        if block_table.is_empty() || seq_len == 0 {
            return Ok(());
        }
        let pool_guard = self.hybrid_pool.lock();
        let Some(pool) = pool_guard.as_ref() else {
            return Ok(());
        };
        let native_len = start_pos + seq_len;

        if let Some(state) = self.compressor_state.lock().as_ref() {
            pool.save_residual_from_state(
                self.layer_idx,
                &state.kv_state,
                &state.score_state,
                native_len,
                block_table,
                false,
            )?;
        }
        if let Some(state) = self.indexer_state.lock().as_ref() {
            pool.save_residual_from_state(
                self.layer_idx,
                &state.compressor_state.kv_state,
                &state.compressor_state.score_state,
                native_len,
                block_table,
                true,
            )?;
            if state.compressed_len > 0 {
                if let Some(layer) = pool.layer(self.layer_idx) {
                    if let Some(ic) = &layer.indexer_compressed {
                        let rows_per_page = 256 / 4;
                        let n = state.compressed_len;
                        let mut abs_row = 0;
                        while abs_row < n {
                            let page_idx = abs_row / rows_per_page;
                            let row = abs_row % rows_per_page;
                            let room = rows_per_page - row;
                            let take = room.min(n - abs_row);
                            let Some(&page) = block_table.get(page_idx) else {
                                abs_row += take;
                                continue;
                            };
                            let page = page as usize;
                            if page >= pool.num_pages {
                                abs_row += take;
                                continue;
                            }
                            let src = state.kv_cache.narrow(0, abs_row, take)?.contiguous()?;
                            let dst = ic.narrow(0, page, 1)?.squeeze(0)?;
                            let hd = dst.dim(1)?;
                            attention_rs::deepseek_v4::copy_contiguous_into(&dst, &src, row * hd)?;
                            abs_row += take;
                        }
                    }
                }
            }
        }

        let sparse_guard = self.sparse_kv.lock();
        let Some(sparse) = sparse_guard.as_ref() else {
            return Ok(());
        };
        let win = sparse.sliding_window;
        let ratio = self.compression.ratio();

        if ratio > 0 && sparse.compressed_slots > 0 && sparse.compressed_len > 0 {
            if start_pos == 0 {
                let rows = sparse
                    .kv
                    .narrow(0, win, sparse.compressed_len)?
                    .contiguous()?;
                pool.write_compressed_rows(self.layer_idx, &rows, 0, block_table)?;
            } else {
                let first = start_pos / ratio;
                let end = sparse
                    .compressed_len
                    .max(first + 1)
                    .min(sparse.compressed_slots);
                if end > first {
                    let rows = sparse
                        .kv
                        .narrow(0, win + first, end - first)?
                        .contiguous()?;
                    pool.write_compressed_rows(self.layer_idx, &rows, first, block_table)?;
                }
            }
        }

        if let Some(token_kv) = token_kv {
            let n = token_kv.dim(0)?;
            if n != seq_len {
                candle_core::bail!("commit_pages token_kv len {n} != seq_len {seq_len}");
            }
            let abs_positions: Vec<i64> =
                (start_pos..start_pos + seq_len).map(|p| p as i64).collect();
            pool.write_swa_rows(self.layer_idx, token_kv, &abs_positions, block_table)?;
        } else {
            // Ring holds only `win` slots — write the chronological tail only.
            let n = seq_len.min(win);
            let abs_start = start_pos + seq_len - n;
            let abs_positions: Vec<i64> = (abs_start..abs_start + n).map(|p| p as i64).collect();
            let idxs: Vec<u32> = (abs_start..abs_start + n)
                .map(|p| (p % win) as u32)
                .collect();
            let idx_t = Tensor::from_vec(idxs, (n,), sparse.kv.device())?;
            let token_kv = sparse.kv.narrow(0, 0, win)?.index_select(&idx_t, 0)?;
            pool.write_swa_rows(self.layer_idx, &token_kv, &abs_positions, block_table)?;
        }
        pool.update_residual_freeze(native_len, block_table);
        Ok(())
    }

    /// Snapshot residual at every native-block boundary after a cold bulk prefill
    /// so later prefix hits can hand off (vLLM block-boundary residual).
    fn freeze_residuals_at_boundaries(
        &self,
        attn_normed: &Tensor,
        seq_len: usize,
        block_table: &[u32],
    ) -> Result<()> {
        if seq_len < V4_NATIVE_BLOCK_SIZE || block_table.is_empty() {
            return Ok(());
        }
        let pool_guard = self.hybrid_pool.lock();
        let Some(pool) = pool_guard.as_ref() else {
            return Ok(());
        };
        let mut boundary = V4_NATIVE_BLOCK_SIZE;
        while boundary <= seq_len {
            let xs = attn_normed.narrow(0, 0, boundary)?.contiguous()?;
            if let Some(compressor) = &self.compressor {
                let mut state = self.compressor_state.lock();
                let state = state.as_mut().expect("compressor state ensured");
                compressor.seed_decode_state_after_prefill(&xs, state, boundary)?;
                pool.save_residual_from_state(
                    self.layer_idx,
                    &state.kv_state,
                    &state.score_state,
                    boundary,
                    block_table,
                    false,
                )?;
            }
            if let Some(indexer) = &self.indexer {
                let mut istate = self.indexer_state.lock();
                let istate = istate.as_mut().expect("indexer state ensured");
                indexer.compressor.seed_decode_state_after_prefill(
                    &xs,
                    &mut istate.compressor_state,
                    boundary,
                )?;
                pool.save_residual_from_state(
                    self.layer_idx,
                    &istate.compressor_state.kv_state,
                    &istate.compressor_state.score_state,
                    boundary,
                    block_table,
                    true,
                )?;
            }
            pool.update_residual_freeze(boundary, block_table);
            boundary += V4_NATIVE_BLOCK_SIZE;
        }
        if let Some(compressor) = &self.compressor {
            let mut state = self.compressor_state.lock();
            let state = state.as_mut().expect("compressor state ensured");
            compressor.seed_decode_state_after_prefill(attn_normed, state, seq_len)?;
        }
        if let Some(indexer) = &self.indexer {
            let mut istate = self.indexer_state.lock();
            let istate = istate.as_mut().expect("indexer state ensured");
            indexer.compressor.seed_decode_state_after_prefill(
                attn_normed,
                &mut istate.compressor_state,
                seq_len,
            )?;
        }
        Ok(())
    }

    /// Write continued-prefill compressed rows (`prefill_continue` output) into
    /// the live sparse / indexer caches at absolute compressed-row indices.
    fn write_continue_compressed(
        &self,
        phase_a: &[Tensor],
        bulk: Option<&Tensor>,
        bulk_start_row: usize,
        start_pos: usize,
        ratio: usize,
        indexer: bool,
    ) -> Result<()> {
        if ratio == 0 {
            return Ok(());
        }
        // Overlap phase-A emits the row that closes the straddling block at
        // `start_pos / ratio` (only when start_pos % ratio != 0).
        let phase_a_row = start_pos / ratio;
        if indexer {
            let mut istate = self.indexer_state.lock();
            let istate = istate.as_mut().expect("indexer state");
            for emitted in phase_a {
                istate.write_compressed_at(emitted, phase_a_row)?;
            }
            if let Some(bulk) = bulk {
                istate.append_compressed_rows_at(bulk, bulk_start_row)?;
            }
        } else {
            let mut sparse = self.sparse_kv.lock();
            let sparse = sparse.as_mut().expect("sparse");
            for emitted in phase_a {
                sparse.write_compressed_row(emitted, phase_a_row)?;
            }
            if let Some(bulk) = bulk {
                sparse.write_compressed_rows_at(bulk, bulk_start_row)?;
            }
        }
        Ok(())
    }

    /// Fused continued prefill (vLLM chrono-gather path).
    ///
    /// Layout matches cold attend: `[chrono SWA gather | compressed]`.
    /// `gather_len = query_len + min(prefix_len, window-1)` — no ring overwrite
    /// before attend, no per-token decode loop, one `sparse_attn`.
    fn prefill_continue_fused(
        &mut self,
        attn_normed: &Tensor,
        qr: &Tensor,
        positions: &Tensor,
        start_pos: usize,
        query_len: usize,
        block_table: &[u32],
    ) -> Result<Tensor> {
        let device = attn_normed.device();
        let win = self.sliding_window;
        let head_dim = self.self_attn.get_head_dim();
        let prefix_win = start_pos.min(win.saturating_sub(1));
        let gather_start = start_pos - prefix_win;
        let gather_len = prefix_win + query_len;

        // 1) Chrono SWA prefix from the live ring (before writing new rows).
        let prev = if prefix_win > 0 {
            let sparse = self.sparse_kv.lock();
            let sparse = sparse.as_ref().expect("sparse cache ensured");
            sparse.gather_chrono_window(start_pos, prefix_win)?
        } else {
            Tensor::zeros((0, head_dim), DType::BF16, device)?
        };

        // 2) Batched token KV for the uncached chunk.
        let kv = self.self_attn.wkv_forward(attn_normed)?.contiguous()?;
        self.rope.apply_from_positions(&kv, positions, 0, false)?;
        attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
            &kv,
            1,
            head_dim,
            self.qk_rope_head_dim,
            64,
        )?;

        let window_portion = if prefix_win > 0 {
            Tensor::cat(&[&prev, &kv], 0)?.contiguous()?
        } else {
            kv.contiguous()?
        };

        // 3) Continue-compress (overlap Phase-A is tiny; bulk is fused CUDA).
        let bufs_guard = self.decode_buffers.lock();
        let bufs = bufs_guard.as_ref().expect("decode buffers ensured");
        if let Some(compressor) = &self.compressor {
            let weighted = bufs
                .compressor_weighted
                .as_ref()
                .expect("compressor weighted buffer");
            let cout = bufs.compressor_out.as_ref().expect("compressor out buffer");
            let (phase_a, bulk, bulk_start) = {
                let mut state = self.compressor_state.lock();
                let state = state.as_mut().expect("compressor state ensured");
                compressor.prefill_continue(
                    attn_normed,
                    state,
                    positions,
                    weighted,
                    cout,
                    Some(&self.rope),
                    false,
                    start_pos,
                )?
            };
            self.write_continue_compressed(
                &phase_a,
                bulk.as_ref(),
                bulk_start,
                start_pos,
                compressor.ratio,
                false,
            )?;
        }
        if let Some(indexer) = &self.indexer {
            let weighted = bufs
                .indexer_compressor_weighted
                .as_ref()
                .expect("indexer compressor weighted");
            let cout = bufs
                .indexer_compressor_out
                .as_ref()
                .expect("indexer compressor out");
            let (phase_a, bulk, bulk_start) = {
                let mut istate = self.indexer_state.lock();
                let istate = istate.as_mut().expect("indexer state ensured");
                indexer.compressor.prefill_continue(
                    attn_normed,
                    &mut istate.compressor_state,
                    positions,
                    weighted,
                    cout,
                    Some(&self.rope),
                    true,
                    start_pos,
                )?
            };
            self.write_continue_compressed(
                &phase_a,
                bulk.as_ref(),
                bulk_start,
                start_pos,
                indexer.compressor.ratio,
                true,
            )?;
        }
        drop(bufs_guard);

        // Default: vLLM chrono-gather batch attend (fast + correct after
        // gather_ring_chrono + RoPE-from-pos fixes).
        // XINFER_V4_CONTINUE=hybrid — fused compress + sequential attend
        // XINFER_V4_CONTINUE=loop — golden per-token decode_one_token
        let use_chrono = match std::env::var("XINFER_V4_CONTINUE") {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                !(v == "hybrid" || v == "loop" || v == "0" || v == "false")
            }
            Err(_) => true,
        };
        if !use_chrono {
            let mut outs: Vec<Tensor> = Vec::with_capacity(query_len);
            for t in 0..query_len {
                let kv_t = kv.narrow(0, t, 1)?.contiguous()?;
                let xs = attn_normed.narrow(0, t, 1)?.contiguous()?;
                let qr_t = qr.narrow(0, t, 1)?.contiguous()?;
                let pos = positions.narrow(0, t, 1)?.contiguous()?;
                {
                    let mut sparse = self.sparse_kv.lock();
                    let sparse = sparse.as_mut().expect("sparse cache ensured");
                    sparse.write_window_token_from_pos(&kv_t, &pos)?;
                }
                outs.push(self.decode_attend_only(&xs, &qr_t, &pos)?);
            }
            self.commit_pages(start_pos, query_len, block_table, Some(&kv))?;
            let refs: Vec<&Tensor> = outs.iter().collect();
            return Ok(Tensor::cat(&refs, 0)?);
        }

        // 4) Chrono batch attend workspace `[SWA gather | compressed]` (cold layout).
        let (compressed_len, compress_ratio) = {
            let sparse = self.sparse_kv.lock();
            let sparse = sparse.as_ref().expect("sparse cache ensured");
            (sparse.compressed_len, sparse.compress_ratio.max(1))
        };
        let (kv_combined, kv_len) = if compressed_len > 0 {
            let compressed = {
                let sparse = self.sparse_kv.lock();
                let sparse = sparse.as_ref().expect("sparse cache ensured");
                sparse.kv.narrow(0, win, compressed_len)?.contiguous()?
            };
            let total = gather_len + compressed_len;
            let combined = Tensor::zeros((total, head_dim), DType::BF16, device)?;
            attention_rs::deepseek_v4::copy_contiguous_into(&combined, &window_portion, 0)?;
            attention_rs::deepseek_v4::copy_contiguous_into(
                &combined,
                &compressed,
                gather_len * head_dim,
            )?;
            (combined, total)
        } else {
            (window_portion, gather_len.max(1))
        };

        // 5) Position-aware topk into the gather workspace.
        let window_topk = attention_rs::deepseek_v4::window_topk_indices_chrono_from_pos(
            positions,
            win,
            query_len,
            gather_start,
            gather_len,
        )?;
        let mut total_topk = win;
        let mut topk_idxs = window_topk;

        if compressed_len > 0 && self.compressor.is_some() {
            let compress_idxs = if let Some(indexer) = &self.indexer {
                let state = self.indexer_state.lock();
                let state = state.as_ref().expect("indexer state ensured");
                let score_len = state.compressed_len.max(1).min(state.max_compressed_len);
                let scores = indexer.scores_prefill_against_cache(
                    attn_normed,
                    qr,
                    &state.kv_cache,
                    score_len,
                    &self.rope,
                    positions,
                )?;
                attention_rs::deepseek_v4::indexer_mask_scores_prefill_by_pos(
                    &scores,
                    positions,
                    indexer.compressor.ratio,
                )?;
                indexer.topk_prefill_from_pos(&scores, positions, score_len, gather_len)?
            } else {
                attention_rs::deepseek_v4::compress_topk_indices_prefill_from_pos(
                    positions,
                    compressed_len,
                    gather_len,
                    compress_ratio,
                    query_len,
                )?
            };
            let compress_topk = compress_idxs.dim(1)?;
            topk_idxs = attention_rs::deepseek_v4::concat_topk_indices(
                &topk_idxs,
                &compress_idxs,
                query_len,
                win,
                compress_topk,
            )?;
            total_topk += compress_topk;
        }

        // 6) One batched sparse attend over the whole uncached chunk.
        let num_heads = self.self_attn.get_num_heads();
        let attn_out = Tensor::zeros((query_len, num_heads, head_dim), DType::BF16, device)?;
        let out = self.self_attn.sparse_attn_from_positions(
            attn_normed,
            qr,
            &self.rope,
            positions,
            &kv_combined,
            &topk_idxs,
            &attn_out,
            kv_len,
            total_topk,
            None,
        )?;

        // 7) Commit ring + pages once (pages remain source of truth).
        {
            let mut sparse = self.sparse_kv.lock();
            let sparse = sparse.as_mut().expect("sparse cache ensured");
            sparse.write_window_rows_from_pos(&kv, positions)?;
        }
        self.commit_pages(start_pos, query_len, block_table, Some(&kv))?;
        Ok(out)
    }

    /// Unified prefill (vLLM-matched continue):
    /// - cold: batched `prefill_fresh` + full-KV page commit
    /// - continue: fused chrono-gather + `prefill_continue` + one `sparse_attn`
    ///   (no per-token decode loop; no cached-prefix recompute)
    fn prefill_unified(
        &mut self,
        attn_normed: &Tensor,
        qr: &Tensor,
        positions: &Tensor,
        start_pos: usize,
        query_len: usize,
        block_table: &[u32],
    ) -> Result<Tensor> {
        self.ensure_decode_buffers(attn_normed.device())?;

        if start_pos == 0 {
            let (out, kv) = self.prefill_fresh(attn_normed, qr, query_len)?;
            self.commit_pages(0, query_len, block_table, Some(&kv))?;
            if query_len >= V4_NATIVE_BLOCK_SIZE {
                self.freeze_residuals_at_boundaries(attn_normed, query_len, block_table)?;
                self.commit_pages(0, query_len, block_table, Some(&kv))?;
            }
            return Ok(out);
        }

        let ratio = self.compression.ratio(); // 0 = SWA-only
                                              // Same-request chunked: keep live compressor/sparse (matches working decode path).
                                              // Prefix-cache hit across requests: hydrate when live does not cover.
        let live_covers = {
            let sparse = self.sparse_kv.lock();
            let r = ratio.max(1);
            sparse
                .as_ref()
                .map(|s| ratio > 0 && s.compressed_len.saturating_mul(r) >= start_pos)
                .unwrap_or(false)
        };
        if start_pos % V4_NATIVE_BLOCK_SIZE == 0 && !live_covers {
            self.load_residual_from_pages(start_pos, block_table)?;
        }

        // `XINFER_V4_CONTINUE`:
        //   unset/chrono — fused compress + chrono-gather batch attend (default)
        //   hybrid — fused compress + sequential attend
        //   loop — golden per-token decode_one_token
        let mode = std::env::var("XINFER_V4_CONTINUE")
            .unwrap_or_else(|_| "chrono".into())
            .to_ascii_lowercase();
        if mode == "loop" {
            let mut outs: Vec<Tensor> = Vec::with_capacity(query_len);
            for t in 0..query_len {
                let xs = attn_normed.narrow(0, t, 1)?.contiguous()?;
                let qr_t = qr.narrow(0, t, 1)?.contiguous()?;
                let pos = positions.narrow(0, t, 1)?.contiguous()?;
                outs.push(self.decode_one_token(&xs, &qr_t, &pos, start_pos + t, block_table)?);
            }
            let refs: Vec<&Tensor> = outs.iter().collect();
            return Ok(Tensor::cat(&refs, 0)?);
        }

        self.prefill_continue_fused(
            attn_normed,
            qr,
            positions,
            start_pos,
            query_len,
            block_table,
        )
    }

    pub fn forward(
        &mut self,
        hc_hidden: &HcHiddenStates,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        _cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        input_ids: Option<&Tensor>,
        block_table: &[u32],
        start_pos: usize,
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
        let attn_output = if input_metadata.is_prefill {
            self.prefill_unified(
                &attn_normed,
                &qr,
                positions,
                start_pos,
                seq_len,
                block_table,
            )?
        } else {
            if seq_len != 1 {
                candle_core::bail!("DeepSeek V4 sparse decode expects one token per request");
            }
            if start_pos >= self.max_seq_len {
                candle_core::bail!(
                    "DeepSeek V4 decode position {start_pos} exceeds sparse cache capacity {}",
                    self.max_seq_len
                );
            }
            // Realign the live ring with the pages at every native block. The
            // incremental decode path advances `compressed_len` on its own, but
            // `commit_pages` places compressed rows by that counter: letting it
            // drift writes misaligned rows into the pages. A later prefix-cache
            // hit reads those pages back, so the drift only surfaces on reuse.
            if start_pos > 0 && start_pos % V4_NATIVE_BLOCK_SIZE == 0 {
                self.load_residual_from_pages(start_pos, block_table)?;
            }
            self.decode_one_token(&attn_normed, &qr, positions, start_pos, block_table)?
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
    hybrid_pool: Arc<Mutex<Option<V4HybridPagePool>>>,
    request_states: Mutex<HashMap<usize, RequestState>>,
    /// State of the most recently finished request, kept so the next turn of a
    /// conversation can adopt it on a prefix-cache hit instead of rebuilding
    /// from page residual. One slot only: this is the same amount of recurrent
    /// state a single-sequence run already keeps resident.
    retired_state: Mutex<Option<RequestState>>,
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

        // Private sparse / indexer / compressor caches must NOT be sized to the
        // full `max_position_embeddings` (often 1M). That pre-allocates multi-GB
        // per GPU and makes prefix snapshots OOM after the first chunk.
        // Prefer engine `max_model_len` when set; otherwise cap to 128k (still
        // above default prefill chunk × several steps). RoPE tables share this.
        let sparse_max_seq_len = config
            .max_model_len
            .unwrap_or(131_072)
            .min(config.max_position_embeddings.max(1))
            .max(16_384);
        if sparse_max_seq_len < config.max_position_embeddings {
            crate::log_warn!(
                "DeepSeek V4 sparse KV capacity capped at {} tokens (model max_position_embeddings={}); set --max-model-len to raise.",
                sparse_max_seq_len,
                config.max_position_embeddings
            );
        }
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
        let hybrid_pool = Arc::new(Mutex::new(None));
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
                hybrid_pool.clone(),
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
            &config.quantization_config,
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
            hybrid_pool,
            request_states: Mutex::new(HashMap::new()),
            retired_state: Mutex::new(None),
        })
    }

    pub fn hybrid_pool_arc(&self) -> Arc<Mutex<Option<V4HybridPagePool>>> {
        self.hybrid_pool.clone()
    }

    /// Move the engine-allocated pool into the model's shared Arc (layers already
    /// hold clones of this Arc). Caller should then store `hybrid_pool_arc()` in
    /// [`GpuKvCache::DeepSeekV4`] so engine and model share one allocation.
    pub fn attach_hybrid_page_pool(&self, pool: Arc<Mutex<Option<V4HybridPagePool>>>) {
        let mut src = pool.lock();
        *self.hybrid_pool.lock() = src.take();
    }

    pub fn clear_seq_state(&self, seq_id: usize) {
        // Retire rather than drop: the next turn of a conversation prefix-hits
        // this sequence's pages, and its recurrent state is the exact state
        // that turn resumes from.
        if let Some(state) = self.request_states.lock().remove(&seq_id) {
            *self.retired_state.lock() = Some(state);
        }
    }

    /// No-op: decode write-through already persists into engine-owned pages.
    pub fn sync_hybrid_pages_after_decode(
        &self,
        _block_table: &[u32],
        _native_len: usize,
    ) -> Result<()> {
        Ok(())
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
        // The V4 kernels accept many query rows, but the compressor and indexer
        // keep one recurrent state machine. Dispatch one request at a time and
        // park that state by sequence id: the scheduler gets a batched API
        // without ever mixing two histories in the same recurrent state.
        let xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };
        let _ = kv_caches;

        let token_count = xs.dim(0)?;
        let request_lens = if input_metadata.is_prefill {
            input_metadata
                .seqlens
                .as_ref()
                .filter(|lengths| !lengths.is_empty())
                // Runner metadata stores the ends of each cumulative query range
                // (`cu_seqlens_q[1..]`); this dispatcher needs per-request lengths.
                .map(|ends| {
                    let mut previous = 0usize;
                    ends.iter()
                        .map(|&end| {
                            let end = end as usize;
                            let len = end.saturating_sub(previous);
                            previous = end;
                            len
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![token_count])
        } else {
            vec![1; token_count]
        };
        if request_lens.iter().sum::<usize>() != token_count {
            candle_core::bail!(
                "DeepSeek V4 request lengths {:?} do not cover {} input rows",
                request_lens,
                token_count
            );
        }

        let request_ids = input_metadata
            .sequence_ids
            .clone()
            .unwrap_or_else(|| (0..request_lens.len()).collect());
        if request_ids.len() != request_lens.len() {
            candle_core::bail!(
                "DeepSeek V4 received {} request ids for {} requests",
                request_ids.len(),
                request_lens.len()
            );
        }
        let context_lens = input_metadata
            .context_lens_host
            .clone()
            .unwrap_or_else(|| vec![input_metadata.max_context_len as u32; request_lens.len()]);
        if context_lens.len() != request_lens.len() {
            candle_core::bail!(
                "DeepSeek V4 received {} context lengths for {} requests",
                context_lens.len(),
                request_lens.len()
            );
        }

        // Host block tables are prepared in runner `prepare_prefill` /
        // `prepare_decode` (CPU Vec — no D2H in the forward).
        let empty_block_table: Vec<u32> = Vec::new();
        let block_tables = input_metadata.block_tables_host.as_ref();

        let mut outputs = Vec::with_capacity(request_lens.len());
        let mut offset = 0usize;
        for (request_idx, (&seq_len, &seq_id)) in
            request_lens.iter().zip(request_ids.iter()).enumerate()
        {
            let input_slice = xs.narrow(0, offset, seq_len)?.contiguous()?;
            let token_id_slice = if !embeded_inputs {
                Some(input_ids.narrow(0, offset, seq_len)?.contiguous()?)
            } else {
                None
            };
            let position_slice = positions.narrow(0, offset, seq_len)?.contiguous()?;
            let block_table: &[u32] = block_tables
                .and_then(|bts| bts.get(request_idx))
                .map(|v| v.as_slice())
                .unwrap_or(&empty_block_table);
            let context_len = context_lens[request_idx] as usize;
            if context_len < seq_len {
                candle_core::bail!(
                    "DeepSeek V4 request {seq_id} context length {context_len} < query length {seq_len}"
                );
            }
            // Unified metadata: start_pos = number of already-cached tokens.
            let start_pos = context_len - seq_len;

            // Only reuse parked state that ends exactly where this request
            // resumes. A prefix-cache hit or a recompute after preemption moves
            // the resume point without finishing the sequence, and that state
            // describes a different history than the current block table.
            let mut parked_state = self
                .request_states
                .lock()
                .remove(&seq_id)
                .filter(|state| state.covers(start_pos, block_table));
            if parked_state.is_none() && start_pos > 0 {
                let mut retired = self.retired_state.lock();
                if retired
                    .as_ref()
                    .map_or(false, |state| state.covers(start_pos, block_table))
                {
                    parked_state = retired.take();
                }
            }
            let mut parked =
                parked_state.map(|state| state.layers.into_iter().map(Some).collect::<Vec<_>>());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let snapshot = parked
                    .as_mut()
                    .and_then(|snapshots| snapshots.get_mut(layer_idx))
                    .and_then(Option::take);
                layer
                    .lock()
                    .restore_state(snapshot, start_pos, block_table, &self.device)?;
            }

            let mut hc_hidden = hc_expand(&input_slice, self.v4_cfg.hc_mult)?;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer = layer.lock();
                let layer_input_ids = if layer_idx < self.v4_cfg.n_hash_layers {
                    token_id_slice.as_ref()
                } else {
                    None
                };
                hc_hidden = layer.forward(
                    &hc_hidden,
                    None,
                    &position_slice,
                    None,
                    input_metadata,
                    layer_input_ids,
                    block_table,
                    start_pos,
                )?;
            }

            // HC head collapse
            let hidden = if let Some(hc_head_weights) = &self.hc_head {
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
            let last = if input_metadata.is_prefill {
                hidden.narrow(0, seq_len.saturating_sub(1), 1)?
            } else {
                hidden
            };
            // Final RMSNorm: owned buffer, single in-place CUDA kernel.
            let last = if last.dtype() == DType::BF16 && last.is_contiguous() {
                last
            } else {
                last.to_dtype(DType::BF16)?.contiguous()?
            };
            self.norm.forward_v4_inplace(&last)?;
            outputs.push(self.lm_head.forward(&last)?);

            let snapshots = self
                .layers
                .iter()
                .map(|layer| layer.lock().take_state())
                .collect::<Vec<_>>();
            let next_pos = start_pos + seq_len;
            let pages = next_pos
                .div_ceil(V4_NATIVE_BLOCK_SIZE)
                .min(block_table.len());
            self.request_states.lock().insert(
                seq_id,
                RequestState {
                    next_pos,
                    block_prefix: block_table[..pages].to_vec(),
                    layers: snapshots,
                },
            );
            offset += seq_len;
        }

        if outputs.len() == 1 {
            return Ok(outputs.remove(0));
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&output_refs, 0)
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
