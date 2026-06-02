use crate::models::gemma3::Gemma3ForConditionalGeneration;
use crate::models::gemma4::Gemma4ForCausalLM;
// src/core/runner.rs
use crate::models::layers::distributed::Comm;
use crate::models::layers::linear::set_linear_is_prefill;
use crate::models::layers::VarBuilderX;
use crate::server::EmbeddingStrategy;
use crate::transfer::Transfer;
#[cfg(all(feature = "cuda", feature = "graph"))]
use crate::utils::graph::{
    planned_graph_capture_batches, CudaGraphFn, CudaGraphWrapper, GraphCapturer, ModelFn,
};
use crate::utils::guidance::{GuidanceState, ParserFactory};
use crate::utils::image::compute_image_slice;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use crate::utils::progress::ProgressLike;
#[cfg(feature = "flashinfer")]
use crate::utils::FlashInferKvParams;
use crate::{
    core::sequence::{DecodeSequence, Sequence, ToDecodeInput},
    models::deepseek3::DeepSeekForCausalLM,
    models::glm4::GLM4ForCausalLM,
    models::glm4_moe::GLM4MoEForCausalLM,
    models::glm4_moe_lite::GLM4MoeLiteForCausalLM,
    models::llama::LLaMaForCausalLM,
    models::llama4::LLama4ForConditionalGeneration,
    models::minimax::MiniMaxForCausalLM,
    models::mistral3_vl::Mistral3ForConditionalGeneration,
    models::phi4::Phi4ForCausalLM,
    models::qwen3::Qwen3ForCausalLM,
    models::qwen3_5::Qwen3_5ForCausalLM,
    models::qwen3_5_moe::Qwen3_5MoEForCausalLM,
    models::qwen3_moe::Qwen3MoEForCausalLM,
    models::qwen3_vl::Qwen3VLForConditionalGeneration,
    utils::config::{Config, EngineConfig, ModelType, SamplingParams},
    utils::kvcache_allocator::KVCacheAllocator,
};
use attention_rs::cache;
#[cfg(feature = "flashinfer")]
use attention_rs::FlashInferMetadata;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use interprocess::local_socket::Stream as LocalStream;
use parking_lot::RwLock;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use toktrie::SimpleVob;

/// Cached sampling parameters computed once during prefill, reused during decode
#[derive(Clone, Debug)]
pub struct CachedSamplingParams {
    pub sampling: Sampling,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

pub enum Seqs<'a> {
    SeqRefs(&'a [&'a Sequence]),
    DecodeVec(&'a Vec<DecodeSequence>),
}

fn sampling_params_for_batch_index<'a>(seqs: &'a Seqs<'a>, index: usize) -> &'a SamplingParams {
    match seqs {
        Seqs::SeqRefs(refs) => &refs[index].sampling_params,
        Seqs::DecodeVec(vec) => &vec[index].sampling_params,
    }
}

fn collect_guided_batch_entries(seqs: &Seqs<'_>, seq_ids: &[usize]) -> Vec<(usize, usize)> {
    seq_ids
        .iter()
        .enumerate()
        .filter_map(|(index, seq_id)| {
            let sampling_params = sampling_params_for_batch_index(seqs, index);
            sampling_params.grammar.as_ref().map(|_| (index, *seq_id))
        })
        .collect()
}

pub enum Model {
    Qwen3(Arc<Qwen3ForCausalLM>),
    Qwen3MoE(Arc<Qwen3MoEForCausalLM>),
    Qwen3_5(Arc<Qwen3_5ForCausalLM>),
    Qwen3_5MoE(Arc<Qwen3_5MoEForCausalLM>),
    LLaMa(Arc<LLaMaForCausalLM>),
    LLaMa4(Arc<LLama4ForConditionalGeneration>),
    Phi4(Arc<Phi4ForCausalLM>),
    GLM4(Arc<GLM4ForCausalLM>),
    GLM4MoE(Arc<GLM4MoEForCausalLM>),
    GLM4MoeLite(Arc<GLM4MoeLiteForCausalLM>),
    DeepSeek(Arc<DeepSeekForCausalLM>),
    Mistral3VL(Arc<Mistral3ForConditionalGeneration>),
    Gemma3(Arc<Gemma3ForConditionalGeneration>),
    Gemma4(Arc<Gemma4ForCausalLM>),
    Qwen3VL(Arc<Qwen3VLForConditionalGeneration>),
    MiniMax(Arc<MiniMaxForCausalLM>),
}

pub enum RunnerType {
    Thread(ModelRunner),
    Process(Vec<LocalStream>),
}

pub struct CpuTqLayerCache {
    pub k_absmax: Option<Tensor>,
    pub k_quant: Option<Tensor>,
    pub v_absmax: Tensor,
    pub v_quant: Tensor,
}

pub struct ModelRunner {
    model: Model,
    gpu_kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    cpu_kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    cpu_tq_cache: Option<Vec<CpuTqLayerCache>>,
    device: Device,
    config: EngineConfig,
    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub capturer: GraphCapturer<CudaGraphWrapper<CudaGraphFn>>,
    #[cfg(feature = "flashinfer")]
    flashinfer_kv_params: Option<FlashInferKvParams>,
    logit_processor: LogitsProcessor,
    /// Cached sampling strategy computed once during prefill, reused during decode
    cached_sampling: RwLock<Option<CachedSamplingParams>>,
    seq_tokens: RwLock<HashMap<usize, Vec<u32>>>,
    restored_prefix_sequences: RwLock<HashSet<usize>>,
    guidance_states: RwLock<HashMap<usize, GuidanceState>>,
    guidance_failed: RwLock<HashSet<usize>>,
    guidance_mismatch: RwLock<HashSet<usize>>,
    llg_factory: Option<Arc<ParserFactory>>,
    transfer: Option<Arc<Transfer>>,
    /// Whether this runner is on the first rank (for logging)
    is_first_rank: bool,
    model_type: ModelType,
}

impl ModelRunner {
    // Mamba slots track concurrent sequence states (not KV token blocks).
    const MAMBA_CACHE_FIXED_CAPACITY: usize = 64;
    #[cfg(all(feature = "cuda", feature = "graph"))]
    const GRAPH_CAPTURE_MIN_BATCH: usize = 16;

    fn is_mla_model(&self) -> bool {
        matches!(
            self.model_type,
            ModelType::GLM4MoeLite | ModelType::DeepSeek
        )
    }

    fn prepare_mamba_slot_mapping(
        &self,
        sequence_ids: &[usize],
        is_prefill: bool,
    ) -> Result<Option<Tensor>> {
        let slots = match &self.model {
            Model::Qwen3_5(model) => Some(if is_prefill {
                model.ensure_mamba_slots_for_sequences(sequence_ids)?
            } else {
                model.get_mamba_slots_for_sequences(sequence_ids)?
            }),
            Model::Qwen3_5MoE(model) => Some(if is_prefill {
                model.ensure_mamba_slots_for_sequences(sequence_ids)?
            } else {
                model.get_mamba_slots_for_sequences(sequence_ids)?
            }),
            Model::Qwen3VL(model) => {
                if is_prefill {
                    model.ensure_mamba_slots_for_sequences(sequence_ids)?
                } else {
                    model.get_mamba_slots_for_sequences(sequence_ids)?
                }
            }
            _ => None,
        };
        if let Some(slots) = slots {
            let slots_i64 = slots.iter().map(|&s| s as i64).collect::<Vec<_>>();
            let len = slots_i64.len();
            Ok(Some(Tensor::from_vec(slots_i64, (len,), &self.device)?))
        } else {
            Ok(None)
        }
    }

    fn effective_mamba_prefix_capacity(
        prefix_cache_enabled: bool,
        mamba_cache_capacity: usize,
    ) -> usize {
        if !prefix_cache_enabled || mamba_cache_capacity == 0 {
            return 0;
        }
        // Keep a larger snapshot pool than active slots so prompt/chunk-prefill
        // boundaries survive decode-time snapshot churn when prefix cache is hot.
        mamba_cache_capacity.saturating_mul(2)
    }

    fn apply_requested_guidance(
        &self,
        logits: &Tensor,
        seqs: &Seqs<'_>,
        seq_ids: &[usize],
    ) -> Result<(Tensor, Option<HashSet<usize>>)> {
        let guided_entries = collect_guided_batch_entries(seqs, seq_ids);
        if guided_entries.is_empty() {
            return Ok((logits.clone(), None));
        }

        let Some(factory) = &self.llg_factory else {
            return Ok((logits.clone(), None));
        };

        let mut guidance_states = self.guidance_states.write();
        let mut guidance_failed = self.guidance_failed.write();
        let mut guidance_mismatch = self.guidance_mismatch.write();
        let mut modified = false;
        let vocab_size = logits.dim(1)?;

        let mut masks: Vec<(usize, usize, SimpleVob)> = Vec::new();
        let mut failed_seq_ids = Vec::new();
        let mut guided_seq_ids = HashSet::new();

        for (i, id) in seq_ids.iter().enumerate() {
            if guided_entries
                .iter()
                .any(|(guided_idx, _)| *guided_idx == i)
            {
                continue;
            }
            let _ = guidance_states.remove(id);
            let _ = guidance_failed.remove(id);
            let _ = guidance_mismatch.remove(id);
        }

        for (i, id) in guided_entries {
            if guidance_failed.contains(&id) {
                continue;
            }

            let grammar = sampling_params_for_batch_index(seqs, i)
                .grammar
                .as_ref()
                .expect("guided batch entries must have a grammar");

            let state = match guidance_states.entry(id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    match GuidanceState::new_from_grammar(factory.clone(), grammar) {
                        Ok(state) => entry.insert(state),
                        Err(err) => {
                            guidance_failed.insert(id);
                            crate::log_warn!(
                            "[Seq {}] Failed to create guidance state: {}. Disabling constraints for this sequence.",
                            id,
                            err
                        );
                            continue;
                        }
                    }
                }
            };

            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    masks.push((i, id, mask));
                    guided_seq_ids.insert(id);
                    modified = true;
                }
                Err(err) => {
                    if guidance_failed.insert(id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to compute guidance mask: {}. Disabling constraints for this sequence.",
                            id,
                            err
                        );
                    }
                    failed_seq_ids.push(id);
                }
            }
        }

        for seq_id in &failed_seq_ids {
            let _ = guidance_states.remove(seq_id);
        }

        if !modified {
            return Ok((logits.clone(), Some(guided_seq_ids)));
        }

        let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        for (seq_idx, seq_id, mask) in masks {
            let start = seq_idx * vocab_size;
            let end = start + vocab_size;
            let row = &mut logits_vec[start..end];
            let mask_len = mask.len();

            if mask_len == 0 {
                if guidance_failed.insert(seq_id) {
                    crate::log_warn!(
                        "[Seq {}] Guidance mask length is 0. Disabling constraints for this sequence.",
                        seq_id
                    );
                }
                failed_seq_ids.push(seq_id);
                continue;
            }

            if mask_len != vocab_size {
                if guidance_mismatch.insert(seq_id) {
                    crate::log_warn!(
                        "[Seq {}] Guidance mask size {} does not match vocab size {}. Clamping mask application.",
                        seq_id,
                        mask_len,
                        vocab_size
                    );
                }
            }

            let apply_len = std::cmp::min(vocab_size, mask_len);
            for tok in 0..apply_len {
                if !mask.is_allowed(tok as u32) {
                    row[tok] = f32::NEG_INFINITY;
                }
            }
            if mask_len < vocab_size {
                for tok in mask_len..vocab_size {
                    row[tok] = f32::NEG_INFINITY;
                }
            }
        }

        for seq_id in &failed_seq_ids {
            let _ = guidance_states.remove(seq_id);
        }

        Ok((
            Tensor::from_vec(logits_vec, logits.shape(), &self.device)?,
            Some(guided_seq_ids),
        ))
    }

    fn sample_processed_logits(&self, logits: &Tensor, sampling: &Sampling) -> Result<Vec<u32>> {
        self.logit_processor.sample_with_strategy(logits, sampling)
    }

    fn commit_guided_tokens(
        &self,
        seq_ids: &[usize],
        tokens: &[u32],
        guided_seq_ids: Option<HashSet<usize>>,
    ) {
        let Some(guided_seq_ids) = guided_seq_ids else {
            return;
        };

        let mut guidance_states = self.guidance_states.write();
        let mut guidance_failed = self.guidance_failed.write();
        for (seq_idx, seq_id) in seq_ids.iter().enumerate() {
            if !guided_seq_ids.contains(seq_id) || guidance_failed.contains(seq_id) {
                continue;
            }

            if let Some(state) = guidance_states.get_mut(seq_id) {
                if state.is_finished() {
                    continue;
                }

                let token = tokens[seq_idx];
                if let Err(err) = state.commit_token(token) {
                    if guidance_failed.insert(*seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to commit guided token {}: {}. Disabling constraints for this sequence.",
                            seq_id,
                            token,
                            err
                        );
                    }
                    let _ = guidance_states.remove(seq_id);
                }
            }
        }
    }

    #[allow(unused)]
    pub fn new(
        model_type: ModelType,
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        econfig: &mut EngineConfig,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: Device,
        reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
        transfer: Option<Arc<Transfer>>,
        llg_factory: Option<Arc<ParserFactory>>,
        stream: Option<LocalStream>,
    ) -> Result<Self> {
        attention_rs::reset_paged_attention_layer_counter();
        let model = crate::build_model!(
            model_type,
            vb,
            comm,
            config,
            dtype,
            is_rope_i,
            &device,
            reporter,
            {
                Qwen3 => Qwen3ForCausalLM,
                Qwen3MoE => Qwen3MoEForCausalLM,
                Qwen3_5 => Qwen3_5ForCausalLM,
                Qwen3_5MoE => Qwen3_5MoEForCausalLM,
                LLaMa => LLaMaForCausalLM,
                LLaMa4 => LLama4ForConditionalGeneration,
                Phi4 => Phi4ForCausalLM,
                GLM4 => GLM4ForCausalLM,
                GLM4MoE => GLM4MoEForCausalLM,
                GLM4MoeLite => GLM4MoeLiteForCausalLM,
                DeepSeek => DeepSeekForCausalLM,
                Mistral3VL => Mistral3ForConditionalGeneration,
                Gemma3 => Gemma3ForConditionalGeneration,
                Gemma4 => Gemma4ForCausalLM,
                Qwen3VL => Qwen3VLForConditionalGeneration,
                MiniMax => MiniMaxForCausalLM,
            }
        )?;

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let wrapper = crate::graph_wrapper!(
            &model,
            device,
            {
                Qwen3 => EmbedInputs,
                Qwen3MoE => EmbedInputs,
                Qwen3_5 => EmbedInputs,
                Qwen3_5MoE => EmbedInputs,
                LLaMa => EmbedInputs,
                LLaMa4 => NoneArg,
                Phi4 => EmbedInputs,
                GLM4 => EmbedInputs,
                GLM4MoE => EmbedInputs,
                GLM4MoeLite => EmbedInputs,
                DeepSeek => EmbedInputs,
                Mistral3VL => NoneArg,
                Gemma3 => NoneArg,
                Gemma4 => EmbedInputs,
                Qwen3VL => NoneArg,
                MiniMax => EmbedInputs,
            }
        );

        let allocator = if let Some(s) = stream {
            use crate::runner::{receive_local, send_local, MessageType};
            use interprocess::TryClone;
            send_local(
                &mut vec![s.try_clone()?],
                &MessageType::InitAck(true),
                false,
            )?;
            let msg = receive_local(&mut s.try_clone()?, true)?;
            if let MessageType::UsableMemoryLeft(ecfg) = msg {
                *econfig = ecfg.clone(); // Update Engine config
            }
            KVCacheAllocator::new(econfig, config, dtype)
        } else {
            let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
            econfig.kvcache_dtype = allocator.resolved_kvcache_dtype();
            let device_ids = econfig.device_ids.clone().unwrap_or(vec![0]);
            match allocator.plan(&device_ids, econfig) {
                Ok(_) => {
                    crate::log_info!("KVCache allocation successfully planned!");
                }
                Err(e) => {
                    candle_core::bail!("KVCache allocation failed: {}", e);
                }
            }
            allocator
        };

        let allocation = crate::utils::kvcache_allocator::KVCacheAllocation {
            num_gpu_blocks: econfig.num_blocks,
            #[cfg(feature = "cuda")]
            num_cpu_blocks: (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.2))
                as usize,
            #[cfg(not(feature = "cuda"))]
            num_cpu_blocks: 1, // dummy for non-CUDA platform
            max_num_seqs: econfig.max_num_seqs,
            max_model_len: econfig.max_model_len.unwrap_or(32768),
            kvcache_memory_bytes: econfig.kvcache_memory_bytes,
            max_num_batched_tokens: econfig.max_num_batched_tokens,
        };

        let is_hybrid_mamba_model = match &model {
            Model::Qwen3_5(_) | Model::Qwen3_5MoE(_) => true,
            Model::Qwen3VL(m) => m.uses_hybrid_mamba_text_model(),
            _ => false,
        };
        let mamba_cache_capacity = if is_hybrid_mamba_model {
            econfig
                .mamba_cache_capacity
                .unwrap_or_else(|| Self::MAMBA_CACHE_FIXED_CAPACITY.max(econfig.max_num_seqs))
        } else {
            0
        };

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let graph_capture_max_num_seqs = if is_hybrid_mamba_model {
            std::cmp::min(
                econfig.max_num_seqs.max(Self::GRAPH_CAPTURE_MIN_BATCH),
                mamba_cache_capacity.max(1),
            )
        } else {
            econfig.max_num_seqs.max(Self::GRAPH_CAPTURE_MIN_BATCH)
        };

        #[cfg(all(feature = "cuda", feature = "graph"))]
        {
            if is_hybrid_mamba_model {
                let capture_capacity = planned_graph_capture_batches(graph_capture_max_num_seqs)
                    .into_iter()
                    .max()
                    .unwrap_or(1);
                if capture_capacity > mamba_cache_capacity {
                    candle_core::bail!(
                        "graph capture batch {} exceeds mamba cache capacity {}",
                        capture_capacity,
                        mamba_cache_capacity
                    );
                }
            }
        }

        let prefix_cache_enabled = econfig.prefix_cache.unwrap_or(false);
        let mut mamba_prefix_capacity =
            Self::effective_mamba_prefix_capacity(prefix_cache_enabled, mamba_cache_capacity);
        if is_hybrid_mamba_model && econfig.mamba_slot_bytes > 0 && econfig.mamba_memory_bytes > 0 {
            let active_reserved = mamba_cache_capacity.saturating_mul(econfig.mamba_slot_bytes);
            let prefix_budget_slots = econfig.mamba_memory_bytes.saturating_sub(active_reserved)
                / econfig.mamba_slot_bytes;
            mamba_prefix_capacity = if prefix_cache_enabled {
                prefix_budget_slots
            } else {
                0
            };
            if mamba_prefix_capacity == 0 && prefix_cache_enabled {
                crate::log_warn!(
                    "Hybrid mamba prefix-state cache disabled because the mamba memory budget leaves no snapshot slots after active slots."
                );
            }
        }
        match &model {
            Model::Qwen3_5(model) => {
                model.preallocate_mamba_cache(mamba_cache_capacity)?;
                model.set_mamba_prefix_cache_capacity(mamba_prefix_capacity);
            }
            Model::Qwen3_5MoE(model) => {
                model.preallocate_mamba_cache(mamba_cache_capacity)?;
                model.set_mamba_prefix_cache_capacity(mamba_prefix_capacity);
            }
            Model::Qwen3VL(model) => {
                model.preallocate_mamba_cache(mamba_cache_capacity)?;
                model.set_mamba_prefix_cache_capacity(mamba_prefix_capacity);
            }
            _ => {}
        }

        if is_hybrid_mamba_model {
            const SIZE_IN_GB: f64 = 1024.0 * 1024.0 * 1024.0;
            const SIZE_IN_MB: f64 = 1024.0 * 1024.0;
            let active_reserved_bytes =
                mamba_cache_capacity.saturating_mul(econfig.mamba_slot_bytes);
            let prefix_budget_bytes = econfig
                .mamba_memory_bytes
                .saturating_sub(active_reserved_bytes);
            crate::log_info!(
                "Hybrid mamba slots preallocated: {} (max_num_seqs={}); prefix-state capacity={} entries; mamba memory budget={:.2}GB (active={:.2}GB, prefix={:.2}GB, per-slot={:.2}MB)",
                mamba_cache_capacity,
                econfig.max_num_seqs,
                mamba_prefix_capacity,
                econfig.mamba_memory_bytes as f64 / SIZE_IN_GB,
                active_reserved_bytes as f64 / SIZE_IN_GB,
                prefix_budget_bytes as f64 / SIZE_IN_GB,
                econfig.mamba_slot_bytes as f64 / SIZE_IN_MB
            );
        }

        let (gpu_kv_cache, cpu_kv_cache) =
            allocator.init_kv_cache(&allocation, dtype, &device, econfig.pd_config.as_ref())?;

        let num_cpu_blocks =
            (econfig.cpu_mem_fold.unwrap_or(0.2f32) * econfig.num_blocks as f32) as usize;
        let cpu_tq_cache = allocator.init_cpu_tq_cache(num_cpu_blocks)?;

        let (temperature, top_k, top_p) = if econfig.generation_cfg.is_some() {
            (
                econfig.generation_cfg.as_ref().unwrap().temperature.clone(),
                econfig.generation_cfg.as_ref().unwrap().top_k.clone(),
                econfig.generation_cfg.as_ref().unwrap().top_p.clone(),
            )
        } else {
            (None, None, None)
        };

        let seed = if econfig.seed.is_none() {
            rand::random::<u64>()
        } else {
            econfig.seed.unwrap()
        };

        #[cfg(feature = "flashinfer")]
        let has_heterogeneous_head_dim =
            matches!(model_type, ModelType::Gemma3) || matches!(model_type, ModelType::Gemma4);

        #[cfg(feature = "flashinfer")]
        let skip_flashinfer_init = config.kvcache_dtype.is_turboquant()
            || (config.kvcache_dtype.is_fp8_keys() && !attention_rs::has_flashinfer_fp8_e4m3())
            || has_heterogeneous_head_dim;
        #[cfg(feature = "flashinfer")]
        let flashinfer_kv_params = if skip_flashinfer_init {
            None
        } else {
            let mut params = None;
            for (k_cache, _) in &gpu_kv_cache {
                if k_cache.rank() != 4 {
                    continue;
                }
                let (_, page_size, num_kv_heads, head_dim) = k_cache.dims4()?;
                let is_mla = matches!(model_type, ModelType::GLM4MoeLite | ModelType::DeepSeek);
                params = Some(FlashInferKvParams {
                    kv_dtype: k_cache.dtype(),
                    out_dtype: dtype,
                    page_size,
                    num_kv_heads,
                    head_dim,
                    num_qo_heads: if is_mla {
                        config.num_attention_heads
                    } else {
                        config.num_attention_heads / comm.world_size()
                    },
                });
                break;
            }
            params
        };
        #[cfg(feature = "flashinfer")]
        if skip_flashinfer_init {
            crate::log_info!(
                "Use native flash backend ({:?} kvcache, flashinfer disabled)",
                config.kvcache_dtype
            );
        } else {
            crate::log_info!("Use flashinfer backend {:?}", flashinfer_kv_params);
        }

        #[cfg(all(feature = "flashattn", not(feature = "flashinfer")))]
        {
            let flashattn_usable = if config.kvcache_dtype.is_turboquant() {
                false
            } else if config.kvcache_dtype.is_fp8_keys() {
                let sm = device
                    .as_cuda_device()
                    .ok()
                    .and_then(|d| attention_rs::cuda_utils::sm_version(d))
                    .unwrap_or(0);
                sm == 90 // FP8 requires SM90
            } else {
                true
            };

            if flashattn_usable {
                crate::log_info!("Use flashattn backend ({:?} kvcache)", config.kvcache_dtype);
            } else {
                crate::log_info!(
                    "Use native flash backend ({:?} kvcache, flashattn not suitable)",
                    config.kvcache_dtype
                );
            }
        }

        if mamba_prefix_capacity > 0
            && comm.rank() == 0
            && matches!(model, Model::Qwen3_5(_) | Model::Qwen3_5MoE(_))
        {
            crate::log_info!(
                "Hybrid mamba prefix-state cache enabled: {} entries",
                mamba_prefix_capacity
            );
        }

        #[cfg(feature = "cuda")]
        if dtype == DType::F16 {
            let sm = device
                .as_cuda_device()
                .ok()
                .and_then(|d| attention_rs::cuda_utils::sm_version(d))
                .unwrap_or(0);
            if sm >= 80 {
                let will_use_native_flash = {
                    #[cfg(feature = "flashinfer")]
                    {
                        skip_flashinfer_init
                    }
                    #[cfg(all(feature = "flashattn", not(feature = "flashinfer")))]
                    {
                        config.kvcache_dtype.is_turboquant()
                            || (config.kvcache_dtype.is_fp8_keys() && sm != 90)
                    }
                    #[cfg(not(any(feature = "flashinfer", feature = "flashattn")))]
                    {
                        true
                    }
                };
                if will_use_native_flash {
                    candle_core::bail!(
                        "F16 dtype is not supported with native flash attention on SM80+ (detected SM{}). \
                         Native flash kernels on SM80+ are compiled for BF16 only. \
                         Use --dtype bf16 (default), or build with flashinfer/flashattn features which support F16.",
                        sm
                    );
                }
            }
        }

        Ok(Self {
            model,
            gpu_kv_cache: Arc::new(Mutex::new(gpu_kv_cache)),
            cpu_kv_cache: Arc::new(Mutex::new(cpu_kv_cache)),
            cpu_tq_cache,
            device,
            config: econfig.clone(),
            #[cfg(all(feature = "cuda", feature = "graph"))]
            capturer: GraphCapturer::new(
                wrapper,
                graph_capture_max_num_seqs,
                econfig.max_model_len.unwrap_or(32768),
                econfig.block_size,
                config.hidden_size,
                #[cfg(feature = "flashinfer")]
                &flashinfer_kv_params,
                matches!(model_type, ModelType::GLM4MoeLite | ModelType::DeepSeek),
            ),
            #[cfg(feature = "flashinfer")]
            flashinfer_kv_params,
            logit_processor: LogitsProcessor::new(seed, temperature, top_k, top_p),
            cached_sampling: RwLock::new(None),
            seq_tokens: RwLock::new(HashMap::new()),
            restored_prefix_sequences: RwLock::new(HashSet::new()),
            guidance_states: RwLock::new(HashMap::new()),
            guidance_failed: RwLock::new(HashSet::new()),
            guidance_mismatch: RwLock::new(HashSet::new()),
            llg_factory,
            transfer,
            is_first_rank: comm.rank() == 0,
            model_type,
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<(Tensor, Tensor)>> {
        loop {
            if let Ok(v) = self.gpu_kv_cache.try_lock() {
                return v;
            }
        }
    }

    pub fn get_cpu_kv_cache(&self) -> MutexGuard<'_, Vec<(Tensor, Tensor)>> {
        loop {
            if let Ok(v) = self.cpu_kv_cache.try_lock() {
                return v;
            }
        }
    }

    fn restore_mamba_prefix_states_for_prefill(&self, seqs: &[&Sequence]) -> Result<()> {
        match &self.model {
            Model::Qwen3_5(_) | Model::Qwen3_5MoE(_) | Model::Qwen3VL(_) => {
                for seq in seqs {
                    if seq.num_cached_tokens == 0 {
                        continue;
                    }
                    let Some(hash) = seq.mamba_prefix_hash else {
                        continue;
                    };
                    if self.restored_prefix_sequences.read().contains(&seq.id) {
                        continue;
                    }
                    let restored = self.restore_mamba_prefix_state(seq.id, hash)?;
                    if !restored {
                        candle_core::bail!(
                            "Missing mamba prefix snapshot for seq {} hash {}",
                            seq.id,
                            hash
                        );
                    }
                    self.restored_prefix_sequences.write().insert(seq.id);
                    crate::log_info!(
                        "Restored mamba prefix state for seq {} (cached {} tokens)",
                        seq.id,
                        seq.num_cached_tokens
                    );
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        match &self.model {
            Model::Qwen3_5(model) => model.restore_mamba_prefix_state(seq_id, hash),
            Model::Qwen3_5MoE(model) => model.restore_mamba_prefix_state(seq_id, hash),
            Model::Qwen3VL(model) => model.restore_mamba_prefix_state(seq_id, hash),
            _ => Ok(true),
        }
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        match &self.model {
            Model::Qwen3_5(model) => model.capture_mamba_prefix_state(seq_id, hash, preserve),
            Model::Qwen3_5MoE(model) => model.capture_mamba_prefix_state(seq_id, hash, preserve),
            Model::Qwen3VL(model) => model.capture_mamba_prefix_state(seq_id, hash, preserve),
            _ => return Ok(true),
        }
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> Result<bool> {
        match &self.model {
            Model::Qwen3_5(model) => Ok(model.has_mamba_prefix_state(hash)),
            Model::Qwen3_5MoE(model) => Ok(model.has_mamba_prefix_state(hash)),
            Model::Qwen3VL(model) => Ok(model.has_mamba_prefix_state(hash)),
            _ => Ok(true),
        }
    }

    pub fn remove_mamba_prefix_state(&self, hash: u64) -> Result<bool> {
        match &self.model {
            Model::Qwen3_5(model) => Ok(model.remove_mamba_prefix_state(hash)),
            Model::Qwen3_5MoE(model) => Ok(model.remove_mamba_prefix_state(hash)),
            Model::Qwen3VL(model) => Ok(model.remove_mamba_prefix_state(hash)),
            _ => Ok(true),
        }
    }

    #[allow(unused)]
    pub fn run(&self, seqs: Seqs, is_prefill: bool) -> Result<Vec<u32>> {
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("{}", if is_prefill { "prefill" } else { "decoding" });
        let (input_ids, positions, mut input_metadata) = if is_prefill {
            match &seqs {
                Seqs::SeqRefs(seqs) => self.prepare_prefill(seqs)?,
                Seqs::DecodeVec(_) => {
                    candle_core::bail!(
                        "Decode sequences are not supported for prefill. Use SeqRefs instead."
                    );
                }
            }
        } else {
            match &seqs {
                Seqs::SeqRefs(seqs) => self.prepare_decode(*seqs)?,
                Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
            }
        };

        if is_prefill {
            if let Seqs::SeqRefs(seqs_ref) = &seqs {
                self.restore_mamba_prefix_states_for_prefill(seqs_ref)?;
            }
        }

        #[cfg(all(feature = "cuda", feature = "graph"))]
        {
            let input_batch = input_ids.dim(0)?;
            let require_exact_graph = input_metadata.mamba_slot_mapping.is_some();
            let can_replay = if require_exact_graph {
                self.capturer.is_exact_captured(input_batch)
            } else {
                self.capturer.is_captured(input_batch)
            };
            if !is_prefill && can_replay {
                let logits = match &self.model {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        } else {
                            self.capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        }
                    }
                    _ => self
                        .capturer
                        .replay(&input_ids, &positions, &input_metadata)?,
                };
                let output_ids = self.sample(&logits, seqs, is_prefill)?;
                return Ok(output_ids);
            }
        }

        #[cfg(feature = "flashinfer")]
        if !is_prefill {
            if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
                if input_metadata.is_mla {
                    if fm.mla_decode_plan_info.is_none() {
                        if let Some(params) = self.flashinfer_kv_params {
                            fm.mla_decode_plan_info = Some(attention_rs::mla::mla_decode_plan(
                                &self.device,
                                params.kv_dtype,
                                &fm.indptr_host,
                                input_ids.dim(0)?,
                                params.num_qo_heads,
                                params.page_size,
                                fm.use_cuda_graph,
                            )?);
                        }
                    }
                } else if fm.decode_plan_info.is_none() {
                    if let Some(params) = self.flashinfer_kv_params {
                        fm.decode_plan_info = Some(attention_rs::flashinfer::decode_plan(
                            &self.device,
                            params.kv_dtype,
                            params.out_dtype,
                            &fm.indptr_host,
                            fm.last_len_host.as_deref(),
                            fm.kv_len_arr_host.as_deref(),
                            input_ids.dim(0)?,
                            params.num_qo_heads,
                            params.num_kv_heads,
                            params.head_dim,
                            params.page_size,
                            fm.use_cuda_graph,
                        )?);
                    }
                }
            }
        }

        let images = if let Seqs::SeqRefs(s) = &seqs {
            // We do not batch multimodel prefill
            if let Some(images) = &s[0].images {
                if images.image_idx == -1 || !is_prefill {
                    None
                } else {
                    compute_image_slice(&s[0].token_ids, s[0].num_cached_tokens, images).map(
                        |(image_idx, token_offset)| {
                            let mut images = images.clone();
                            images.image_idx = image_idx;
                            images.image_token_offset = token_offset;
                            images
                        },
                    )
                }
            } else {
                None
            }
        } else {
            None
        };
        let images = images.as_ref();

        let _prefill_guard = set_linear_is_prefill(is_prefill);
        let logits = crate::model_call!(
            &self.model,
            forward,
            (&input_ids, &positions, Some(&self.get_kv_cache()), &input_metadata),
            {
                Qwen3 => false,
                Qwen3MoE => false,
                Qwen3_5 => false,
                Qwen3_5MoE => false,
                LLaMa => false,
                LLaMa4 => images,
                Phi4 => false,
                GLM4 => false,
                GLM4MoE => false,
                GLM4MoeLite => false,
                DeepSeek => false,
                Mistral3VL => images,
                Gemma3 => images,
                Gemma4 => false,
                Qwen3VL => images,
                MiniMax => false,
            }
        )?;
        let output_ids = self.sample(&logits, seqs, is_prefill)?;
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        Ok(output_ids)
    }

    pub fn embed(&self, seqs: &[&Sequence], strategy: &EmbeddingStrategy) -> Result<Vec<Vec<f32>>> {
        let (input_ids, positions, input_metadata) = self.prepare_prefill(seqs)?;

        let _prefill_guard = set_linear_is_prefill(true);
        let hidden = crate::model_call!(
            &self.model,
            forward_embedding,
            (&input_ids, &positions, Some(&self.get_kv_cache()), &input_metadata),
            {
                Qwen3 => false,
                Qwen3MoE => false,
                Qwen3_5 => false,
                Qwen3_5MoE => false,
                LLaMa => false,
                LLaMa4 => None,
                Phi4 => false,
                GLM4 => false,
                Gemma3 => None,
                Gemma4 => false,
                MiniMax => false,
            },
            candle_core::bail!("Embedding is not supported for this model type")
        )?;

        crate::log_info!(
            "Embedding forward finished with hidden shape {:?}",
            hidden.shape()
        );
        let hidden = hidden.to_dtype(DType::F32)?;
        let dims = hidden.dims();
        if dims.len() != 2 {
            candle_core::bail!("Unexpected embedding tensor dims {:?}", dims);
        }

        let mut start = 0;
        let mut outputs = Vec::new();
        for seq in seqs {
            let len = seq.len().saturating_sub(seq.num_cached_tokens);
            crate::log_info!(
                "Extracting embedding state for Seq {} (start {start}, len {len})",
                seq.id
            );
            let slice = hidden.narrow(0, start, len)?;
            let pooled = match strategy {
                EmbeddingStrategy::Mean => slice.mean(D::Minus2)?,
                EmbeddingStrategy::Last => slice.narrow(0, len.saturating_sub(1), 1)?.squeeze(0)?,
            };
            outputs.push(pooled.to_vec1::<f32>()?);
            start += len;
        }

        Ok(outputs)
    }

    fn prepare_block_tables<'a, I, S>(&self, seqs: I) -> Result<Tensor>
    where
        I: IntoIterator<Item = &'a S>,
        S: ToDecodeInput + 'a,
    {
        let seq_refs: Vec<&'a S> = seqs.into_iter().collect(); // only references, no clone
        let len = seq_refs.len();

        let max_len = seq_refs
            .iter()
            .map(|seq| seq.block_table().len())
            .max()
            .unwrap_or(0);

        let mut flat: Vec<u32> = Vec::with_capacity(len * max_len);
        for seq in &seq_refs {
            let bt = seq.block_table();
            flat.extend_from_slice(bt);
            flat.extend(std::iter::repeat(0).take(max_len - bt.len()));
        }

        Tensor::from_vec(flat, (len, max_len), &self.device)
    }

    #[allow(non_snake_case)]
    #[allow(unused_mut)]
    fn prepare_prefill(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut batch_indices_vec: Vec<u32> = Vec::new();
        let mut positions_vec: Vec<u32> = Vec::new();
        let mut prefill_tokens: Vec<usize> = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let chunk_size = self.config.effective_prefill_chunk_size();
        let mut max_context_len = 0;
        for (seq_idx, seq) in seqs.iter().enumerate() {
            let seqlen = seq.len();
            let num_tokens = std::cmp::min(chunk_size, seqlen - seq.num_cached_tokens);
            input_ids
                .extend(&seq.token_ids[seq.num_cached_tokens..seq.num_cached_tokens + num_tokens]);
            positions.extend(
                (seq.num_cached_tokens as i64..(seq.num_cached_tokens + num_tokens) as i64)
                    .collect::<Vec<_>>(),
            );
            for pos in 0..num_tokens {
                batch_indices_vec.push(seq_idx as u32);
                positions_vec.push((seq.num_cached_tokens + pos) as u32);
            }
            prefill_tokens.push(num_tokens);
            let seqlen_q = num_tokens;
            let seqlen_k = if seq.num_cached_tokens > 0 {
                seq.num_cached_tokens + num_tokens
            } else {
                num_tokens
            };
            let effective_context = seq.num_cached_tokens + num_tokens;
            if effective_context > max_context_len {
                max_context_len = effective_context;
            }
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
            max_seqlen_q = std::cmp::max(max_seqlen_q, seqlen_q);
            max_seqlen_k = std::cmp::max(max_seqlen_k, seqlen_k);

            let mut slot_mapping_tokens: i64 = 0;
            for i in seq.num_cached_blocks()..seq.num_blocks() {
                let start = (seq.block_table[i] * self.config.block_size as u32) as i64;
                let start = if i == seq.num_cached_blocks() {
                    start + (seq.num_cached_tokens as i64 % self.config.block_size as i64)
                } else {
                    start
                };
                let end = start
                    + std::cmp::min(
                        num_tokens as i64 - slot_mapping_tokens,
                        self.config.block_size as i64,
                    );
                slot_mapping.extend((start..end).collect::<Vec<i64>>());
                slot_mapping_tokens += end - start;
                if slot_mapping_tokens >= num_tokens as i64 {
                    break;
                }
            }
        }

        assert!(
            input_ids.len() > 0 && positions.len() > 0 && slot_mapping.len() > 0,
            "Invalid inputs!"
        );
        // Validate lengths
        if input_ids.len() != slot_mapping.len() {
            candle_core::bail!(
                "input_ids and slot_mapping must have same length: {}, {}",
                input_ids.len(),
                slot_mapping.len()
            );
        }
        if input_ids.len() != *cu_seqlens_q.last().unwrap() as usize {
            candle_core::bail!("input_ids length must match last cu_seqlens_q",);
        }
        // crate::log_info!("input_ids {:?}, positions {:?}, slot_mapping {:?}", input_ids, positions, slot_mapping);

        // Create tensors
        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let q_len = cu_seqlens_q.len();
        let k_len = cu_seqlens_k.len();
        let s_len = slot_mapping.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;

        let block_tables_t = self.prepare_block_tables(seqs)?;
        let context_lens_vec: Vec<u32> = seqs
            .iter()
            .zip(prefill_tokens.iter())
            .map(|(seq, &num_tokens)| (seq.num_cached_tokens + num_tokens) as u32)
            .collect();
        let context_lens_t = Tensor::from_vec(context_lens_vec, seqs.len(), &self.device)?;
        let block_tables = Some(block_tables_t);
        let context_lens = Some(context_lens_t);
        let cu_seqlens_q_vec = cu_seqlens_q.clone();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), &self.device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), &self.device)?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params.is_some() {
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for (seq, &num_tokens) in seqs.iter().zip(prefill_tokens.iter()) {
                let effective_len = seq.num_cached_tokens + num_tokens;
                let max_blocks = seq.block_table.len();
                let num_blocks = if effective_len == 0 {
                    0
                } else {
                    (effective_len + self.config.block_size - 1) / self.config.block_size
                };
                let num_blocks = std::cmp::min(num_blocks, max_blocks);
                let bt = &seq.block_table[..num_blocks];
                indices.extend(bt.iter().map(|&x| x as u32));
                indptr.push(indices.len() as u32);
                let last = if effective_len == 0 {
                    0
                } else {
                    (effective_len - 1) % self.config.block_size + 1
                };
                last_len.push(last as u32);
            }

            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            if let Some((pos, &bad_idx)) = indices
                .iter()
                .enumerate()
                .find(|(_, &idx)| idx as usize >= self.config.num_blocks)
            {
                candle_core::bail!(
                    "flashinfer prefill block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                    pos,
                    bad_idx,
                    self.config.num_blocks
                );
            }
            let indptr_len = indptr.len();
            let indices_len = indices.len();
            let last_len_val = last_len.len();
            let batch_indices_len = batch_indices_vec.len();
            let positions_len = positions_vec.len();

            let indptr = Tensor::from_vec(indptr, (indptr_len,), &self.device)?;
            let indices = Tensor::from_vec(indices, (indices_len,), &self.device)?;
            let last_len = Tensor::from_vec(last_len, (last_len_val,), &self.device)?;
            let batch_indices =
                Tensor::from_vec(batch_indices_vec, (batch_indices_len,), &self.device)?;
            let positions = Tensor::from_vec(positions_vec, (positions_len,), &self.device)?;

            let cu_seqlens_q_host_u32: Vec<u32> =
                cu_seqlens_q_vec.iter().map(|&x| x as u32).collect();

            let mut prefill_plan_info: Option<Vec<i64>> = None;
            let mut mla_prefill_plan_info: Option<Vec<i64>> = None;

            if self.is_mla_model() {
                if let Some(params) = self.flashinfer_kv_params {
                    mla_prefill_plan_info = Some(attention_rs::mla::mla_prefill_plan(
                        &self.device,
                        &cu_seqlens_q_host_u32,
                        &indptr_host,
                        &kv_len_arr_host,
                        last_len_host.len(),
                        params.num_qo_heads,
                        params.head_dim,
                        true,
                    )?)
                }
            };

            if !self.is_mla_model() {
                if let Some(params) = self.flashinfer_kv_params {
                    prefill_plan_info = Some(attention_rs::flashinfer::prefill_plan(
                        &self.device,
                        &cu_seqlens_q_host_u32,
                        &indptr_host,
                        &kv_len_arr_host,
                        *cu_seqlens_q_vec.last().unwrap() as u32,
                        last_len_host.len(),
                        params.num_qo_heads,
                        params.num_kv_heads,
                        params.head_dim,
                        params.page_size,
                        params.out_dtype,
                        None,
                        Some(params.kv_dtype),
                    )?)
                }
            };

            Some(FlashInferMetadata {
                indptr,
                indptr_host,
                indices,
                last_len,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: Some(*cu_seqlens_q_vec.last().unwrap() as u32),
                batch_indices: Some(batch_indices),
                positions: Some(positions),
                use_cuda_graph: false,
                decode_plan_info: None,
                prefill_plan_info,
                mla_decode_plan_info: None,
                mla_prefill_plan_info,
            })
        } else {
            None
        };

        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let sequence_ids_vec = seqs.iter().map(|s| s.id()).collect::<Vec<_>>();
        let mamba_slot_mapping = self.prepare_mamba_slot_mapping(&sequence_ids_vec, true)?;
        let sequence_ids = Some(sequence_ids_vec);

        let input_metadata = InputMetadata {
            is_prefill: true,
            is_mla: self.is_mla_model(),
            sequence_ids,
            mamba_slot_mapping,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len,
            seqlens: Some(cu_seqlens_q_vec[1..].to_vec()),
            flashinfer_metadata,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn prepare_decode<'a, I, S>(&self, seqs: I) -> Result<(Tensor, Tensor, InputMetadata)>
    where
        I: IntoIterator<Item = &'a S>,
        S: ToDecodeInput + 'a,
    {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();

        let seq_refs: Vec<&'a S> = seqs.into_iter().collect(); // only references, no clone

        for seq in &seq_refs {
            input_ids.push(seq.last_token());
            positions.push((seq.len() - 1) as i64);
            context_lens.push(seq.len() as u32);
            let slot = seq.block_table_last() * self.config.block_size as u32
                + seq.last_block_tokens() as u32
                - 1;
            slot_mapping.push(slot as i64);
        }

        // Create tensors
        let length = positions.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let s_len = slot_mapping.len();
        let c_len = context_lens.len();
        let max_context_len = context_lens.clone().into_iter().max().unwrap() as usize;

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;
        let context_lens = Tensor::from_vec(context_lens, (c_len,), &self.device)?;
        let block_tables = self.prepare_block_tables(seq_refs.clone())?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params.is_some() {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_cuda_graph = {
                let require_exact_graph = match &self.model {
                    Model::Qwen3_5(_) | Model::Qwen3_5MoE(_) => true,
                    Model::Qwen3VL(model) => model.uses_hybrid_mamba_text_model(),
                    _ => false,
                };
                if require_exact_graph {
                    self.capturer.is_exact_captured(seq_refs.len())
                } else {
                    self.capturer.is_captured(seq_refs.len())
                }
            };
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_cuda_graph = false;

            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for seq in &seq_refs {
                let bt = seq.block_table();
                indices.extend(bt.iter().map(|&x| x as u32));
                indptr.push(indices.len() as u32);
                let len = seq.len();
                let last = if len == 0 {
                    0
                } else {
                    (len - 1) % self.config.block_size + 1
                };
                last_len.push(last as u32);
            }
            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            if let Some((pos, &bad_idx)) = indices
                .iter()
                .enumerate()
                .find(|(_, &idx)| idx as usize >= self.config.num_blocks)
            {
                candle_core::bail!(
                    "flashinfer decode block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                    pos,
                    bad_idx,
                    self.config.num_blocks
                );
            }
            let indptr_len = indptr.len();
            let indices_len = indices.len();
            let last_len_val = last_len.len();

            let indptr = Tensor::from_vec(indptr, (indptr_len,), &self.device)?;
            let indices = Tensor::from_vec(indices, (indices_len,), &self.device)?;
            let last_len = Tensor::from_vec(last_len, (last_len_val,), &self.device)?;

            Some(FlashInferMetadata {
                indptr,
                indptr_host,
                indices,
                last_len,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: None,
                batch_indices: None,
                positions: None,
                use_cuda_graph,
                decode_plan_info: None,
                prefill_plan_info: None,
                mla_decode_plan_info: None,
                mla_prefill_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let sequence_ids = Some(seq_refs.iter().map(|s| s.id()).collect::<Vec<_>>());
        let mamba_slot_mapping = self.prepare_mamba_slot_mapping(
            sequence_ids
                .as_ref()
                .expect("sequence_ids should exist for decode"),
            false,
        )?;

        let input_metadata = InputMetadata {
            is_prefill: false,
            is_mla: self.is_mla_model(),
            sequence_ids,
            mamba_slot_mapping,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len,
            seqlens: None,
            flashinfer_metadata,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn sample(&self, logits: &Tensor, seqs: Seqs, is_prefill: bool) -> Result<Vec<u32>> {
        let seq_ids: Vec<usize> = match &seqs {
            Seqs::SeqRefs(seqs) => seqs.iter().map(|s| s.id()).collect(),
            Seqs::DecodeVec(v) => v.iter().map(|s| s.id()).collect(),
        };

        // Get the batch size for deciding whether to use parallel sampling
        let batch_size = match seqs {
            Seqs::SeqRefs(seqs) => seqs.len(),
            Seqs::DecodeVec(v) => v.len(),
        };

        // Compute and cache sampling params (including penalties) during prefill, reuse during decode
        let cached_params = match (is_prefill, &seqs) {
            // Prefill: compute sampling strategy and penalties, cache for decode phase
            (true, Seqs::SeqRefs(seqs)) => {
                // Check if generation_cfg has valid sampling params (temperature AND top_k/top_p)
                let has_valid_sampling_cfg =
                    self.config.generation_cfg.as_ref().map_or(false, |cfg| {
                        cfg.temperature.is_some() && (cfg.top_k.is_some() || cfg.top_p.is_some())
                    });
                let user_params = &seqs[0].sampling_params;

                // Log thinking parameter only from first rank to avoid duplicate logs in multi-GPU
                if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                    crate::log_info!(
                        "User's thinking preference for reasoning models: {:?}",
                        user_params.thinking
                    );
                }

                // Determine frequency/presence penalties (user params > generation_cfg)
                let gen_cfg_freq = self
                    .config
                    .generation_cfg
                    .as_ref()
                    .and_then(|c| c.frequency_penalty);
                let gen_cfg_pres = self
                    .config
                    .generation_cfg
                    .as_ref()
                    .and_then(|c| c.presence_penalty);
                let frequency_penalty = user_params.frequency_penalty.or(gen_cfg_freq);
                let presence_penalty = user_params.presence_penalty.or(gen_cfg_pres);

                let user_has_temperature = user_params.temperature.is_some();
                let user_wants_greedy = matches!(user_params.temperature, Some(t) if t == 0.0);
                let has_user_config = user_has_temperature
                    || matches!(user_params.top_k, Some(k) if k > 0)
                    || matches!(user_params.top_p, Some(p) if p > 0.0 && p < 1.0);

                let sampling = if user_wants_greedy {
                    if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                        crate::log_warn!("Using greedy decoding (temperature=0.0)");
                    }
                    Sampling::ArgMax
                } else if has_user_config {
                    if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                        crate::log_warn!(
                            "Using user's sampling params: temp={:?}, top_k={:?}, top_p={:?}, freq_penalty={:?}, pres_penalty={:?}",
                            user_params.temperature,
                            user_params.top_k,
                            user_params.top_p,
                            frequency_penalty,
                            presence_penalty
                        );
                    }
                    LogitsProcessor::get_strategy(
                        user_params.temperature,
                        user_params.top_k,
                        user_params.top_p,
                    )
                } else if has_valid_sampling_cfg {
                    let cfg = self.config.generation_cfg.as_ref().unwrap();
                    if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                        crate::log_warn!(
                            "Using sampling from generation_config: temp={:?}, top_k={:?}, top_p={:?}, freq_penalty={:?}, pres_penalty={:?}",
                            cfg.temperature,
                            cfg.top_k,
                            cfg.top_p,
                            frequency_penalty,
                            presence_penalty
                        );
                    }
                    LogitsProcessor::get_strategy(cfg.temperature, cfg.top_k, cfg.top_p)
                } else {
                    if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                        crate::log_warn!(
                            "No generation_config, using default sampling (temperature=0.7, top_k=32, top_p=0.95)"
                        );
                    }
                    Sampling::TopKThenTopP {
                        k: 32,
                        p: 0.95,
                        temperature: 0.7,
                    }
                };

                let cached = CachedSamplingParams {
                    sampling,
                    frequency_penalty,
                    presence_penalty,
                };

                // Cache for decode phase
                *self.cached_sampling.write() = Some(cached.clone());
                cached
            }
            // Decode or non-SeqRefs: use cached parameters
            _ => self
                .cached_sampling
                .read()
                .clone()
                .unwrap_or(CachedSamplingParams {
                    sampling: Sampling::TopKThenTopP {
                        k: 32,
                        p: 0.95,
                        temperature: 0.7,
                    },
                    frequency_penalty: None,
                    presence_penalty: None,
                }),
        };

        let (guided_logits, guided_seq_ids) =
            self.apply_requested_guidance(logits, &seqs, &seq_ids)?;

        // Apply penalties using cached values (same for all sequences in batch)
        // This is done AFTER LLG masking so penalties only affect tokens allowed by grammar
        let has_any_penalty =
            cached_params.frequency_penalty.is_some() || cached_params.presence_penalty.is_some();

        let logits = if !is_prefill && has_any_penalty {
            let seq_tokens = self.seq_tokens.write();
            let reference_tokens: Vec<Vec<u32>> = seq_ids
                .iter()
                .map(|id| {
                    if let Some(tokens) = seq_tokens.get(&id) {
                        if tokens.len() > 128 {
                            tokens[tokens.len().saturating_sub(128)..].to_vec()
                        } else {
                            vec![]
                        }
                    } else {
                        vec![]
                    }
                })
                .collect();

            self.logit_processor.apply_batch_repeat_penalty(
                &guided_logits,
                vec![cached_params.frequency_penalty.unwrap_or(0.0); batch_size],
                vec![cached_params.presence_penalty.unwrap_or(0.0); batch_size],
                reference_tokens,
            )?
        } else {
            guided_logits.to_owned()
        };

        let tokens = self.sample_processed_logits(&logits, &cached_params.sampling)?;

        self.commit_guided_tokens(&seq_ids, &tokens, guided_seq_ids);

        // Track tokens for sequences when penalties are enabled
        if has_any_penalty {
            let mut seq_tokens = self.seq_tokens.write();
            for i in 0..seq_ids.len() {
                if seq_tokens.contains_key(&seq_ids[i]) {
                    seq_tokens
                        .get_mut(&seq_ids[i])
                        .expect("no entry")
                        .push(tokens[i]);
                } else {
                    seq_tokens.insert(seq_ids[i], vec![tokens[i]].into());
                }
            }
        }

        // Guided token commits are handled immediately after sampling.
        Ok(tokens)
    }

    pub fn finished(&self, id: usize) {
        let mut seq_tokens = self.seq_tokens.write();
        let _ = seq_tokens.remove(&id);
        let mut restored = self.restored_prefix_sequences.write();
        let _ = restored.remove(&id);
        let mut guidance_states = self.guidance_states.write();
        let _ = guidance_states.remove(&id);
        let mut guidance_failed = self.guidance_failed.write();
        let _ = guidance_failed.remove(&id);
        let mut guidance_mismatch = self.guidance_mismatch.write();
        let _ = guidance_mismatch.remove(&id);
        match &self.model {
            Model::Qwen3_5(model) => model.release_sequence_state(id),
            Model::Qwen3_5MoE(model) => model.release_sequence_state(id),
            Model::Qwen3VL(model) => model.release_sequence_state(id),
            _ => {}
        }
    }

    pub fn get_model_vocab_size(&self) -> usize {
        match &self.model {
            Model::Qwen3(model) => model.get_vocab_size(),
            Model::Qwen3MoE(model) => model.get_vocab_size(),
            Model::Qwen3_5(model) => model.get_vocab_size(),
            Model::Qwen3_5MoE(model) => model.get_vocab_size(),
            Model::LLaMa(model) => model.get_vocab_size(),
            Model::LLaMa4(model) => model.get_vocab_size(),
            Model::Phi4(model) => model.get_vocab_size(),
            Model::GLM4(model) => model.get_vocab_size(),
            Model::GLM4MoE(model) => model.get_vocab_size(),
            Model::GLM4MoeLite(model) => model.get_vocab_size(),
            Model::DeepSeek(model) => model.get_vocab_size(),
            Model::Mistral3VL(model) => model.get_vocab_size(),
            Model::Gemma3(model) => model.get_vocab_size(),
            Model::Gemma4(model) => model.get_vocab_size(),
            Model::Qwen3VL(model) => model.get_vocab_size(),
            Model::MiniMax(model) => model.get_vocab_size(),
        }
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub fn warmup_capture(&mut self) -> Result<()> {
        let kv_cache_lock = self.gpu_kv_cache.lock().unwrap(); // no custom method call on `self`
        self.capturer.capture(&self.device, Some(&kv_cache_lock))?;
        match &self.model {
            Model::Qwen3_5(model) => model.reset_mamba_cache()?,
            Model::Qwen3_5MoE(model) => model.reset_mamba_cache()?,
            Model::Qwen3VL(model) => model.reset_mamba_cache()?,
            _ => {}
        }
        self.restored_prefix_sequences.write().clear();
        Ok(())
    }

    pub fn swap_kvcache(&self, mappings: HashMap<usize, usize>, swap_in: bool) -> Result<bool> {
        let tq_mode = attention_rs::get_turboquant_mode();
        let tq_full = matches!(
            tq_mode,
            Some(attention_rs::TurboquantMode::Turbo4) | Some(attention_rs::TurboquantMode::Turbo3)
        );

        if !tq_full {
            let gpu_cache = self.get_kv_cache();
            let cpu_cache = self.get_cpu_kv_cache();
            assert!(
                gpu_cache.len() > 0 && cpu_cache.len() > 0,
                "Invalid kvcache tensors!"
            );
            let block_size_bytes = cpu_cache[0].0.elem_count() / cpu_cache[0].0.dim(0)?
                * cpu_cache[0].0.dtype().size_in_bytes();
            for i in 0..gpu_cache.len() {
                if swap_in {
                    cache::swap_blocks(&cpu_cache[i].0, &gpu_cache[i].0, &mappings)?;
                    cache::swap_blocks(&cpu_cache[i].1, &gpu_cache[i].1, &mappings)?;
                } else {
                    cache::swap_blocks(&gpu_cache[i].0, &cpu_cache[i].0, &mappings)?;
                    cache::swap_blocks(&gpu_cache[i].1, &cpu_cache[i].1, &mappings)?;
                }
            }
            let total_mb =
                (block_size_bytes * mappings.len() * gpu_cache.len() * 2) as f32 / 1024.0 / 1024.0;
            if swap_in {
                crate::log_info!("{:.2} MB CPU KV cached blocks swapped in GPU!", total_mb);
            } else {
                crate::log_info!(
                    "{:.2} MB GPU KV cached blocks swapped out to CPU!",
                    total_mb
                );
            }
        }

        if let Some(cpu_tq) = &self.cpu_tq_cache {
            let num_layers = cpu_tq.len();
            for layer_idx in 0..num_layers {
                let cpu_layer = &cpu_tq[layer_idx];
                attention_rs::with_turboquant_layer(layer_idx, |gpu_layer, _| -> Result<()> {
                    if swap_in {
                        cache::swap_blocks(&cpu_layer.v_absmax, &gpu_layer.v_absmax, &mappings)?;
                        cache::swap_blocks(&cpu_layer.v_quant, &gpu_layer.v_quant, &mappings)?;
                        if let (Some(cpu_ka), Some(gpu_ka)) =
                            (&cpu_layer.k_absmax, &gpu_layer.k_absmax)
                        {
                            cache::swap_blocks(cpu_ka, gpu_ka, &mappings)?;
                        }
                        if let (Some(cpu_kq), Some(gpu_kq)) =
                            (&cpu_layer.k_quant, &gpu_layer.k_quant)
                        {
                            cache::swap_blocks(cpu_kq, gpu_kq, &mappings)?;
                        }
                    } else {
                        cache::swap_blocks(&gpu_layer.v_absmax, &cpu_layer.v_absmax, &mappings)?;
                        cache::swap_blocks(&gpu_layer.v_quant, &cpu_layer.v_quant, &mappings)?;
                        if let (Some(gpu_ka), Some(cpu_ka)) =
                            (&gpu_layer.k_absmax, &cpu_layer.k_absmax)
                        {
                            cache::swap_blocks(gpu_ka, cpu_ka, &mappings)?;
                        }
                        if let (Some(gpu_kq), Some(cpu_kq)) =
                            (&gpu_layer.k_quant, &cpu_layer.k_quant)
                        {
                            cache::swap_blocks(gpu_kq, cpu_kq, &mappings)?;
                        }
                    }
                    Ok(())
                })
                .transpose()?;
            }
            crate::log_info!(
                "TQ buffers {} ({} layers, {} blocks)",
                if swap_in { "swapped in" } else { "swapped out" },
                num_layers,
                mappings.len()
            );
        }

        Ok(true)
    }

    pub fn transfer_prefill(&self, seq: &Sequence) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!(
                    "PD server does not support prefill transfer, call this in the client!"
                )
            }
            transfer.transfer_prefill(seq)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn try_receive_prefill(&self, available_tokens: usize) -> Result<(bool, Option<Sequence>)> {
        if let Some(transfer) = &self.transfer {
            if transfer.is_client() {
                candle_core::bail!("PD client does not support try_receive_prefill!");
            }
            transfer.try_receive_prefill_request(available_tokens)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!");
        }
    }

    pub fn check_prefill_status(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!("PD server does not support check prefill status!");
            }
            transfer.check_prefill_finished(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!");
        }
    }

    pub fn send_kvcache(&self, seq: &Sequence, first_token: u32) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_server() {
                candle_core::bail!(
                    "PD client does not support send_kvcache, call this in the PD server!"
                )
            }
            transfer.transfer_kv_cache(seq, &*self.get_kv_cache(), first_token)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn receive_kvcache(&self, seq: &Sequence) -> Result<(bool, u32, usize, usize)> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!(
                    "PD server does not support receive_kvcache, call this in the PD client!"
                )
            }
            transfer.receive_kv_cache(seq, &*self.get_kv_cache())
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn release_remote_kvcache(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!("release_remote_kvcache should be called from PD client!")
            }
            transfer.release_remote_kvcache(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn check_kvcache_release(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if transfer.is_client() {
                candle_core::bail!("try_check_kvcache_release should be called from PD server!")
            }
            transfer.check_kvcache_release(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn clear_blocks(&self, _block_ids: Vec<u32>) -> Result<bool> {
        Ok(true)
        // fn cache_clear(gpu_cache: &Vec<(Tensor, Tensor)>, block_ids: &Vec<u32>) -> Result<bool> {
        //     if gpu_cache.is_empty() || block_ids.is_empty() {
        //         return Ok(true);
        //     }

        //     for i in 0..gpu_cache.len() {
        //         cache::clear_blocks(&gpu_cache[i].0, block_ids)?;
        //         cache::clear_blocks(&gpu_cache[i].1, block_ids)?;
        //     }

        //     Ok(true)
        // }

        // cache_clear(&*self.get_kv_cache(), &block_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::{collect_guided_batch_entries, Seqs};
    use crate::core::sequence::DecodeSequence;
    use crate::utils::config::SamplingParams;
    use llguidance::api::TopLevelGrammar;

    fn decode_sequence(id: usize, grammar: Option<TopLevelGrammar>) -> DecodeSequence {
        let mut sampling_params = SamplingParams::new_with_max_tokens(16);
        sampling_params.grammar = grammar;
        DecodeSequence {
            id,
            last_token: 0,
            len: 1,
            last_block_tokens: 1,
            block_table_last: 0,
            block_tables: vec![0],
            sampling_params,
        }
    }

    #[test]
    fn test_collect_guided_batch_entries_only_returns_constrained_rows() {
        let seqs = vec![
            decode_sequence(10, None),
            decode_sequence(11, Some(TopLevelGrammar::from_regex("a+"))),
            decode_sequence(12, None),
            decode_sequence(13, Some(TopLevelGrammar::from_regex("b+"))),
        ];
        let seq_ids = seqs.iter().map(|seq| seq.id).collect::<Vec<_>>();

        let guided = collect_guided_batch_entries(&Seqs::DecodeVec(&seqs), &seq_ids);

        assert_eq!(guided, vec![(1, 11), (3, 13)]);
    }
}
