//src/core/engine.rs
use super::runner::{ModelRunner, RunnerType, Seqs};
use super::scheduler::{Scheduler, KVCACHE_SWAP_THRESHOLD};
use super::sequence::Sequence;
use crate::core::scheduler::PD_PREFILL_STATUS_CHECK_COOLING_PERIOD;
use crate::core::sequence::{DecodeSequence, SequenceStatus};
use crate::core::GenerationOutput;
use crate::models::layers::distributed::Comm;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;
use crate::models::layers::VarBuilderX;
use crate::runner::{
    receive_local, send_and_expect_ack, send_local, MessageType, RunnerInitRequest,
};
use crate::server::logger::ChatCompletionLogger;
use crate::server::parser::{detect_prefilled_reasoning_end_marker, ToolConfig};
use crate::server::{EmbeddingStrategy, UsageResponse};
use crate::tools::Tool;
use crate::transfer::PdRole;
use crate::transfer::Transfer;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, EosTokenId, ModelType, SamplingParams};
use crate::utils::guidance::{build_llg_factory, extract_guidance_tokens, GuidanceTokens};
use crate::utils::heartbeat::heartbeat_worker;
use crate::utils::image::{get_image_config, ImageData, ImageProcessConfig};
use crate::utils::kvcache_allocator::KVCacheAllocator;
use crate::utils::progress::{progress_worker, ProgressReporter};
use crate::utils::progress::{spawn_progress_thread, ProgressLike};
use crate::utils::{chat_template::ChatTemplate, prepare_engine_config};
use crate::utils::{get_runner_path, init_config_tokenizer, spawn_runner};
use crate::{log_info, log_warn};
use candle_core::{DType, Result};
use colored::Colorize;
use either::Either;
use futures::future::join_all;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use interprocess::TryClone;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
pub static GLOBAL_RT: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build global Tokio runtime")
});

const REQUEST_ADMISSION_DECODE_BUDGET_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub enum StreamItem {
    Token(String, u32), //streaming: (text, token_id)
    TokenID(u32),       //completion
    Completion((usize, usize, usize, Vec<u32>, Option<String>)), //completion
    Done((usize, usize, usize, usize, Option<String>)), //streaming end
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestType {
    Stream,
    Completion,
}

/// Output of `LLMEngine::preprocess`. Carries the per-request CPU work
/// (chat-template render + tokenize) the caller has already done so the
/// engine write lock can be acquired briefly, for the scheduler-add step
/// only.
#[derive(Debug, Clone)]
pub struct PreprocessedRequest {
    pub params: SamplingParams,
    pub prompt: String,
    pub image_idx: i32,
    pub token_ids: Vec<u32>,
}

#[allow(dead_code)]
pub struct LLMEngine {
    pub runners: Arc<RwLock<RunnerType>>,
    pub scheduler: Scheduler,
    pub tokenizer: Tokenizer,
    pub econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
    prompt_replay_candidates: Vec<Vec<u32>>,
    stream_decoders: HashMap<usize, super::DecodeStreamType>,
    stream_senders: HashMap<usize, Sender<StreamItem>>,
    request_types: HashMap<usize, RequestType>,
    decode_start_times: HashMap<usize, usize>,
    decode_length: HashMap<usize, usize>,
    seq_prefilled_reasoning_end: HashMap<usize, String>,
    seq_prompt_replays: HashMap<usize, Vec<u32>>,
    last_check_throughput_time: usize,
    active_requests: HashSet<usize>,
    cancelled_sequences: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    has_vision: bool,
    model_name: String,
    pub model_type: ModelType,
    pub tool_config: ToolConfig,
    pub img_cfg: Option<ImageProcessConfig>,
    pub guidance_tokens: GuidanceTokens,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<RwLock<Self>>> {
        let (model_pathes, is_gguf, mut config, config_tokenizer, tokenizer, mut generation_cfg) =
            init_config_tokenizer(econfig)?;
        let llg_factory = match build_llg_factory(tokenizer.clone(), config.vocab_size) {
            Ok(f) => Some(f),
            Err(e) => {
                crate::log_warn!("Failed to build llguidance factory: {}", e);
                None
            }
        };

        let stop_flag = Arc::new(AtomicBool::new(false));
        let model_loaded = Arc::new(AtomicBool::new(false));
        let (mut econfig, use_runner) =
            prepare_engine_config(econfig, &config, &config_tokenizer, &mut generation_cfg);
        config.kvcache_dtype = econfig.kvcache_dtype;
        config.is_f16_mode = dtype == DType::F16;

        // In case config file missing bos and eos configuration
        config.apply_generation_cfg(generation_cfg.as_ref());
        if config.eos_token_id.is_none() {
            if let Some(eos) = &config_tokenizer.eos_token {
                if let Some(token) = tokenizer.get_vocab(true).get(eos).copied() {
                    config.eos_token_id = Some(EosTokenId::Single(token));
                };
            }
        }
        let guidance_tokens = extract_guidance_tokens(
            &tokenizer,
            config
                .eos_token_id
                .as_ref()
                .map(EosTokenId::to_vec)
                .unwrap_or_default(),
        );
        assert!(
            config.architectures.is_some() && config.architectures.as_ref().unwrap().len() == 1,
            "Only one architecture is supported at the moment!"
        );
        let arch = config.architectures.as_ref().unwrap()[0].clone();
        if crate::utils::is_qwen3_hybrid_arch_name(arch.as_str()) {
            if let Some(p_cfg) = &econfig.pd_config {
                let role = match p_cfg.role {
                    PdRole::Server => "pd-server",
                    PdRole::Client => "pd-client",
                };
                candle_core::bail!(
                    "{} is enabled with {} (hybrid Mamba model), but PD separation is not supported because Mamba states are not transferred across PD nodes.",
                    role,
                    arch
                );
            }
        }
        let (model_type, default_chat_template, is_rope_i) =
            crate::utils::get_arch_rope(&tokenizer, arch.clone())?;
        log_info!("Use ROPE interleaved {is_rope_i}");

        let is_pd_server = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        };

        log_info!("{:?}", econfig);

        log_info!("{:?}\n", config_tokenizer);

        log_info!("{:?}\n", config);

        let device_ids = if let Some(ids) = &econfig.device_ids {
            log_info!("Using device IDs: {:?}", ids);
            ids.clone()
        } else {
            log_info!("Using default device ID: [0]");
            vec![0]
        };
        #[cfg(feature = "nvtx")]
        assert!(
            device_ids.len() == 1,
            "nvtx profiler only workable on single rank!"
        );

        #[cfg(feature = "nvtx")]
        let use_runner = false;

        let runners = if !use_runner {
            let device = crate::utils::new_device(device_ids[0])?;
            log_info!("Loading model...");
            let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                Arc::new(RwLock::new(Box::new(ProgressReporter::new(0))));
            let handle = progress_worker(1, config.num_hidden_layers, &reporter);
            let mut model_runner = {
                let _guard = candle_core::InferenceMode::enter();
                let vb = VarBuilderX::new(&model_pathes, is_gguf, dtype, &device)?;
                let transfer = if let Some(p_cfg) = &econfig.pd_config {
                    Some(Arc::new(Transfer::new(
                        p_cfg.clone(),
                        0,
                        model_loaded.clone(),
                        stop_flag.clone(),
                    )?))
                } else {
                    None
                };

                let runner = ModelRunner::new(
                    model_type.clone(),
                    &vb,
                    #[cfg(not(feature = "nccl"))]
                    Rc::new(Comm::default()),
                    #[cfg(feature = "nccl")]
                    Rc::new(
                        Comm::from_rank(
                            device.as_cuda_device().unwrap().cuda_device(),
                            0,
                            1,
                            Id::new().unwrap(),
                        )
                        .unwrap(),
                    ),
                    &mut econfig,
                    &config,
                    dtype,
                    is_rope_i,
                    device.clone(),
                    reporter,
                    transfer,
                    llg_factory.clone(),
                    None,
                )?;
                drop(vb);
                runner
            };

            if !is_pd_server {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if econfig.disable_cuda_graph {
                    log_info!("CUDA graph capture disabled by --disable-cuda-graph");
                } else if crate::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    log_info!("{arch} does not supprt CUDA graph");
                } else {
                    match model_runner.warmup_capture() {
                        Ok(_) => {
                            log_info!("Cuda graph captured for performance enhancement!")
                        }
                        Err(e) => crate::log_error!("Unable to capture cuda graph {:?}!", e),
                    }
                }
            }

            model_loaded.store(true, Ordering::SeqCst);
            let _ = handle.join();
            RunnerType::Thread(model_runner)
        } else {
            log_info!("Loading model with runner(s)...");

            #[cfg(feature = "nccl")]
            let nccl_id = Id::new().unwrap();

            let runner_path = get_runner_path()?;

            let uuid_str = uuid::Uuid::new_v4();
            let uuid_str = uuid_str.to_string();
            let unique_id = if let Some(l_uuid) = uuid_str.split('-').last() {
                l_uuid
            } else {
                ""
            };

            #[cfg(feature = "python")]
            pyo3::Python::with_gil(|py| {
                for (rank, _) in device_ids.iter().enumerate() {
                    let sock_name = format!("{}@xinfer-runner-{}", unique_id, rank);
                    spawn_runner(py, &runner_path.display().to_string(), &sock_name, unique_id)
                        .expect("Failed to spawn runner. \n\r*****Tips: xinfer binary not found in this package, use 'build.sh' script to build.");
                }
            });

            #[cfg(not(feature = "python"))]
            for (rank, _) in device_ids.iter().enumerate() {
                let sock_name = format!("{}@xinfer-runner-{}", unique_id, rank);
                spawn_runner(&runner_path.display().to_string(), &sock_name, unique_id)
                    .expect("Failed to spawn runner subprocess. \n\r*****Tips: ensure xinfer is built with './build.sh --install --features cuda,nccl'.");
            }

            let progress_sock_name = format!("{}@xinfer-progress", unique_id);
            let progress_handle = spawn_progress_thread(
                econfig.num_shards.unwrap_or(1),
                config.num_hidden_layers,
                progress_sock_name,
            );
            heartbeat_worker(Some(device_ids.len()), false, stop_flag.clone(), unique_id);
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;
            let engine_config = std::sync::OnceLock::<EngineConfig>::new();
            let runner_streams: Result<Vec<LocalStream>> = device_ids
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    let model_type = model_type.clone();
                    let sock_name = format!("{}@xinfer-runner-{}", unique_id, rank);
                    let listener = ListenerOptions::new()
                        .name(
                            sock_name
                                .clone()
                                .to_ns_name::<GenericNamespaced>()
                                .expect("Failed to to_ns_name"),
                        )
                        .create_sync()
                        .expect("Failed to create listener");

                    crate::log_info!("listener starting accepting runner {}", rank);

                    // Accept one connection
                    let mut stream = listener.accept()?;
                    crate::log_info!("Accepted runner {}", rank);

                    // Wait for "ready"
                    let mut reader = BufReader::new(&mut stream);
                    let mut message = String::new();
                    reader.read_line(&mut message)?;
                    if message.trim() != "ready" {
                        return Err(candle_core::Error::Msg(format!(
                            "Runner {} did not send ready",
                            rank
                        )));
                    }

                    // Build init message
                    let mut econfig_rank = econfig.clone();
                    econfig_rank.device_ids = Some(vec![*dev_id]);
                    let init_msg = MessageType::Init(RunnerInitRequest {
                        rank,
                        dev_id: *dev_id,
                        num_shards: econfig.num_shards.unwrap_or(1),
                        model_type,
                        config: config.clone(),
                        econfig: econfig_rank,
                        model_pathes: model_pathes.clone(),
                        is_gguf,
                        dtype: dtype.into(),
                        is_rope_i,
                        #[cfg(feature = "nccl")]
                        nccl_id: crate::runner::NcclId(nccl_id.clone()),
                    });

                    send_and_expect_ack(&mut stream, &init_msg, "initialize", rank)?;

                    // Always validate/plan allocation after model loading for accurate memory readings
                    let ecfg = engine_config.get_or_init(|| {
                        let mut econfig = econfig.clone();
                        // Use new KVCacheAllocator for multi-rank negotiation
                        let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
                        econfig.kvcache_dtype = allocator.resolved_kvcache_dtype();
                        let device_ids = econfig.device_ids.clone().unwrap_or(vec![0]);
                        match allocator.plan(&device_ids, &mut econfig) {
                            Ok(_) => {
                                crate::log_info!("KVCache allocation successfully planned!");
                            }
                            Err(e) => {
                                crate::log_error!("Failed to allocate KVCache: {:?}", e);
                                panic!("Failed to allocate KVCache: {:?}", e);
                            }
                        }
                        econfig
                    });

                    send_and_expect_ack(
                        &mut stream,
                        &MessageType::UsableMemoryLeft(ecfg.clone()),
                        "init kvcache",
                        rank,
                    )?;

                    crate::log_info!("Runner {} started!", rank);
                    Ok(stream)
                })
                .collect();

            if let Ok(Some(handle)) = progress_handle.join() {
                let _ = handle.join();
            }
            if let Some(cfg) = engine_config.get() {
                econfig = cfg.clone();
            }
            RunnerType::Process(runner_streams?)
        };

        if econfig.max_model_len.is_none() {
            println!(
                "\n{} is not given, default to {}, Max usable kvcache tokens {}.\n",
                format!("Warn: max_model_len").yellow().bold(),
                format!("{}", 32768).red().bold(),
                format!("{}", 32768 * econfig.max_num_seqs).red().bold(),
            );
            econfig.max_model_len = Some(32768);
        }
        let runners = Arc::new(RwLock::new(runners));

        let mut scheduler = Scheduler::new(runners.clone(), &econfig, &config);

        // Preserve model-specific tool token detection for non-guided paths.
        let mut tool_config = ToolConfig::for_model_type(&model_type);
        tool_config.validate_with_tokenizer(&tokenizer, &model_type);
        let tool_call_start_ids = tool_config.tool_call_start_ids(&tokenizer);
        let tool_call_end_ids = tool_config.tool_call_end_ids(&tokenizer);

        if !tool_call_start_ids.is_empty() {
            scheduler.set_tool_call_start_tokens(tool_call_start_ids.clone());
            log_info!(
                "Tool call start token IDs set to: {:?}",
                tool_call_start_ids
            );
        } else {
            log_info!("Tool call start token IDs not set (no reliable start token)");
        }

        if !tool_call_end_ids.is_empty() {
            scheduler.set_tool_call_end_tokens(tool_call_end_ids.clone());
            log_info!("Tool call end token IDs set to: {:?}", tool_call_end_ids);
        } else {
            log_info!("Tool call end token IDs not set (no reliable end token)");
        }

        // Set tokenizer for JSON tool call detection (for models like Qwen3 that output raw JSON)
        scheduler.set_tokenizer(Arc::new(tokenizer.clone()));

        log_warn!(
            "Maximum batched tokens {} ({} blocks x Block_Size {} for KV cache). Additional CPU KV Cache blocks {}.",
            econfig.max_num_batched_tokens,
            econfig.num_blocks,
            econfig.block_size,
            (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.2f32)) as usize
        );

        let mut template = ChatTemplate::new(
            None,
            config_tokenizer.chat_template.clone(),
            config_tokenizer.bos_token.clone(),
            config_tokenizer.eos_token.clone(),
            None,
            true,
            true,
        );
        let escaped_special_tokens = ChatTemplate::collect_escape_tokens(
            &tokenizer,
            &[&tool_config.start_token_str, &tool_config.end_token_str],
        );
        template.set_escape_tokens(escaped_special_tokens);

        let img_cfg = get_image_config(model_type.clone(), &config)?;
        if let Some(cfg) = img_cfg.as_ref() {
            template.set_preserve_tokens(cfg.prompt_marker_tokens());
        }
        let prompt_replay_candidates = Self::build_prompt_replay_candidates(
            &tokenizer,
            &template,
            &Vec::new(),
            &guidance_tokens,
        );
        let model_name = if let Some(archs) = &config.architectures {
            archs[0].to_string()
        } else {
            "default".to_string()
        };

        log_warn!("Model loaded.\n");

        let engine = Arc::new(RwLock::new(Self {
            runners,
            scheduler,
            tokenizer,
            econfig,
            default_chat_template,
            template,
            prompt_replay_candidates,
            stream_decoders: HashMap::new(),
            stream_senders: HashMap::new(),
            request_types: HashMap::new(),
            decode_start_times: HashMap::new(),
            decode_length: HashMap::new(),
            seq_prefilled_reasoning_end: HashMap::new(),
            seq_prompt_replays: HashMap::new(),
            last_check_throughput_time: 0,
            active_requests: HashSet::new(),
            cancelled_sequences: Vec::new(),
            stop_flag: stop_flag.clone(),
            has_vision: config.is_multi_model.unwrap_or(false),
            model_type,
            tool_config,
            img_cfg,
            model_name,
            guidance_tokens,
        }));

        Self::start_engine(engine.clone());
        Ok(engine)
    }

    fn add_request_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: &RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize)> {
        let tokens = self
            .tokenizer
            .encode_fast(prompt, true)
            .expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().to_vec();
        self.add_request_pretokenized_(params, prompt, token_ids, request_type, images, image_idx)
    }

    /// Same body as `add_request_` but accepts a pre-computed `token_ids`
    /// so the caller can run tokenize outside the engine write lock. Hot-path
    /// serialization of concurrent server handlers gets cut by ~
    /// chat-template + tokenize wall × concurrency in chat-template-heavy
    /// workloads.
    fn add_request_pretokenized_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        token_ids: Vec<u32>,
        request_type: &RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize)> {
        let length = token_ids.len();
        let raw_replay_token_ids = self.match_prompt_replay_candidate(&token_ids);
        if let Some(max_model_len) = self.econfig.max_model_len {
            if length > max_model_len - 1 {
                candle_core::bail!(
                    "Inputs token length {} exceed max_model_len {}",
                    length,
                    max_model_len
                );
            }
        }
        let mut params = params.clone();
        params.max_tokens = Some(
            params
                .max_tokens
                .unwrap_or(self.econfig.max_tokens.unwrap_or(16384)),
        );
        let mut max_tokens = params.max_tokens.unwrap();
        let requested_max_tokens = max_tokens;

        let max_model_len = self.econfig.max_model_len.unwrap_or(max_tokens);
        //we also need to consider prompt length
        if length + max_tokens > max_model_len {
            max_tokens = max_model_len - length;
            params.max_tokens = Some(max_tokens);
            log_warn!(
                "Adjusted max_tokens to {} to fit within max_model_len {} (with prompt length {}, requested max_tokens {})",
                max_tokens,
                max_model_len,
                length,
                requested_max_tokens
            );
        }
        if max_tokens == 0 {
            candle_core::bail!(
                "Inputs token length {} leaves no room for generated tokens within max_model_len {}",
                length,
                max_model_len
            );
        }

        if let Some(gen_cfg) = &self.econfig.generation_cfg {
            let temperature = params.temperature.or(gen_cfg.temperature);
            let top_k = params.top_k.or(gen_cfg.top_k);
            let top_p = params.top_p.or(gen_cfg.top_p);
            let frequency_penalty = params.frequency_penalty.or(gen_cfg.frequency_penalty);
            let presence_penalty = params.presence_penalty.or(gen_cfg.presence_penalty);
            params.temperature = temperature;
            params.top_k = top_k;
            params.top_p = top_p;
            params.frequency_penalty = frequency_penalty;
            params.presence_penalty = presence_penalty;
        }

        if let Some(stop_sequences) = &params.stop_sequences {
            let mut stop_token_ids = Vec::new();
            let mut resolved_stop_sequences = Vec::new();
            for sequence in stop_sequences {
                if sequence.is_empty() {
                    continue;
                }
                match self.tokenizer.encode(sequence.as_str(), false) {
                    Ok(encoding) => {
                        let ids = encoding.get_ids();
                        if !ids.is_empty() {
                            stop_token_ids.push(ids.to_vec());
                            resolved_stop_sequences.push(sequence.clone());
                        }
                    }
                    Err(err) => {
                        crate::log_warn!(
                            "Failed to encode stop sequence '{}': {:?}",
                            sequence,
                            err
                        );
                    }
                }
            }
            if !stop_token_ids.is_empty() {
                params.stop_token_ids = Some(stop_token_ids);
                params.stop_sequences = Some(resolved_stop_sequences);
            }
        }
        let seq = Sequence::new(
            token_ids,
            self.econfig.block_size,
            params,
            images,
            image_idx,
        );

        let prompt_required_blocks = self.scheduler.block_manager.required_blocks(&seq);
        let requested_decode_blocks = max_tokens.div_ceil(self.econfig.block_size);
        let minimum_decode_budget_tokens = max_tokens.min(REQUEST_ADMISSION_DECODE_BUDGET_TOKENS);
        let minimum_decode_blocks = minimum_decode_budget_tokens.div_ceil(self.econfig.block_size);
        let mut target_required_blocks =
            prompt_required_blocks.saturating_add(requested_decode_blocks);
        let mut minimum_required_blocks =
            prompt_required_blocks.saturating_add(minimum_decode_blocks);
        let mut available_blocks = self.scheduler.block_manager.get_num_free_blocks();

        while target_required_blocks > available_blocks {
            let prev_target = target_required_blocks;
            let target_required_tokens = target_required_blocks * self.econfig.block_size;
            let evicted = self
                .scheduler
                .evict_prefix_cache_until_free(target_required_blocks);
            if evicted > 0 {
                crate::log_warn!(
                    "Evicted {} prefix cache block(s) to reserve {} KV tokens for request admission (prompt + {} requested decode tokens).",
                    evicted,
                    target_required_tokens,
                    max_tokens
                );
            }
            if evicted == 0 && !self.try_release_cache(target_required_tokens) {
                break;
            }
            let prompt_required_blocks = self.scheduler.block_manager.required_blocks(&seq);
            target_required_blocks = prompt_required_blocks.saturating_add(requested_decode_blocks);
            minimum_required_blocks = prompt_required_blocks.saturating_add(minimum_decode_blocks);
            available_blocks = self.scheduler.block_manager.get_num_free_blocks();

            if evicted > 0 && target_required_blocks >= prev_target {
                break;
            }
        }

        if minimum_required_blocks > available_blocks {
            let available_tokens = available_blocks * self.econfig.block_size;
            let required_tokens = minimum_required_blocks * self.econfig.block_size;
            candle_core::bail!(
                "Remaining {} kvcache tokens, but your request requires at least {} tokens (prompt plus {} decode budget tokens), please request later!",
                available_tokens,
                required_tokens,
                minimum_decode_budget_tokens
            );
        }
        if target_required_blocks > available_blocks {
            crate::log_warn!(
                "Request admitted with {} KV tokens available, below requested reservation {} tokens but enough for prompt plus {} decode budget tokens.",
                available_blocks * self.econfig.block_size,
                target_required_blocks * self.econfig.block_size,
                minimum_decode_budget_tokens
            );
        }
        let seq_id = self.scheduler.add(seq);
        if let Some(end_marker) = detect_prefilled_reasoning_end_marker(prompt) {
            self.seq_prefilled_reasoning_end.insert(seq_id, end_marker);
        }
        if let Some(replay_ids) = raw_replay_token_ids {
            self.seq_prompt_replays.insert(seq_id, replay_ids);
        }

        if *request_type == RequestType::Stream {
            let tokenizer = self.tokenizer.clone();
            let leaked: &'static _ = Box::leak(Box::new(tokenizer));
            let decoder = leaked.decode_stream(false);
            let wrapped = super::StreamWithTokenizer {
                _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                stream: decoder,
            };
            let boxed_decoder: Box<dyn super::DecodeStreamTrait + Send + Sync> = Box::new(wrapped);
            self.stream_decoders.insert(seq_id, boxed_decoder);
        }
        self.active_requests.insert(seq_id);
        Ok((seq_id, length))
    }

    pub fn add_request(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) =
            self.add_request_(params, prompt, &request_type, images, image_idx)?;
        let (tx, rx) = channel(1024);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if self.econfig.server_mode.unwrap_or(true) && request_type != RequestType::Completion {
            log_warn!(
                "[{:?}] New request [Seq_id {}, {} tokens] received! (session_id: {:?})\n",
                request_type,
                seq_id,
                prompt_length,
                params.session_id,
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    /// Pretokenized variant of `add_request`. Caller has already done
    /// tokenize (typically via `preprocess`) outside the engine write
    /// lock, so the only work that needs `&mut self` is the scheduler-add.
    pub fn add_request_pretokenized(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        token_ids: Vec<u32>,
        request_type: RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) = self.add_request_pretokenized_(
            params,
            prompt,
            token_ids,
            &request_type,
            images,
            image_idx,
        )?;
        let (tx, rx) = channel(1024);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if self.econfig.server_mode.unwrap_or(true) && request_type != RequestType::Completion {
            log_warn!(
                "[{:?}] New request [Seq_id {}, {} tokens] received! (session_id: {:?})\n",
                request_type,
                seq_id,
                prompt_length,
                params.session_id,
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    /// Apply chat template + tokenize each `(params, messages)` pair under
    /// shared access (`&self`) so N concurrent server handlers can run this
    /// in parallel under their own `engine.read()` guards. Returns a list
    /// of `PreprocessedRequest`s that feed `generate_sync`
    /// or `generate_stream`, which acquire `engine.write()`
    /// only briefly for the scheduler-add step.
    pub fn preprocess(
        &self,
        params: &[SamplingParams],
        message_list: &[Vec<Message>],
        tools: &[Tool],
        log: bool,
    ) -> Result<Vec<PreprocessedRequest>> {
        if params.len() != message_list.len() {
            candle_core::bail!("size of sampling parameters is not match with size of prompts!");
        }
        let tools_vec: Vec<Tool> = tools.to_vec();
        let mut out = Vec::with_capacity(params.len());
        for (param, messages) in params.iter().zip(message_list.iter()) {
            let (prompt, image_idx) = self.apply_chat_template(param, messages, &tools_vec, log);
            let tokens = self
                .tokenizer
                .encode_fast(prompt.as_str(), true)
                .expect("encode failed!");
            let token_ids: Vec<u32> = tokens.get_ids().to_vec();
            out.push(PreprocessedRequest {
                params: param.clone(),
                prompt,
                image_idx,
                token_ids,
            });
        }
        Ok(out)
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        self.scheduler.get_num_cached_tokens()
    }

    /// See [`Scheduler::get_num_cached_tokens_for_seq`].
    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.scheduler.get_num_cached_tokens_for_seq(seq_id)
    }

    fn trim_prompt_replay_prefix(
        replay_ids: &[u32],
        guidance_tokens: &GuidanceTokens,
    ) -> Option<Vec<u32>> {
        let start_idx = replay_ids
            .iter()
            .position(|token_id| guidance_tokens.reasoning_start_ids.contains(token_id))?;
        Some(replay_ids[start_idx..].to_vec())
    }

    fn build_prompt_replay_candidates(
        tokenizer: &Tokenizer,
        template: &ChatTemplate,
        tools: &Vec<Tool>,
        guidance_tokens: &GuidanceTokens,
    ) -> Vec<Vec<u32>> {
        let synthetic_messages = vec![Message {
            role: "user".to_string(),
            content: "__XINFER_REPLAY_PROBE__".to_string(),
            num_images: 0,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mut candidates = Vec::new();

        for enable_thinking in [true, false] {
            let mut replay_template = template.clone();
            replay_template.set_enable_thinking(enable_thinking);
            replay_template.set_messages(&synthetic_messages);
            let rendered = match replay_template.apply_chat_template(tools, false) {
                Ok(rendered) => rendered,
                Err(_) => continue,
            };
            let Some(replay_suffix) =
                replay_template.generation_prompt_replay_suffix(tools, &rendered)
            else {
                continue;
            };
            let Ok(encoding) = tokenizer.encode_fast(replay_suffix.as_str(), true) else {
                continue;
            };
            let ids = encoding.get_ids().to_vec();
            if let Some(replay_ids) = Self::trim_prompt_replay_prefix(&ids, guidance_tokens) {
                // crate::log_info!(
                //     "Missing suffix detected {} -> {:?}",
                //     replay_suffix,
                //     replay_ids
                // );
                candidates.push(replay_ids);
            }
        }

        candidates.sort_by_key(|ids| std::cmp::Reverse(ids.len()));
        candidates.dedup();
        candidates
    }

    fn match_prompt_replay_candidate(&self, prompt_token_ids: &[u32]) -> Option<Vec<u32>> {
        self.prompt_replay_candidates
            .iter()
            .find(|candidate| prompt_token_ids.ends_with(candidate.as_slice()))
            .cloned()
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        self.scheduler.get_available_kv_tokens()
    }

    fn take_prompt_replay_token_ids(&mut self, seq_id: usize) -> Option<Vec<u32>> {
        self.seq_prompt_replays.remove(&seq_id)
    }

    pub fn notify_runner_finished(&mut self, id: usize) -> Result<()> {
        match &mut *self.runners.write() {
            RunnerType::Thread(model_runner) => Ok(model_runner.finished(id)),
            RunnerType::Process(ref mut runner_streams) => {
                for stream in runner_streams {
                    send_local(
                        &mut vec![stream.try_clone()?],
                        &MessageType::FinishDecode(id),
                        false,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Phase 1: Schedule sequences and prepare data for the forward pass.
    /// Returns (scheduled_ids, is_prefill, cloned_sequences) or None if no work.
    pub fn prepare_step(&mut self) -> Result<Option<(Vec<usize>, bool, Vec<Sequence>)>> {
        let (scheduled_ids, is_prefill) = match self.scheduler.schedule() {
            Ok((ids, prefill)) => (ids, prefill),
            Err(_) => (vec![], true),
        };
        if scheduled_ids.is_empty() {
            return Ok(None);
        }

        if is_prefill {
            let downgraded = self
                .scheduler
                .fallback_missing_mamba_prefix_snapshots(&scheduled_ids)?;
            if downgraded > 0 {
                crate::log_warn!(
                    "Prefill fallback: {} sequence(s) downgraded to full prefill due to missing mamba snapshots.",
                    downgraded
                );
            }
        }

        let seqs = self.scheduler.get_sequences(&scheduled_ids);
        let owned_seqs: Vec<Sequence> = seqs.iter().map(|s| (*s).clone()).collect();
        Ok(Some((scheduled_ids, is_prefill, owned_seqs)))
    }

    /// Phase 2: Run the forward pass. Does NOT require the engine lock - only the runner lock.
    pub fn run_forward(
        runners: &Arc<RwLock<RunnerType>>,
        owned_seqs: &[Sequence],
        is_prefill: bool,
    ) -> Result<Vec<u32>> {
        match &mut *runners.write() {
            RunnerType::Thread(model_runner) => {
                let seq_refs: Vec<&Sequence> = owned_seqs.iter().collect();
                model_runner.run(Seqs::SeqRefs(&seq_refs), is_prefill)
            }
            RunnerType::Process(ref mut runner_streams) => {
                let request = if is_prefill {
                    MessageType::RunPrefill((owned_seqs.to_vec(), true))
                } else {
                    let sequences = owned_seqs
                        .iter()
                        .map(|s| DecodeSequence::new(s))
                        .collect::<Vec<_>>();
                    MessageType::RunDecode((sequences, false))
                };

                let cloned_streams: Vec<LocalStream> = runner_streams
                    .iter_mut()
                    .map(|s| s.try_clone().expect("clone failed"))
                    .collect();

                let all_outputs: Result<Vec<Vec<u32>>> = cloned_streams
                    .into_par_iter()
                    .map(|mut stream| {
                        let msg = request.clone();
                        send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                        let response = receive_local(&mut stream, false)?;

                        match response {
                            MessageType::RunResponse(output_ids) => {
                                if output_ids.len() == 0 {
                                    candle_core::bail!("Runner step error, no response!")
                                } else {
                                    Ok(output_ids)
                                }
                            }
                            other => {
                                candle_core::bail!("Unexpected response type: {:?}", other)
                            }
                        }
                    })
                    .collect();

                let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
                if let Some(output_ids) = all_outputs.first() {
                    Ok(output_ids.clone())
                } else {
                    candle_core::bail!("No output ids received from model runners");
                }
            }
        }
    }

    /// Phase 3: Postprocess forward pass results, deliver tokens, and do maintenance.
    pub fn finish_step(
        &mut self,
        scheduled_ids: Vec<usize>,
        is_prefill: bool,
        output_ids: Vec<u32>,
    ) -> Result<usize> {
        pub struct DecodedIds(Either<Vec<usize>, Vec<usize>>);

        let decoded_ids = if is_prefill {
            let (indices, finished_indices) =
                self.scheduler.filter_prefill_finished(&scheduled_ids);
            if indices.is_empty() {
                self.check_canceled(None);
                return Ok(0);
            } else {
                let output_ids: Vec<u32> = indices.iter().map(|&i| output_ids[i]).collect();
                self.scheduler.postprocess(&finished_indices, &output_ids);
                DecodedIds(Either::Left(finished_indices))
            }
        } else {
            self.scheduler.postprocess(&scheduled_ids, &output_ids);
            DecodedIds(Either::Left(scheduled_ids))
        };

        let (indices, is_running): (&Vec<usize>, bool) = match &decoded_ids.0 {
            Either::Left(indices) => (indices, true),
            Either::Right(indices) => (indices, false),
        };

        let mut prefill_batch: Vec<(usize, usize, f32, bool)> = Vec::new();
        for &idx in indices {
            let sq = if is_running {
                self.scheduler.get_running(idx)
            } else {
                self.scheduler.get_waiting(idx)
            };
            if let Some(s) = sq {
                let seq_id = s.id;
                if s.is_finished() {
                    // Normal finish handling

                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            let prompt_start_time = s.created_time();

                            let decode_finish_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis()
                                as usize;
                            let decode_start_time = if let Some(decode_start_time) =
                                self.decode_start_times.get(&seq_id)
                            {
                                *decode_start_time
                            } else {
                                decode_finish_time
                            };

                            if s.is_tool_call_end {
                                //finish early, we need to send the last token
                                if *request_type == RequestType::Stream {
                                    if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                        if let Some(tok) = decoder.step(s.last_token)? {
                                            let _ = sender
                                                .try_send(StreamItem::Token(tok, s.last_token));
                                        }
                                    }
                                } else {
                                    let _ = sender.try_send(StreamItem::TokenID(s.last_token));
                                }
                            }

                            if *request_type == RequestType::Stream {
                                let _ = sender.try_send(StreamItem::Done((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.len(),
                                    s.stop_sequence.clone(),
                                )));
                            } else {
                                let _ = sender.try_send(StreamItem::Completion((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.clone(),
                                    s.stop_sequence.clone(),
                                )));
                            }
                        }
                    }

                    self.stream_senders.remove(&seq_id);
                    self.stream_decoders.remove(&seq_id);
                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                    self.decode_start_times.remove(&seq_id);
                    self.decode_length.remove(&seq_id);
                    self.seq_prefilled_reasoning_end.remove(&seq_id);
                    self.seq_prompt_replays.remove(&seq_id);
                    let _ = self.notify_runner_finished(seq_id);
                    if self.econfig.server_mode.unwrap_or(true) {
                        self.scheduler.print_free_blocks();
                    }
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        let cur_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        self.decode_start_times.insert(seq_id, cur_time);
                        self.decode_length.insert(seq_id, 1);

                        let time_costs = cur_time - s.created_time();
                        if time_costs / 100 > 0 && s.len() > 0 {
                            prefill_batch.push((
                                seq_id,
                                s.len(),
                                time_costs as f32 / 1000f32,
                                s.num_cached_tokens > 0,
                            ));
                        }
                    }

                    if let Some(length) = self.decode_length.get_mut(&seq_id) {
                        *length = s.output_len();
                    }

                    let mut token_ids =
                        if self.is_pd_mode() && s.pd_first_token.is_some() && s.output_len() == 2 {
                            // Special case, the real first token is generated on PD server
                            vec![s.pd_first_token.unwrap_or(s.last_token), s.last_token]
                        } else {
                            vec![s.last_token]
                        };
                    if let Some(mut replay_ids) = self.take_prompt_replay_token_ids(seq_id) {
                        replay_ids.extend(token_ids);
                        token_ids = replay_ids;
                    }

                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            if *request_type == RequestType::Stream {
                                if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                    for token_id in token_ids {
                                        if let Some(tok) = decoder.step(token_id)? {
                                            let result =
                                                sender.try_send(StreamItem::Token(tok, token_id));
                                            if result.is_err() {
                                                crate::log_error!(
                                                    "Error when sending token to client [seq_id {}]",
                                                    seq_id
                                                );
                                                self.cancelled_sequences.push(seq_id);
                                            }
                                        }
                                    }
                                }
                            } else {
                                //completion request will be decoded at the final stage (at once)
                                for token_id in token_ids {
                                    /*
                                        Check if the receiver is still active.
                                        If the client disconnected, collect_sync_results will be dropped,
                                        dropping the receiver, causing try_send to fail.
                                    */
                                    if let Err(_) = sender.try_send(StreamItem::TokenID(token_id)) {
                                        crate::log_error!(
                                            "Error when sending token to client [seq_id {}]",
                                            seq_id
                                        );
                                        self.cancelled_sequences.push(seq_id);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !prefill_batch.is_empty() {
            let seq_ids: Vec<usize> = prefill_batch.iter().map(|(id, _, _, _)| *id).collect();
            let total_tokens: usize = prefill_batch.iter().map(|(_, t, _, _)| *t).sum();
            let max_time: f32 = prefill_batch
                .iter()
                .map(|(_, _, t, _)| *t)
                .fold(0.0f32, f32::max);
            let has_cache = prefill_batch.iter().any(|(_, _, _, c)| *c);
            if max_time > 0.0 {
                crate::log_info!(
                    "Prefilling {} seq(s) {:?}: {} total tokens in {:.2}s ({:.2} tokens/s{})",
                    prefill_batch.len(),
                    seq_ids,
                    total_tokens,
                    max_time,
                    total_tokens as f32 / max_time,
                    if has_cache { ", cache included" } else { "" },
                );
            }
        }
        self.scheduler.clear_finished();

        if indices.is_empty()
            && self.scheduler.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD + 0.01f32
        {
            if !self.try_release_cache(0) {
                if let Some(oldest_seq_id) = self.active_requests.clone().iter().min() {
                    crate::log_error!(
                        "Unable to schedule task(s), drop the oldest active request (seq_id: {:?})",
                        oldest_seq_id
                    );
                    self.cancelled_sequences.push(*oldest_seq_id);
                    self.check_canceled(Some(
                        "Unable to schedule task(s), this request has been dropped!".to_string(),
                    ));
                }
                if self.scheduler.kv_cache_usage_percent() > 0.99f32 {
                    self.free_resources();
                }
            }
        } else {
            self.check_canceled(None);
        }
        if self.econfig.server_mode.unwrap_or(true) && is_running {
            self.may_print_decoding_throughput(&indices);
        }
        Ok(indices.len())
    }

    pub fn try_release_cache(&mut self, tokens_required: usize) -> bool {
        if tokens_required == 0 {
            return self.scheduler.evict_prefix_cache_blocks(1) > 0;
        }

        let required_blocks = tokens_required.div_ceil(self.econfig.block_size);
        self.scheduler
            .evict_prefix_cache_until_free(required_blocks)
            > 0
    }

    pub fn check_canceled(&mut self, reason: Option<String>) {
        if self.cancelled_sequences.is_empty() {
            return;
        }
        for i in 0..self.cancelled_sequences.len() {
            let seq_id = self.cancelled_sequences[i];
            self.scheduler.cancel(seq_id);
            // Ensure model-side per-sequence state (e.g., Qwen3.5 Mamba cache slot) is released.
            let _ = self.notify_runner_finished(seq_id);
            if let Some(_) = self.active_requests.get(&seq_id) {
                self.active_requests.remove(&seq_id);
            }
            self.stream_decoders.remove(&seq_id);
            self.decode_start_times.remove(&seq_id);
            self.seq_prefilled_reasoning_end.remove(&seq_id);
            self.seq_prompt_replays.remove(&seq_id);
            if let Some(r) = &reason {
                if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                    if let Some(request_type) = self.request_types.get(&seq_id) {
                        if *request_type == RequestType::Stream {
                            let _ = sender.try_send(StreamItem::Error(r.clone()));
                        }
                    }
                }
            }
            self.stream_senders.remove(&seq_id);
        }
        self.scheduler.clear_finished();
        self.cancelled_sequences.clear();
    }

    pub fn may_print_decoding_throughput(&mut self, active_indices: &Vec<usize>) {
        if active_indices.is_empty() || self.active_requests.is_empty() {
            return;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        if cur_time - self.last_check_throughput_time < 5000 {
            return;
        }
        self.last_check_throughput_time = cur_time;

        let mut total_decoded_length = 0;
        let mut total_decoded_time_costs = 0;

        let mut seq_ids = Vec::new();
        for &idx in active_indices {
            if let Some(seq) = self.scheduler.get_running(idx) {
                if let Some(length) = self.decode_length.get(&seq.id) {
                    total_decoded_length += length;
                }
                if let Some(start_time) = self.decode_start_times.get(&seq.id) {
                    total_decoded_time_costs += cur_time - start_time;
                }
                seq_ids.push(seq.id);
            }
        }

        if total_decoded_length > 0 && total_decoded_time_costs / 1000 > 0 {
            let avg_throughput = total_decoded_length / (total_decoded_time_costs / 1000);
            crate::log_info!(
                "Decoding: {} active request(s) [Seq: {:?}], avg. {} tokens/s per request (total: {} tokens/s)",
                active_indices.len(),
                seq_ids,
                avg_throughput,
                avg_throughput * active_indices.len()
            )
        }

        if total_decoded_length % 100 > 50 {
            self.scheduler.print_free_blocks();
        }
    }

    pub fn cancel(&mut self, seq_id: usize) {
        self.cancelled_sequences.push(seq_id);
    }

    pub fn cancel_all_with_reason(&mut self, reason: Option<String>) -> bool {
        for seq_id in &self.active_requests {
            self.cancelled_sequences.push(*seq_id);
        }
        let has_requests_to_cancel = self.cancelled_sequences.len() > 0;
        self.check_canceled(reason);
        has_requests_to_cancel
    }

    pub fn apply_chat_template(
        &self,
        params: &SamplingParams,
        messages: &Vec<Message>,
        tools: &Vec<Tool>,
        log: bool,
    ) -> (String, i32) {
        // let mut collected_images = Vec::new();
        let mut prompt_template = self.template.clone();
        prompt_template
            .set_enable_thinking(params.thinking.unwrap_or(!self.econfig.disable_reasoning));
        prompt_template.set_messages(messages);
        let image_idx: i32 = 0;
        let prompt_processed = prompt_template
            .apply_chat_template(tools, log)
            .map_err(candle_core::Error::wrap);
        let prompt = if prompt_processed.is_ok() {
            prompt_processed.unwrap()
        } else {
            if log {
                crate::log_error!(
                    "Applying Chat Template failed: {:?}, use default template!",
                    prompt_processed
                );
            }

            let mut prompt = "".to_string();
            for message in messages {
                if message.role == "user" {
                    let escaped = prompt_template.escape_text(&message.content);
                    prompt += &self.default_chat_template.replace("{}", &escaped);
                    prompt += "\n";
                }
            }
            prompt
        };
        if log {
            log_info!(
                "Prompt after applying Chat Template: {}",
                prompt.replace("\n", "")
            );
        }
        (prompt, image_idx)
    }

    pub fn is_idle(&self) -> bool {
        self.active_requests.is_empty()
    }

    pub fn is_pd_mode(&self) -> bool {
        self.econfig.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    /// Registers preprocessed completion requests. Callers should run
    /// `preprocess(...)` under `engine.read()` first, then call this under
    /// `engine.write()` so the write lock is held only for request admission.
    pub fn generate_sync(
        &mut self,
        preprocessed: Vec<PreprocessedRequest>,
        images: Option<ImageData>,
        logger: &Option<Arc<ChatCompletionLogger>>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        let mut receivers = Vec::with_capacity(preprocessed.len());
        for pp in preprocessed {
            if let Some(ref l) = logger {
                l.log_prompt(&pp.prompt);
            }
            if let Ok((seq_id, prompt_length, rx)) = self.add_request_pretokenized(
                &pp.params,
                &pp.prompt,
                pp.token_ids,
                RequestType::Completion,
                &images,
                pp.image_idx,
            ) {
                receivers.push((seq_id, prompt_length, rx));
            }
        }
        Ok(receivers)
    }

    pub async fn collect_sync_results(
        receivers: Vec<(usize, usize, mpsc::Receiver<StreamItem>)>,
        tokenizer: Arc<Tokenizer>,
        logger: Option<Arc<ChatCompletionLogger>>,
    ) -> Result<Vec<GenerationOutput>> {
        let decoded_tokens = Arc::new(AtomicUsize::new(0));
        let decode_start_time = Arc::new(AtomicUsize::new(0));

        // Create futures for each receiver (do NOT spawn detached tasks)
        let tasks = receivers
            .into_iter()
            .map(|(seq_id, prompt_length, mut rx)| {
                let decoded_tokens = decoded_tokens.clone();
                let decode_start_time_clone = decode_start_time.clone();
                let tokenizer = Arc::clone(&tokenizer);
                let logger = logger.clone();
                async move {
                    let mut output: Option<GenerationOutput> = None;
                    let mut collected_token_ids: Vec<u32> = Vec::new();

                    // Initialize decoder for incremental logging if needed
                    let mut decoder = if logger.is_some() {
                        let tokenizer_clone = tokenizer.as_ref().clone();
                        let leaked: &'static _ = Box::leak(Box::new(tokenizer_clone));
                        let decoder = leaked.decode_stream(true);
                        let wrapped = super::StreamWithTokenizer {
                            _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                            stream: decoder,
                        };
                        Some(Box::new(wrapped) as Box<dyn super::DecodeStreamTrait + Send + Sync>)
                    } else {
                        None
                    };

                    while let Some(msg) = rx.recv().await {
                        match msg {
                            StreamItem::Completion((
                                prompt_start,
                                decode_start,
                                decode_finish,
                                decoded_ids,
                                stop_sequence,
                            )) => {
                                let decoded_len = decoded_ids.len();
                                let decode_output = tokenizer
                                    .decode(&decoded_ids, true)
                                    .expect("unable to decode!");

                                output = Some(GenerationOutput {
                                    seq_id,
                                    prompt_length,
                                    prompt_start_time: prompt_start,
                                    decode_start_time: decode_start,
                                    decode_finish_time: decode_finish,
                                    decoded_length: decoded_len,
                                    decode_output,
                                    stop_sequence,
                                });
                                break;
                            }
                            StreamItem::Token(_, _) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);

                                decode_start_time_clone
                                    .compare_exchange(
                                        0,
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as usize,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    )
                                    .ok();
                            }
                            StreamItem::TokenID(id) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);
                                collected_token_ids.push(id);

                                if let Some(d) = decoder.as_mut() {
                                    if let Ok(Some(text)) = d.step(id) {
                                        if let Some(l) = &logger {
                                            l.log_stream_token(&text);
                                        }
                                    }
                                }

                                decode_start_time_clone
                                    .compare_exchange(
                                        0,
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as usize,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    )
                                    .ok();
                            }
                            StreamItem::Done(_) => {
                                crate::log_info!("Sequence [seq_id {}] finished!", seq_id);
                                break;
                            }
                            StreamItem::Error(e) => {
                                crate::log_error!("Error in Seq {}: {}", seq_id, e);
                                break;
                            }
                        }
                    }

                    output
                }
            });

        // Wait for all decoding tasks
        let outputs = join_all(tasks).await;

        // Collect successful outputs
        let results: Vec<_> = outputs.into_iter().filter_map(|r| r).collect();

        Ok(results)
    }

    pub fn free_resources(&mut self) {
        crate::log_error!("***Release all resources for future usage!");
        self.scheduler.clear_finished();
        self.scheduler.release_waitings();
        self.seq_prefilled_reasoning_end.clear();
        self.seq_prompt_replays.clear();
    }

    /// Returns the reasoning end marker when the prompt for this sequence
    /// already ended with an opening think marker.
    pub fn get_prefilled_reasoning_end_marker(&self, seq_id: usize) -> Option<String> {
        self.seq_prefilled_reasoning_end.get(&seq_id).cloned()
    }

    /// Registers a preprocessed streaming request. Callers should run
    /// `preprocess(...)` under `engine.read()` first, then call this under
    /// `engine.write()` so the write lock is held only for request admission.
    pub fn generate_stream(
        &mut self,
        preprocessed: PreprocessedRequest,
        images: Option<ImageData>,
        logger: &Option<Arc<ChatCompletionLogger>>,
    ) -> Result<(usize, usize, Option<String>, mpsc::Receiver<StreamItem>)> {
        if let Some(ref l) = logger {
            l.log_prompt(&preprocessed.prompt);
        }
        match self.add_request_pretokenized(
            &preprocessed.params,
            &preprocessed.prompt,
            preprocessed.token_ids,
            RequestType::Stream,
            &images,
            preprocessed.image_idx,
        ) {
            Ok((seq_id, prompt_length, rx)) => {
                let prefilled_reasoning_end = self.get_prefilled_reasoning_end_marker(seq_id);
                Ok((seq_id, prompt_length, prefilled_reasoning_end, rx))
            }
            Err(e) => {
                candle_core::bail!("{:?}", e)
            }
        }
    }

    pub fn get_usage_stats(&self, session_id: Option<String>) -> Result<UsageResponse> {
        match session_id {
            Some(sid) => {
                let available_kvcache_tokens = self.scheduler.get_available_kv_tokens();
                let max_model_len = self
                    .econfig
                    .max_model_len
                    .ok_or_else(|| candle_core::Error::msg("max_model_len not set!"))?;
                let (seq_id, status) =
                    self.scheduler.find_seq_by_session_id(&sid).ok_or_else(|| {
                        candle_core::Error::msg(format!("Seq with session_id {} not found", sid))
                    })?;
                let token_used = self.scheduler.get_seq_token_usage(seq_id)?;

                let total_kv_cache_tokens = self.scheduler.get_total_kv_tokens();
                let (swap_used, total_swap_memory) = self.scheduler.get_cpu_swap_usage();

                let session_status = match status {
                    SequenceStatus::FinishSwapped | SequenceStatus::Swapped => {
                        "Swapped".to_string()
                    }
                    _ => status.to_string(),
                };

                Ok(UsageResponse {
                    token_used,
                    max_model_len,
                    used_kvcache_tokens: total_kv_cache_tokens - available_kvcache_tokens,
                    total_kv_cache_tokens,
                    swap_used,
                    total_swap_memory,
                    session_status,
                })
            }
            _ => {
                candle_core::bail!("No session_id provided")
            }
        }
    }

    pub fn embed(
        &mut self,
        inputs: &[String],
        strategy: EmbeddingStrategy,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let mut outputs = Vec::new();
        let mut prompt_tokens = 0;

        for input in inputs {
            let tokens = self
                .tokenizer
                .encode_fast(input.as_str(), true)
                .map_err(candle_core::Error::wrap)?;
            let token_ids: Vec<u32> = tokens.get_ids().to_vec();
            if token_ids.is_empty() {
                candle_core::bail!("Embedding input cannot be empty");
            }

            if let Some(max_model_len) = self.econfig.max_model_len {
                if token_ids.len() > max_model_len - 1 {
                    candle_core::bail!(
                        "Embedding input length {} exceeds max_model_len {}",
                        token_ids.len(),
                        max_model_len
                    );
                }
            }

            let mut seq = Sequence::new(
                token_ids.clone(),
                self.econfig.block_size,
                SamplingParams::default(),
                &None,
                -1,
            );
            let required_blocks = self.scheduler.block_manager.required_blocks(&seq);
            let available_blocks = self.scheduler.block_manager.get_num_free_blocks();
            if required_blocks > available_blocks {
                let available_tokens = available_blocks * self.econfig.block_size;
                let required_tokens = required_blocks * self.econfig.block_size;
                candle_core::bail!(
                    "Remaining {} kvcache tokens, but embedding requires {} new tokens, please retry later",
                    available_tokens,
                    required_tokens
                );
            }
            self.scheduler.block_manager.allocate(&mut seq)?;

            let mut chunked_mean: Option<Vec<f32>> = None;
            let mut last_vec: Option<Vec<f32>> = None;
            let mut processed_tokens = 0usize;
            let chunk_size = self.econfig.effective_prefill_chunk_size();

            while seq.num_cached_tokens < seq.len() {
                let remaining = seq.len() - seq.num_cached_tokens;
                let chunk_tokens = std::cmp::min(chunk_size, remaining);
                let embedding_result = match &mut *self.runners.write() {
                    RunnerType::Thread(model_runner) => model_runner.embed(&[&seq], &strategy),
                    RunnerType::Process(ref mut runner_streams) => {
                        let request = MessageType::RunEmbed((vec![seq.clone()], strategy.clone()));
                        let cloned_streams: Vec<LocalStream> = runner_streams
                            .iter_mut()
                            .map(|s| s.try_clone().expect("clone failed"))
                            .collect();

                        let all_outputs: Result<Vec<Vec<Vec<f32>>>> = cloned_streams
                            .into_par_iter()
                            .map(|mut stream| {
                                let msg = request.clone();
                                send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                                let response = receive_local(&mut stream, false)?;

                                match response {
                                    MessageType::RunResponseEmbed(output_embed) => {
                                        if output_embed.len() == 0 {
                                            candle_core::bail!("Runner step error, no response!")
                                        } else {
                                            Ok(output_embed)
                                        }
                                    }
                                    other => {
                                        candle_core::bail!("Unexpected response type: {:?}", other)
                                    }
                                }
                            })
                            .collect();
                        let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
                        if let Some(output_embed) = all_outputs.first() {
                            Ok(output_embed.clone())
                        } else {
                            candle_core::bail!(
                                "No output embedding states received from model runners"
                            );
                        }
                    }
                };
                let embedding_vecs = match embedding_result {
                    Ok(v) => v,
                    Err(e) => {
                        self.scheduler.block_manager.deallocate(&seq);
                        return Err(e);
                    }
                };
                if let Some(vec) = embedding_vecs.into_iter().next() {
                    match strategy {
                        EmbeddingStrategy::Mean => {
                            if chunked_mean.is_none() {
                                chunked_mean = Some(vec![0.0; vec.len()]);
                            }
                            if let Some(sum) = chunked_mean.as_mut() {
                                for (dst, val) in sum.iter_mut().zip(vec.iter()) {
                                    *dst += val * chunk_tokens as f32;
                                }
                            }
                        }
                        EmbeddingStrategy::Last => {
                            last_vec = Some(vec);
                        }
                    }
                }

                seq.num_cached_tokens += chunk_tokens;
                processed_tokens += chunk_tokens;
                if seq.len() > chunk_size {
                    if chunk_tokens < chunk_size {
                        crate::log_info!(
                            "Embedding chunk prefilled finished {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    } else {
                        crate::log_info!(
                            "Embedding chunk prefilled {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    }
                }
            }

            self.scheduler.block_manager.deallocate(&seq);
            prompt_tokens += seq.len();

            let final_vec = match strategy {
                EmbeddingStrategy::Mean => {
                    let mut sum = chunked_mean.unwrap_or_default();
                    if processed_tokens > 0 {
                        for v in sum.iter_mut() {
                            *v /= processed_tokens as f32;
                        }
                    }
                    sum
                }
                EmbeddingStrategy::Last => last_vec.unwrap_or_default(),
            };
            outputs.push(final_vec);
        }

        Ok((outputs, prompt_tokens))
    }

    pub fn start_engine(engine: Arc<RwLock<Self>>) {
        GLOBAL_RT.spawn(async move {
            let engine = engine.clone();
            let (is_pd_server, runners) = {
                let guard = engine.read();
                (
                    guard.is_pd_mode() && guard.is_pd_server(),
                    guard.runners.clone(),
                )
            };
            loop {
                let idle = {
                    let guard = engine.read();
                    guard.is_idle() && !is_pd_server
                };

                if idle {
                    tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                    continue;
                }

                let mut task_processed = 0;

                // Phase 1: Schedule (engine lock held briefly)
                let prep = {
                    let mut guard = engine.write();
                    match guard.prepare_step() {
                        Ok(p) => p,
                        Err(e) => {
                            crate::log_error!("[Engine Loop] Prepare error: {:?}", e);
                            if !guard.cancel_all_with_reason(Some(e.to_string())) {
                                std::process::exit(1);
                            }
                            None
                        }
                    }
                };
                // Engine lock released -- server can accept new requests during forward pass

                if let Some((scheduled_ids, is_prefill, owned_seqs)) = prep {
                    // Phase 2: Forward pass (only runner lock, engine lock FREE)
                    match Self::run_forward(&runners, &owned_seqs, is_prefill) {
                        Ok(output_ids) => {
                            // Phase 3: Postprocess (engine lock held briefly)
                            let mut guard = engine.write();
                            match guard.finish_step(scheduled_ids, is_prefill, output_ids) {
                                Ok(n) => task_processed = n,
                                Err(e) => {
                                    crate::log_error!("[Engine Loop] Finish error: {:?}", e);
                                    if !guard.cancel_all_with_reason(Some(e.to_string())) {
                                        std::process::exit(1);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            crate::log_error!("[Engine Loop] Forward error: {:?}", e);
                            let mut guard = engine.write();
                            if !guard.cancel_all_with_reason(Some(e.to_string())) {
                                std::process::exit(1);
                            }
                        }
                    }
                }

                if task_processed == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(if is_pd_server {
                        PD_PREFILL_STATUS_CHECK_COOLING_PERIOD as u64
                    } else {
                        1
                    }))
                    .await;
                }
                tokio::task::yield_now().await;
            }
        });
    }

    pub fn get_model_info(&self) -> (bool, String, Option<usize>) {
        (
            self.has_vision,
            self.model_name.clone(),
            self.econfig.max_model_len,
        )
    }

    /// Get a clone of the chat template for external use (e.g., tokenization without generation)
    pub fn get_chat_template(&self) -> ChatTemplate {
        self.template.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::LLMEngine;
    use crate::utils::guidance::GuidanceTokens;

    #[test]
    fn trim_prompt_replay_prefix_accepts_single_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42, 99],
            reasoning_end_ids: vec![100],
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[99], &guidance_tokens),
            Some(vec![99])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_accepts_multi_token_suffix_when_first_token_is_reasoning() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[42, 7], &guidance_tokens),
            Some(vec![42, 7])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_trims_leading_non_reasoning_tokens() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 42, 8], &guidance_tokens),
            Some(vec![42, 8])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_suffix_without_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 8], &guidance_tokens),
            None
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_empty_suffix() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[], &guidance_tokens),
            None
        );
    }
}
