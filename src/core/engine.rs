//src/core/engine.rs
use super::runner::ModelRunner;
use super::block_manager::BlockManager;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use super::qos::SeqReport;
use crate::core::scheduler::PD_PREFILL_STATUS_CHECK_COOLING_PERIOD;
use crate::core::sequence::SequenceStatus;
use crate::core::lora::{DEFAULT_LORA_MAX_BYTES, DEFAULT_LORA_MAX_TENANTS};
use crate::core::GenerationOutput;
use crate::models::layers::distributed::Comm;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;
use crate::models::layers::VarBuilderX;
use crate::runner::{
    send_and_expect_ack, MessageType, RunnerInitRequest,
};
use crate::server::logger::ChatCompletionLogger;
use crate::server::parser::{detect_prefilled_reasoning_end_marker, ToolConfig};
use crate::server::{EmbeddingStrategy, UsageResponse};
use crate::tools::Tool;
use crate::domain::fwd::FwdMessage;
use crate::domain::http::HttpMessage;
use crate::domain::sched::SchedMessage;
use crate::domain::stream::StreamMessage;
use crate::domain::{start_fwd_actor, start_http_actor, start_kv_actor, start_sched_actor, start_stream_actor};
use ractor::rpc::CallResult;
use crate::transfer::PdRole;
use crate::transfer::Transfer;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, EosTokenId, ModelType, SamplingParams};
use crate::utils::guidance::{build_llg_factory, extract_guidance_tokens, GuidanceTokens};
use crate::utils::guidance_grammar::{get_reasoning_token_strings, is_reasoning_grammar};
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
use futures::future::join_all;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Receiver;
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

/// Extract the success value from a Ractor RPC call, or a default. Wraps the
/// `ActorRef::call` + `CallResult` extraction so call sites stay concise.
async fn rpc_value<M, T>(
    ref_: &ractor::ActorRef<M>,
    builder: impl FnOnce(ractor::RpcReplyPort<T>) -> M,
    default: T,
) -> T
where
    M: ractor::Message,
    T: Send + 'static,
{
    match ref_.call(builder, None).await {
        Ok(CallResult::Success(v)) => v,
        _ => default,
    }
}

pub struct LLMEngine {
    /// The forward-pass actor (the sole path to the model transport). Every
    /// other component reaches the model through this `ActorRef<FwdMessage>`,
    /// never a direct transport reference. This is what makes the desync class
    /// of bug structurally impossible: the `IpcChannel` actor is the sole socket
    /// writer, and the `FwdD` actor is the sole driver of it.
    pub fwd: ractor::ActorRef<FwdMessage>,
    /// The stream delivery actor (the per-request channels, the token delivery).
    pub stream: ractor::ActorRef<StreamMessage>,
    /// The scheduler actor (the engine loop). Owns the `Scheduler` state + drives
    /// the `FwdD` + `StreamD` actors.
    pub sched: ractor::ActorRef<SchedMessage>,
    /// The HTTP ingestion actor (the request queue, the per-request stream channels).
    pub http: ractor::ActorRef<HttpMessage>,
    pub tokenizer: Tokenizer,
    pub econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
    prompt_replay_candidates: Vec<Vec<u32>>,
    has_vision: bool,
    model_name: String,
    pub model_type: ModelType,
    pub tool_config: ToolConfig,
    pub img_cfg: Option<ImageProcessConfig>,
    pub guidance_tokens: GuidanceTokens,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub async fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<Self>> {
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
        if config.bos_token_id.is_none() {
            if let Some(bos) = &config_tokenizer.bos_token {
                if let Some(token) = tokenizer.get_vocab(true).get(bos).copied() {
                    config.bos_token_id = Some(token as usize);
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
            config
                .bos_token_id
                .map_or(Vec::new(), |bos| vec![bos as u32]),
            &config_tokenizer,
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

        let fwd = if !use_runner {
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
                    &mut config,
                    dtype,
                    is_rope_i,
                    device.clone(),
                    reporter,
                    transfer,
                    llg_factory.clone(),
                    None,
                )?;
                // Async CUDA weight H2D keeps mmap live until this sync.
                #[cfg(feature = "cuda")]
                device.synchronize()?;
                drop(vb);
                runner
            };

            if !is_pd_server {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if econfig.disable_cuda_graph {
                    log_info!("CUDA graph capture disabled by --disable-cuda-graph");
                } else if crate::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    log_info!("{arch} does not supprt CUDA graph");
                } else if crate::utils::qwen4_has_ple(&config) {
                    log_info!(
                        "CUDA graph disabled: Qwen4 PLE n-gram embedding gathers on host memory"
                    );
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
            // The in-process model (the non-nccl build). Ractor actors require
            // `Send` state, but the in-process ModelRunner (the GPU tensors) is
            // not Send. The Ractor conversion supports the IPC runner path (the
            // nccl build); the in-process path is not supported.
            candle_core::bail!(
                "Ractor engine requires the IPC runner path (the nccl build). \
                 The in-process (non-nccl) ModelRunner is not Send and cannot be \
                 a Ractor actor's. Build with the nccl feature."
            );
        } else {
            log_info!("Loading model with runner(s)...");

            let is_multi_node = econfig.num_nodes > 1;
            let node_rank = econfig.node_rank;
            let num_nodes = econfig.num_nodes;
            let local_num_gpus = device_ids.len();
            let global_rank_offset = node_rank * local_num_gpus;

            #[cfg(feature = "nccl")]
            let nccl_id = if is_multi_node {
                let master_addr = econfig
                    .master_addr
                    .as_ref()
                    .expect("master_addr required for multi-node");
                crate::utils::multi_node::coordinate_nccl_id(
                    num_nodes,
                    node_rank,
                    master_addr,
                    econfig.master_port,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Multi-node NCCL ID exchange failed: {}", e))
                })?
            } else {
                Id::new().unwrap()
            };

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
            let progress_handle =
                spawn_progress_thread(local_num_gpus, config.num_hidden_layers, progress_sock_name);
            heartbeat_worker(Some(device_ids.len()), false, stop_flag.clone(), unique_id);
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;
            let engine_config = std::sync::OnceLock::<EngineConfig>::new();
            let runner_streams: Result<Vec<LocalStream>> = device_ids
                .par_iter()
                .enumerate()
                .map(|(local_rank, dev_id)| {
                    let global_rank = global_rank_offset + local_rank;
                    let model_type = model_type.clone();
                    let sock_name = format!("{}@xinfer-runner-{}", unique_id, local_rank);
                    let listener = ListenerOptions::new()
                        .name(
                            sock_name
                                .clone()
                                .to_ns_name::<GenericNamespaced>()
                                .expect("Failed to to_ns_name"),
                        )
                        .create_sync()
                        .expect("Failed to create listener");

                    crate::log_info!(
                        "listener starting accepting runner {} (global rank {})",
                        local_rank,
                        global_rank
                    );

                    // Accept one connection
                    let mut stream = listener.accept()?;
                    crate::log_info!(
                        "Accepted runner {} (global rank {})",
                        local_rank,
                        global_rank
                    );

                    // Wait for "ready"
                    let mut reader = BufReader::new(&mut stream);
                    let mut message = String::new();
                    reader.read_line(&mut message)?;
                    if message.trim() != "ready" {
                        return Err(candle_core::Error::Msg(format!(
                            "Runner {} did not send ready",
                            local_rank
                        )));
                    }

                    // Build init message with global rank for NCCL
                    let mut econfig_rank = econfig.clone();
                    econfig_rank.device_ids = Some(vec![*dev_id]);
                    let init_msg = MessageType::Init(RunnerInitRequest {
                        rank: global_rank,
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

                    send_and_expect_ack(&mut stream, &init_msg, "initialize", global_rank)?;

                    // Always validate/plan allocation after model loading for accurate memory readings
                    let ecfg = engine_config.get_or_init(|| {
                        let mut econfig = econfig.clone();
                        let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
                        // Propagate resolved dtype (e.g. TurboQuant → Auto on MLA) before plan/IPC.
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
                        global_rank,
                    )?;

                    crate::log_info!(
                        "Runner {} (global rank {}) started!",
                        local_rank,
                        global_rank
                    );
                    Ok(stream)
                })
                .collect();

            if let Ok(Some(handle)) = progress_handle.join() {
                let _ = handle.join();
            }
            if let Some(cfg) = engine_config.get() {
                econfig = cfg.clone();
                config.kvcache_dtype = econfig.kvcache_dtype;
            }

            let local_streams = runner_streams?;

            // Multi-node forward coordination setup
            if is_multi_node && node_rank == 0 {
                let mn_config = crate::utils::multi_node::MultiNodeConfig {
                    num_nodes,
                    node_rank,
                    master_addr: econfig.master_addr.clone().unwrap_or_default(),
                    master_port: econfig.master_port,
                    local_num_gpus,
                };
                let _worker_tcp_streams = crate::utils::multi_node::master_accept_worker_streams(
                    &mn_config,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Multi-node forward coord failed: {}", e))
                })?;
                // The multi-node case: the local streams drive the FwdD actor (the
                // remote TCP streams are coordinated separately by the multi-node
                // worker daemon, the NCCL all-reduce synchronizes the compute).
                start_fwd_actor(local_streams).await
            } else {
                start_fwd_actor(local_streams).await
            }
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

        // The KV-cache actor (the block pool, the prefix cache, the mamba slots, the
        // tier ladder). Owned before the scheduler so the scheduler can drive it
        // through the KvRef proxy (the no shared block_manager).
        let prefix_cache_cfg = crate::core::scheduler::build_prefix_cache_config(&econfig);
        let mamba_snapshot_default_stride_blocks = econfig
            .effective_prefill_chunk_size()
            .div_ceil(econfig.block_size)
            .max(1);
        let block_manager = BlockManager::new(
            fwd.clone(),
            econfig.num_blocks,
            (econfig.cpu_mem_fold.unwrap_or(0.5f32) * econfig.num_blocks as f32) as usize,
            econfig.block_size,
            prefix_cache_cfg,
            config
                .architectures
                .as_ref()
                .and_then(|arches| arches.first())
                .map(|arch| crate::utils::is_qwen3_hybrid_arch_name(arch))
                .unwrap_or(false),
            mamba_snapshot_default_stride_blocks,
        );
        let kv = start_kv_actor(block_manager).await;

        let mut scheduler = Scheduler::new(kv.clone(), &econfig, &config);

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
            "KV-cache pool {} tokens ({} blocks x block_size {}). Per-step scheduling budget {} tokens (max {} parallel reqs, max {} reqs). Additional CPU KV Cache blocks {}.",
            econfig.max_kv_cache_tokens,
            econfig.num_blocks,
            econfig.block_size,
            econfig.max_num_batched_tokens,
            econfig.max_num_parallel_reqs,
            econfig.max_num_seqs,
            (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.5f32)) as usize
        );

        // The official DeepSeek-V4 `chat` encoding starts generation after
        // `</think>`; thinking mode is opt-in for this model. Other reasoning
        // models retain the existing enabled-by-default behavior.
        let default_enable_thinking =
            !econfig.disable_reasoning && !matches!(model_type, ModelType::DeepSeekV4);
        let mut template = ChatTemplate::new(
            None,
            config_tokenizer.chat_template.clone(),
            config_tokenizer.bos_token.clone(),
            config_tokenizer.eos_token.clone(),
            None,
            true,
            default_enable_thinking,
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

        // The LoRA startup log (the #4, the operator visibility): the loaded
        // adapters + the residency budget. The no-op when no adapter is loaded
        // (the base model only, the zero overhead).
        if !econfig.lora.adapters.is_empty() {
            crate::log_info!(
                "LoRA: {} adapter(s) loaded ({:?}), budget {} tenants / {} bytes, policy {:?}, pinned {:?}",
                econfig.lora.adapters.len(),
                econfig.lora.adapters.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                econfig.lora.max_tenants.unwrap_or(DEFAULT_LORA_MAX_TENANTS),
                econfig.lora.max_bytes.unwrap_or(DEFAULT_LORA_MAX_BYTES),
                econfig.lora.eviction_policy,
                econfig.lora.pinned
            );
        }

        // The stream delivery actor (the per-request channels, the token delivery).
        let stream = start_stream_actor().await;
        // The scheduler actor (the engine loop): owns the Scheduler state + drives
        // the FwdD + StreamD actors. The mtp/dflash flags come from the econfig.
        let mtp_enabled = crate::speculative::uses_builtin_mtp(&econfig);
        let dflash_enabled = crate::speculative::uses_dflash(&econfig);
        let sched = start_sched_actor(
            scheduler,
            fwd.clone(),
            stream.clone(),
            mtp_enabled,
            dflash_enabled,
        ).await;
        // The HTTP ingestion actor (the request queue, the per-request stream
        // channels): forwards sequences to the SchedD actor + registers the stream
        // channels with the StreamD actor.
        let http = start_http_actor(sched.clone(), stream.clone()).await;
        // Resolve the circular dependency: SchedD needs HttpRef for
        // take_prompt_replay and remove_active_request. HttpD needs SchedRef
        // for cancel forwarding. Both are created, now wire them.
        let _ = sched.send_message(SchedMessage::SetHttp(http.clone()));

        let engine = Arc::new(Self {
            fwd,
            stream,
            sched,
            http,
            tokenizer,
            econfig,
            default_chat_template,
            template,
            prompt_replay_candidates,
            has_vision: config.is_multi_model.unwrap_or(false),
            model_type,
            tool_config,
            img_cfg,
            model_name,
            guidance_tokens,
        });

        Self::start_engine(engine.clone());
        Ok(engine)
    }

    async fn add_request_(
        &self,
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
        self.add_request_pretokenized_(params, prompt, token_ids, request_type, images, image_idx).await
    }

    /// Same body as `add_request_` but accepts a pre-computed `token_ids`
    /// so the caller can run tokenize outside the engine write lock. Hot-path
    /// serialization of concurrent server handlers gets cut by ~
    /// chat-template + tokenize wall × concurrency in chat-template-heavy
    /// workloads.
    async fn add_request_pretokenized_(
        &self,
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

        let mut max_tokens = if let Some(max_t) = params.max_tokens {
            max_t
        } else {
            let default_max = self.econfig.max_tokens.unwrap_or(16384);
            params.max_tokens = Some(default_max);
            default_max
        };
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

        let prompt_required_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockRequired(seq.clone(), r), 0).await;
        let requested_decode_blocks = max_tokens.div_ceil(self.econfig.block_size);
        let minimum_decode_budget_tokens = max_tokens.min(REQUEST_ADMISSION_DECODE_BUDGET_TOKENS);
        let minimum_decode_blocks = minimum_decode_budget_tokens.div_ceil(self.econfig.block_size);
        let mut target_required_blocks =
            prompt_required_blocks.saturating_add(requested_decode_blocks);
        let mut minimum_required_blocks =
            prompt_required_blocks.saturating_add(minimum_decode_blocks);
        let mut available_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockFree(r), 0).await;

        while target_required_blocks > available_blocks {
            let prev_target = target_required_blocks;
            let target_required_tokens = target_required_blocks * self.econfig.block_size;
            let evicted = rpc_value(&self.sched, |r| SchedMessage::EvictPrefixUntilFree(target_required_blocks, r), 0).await;
            if evicted > 0 {
                crate::log_warn!(
                    "Evicted {} prefix cache block(s) to reserve {} KV tokens for request admission (prompt + {} requested decode tokens).",
                    evicted,
                    target_required_tokens,
                    max_tokens
                );
            }
            if evicted == 0 && !self.try_release_cache(target_required_tokens).await {
                break;
            }
            let prompt_required_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockRequired(seq.clone(), r), 0).await;
            target_required_blocks = prompt_required_blocks.saturating_add(requested_decode_blocks);
            minimum_required_blocks = prompt_required_blocks.saturating_add(minimum_decode_blocks);
            available_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockFree(r), 0).await;

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
        let seq_id = rpc_value(&self.sched, |r| SchedMessage::Add(seq, r), 0).await;
        if let Some(end_marker) = detect_prefilled_reasoning_end_marker(prompt) {
            let _ = self.stream.send_message(StreamMessage::RecordPrefilledReasoningEnd(seq_id, end_marker));
        }
        if let Some(replay_ids) = raw_replay_token_ids {
            let _ = self.http.send_message(HttpMessage::SetPromptReplay(seq_id, replay_ids));
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
            let _ = self.stream.send_message(StreamMessage::SetDecoder(seq_id, boxed_decoder));
        }
        let _ = self.http.send_message(HttpMessage::AddActiveRequest(seq_id));
        Ok((seq_id, length))
    }

    pub async fn add_request(
        &self,
        params: &SamplingParams,
        prompt: &str,
        request_type: RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) =
            self.add_request_(params, prompt, &request_type, images, image_idx).await?;
        // The stream channel is registered through the HttpD actor (the request
        // ingestion domain), which forwards it to the StreamD actor.
        let rx = match self.http.call(|r| HttpMessage::RegisterStream(seq_id, r), None).await {
            Ok(CallResult::Success(rx)) => rx,
            _ => return Err(candle_core::Error::msg("register_stream failed")),
        };
        let _ = self.stream.send_message(StreamMessage::SetRequestType(seq_id, request_type.clone()));
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
    pub async fn add_request_pretokenized(
        &self,
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
        ).await?;
        let rx = match self.http.call(|r| HttpMessage::RegisterStream(seq_id, r), None).await {
            Ok(CallResult::Success(rx)) => rx,
            _ => return Err(candle_core::Error::msg("register_stream failed")),
        };
        let _ = self.stream.send_message(StreamMessage::SetRequestType(seq_id, request_type.clone()));
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
    /// in parallel under their own `engine.as_ref()` guards. Returns a list
    /// of `PreprocessedRequest`s that feed `generate_sync`
    /// or `generate_stream`, which acquire `engine.as_ref()`
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

    pub async fn get_num_cached_tokens(&self) -> usize {
        rpc_value(&self.sched, |r| SchedMessage::GetNumCachedTokens(r), 0).await
    }

    /// See [`Scheduler::get_num_cached_tokens_for_seq`].
    pub async fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        rpc_value(&self.sched, |r| SchedMessage::GetNumCachedTokensForSeq(seq_id, r), None).await
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

pub async fn get_available_kv_tokens(&self) -> usize {
        rpc_value(&self.sched, |r| SchedMessage::GetAvailableKvTokens(r), 0).await
    }

pub async fn notify_runner_finished(&self, id: usize) -> Result<()> {
        let _ = self.fwd.send_message(FwdMessage::Finished(id));
        Ok(())
    }

    /// The end-of-sequence scheduling report (cached tokens + QoS stats) for the
    /// server's per-sequence display. Delegates to the scheduler (same process).
    pub async fn get_seq_report(&self, seq_id: usize) -> Option<SeqReport> {
        rpc_value(&self.sched, |r| SchedMessage::GetSeqReport(seq_id, r), None).await
    }

    /// Release hybrid GDN/Mamba active slots queued by scheduler-side abort paths.
    pub async fn try_release_cache(&self, tokens_required: usize) -> bool {
        if tokens_required == 0 {
            return rpc_value(&self.sched, |r| SchedMessage::EvictPrefixCacheBlocks(1, r), 0).await > 0;
        }

        let required_blocks = tokens_required.div_ceil(self.econfig.block_size);
        rpc_value(&self.sched, |r| SchedMessage::EvictPrefixUntilFree(required_blocks, r), 0).await > 0
    }

    pub async fn check_canceled(&self, reason: Option<String>) {
        // Process cancellations + drain the pending runner releases in a loop
        // (the no mutual recursion, which would create an infinitely-sized future).
        loop {
            let cancelled = rpc_value(&self.http, |r| HttpMessage::TakeCancellations(r), Vec::new()).await;
            if !cancelled.is_empty() {
                for seq_id in cancelled {
                    let _ = self.sched.send_message(SchedMessage::Cancel(seq_id));
                    // Ensure model-side per-sequence state (e.g., Qwen3.5 Mamba cache slot) is released.
                    let _ = self.notify_runner_finished(seq_id).await;
                    let _ = self.http.send_message(HttpMessage::RemoveActiveRequest(seq_id));
                    let _ = self.stream.send_message(StreamMessage::ForgetSequence(seq_id));
                    let _ = rpc_value(&self.http, |r| HttpMessage::TakePromptReplay(seq_id, r), None).await;
                    if let Some(r) = &reason {
                        let request_type = rpc_value(&self.stream, |r| StreamMessage::GetRequestType(seq_id, r), None).await;
                        if let Some(request_type) = request_type {
                            if request_type == RequestType::Stream {
                                let _ = self.stream.send_message(StreamMessage::DeliverDone(seq_id, StreamItem::Error(r.clone())));
                            }
                        }
                    }
                }
                let _ = self.sched.send_message(SchedMessage::ClearFinished);
            }
            // Drain the pending runner releases (mark them cancelled, the loop
            // processes them on the next iteration).
            let pending = rpc_value(&self.sched, |r| SchedMessage::TakePendingRunnerReleases(r), Vec::new()).await;
            if pending.is_empty() {
                break;
            }
            for seq_id in &pending {
                let _ = self.http.send_message(HttpMessage::MarkCancelled(*seq_id));
            }
        }
    }

    pub async fn may_print_decoding_throughput(&self, active_indices: &Vec<usize>) {
        if active_indices.is_empty() || rpc_value(&self.http, |r| HttpMessage::IsIdle(r), false).await {
            return;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        if !rpc_value(&self.stream, |r| StreamMessage::ThroughputGate(cur_time, r), false).await {
            return;
        }

        let mut total_decoded_length = 0;
        let mut total_decoded_time_costs = 0;

        let mut seq_ids = Vec::new();
        for &idx in active_indices {
            if let Some(seq) = rpc_value(&self.sched, |r| SchedMessage::GetRunning(idx, r), None).await {
                if let Some(length) = rpc_value(&self.stream, |r| StreamMessage::GetDecodeLength(seq.id, r), None).await {
                    total_decoded_length += length;
                }
                if let Some(start_time) = rpc_value(&self.stream, |r| StreamMessage::GetDecodeStartTime(seq.id, r), None).await {
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
            let _ = self.sched.send_message(SchedMessage::PrintFreeBlocks);
        }
    }

    pub fn cancel(&self, seq_id: usize) {
        let _ = self.http.send_message(HttpMessage::MarkCancelled(seq_id));
    }

    pub async fn cancel_all_with_reason(&self, reason: Option<String>) -> bool {
        let had_active = !rpc_value(&self.http, |r| HttpMessage::IsIdle(r), false).await;
        let _ = self.http.send_message(HttpMessage::CancelAllActive);
        self.check_canceled(reason).await;
        had_active
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
        let default_enable_thinking = self.template.enable_thinking();
        if let Some(grammar) = &params.grammar {
            if is_reasoning_grammar(&grammar) {
                prompt_template.set_enable_thinking(true);
            } else {
                // Grammar without reasoning rules: thinking is handled at the mask
                // level (GuidanceState defers grammar until after </think>), so use
                // the default thinking setting — same as the no-grammar path.
                prompt_template
                    .set_enable_thinking(params.thinking.unwrap_or(default_enable_thinking));
            }
        } else {
            prompt_template.set_enable_thinking(params.thinking.unwrap_or(default_enable_thinking));
        };
        prompt_template.set_reasoning_effort(
            params
                .reasoning_effort
                .as_ref()
                .and_then(|effort| effort.chat_template_value()),
        );
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
        // Generation alignment and open/close parity enforcement
        if let Some(grammar) = &params.grammar {
            if self.guidance_tokens.add_bos_token {
                // BOS-based trimming: trim the last BOS token from the prompt tail.
                // Only trim if the prompt actually ends with the BOS string to avoid
                // splitting in the middle of a multi-turn conversation.
                if let Ok(bos_string) = self
                    .tokenizer
                    .decode(&self.guidance_tokens.bos_token_ids, false)
                {
                    if prompt.trim_end().ends_with(&bos_string) {
                        if let Some((prefix, _)) = prompt.rsplit_once(&bos_string) {
                            return (prefix.to_string(), image_idx);
                        }
                    }
                }
                for bos_token in self.guidance_tokens.bos_token_ids.iter() {
                    if let Ok(bos_string) = self.tokenizer.decode(&[*bos_token], false) {
                        if prompt.trim_end().ends_with(&bos_string) {
                            if let Some((prefix, _)) = prompt.rsplit_once(&bos_string) {
                                return (prefix.to_string(), image_idx);
                            }
                        }
                    }
                }
            } else {
                // Reasoning tag-based trimming: check for reasoning start/end tokens
                if let Some((start_str, end_str)) =
                    get_reasoning_token_strings(&self.guidance_tokens, &self.tokenizer)
                {
                    if is_reasoning_grammar(&grammar) {
                        // Control entire reasoning block via guidance
                        if prompt.trim().ends_with(&start_str) || prompt.trim().ends_with(&end_str)
                        {
                            if let Some((prompt, _trimmed)) = prompt.rsplit_once(&start_str) {
                                return (prompt.to_string(), image_idx);
                            }
                        }
                    } else if params.guidance_reasoning_end_ids.is_empty() {
                        // Only trim <think> when NOT using two-phase reasoning.
                        // With two-phase reasoning, the model needs the <think> prefix
                        // to generate reasoning freely before grammar constraints kick in.
                        if prompt.trim().ends_with(&start_str) {
                            if let Some((prompt, _trimmed)) = prompt.rsplit_once(&start_str) {
                                return (prompt.to_string(), image_idx);
                            }
                        }
                    }
                }
            }
        }
        (prompt, image_idx)
    }

    pub async fn is_idle(&self) -> bool {
        rpc_value(&self.http, |r| HttpMessage::IsIdle(r), false).await
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
    /// `preprocess(...)` under `engine.as_ref()` first, then call this under
    /// `engine.as_ref()` so the write lock is held only for request admission.
    pub async fn generate_sync(
        &self,
        preprocessed: Vec<PreprocessedRequest>,
        images: Option<ImageData>,
        logger: &Option<Arc<ChatCompletionLogger>>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        let mut receivers = Vec::with_capacity(preprocessed.len());
        for pp in preprocessed {
            if let Some(ref l) = logger {
                l.log_prompt(&pp.prompt);
            }
            let (seq_id, prompt_length, rx) = self.add_request_pretokenized(
                &pp.params,
                &pp.prompt,
                pp.token_ids,
                RequestType::Completion,
                &images,
                pp.image_idx,
            ).await?;
            receivers.push((seq_id, prompt_length, rx));
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

    pub async fn free_resources(&self) {
        crate::log_error!("***Release all resources for future usage!");
        let _ = self.sched.send_message(SchedMessage::ClearFinished);
        let released_ids = rpc_value(&self.sched, |r| SchedMessage::ReleaseWaitings(r), Vec::new()).await;
        for seq_id in released_ids {
            let _ = self.notify_runner_finished(seq_id).await;
        }
        let _ = self.stream.send_message(StreamMessage::ResetAll);
        let _ = self.http.send_message(HttpMessage::ClearPromptReplays);
    }

    /// Returns the reasoning end marker when the prompt for this sequence
    /// already ended with an opening think marker.
    pub async fn get_prefilled_reasoning_end_marker(&self, seq_id: usize) -> Option<String> {
        rpc_value(&self.stream, |r| StreamMessage::GetPrefilledReasoningEnd(seq_id, r), None).await
    }

    /// Registers a preprocessed streaming request. Callers should run
    /// `preprocess(...)` under `engine.as_ref()` first, then call this under
    /// `engine.as_ref()` so the write lock is held only for request admission.
    pub async fn generate_stream(
        &self,
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
        ).await {
            Ok((seq_id, prompt_length, rx)) => {
                let prefilled_reasoning_end = self.get_prefilled_reasoning_end_marker(seq_id).await;
                Ok((seq_id, prompt_length, prefilled_reasoning_end, rx))
            }
            Err(e) => {
                candle_core::bail!("{:?}", e)
            }
        }
    }

    pub async fn get_usage_stats(&self, session_id: Option<String>) -> Result<UsageResponse> {
        match session_id {
            Some(sid) => {
                let available_kvcache_tokens = rpc_value(&self.sched, |r| SchedMessage::GetAvailableKvTokens(r), 0).await;
                let max_model_len = self
                    .econfig
                    .max_model_len
                    .ok_or_else(|| candle_core::Error::msg("max_model_len not set!"))?;
                let (seq_id, status) =
                    rpc_value(&self.sched, |r| SchedMessage::FindSeqBySessionId(sid.clone(), r), None).await.ok_or_else(|| {
                        candle_core::Error::msg(format!("Seq with session_id {sid} not found"))
                    })?;
                let token_used = rpc_value(&self.sched, |r| SchedMessage::GetSeqTokenUsage(seq_id, r), Err("RPC failed".to_string())).await
                    .map_err(|e| candle_core::Error::Msg(e))?;

                let total_kv_cache_tokens = rpc_value(&self.sched, |r| SchedMessage::GetTotalKvTokens(r), 0).await;
                let (swap_used, total_swap_memory) = rpc_value(&self.sched, |r| SchedMessage::GetCpuSwapUsage(r), (0.0, 0.0)).await;

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

    pub async fn embed(
        &self,
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
            let required_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockRequired(seq.clone(), r), 0).await;
            let available_blocks = rpc_value(&self.sched, |r| SchedMessage::BlockFree(r), 0).await;
            if required_blocks > available_blocks {
                let available_tokens = available_blocks * self.econfig.block_size;
                let required_tokens = required_blocks * self.econfig.block_size;
                candle_core::bail!(
                    "Remaining {} kvcache tokens, but embedding requires {} new tokens, please retry later",
                    available_tokens,
                    required_tokens
                );
            }
            seq = rpc_value(&self.sched, |r| SchedMessage::BlockAllocate(seq, r), None).await.ok_or_else(|| {
                candle_core::Error::msg("Failed to allocate blocks for embedding sequence")
            })?;

            let mut chunked_mean: Option<Vec<f32>> = None;
            let mut last_vec: Option<Vec<f32>> = None;
            let mut processed_tokens = 0usize;
            let chunk_size = self.econfig.effective_prefill_chunk_size();

            while seq.num_cached_tokens < seq.len() {
                let remaining = seq.len() - seq.num_cached_tokens;
                let chunk_tokens = std::cmp::min(chunk_size, remaining);
                let embedding_result = rpc_value(&self.fwd, |r| FwdMessage::Embed(vec![seq.clone()], strategy.clone(), r), Err("embed RPC failed".to_string())).await;
                let embedding_vecs = match embedding_result {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.sched.send_message(SchedMessage::BlockDeallocate(seq.clone()));
                        return Err(candle_core::Error::Msg(e));
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

            let _ = self.sched.send_message(SchedMessage::BlockDeallocate(seq.clone()));
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

    pub fn start_engine(engine: Arc<Self>) {
        // The engine loop runs as a tokio task (the Ractor actors are async). It
        // drives the SchedD actor's Step RPC (the schedule -> forward -> postprocess
        // -> delivery ->).
        GLOBAL_RT.spawn(async move {
            let (is_pd_server, sched) = {
                (engine.is_pd_mode() && engine.is_pd_server(), engine.sched.clone())
            };
            loop {
                let idle = engine.is_idle().await && !is_pd_server;
                if idle {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    continue;
                }
                // The inference step is driven by the SchedD actor (the schedule -> forward -> postprocess -> stream delivery).
                let task_processed = rpc_value(&sched, |r| SchedMessage::Step(r), 0).await;
                if task_processed == 0 {
                    engine.check_canceled(None).await;
                    tokio::time::sleep(std::time::Duration::from_millis(if is_pd_server {
                        PD_PREFILL_STATUS_CHECK_COOLING_PERIOD as u64
                    } else {
                        1
                    })).await;
                }
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

    /// Get a clone of the default chat template for grammar generation
    pub fn get_default_chat_template(&self) -> String {
        self.default_chat_template.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::LLMEngine;
    use crate::utils::guidance::GuidanceTokens;

    #[test]
    fn trim_prompt_replay_prefix_accepts_single_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42, 99],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            add_bos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[99], &guidance_tokens),
            Some(vec![99])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_accepts_multi_token_suffix_when_first_token_is_reasoning() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            add_bos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[42, 7], &guidance_tokens),
            Some(vec![42, 7])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_trims_leading_non_reasoning_tokens() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            add_bos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 42, 8], &guidance_tokens),
            Some(vec![42, 8])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_suffix_without_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            add_bos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 8], &guidance_tokens),
            None
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_empty_suffix() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            add_bos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[], &guidance_tokens),
            None
        );
    }
}
