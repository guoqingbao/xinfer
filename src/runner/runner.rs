use interprocess::local_socket::traits::Stream;
use interprocess::local_socket::Stream as LocalStream;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::TryClone;
use parking_lot::RwLock;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use xinfer::core::runner::{ModelRunner, Seqs};
use xinfer::models::layers::distributed::Comm;
use xinfer::models::layers::VarBuilderX;
use xinfer::runner::{receive_local, send_local, MessageType};
use xinfer::transfer::PdRole;
use xinfer::transfer::Transfer;
use xinfer::utils::gguf_helper::load_gguf_info_from_files;
use xinfer::utils::guidance::build_llg_factory;
use xinfer::utils::heartbeat::heartbeat_worker;
use xinfer::utils::new_device;
use xinfer::utils::progress::{ProgressLike, ProgressReporter, RemoteProgressReporter};

pub fn run_runner() -> anyhow::Result<()> {
    xinfer::log_info!("runner started");

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args: Vec<String> = if let Ok(encoded) = std::env::var("XINFER_RUNNER_ARGS") {
        encoded.split('\x1f').map(String::from).collect()
    } else {
        std::env::args().collect()
    };
    let sock = args
        .iter()
        .position(|s| s == "--sock")
        .and_then(|i| args.get(i + 1))
        .expect("Socket name missing");
    let uuid_str: String = args
        .iter()
        .position(|s| s == "--uuid")
        .and_then(|i| args.get(i + 1))
        .map_or("", |v| v)
        .to_string();
    let sock_name = sock.clone().to_ns_name::<GenericNamespaced>()?;
    let mut stream = LocalStream::connect(sock_name.clone());
    // shared flag for model loaded
    let model_loaded = Arc::new(AtomicBool::new(false));
    let model_loaded_ctrlc = model_loaded.clone();

    loop {
        if stream.is_ok() {
            break;
        }
        xinfer::log_info!("Runner retry connecting to socket: {}", sock);
        stream = LocalStream::connect(sock_name.clone());
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let mut stream = stream.expect("Failed to connect to socket");
    stream.write_all(b"ready\n")?;
    stream.flush()?;

    ctrlc::set_handler(move || {
        if model_loaded_ctrlc.load(Ordering::SeqCst) {
            xinfer::log_info!("Runner break session!");
        } else {
            xinfer::log_warn!("Runner break model loading (Ctrl+C detected)!");
            std::process::exit(0);
        }
    })
    .expect("Error setting Ctrl+C handler");

    xinfer::log_info!("Runner connected to socket: {}", sock);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _ = heartbeat_worker(None, true, stop_flag.clone(), &uuid_str);

    let msg = receive_local(&mut stream, true)?;
    let runner = match msg {
        MessageType::Init(init_req) => {
            xinfer::log_info!("Received init request: {:?}", init_req);
            // Use init_req.rank to pick device
            let device = new_device(init_req.dev_id)?;

            #[cfg(feature = "nccl")]
            let comm = Rc::new(
                Comm::from_rank(
                    device.as_cuda_device().unwrap().cuda_device(),
                    init_req.rank,
                    init_req.num_shards,
                    init_req.nccl_id.0,
                )
                .unwrap(),
            );

            #[cfg(not(feature = "nccl"))]
            let comm = Rc::new(Comm::default());

            xinfer::log_info!("Loading model at rank {}", init_req.rank);

            let progress_sock_name = format!("{}@xinfer-progress", uuid_str);

            let progress_reporter = match RemoteProgressReporter::new(
                init_req.rank,
                init_req.num_shards,
                progress_sock_name,
                true,
            ) {
                Ok(reporter) => {
                    let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                        Arc::new(RwLock::new(Box::new(reporter)));
                    reporter
                }
                _ => {
                    xinfer::log_error!("Unable to create remote progress reporter!");
                    let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                        Arc::new(RwLock::new(Box::new(ProgressReporter::new(init_req.rank))));
                    reporter
                }
            };

            let (transfer, is_pd_server) = if let Some(t_cfg) = &init_req.econfig.pd_config {
                (
                    Some(Arc::new(Transfer::new(
                        t_cfg.clone(),
                        init_req.rank,
                        model_loaded.clone(),
                        stop_flag.clone(),
                    )?)),
                    matches!(t_cfg.role, PdRole::Server),
                )
            } else {
                (None, false)
            };

            let stream_kv = Some(stream.try_clone()?);
            let mut econfig = init_req.econfig.clone();
            let tokenizer_path = init_req.model_pathes.get_tokenizer_filename();
            let llg_factory = if init_req.is_gguf {
                match load_gguf_info_from_files(&init_req.model_pathes.get_weight_filenames()) {
                    Ok(info) => match build_llg_factory(info.tokenizer, init_req.config.vocab_size)
                    {
                        Ok(f) => Some(f),
                        Err(e) => {
                            xinfer::log_warn!("Failed to build llguidance factory: {}", e);
                            None
                        }
                    },
                    Err(e) => {
                        xinfer::log_warn!(
                            "Failed to load GGUF tokenizer metadata; disabling optional llguidance: {}",
                            e
                        );
                        None
                    }
                }
            } else if tokenizer_path.exists() {
                match Tokenizer::from_file(&tokenizer_path) {
                    Ok(tokenizer) => match build_llg_factory(tokenizer, init_req.config.vocab_size)
                    {
                        Ok(f) => Some(f),
                        Err(e) => {
                            xinfer::log_warn!("Failed to build llguidance factory: {}", e);
                            None
                        }
                    },
                    Err(e) => {
                        xinfer::log_warn!(
                            "Failed to load tokenizer from {:?}; disabling optional llguidance: {}",
                            tokenizer_path,
                            e
                        );
                        None
                    }
                }
            } else {
                xinfer::log_warn!(
                    "Tokenizer file {:?} not found; disabling optional llguidance",
                    tokenizer_path
                );
                None
            };
            #[allow(unused_mut)]
            let mut runner = {
                let _guard = candle_core::InferenceMode::enter();
                let vb = VarBuilderX::new(
                    &init_req.model_pathes,
                    init_req.is_gguf,
                    init_req.dtype.into(),
                    &device,
                )?;
                let runner = ModelRunner::new(
                    init_req.model_type,
                    &vb,
                    comm,
                    &mut econfig,
                    &init_req.config,
                    init_req.dtype.into(),
                    init_req.is_rope_i,
                    device,
                    progress_reporter,
                    transfer,
                    llg_factory,
                    stream_kv,
                )?;
                drop(vb);
                runner
            };

            xinfer::log_info!(
                "Runner at rank {} created (PD config: {:?})!",
                init_req.rank,
                init_req.econfig.pd_config
            );

            if !is_pd_server {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                let arch = init_req.config.architectures.as_ref().unwrap()[0].clone();
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if init_req.econfig.disable_cuda_graph {
                    xinfer::log_info!("CUDA graph capture disabled by --disable-cuda-graph");
                } else if xinfer::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    xinfer::log_info!("{arch} does not supprt CUDA graph");
                } else {
                    match runner.warmup_capture() {
                        Ok(_) => {
                            use colored::Colorize;
                            eprintln!("{}", String::from("Cuda graph captured").yellow());
                        }
                        Err(e) => {
                            use colored::Colorize;
                            let s = format!("Graph capture failed: {:?}", e);
                            eprintln!("{}", s.red());
                        }
                    }
                }
            }

            send_local(
                &mut vec![stream.try_clone()?],
                &MessageType::InitAck(true),
                false,
            )?;
            runner
        }
        _ => {
            xinfer::log_error!("Unexpected message type: {:?}", msg);
            panic!("Unexpected message type");
        }
    };

    // mark model as loaded
    model_loaded.store(true, Ordering::SeqCst);
    loop {
        match receive_local(&mut stream, false) {
            Ok(MessageType::Shutdown) => {
                xinfer::log_info!("Runner exit");
                break;
            }
            Ok(MessageType::RunPrefill((sequences, is_prefill))) => {
                let outputs = runner.run(
                    Seqs::SeqRefs(&sequences.iter().collect::<Vec<_>>()),
                    is_prefill,
                );
                if outputs.is_err() {
                    xinfer::log_error!("Runner prefill error: {:?}", outputs);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponse(outputs.unwrap_or(vec![])),
                    false,
                )?;
            }
            Ok(MessageType::RunDecode((sequences, is_prefill))) => {
                let outputs = runner.run(Seqs::DecodeVec(&sequences), is_prefill);
                if outputs.is_err() {
                    xinfer::log_error!("Runner decode error: {:?}", outputs);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponse(outputs.unwrap_or(vec![])),
                    false,
                )?;
            }
            Ok(MessageType::RunEmbed((sequences, strategy))) => {
                use xinfer::core::sequence::Sequence;
                let refs: Vec<&Sequence> = sequences.iter().collect();
                let slice: &[&Sequence] = &refs;
                let outputs = runner.embed(&slice, &strategy);
                if outputs.is_err() {
                    xinfer::log_error!("Runner embedding error: {:?}", outputs);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponseEmbed(outputs.unwrap_or(vec![vec![]])),
                    false,
                )?;
            }
            Ok(MessageType::LoadingProgress(_)) => {
                xinfer::log_info!("Received loading progress message");
            }
            Ok(MessageType::KVCacheSwap((mappings, swap_in))) => {
                xinfer::log_info!(
                    "Received KVCacheSwap message: {} kv cache blocks need to {}!",
                    mappings.len(),
                    if swap_in { "swap in" } else { "swap out" },
                );
                let ret = runner.swap_kvcache(mappings, swap_in);
                if ret.is_err() {
                    xinfer::log_error!("KvCache Swap failed: {:?}", ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::KVCacheSwapResponse(ret.is_ok()),
                    false,
                )?;
            }
            Ok(MessageType::FinishDecode(id)) => {
                runner.finished(id);
            }
            Ok(MessageType::CaptureMambaPrefixState((seq_id, hash, preserve))) => {
                let ret = runner.capture_mamba_prefix_state(seq_id, hash, preserve);
                if ret.is_err() {
                    xinfer::log_error!(
                        "CaptureMambaPrefixState failed for seq {} hash {} preserve={} : {:?}",
                        seq_id,
                        hash,
                        preserve,
                        ret
                    );
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::CaptureMambaPrefixStateResponse(ret.unwrap_or(false)),
                    false,
                )?;
            }
            Ok(MessageType::HasMambaPrefixState(hash)) => {
                let ret = runner.has_mamba_prefix_state(hash);
                if ret.is_err() {
                    xinfer::log_error!("HasMambaPrefixState failed for hash {}: {:?}", hash, ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::HasMambaPrefixStateResponse(ret.unwrap_or(false)),
                    false,
                )?;
            }
            Ok(MessageType::RemoveMambaPrefixState(hash)) => {
                let ret = runner.remove_mamba_prefix_state(hash);
                if ret.is_err() {
                    xinfer::log_error!("RemoveMambaPrefixState failed for hash {}: {:?}", hash, ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RemoveMambaPrefixStateResponse(ret.unwrap_or(false)),
                    false,
                )?;
            }
            Ok(MessageType::TransferPrefill(sequence)) => {
                let ret = runner.transfer_prefill(&sequence);
                // if ret.is_err() {
                //     xinfer::log_error!("Prefill transfer failed: {:?}", ret);
                // }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::TransferPrefillResponse(ret.is_ok()),
                    false,
                )?;
            }
            Ok(MessageType::ReceivePrefill(id)) => {
                let ret = runner.try_receive_prefill(id);
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::ReceivePrefillResponse(ret.unwrap_or((false, None))),
                    false,
                )?;
            }
            Ok(MessageType::CheckPrefillStatus(id)) => {
                let status = runner.check_prefill_status(id);
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::CheckPrefillStatusResponse(
                        status.is_ok() && status.unwrap_or(false),
                    ),
                    false,
                )?;
            }
            Ok(MessageType::KvCacheSend((sequence, token))) => {
                let ret = runner.send_kvcache(&sequence, token);
                if ret.is_err() {
                    xinfer::log_error!("KvCacheSend failed: {:?}", ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::KvCacheSendResponse(ret.is_ok()),
                    false,
                )?;
            }
            Ok(MessageType::KvCacheReceive(sequence)) => {
                let ret = runner.receive_kvcache(&sequence);
                if ret.is_err() {
                    xinfer::log_error!("KvCacheReceive failed: {:?}", ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::KvCacheReceiveResponse(ret.unwrap_or((false, 0, 0, 0))),
                    false,
                )?;
            }
            Ok(MessageType::KvCacheRelease(id)) => {
                let status = runner.release_remote_kvcache(id);
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::KvCacheReleaseResponse(status.is_ok() && status.unwrap_or(false)),
                    false,
                )?;
            }
            Ok(MessageType::CheckKvCacheRelease(id)) => {
                let status = runner.check_kvcache_release(id);
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::CheckKvCacheReleaseResponse(
                        status.is_ok() && status.unwrap_or(false),
                    ),
                    false,
                )?;
            }
            Ok(MessageType::ClearBlocks(block_ids)) => {
                let ret = runner.clear_blocks(block_ids);
                if ret.is_err() {
                    xinfer::log_error!("ClearBlocks failed: {:?}", ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::ClearBlocksResponse(ret.is_ok()),
                    false,
                )?;
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    xinfer::log_error!("Runner exit with error: {:?}", e);
                }
                break;
            }
            _ => {
                xinfer::log_error!("Unexpected message type");
            }
        }
    }
    stop_flag.store(true, Ordering::Relaxed);
    xinfer::log_info!("Runner finished");
    std::process::exit(0);
}
