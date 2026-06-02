use crate::core::sequence::{DecodeSequence, Sequence};
use crate::models::layers::distributed::Id;
use crate::server::EmbeddingStrategy;
use crate::utils::config::{Config, EngineConfig, ModelType};
use crate::utils::downloader::ModelPaths;
#[cfg(feature = "nccl")]
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use candle_core::DType;
use interprocess::local_socket::Stream as LocalStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunnerInitRequest {
    pub rank: usize,
    pub dev_id: usize,
    pub num_shards: usize,
    pub model_type: ModelType,
    pub config: Config,
    pub econfig: EngineConfig,
    pub model_pathes: ModelPaths,
    pub is_gguf: bool,
    pub dtype: SerializableDType,
    pub is_rope_i: bool,
    #[cfg(feature = "nccl")]
    pub nccl_id: NcclId,
}

#[derive(Debug, Clone)]
pub struct NcclId(pub Id);

#[cfg(feature = "nccl")]
impl Serialize for NcclId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Detect if JSON serializer
        if serializer.is_human_readable() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            let encoded = STANDARD_NO_PAD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            serializer.serialize_bytes(bytes)
        }
    }
}

#[cfg(feature = "nccl")]
impl<'de> Deserialize<'de> for NcclId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s: &str = Deserialize::deserialize(deserializer)?;
            let bytes = STANDARD_NO_PAD
                .decode(s)
                .map_err(serde::de::Error::custom)?;
            if bytes.len() != 128 {
                return Err(serde::de::Error::custom(format!(
                    "Expected 128 bytes but got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0i8; 128];
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
            }
            #[cfg(not(target_arch = "aarch64"))]
            return Ok(NcclId(Id::uninit(arr)));
            #[cfg(target_arch = "aarch64")]
            {
                let arr_u8 = arr.map(|b| b as u8);
                return Ok(NcclId(Id::uninit(arr_u8)));
            }
        } else {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = NcclId;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "128-byte NCCL ID")
                }

                fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    if v.len() != 128 {
                        return Err(E::custom(format!("Expected 128 bytes but got {}", v.len())));
                    }
                    let mut arr = [0i8; 128];
                    unsafe {
                        std::ptr::copy_nonoverlapping(v.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    return Ok(NcclId(Id::uninit(arr)));
                    #[cfg(target_arch = "aarch64")]
                    {
                        let arr_u8 = arr.map(|b| b as u8);
                        return Ok(NcclId(Id::uninit(arr_u8)));
                    }
                }
            }

            deserializer.deserialize_bytes(Visitor)
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum SerializableDType {
    U8 = 0,
    U32 = 1,
    I64 = 2,
    BF16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
}

impl From<DType> for SerializableDType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::U8 => Self::U8,
            DType::U32 => Self::U32,
            DType::I64 => Self::I64,
            DType::BF16 => Self::BF16,
            DType::F16 => Self::F16,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
        }
    }
}

impl From<SerializableDType> for DType {
    fn from(sdt: SerializableDType) -> Self {
        match sdt {
            SerializableDType::U8 => DType::U8,
            SerializableDType::U32 => DType::U32,
            SerializableDType::I64 => DType::I64,
            SerializableDType::BF16 => DType::BF16,
            SerializableDType::F16 => DType::F16,
            SerializableDType::F32 => DType::F32,
            SerializableDType::F64 => DType::F64,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct InitAck {
    pub ok: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    /// Sent by main process to initialize the runner.
    Init(RunnerInitRequest),

    /// Sent by runner in response to `Init` with initialization status.
    InitAck(bool),

    LoadingProgress((usize, usize)),

    /// Sent by main process to request prefill on sequences.
    RunPrefill((Vec<Sequence>, bool)),

    /// Sent by main process to request inference on sequences.
    RunDecode((Vec<DecodeSequence>, bool)),

    /// Sent by runner in response to `Run` with generated token IDs.
    RunResponse(Vec<u32>),

    /// Sent by main process to request embedding on sequences.
    RunEmbed((Vec<Sequence>, EmbeddingStrategy)),

    /// Sent by runner in response to `Run` with generated embedding states
    RunResponseEmbed(Vec<Vec<f32>>),

    /// Sent by main process to notify the finished decoding sequences.
    FinishDecode(usize),

    // Hybrid mamba-prefix state management.
    CaptureMambaPrefixState((usize, u64, bool)),
    CaptureMambaPrefixStateResponse(bool),
    HasMambaPrefixState(u64),
    HasMambaPrefixStateResponse(bool),
    RemoveMambaPrefixState(u64),
    RemoveMambaPrefixStateResponse(bool),

    /// Optional: runner can send back an error message.
    Error(String),

    Heartbeat,

    // Prefill transfer to PD server
    TransferPrefill(Sequence),
    TransferPrefillResponse(bool),

    // Prefill transfer receive
    ReceivePrefill(usize),
    ReceivePrefillResponse((bool, Option<Sequence>)),

    // Client: Check PD prefill status
    CheckPrefillStatus(usize),
    CheckPrefillStatusResponse(bool),

    KVCacheSwap((HashMap<usize, usize>, bool)),

    KVCacheSwapResponse(bool),

    // send kvcache to client (seq_id, first_token)
    KvCacheSend((Sequence, u32)),
    KvCacheSendResponse(bool),

    // receive kvcache from PD server
    KvCacheReceive(Sequence),
    KvCacheReceiveResponse((bool, u32, usize, usize)),

    // notify PD server to release kvcache
    KvCacheRelease(usize),
    KvCacheReleaseResponse(bool),

    // Server: Check if a prefilled seq need to release kvcache
    CheckKvCacheRelease(usize),
    CheckKvCacheReleaseResponse(bool),

    ClearBlocks(Vec<u32>),
    ClearBlocksResponse(bool),

    UsableMemoryLeft(EngineConfig),
    /// shutdown subprocesses
    Shutdown,
}

//inter-node communication
pub fn send_local(
    streams: &mut Vec<LocalStream>,
    message: &MessageType,
    use_json: bool,
) -> std::io::Result<()> {
    let serialized = if use_json {
        serde_json::to_vec(message).expect("JSON serialization failed")
    } else {
        bincode::serialize(message).expect("Bincode serialization failed")
    };

    for stream in streams.iter_mut() {
        stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
        stream.write_all(&serialized)?;
        stream.flush()?; // Ensure data is sent immediately
                         // Wait for acknowledgment
        let mut ack_buf = [0u8; 1];
        if let Err(e) = stream.read_exact(&mut ack_buf) {
            eprintln!(
                "Timeout waiting for acknowledgment from subprocess: {:?}",
                e
            );
        } else if ack_buf[0] != 1 {
            eprintln!("Unexpected acknowledgment value from subprocess");
        }
    }
    Ok(())
}

pub fn receive_local(stream: &mut LocalStream, use_json: bool) -> std::io::Result<MessageType> {
    let mut length_buf = [0u8; 4];
    stream.read_exact(&mut length_buf)?;
    let length = u32::from_le_bytes(length_buf) as usize;

    let mut serialized = vec![0u8; length];
    stream.read_exact(&mut serialized)?;

    let message: MessageType = if use_json {
        serde_json::from_slice(&serialized).expect("JSON deserialization failed")
    } else {
        bincode::deserialize(&serialized).expect("Bincode deserialization failed")
    };

    // Send acknowledgment
    stream.write_all(&[1])?;
    stream.flush()?;
    Ok(message)
}

pub fn send_and_expect_ack(
    stream: &mut LocalStream,
    msg: &MessageType,
    stage: &str,
    rank: usize,
) -> candle_core::Result<()> {
    use interprocess::TryClone;
    send_local(&mut vec![stream.try_clone()?], msg, true)?;

    crate::log_info!("Waiting runner {} {} response...", rank, stage);

    match receive_local(stream, false)? {
        MessageType::InitAck(true) => Ok(()),
        _ => candle_core::bail!("Runner {} failed during {}", rank, stage),
    }
}

///
/// Defines a function that broadcasts an operation to all runners and expects a `Result<T>`.
/// It handles both `Thread` (direct call) and `Process` (IPC message) runners.
///
/// In Process mode, it expects the response variant to contain the value `T`.
/// It collects all values and verifies they are identical before returning one.
///
#[macro_export]
macro_rules! def_broadcast_message_to_runners {
    (
        // The visibility (e.g., `pub`)
        $vis:vis,
        // The name of the function to create (e.g., `try_receive_kv_cache`)
        $fn_name:ident,
        // The name of the method on the thread-mode runner (e.g., `receive_kv_cache`)
        $thread_fn_name:ident,
        // The arguments for the function (e.g., `(seq: Sequence)`)
        ($($arg_name:ident: $arg_type:ty),*),
        // The MessageType variant to send (e.g., `MessageType::KvCacheReceive`)
        $msg_variant:path,
        // The expression to build the message payload (e.g., `(seq.clone())`)
        ($($msg_arg:expr),*),
        // The MessageType response variant to match (e.g., `MessageType::KvCacheReceiveResponse`)
        $resp_variant:path,
        // The inner return type (e.g., `u32`)
        $return_ty:ty
    ) => {
        $vis fn $fn_name(&self, $($arg_name: $arg_type),*) -> Result<$return_ty>
        where
            $return_ty: std::fmt::Debug + Send,
        {
            match &mut *self.runners.write() {
                RunnerType::Thread(model_runner) => {
                    // Thread Mode: Call the method directly.
                    model_runner.$thread_fn_name($($arg_name),*)
                }
                RunnerType::Process(ref mut runner_streams) => {
                    // Process Mode: Broadcast to all subprocess runners.
                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s| s.try_clone().expect("Failed to clone runner stream"))
                        .collect();

                    // Use Rayon for parallel broadcast
                    let all_results: Result<Vec<$return_ty>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            // Send the message
                            send_local(
                                &mut vec![stream.try_clone()?],
                                &$msg_variant($($msg_arg),*),
                                false,
                            )?;

                            // Wait for the response
                            let response = receive_local(&mut stream, false)?;
                            match response {
                                // Match on the expected response containing the value
                                $resp_variant(value) => {
                                    Ok(value)
                                }
                                other => {
                                    candle_core::bail!("Unexpected response for {}: {:?}", stringify!($fn_name), other)
                                }
                            }
                        })
                        .collect(); // Collects into a Result<Vec<T>>

                    // Check that all ranks returned the same value
                    match all_results {
                        Ok(mut values) => {
                            if values.is_empty() {
                                candle_core::bail!("No values received from runners for {}", stringify!($fn_name));
                            }
                            // Pop first element to return, then check rest for consistency
                            let first_val = values.pop().unwrap();
                            Ok(first_val)
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        }
    };
}

pub fn run_runner_process(args: Vec<String>) -> anyhow::Result<()> {
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

    use crate::core::runner::{ModelRunner, Seqs};
    use crate::models::layers::distributed::Comm;
    use crate::models::layers::VarBuilderX;
    use crate::transfer::PdRole;
    use crate::transfer::Transfer;
    use crate::utils::gguf_helper::load_gguf_info_from_files;
    use crate::utils::guidance::build_llg_factory;
    use crate::utils::heartbeat::heartbeat_worker;
    use crate::utils::new_device;
    use crate::utils::progress::{ProgressLike, ProgressReporter, RemoteProgressReporter};

    crate::log_info!("runner started");
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

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
    let model_loaded = Arc::new(AtomicBool::new(false));
    let model_loaded_ctrlc = model_loaded.clone();

    loop {
        if stream.is_ok() {
            break;
        }
        crate::log_info!("Runner retry connecting to socket: {}", sock);
        stream = LocalStream::connect(sock_name.clone());
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let mut stream = stream.expect("Failed to connect to socket");
    stream.write_all(b"ready\n")?;
    stream.flush()?;

    ctrlc::set_handler(move || {
        if model_loaded_ctrlc.load(Ordering::SeqCst) {
            crate::log_info!("Runner break session!");
        } else {
            crate::log_warn!("Runner break model loading (Ctrl+C detected)!");
            std::process::exit(0);
        }
    })
    .expect("Error setting Ctrl+C handler");

    crate::log_info!("Runner connected to socket: {}", sock);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _ = heartbeat_worker(None, true, stop_flag.clone(), &uuid_str);

    let msg = receive_local(&mut stream, true)?;
    let runner = match msg {
        MessageType::Init(init_req) => {
            crate::log_info!("Received init request: {:?}", init_req);
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

            crate::log_info!("Loading model at rank {}", init_req.rank);

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
                    crate::log_error!("Unable to create remote progress reporter!");
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
                    Ok(info) => {
                        match build_llg_factory(info.tokenizer, init_req.config.vocab_size) {
                            Ok(f) => Some(f),
                            Err(e) => {
                                crate::log_warn!("Failed to build llguidance factory: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        crate::log_warn!(
                            "Failed to load GGUF tokenizer metadata; disabling optional llguidance: {}",
                            e
                        );
                        None
                    }
                }
            } else if tokenizer_path.exists() {
                match Tokenizer::from_file(&tokenizer_path) {
                    Ok(tokenizer) => {
                        match build_llg_factory(tokenizer, init_req.config.vocab_size) {
                            Ok(f) => Some(f),
                            Err(e) => {
                                crate::log_warn!("Failed to build llguidance factory: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        crate::log_warn!(
                            "Failed to load tokenizer from {:?}; disabling optional llguidance: {}",
                            tokenizer_path,
                            e
                        );
                        None
                    }
                }
            } else {
                crate::log_warn!(
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

            crate::log_info!(
                "Runner at rank {} created (PD config: {:?})!",
                init_req.rank,
                init_req.econfig.pd_config
            );

            if !is_pd_server {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                let arch = init_req.config.architectures.as_ref().unwrap()[0].clone();
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if init_req.econfig.disable_cuda_graph {
                    crate::log_info!("CUDA graph capture disabled by --disable-cuda-graph");
                } else if crate::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    crate::log_info!("{arch} does not supprt CUDA graph");
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
            crate::log_error!("Unexpected message type: {:?}", msg);
            panic!("Unexpected message type");
        }
    };

    model_loaded.store(true, Ordering::SeqCst);
    loop {
        match receive_local(&mut stream, false) {
            Ok(MessageType::Shutdown) => {
                crate::log_info!("Runner exit");
                break;
            }
            Ok(MessageType::RunPrefill((sequences, is_prefill))) => {
                let outputs = runner.run(
                    Seqs::SeqRefs(&sequences.iter().collect::<Vec<_>>()),
                    is_prefill,
                );
                if outputs.is_err() {
                    crate::log_error!("Runner prefill error: {:?}", outputs);
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
                    crate::log_error!("Runner decode error: {:?}", outputs);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponse(outputs.unwrap_or(vec![])),
                    false,
                )?;
            }
            Ok(MessageType::RunEmbed((sequences, strategy))) => {
                use crate::core::sequence::Sequence;
                let refs: Vec<&Sequence> = sequences.iter().collect();
                let slice: &[&Sequence] = &refs;
                let outputs = runner.embed(&slice, &strategy);
                if outputs.is_err() {
                    crate::log_error!("Runner embedding error: {:?}", outputs);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponseEmbed(outputs.unwrap_or(vec![vec![]])),
                    false,
                )?;
            }
            Ok(MessageType::LoadingProgress(_)) => {
                crate::log_info!("Received loading progress message");
            }
            Ok(MessageType::KVCacheSwap((mappings, swap_in))) => {
                crate::log_info!(
                    "Received KVCacheSwap message: {} kv cache blocks need to {}!",
                    mappings.len(),
                    if swap_in { "swap in" } else { "swap out" },
                );
                let ret = runner.swap_kvcache(mappings, swap_in);
                if ret.is_err() {
                    crate::log_error!("KvCache Swap failed: {:?}", ret);
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
                    crate::log_error!(
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
                    crate::log_error!("HasMambaPrefixState failed for hash {}: {:?}", hash, ret);
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
                    crate::log_error!("RemoveMambaPrefixState failed for hash {}: {:?}", hash, ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RemoveMambaPrefixStateResponse(ret.unwrap_or(false)),
                    false,
                )?;
            }
            Ok(MessageType::TransferPrefill(sequence)) => {
                let ret = runner.transfer_prefill(&sequence);
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
                    crate::log_error!("KvCacheSend failed: {:?}", ret);
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
                    crate::log_error!("KvCacheReceive failed: {:?}", ret);
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
                    crate::log_error!("ClearBlocks failed: {:?}", ret);
                }
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::ClearBlocksResponse(ret.is_ok()),
                    false,
                )?;
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    crate::log_error!("Runner exit with error: {:?}", e);
                }
                break;
            }
            _ => {
                crate::log_error!("Unexpected message type");
            }
        }
    }
    stop_flag.store(true, Ordering::Relaxed);
    crate::log_info!("Runner finished");
    std::process::exit(0);
}
