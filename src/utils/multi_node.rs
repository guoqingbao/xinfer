use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const MAX_TCP_MESSAGE_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;

/// TCP-based NCCL unique ID exchange for multi-node tensor parallelism.
///
/// Node 0 generates the NCCL ID, listens on `master_addr:master_port`,
/// and sends the 128-byte ID to each connecting worker node.
/// Worker nodes connect, receive the ID, and use it to join the NCCL
/// communicator with their global rank.

#[cfg(feature = "nccl")]
pub fn coordinate_nccl_id(
    num_nodes: usize,
    node_rank: usize,
    master_addr: &str,
    master_port: u16,
) -> std::io::Result<Id> {
    if node_rank == 0 {
        master_distribute_nccl_id(num_nodes, master_addr, master_port)
    } else {
        worker_receive_nccl_id(master_addr, master_port)
    }
}

/// Master (node 0): generate NCCL ID, accept connections from (num_nodes - 1)
/// worker nodes, send them the 128-byte raw ID.
#[cfg(feature = "nccl")]
fn master_distribute_nccl_id(
    num_nodes: usize,
    master_addr: &str,
    master_port: u16,
) -> std::io::Result<Id> {
    let id = Id::new().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to create NCCL ID: {:?}", e),
        )
    })?;

    let raw_id: &[i8; 128] = id.internal();
    let raw_bytes: &[u8] = unsafe { std::slice::from_raw_parts(raw_id.as_ptr() as *const u8, 128) };

    let bind_addr = format!("{}:{}", master_addr, master_port);
    crate::log_info!(
        "[Multi-Node] Master node listening on {} for {} worker node(s)...",
        bind_addr,
        num_nodes - 1
    );

    let listener = TcpListener::bind(&bind_addr).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "Failed to bind multi-node coordinator at {}: {}",
                bind_addr, e
            ),
        )
    })?;

    let mut connected = 0;
    while connected < num_nodes - 1 {
        let (mut stream, peer_addr) = listener.accept()?;
        crate::log_info!(
            "[Multi-Node] Worker connected from {} ({}/{})",
            peer_addr,
            connected + 1,
            num_nodes - 1
        );
        stream.write_all(raw_bytes)?;
        stream.flush()?;

        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack)?;
        if ack[0] != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Worker did not acknowledge NCCL ID",
            ));
        }
        connected += 1;
    }

    crate::log_info!(
        "[Multi-Node] All {} worker node(s) received NCCL ID",
        num_nodes - 1
    );
    Ok(id)
}

/// Worker (node > 0): connect to master, receive 128-byte NCCL ID.
#[cfg(feature = "nccl")]
fn worker_receive_nccl_id(master_addr: &str, master_port: u16) -> std::io::Result<Id> {
    let addr = format!("{}:{}", master_addr, master_port);
    crate::log_info!("[Multi-Node] Worker connecting to master at {}...", addr);

    let mut stream = {
        let mut attempts = 0;
        loop {
            match TcpStream::connect(&addr) {
                Ok(s) => break s,
                Err(e) => {
                    attempts += 1;
                    if attempts > 120 {
                        return Err(std::io::Error::new(
                            e.kind(),
                            format!(
                                "Failed to connect to master at {} after {} attempts: {}",
                                addr, attempts, e
                            ),
                        ));
                    }
                    crate::log_info!(
                        "[Multi-Node] Waiting for master at {} (attempt {})...",
                        addr,
                        attempts
                    );
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }
    };

    let mut raw_bytes = [0u8; 128];
    stream.read_exact(&mut raw_bytes)?;

    stream.write_all(&[1u8])?;
    stream.flush()?;

    let mut arr = [0i8; 128];
    unsafe {
        std::ptr::copy_nonoverlapping(raw_bytes.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
    }

    #[cfg(not(target_arch = "aarch64"))]
    let id = Id::uninit(arr);
    #[cfg(target_arch = "aarch64")]
    let id = Id::uninit(arr.map(|b| b as u8));

    crate::log_info!("[Multi-Node] Received NCCL ID from master");
    Ok(id)
}

/// Multi-node forward-pass coordination.
///
/// On non-master nodes, runners need to receive forward-pass commands
/// from the master node via TCP and respond with results.
/// The master node broadcasts forward commands to all remote worker nodes'
/// TCP listeners and collects their responses.

/// Configuration for a multi-node setup
#[derive(Clone, Debug)]
pub struct MultiNodeConfig {
    pub num_nodes: usize,
    pub node_rank: usize,
    pub master_addr: String,
    pub master_port: u16,
    pub local_num_gpus: usize,
}

impl MultiNodeConfig {
    pub fn global_world_size(&self) -> usize {
        self.num_nodes * self.local_num_gpus
    }

    pub fn global_rank_offset(&self) -> usize {
        self.node_rank * self.local_num_gpus
    }

    pub fn is_master(&self) -> bool {
        self.node_rank == 0
    }

    /// Port used for forward-pass coordination TCP connections.
    /// Offset from master_port by 1 to avoid collision with NCCL ID exchange.
    pub fn forward_coord_port(&self) -> std::io::Result<u16> {
        self.master_port.checked_add(1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--master-port must be less than 65535 for multi-node coordination",
            )
        })
    }
}

/// On the master node: accept TCP connections from worker nodes for
/// forward-pass coordination. Returns one TcpStream per worker node.
pub fn master_accept_worker_streams(config: &MultiNodeConfig) -> std::io::Result<Vec<TcpStream>> {
    let bind_addr = format!("{}:{}", config.master_addr, config.forward_coord_port()?);
    crate::log_info!(
        "[Multi-Node] Master listening for forward coordination on {}...",
        bind_addr
    );

    let listener = TcpListener::bind(&bind_addr)?;
    let mut streams = Vec::new();
    let expected = config.num_nodes - 1;

    while streams.len() < expected {
        let (stream, peer_addr) = listener.accept()?;
        stream.set_nodelay(true)?;
        crate::log_info!(
            "[Multi-Node] Forward coord: worker connected from {} ({}/{})",
            peer_addr,
            streams.len() + 1,
            expected
        );
        streams.push(stream);
    }

    crate::log_info!("[Multi-Node] All worker nodes connected for forward coordination");
    Ok(streams)
}

/// On worker nodes: connect to master for forward-pass coordination.
pub fn worker_connect_to_master(config: &MultiNodeConfig) -> std::io::Result<TcpStream> {
    let addr = format!("{}:{}", config.master_addr, config.forward_coord_port()?);
    crate::log_info!(
        "[Multi-Node] Worker node {} connecting to master for forward coordination at {}...",
        config.node_rank,
        addr
    );

    let stream = {
        let mut attempts = 0;
        loop {
            match TcpStream::connect(&addr) {
                Ok(s) => break s,
                Err(e) => {
                    attempts += 1;
                    if attempts > 120 {
                        return Err(std::io::Error::new(
                            e.kind(),
                            format!(
                                "Failed to connect to master forward coord at {} after {} attempts: {}",
                                addr, attempts, e
                            ),
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }
    };
    stream.set_nodelay(true)?;
    crate::log_info!(
        "[Multi-Node] Worker node {} connected to master for forward coordination",
        config.node_rank
    );
    Ok(stream)
}

/// Send a length-prefixed bincode message over a TcpStream.
pub fn send_tcp(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("TCP message too large: {} bytes", data.len()),
        )
    })?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(data)?;
    stream.flush()?;
    Ok(())
}

/// Receive a length-prefixed bincode message from a TcpStream.
pub fn recv_tcp(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_TCP_MESSAGE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("TCP message length {} exceeds limit", len),
        ));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(feature = "nccl")]
fn forward_to_local_runners(
    runner_streams: &mut Vec<interprocess::local_socket::Stream>,
    msg: &crate::runner::MessageType,
) -> candle_core::Result<crate::runner::MessageType> {
    use crate::runner::{receive_local, send_local, MessageType};
    use interprocess::local_socket::Stream as LocalStream;
    use interprocess::TryClone;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let cloned_streams: Vec<LocalStream> = runner_streams
        .iter_mut()
        .map(|s| s.try_clone().expect("clone failed"))
        .collect();

    let responses: candle_core::Result<Vec<MessageType>> = cloned_streams
        .into_par_iter()
        .map(|mut stream| {
            send_local(&mut vec![stream.try_clone()?], msg, false)?;
            receive_local(&mut stream, false).map_err(candle_core::Error::wrap)
        })
        .collect();

    responses?
        .into_iter()
        .next()
        .ok_or_else(|| candle_core::Error::Msg("No response from local runners".to_string()))
}

/// Run the worker daemon on non-master nodes.
///
/// This function:
/// 1. Loads the model on local GPUs using global NCCL ranks
/// 2. Connects to the master node for forward-pass coordination
/// 3. Loops: receives forward commands via TCP, dispatches to local runners
///    via their existing local IPC, sends responses back to master
#[cfg(feature = "nccl")]
pub fn run_worker_daemon(
    econfig: &crate::utils::config::EngineConfig,
    dtype: candle_core::DType,
) -> candle_core::Result<()> {
    use crate::core::engine::LLMEngine;
    use crate::core::runner::RunnerType;
    use crate::runner::MessageType;

    crate::log_info!(
        "[Multi-Node Worker] Starting daemon on node {} with {} local GPUs",
        econfig.node_rank,
        econfig.device_ids.as_ref().map(|d| d.len()).unwrap_or(1)
    );

    let engine = LLMEngine::new(econfig, dtype)?;

    let mn_config = MultiNodeConfig {
        num_nodes: econfig.num_nodes,
        node_rank: econfig.node_rank,
        master_addr: econfig.master_addr.clone().unwrap_or_default(),
        master_port: econfig.master_port,
        local_num_gpus: econfig.device_ids.as_ref().map(|d| d.len()).unwrap_or(1),
    };

    let mut master_stream = worker_connect_to_master(&mn_config)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to connect to master: {}", e)))?;

    crate::log_info!("[Multi-Node Worker] Entering forward daemon loop...");

    loop {
        let data = match recv_tcp(&mut master_stream) {
            Ok(d) => d,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    crate::log_info!("[Multi-Node Worker] Master disconnected, shutting down");
                    break;
                }
                crate::log_error!("[Multi-Node Worker] TCP recv error: {:?}", e);
                break;
            }
        };

        let msg: MessageType = match bincode::deserialize(&data) {
            Ok(m) => m,
            Err(e) => {
                crate::log_error!("[Multi-Node Worker] Failed to deserialize message: {:?}", e);
                continue;
            }
        };

        match msg {
            MessageType::RunPrefill(_)
            | MessageType::RunDecode(_)
            | MessageType::RunEmbed(_)
            | MessageType::KVCacheSwap(_)
            | MessageType::CaptureMambaPrefixState(_)
            | MessageType::HasMambaPrefixState(_)
            | MessageType::RemoveMambaPrefixState(_)
            | MessageType::TransferPrefill(_)
            | MessageType::ReceivePrefill(_)
            | MessageType::CheckPrefillStatus(_)
            | MessageType::KvCacheSend(_)
            | MessageType::KvCacheReceive(_)
            | MessageType::KvCacheRelease(_)
            | MessageType::CheckKvCacheRelease(_)
            | MessageType::ClearBlocks(_) => {
                let runners = engine.read().runners.clone();
                let response = {
                    let mut guard = runners.write();
                    match &mut *guard {
                        RunnerType::Process(ref mut runner_streams) => {
                            match forward_to_local_runners(runner_streams, &msg) {
                                Ok(response) => response,
                                Err(e) => MessageType::Error(format!("{:?}", e)),
                            }
                        }
                        _ => {
                            crate::log_error!("[Multi-Node Worker] Expected Process runner type");
                            MessageType::Error("Expected Process runner type".to_string())
                        }
                    }
                };

                let response_data =
                    bincode::serialize(&response).expect("Failed to serialize response");
                if let Err(e) = send_tcp(&mut master_stream, &response_data) {
                    crate::log_error!("[Multi-Node Worker] Failed to send response: {:?}", e);
                    break;
                }
            }
            MessageType::FinishDecode(id) => {
                let mut guard = engine.write();
                let _ = guard.notify_runner_finished(id);
            }
            MessageType::Shutdown => {
                crate::log_info!("[Multi-Node Worker] Received shutdown, exiting");
                break;
            }
            _ => {
                crate::log_warn!("[Multi-Node Worker] Ignoring unhandled message type");
            }
        }
    }

    crate::log_info!("[Multi-Node Worker] Daemon exited");
    Ok(())
}

#[cfg(not(feature = "nccl"))]
pub fn run_worker_daemon(
    _econfig: &crate::utils::config::EngineConfig,
    _dtype: candle_core::DType,
) -> candle_core::Result<()> {
    candle_core::bail!("Multi-node inference requires the `nccl` feature to be enabled");
}
