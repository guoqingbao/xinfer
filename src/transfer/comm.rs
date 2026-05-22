// src/core/transfer/comm.rs
use super::{FinishedPrefillData, PdConfig, PdRole, TransferMessage};
use candle_core::Result;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::traits::Stream;
use interprocess::local_socket::{
    GenericFilePath, ListenerOptions, Stream as LocalStream, ToFsName,
};
use interprocess::TryClone;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use url::Url;
/// An internal enum to abstract over the two stream types.
/// It implements Read and Write to be used generically.
enum CommStream {
    Local(LocalStream),
    Remote(TcpStream),
}

impl Read for CommStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            CommStream::Local(s) => s.read(buf),
            CommStream::Remote(s) => s.read(buf),
        }
    }
}

impl Write for CommStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            CommStream::Local(s) => s.write(buf),
            CommStream::Remote(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            CommStream::Local(s) => s.flush(),
            CommStream::Remote(s) => s.flush(),
        }
    }
}

/// The main Communicator struct.
/// This holds a thread-safe, persistent connection (once established)
/// and can be cloned to share between the main thread (for sending)
/// and the listener thread (for receiving).
#[derive(Clone)]
pub struct Communicator {
    /// The stream is None until a connection is established.
    /// Read-half of the connection
    reader: Arc<Mutex<Option<CommStream>>>,
    /// Write-half of the connection
    writer: Arc<Mutex<Option<CommStream>>>,
    config: PdConfig,
    role: PdRole,
    rank: usize,
}

impl Communicator {
    /// Creates a new Communicator, ready to connect.
    pub fn new(config: PdConfig, role: PdRole, rank: usize) -> Self {
        Self {
            reader: Arc::new(Mutex::new(None)),
            writer: Arc::new(Mutex::new(None)),
            config,
            role,
            rank,
        }
    }

    /// Public method to send a message.
    /// This is called from the *main thread* (e.g., Scheduler).
    /// It locks the stream, sends the data, and unlocks.
    pub fn send(&self, msg: &TransferMessage) -> Result<bool> {
        let mut guard = self.writer.lock();
        if let Some(stream) = guard.as_mut() {
            send_message_generic(stream, msg)
        } else {
            candle_core::bail!(
                "[{:?} Rank {}] Communicator not connected, cannot send message.",
                self.role,
                self.rank
            );
        }
    }

    /// Internal method to receive a message.
    /// This is called from the *listener thread* in a loop.
    fn receive(&self) -> Result<TransferMessage> {
        let mut guard = self.reader.lock();
        if let Some(stream) = guard.as_mut() {
            receive_message_generic(stream)
        } else {
            // This should ideally not happen if called from run_listener_loop
            // as the stream is guaranteed to be Some.
            candle_core::bail!(
                "[{:?} Rank {}] Communicator not connected, cannot receive message.",
                self.role,
                self.rank
            );
        }
    }

    /// The main listener loop, intended to be run in a separate thread.
    /// It handles establishing the connection and then transitions
    /// into a loop to receive and dispatch messages.
    /// It also handles connection drops and retries.
    pub fn run_listener_loop(
        &self,
        pending_prefills: Arc<Mutex<VecDeque<crate::core::sequence::Sequence>>>,
        finished_data: Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
        server_tasks: Arc<RwLock<Vec<usize>>>,
        available_tokens: Arc<RwLock<usize>>,
        model_loaded: Arc<AtomicBool>,
        stop_flag: Arc<AtomicBool>,
    ) {
        // --- Outer loop: Connection Establishing ---
        loop {
            match self.establish_connection() {
                Ok(_) => {
                    if self.rank == 0 {
                        crate::log_info!("[{:?}] PD Connection established.", self.role,);
                    }
                }
                Err(e) => {
                    if model_loaded.load(Ordering::SeqCst) {
                        if self.rank == 0 {
                            crate::log_error!(
                                "[{:?}] Failed to establish PD connection: {}. Retrying in 5s...",
                                self.role,
                                e
                            );
                        }
                    }
                    if stop_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    std::thread::sleep(Duration::from_secs(5));
                    continue; // Retry connection
                }
            }

            // --- Inner loop: Message Receiving ---
            // Now that connection is established, self.stream is Some.
            loop {
                match self.receive() {
                    Ok(msg) => self.handle_received_message(
                        msg,
                        &pending_prefills,
                        &finished_data,
                        &server_tasks,
                        &available_tokens,
                    ),
                    Err(e) => {
                        crate::log_error!(
                            "[{:?} Rank {}] Connection error: {}. Re-establishing...",
                            self.role,
                            self.rank,
                            e
                        );
                        *self.reader.lock() = None;
                        *self.writer.lock() = None;
                        break; // Break inner loop to re-establish connection
                    }
                }
            }
        }
    }

    /// Blocks until a connection is established based on role and config.
    /// On success, it populates `self.stream`.
    ///
    /// Connection modes:
    /// - `http://` or `tcp://` → RemoteIPC (TCP connection)
    /// - `file://` → LocalIPC with filesystem-based socket at the specified path
    /// - `None` → LocalIPC with filesystem-based socket at default path
    fn establish_connection(&self) -> Result<()> {
        let (read_stream, write_stream) = match &self.config.url {
            Some(url_str) => {
                if let Ok(url) = Url::parse(url_str) {
                    match url.scheme() {
                        "http" | "tcp" => {
                            // --- Remote (TCP) Mode ---
                            self.connect_remote_tcp(&url)?
                        }
                        "file" => {
                            // --- Local (File-based IPC) Mode ---
                            let sock_name = format!("{}@xinfer-transfer-{}", url.path(), self.rank);
                            self.connect_local_ipc(&sock_name)?
                        }
                        _ => {
                            panic!("{} is not a supported URL scheme", url.scheme());
                        }
                    }
                } else {
                    // URL parsing failed, treat as invalid
                    panic!("Invalid pd_url: {}", url_str);
                }
            }
            None => {
                // --- Local (File-based IPC) Mode with default path ---
                let sock_name = format!("@xinfer-transfer-{}", self.rank);
                self.connect_local_ipc(&sock_name)?
            }
        };

        // Store the newly established streams
        *self.reader.lock() = Some(read_stream);
        *self.writer.lock() = Some(write_stream);
        Ok(())
    }

    /// Establishes a TCP connection for remote IPC.
    fn connect_remote_tcp(&self, url: &Url) -> Result<(CommStream, CommStream)> {
        match self.role {
            PdRole::Client => {
                let addr = format!(
                    "{}:{}",
                    url.host_str().unwrap_or("127.0.0.1"),
                    url.port().unwrap_or(8100)
                );
                let stream = TcpStream::connect(&addr)?;
                stream.set_nodelay(true)?;
                crate::log_info!(
                    "[PD Client Rank {}] Connected to TCP server at {}",
                    self.rank,
                    addr
                );
                Ok((
                    CommStream::Remote(stream.try_clone()?),
                    CommStream::Remote(stream),
                ))
            }
            PdRole::Server => {
                let addr = format!(
                    "{}:{}",
                    url.host_str().unwrap_or("0.0.0.0"),
                    url.port().unwrap_or(8100)
                );
                let listener = TcpListener::bind(&addr)?;
                crate::log_info!(
                    "[PD Server Rank {}] TCP listener bound. Waiting for client on {}",
                    self.rank,
                    addr
                );
                let (stream, client_addr) = listener.accept()?;
                stream.set_nodelay(true)?;
                crate::log_info!(
                    "[PD Server Rank {}] Accepted TCP connection from {}",
                    self.rank,
                    client_addr
                );
                Ok((
                    CommStream::Remote(stream.try_clone()?),
                    CommStream::Remote(stream),
                ))
            }
        }
    }

    /// Establishes a local IPC connection using filesystem-based sockets.
    fn connect_local_ipc(&self, sock_name: &str) -> Result<(CommStream, CommStream)> {
        let fs_name = sock_name.to_string().to_fs_name::<GenericFilePath>()?;
        match self.role {
            PdRole::Client => {
                // Retry connecting to the local socket
                loop {
                    match LocalStream::connect(fs_name.clone()) {
                        Ok(stream) => {
                            if self.rank == 0 {
                                crate::log_info!(
                                    "PD Client: Connected to LocalIPC server at {}",
                                    sock_name
                                );
                            }
                            break Ok((
                                CommStream::Local(stream.try_clone()?),
                                CommStream::Local(stream),
                            ));
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            // Server might not be ready, wait and retry
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            PdRole::Server => {
                // Ensure the socket file is removed before binding
                if std::path::Path::new(sock_name).exists() {
                    let _ = std::fs::remove_file(sock_name);
                }
                let listener = ListenerOptions::new().name(fs_name).create_sync()?;
                let stream = listener.accept()?;
                if self.rank == 0 {
                    crate::log_info!("PD Server: Accepted LocalIPC connection at {}", sock_name);
                }
                Ok((
                    CommStream::Local(stream.try_clone()?),
                    CommStream::Local(stream),
                ))
            }
        }
    }

    /// Internal helper to dispatch received messages to the correct queue.
    fn handle_received_message(
        &self,
        msg: TransferMessage,
        pending_prefills: &Arc<Mutex<VecDeque<crate::core::sequence::Sequence>>>,
        finished_data: &Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
        server_tasks: &Arc<RwLock<Vec<usize>>>,
        available_tokens: &Arc<RwLock<usize>>,
    ) {
        match (&self.role, msg) {
            // Client receives KV cache
            (PdRole::Client, TransferMessage::TransferKvCache(data)) => {
                if self.rank == 0 {
                    crate::log_info!("PD Client: KvCache for Seq {} received", data.seq_id);
                }
                finished_data.write().insert(data.seq_id, data);
            }
            // Server receives Prefill request
            (PdRole::Server, TransferMessage::TransferPrefill(seq)) => {
                if self.rank == 0 {
                    crate::log_info!(
                        "PD Server: received TransferPrefill for Seq {} ({} tokens)",
                        seq.id,
                        seq.len()
                    );
                }
                server_tasks.write().push(seq.id); // indicate working in progress
                pending_prefills.lock().push_back(seq);
            }
            (PdRole::Client, TransferMessage::AvailableTokenResponse(num_tokens)) => {
                *available_tokens.write() = num_tokens;
            }
            (PdRole::Server, TransferMessage::AvailableTokenResponse(_)) => {
                crate::log_warn!(
                    "[PD Client Rank {}] Received unexpected AvailableTokenResponse msg",
                    self.rank
                );
            }
            (PdRole::Server, TransferMessage::ReleaseKvCache(seq_id)) => {
                // if self.rank == 0 {
                //     crate::log_info!("PD Server: received ReleaseKvCache for Seq {}", seq_id);
                // }
                server_tasks.write().retain(|&id| id != seq_id); // remove, indicate the server need to release this cache
            }
            // Mismatched messages (warn and drop)
            (PdRole::Client, TransferMessage::TransferPrefill(_)) => {
                crate::log_warn!(
                    "[PD Client Rank {}] Received unexpected TransferPrefill msg",
                    self.rank
                );
            }
            (PdRole::Server, TransferMessage::TransferKvCache(_)) => {
                crate::log_warn!(
                    "[PD Server Rank {}] Received unexpected TransferKvCache msg",
                    self.rank
                );
            }
            (PdRole::Client, TransferMessage::ReleaseKvCache(_)) => {
                crate::log_warn!(
                    "[PD Client Rank {}] Received unexpected ReleaseKvCache msg",
                    self.rank
                );
            }
        }
    }
}

/// Generic, standardized function to send a message.
/// Uses a 4-byte LE length prefix followed by bincode data.
fn send_message_generic(stream: &mut (impl Read + Write), msg: &TransferMessage) -> Result<bool> {
    let serialized: Vec<u8> = bincode::serialize(msg).map_err(candle_core::Error::wrap)?;
    let len = serialized.len() as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&serialized)?;
    stream.flush()?;
    Ok(true)
}

/// Generic, standardized function to receive a message.
/// Reads a 4-byte LE length prefix then bincode data.
fn receive_message_generic(stream: &mut (impl Read + Write)) -> Result<TransferMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    if len > 1_000_000_000 {
        // 1GB sanity check
        candle_core::bail!("Message size too large: {} bytes", len);
    }

    let mut msg_buf = vec![0u8; len];
    stream.read_exact(&mut msg_buf)?;

    let msg = bincode::deserialize(&msg_buf).map_err(candle_core::Error::wrap)?;
    Ok(msg)
}
