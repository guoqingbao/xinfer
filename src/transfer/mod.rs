// src/core/transfer/mod.rs
use crate::core::sequence::Sequence;
use crate::runner::SerializableDType;
use crate::transfer::comm::Communicator;
use candle_core::{DType, Result, Tensor, WithDType};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;
mod comm;
#[cfg(feature = "cuda")]
mod cuda_remote;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::CudaDType as MsgDtype;
#[cfg(feature = "python")]
use pyo3::pyclass;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(not(feature = "cuda"))]
use Sized as MsgDtype;

/// Defines the role of the current inference engine instance.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdRole {
    /// The main instance, handles decoding and orchestrates prefills.
    Client = 1,
    /// A worker instance, dedicated to executing prefills.
    Server = 2,
}

/// The mechanism used to transfer KV cache data.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdMethod {
    /// Use CUDA IPC handles for D2D transfer (fastest, local machine only).
    LocalIpc = 1,
    /// Use TCP for remote transfer (inter-machine).
    RemoteTcp = 2,
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PdConfig {
    /// Is this instance a Client or a PDServer?
    pub role: PdRole,
    /// The chosen transfer method.
    pub method: PdMethod,
    // The network address for the PD Server or client to listen on/connect to (e.g., "0.0.0.0:9000")
    pub url: Option<String>,
}

/// Configuration for the Transfer sub-system.
#[cfg(feature = "python")]
#[pyclass]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PdConfig {
    /// Is this instance a Client or a PDServer?
    #[pyo3(get, set)]
    pub role: PdRole,
    /// The chosen transfer method.
    #[pyo3(get, set)]
    pub method: PdMethod,
    // The network address for the PD Server or client to listen on/connect to (e.g., "0.0.0.0:9000")
    #[pyo3(get, set)]
    pub url: Option<String>,
}

/// Serializable handle for a CUDA IPC memory region.
/// This is a wrapper around the `cudaIpcMemHandle_t` struct.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaIpcMemHandle(Vec<u8>, Vec<usize>, SerializableDType); // Simplified as bytes. See cuda.rs for real impl.
#[cfg(not(target_arch = "aarch64"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaIpcMemHandle(Vec<i8>, Vec<usize>, SerializableDType); // Simplified as bytes. See cuda.rs for real impl.

/// TurboQuant buffer handles for a single layer (IPC path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TqLayerIpcHandles {
    pub k_absmax: Option<CudaIpcMemHandle>,
    pub k_quant: Option<CudaIpcMemHandle>,
    pub v_absmax: CudaIpcMemHandle,
    pub v_quant: CudaIpcMemHandle,
}

/// TurboQuant buffer data for a single layer (TCP path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TqLayerTcpData {
    pub k_absmax: Option<Vec<u8>>,
    pub k_quant: Option<Vec<u8>>,
    pub v_absmax: Vec<u8>,
    pub v_quant: Vec<u8>,
}

/// A handle abstracting *how* to get the KV cache data.
/// This is serialized and sent from the PDServer to the Client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KVTransferHandle {
    /// For local D2D copy. Contains IPC handles for all layers' K/V tensors.
    LocalIpc {
        /// `Vec` is over `num_layers`. Each `(k, v)` is an IPC handle.
        /// These handles are specific to the server's *rank*.
        layer_handles: Vec<(CudaIpcMemHandle, CudaIpcMemHandle)>,
        /// The *server's* GPU block IDs that contain the data for this sequence.
        server_block_ids: Vec<u32>,
        /// TurboQuant buffer IPC handles (one per layer). Present when TQ is active.
        tq_layer_handles: Option<Vec<TqLayerIpcHandles>>,
    },
    /// For remote HtoD copy. Contains the raw KV cache data.
    RemoteTcp {
        /// Raw bytes of the KV cache data, copied block-by-block from the server's CPU.
        /// `Vec` is over `num_layers`. `(k_bytes, v_bytes)`.
        /// The inner `Vec<u8>` contains all blocks for that layer, concatenated.
        layer_data: Vec<(Vec<u8>, Vec<u8>)>,
        /// The number of blocks (tokens) in the sequence.
        num_blocks: usize,
        /// TurboQuant buffer data (one per layer). Present when TQ is active.
        tq_layer_data: Option<Vec<TqLayerTcpData>>,
    },
}

/// Data sent from the PDServer to the Client upon prefill completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishedPrefillData {
    pub seq_id: usize,
    pub first_token: u32,
    pub transfer_handle: KVTransferHandle,
    pub sending_time: usize,
    /// Prefix-cache hit length recorded on the prefill server. Restored onto
    /// the decode-side seq so `Usage.prompt_tokens_details.cached_tokens`
    /// reports the right value across PD.
    pub num_cached_tokens: usize,
}

/// Messages for communication between Client and PDServer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMessage {
    /// Client -> Server: Request to prefill a new sequence.
    TransferPrefill(Sequence),
    AvailableTokenResponse(usize),
    /// Server -> Client: Prefill finished, contains KV cache handle.
    TransferKvCache(FinishedPrefillData),
    // Client -> Server: Request to release prefill kvcache
    ReleaseKvCache(usize),
}

/// The main Transfer struct that orchestrates PD disaggregation.
#[allow(dead_code)]
pub struct Transfer {
    config: PdConfig,
    /// (Client-side) Holds completed prefill data from the server.
    finished_data: Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
    /// (Server-side) Holds incoming prefill requests from the client.
    pending_prefills: Arc<Mutex<VecDeque<Sequence>>>,
    /// (Server-side) Holds sequences that kvcache unreleased
    server_tasks: Arc<RwLock<Vec<usize>>>,
    available_tokens: Arc<RwLock<usize>>,
    /// Handle to the background communication thread.
    comm_handle: Option<JoinHandle<()>>,
    rank: usize,
    communicator: Communicator,
}

impl Transfer {
    /// Creates a new Transfer instance and starts its communication thread.
    pub fn new(
        config: PdConfig,
        rank: usize,
        model_loaded: Arc<AtomicBool>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<Self> {
        let finished_data = Arc::new(RwLock::new(HashMap::new()));
        let pending_prefills = Arc::new(Mutex::new(VecDeque::new()));
        let server_tasks = Arc::new(RwLock::new(Vec::new()));
        let available_tokens = Arc::new(RwLock::new(0));
        // Clone Arcs for the background thread
        let thread_finished_data = finished_data.clone();
        let thread_pending_prefills = pending_prefills.clone();
        let thread_server_tasks = server_tasks.clone();
        let thread_available_tokens = available_tokens.clone();

        let communicator = Communicator::new(config.clone(), config.role.clone(), rank);
        // Start the background communication thread
        let thread_comm = communicator.clone();
        let comm_handle = Some(std::thread::spawn(move || {
            thread_comm.run_listener_loop(
                thread_pending_prefills,
                thread_finished_data,
                thread_server_tasks,
                thread_available_tokens,
                model_loaded,
                stop_flag,
            );
        }));

        Ok(Self {
            config,
            finished_data,
            pending_prefills,
            server_tasks,
            comm_handle,
            rank,
            communicator,
            available_tokens,
        })
    }

    pub fn is_client(&self) -> bool {
        matches!(self.config.role, PdRole::Client)
    }

    pub fn is_server(&self) -> bool {
        matches!(self.config.role, PdRole::Server)
    }

    // --- Client-side API ---

    /// (Client) Sends a sequence to the PDServer for prefill.
    pub fn transfer_prefill(&self, seq: &Sequence) -> Result<bool> {
        if !self.is_client() {
            return Ok(false);
        }
        let available_tokens = *self.available_tokens.read();
        if available_tokens != 0 && seq.len() + 1 > available_tokens {
            candle_core::bail!("PD Client: transfer prefill failed because prefill length {} > available tokens {} on PD Server", seq.len(), available_tokens);
        }
        self.communicator
            .send(&TransferMessage::TransferPrefill(seq.clone()))
    }

    /// (Client) Checks if a specific prefill has finished.
    pub fn check_prefill_finished(&self, seq_id: usize) -> Result<bool> {
        Ok(self.finished_data.read().contains_key(&seq_id))
    }

    /// (Client) Remove finished data for a seq after successful receive to avoid unbounded memory.
    pub fn cleanup_finished_data(&self, seq_id: usize) {
        self.finished_data.write().remove(&seq_id);
    }

    /// (Client) Receives the KV cache data and copies it into local GPU blocks.
    /// Returns `(success, first_token, sending_time, num_cached_tokens)`.
    #[allow(unused)]
    pub fn receive_kv_cache(
        &self,
        seq: &Sequence,
        local_gpu_cache: &Vec<(Tensor, Tensor)>,
    ) -> Result<(bool, u32, usize, usize)> {
        let status = self.check_prefill_finished(seq.id)?;
        if !status {
            candle_core::bail!("Unable to receive kvcache from the PD server since this sequence is not prefill completed!")
        }

        fn read_data<T: WithDType + MsgDtype>(
            sf: &Transfer,
            seq: &Sequence,
            local_gpu_cache: &Vec<(Tensor, Tensor)>,
        ) -> Result<(bool, u32, usize, usize)> {
            let local_gpu_ids = seq.block_table.clone();
            let local_device = local_gpu_cache[0].0.device();

            let data_guard = sf.finished_data.read();
            let data = &data_guard.get(&seq.id).unwrap();
            let token = data.first_token;
            match &data.transfer_handle {
                KVTransferHandle::LocalIpc {
                    layer_handles,
                    server_block_ids,
                    tq_layer_handles,
                } => {
                    let mapping: HashMap<usize, usize> = server_block_ids
                        .iter()
                        .zip(local_gpu_ids)
                        .map(|(server_id, local_id)| (*server_id as usize, local_id as usize))
                        .collect();

                    let tq_full = matches!(
                        attention_rs::get_turboquant_mode(),
                        Some(attention_rs::TurboquantMode::Turbo4)
                            | Some(attention_rs::TurboquantMode::Turbo3)
                    );
                    #[cfg(feature = "cuda")]
                    if !tq_full {
                        for (i, (k_handle, v_handle)) in layer_handles.iter().enumerate() {
                            let remote_k_tensor =
                                cuda_remote::open_ipc_handle::<T>(k_handle, local_device)?;
                            let remote_v_tensor =
                                cuda_remote::open_ipc_handle::<T>(v_handle, local_device)?;

                            attention_rs::cache::swap_blocks(
                                &remote_k_tensor,
                                &local_gpu_cache[i].0,
                                &mapping,
                            )?;
                            attention_rs::cache::swap_blocks(
                                &remote_v_tensor,
                                &local_gpu_cache[i].1,
                                &mapping,
                            )?;
                        }
                    }

                    #[cfg(feature = "cuda")]
                    if let Some(tq_handles) = tq_layer_handles {
                        for (layer_idx, tq_h) in tq_handles.iter().enumerate() {
                            attention_rs::with_turboquant_layer(layer_idx, |tq, _| -> Result<()> {
                                let remote_va =
                                    cuda_remote::open_ipc_handle_any(&tq_h.v_absmax, local_device)?;
                                let remote_vq =
                                    cuda_remote::open_ipc_handle_any(&tq_h.v_quant, local_device)?;
                                attention_rs::cache::swap_blocks(
                                    &remote_va,
                                    &tq.v_absmax,
                                    &mapping,
                                )?;
                                attention_rs::cache::swap_blocks(
                                    &remote_vq,
                                    &tq.v_quant,
                                    &mapping,
                                )?;

                                if let (Some(ka_handle), Some(local_ka)) =
                                    (&tq_h.k_absmax, &tq.k_absmax)
                                {
                                    let remote_ka =
                                        cuda_remote::open_ipc_handle_any(ka_handle, local_device)?;
                                    attention_rs::cache::swap_blocks(
                                        &remote_ka, local_ka, &mapping,
                                    )?;
                                }
                                if let (Some(kq_handle), Some(local_kq)) =
                                    (&tq_h.k_quant, &tq.k_quant)
                                {
                                    let remote_kq =
                                        cuda_remote::open_ipc_handle_any(kq_handle, local_device)?;
                                    attention_rs::cache::swap_blocks(
                                        &remote_kq, local_kq, &mapping,
                                    )?;
                                }
                                Ok(())
                            })
                            .transpose()?;
                        }
                    }
                }
                KVTransferHandle::RemoteTcp {
                    layer_data,
                    num_blocks,
                    tq_layer_data,
                } => {
                    if local_gpu_ids.len() < *num_blocks {
                        candle_core::bail!("Not enough blocks allocated to receive KV cache");
                    }

                    let mapping: HashMap<usize, usize> = (0..*num_blocks)
                        .zip(local_gpu_ids)
                        .map(|(i, local_id)| (i, local_id as usize))
                        .collect();

                    let tq_full = matches!(
                        attention_rs::get_turboquant_mode(),
                        Some(attention_rs::TurboquantMode::Turbo4)
                            | Some(attention_rs::TurboquantMode::Turbo3)
                    );
                    if !tq_full {
                        for (i, (k_bytes, v_bytes)) in layer_data.into_iter().enumerate() {
                            let (local_k_cache, local_v_cache) = &local_gpu_cache[i];

                            let remote_k_tensor = super::transfer::bytes_to_cpu_tensor(
                                k_bytes,
                                *num_blocks,
                                local_k_cache,
                            )?;
                            let remote_v_tensor = super::transfer::bytes_to_cpu_tensor(
                                v_bytes,
                                *num_blocks,
                                local_v_cache,
                            )?;

                            attention_rs::cache::swap_blocks(
                                &remote_k_tensor,
                                &local_k_cache,
                                &mapping,
                            )?;
                            attention_rs::cache::swap_blocks(
                                &remote_v_tensor,
                                &local_v_cache,
                                &mapping,
                            )?;
                        }
                    }

                    if let Some(tq_data) = tq_layer_data {
                        for (layer_idx, tq_d) in tq_data.iter().enumerate() {
                            attention_rs::with_turboquant_layer(layer_idx, |tq, _| -> Result<()> {
                                let va_cpu = super::transfer::bytes_to_tq_cpu_tensor(
                                    &tq_d.v_absmax,
                                    *num_blocks,
                                    &tq.v_absmax,
                                )?;
                                let vq_cpu = super::transfer::bytes_to_tq_cpu_tensor(
                                    &tq_d.v_quant,
                                    *num_blocks,
                                    &tq.v_quant,
                                )?;
                                attention_rs::cache::swap_blocks(&va_cpu, &tq.v_absmax, &mapping)?;
                                attention_rs::cache::swap_blocks(&vq_cpu, &tq.v_quant, &mapping)?;

                                if let (Some(ka_bytes), Some(local_ka)) =
                                    (&tq_d.k_absmax, &tq.k_absmax)
                                {
                                    let ka_cpu = super::transfer::bytes_to_tq_cpu_tensor(
                                        ka_bytes,
                                        *num_blocks,
                                        local_ka,
                                    )?;
                                    attention_rs::cache::swap_blocks(&ka_cpu, local_ka, &mapping)?;
                                }
                                if let (Some(kq_bytes), Some(local_kq)) =
                                    (&tq_d.k_quant, &tq.k_quant)
                                {
                                    let kq_cpu = super::transfer::bytes_to_tq_cpu_tensor(
                                        kq_bytes,
                                        *num_blocks,
                                        local_kq,
                                    )?;
                                    attention_rs::cache::swap_blocks(&kq_cpu, local_kq, &mapping)?;
                                }
                                Ok(())
                            })
                            .transpose()?;
                        }
                    }
                }
            }
            Ok((true, token, data.sending_time, data.num_cached_tokens))
        }

        let dtype = local_gpu_cache[0].0.dtype();
        let result = match dtype {
            DType::F16 => read_data::<half::f16>(&self, seq, local_gpu_cache),
            DType::BF16 => read_data::<half::bf16>(&self, seq, local_gpu_cache),
            DType::U8 => read_data::<u8>(&self, seq, local_gpu_cache),
            _ => candle_core::bail!("Invalid kvcache dtype!"),
        };
        self.cleanup_finished_data(seq.id);
        result
    }

    /// (Client) Notify the server to release kvcache
    pub fn release_remote_kvcache(&self, seq_id: usize) -> Result<bool> {
        self.communicator
            .send(&TransferMessage::ReleaseKvCache(seq_id))
    }

    // --- Server-side API ---

    /// (Server) Tries to receive a new prefill request from the queue.
    pub fn try_receive_prefill_request(
        &self,
        available_tokens: usize,
    ) -> Result<(bool, Option<Sequence>)> {
        *self.available_tokens.write() = available_tokens;
        self.communicator
            .send(&TransferMessage::AvailableTokenResponse(available_tokens))?;
        if let Some(seq) = self.pending_prefills.lock().pop_front() {
            if seq.len() + 1 < available_tokens {
                Ok((true, Some(seq)))
            } else {
                crate::log_warn!(
                    "A new prefill request requires {} KvCache tokens, but KvCache only left {}",
                    seq.len() + 1,
                    available_tokens
                );
                Ok((false, Some(seq)))
            }
        } else {
            Ok((true, None))
        }
    }

    /// (Server) Creates the KVTransferHandle and sends it to the client.
    #[allow(unused)]
    pub fn transfer_kv_cache(
        &self,
        seq: &Sequence,
        server_gpu_cache: &Vec<(Tensor, Tensor)>,
        first_token: u32,
    ) -> Result<bool> {
        fn transfer_data<T: WithDType + MsgDtype>(
            sf: &Transfer,
            config: &PdConfig,
            seq: &Sequence,
            server_gpu_cache: &Vec<(Tensor, Tensor)>,
            first_token: u32,
        ) -> Result<bool> {
            let sending_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize;
            let tq_mode = attention_rs::get_turboquant_mode();
            let num_layers = server_gpu_cache.len();

            let transfer_handle = match config.method {
                PdMethod::LocalIpc => {
                    let mut layer_handles = Vec::new();
                    #[cfg(feature = "cuda")]
                    for (k_tensor, v_tensor) in server_gpu_cache.iter() {
                        let k_handle = cuda_remote::get_ipc_handle::<T>(k_tensor)?;
                        let v_handle = cuda_remote::get_ipc_handle::<T>(v_tensor)?;
                        layer_handles.push((k_handle, v_handle));
                    }

                    let tq_layer_handles = if tq_mode.is_some() {
                        let mut tq_handles = Vec::with_capacity(num_layers);
                        #[cfg(feature = "cuda")]
                        for layer_idx in 0..num_layers {
                            let h = attention_rs::with_turboquant_layer(
                                layer_idx,
                                |tq, _mode| -> Result<TqLayerIpcHandles> {
                                    let va_handle = cuda_remote::get_ipc_handle_any(&tq.v_absmax)?;
                                    let vq_handle = cuda_remote::get_ipc_handle_any(&tq.v_quant)?;
                                    let ka_handle = tq
                                        .k_absmax
                                        .as_ref()
                                        .map(|t| cuda_remote::get_ipc_handle_any(t))
                                        .transpose()?;
                                    let kq_handle = tq
                                        .k_quant
                                        .as_ref()
                                        .map(|t| cuda_remote::get_ipc_handle_any(t))
                                        .transpose()?;
                                    Ok(TqLayerIpcHandles {
                                        k_absmax: ka_handle,
                                        k_quant: kq_handle,
                                        v_absmax: va_handle,
                                        v_quant: vq_handle,
                                    })
                                },
                            )
                            .ok_or_else(|| {
                                candle_core::Error::Msg(format!(
                                    "TQ layer {} not found on server",
                                    layer_idx
                                ))
                            })??;
                            tq_handles.push(h);
                        }
                        Some(tq_handles)
                    } else {
                        None
                    };

                    KVTransferHandle::LocalIpc {
                        layer_handles,
                        server_block_ids: seq.block_table.clone(),
                        tq_layer_handles,
                    }
                }
                PdMethod::RemoteTcp => {
                    let server_block_ids = &seq.block_table;

                    let mapping: HashMap<usize, usize> = server_block_ids
                        .iter()
                        .enumerate()
                        .map(|(i, &server_id)| (server_id as usize, i))
                        .collect();

                    let tq_full = matches!(
                        tq_mode,
                        Some(attention_rs::TurboquantMode::Turbo4)
                            | Some(attention_rs::TurboquantMode::Turbo3)
                    );

                    let mut layer_data = Vec::new();
                    if !tq_full {
                        for (k_tensor, v_tensor) in server_gpu_cache.iter() {
                            let k_cpu_tensor = super::transfer::copy_blocks_to_cpu(
                                k_tensor,
                                &mapping,
                                server_block_ids.len(),
                            )?;
                            let v_cpu_tensor = super::transfer::copy_blocks_to_cpu(
                                v_tensor,
                                &mapping,
                                server_block_ids.len(),
                            )?;

                            layer_data.push((
                                super::transfer::cpu_tensor_to_bytes::<T>(&k_cpu_tensor)?,
                                super::transfer::cpu_tensor_to_bytes::<T>(&v_cpu_tensor)?,
                            ));
                        }
                    }

                    let tq_layer_data = if tq_mode.is_some() {
                        let mut tq_data = Vec::with_capacity(num_layers);
                        for layer_idx in 0..num_layers {
                            let d = attention_rs::with_turboquant_layer(
                                layer_idx,
                                |tq, _mode| -> Result<TqLayerTcpData> {
                                    let va_cpu = super::transfer::copy_blocks_to_cpu(
                                        &tq.v_absmax,
                                        &mapping,
                                        server_block_ids.len(),
                                    )?;
                                    let vq_cpu = super::transfer::copy_blocks_to_cpu(
                                        &tq.v_quant,
                                        &mapping,
                                        server_block_ids.len(),
                                    )?;
                                    let ka_cpu = tq
                                        .k_absmax
                                        .as_ref()
                                        .map(|t| {
                                            super::transfer::copy_blocks_to_cpu(
                                                t,
                                                &mapping,
                                                server_block_ids.len(),
                                            )
                                        })
                                        .transpose()?;
                                    let kq_cpu = tq
                                        .k_quant
                                        .as_ref()
                                        .map(|t| {
                                            super::transfer::copy_blocks_to_cpu(
                                                t,
                                                &mapping,
                                                server_block_ids.len(),
                                            )
                                        })
                                        .transpose()?;
                                    Ok(TqLayerTcpData {
                                        k_absmax: ka_cpu
                                            .map(|t| super::transfer::cpu_tensor_to_bytes_any(&t))
                                            .transpose()?,
                                        k_quant: kq_cpu
                                            .map(|t| super::transfer::cpu_tensor_to_bytes_any(&t))
                                            .transpose()?,
                                        v_absmax: super::transfer::cpu_tensor_to_bytes_any(
                                            &va_cpu,
                                        )?,
                                        v_quant: super::transfer::cpu_tensor_to_bytes_any(&vq_cpu)?,
                                    })
                                },
                            )
                            .ok_or_else(|| {
                                candle_core::Error::Msg(format!(
                                    "TQ layer {} not found on server",
                                    layer_idx
                                ))
                            })??;
                            tq_data.push(d);
                        }
                        Some(tq_data)
                    } else {
                        None
                    };

                    KVTransferHandle::RemoteTcp {
                        layer_data,
                        num_blocks: seq.block_table.len(),
                        tq_layer_data,
                    }
                }
            };

            let msg = TransferMessage::TransferKvCache(FinishedPrefillData {
                seq_id: seq.id,
                first_token,
                transfer_handle,
                sending_time,
                num_cached_tokens: seq.num_cached_tokens,
            });
            sf.communicator.send(&msg)
        }

        let dtype = server_gpu_cache[0].0.dtype();
        match dtype {
            DType::F16 => {
                transfer_data::<half::f16>(&self, &self.config, seq, server_gpu_cache, first_token)?
            }
            DType::BF16 => transfer_data::<half::bf16>(
                &self,
                &self.config,
                seq,
                server_gpu_cache,
                first_token,
            )?,
            DType::U8 => {
                transfer_data::<u8>(&self, &self.config, seq, server_gpu_cache, first_token)?
            }
            _ => candle_core::bail!("Invalid kvcache dtype!"),
        };
        Ok(true)
    }

    /// (Server) Checks if a specific prefill need to release kvcache.
    pub fn check_kvcache_release(&self, seq_id: usize) -> Result<bool> {
        Ok(!self.server_tasks.read().contains(&seq_id))
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        // TODO: Add logic to gracefully shut down the comm_handle thread
    }
}

/// (Client) Converts raw TQ bytes back into a CPU tensor for HtoD copy.
/// Uses the GPU tensor's shape with dim(0) replaced by `num_blocks`.
pub fn bytes_to_tq_cpu_tensor(
    bytes: &Vec<u8>,
    num_blocks: usize,
    gpu_tensor_template: &Tensor,
) -> Result<Tensor> {
    let dtype = gpu_tensor_template.dtype();
    let mut cpu_shape = gpu_tensor_template.shape().dims().to_vec();
    cpu_shape[0] = num_blocks;
    Tensor::from_raw_buffer(bytes, dtype, &cpu_shape, &candle_core::Device::Cpu)
}

/// (Client) Converts raw bytes back into a CPU tensor for HtoD copy.
pub fn bytes_to_cpu_tensor(
    bytes: &Vec<u8>,
    num_blocks: usize,
    gpu_tensor_template: &Tensor, // Used for shape/dtype
) -> Result<Tensor> {
    let dtype = gpu_tensor_template.dtype();
    let mut cpu_shape = gpu_tensor_template.shape().dims().to_vec();
    cpu_shape[0] = num_blocks;

    // This is a candle-specific way to create a tensor from raw bytes
    Tensor::from_raw_buffer(bytes, dtype, &cpu_shape, &candle_core::Device::Cpu)
}

/// (Server) Converts a CPU tensor to raw bytes for network transfer.
pub fn cpu_tensor_to_bytes<T: WithDType>(cpu_tensor: &Tensor) -> Result<Vec<u8>> {
    use candle_core::Storage;
    if !cpu_tensor.is_contiguous() {
        candle_core::bail!("CPU tensor must be contiguous to serialize");
    }
    let (storage, _) = cpu_tensor.storage_and_layout();
    let Storage::Cpu(src_storage) = &*storage else {
        candle_core::bail!("Invalid source kvcache storage!")
    };
    let src_slice: &[T] = src_storage.as_slice()?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            src_slice.as_ptr() as *const u8,
            src_slice.len() * std::mem::size_of::<T>(),
        )
    };
    Ok(bytes.to_vec())
}

/// Dtype-dispatching version of cpu_tensor_to_bytes.
pub fn cpu_tensor_to_bytes_any(cpu_tensor: &Tensor) -> Result<Vec<u8>> {
    match cpu_tensor.dtype() {
        DType::U8 => cpu_tensor_to_bytes::<u8>(cpu_tensor),
        DType::F32 => cpu_tensor_to_bytes::<f32>(cpu_tensor),
        DType::F16 => cpu_tensor_to_bytes::<half::f16>(cpu_tensor),
        DType::BF16 => cpu_tensor_to_bytes::<half::bf16>(cpu_tensor),
        dt => candle_core::bail!("Unsupported dtype for tensor bytes: {:?}", dt),
    }
}

/// (Server) Copies specific blocks from a GPU tensor to a new, contiguous CPU tensor.
pub fn copy_blocks_to_cpu(
    gpu_tensor: &Tensor,
    mapping: &HashMap<usize, usize>,
    num_blocks: usize,
) -> Result<Tensor> {
    // Create a new destination tensor on the CPU
    let mut cpu_shape = gpu_tensor.shape().dims().to_vec();
    cpu_shape[0] = num_blocks;
    let cpu_tensor = Tensor::zeros(cpu_shape, gpu_tensor.dtype(), &candle_core::Device::Cpu)?;

    // Use swap_blocks to copy from sparse locations in `gpu_tensor` to
    //    contiguous locations in `cpu_tensor`.
    //    mapping: { server_block_id -> contiguous_index (0, 1, 2...) }
    attention_rs::cache::swap_blocks(gpu_tensor, &cpu_tensor, mapping)?;

    Ok(cpu_tensor)
}
