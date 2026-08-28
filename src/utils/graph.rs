#[cfg(feature = "flashinfer")]
use super::FlashInferKvParams;
use crate::models::layers::linear::set_linear_is_prefill;
use attention_rs::InputMetadata;
use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::sys::{
    lib, CUgraphInstantiate_flags, CUmemPool_attribute, CUmemoryPool, CUstreamCaptureMode,
    CUstreamCaptureStatus,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{DType, Device, Result, Tensor};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use tqdm::tqdm;

#[allow(dead_code)]
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: sys::CUstream,
}

impl CudaGraph {
    pub fn begin_capture(stream: sys::CUstream, mode: sys::CUstreamCaptureMode) -> Result<()> {
        unsafe {
            lib()
                .cuStreamBeginCapture_v2(stream, mode)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("begin_capture failed: {e:?}")))
        }
    }

    pub fn end_capture(stream: sys::CUstream, flags: u64) -> Result<CudaGraph> {
        let mut graph = MaybeUninit::uninit();
        let cu_graph = unsafe {
            lib()
                .cuStreamEndCapture(stream, graph.as_mut_ptr())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuStreamEndCapture failed: {e:?}"))
                })?;
            graph.assume_init()
        };

        let mut graph_exec = MaybeUninit::uninit();
        let cu_graph_exec = unsafe {
            lib()
                .cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), cu_graph, flags)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
            graph_exec.assume_init()
        };
        Ok(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream,
        })
    }

    pub fn capture_status(stream: sys::CUstream) -> Result<sys::CUstreamCaptureStatus> {
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            lib()
                .cuStreamIsCapturing(stream, &mut status)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
        }
        Ok(status)
    }

    pub fn launch(&self) -> Result<()> {
        unsafe {
            lib()
                .cuGraphLaunch(self.cu_graph_exec, self.stream)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuGraphLaunch failed: {e:?}")))
        }
    }
}

pub trait CudaGraphModule {
    fn start_capture(&mut self, bs: usize) -> Result<()>;
    fn end_capture(&mut self, save: bool) -> Result<()>;
    fn replay(&self, bs: usize) -> Result<()>;
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor>;
    fn report_graph_pool_usage(&self) -> Result<()>;
}

pub struct CudaGraphHandle {
    graph: Arc<CudaGraph>,
}

impl CudaGraphHandle {
    pub fn new(graph: Arc<CudaGraph>) -> Self {
        Self { graph }
    }

    pub fn replay(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("CUDA Graph launch failed: {:?}", e)))?;
        Ok(())
    }
}

pub struct CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    module: M,
    captured_graphs: BTreeMap<usize, CudaGraphHandle>,
    capturing: bool,
    current_bs: Option<usize>,
    device: Arc<CudaDevice>,
    pub pool_handle: RwLock<Option<i64>>,
    captured_bs: Vec<usize>,
}

impl<M> CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    pub fn new(module: M, device: Arc<CudaDevice>) -> Self {
        Self {
            module,
            captured_graphs: BTreeMap::new(),
            capturing: false,
            current_bs: None,
            device,
            pool_handle: RwLock::new(None),
            captured_bs: Vec::new(),
        }
    }

    fn sync_stream(&self) -> Result<()> {
        unsafe {
            lib()
                .cuStreamSynchronize(self.device.cu_stream().clone())
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuStreamSynchronize failed: {e:?}")))
        }
    }

    fn create_capture_pool(&self) -> Result<CUmemoryPool> {
        let mut pool: CUmemoryPool = ptr::null_mut();
        unsafe {
            lib()
                .cuDeviceGetDefaultMemPool(&mut pool, *self.device.cu_device())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
                })?;

            let handle = pool as *mut std::ffi::c_void as usize as i64;
            *self.pool_handle.write() = Some(handle);

            let threshold: u64 = u64::MAX;
            lib()
                .cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const _ as _,
                )
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}"))
                })?;
        }
        Ok(pool)
    }

    fn set_capture_mem_pool(&self) -> Result<()> {
        if self.pool_handle.read().is_some() {
            return Ok(());
        }

        unsafe {
            let status = CudaGraph::capture_status(self.device.cu_stream().clone())?;
            if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
                let pool = self.create_capture_pool()?;
                lib()
                    .cuDeviceSetMemPool(*self.device.cu_device(), pool)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}"))
                    })?;
            }
        }

        Ok(())
    }

    /// Reads a usize attribute from the given CUDA memory pool.
    fn get_mem_pool_attribute(pool: CUmemoryPool, attr: CUmemPool_attribute) -> Result<usize> {
        let mut value: usize = 0;
        unsafe {
            sys::lib()
                .cuMemPoolGetAttribute(pool, attr, &mut value as *mut _ as *mut std::ffi::c_void)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolGetAttribute failed: {e:?}"))
                })?;
        }
        Ok(value)
    }

    /// Returns peak memory used (in bytes) from a given CUDA memory pool.
    pub fn get_peak_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    }

    /// Returns current memory usage (in bytes) from a given CUDA memory pool.
    pub fn get_current_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    }

    /// Retrieves the default CUDA memory pool for a device.
    pub fn get_current_mem_pool(&self) -> Result<CUmemoryPool> {
        if self.pool_handle.read().is_some() {
            let pool_handle = self.pool_handle.read().unwrap();
            let pool: CUmemoryPool = pool_handle as usize as *mut sys::CUmemPoolHandle_st;
            Ok(pool)
        } else {
            candle_core::bail!("Memory pool for graph is not init!")
        }
    }
}

impl<M> CudaGraphModule for CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    fn start_capture(&mut self, bs: usize) -> Result<()> {
        self.capturing = true;
        self.current_bs = Some(bs);
        self.sync_stream()?;
        self.set_capture_mem_pool()?;
        CudaGraph::begin_capture(
            self.device.cu_stream().clone(),
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        Ok(())
    }

    fn end_capture(&mut self, save: bool) -> Result<()> {
        self.capturing = false;
        let bs = self.current_bs.take().unwrap();

        // AUTO_FREE_ON_LAUNCH: graph pool allocs are freed after each launch and
        // re-created on the next. Required for V4's large capture pool to launch
        // successfully. Logits must be D2D-copied into a non-pool buffer during
        // capture so replay reads a stable address (see capture()).
        let graph = CudaGraph::end_capture(
            self.device.cu_stream().clone(),
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u32 as u64,
        )?;
        self.captured_graphs
            .insert(bs, CudaGraphHandle::new(Arc::new(graph)));

        if save {
            self.captured_bs.push(bs);
            self.captured_bs.sort_unstable(); // keep it sorted for binary search
        }
        self.sync_stream()?;
        Ok(())
    }

    fn replay(&self, bs: usize) -> Result<()> {
        if let Some(&next_bs) = self.captured_bs.iter().find(|&&x| x >= bs) {
            if let Some(graph) = self.captured_graphs.get(&next_bs) {
                graph.replay()?;
                self.sync_stream()
            } else {
                candle_core::bail!("No suitable graph is found for batch size {}!", next_bs)
            }
        } else {
            candle_core::bail!("Batch size {} is not captured in graph!", bs)
        }
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        (self.module)(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
        )
    }

    fn report_graph_pool_usage(&self) -> Result<()> {
        let pool = self.get_current_mem_pool()?;
        let peak = Self::get_peak_memory_usage(pool)?;
        let current = Self::get_current_memory_usage(pool)?;
        println!(
            "Default pool usage: {:.2} MB (current), {:.2} MB (peak)",
            current as f64 / 1e6,
            peak as f64 / 1e6
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum CapturePhase {
    CachePrewarm,
    Warmup,
    Capture,
}

impl CapturePhase {
    const ALL: [Self; 3] = [Self::CachePrewarm, Self::Warmup, Self::Capture];

    fn is_cache_prewarm(self) -> bool {
        matches!(self, Self::CachePrewarm)
    }

    fn is_warmup(self) -> bool {
        !matches!(self, Self::Capture)
    }
}

pub struct GraphCaptureVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub mamba_slot_mapping: Tensor,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indptr: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_last_len: Tensor,
    pub outputs: BTreeMap<usize, Tensor>,
}

/// Per-tier (verify_len) buffers for the MTP/DFlash verify graphs: the shapes
/// that track the tier. Size-independent buffers live in `MtpGraphCaptureVars`.
/// The captured graph references these storages, so replay overwrites them in
/// place before launch (shallow clones in the capture metadata share storage).
pub struct MtpGraphSizeVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub slot_mapping: Tensor,
    pub cu_seqlens_q: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_batch_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_positions: Tensor,
    /// Stable (default-pool) logits buffer the captured graph D2D-copies into,
    /// so replay reads a stable address (AUTO_FREE_ON_LAUNCH contract).
    pub output: Tensor,
}

pub struct MtpGraphCaptureVars {
    pub mamba_slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    pub cu_seqlens_k: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indptr: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_last_len: Tensor,
    /// Per-tier buffers, keyed by verify_len (one captured graph per tier).
    pub sizes: BTreeMap<usize, MtpGraphSizeVars>,
}

pub struct GraphCapturer<M: CudaGraphModule> {
    pub model: M,
    pub graph_bs: Vec<usize>,
    pub graph_vars: Option<GraphCaptureVars>,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub block_size: usize,
    pub hidden_size: usize,
    pub device: Option<Device>,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_kv_params: Option<FlashInferKvParams>,
    pub is_mla: bool,
    pub mtp_graph_vars: Option<MtpGraphCaptureVars>,
}

pub fn planned_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
    // Capture every exact batch up to 32. This is especially important for
    // hybrid GDN/Mamba models, whose graph replay requires an exact batch size
    // because the state-slot mapping cannot be padded safely.
    (1..=max_num_seqs.clamp(1, 32)).collect()
}

#[cfg(feature = "flashinfer")]
fn graph_decode_plan(
    device: &Device,
    params: &FlashInferKvParams,
    indptr_host: &[u32],
    last_len_host: &[u32],
    kv_len_arr_host: &[u32],
    batch_size: usize,
    is_mla: bool,
    enable_cuda_graph: bool,
) -> Result<(Option<Vec<i64>>, Option<Vec<i64>>)> {
    if is_mla {
        let plan = attention_rs::mla::mla_decode_plan(
            device,
            params.kv_dtype,
            indptr_host,
            batch_size,
            params.num_qo_heads,
            params.page_size,
            enable_cuda_graph,
        )?;
        Ok((None, Some(plan)))
    } else {
        let plan = attention_rs::flashinfer::decode_plan(
            device,
            params.kv_dtype,
            params.out_dtype,
            indptr_host,
            Some(last_len_host),
            Some(kv_len_arr_host),
            batch_size,
            params.num_qo_heads,
            params.num_kv_heads,
            params.head_dim,
            params.page_size,
            enable_cuda_graph,
        )?;
        Ok((Some(plan), None))
    }
}

impl<M: CudaGraphModule> GraphCapturer<M> {
    pub fn new(
        model: M,
        max_num_seqs: usize,
        max_model_len: usize,
        block_size: usize,
        hidden_size: usize,
        #[cfg(feature = "flashinfer")] flashinfer_kv_params: &Option<FlashInferKvParams>,
        is_mla: bool,
    ) -> Self {
        let graph_bs = planned_graph_capture_batches(max_num_seqs);
        println!("The following batches for capture: {:?}", graph_bs);

        Self {
            model,
            graph_bs,
            graph_vars: None,
            max_num_seqs,
            max_model_len,
            block_size,
            hidden_size,
            device: None,
            #[cfg(feature = "flashinfer")]
            flashinfer_kv_params: flashinfer_kv_params.clone(),
            is_mla,
            mtp_graph_vars: None,
        }
    }

    pub fn capture(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
    ) -> Result<()> {
        let _fp8_domain = attention_rs::fp8_linear::set_fp8_execution_domain(
            attention_rs::fp8_linear::Fp8ExecutionDomain::DecodeGraph,
        );
        let _prefill_guard = set_linear_is_prefill(false);
        self.device = Some(device.clone());
        let max_bs = self.graph_bs[self.graph_bs.len() - 1];
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        let input_ids = Tensor::zeros((max_bs,), DType::U32, device)?;
        let positions = Tensor::zeros((max_bs,), DType::I64, device)?;
        let mamba_slot_mapping = Tensor::from_vec(
            (0..max_bs).map(|i| i as i64).collect::<Vec<_>>(),
            (max_bs,),
            device,
        )?;
        let slot_mapping = Tensor::zeros((max_bs,), DType::I64, device)?;
        let context_lens = Tensor::zeros((max_bs,), DType::U32, device)?;
        let block_tables = Tensor::zeros((max_bs, max_num_blocks), DType::U32, device)?;
        #[cfg(feature = "flashinfer")]
        let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len, last_len_host) = {
            let mut indptr = Vec::with_capacity(max_bs + 1);
            indptr.push(0u32);
            let mut indices = Vec::with_capacity(max_bs * max_num_blocks);
            for _ in 0..max_bs {
                for i in 0..max_num_blocks {
                    indices.push(i as u32);
                }
                indptr.push(indices.len() as u32);
            }
            let last = if self.max_model_len == 0 {
                0u32
            } else {
                ((self.max_model_len - 1) % self.block_size + 1) as u32
            };
            let last_len = vec![last; max_bs];

            (
                Tensor::from_vec(indptr, (max_bs + 1,), device)?,
                Tensor::from_vec(indices, (max_bs * max_num_blocks,), device)?,
                Tensor::from_vec(last_len.clone(), (max_bs,), device)?,
                last_len,
            )
        };
        #[cfg(feature = "flashinfer")]
        let capture_in_warmup = self.flashinfer_kv_params.is_some();
        #[cfg(not(feature = "flashinfer"))]
        let capture_in_warmup = false;

        let mut outputs = BTreeMap::<usize, Tensor>::new();
        let _guard = candle_core::cuda_backend::cuda_param_cache_scope(true);
        for phase in CapturePhase::ALL {
            let iter: Box<dyn Iterator<Item = usize>> = if phase.is_warmup() {
                Box::new(0..self.graph_bs.len())
            } else {
                Box::new(tqdm(0..self.graph_bs.len()).desc(Some("Graph capturing")))
            };
            for i in iter {
                let bs = self.graph_bs[self.graph_bs.len() - i - 1];
                let input_ids_bs = input_ids.narrow(0, 0, bs)?;
                let positions_bs = positions.narrow(0, 0, bs)?;
                #[cfg(feature = "flashinfer")]
                let flashinfer_metadata = if self.flashinfer_kv_params.is_none() {
                    None
                } else {
                    let mut indptr_host = Vec::with_capacity(bs + 1);
                    indptr_host.push(0u32);
                    for i in 0..bs {
                        indptr_host.push(((i + 1) * max_num_blocks) as u32);
                    }

                    let (decode_plan_info, mla_decode_plan_info, kv_len_arr_host) =
                        if let Some(params) = self.flashinfer_kv_params {
                            let mut kv_len_arr_host_bs = Vec::with_capacity(bs);
                            for i in 0..bs {
                                let num_pages = indptr_host[i + 1] - indptr_host[i];
                                if num_pages == 0 {
                                    kv_len_arr_host_bs.push(0);
                                } else {
                                    let full = (num_pages - 1) * params.page_size as u32;
                                    kv_len_arr_host_bs.push(full + last_len_host[i]);
                                }
                            }
                            let (dp, mdp) = graph_decode_plan(
                                device,
                                &params,
                                &indptr_host,
                                &last_len_host[..bs],
                                &kv_len_arr_host_bs,
                                bs,
                                self.is_mla,
                                true, //must be true for graph capture
                            )?;
                            (dp, mdp, Some(kv_len_arr_host_bs))
                        } else {
                            (None, None, None)
                        };

                    Some(attention_rs::FlashInferMetadata {
                        indptr: flashinfer_indptr.narrow(0, 0, bs + 1)?,
                        indptr_host,
                        indices: flashinfer_indices.narrow(0, 0, bs * max_num_blocks)?,
                        last_len: flashinfer_last_len.narrow(0, 0, bs)?,
                        last_len_host: Some(last_len_host[..bs].to_vec()),
                        kv_len_arr_host,
                        total_num_rows: None,
                        batch_indices: None,
                        positions: None,
                        use_cuda_graph: true,
                        decode_plan_info,
                        prefill_plan_info: None,
                        mla_decode_plan_info,
                        mla_prefill_plan_info: None,
                    })
                };
                #[cfg(not(feature = "flashinfer"))]
                let flashinfer_metadata = None;

                let input_metadata = InputMetadata {
                    is_prefill: false,
                    is_mla: self.is_mla,
                    sequence_ids: None,
                    mamba_slot_mapping: Some(mamba_slot_mapping.narrow(0, 0, bs)?),
                    slot_mapping: slot_mapping.narrow(0, 0, bs)?,
                    block_tables: Some(block_tables.narrow(0, 0, bs)?),
                    block_tables_host: None,
                    context_lens_host: None,
                    context_lens: Some(context_lens.narrow(0, 0, bs)?),
                    cu_seqlens_q: None,
                    cu_seqlens_k: None,
                    max_seqlen_q: 0,
                    max_seqlen_k: 0,
                    max_context_len: self.max_model_len,
                    seqlens: None,
                    flashinfer_metadata,
                    is_mtp_verify: false,
                };

                let should_capture =
                    !phase.is_cache_prewarm() && (!phase.is_warmup() || capture_in_warmup);
                if should_capture {
                    self.model.start_capture(bs)?;
                }
                if phase.is_warmup() {
                    let _ = self.model.forward(
                        &input_ids_bs,
                        &positions_bs,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    #[cfg(feature = "cuda")]
                    if !should_capture {
                        device.synchronize()?;
                    }
                } else {
                    let out = self.model.forward(
                        &input_ids_bs,
                        &positions_bs,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    outputs.insert(bs, out);
                }
                if should_capture {
                    self.model.end_capture(!phase.is_warmup())?;
                }
            }
            #[cfg(feature = "cuda")]
            device.synchronize()?;
        }
        let _ = self.model.report_graph_pool_usage();
        crate::log_warn!("Captured batches {:?}", outputs.keys());
        self.graph_vars = Some(GraphCaptureVars {
            input_ids,
            positions,
            mamba_slot_mapping,
            slot_mapping,
            context_lens,
            block_tables,
            #[cfg(feature = "flashinfer")]
            flashinfer_indptr,
            #[cfg(feature = "flashinfer")]
            flashinfer_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_last_len,
            outputs,
        });

        Ok(())
    }

    pub fn is_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .keys()
                .find(|&&x| x >= batch)
                .is_some()
    }

    pub fn is_exact_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .contains_key(&batch)
    }

    pub fn replay(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let _fp8_domain = attention_rs::fp8_linear::set_fp8_execution_domain(
            attention_rs::fp8_linear::Fp8ExecutionDomain::DecodeGraph,
        );
        if input_metadata.is_prefill {
            candle_core::bail!("Graph replay is not used for prefill!")
        }
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
        let input_batch = input_ids.dim(0)?;
        let require_exact_batch = input_metadata.mamba_slot_mapping.is_some();
        if let Some(graph_vars) = &self.graph_vars {
            let selected_batch = if require_exact_batch {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x == input_batch)
                    .copied()
            } else {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x >= input_batch)
                    .copied()
            };
            if let Some(batch) = selected_batch {
                graph_vars.input_ids.zero_()?;
                graph_vars.input_ids.copy_(&input_ids, 0)?;
                graph_vars.positions.zero_()?;
                graph_vars.positions.copy_(&positions, 0)?;

                if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
                    graph_vars.mamba_slot_mapping.zero_()?;
                    graph_vars.mamba_slot_mapping.copy_(&ms_mapping, 0)?;
                } else {
                    graph_vars.mamba_slot_mapping.zero_()?;
                }

                let s_mapping = input_metadata.slot_mapping.as_ref();
                graph_vars.slot_mapping.zero_()?;
                graph_vars.slot_mapping.copy_(&s_mapping, 0)?;

                let c_lens = input_metadata.context_lens.as_ref().unwrap();
                graph_vars.context_lens.zero_()?;
                graph_vars.context_lens.copy_(&c_lens, 0)?;

                let b_tables = input_metadata.block_tables.as_ref().unwrap();
                let padded_table = b_tables
                    .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                    .contiguous()?;

                graph_vars.block_tables.zero_()?;
                graph_vars.block_tables.copy_(&padded_table, 0)?;

                #[cfg(feature = "flashinfer")]
                if let Some(fm) = &input_metadata.flashinfer_metadata {
                    let mut indptr_host = fm.indptr_host.clone();
                    if input_batch == batch {
                        graph_vars.flashinfer_indptr.zero_()?;
                        graph_vars.flashinfer_indptr.copy_(&fm.indptr, 0)?;
                    } else {
                        // Pad indptr to the captured batch size so graph replay sees valid lengths.
                        let last = *indptr_host.last().unwrap_or(&0);
                        for _ in (input_batch + 1)..=batch {
                            indptr_host.push(last);
                        }

                        let indptr_padded = Tensor::from_vec(
                            indptr_host.clone(),
                            (batch + 1,),
                            graph_vars.input_ids.device(),
                        )?;
                        graph_vars.flashinfer_indptr.copy_(&indptr_padded, 0)?;
                    }

                    graph_vars.flashinfer_last_len.zero_()?;
                    graph_vars.flashinfer_last_len.copy_(&fm.last_len, 0)?;

                    graph_vars.flashinfer_indices.zero_()?;
                    graph_vars.flashinfer_indices.copy_(&fm.indices, 0)?;

                    if let Some(params) = self.flashinfer_kv_params {
                        let dev = self
                            .device
                            .as_ref()
                            .ok_or_else(|| candle_core::Error::msg("graph device is missing"))?;
                        let last_len_host = fm.last_len_host.as_deref().ok_or_else(|| {
                            candle_core::Error::msg("graph replay requires last_len_host")
                        })?;
                        let kv_len_arr_host = fm.kv_len_arr_host.as_deref().ok_or_else(|| {
                            candle_core::Error::msg("graph replay requires kv_len_arr_host")
                        })?;
                        let _ = graph_decode_plan(
                            dev,
                            &params,
                            &indptr_host,
                            last_len_host,
                            kv_len_arr_host,
                            batch,
                            self.is_mla,
                            fm.use_cuda_graph,
                        )?;
                    }
                }

                let result = self.model.replay(batch);
                if result.is_err() {
                    eprintln!("Error when replaying graph {:?}", result);
                }

                graph_vars.outputs[&batch]
                    .narrow(0, 0, input_batch)?
                    .contiguous()
            } else {
                candle_core::bail!("Input batch {} is not captured!", input_batch)
            }
        } else {
            candle_core::bail!("Graph is not captured!")
        }
    }

    pub fn capture_mtp(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        verify_lens: &[usize],
        label: &'static str,
    ) -> Result<()> {
        if verify_lens.is_empty() {
            return Ok(());
        }

        let _fp8_domain = attention_rs::fp8_linear::set_fp8_execution_domain(
            attention_rs::fp8_linear::Fp8ExecutionDomain::MtpGraph,
        );
        let _prefill_guard = set_linear_is_prefill(true);
        self.device = Some(device.clone());
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        // Size-independent buffers shared by all tiers. Capture must use
        // in-bounds, decode-consistent page metadata. Zeroed cu_seqlens /
        // last_len=max_model_len previously caused FlashInfer/GDN OOB during
        // capture; multirank NCCL sync surfaces that as
        // CUDA_ERROR_ILLEGAL_ADDRESS on InputMetadata drop.
        let mamba_slot_mapping = Tensor::zeros((1,), DType::I64, device)?;
        let context_lens = Tensor::from_vec(vec![self.max_model_len as u32], (1,), device)?;
        let block_tables = Tensor::zeros((1, max_num_blocks), DType::U32, device)?;
        let cu_seqlens_k = Tensor::from_vec(vec![0u32, self.max_model_len as u32], (2,), device)?;

        #[cfg(feature = "flashinfer")]
        let last_page_len = if self.max_model_len == 0 {
            0u32
        } else {
            ((self.max_model_len - 1) % self.block_size + 1) as u32
        };

        #[cfg(feature = "flashinfer")]
        let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len) = {
            let indices: Vec<u32> = (0..max_num_blocks as u32).collect();
            (
                Tensor::from_vec(vec![0u32, max_num_blocks as u32], (2,), device)?,
                Tensor::from_vec(indices, (max_num_blocks,), device)?,
                Tensor::from_vec(vec![last_page_len], (1,), device)?,
            )
        };

        #[cfg(feature = "flashinfer")]
        let use_flashinfer = self.flashinfer_kv_params.is_some();
        #[cfg(not(feature = "flashinfer"))]
        let use_flashinfer = false;

        let capture_in_warmup = use_flashinfer;

        let mut sizes: BTreeMap<usize, MtpGraphSizeVars> = BTreeMap::new();
        let _guard = candle_core::cuda_backend::cuda_param_cache_scope(true);

        // Largest tier first: the shared capture pool peaks at the biggest
        // graph; smaller tiers reuse it (AUTO_FREE_ON_LAUNCH).
        let mut lens: Vec<usize> = verify_lens.to_vec();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        for verify_len in lens {
            // Per-tier buffers (shapes track the tier).
            let input_ids = Tensor::zeros((verify_len,), DType::U32, device)?;
            let positions = Tensor::zeros((verify_len,), DType::I64, device)?;
            let slot_mapping = Tensor::zeros((verify_len,), DType::I64, device)?;
            let cu_seqlens_q = Tensor::from_vec(vec![0u32, verify_len as u32], (2,), device)?;

            #[cfg(feature = "flashinfer")]
            let flashinfer_batch_indices = Tensor::zeros((verify_len,), DType::U32, device)?;
            #[cfg(feature = "flashinfer")]
            let flashinfer_positions = {
                // Mirror runtime MTP verify append positions at max context:
                // [max_model_len - verify_len, ..., max_model_len - 1].
                let start = self.max_model_len.saturating_sub(verify_len) as u32;
                let pos: Vec<u32> = (0..verify_len as u32).map(|i| start + i).collect();
                Tensor::from_vec(pos, (verify_len,), device)?
            };

            #[cfg(feature = "flashinfer")]
            let flashinfer_metadata = if let Some(params) = self.flashinfer_kv_params {
                let indptr_host = vec![0u32, max_num_blocks as u32];
                let kv_len_arr_host = vec![self.max_model_len as u32];
                let q_cu_seqlens_host = vec![0u32, verify_len as u32];
                let last_len_host = vec![last_page_len];

                let prefill_plan_info = attention_rs::flashinfer::graph_prefill_plan(
                    device,
                    &q_cu_seqlens_host,
                    &indptr_host,
                    &kv_len_arr_host,
                    verify_len as u32,
                    1,
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                )?;

                Some(attention_rs::FlashInferMetadata {
                    indptr: flashinfer_indptr.clone(),
                    indptr_host,
                    indices: flashinfer_indices.clone(),
                    last_len: flashinfer_last_len.clone(),
                    last_len_host: Some(last_len_host),
                    kv_len_arr_host: Some(kv_len_arr_host),
                    total_num_rows: Some(verify_len as u32),
                    batch_indices: Some(flashinfer_batch_indices.clone()),
                    positions: Some(flashinfer_positions.clone()),
                    use_cuda_graph: true,
                    decode_plan_info: None,
                    prefill_plan_info: Some(prefill_plan_info),
                    mla_decode_plan_info: None,
                    mla_prefill_plan_info: None,
                })
            } else {
                None
            };
            #[cfg(not(feature = "flashinfer"))]
            let flashinfer_metadata = None;

            let input_metadata = InputMetadata {
                is_prefill: true,
                is_mla: self.is_mla,
                sequence_ids: Some(vec![0]),
                mamba_slot_mapping: Some(mamba_slot_mapping.clone()),
                slot_mapping: slot_mapping.clone(),
                block_tables: Some(block_tables.clone()),
                block_tables_host: None,
                context_lens_host: None,
                context_lens: Some(context_lens.clone()),
                cu_seqlens_q: Some(cu_seqlens_q.clone()),
                cu_seqlens_k: Some(cu_seqlens_k.clone()),
                max_seqlen_q: verify_len,
                max_seqlen_k: self.max_model_len,
                max_context_len: self.max_model_len,
                seqlens: None,
                flashinfer_metadata,
                is_mtp_verify: true,
            };

            let mut stable_logits: Option<Tensor> = None;
            for phase in CapturePhase::ALL {
                let should_capture =
                    !phase.is_cache_prewarm() && (!phase.is_warmup() || capture_in_warmup);
                if should_capture {
                    self.model.start_capture(verify_len)?;
                }
                if phase.is_warmup() {
                    let out = self.model.forward(
                        &input_ids,
                        &positions,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    // Allocate the stable logits buffer on the default pool during
                    // uncaptured cache-prewarm so AUTO_FREE_ON_LAUNCH does not
                    // invalidate the address read after graph replay.
                    if phase.is_cache_prewarm() {
                        stable_logits = Some(Tensor::zeros(out.shape(), out.dtype(), device)?);
                    }
                    #[cfg(feature = "cuda")]
                    if !should_capture {
                        device.synchronize()?;
                    }
                } else {
                    let out = self.model.forward(
                        &input_ids,
                        &positions,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    let stable = stable_logits.as_ref().ok_or_else(|| {
                        candle_core::Error::msg(
                            "MTP graph capture missing stable logits buffer (cache prewarm failed)",
                        )
                    })?;
                    // D2D into non-pool storage; this copy is part of the captured graph.
                    stable.copy_(&out, 0)?;
                }
                if should_capture {
                    self.model.end_capture(!phase.is_warmup())?;
                }
                #[cfg(feature = "cuda")]
                device.synchronize()?;
            }

            // Move the per-tier buffers into the map once, after all phases are done
            // (the loop only borrows them, so this is a true move, not a clone).
            let stable = stable_logits.take().ok_or_else(|| {
                candle_core::Error::msg(
                    "MTP graph capture missing stable logits buffer (cache prewarm failed)",
                )
            })?;
            sizes.insert(
                verify_len,
                MtpGraphSizeVars {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    #[cfg(feature = "flashinfer")]
                    flashinfer_batch_indices,
                    #[cfg(feature = "flashinfer")]
                    flashinfer_positions,
                    output: stable,
                },
            );

            crate::log_warn!(
                "Captured {} verify graph len={} (flashinfer={})",
                label,
                verify_len,
                use_flashinfer
            );
        }

        self.mtp_graph_vars = Some(MtpGraphCaptureVars {
            mamba_slot_mapping,
            context_lens,
            block_tables,
            cu_seqlens_k,
            #[cfg(feature = "flashinfer")]
            flashinfer_indptr,
            #[cfg(feature = "flashinfer")]
            flashinfer_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_last_len,
            sizes,
        });
        Ok(())
    }

    pub fn is_mtp_captured(&self, verify_len: usize) -> bool {
        self.mtp_graph_vars
            .as_ref()
            .map_or(false, |v| v.sizes.contains_key(&verify_len))
    }

    pub fn replay_mtp(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let _fp8_domain = attention_rs::fp8_linear::set_fp8_execution_domain(
            attention_rs::fp8_linear::Fp8ExecutionDomain::MtpGraph,
        );
        let verify_len = input_ids.dim(0)?;
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        let mtp_vars = self
            .mtp_graph_vars
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("MTP graphs not captured"))?;

        let size_vars = mtp_vars.sizes.get(&verify_len).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "MTP verify graph for len {verify_len} is not captured!"
            ))
        })?;

        size_vars.input_ids.zero_()?;
        size_vars.input_ids.copy_(input_ids, 0)?;
        size_vars.positions.zero_()?;
        size_vars.positions.copy_(positions, 0)?;

        if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
            mtp_vars.mamba_slot_mapping.zero_()?;
            mtp_vars.mamba_slot_mapping.copy_(ms_mapping, 0)?;
        }

        size_vars.slot_mapping.zero_()?;
        size_vars
            .slot_mapping
            .copy_(&input_metadata.slot_mapping, 0)?;

        if let Some(c_lens) = input_metadata.context_lens.as_ref() {
            mtp_vars.context_lens.zero_()?;
            mtp_vars.context_lens.copy_(c_lens, 0)?;
        }

        if let Some(b_tables) = input_metadata.block_tables.as_ref() {
            let padded_table = b_tables
                .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                .contiguous()?;
            mtp_vars.block_tables.zero_()?;
            mtp_vars.block_tables.copy_(&padded_table, 0)?;
        }

        if let Some(cu_q) = input_metadata.cu_seqlens_q.as_ref() {
            size_vars.cu_seqlens_q.copy_(cu_q, 0)?;
        }
        if let Some(cu_k) = input_metadata.cu_seqlens_k.as_ref() {
            mtp_vars.cu_seqlens_k.copy_(cu_k, 0)?;
        }

        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_ref() {
            mtp_vars.flashinfer_indptr.zero_()?;
            mtp_vars.flashinfer_indptr.copy_(&fm.indptr, 0)?;
            mtp_vars.flashinfer_indices.zero_()?;
            mtp_vars.flashinfer_indices.copy_(&fm.indices, 0)?;
            mtp_vars.flashinfer_last_len.zero_()?;
            mtp_vars.flashinfer_last_len.copy_(&fm.last_len, 0)?;
            let batch_indices = fm.batch_indices.as_ref().ok_or_else(|| {
                candle_core::Error::msg("mtp replay requires flashinfer batch_indices")
            })?;
            let positions = fm.positions.as_ref().ok_or_else(|| {
                candle_core::Error::msg("mtp replay requires flashinfer positions")
            })?;
            size_vars.flashinfer_batch_indices.zero_()?;
            size_vars.flashinfer_batch_indices.copy_(batch_indices, 0)?;
            size_vars.flashinfer_positions.zero_()?;
            size_vars.flashinfer_positions.copy_(positions, 0)?;

            if let Some(params) = self.flashinfer_kv_params {
                let dev = self
                    .device
                    .as_ref()
                    .ok_or_else(|| candle_core::Error::msg("graph device is missing"))?;
                let kv_len_arr_host = fm.kv_len_arr_host.as_deref().ok_or_else(|| {
                    candle_core::Error::msg("mtp replay requires kv_len_arr_host")
                })?;
                let q_cu_seqlens_host = vec![0u32, verify_len as u32];
                let _ = attention_rs::flashinfer::graph_prefill_plan(
                    dev,
                    &q_cu_seqlens_host,
                    &fm.indptr_host,
                    kv_len_arr_host,
                    verify_len as u32,
                    1,
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                )?;
            }
        }

        self.model.replay(verify_len)?;

        size_vars.output.contiguous()
    }
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

pub type ModelFn = dyn for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>
    + Send
    + Sync;

pub type CudaGraphFn = Box<
    dyn for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
            bool,
        ) -> Result<Tensor>
        + Send
        + Sync,
>;

/// CUDA graph for the DFlash draft transformer (opt-in, `XINFER_SPEC_GRAPH`).
/// Captures the draft forward `(target_hidden, noise_embedding, positions) -> draft_hidden` at
/// the full context cap. Inputs are preallocated default-pool buffers; the output is D2D-copied
/// into a stable default-pool buffer (the `AUTO_FREE_ON_LAUNCH` contract), so replay reads a
/// stable address. The lm_head + selection run eagerly on the replayed output.
pub struct DFlashDraftGraph {
    graph: Option<Arc<CudaGraph>>,
    target_hidden: Tensor,
    noise_embedding: Tensor,
    positions: Tensor,
    out: Tensor,
    device: Device,
}

impl DFlashDraftGraph {
    pub fn new(cap: usize, block: usize, hidden: usize, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            graph: None,
            target_hidden: Tensor::zeros((cap, hidden), dtype, device)?,
            noise_embedding: Tensor::zeros((block, hidden), dtype, device)?,
            positions: Tensor::zeros((cap + block,), DType::I64, device)?,
            out: Tensor::zeros((cap + block, hidden), dtype, device)?,
            device: device.clone(),
        })
    }

    /// Capture the draft forward (`draft_fwd` runs the draft transformer) into a CUDA graph.
    pub fn capture<F: FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>>(
        &mut self,
        draft_fwd: F,
    ) -> Result<()> {
        let stream = self.device.as_cuda_device()?.cu_stream().clone();
        CudaGraph::begin_capture(
            stream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        {
            let out = draft_fwd(&self.target_hidden, &self.noise_embedding, &self.positions)?;
            self.out.copy_(&out, 0)?;
            // `out` is a graph-pool allocation; dropping it while capture is still
            // active makes cudarc skip the cuMemFree (the graph pool owns it and
            // AUTO_FREE_ON_LAUNCH frees it at replay). Dropping it after end_capture
            // faults with CUDA_ERROR_INVALID_VALUE.
        }
        let graph = CudaGraph::end_capture(
            stream,
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u32 as u64,
        )?;
        self.graph = Some(Arc::new(graph));
        Ok(())
    }

    /// Replay the captured draft forward with the real inputs; returns the draft hidden
    /// (`[cap + block, hidden]`; the caller narrows the trailing `block` rows for the lm_head).
    pub fn replay(
        &self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("DFlash draft graph not captured".into()))?;
        self.target_hidden.copy_(target_hidden, 0)?;
        self.noise_embedding.copy_(noise_embedding, 0)?;
        self.positions.copy_(positions, 0)?;
        graph.launch()?;
        Ok(self.out.clone())
    }

    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// The context cap the graph was captured for (the draft context window size).
    pub fn cap(&self) -> usize {
        self.target_hidden.dim(0).unwrap_or(0)
    }

    /// The draft block width (anchor + MASK rows) the graph was captured for.
    pub fn block(&self) -> usize {
        self.noise_embedding.dim(0).unwrap_or(0)
    }

    /// The draft model hidden size.
    pub fn hidden(&self) -> usize {
        self.target_hidden.dim(1).unwrap_or(0)
    }
}
