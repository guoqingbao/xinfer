use crate::models::layers::linear::{
    linear_b_x as linear_b, linear_no_bias_x as linear, LinearX as Linear, LnFp8, LnMxfp4, LnNvfp4,
    QLinear,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
use crate::utils::gguf_helper::restore_qwen35_qkv_weight;
#[cfg(feature = "nccl")]
pub use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::CustomOp1;
use candle_core::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor};
use candle_nn::var_builder::Shard;
use either::Either;
#[cfg(not(feature = "nccl"))]
pub struct Comm {}
#[cfg(not(feature = "nccl"))]
impl Comm {
    //dummy Comm
    pub fn default() -> Self {
        Self {}
    }
    pub fn dim(&self) -> usize {
        0
    }

    pub fn rank(&self) -> usize {
        0
    }
    pub fn world_size(&self) -> usize {
        1
    }
}

#[cfg(not(feature = "nccl"))]
#[derive(Debug, Clone, Copy)]
pub struct Id {
    pub internal: [::core::ffi::c_char; 128usize],
}

#[cfg(not(feature = "nccl"))]
impl Id {
    pub fn as_bytes(&self) -> &[u8] {
        // Safe reinterpretation of `c_char` as `u8`
        unsafe { std::slice::from_raw_parts(self.internal.as_ptr() as *const u8, 128) }
    }
}

pub use std::rc::Rc;

pub struct ReplicatedLinear {
    linear: Linear,
}

pub struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

pub fn tensor_parallel_chunk(
    x: &Tensor,
    dim: usize,
    rank: usize,
    world_size: usize,
    name: &str,
) -> Result<Tensor> {
    if world_size <= 1 {
        return Ok(x.clone());
    }
    let dim_size = x.dim(dim)?;
    if dim_size % world_size != 0 {
        candle_core::bail!(
            "tensor-parallel chunk for {} dim {} size {} is not divisible by world_size {}",
            name,
            dim,
            dim_size,
            world_size
        );
    }
    let chunk_size = dim_size / world_size;
    x.narrow(dim, rank * chunk_size, chunk_size)?.contiguous()
}

pub fn load_restored_gguf_column_linear(
    vb: &VarBuilderX,
    in_dim: usize,
    out_dim: usize,
    name: &str,
    comm: Rc<Comm>,
    dtype: DType,
    restore_weight: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<TensorParallelColumnLinear> {
    let qvb = match &vb.pp(name).0 {
        Either::Right(vb) => vb.clone(),
        _ => candle_core::bail!("expected GGUF varbuilder for {}", name),
    };
    let ws = qvb.get((out_dim, in_dim), "weight")?;
    let mut wdtype = ws.dtype();
    let weight = restore_weight(ws.dequantize_f16(qvb.device())?)?;
    let local_weight = tensor_parallel_chunk(&weight, 0, comm.rank(), comm.world_size(), name)?;
    if local_weight.dim(0)? % wdtype.block_size() != 0 {
        wdtype = GgmlDType::Q8_0;
    }
    let qtensor = QTensor::quantize(&local_weight, wdtype)?;
    let qlinear = QLinear::from_qparts_x(qtensor, None, dtype)?;
    Ok(TensorParallelColumnLinear::new(Linear::QLinear(qlinear)))
}

pub struct MergedParallelColumnLinear {
    linears: Vec<TensorParallelColumnLinear>,
    biases: Vec<Option<Tensor>>,
    output_splits: Option<Vec<usize>>,
}

impl MergedParallelColumnLinear {
    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self {
            linears,
            biases: Vec::new(),
            output_splits: None,
        }
    }

    pub fn from_packed_local(
        weight: Tensor,
        bias: Option<Tensor>,
        output_splits: Vec<usize>,
        quant: &Option<String>,
    ) -> Result<Self> {
        let linear = Linear::new(weight, None, quant)?;
        let tp_linear = TensorParallelColumnLinear { linear };
        Ok(Self {
            linears: vec![tp_linear],
            biases: vec![bias],
            output_splits: Some(output_splits),
        })
    }

    pub fn from_packed_local_qlinear(
        qlinear: Linear,
        bias: Option<Tensor>,
        output_splits: Vec<usize>,
    ) -> Result<Self> {
        let tp_linear = TensorParallelColumnLinear { linear: qlinear };
        Ok(Self {
            linears: vec![tp_linear],
            biases: vec![bias],
            output_splits: Some(output_splits),
        })
    }

    pub fn from_packed_local_fp8(
        weight: Tensor,
        weight_scale: Tensor,
        bias: Option<Tensor>,
        weight_block_size: Vec<usize>,
        sm_version: usize,
        output_splits: Vec<usize>,
    ) -> Result<Self> {
        #[cfg(feature = "cutlass")]
        let weight_scale_cutlass = if sm_version >= 100 {
            Some(weight_scale.t()?)
        } else if sm_version >= 90 {
            Some(weight_scale.t()?.contiguous()?)
        } else {
            None
        };

        #[cfg(not(feature = "cutlass"))]
        let weight_scale_cutlass = None;

        let linear = Linear::LnFp8(LnFp8 {
            weight,
            weight_scale,
            weight_scale_cutlass,
            bias: None,
            weight_block_size,
            sm_version,
        });
        let tp_linear = TensorParallelColumnLinear { linear };
        Ok(Self {
            linears: vec![tp_linear],
            biases: vec![bias],
            output_splits: Some(output_splits),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        if let Some(output_splits) = &self.output_splits {
            if self.linears.len() != 1 {
                candle_core::bail!(
                    "MergedParallelColumnLinear expected exactly 1 linear for split outputs, got {}",
                    self.linears.len()
                );
            }
            let mut ys = self.linears[0].forward(x)?;
            if let Some(Some(bias)) = self.biases.first() {
                ys = ys.broadcast_add(bias)?;
            }
            let split_dim = ys.dims().len().saturating_sub(1);
            let total_dim = ys.dim(split_dim)?;
            let expected_dim: usize = output_splits.iter().sum();
            if total_dim != expected_dim {
                candle_core::bail!(
                    "MergedParallelColumnLinear split mismatch: output dim {} vs expected {}",
                    total_dim,
                    expected_dim
                );
            }
            let mut outputs = Vec::with_capacity(output_splits.len());
            let mut start = 0usize;
            for split_size in output_splits {
                outputs.push(ys.narrow(split_dim, start, *split_size)?.contiguous()?);
                start += *split_size;
            }
            return Ok(outputs);
        }

        let mut xss = Vec::<Tensor>::new();
        for i in 0..self.linears.len() {
            let mut xs = self.linears[i].forward(x)?;
            if self.biases.len() > 0 && i < self.biases.len() {
                if let Some(bias) = &self.biases[i] {
                    xs = xs.broadcast_add(bias)?;
                }
            }
            xss.push(xs);
        }
        Ok(xss)
    }
}

pub fn load_restored_gguf_merged_qkv_linear(
    vb: &VarBuilderX,
    hidden_size: usize,
    key_dim_global: usize,
    value_dim_global: usize,
    num_k_heads_global: usize,
    num_v_heads_global: usize,
    head_v_dim: usize,
    name: &str,
    comm: Rc<Comm>,
    dtype: DType,
) -> Result<MergedParallelColumnLinear> {
    let qvb = match &vb.pp(name).0 {
        Either::Right(vb) => vb.clone(),
        _ => candle_core::bail!("expected GGUF varbuilder for {}", name),
    };
    let out_dim = key_dim_global * 2 + value_dim_global;
    let ws = qvb.get((out_dim, hidden_size), "weight")?;
    let mut wdtype = ws.dtype();
    let weight = restore_qwen35_qkv_weight(
        &ws.dequantize_f16(qvb.device())?,
        key_dim_global,
        num_k_heads_global,
        num_v_heads_global,
        head_v_dim,
    )?;

    let chunk_sizes = [key_dim_global, key_dim_global, value_dim_global];
    let mut local_chunks = Vec::with_capacity(chunk_sizes.len());
    let mut output_splits = Vec::with_capacity(chunk_sizes.len());
    let mut offset = 0usize;
    for (chunk_idx, &chunk_size) in chunk_sizes.iter().enumerate() {
        let chunk = weight.narrow(0, offset, chunk_size)?;
        let local_chunk = tensor_parallel_chunk(
            &chunk,
            0,
            comm.rank(),
            comm.world_size(),
            &format!("{name}[{chunk_idx}]"),
        )?;
        if local_chunk.dim(0)? % wdtype.block_size() != 0 {
            wdtype = GgmlDType::Q8_0;
        }
        local_chunks.push(local_chunk);
        output_splits.push(local_chunks.last().unwrap().dim(0)?);
        offset += chunk_size;
    }

    let local_chunk_refs = local_chunks.iter().collect::<Vec<_>>();
    let local_weight = Tensor::cat(&local_chunk_refs, 0)?;
    let qtensor = QTensor::quantize(&local_weight, wdtype)?;
    let qlinear = QLinear::from_qparts_x(qtensor, None, dtype)?;
    MergedParallelColumnLinear::from_packed_local_qlinear(
        Linear::QLinear(qlinear),
        None,
        output_splits,
    )
}

#[allow(dead_code)]
pub struct TensorParallelRowLinear {
    linear: Linear,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    bias: Option<Tensor>,
    dtype: DType,
}

#[allow(dead_code)]
pub struct AllReduce {
    comm: Rc<Comm>,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl AllReduce {
    pub fn new(comm: Rc<Comm>) -> Self {
        Self { comm: comm.clone() }
    }
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(self)
    }
}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
        use candle_core::cuda_backend::cudarc::nccl::safe::ReduceOp;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let start_offset = l.start_offset();

        let dst = match s.dtype() {
            DType::BF16 => {
                let full_slice = s.as_cuda_slice::<bf16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce BF16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce F16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    #[allow(unused_variables)]
    pub fn new(linear: Linear, comm: Rc<Comm>, dtype: DType) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias: None,
            dtype,
        }
    }

    #[allow(unused_variables)]
    pub fn new_with_bias(
        linear: Linear,
        bias: Option<Tensor>,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias,
            dtype,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            if xs.dtype() != self.dtype {
                //only bf16/fp16 supported in all reduce
                let xs_reduce = xs.to_dtype(self.dtype)?.apply_op1_no_bwd(all_reduce)?;
                xs = xs_reduce.to_dtype(xs.dtype())?
            } else {
                xs = xs.apply_op1_no_bwd(all_reduce)?;
            }
        }
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(&bias)?;
        }
        Ok(xs)
    }
}

pub fn load_restored_gguf_row_linear(
    vb: &VarBuilderX,
    in_dim: usize,
    out_dim: usize,
    name: &str,
    comm: Rc<Comm>,
    dtype: DType,
    restore_weight: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<TensorParallelRowLinear> {
    let qvb = match &vb.pp(name).0 {
        Either::Right(vb) => vb.clone(),
        _ => candle_core::bail!("expected GGUF varbuilder for {}", name),
    };
    let ws = qvb.get((out_dim, in_dim), "weight")?;
    let mut wdtype = ws.dtype();
    let weight = restore_weight(ws.dequantize_f16(qvb.device())?)?;
    let local_weight = tensor_parallel_chunk(&weight, 1, comm.rank(), comm.world_size(), name)?;
    if local_weight.dim(1)? % wdtype.block_size() != 0 {
        wdtype = GgmlDType::Q8_0;
    }
    let qtensor = QTensor::quantize(&local_weight, wdtype)?;
    let qlinear = QLinear::from_qparts_x(qtensor, None, dtype)?;
    Ok(TensorParallelRowLinear::new(
        Linear::QLinear(qlinear),
        comm,
        dtype,
    ))
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

/// Determine local KV heads and shard mapping for tensor-parallel attention.
///
/// In replicated KV mode (`total_num_kv_heads < world_size`), query heads are sharded in
/// contiguous rank ranges, so KV head replication must follow the same contiguous grouping.
pub fn kv_head_shard(
    total_num_kv_heads: usize,
    rank: usize,
    world_size: usize,
) -> Result<(usize, Shard)> {
    if total_num_kv_heads == 0 {
        candle_core::bail!("num_kv_heads must be > 0");
    }
    if world_size == 0 {
        candle_core::bail!("tensor parallel world_size must be > 0");
    }
    if rank >= world_size {
        candle_core::bail!(
            "rank out of bounds for tensor parallel group (rank={}, world_size={})",
            rank,
            world_size
        );
    }

    if total_num_kv_heads >= world_size {
        if total_num_kv_heads % world_size != 0 {
            candle_core::bail!(
                "kv heads must be divisible by tensor parallel world_size when partitioned (num_kv_heads={}, world_size={})",
                total_num_kv_heads,
                world_size
            );
        }
        Ok((total_num_kv_heads / world_size, shard(0, rank, world_size)))
    } else {
        if world_size % total_num_kv_heads != 0 {
            candle_core::bail!(
                "tensor parallel world_size must be divisible by kv heads when kv heads are replicated (num_kv_heads={}, world_size={})",
                total_num_kv_heads,
                world_size
            );
        }
        let ranks_per_kv_head = world_size / total_num_kv_heads;
        let kv_head_rank = rank / ranks_per_kv_head;
        Ok((1, shard(0, kv_head_rank, total_num_kv_heads)))
    }
}

impl TensorParallelColumnLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear_b(
            in_dim,
            out_dim,
            bias,
            vb,
            shard(0, comm.rank(), comm.world_size()),
            quant_cfg,
            quant,
            dtype,
        )?;
        Ok(Self { linear })
    }

    pub fn load_with_shard(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        shard: Shard,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear_b(in_dim, out_dim, bias, vb, shard, quant_cfg, quant, dtype)?;
        Ok(Self { linear })
    }
}

impl MergedParallelColumnLinear {
    pub fn load_merged_with_hints(
        in_dim: usize,
        out_dim: usize,
        chunk_dim: usize,
        chunks: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        if quant.is_some() {
            candle_core::bail!("Merged quantized weight is not supported at the moment!");
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        for chunk_idx in 0..chunks {
            let linear = linear(
                in_dim,
                out_dim,
                vb.clone(),
                shard(
                    chunk_dim,
                    chunk_idx * comm.world_size() + comm.rank(),
                    comm.world_size() * chunks,
                ),
                quant_cfg,
                quant,
                dtype,
            )?;

            let ln = TensorParallelColumnLinear { linear };
            vec_linear.push(ln);
        }
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; chunks],
            output_splits: None,
        })
    }

    pub fn load_merged_chunks(
        in_dim: usize,
        out_dim: usize,
        chunk_dim: usize,
        chunks: Vec<usize>,
        chunk_shards: Option<Vec<Shard>>,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let is_fp8_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "fp8"
        } else {
            false
        };

        let is_nvfp4_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "nvfp4"
        } else {
            false
        };
        let is_mxfp4_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "mxfp4"
        } else {
            false
        };

        if vb.is_qvar_builder() {
            candle_core::bail!(
                "Merged quantized weight is not supported for GGUF varbuilder at the moment!"
            );
        }
        if quant_cfg.is_some() && !is_fp8_quant && !is_nvfp4_quant && !is_mxfp4_quant {
            candle_core::bail!(
                "Merged quantized weight is not supported at the moment, using ISQ instead!"
            );
        }
        if let Some(chunk_shards) = &chunk_shards {
            if chunk_shards.len() != chunks.len() {
                candle_core::bail!(
                    "chunk_shards length mismatch: expected {}, got {}",
                    chunks.len(),
                    chunk_shards.len()
                );
            }
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        let mut output_splits: Option<Vec<usize>> = None;
        use crate::models::layers::linear::{LinearX, LnFp8, QLinear};
        match vb.0 {
            Either::Left(v) => {
                if is_fp8_quant {
                    if chunk_dim != 0 {
                        candle_core::bail!(
                            "FP8 merged chunk loading currently supports chunk_dim=0 only, got {}",
                            chunk_dim
                        );
                    }
                    let quant_cfg = quant_cfg.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "FP8 merged chunk loading requires quantization config".to_string(),
                        )
                    })?;

                    let block_size = quant_cfg
                        .weight_block_size
                        .clone()
                        .unwrap_or(vec![128, 128]);
                    if block_size.len() != 2 {
                        candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
                    }
                    let by = block_size[0];
                    let bx = block_size[1];
                    if by == 0 || bx == 0 {
                        candle_core::bail!("LnFp8: invalid zero in weight_block_size");
                    }

                    let weight = v.get_with_hints_dtype(
                        (out_dim, in_dim),
                        "weight",
                        Shard::default(),
                        DType::U8,
                    )?;
                    let scale_dim0 = (out_dim + by - 1) / by;
                    let scale_dim1 = (in_dim + bx - 1) / bx;
                    let weight_scale = match v.get_with_hints_dtype(
                        (scale_dim0, scale_dim1),
                        "weight_scale",
                        Shard::default(),
                        DType::F32,
                    ) {
                        Ok(s) => s,
                        Err(_) => v
                            .get_with_hints_dtype(
                                (scale_dim0, scale_dim1),
                                "weight_scale_inv",
                                Shard::default(),
                                DType::F32,
                            )
                            .map_err(|_| {
                                candle_core::Error::Msg(
                                    "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                                )
                            })?,
                    };

                    #[cfg(feature = "cuda")]
                    let sm_version =
                        attention_rs::cuda_utils::sm_version(v.device().as_cuda_device()?)
                            .unwrap_or(0) as usize;

                    #[cfg(not(feature = "cuda"))]
                    let sm_version = 0;

                    let mut chunk_start = 0;
                    let mut local_weight_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_scale_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_output_splits = Vec::<usize>::with_capacity(chunks.len());
                    for chunk_idx in 0..chunks.len() {
                        let chunk_size = chunks[chunk_idx];
                        let ws = weight.narrow(0, chunk_start, chunk_size)?;
                        let chunk_shard = if let Some(chunk_shards) = &chunk_shards {
                            chunk_shards[chunk_idx]
                        } else {
                            shard(0, comm.rank(), comm.world_size())
                        };
                        if chunk_shard.dim != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} shard dim {} is unsupported, expected 0",
                                chunk_idx,
                                chunk_shard.dim
                            );
                        }
                        if ws.dim(0)? % chunk_shard.world_size != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} dim {} is not divisible by shard world_size {}",
                                chunk_idx,
                                ws.dim(0)?,
                                chunk_shard.world_size
                            );
                        }
                        let local_out = ws.dim(0)? / chunk_shard.world_size;
                        if local_out == 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} produced empty shard",
                                chunk_idx
                            );
                        }
                        let local_out_start = chunk_start + chunk_shard.rank * local_out;
                        if local_out_start % by != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} local start {} is not aligned to block_size_y {}",
                                chunk_idx,
                                local_out_start,
                                by
                            );
                        }
                        let ws_chunk = ws
                            .narrow(0, chunk_shard.rank * local_out, local_out)?
                            .contiguous()?;
                        local_output_splits.push(local_out);

                        let scale_row_start = local_out_start / by;
                        let scale_rows = (local_out + by - 1) / by;
                        if scale_row_start + scale_rows > scale_dim0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} scale slice out of bounds: start={}, rows={}, total={}",
                                chunk_idx,
                                scale_row_start,
                                scale_rows,
                                scale_dim0
                            );
                        }
                        let ws_scale = weight_scale
                            .narrow(0, scale_row_start, scale_rows)?
                            .contiguous()?;
                        local_weight_chunks.push(ws_chunk);
                        local_scale_chunks.push(ws_scale);
                        chunk_start += chunk_size;
                    }

                    let merged_weight = if local_weight_chunks.len() == 1 {
                        local_weight_chunks.remove(0)
                    } else {
                        let weight_refs = local_weight_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&weight_refs, 0)?
                    };

                    let merged_scale = if local_scale_chunks.len() == 1 {
                        local_scale_chunks.remove(0)
                    } else {
                        let scale_refs = local_scale_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&scale_refs, 0)?
                    };

                    #[cfg(feature = "cutlass")]
                    let merged_scale_cutlass = if sm_version >= 100 {
                        Some(merged_scale.t()?)
                    } else if sm_version >= 90 {
                        Some(merged_scale.t()?.contiguous()?)
                    } else {
                        None
                    };

                    #[cfg(not(feature = "cutlass"))]
                    let merged_scale_cutlass = None;

                    let linear = LinearX::LnFp8(LnFp8 {
                        weight: merged_weight,
                        weight_scale: merged_scale,
                        weight_scale_cutlass: merged_scale_cutlass,
                        bias: None,
                        weight_block_size: block_size.clone(),
                        sm_version,
                    });
                    let ln = TensorParallelColumnLinear { linear };
                    vec_linear.push(ln);
                    output_splits = Some(local_output_splits);
                } else if is_nvfp4_quant || is_mxfp4_quant {
                    if chunk_dim != 0 {
                        candle_core::bail!(
                            "FP4 merged chunk loading currently supports chunk_dim=0 only, got {}",
                            chunk_dim
                        );
                    }

                    let is_mlx = quant_cfg.as_ref().map_or(false, |c| c.is_mlx_nvfp4);
                    let pack_factor = 2; // 2 FP4 values per byte
                    let blocks = if is_mlx {
                        // MLX: weight is U32 [out_dim, in_dim/8], repack to U8 on GPU
                        let w_u32 = v.get_with_hints_dtype(
                            (out_dim, in_dim / 8),
                            "weight",
                            Shard::default(),
                            DType::U32,
                        )?;
                        attention_rs::nvfp4_linear::mlx_repack_u32_to_u8(&w_u32)?
                    } else if v.contains_tensor("weight_packed") {
                        v.get_with_hints_dtype(
                            (out_dim, in_dim / pack_factor),
                            "weight_packed",
                            Shard::default(),
                            DType::U8,
                        )?
                    } else if v.contains_tensor("weight") {
                        v.get_with_hints_dtype(
                            (out_dim, in_dim / pack_factor),
                            "weight",
                            Shard::default(),
                            DType::U8,
                        )?
                    } else {
                        v.get_with_hints_dtype(
                            (out_dim, in_dim / pack_factor),
                            "blocks",
                            Shard::default(),
                            DType::U8,
                        )?
                    };

                    let scale_block_size: usize = if is_nvfp4_quant { 16 } else { 32 };
                    let scale_cols = in_dim / scale_block_size;
                    let scales = if v.contains_tensor("weight_scale") {
                        v.get_with_hints_dtype(
                            (out_dim, scale_cols),
                            "weight_scale",
                            Shard::default(),
                            DType::U8,
                        )?
                    } else {
                        v.get_with_hints_dtype(
                            (out_dim, scale_cols),
                            "scales",
                            Shard::default(),
                            DType::U8,
                        )?
                    };

                    #[allow(unused_variables)]
                    let (nvfp4_global_scale, nvfp4_input_scale, nvfp4_sm) = if is_nvfp4_quant {
                        if is_mlx {
                            #[cfg(feature = "cuda")]
                            let sm =
                                attention_rs::cuda_utils::sm_version(v.device().as_cuda_device()?)
                                    .unwrap_or(0) as usize;
                            #[cfg(not(feature = "cuda"))]
                            let sm = 0usize;
                            (1.0f32, 1.0f32, sm)
                        } else {
                            let no_shard = Shard::default();
                            let global_scale = if v.contains_tensor("weight_global_scale") {
                                let t = match v.get_with_hints_dtype(
                                    (1,),
                                    "weight_global_scale",
                                    no_shard,
                                    DType::F32,
                                ) {
                                    Ok(t) => t,
                                    Err(_) => v.get_with_hints_dtype(
                                        (),
                                        "weight_global_scale",
                                        no_shard,
                                        DType::F32,
                                    )?,
                                };
                                let raw = t.flatten_all()?.to_vec1::<f32>()?[0];
                                if raw != 0.0 {
                                    1.0 / raw
                                } else {
                                    1.0
                                }
                            } else if v.contains_tensor("weight_scale_2") {
                                let t = match v.get_with_hints_dtype(
                                    (1,),
                                    "weight_scale_2",
                                    no_shard,
                                    DType::F32,
                                ) {
                                    Ok(t) => t,
                                    Err(_) => v.get_with_hints_dtype(
                                        (),
                                        "weight_scale_2",
                                        no_shard,
                                        DType::F32,
                                    )?,
                                };
                                t.flatten_all()?.to_vec1::<f32>()?[0]
                            } else {
                                1.0f32
                            };

                            let input_scale = if v.contains_tensor("input_scale") {
                                let t = match v.get_with_hints_dtype(
                                    (1,),
                                    "input_scale",
                                    no_shard,
                                    DType::F32,
                                ) {
                                    Ok(t) => t,
                                    Err(_) => v.get_with_hints_dtype(
                                        (),
                                        "input_scale",
                                        no_shard,
                                        DType::F32,
                                    )?,
                                };
                                t.flatten_all()?.to_vec1::<f32>()?[0]
                            } else if v.contains_tensor("input_global_scale") {
                                let t = match v.get_with_hints_dtype(
                                    (1,),
                                    "input_global_scale",
                                    no_shard,
                                    DType::F32,
                                ) {
                                    Ok(t) => t,
                                    Err(_) => v.get_with_hints_dtype(
                                        (),
                                        "input_global_scale",
                                        no_shard,
                                        DType::F32,
                                    )?,
                                };
                                let raw = t.flatten_all()?.to_vec1::<f32>()?[0];
                                if raw != 0.0 {
                                    1.0 / raw
                                } else {
                                    1.0
                                }
                            } else {
                                1.0f32
                            };

                            #[cfg(feature = "cuda")]
                            let sm =
                                attention_rs::cuda_utils::sm_version(v.device().as_cuda_device()?)
                                    .unwrap_or(0) as usize;
                            #[cfg(not(feature = "cuda"))]
                            let sm = 0usize;

                            (global_scale, input_scale, sm)
                        }
                    } else {
                        (1.0f32, 1.0f32, 0usize)
                    };

                    let mut chunk_start = 0;
                    let mut local_block_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_scale_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_output_splits = Vec::<usize>::with_capacity(chunks.len());
                    for chunk_idx in 0..chunks.len() {
                        let chunk_size = chunks[chunk_idx];
                        let chunk_shard = if let Some(chunk_shards) = &chunk_shards {
                            chunk_shards[chunk_idx]
                        } else {
                            shard(0, comm.rank(), comm.world_size())
                        };
                        if chunk_size % chunk_shard.world_size != 0 {
                            candle_core::bail!(
                                "FP4 merged chunk {} dim {} is not divisible by shard world_size {}",
                                chunk_idx,
                                chunk_size,
                                chunk_shard.world_size
                            );
                        }
                        let local_out = chunk_size / chunk_shard.world_size;
                        let local_start = chunk_start + chunk_shard.rank * local_out;

                        local_block_chunks
                            .push(blocks.narrow(0, local_start, local_out)?.contiguous()?);
                        local_scale_chunks
                            .push(scales.narrow(0, local_start, local_out)?.contiguous()?);
                        local_output_splits.push(local_out);
                        chunk_start += chunk_size;
                    }

                    let merged_blocks = if local_block_chunks.len() == 1 {
                        local_block_chunks.remove(0)
                    } else {
                        let refs = local_block_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&refs, 0)?
                    };
                    let merged_scales = if local_scale_chunks.len() == 1 {
                        local_scale_chunks.remove(0)
                    } else {
                        let refs = local_scale_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&refs, 0)?
                    };

                    let linear = if is_nvfp4_quant {
                        #[cfg(feature = "cuda")]
                        let weight_scale_swizzled = if nvfp4_sm >= 100 {
                            Some(attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                                &merged_scales,
                            )?)
                        } else {
                            None
                        };
                        #[cfg(not(feature = "cuda"))]
                        let weight_scale_swizzled: Option<Tensor> = None;

                        LinearX::LnNvfp4(LnNvfp4 {
                            blocks: merged_blocks,
                            scales: merged_scales,
                            global_scale: nvfp4_global_scale,
                            input_scale: nvfp4_input_scale,
                            bias: None,
                            weight_scale_swizzled,
                        })
                    } else {
                        LinearX::LnMxfp4(LnMxfp4 {
                            blocks: merged_blocks,
                            scales: merged_scales,
                            bias: None,
                        })
                    };

                    let ln = TensorParallelColumnLinear { linear };
                    vec_linear.push(ln);
                    output_splits = Some(local_output_splits);
                } else {
                    let weight = v.get((out_dim, in_dim), "weight")?;
                    let weight = if weight.dtype() != dtype {
                        weight.to_dtype(dtype)?
                    } else {
                        weight
                    };
                    let mut chunk_start = 0;
                    for chunk_idx in 0..chunks.len() {
                        let chunk_size = chunks[chunk_idx];
                        let ws = weight.narrow(chunk_dim, chunk_start, chunk_size)?;
                        let ws_chunk = if let Some(chunk_shards) = &chunk_shards {
                            let chunk_shard = &chunk_shards[chunk_idx];
                            if ws.dim(0)? % chunk_shard.world_size != 0 {
                                candle_core::bail!(
                                    "merged chunk {} dim {} is not divisible by shard world_size {}",
                                    chunk_idx,
                                    ws.dim(0)?,
                                    chunk_shard.world_size
                                );
                            }
                            let c_chunk_size = ws.dim(0)? / chunk_shard.world_size;
                            ws.narrow(0, chunk_shard.rank * c_chunk_size, c_chunk_size)?
                                .contiguous()?
                        } else {
                            if ws.dim(0)? % comm.world_size() != 0 {
                                candle_core::bail!(
                                    "merged chunk {} dim {} is not divisible by comm world_size {}",
                                    chunk_idx,
                                    ws.dim(0)?,
                                    comm.world_size()
                                );
                            }
                            let c_chunk_size = ws.dim(0)? / comm.world_size();
                            ws.narrow(0, comm.rank() * c_chunk_size, c_chunk_size)?
                                .contiguous()?
                        };
                        chunk_start += chunk_size;

                        let ln = crate::models::layers::linear::Linear::new(ws_chunk, None);
                        let linear = if let Some(quantized_type) = quant {
                            let quantized_type = if chunk_idx == chunks.len() - 1 {
                                "q8_0".to_string()
                            } else {
                                quantized_type.clone()
                            };
                            LinearX::QLinear(QLinear::from_linear_x(ln, quantized_type, dtype)?)
                        } else {
                            LinearX::Linear(ln)
                        };
                        let ln = TensorParallelColumnLinear { linear };
                        vec_linear.push(ln);
                    }
                }
            }
            _ => {
                candle_core::bail!("Quantized qkv weight not supported!")
            }
        }

        let linear_count = vec_linear.len();
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; linear_count],
            output_splits,
        })
    }
}

impl TensorParallelRowLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear(
            in_dim,
            out_dim,
            vb,
            shard(1, comm.rank(), comm.world_size()),
            quant_cfg,
            quant,
            dtype,
        )?;
        Ok(Self::new(linear, comm, dtype))
    }
}

impl ReplicatedLinear {
    pub fn from(linear: Linear) -> Result<Self> {
        Ok(Self { linear })
    }

    pub fn from_weight_bias(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = Linear::new(weight, bias, &None)?;
        Ok(Self { linear })
    }

    pub fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb, shard(0, 0, 1), quant_cfg, quant, dtype)?;
        Ok(Self { linear })
    }

    pub fn load_b(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        if !bias {
            ReplicatedLinear::load_no_bias(in_dim, out_dim, vb, quant_cfg, quant, dtype)
        } else {
            let linear = linear_b(
                in_dim,
                out_dim,
                bias,
                vb,
                shard(0, 0, 1),
                quant_cfg,
                quant,
                dtype,
            )?;
            Ok(Self { linear })
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

/// Vocabulary-parallel linear layer for lm_head.
///
/// Shards the output (vocab) dimension across TP ranks. Each rank holds
/// `vocab_size_padded / world_size` rows of the weight matrix and computes
/// partial logits. An all-gather on the last dimension assembles the full
/// logits tensor, which is then trimmed to the original vocab_size.
///
#[allow(dead_code)]
pub struct VocabParallelLinear {
    linear: Linear,
    #[cfg(feature = "nccl")]
    all_gather: Option<AllGather>,
    org_vocab_size: usize,
    dtype: DType,
}

#[allow(dead_code)]
pub struct AllGather {
    comm: Rc<Comm>,
    world_size: usize,
}

unsafe impl Sync for AllGather {}
unsafe impl Send for AllGather {}

impl AllGather {
    pub fn new(comm: Rc<Comm>) -> Self {
        let world_size = comm.world_size();
        Self { comm, world_size }
    }

    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(self)
    }
}

impl CustomOp1 for AllGather {
    fn name(&self) -> &'static str {
        "allgather"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllGather is never used on cpu")
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let start_offset = l.start_offset();
        let total_elems = elem_count * self.world_size;

        let dst = match s.dtype() {
            DType::BF16 => {
                let full_slice = s.as_cuda_slice::<bf16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather BF16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<bf16>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather F16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f16>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F32 => {
                let full_slice = s.as_cuda_slice::<f32>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather F32 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f32>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype for all_gather: {dtype:?}"),
        };

        // NCCL all_gather output layout: [rank0_chunk | rank1_chunk | ... | rankN_chunk]
        // Each chunk has `elem_count` elements. We report the raw gathered shape as
        // [world_size * dim0, dim1, ..., dimN] (concat along dim 0).
        // The caller (VocabParallelLinear::forward) handles the reshape/transpose
        // to get the correct [batch, full_vocab] layout.
        let dims = l.shape().dims();
        let mut out_dims = dims.to_vec();
        out_dims[0] *= self.world_size;
        Ok((dst, Shape::from_dims(&out_dims)))
    }
}

const VOCAB_PADDING_SIZE: usize = 64;

fn pad_vocab_size(vocab_size: usize, world_size: usize) -> usize {
    let padded = ((vocab_size + VOCAB_PADDING_SIZE - 1) / VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE;
    let per_rank = ((padded + world_size - 1) / world_size) * world_size;
    ((per_rank + VOCAB_PADDING_SIZE - 1) / VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE
}

impl VocabParallelLinear {
    #[allow(unused_variables)]
    pub fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let world_size = comm.world_size();
        if world_size <= 1 {
            let linear = linear(in_dim, out_dim, vb, shard(0, 0, 1), quant_cfg, quant, dtype)?;
            return Ok(Self {
                linear,
                #[cfg(feature = "nccl")]
                all_gather: None,
                org_vocab_size: out_dim,
                dtype,
            });
        }

        let padded_vocab = pad_vocab_size(out_dim, world_size);
        let linear = linear(
            in_dim,
            padded_vocab,
            vb,
            shard(0, comm.rank(), world_size),
            quant_cfg,
            quant,
            dtype,
        )?;

        #[cfg(feature = "nccl")]
        let all_gather = Some(AllGather::new(comm));

        Ok(Self {
            linear,
            #[cfg(feature = "nccl")]
            all_gather,
            org_vocab_size: out_dim,
            dtype,
        })
    }

    pub fn from_weight_bias(
        weight: Tensor,
        bias: Option<Tensor>,
        comm: Rc<Comm>,
        org_vocab_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let world_size = comm.world_size();
        if world_size <= 1 {
            let linear = Linear::new(weight, bias, &None)?;
            return Ok(Self {
                linear,
                #[cfg(feature = "nccl")]
                all_gather: None,
                org_vocab_size,
                dtype,
            });
        }

        let vocab_dim = weight.dim(0)?;
        let padded_vocab = pad_vocab_size(org_vocab_size, world_size);
        let local_vocab = padded_vocab / world_size;
        let rank = comm.rank();

        let weight = if vocab_dim < padded_vocab {
            let hidden = weight.dim(1)?;
            let pad_rows = padded_vocab - vocab_dim;
            let padding = Tensor::zeros((pad_rows, hidden), weight.dtype(), weight.device())?;
            Tensor::cat(&[&weight, &padding], 0)?
        } else {
            weight
        };

        let local_weight = weight
            .narrow(0, rank * local_vocab, local_vocab)?
            .contiguous()?;

        let linear = Linear::new(local_weight, bias, &None)?;

        #[cfg(feature = "nccl")]
        let all_gather = Some(AllGather::new(comm));

        Ok(Self {
            linear,
            #[cfg(feature = "nccl")]
            all_gather,
            org_vocab_size,
            dtype,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.linear.forward(x)?;

        #[cfg(feature = "nccl")]
        if let Some(all_gather) = &self.all_gather {
            // logits shape: [batch, local_vocab]
            // After NCCL all_gather: [world_size * batch, local_vocab] (concat along dim 0)
            // We need to reshape to [world_size, batch, local_vocab], transpose to
            // [batch, world_size, local_vocab], then flatten last two dims to [batch, vocab]
            let gathered = if logits.dtype() != self.dtype {
                let g = all_gather.apply(&logits.to_dtype(self.dtype)?)?;
                g.to_dtype(logits.dtype())?
            } else {
                all_gather.apply(&logits)?
            };

            let ws = all_gather.world_size;
            let local_vocab = logits.dim(logits.dims().len() - 1)?;
            let batch = logits.dims()[..logits.dims().len() - 1]
                .iter()
                .product::<usize>();

            // gathered is [ws * batch, local_vocab]
            // reshape to [ws, batch, local_vocab]
            let gathered = gathered.reshape((ws, batch, local_vocab))?;
            // transpose to [batch, ws, local_vocab]
            let gathered = gathered.transpose(0, 1)?.contiguous()?;
            // flatten to [batch, ws * local_vocab]
            let full_vocab = ws * local_vocab;
            let logits = gathered.reshape((batch, full_vocab))?;

            // Restore original batch dimensions if input was multi-dimensional
            let orig_dims = &x.dims()[..x.dims().len() - 1];
            let logits = if orig_dims.len() > 1 {
                let mut shape = orig_dims.to_vec();
                shape.push(full_vocab);
                logits.reshape(shape)?
            } else {
                logits
            };

            if full_vocab > self.org_vocab_size {
                let last_dim = logits.dims().len() - 1;
                return logits.narrow(last_dim, 0, self.org_vocab_size);
            }
            return Ok(logits);
        }

        Ok(logits)
    }
}
