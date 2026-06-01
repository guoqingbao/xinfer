//! Linear layer (GGUF and unquantized safetensors)
use super::wna16::WNA16;
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
use crate::utils::should_skip_fp8_for_module;
use crate::utils::should_skip_quant_for_module;
use candle_core::quantized;
use candle_core::quantized::GgmlDType;
use candle_core::Module;
use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Result, Tensor,
};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    static LINEAR_IS_PREFILL: Cell<bool> = const { Cell::new(false) };
}

pub struct LinearPrefillGuard {
    prev: bool,
}

impl Drop for LinearPrefillGuard {
    fn drop(&mut self) {
        LINEAR_IS_PREFILL.with(|flag| flag.set(self.prev));
    }
}

pub fn set_linear_is_prefill(is_prefill: bool) -> LinearPrefillGuard {
    let prev = LINEAR_IS_PREFILL.with(|flag| {
        let prev = flag.get();
        flag.set(is_prefill);
        prev
    });
    LinearPrefillGuard { prev }
}

pub fn linear_is_prefill() -> bool {
    LINEAR_IS_PREFILL.with(|flag| flag.get())
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [b1, seq_len, _, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left((b1, seq_len))?.t()?
                } else {
                    self.weight.t()?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left(bsize)?.t()?
                } else {
                    self.weight.t()?
                }
            }
            _ => self.weight.t()?,
        };
        let x = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim1, dim2))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, dim1, wdim))?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, wdim))?
                }
            }
            _ => x.matmul(&w)?,
        };

        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    Ok(Linear::new(weight, None))
}

pub fn linear_no_bias_merged(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let sd = shard(shards.dim + 1, shards.rank, shards.world_size);
    let weight = vb.get_with_hints((num_experts, out_dim, in_dim), "weight", sd)?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    Ok(Linear::new(weight, None))
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let ws = if ws.dtype() != dtype {
        ws.to_dtype(dtype)?
    } else {
        ws
    };
    let bs = vb.get((out_dim,), "bias");
    let bs = if bs.is_ok() {
        let bs = bs.unwrap();
        let bs = if shard.world_size > 1 {
            let dim_size = bs.dim(0)?;
            let start = shard.rank * (dim_size / shard.world_size);
            bs.narrow(0, start, dim_size / shard.world_size)?
                .contiguous()?
        } else {
            bs
        };
        let bs = if bs.dtype() != dtype {
            bs.to_dtype(dtype)?
        } else {
            bs
        };
        Some(bs)
    } else {
        None
    };

    Ok(Linear::new(ws, bs))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb, shard, dtype)
    } else {
        linear_no_bias(in_dim, out_dim, vb, shard, dtype)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QLinear {
    pub inner: Option<QMatMul>,
    pub bias: Option<Tensor>,
    pub wna16: Option<WNA16>,
    pub dtype: DType,
}

impl QLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: crate::utils::gguf_varbuilder::VarBuilder,
        shards: Shard,
        dtype: DType,
    ) -> Result<Self> {
        let ws = vb.get((out_dim, in_dim), "weight")?;
        let mut wdtype = ws.dtype();
        let ws = if shards.world_size > 1 {
            let ws = ws.dequantize_f16(&vb.device())?;
            let chunk_size = ws.shape().dims()[shards.dim] / shards.world_size;
            if chunk_size % wdtype.block_size() != 0 {
                // crate::log_warn!(
                //     "Invalid dim_size to chunk {} (start {}, size {}) for block_size {}, switching to Q8_0 format!",
                //     ws.shape().dims()[shards.dim],
                //     shards.rank * chunk_size,
                //     chunk_size,
                //     wdtype.block_size()
                // );
                wdtype = GgmlDType::Q8_0;
            }

            let ws = ws
                .narrow(shards.dim, shards.rank * chunk_size, chunk_size)?
                .contiguous()?;
            let qtensor = QTensor::quantize(&ws, wdtype)?;
            Arc::new(qtensor)
        } else {
            ws.to_owned()
        };
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let b = vb.get(out_dim, "bias");
        let bias = if b.is_ok() {
            let bw = b.unwrap().dequantize(vb.device())?;
            if shards.world_size > 1 {
                let bw_chunk = bw.dim(0)? / shards.world_size;
                Some(
                    bw.narrow(0, shards.rank * bw_chunk, bw_chunk)?
                        .contiguous()?,
                )
            } else {
                Some(bw)
            }
        } else {
            None
        };
        Ok(Self {
            inner: Some(inner),
            bias,
            wna16: None,
            dtype,
        })
    }

    pub fn new_fused(
        num_experts: usize,
        in_dim: usize,
        out_dim: usize,
        vb: crate::utils::gguf_varbuilder::VarBuilder,
        shards: Shard,
        dtype: DType,
    ) -> Result<Self> {
        let ws = vb.get((num_experts, out_dim, in_dim), "weight")?;
        let wdtype = ws.dtype();
        let ws = if shards.world_size > 1 {
            let ws = ws.dequantize_f16(&vb.device())?;
            let chunk_size = ws.shape().dims()[shards.dim + 1] / shards.world_size;
            assert!(
                chunk_size % wdtype.block_size() == 0,
                "chunk position invalid dim {}, start {}, size {}",
                ws.shape().dims()[shards.dim],
                shards.rank * chunk_size,
                chunk_size
            );
            let ws = ws
                .narrow(shards.dim + 1, shards.rank * chunk_size, chunk_size)?
                .contiguous()?;
            let qtensor = QTensor::quantize(&ws, wdtype)?;
            Arc::new(qtensor)
        } else {
            ws.to_owned()
        };

        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let b = vb.get(out_dim, "bias");
        let bias = if b.is_ok() {
            let bw = b.unwrap().dequantize(vb.device())?;
            if shards.world_size > 1 {
                let bw_chunk = bw.dim(0)? / shards.world_size;
                Some(
                    bw.narrow(0, shards.rank * bw_chunk, bw_chunk)?
                        .contiguous()?,
                )
            } else {
                Some(bw)
            }
        } else {
            None
        };
        Ok(Self {
            inner: Some(inner),
            bias,
            wna16: None,
            dtype,
        })
    }

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType) -> Result<Self> {
        let bx = match b {
            Some(b_) => Some(b_.to_dtype(DType::F32)?),
            _ => None,
        };

        Ok(Self {
            inner: Some(QMatMul::QTensor(Arc::new(w))),
            bias: bx,
            wna16: None,
            dtype,
        })
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match &self.inner {
            Some(QMatMul::QTensor(t)) => t.dequantize(&t.device()),
            _ => {
                panic!("Not supported!");
            }
        }
    }
    //in-situ quantization
    pub fn from_linear_x(linear: Linear, quant: String, dtype: DType) -> Result<Self> {
        use quantized::GgmlDType;
        let ggml_dtype = match quant.as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };
        let weight = linear.weight();
        let qbias = linear.bias().cloned();
        let last_dim = weight.dim(candle_core::D::Minus1)?;
        let actual_ggml_dtype = if last_dim % ggml_dtype.block_size() != 0 {
            if last_dim % GgmlDType::Q8_0.block_size() == 0 {
                crate::log_warn!(
                    "ISQ: weight {:?} incompatible with {:?} (block_size {}), \
                    falling back to Q8_0 (block_size {})",
                    weight.shape(),
                    ggml_dtype,
                    ggml_dtype.block_size(),
                    GgmlDType::Q8_0.block_size()
                );
                GgmlDType::Q8_0
            } else {
                crate::log_warn!(
                    "ISQ: weight {:?} incompatible with any GGUF dtype, keeping unquantized",
                    weight.shape()
                );
                let inner = QMatMul::Tensor(weight.clone());
                return Ok(QLinear {
                    inner: Some(inner),
                    bias: qbias,
                    wna16: None,
                    dtype,
                });
            }
        } else {
            ggml_dtype
        };
        let qtensor = QTensor::quantize(weight, actual_ggml_dtype)?;
        QLinear::from_qparts_x(qtensor, qbias, dtype)
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(wna16) = &self.wna16 {
            wna16.forward(x)
        } else if let Some(inner) = &self.inner {
            let xs = if x.dtype() != DType::F32 {
                x.to_dtype(DType::F32)?
            } else {
                x.to_owned()
            };
            let xs = QMatMul::forward(inner, &xs)?;

            if let Some(bias) = &self.bias {
                xs.broadcast_add(bias)
            } else {
                Ok(xs)
            }
        } else {
            candle_core::bail!("Invalid quantization type!")
        }
    }
}

impl QLinear {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        if let Some(inner) = &self.inner {
            let xs = inner.indexed_moe_forward(&x.to_dtype(DType::F32)?, ids)?;
            if let Some(bias) = &self.bias {
                xs.broadcast_add(bias)?.to_dtype(self.dtype)
            } else {
                xs.to_dtype(self.dtype)
            }
        } else {
            candle_core::bail!("Invalid quantization type!")
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    LnFp8(LnFp8),
    LnMxfp4(LnMxfp4),
    LnNvfp4(LnNvfp4),
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(ln) => ln.forward(x),
            Self::QLinear(ln) => ln.forward(x),
            Self::LnMxfp4(ln) => ln.forward(x),
            Self::LnNvfp4(ln) => ln.forward(x),
            Self::LnFp8(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("No supported!")
            }
            Self::QLinear(ln) => ln.indexed_moe_forward(x, ids),
            Self::LnFp8(_) => panic!("LnFp8 does not support indexed_moe_forward yet"),
            Self::LnMxfp4(_) => panic!("LnMxfp4 does not support indexed_moe_forward yet"),
            Self::LnNvfp4(_) => panic!("LnNvfp4 does not support indexed_moe_forward yet"),
        }
    }
}

impl LinearX {
    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Result<Self> {
        let dtype = weight.dtype();
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            Ok(Self::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                dtype,
            )?))
        } else {
            Ok(Self::Linear(ln))
        }
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("Unquantized tensor unable to be dequantized!")
            }
            Self::QLinear(ln) => ln.dequantize(),
            Self::LnFp8(_) => panic!("LnFp8 unable to be dequantized"),
            Self::LnMxfp4(_) => panic!("LnMxfp4 unable to be dequantized"),
            Self::LnNvfp4(_) => panic!("LnNvfp4 unable to be dequantized"),
        }
    }
}

fn has_fp4_scale_tensors(vb: &VarBuilder, is_mlx_nvfp4: bool) -> bool {
    if is_mlx_nvfp4 {
        vb.contains_tensor("weight") && vb.contains_tensor("scales")
    } else {
        vb.contains_tensor("weight_scale")
            || vb.contains_tensor("weight_scale_2")
            || vb.contains_tensor("weight_global_scale")
            || vb.contains_tensor("weight_packed")
            || vb.contains_tensor("blocks")
    }
}

fn has_nvfp4_specific_tensors(vb: &VarBuilder, is_mlx_nvfp4: bool) -> bool {
    if is_mlx_nvfp4 {
        vb.contains_tensor("weight") && vb.contains_tensor("scales")
    } else {
        vb.contains_tensor("weight_packed")
            || vb.contains_tensor("blocks")
            || vb.contains_tensor("weight_scale_2")
            || vb.contains_tensor("weight_global_scale")
    }
}

fn has_fp8_tensors(vb: &VarBuilder) -> bool {
    vb.contains_tensor("weight_scale") || vb.contains_tensor("weight_scale_inv")
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let module_path = vb.module_path().to_string();
    match &vb.0 {
        Either::Left(vb) => {
            if let Some(cfg) = quant_cfg {
                if cfg.quant_method == "fp8" {
                    if should_skip_fp8_for_module(&module_path, cfg) {
                        let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }

                    let has_fp8_scale = vb.contains_tensor("weight_scale")
                        || vb.contains_tensor("weight_scale_inv");
                    if !has_fp8_scale {
                        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shards)?;
                        if matches!(
                            weight.dtype(),
                            DType::BF16 | DType::F16 | DType::F32 | DType::F64
                        ) {
                            let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                            return Ok(LinearX::Linear(ln));
                        }
                    }

                    match load_ln_fp8_with_hints(in_dim, out_dim, vb.clone(), shards, cfg, true) {
                        Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                        Err(err) => return Err(err),
                    }
                }

                if cfg.quant_method == "mxfp4" {
                    if should_skip_quant_for_module(&module_path, cfg) {
                        let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    if !has_fp4_scale_tensors(&vb, false) {
                        let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    let ln = LnMxfp4::load(in_dim, out_dim, vb.clone(), shards, true)?;
                    return Ok(LinearX::LnMxfp4(ln));
                }

                if cfg.quant_method == "nvfp4" {
                    let is_mlx = cfg.is_mlx_nvfp4;
                    if should_skip_quant_for_module(&module_path, cfg)
                        || !has_fp4_scale_tensors(&vb, is_mlx)
                    {
                        let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    if !is_mlx && !has_nvfp4_specific_tensors(&vb, false) && has_fp8_tensors(&vb) {
                        match load_ln_fp8_with_hints(in_dim, out_dim, vb.clone(), shards, cfg, true)
                        {
                            Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                            Err(_) => {
                                let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                                return Ok(LinearX::Linear(ln));
                            }
                        }
                    }
                    let ln = if is_mlx {
                        LnNvfp4::load_mlx(in_dim, out_dim, vb.clone(), shards, true)?
                    } else {
                        LnNvfp4::load(in_dim, out_dim, vb.clone(), shards, true)?
                    };
                    return Ok(LinearX::LnNvfp4(ln));
                }

                let wna16 = WNA16::new(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    true,
                    dtype,
                    true,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX::QLinear(ln))
            } else {
                let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX::QLinear(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?))
                } else {
                    Ok(LinearX::Linear(ln))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let module_path = vb.module_path().to_string();
    match &vb.0 {
        Either::Left(vb) => {
            if let Some(cfg) = quant_cfg {
                if cfg.quant_method == "fp8" {
                    if should_skip_fp8_for_module(&module_path, cfg) {
                        let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }

                    let has_fp8_scale = vb.contains_tensor("weight_scale")
                        || vb.contains_tensor("weight_scale_inv");
                    if !has_fp8_scale {
                        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shards)?;
                        if matches!(
                            weight.dtype(),
                            DType::BF16 | DType::F16 | DType::F32 | DType::F64
                        ) {
                            let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                            return Ok(LinearX::Linear(ln));
                        }
                    }

                    match load_ln_fp8_with_hints(in_dim, out_dim, vb.clone(), shards, cfg, false) {
                        Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                        Err(err) => return Err(err),
                    }
                }

                if cfg.quant_method == "mxfp4" {
                    if should_skip_quant_for_module(&module_path, cfg)
                        || !has_fp4_scale_tensors(&vb, false)
                    {
                        let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    let ln = LnMxfp4::load(in_dim, out_dim, vb.clone(), shards, false)?;
                    return Ok(LinearX::LnMxfp4(ln));
                }

                if cfg.quant_method == "nvfp4" {
                    let is_mlx = cfg.is_mlx_nvfp4;
                    if should_skip_quant_for_module(&module_path, cfg) {
                        let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    if !has_fp4_scale_tensors(&vb, is_mlx) {
                        let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                        return Ok(LinearX::Linear(ln));
                    }
                    if !is_mlx && !has_nvfp4_specific_tensors(&vb, false) && has_fp8_tensors(&vb) {
                        match load_ln_fp8_with_hints(
                            in_dim,
                            out_dim,
                            vb.clone(),
                            shards,
                            cfg,
                            false,
                        ) {
                            Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                            Err(_) => {
                                let ln =
                                    linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                                return Ok(LinearX::Linear(ln));
                            }
                        }
                    }
                    let ln = if is_mlx {
                        LnNvfp4::load_mlx(in_dim, out_dim, vb.clone(), shards, false)?
                    } else {
                        LnNvfp4::load(in_dim, out_dim, vb.clone(), shards, false)?
                    };
                    return Ok(LinearX::LnNvfp4(ln));
                }

                let wna16 = WNA16::new(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    false,
                    dtype,
                    true,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX::QLinear(ln))
            } else {
                let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX::QLinear(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?))
                } else {
                    Ok(LinearX::Linear(ln))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
    }
}

pub fn linear_no_bias_merged_x(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    _: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            let ln =
                linear_no_bias_merged(num_experts, in_dim, out_dim, vb.clone(), shards, dtype)?;
            if let Some(quantized_type) = quant {
                Ok(LinearX::QLinear(QLinear::from_linear_x(
                    ln,
                    quantized_type.clone(),
                    dtype,
                )?))
            } else {
                Ok(LinearX::Linear(ln))
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new_fused(
            num_experts,
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilderX,
    shard: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    }
}

#[derive(Debug, Clone)]
pub struct LnFp8 {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub weight_scale_cutlass: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub weight_block_size: Vec<usize>,
    pub sm_version: usize,
}

impl LnFp8 {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        // Expected format:
        // weight: [out_dim, in_dim]
        // weight_scale: [out_dim, in_dim // block_size[1]]  (assuming block_size_y = 1)
        // Or weight_scale: [out_dim // block_size_y, in_dim // block_size_x]

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
        }

        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
        let weight = if weight.dtype() != DType::U8 {
            weight.to_dtype(DType::U8)?
        } else {
            weight
        };

        let by = block_size[0];
        let bx = block_size[1];

        let scale_dim0 = (out_dim + by - 1) / by;
        let scale_dim1 = (in_dim + bx - 1) / bx;

        let weight_scale = match vb.get_with_hints((scale_dim0, scale_dim1), "weight_scale", shard)
        {
            Ok(s) => s,
            Err(_) => vb
                .get_with_hints((scale_dim0, scale_dim1), "weight_scale_inv", shard)
                .map_err(|_| {
                    candle_core::Error::Msg(
                        "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                    )
                })?,
        }
        .to_dtype(DType::F32)?;

        #[cfg(feature = "cuda")]
        let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
            .unwrap_or(0) as usize;

        #[cfg(not(feature = "cuda"))]
        let sm_version = 0;

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

        // Load bias if present
        let bias = vb.get((out_dim,), "bias");
        let bias = if bias.is_ok() {
            let bs = bias.unwrap();
            let bs = if shard.world_size > 1 {
                let dim_size = bs.dim(0)?;
                let start = shard.rank * (dim_size / shard.world_size);
                bs.narrow(0, start, dim_size / shard.world_size)?
                    .contiguous()?
            } else {
                bs
            };
            Some(bs)
        } else {
            None
        };

        Ok(Self {
            weight,
            weight_scale,
            weight_scale_cutlass,
            bias,
            weight_block_size: block_size,
            sm_version,
        })
    }
}

fn load_ln_fp8_with_hints(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    quant_cfg: &QuantConfig,
    load_bias: bool,
) -> Result<LnFp8> {
    fn normalize_sharded_2d(
        t: Tensor,
        shard: Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            candle_core::bail!("LnFp8: unsupported shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "LnFp8: unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            )
        } else {
            let local = global_dim1 / shard.world_size;
            if d1 == local {
                return Ok(t);
            }
            if d1 == global_dim1 {
                return t.narrow(1, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "LnFp8: unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim1
            )
        }
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "LnFp8: unexpected {} shape ({}), expects local {} or global {}",
            name,
            d0,
            local,
            global_dim
        )
    }

    let block_size = quant_cfg
        .weight_block_size
        .clone()
        .unwrap_or(vec![128, 128]);
    if block_size.len() != 2 {
        candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
    }

    let by = block_size[0];
    let bx = block_size[1];
    let scale_dim0 = (out_dim + by - 1) / by;
    let scale_dim1 = (in_dim + bx - 1) / bx;

    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8)?;
    let weight = normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;
    let weight_scale = match vb.get_with_hints_dtype(
        (scale_dim0, scale_dim1),
        "weight_scale",
        shard,
        DType::F32,
    ) {
        Ok(s) => s,
        Err(_) => vb
            .get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            )
            .map_err(|_| {
                candle_core::Error::Msg("LnFp8: Missing weight_scale or weight_scale_inv".into())
            })?,
    };
    let weight_scale = normalize_sharded_2d(
        weight_scale,
        shard,
        scale_dim0,
        scale_dim1,
        "weight_scale(_inv)",
    )?;

    #[cfg(feature = "cuda")]
    let sm_version =
        attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?).unwrap_or(0) as usize;

    #[cfg(not(feature = "cuda"))]
    let sm_version = 0;

    #[cfg(feature = "cutlass")]
    let weight_scale_cutlass = if sm_version >= 100 {
        // SM100+: Column-major scale layout
        Some(weight_scale.t()?)
    } else if sm_version >= 90 {
        // SM90: CUTLASS expects scales_b as [K/128, N/128] row-major contiguous
        // Original weight_scale: [N/128, K/128] row-major
        // Transpose + contiguous gives [K/128, N/128] row-major
        Some(weight_scale.t()?.contiguous()?)
    } else {
        None
    };

    #[cfg(not(feature = "cutlass"))]
    let weight_scale_cutlass = None;

    let bias = if load_bias {
        vb.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32)
            .ok()
            .map(|b| normalize_sharded_1d(b, shard, out_dim, "bias"))
            .transpose()?
    } else {
        None
    };

    Ok(LnFp8 {
        weight,
        weight_scale,
        weight_scale_cutlass,
        bias,
        weight_block_size: block_size,
        sm_version,
    })
}

impl Module for LnFp8 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, in_dim) = match x.dims() {
            [b, s, d] => (*b, *s, *d),
            [b, d] => (*b, 1, *d),
            _ => candle_core::bail!("LnFp8: Input should be 2D or 3D"),
        };

        let x_2d = x.reshape((b_sz * seq_len, in_dim))?;

        let out = attention_rs::fp8_linear::fp8_matmul(
            &x_2d,
            &self.weight,
            &self.weight_scale,
            self.weight_scale_cutlass.as_ref(),
            &self.weight_block_size,
            linear_is_prefill(),
        )?;

        let (_, out_dim) = out.dims2()?;
        let out = if seq_len > 1 {
            out.reshape((b_sz, seq_len, out_dim))?
        } else {
            out
        };

        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

/// MXFP4 linear layer: packed FP4 E2M1 weights with E8M0 block scales.
#[derive(Debug, Clone)]
pub struct LnMxfp4 {
    pub blocks: Tensor,
    pub scales: Tensor,
    pub bias: Option<Tensor>,
}

impl LnMxfp4 {
    pub fn load(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        load_bias: bool,
    ) -> Result<Self> {
        let blocks = if vb.contains_tensor("weight_packed") {
            vb.get_with_hints_dtype((out_dim, in_dim / 2), "weight_packed", shard, DType::U8)?
        } else {
            vb.get_with_hints_dtype((out_dim, in_dim / 2), "blocks", shard, DType::U8)?
        };
        let scales = if vb.contains_tensor("weight_scale") {
            vb.get_with_hints_dtype((out_dim, in_dim / 32), "weight_scale", shard, DType::U8)?
        } else {
            vb.get_with_hints_dtype((out_dim, in_dim / 32), "scales", shard, DType::U8)?
        };
        let bias = if load_bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };
        Ok(Self {
            blocks,
            scales,
            bias,
        })
    }
}

impl Module for LnMxfp4 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let result = attention_rs::mxfp4_linear::mxfp4_matmul(
            &x_2d,
            &self.blocks,
            &self.scales,
            self.bias.as_ref(),
            linear_is_prefill(),
        )?;

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result.reshape(new_dims)
        } else {
            Ok(result)
        }
    }
}

/// NVFP4 linear layer: packed FP4 E2M1 weights with FP8 E4M3 block scales + F32 global scale.
///
/// Scale factors:
/// - `global_scale`: weight-side multiplier for the hardware FP4 path
///   (from `weight_scale_2` or `1/weight_global_scale`)
/// - `input_scale`: activation-side multiplier for the hardware FP4 path.
///   ModelOpt checkpoints store this directly as `input_scale`.
///   Compressed-tensors checkpoints store `input_global_scale` as a divisor, so
///   we invert it here to keep the hardware FP4 contract consistent.
///   For the software path (Hopper and below), this is ignored since activations
///   stay in FP16/BF16. When the checkpoint doesn't provide an activation scale,
///   defaults to 1.0.
#[derive(Debug, Clone)]
pub struct LnNvfp4 {
    pub blocks: Tensor,
    pub scales: Tensor,
    pub global_scale: f32,
    pub input_scale: f32,
    pub bias: Option<Tensor>,
    pub weight_scale_swizzled: Option<Tensor>,
}

impl LnNvfp4 {
    pub fn load(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        load_bias: bool,
    ) -> Result<Self> {
        Self::load_inner(in_dim, out_dim, vb, shard, load_bias, false)
    }

    /// Load with MLX NVFP4 format support. MLX stores weights as U32
    /// (8 FP4 nibbles per U32) which we reinterpret as U8 bytes at load time.
    pub fn load_mlx(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        load_bias: bool,
    ) -> Result<Self> {
        Self::load_inner(in_dim, out_dim, vb, shard, load_bias, true)
    }

    fn load_inner(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        load_bias: bool,
        is_mlx: bool,
    ) -> Result<Self> {
        let blocks = if is_mlx {
            // MLX stores weights as U32 [out_dim, in_dim/8] — 8 nibbles per U32.
            // Load as U32 on GPU, then repack to U8 [out_dim, in_dim/2] using a
            // GPU kernel (no CPU round-trip).
            let w_u32 =
                vb.get_with_hints_dtype((out_dim, in_dim / 8), "weight", shard, DType::U32)?;
            attention_rs::nvfp4_linear::mlx_repack_u32_to_u8(&w_u32)?
        } else if vb.contains_tensor("weight_packed") {
            vb.get_with_hints_dtype((out_dim, in_dim / 2), "weight_packed", shard, DType::U8)?
        } else if vb.contains_tensor("weight") {
            vb.get_with_hints_dtype((out_dim, in_dim / 2), "weight", shard, DType::U8)?
        } else {
            vb.get_with_hints_dtype((out_dim, in_dim / 2), "blocks", shard, DType::U8)?
        };

        let scale_dim = in_dim / 16;
        let scales = if vb.contains_tensor("weight_scale") {
            vb.get_with_hints_dtype((out_dim, scale_dim), "weight_scale", shard, DType::U8)?
        } else {
            vb.get_with_hints_dtype((out_dim, scale_dim), "scales", shard, DType::U8)?
        };

        let no_shard = Shard::default();
        let global_scale = if is_mlx {
            // MLX bakes the global scale into per-block FP8 E4M3 scales.
            1.0f32
        } else if vb.contains_tensor("weight_global_scale") {
            // compressed-tensors format: weight_global_scale is a divisor, invert it
            let t = match vb.get_with_hints_dtype((1,), "weight_global_scale", no_shard, DType::F32)
            {
                Ok(t) => t,
                Err(_) => {
                    vb.get_with_hints_dtype((), "weight_global_scale", no_shard, DType::F32)?
                }
            };
            let raw = t.flatten_all()?.to_vec1::<f32>()?[0];
            if raw != 0.0 {
                1.0 / raw
            } else {
                1.0
            }
        } else if vb.contains_tensor("weight_scale_2") {
            // modelopt format: weight_scale_2 is the direct multiplier
            let t = match vb.get_with_hints_dtype((1,), "weight_scale_2", no_shard, DType::F32) {
                Ok(t) => t,
                Err(_) => vb.get_with_hints_dtype((), "weight_scale_2", no_shard, DType::F32)?,
            };
            t.flatten_all()?.to_vec1::<f32>()?[0]
        } else {
            1.0f32
        };

        let input_scale = if is_mlx {
            1.0f32
        } else if vb.contains_tensor("input_scale") {
            // modelopt format: input_scale is a per-tensor activation scale
            let t = match vb.get_with_hints_dtype((1,), "input_scale", no_shard, DType::F32) {
                Ok(t) => t,
                Err(_) => vb.get_with_hints_dtype((), "input_scale", no_shard, DType::F32)?,
            };
            t.flatten_all()?.to_vec1::<f32>()?[0]
        } else if vb.contains_tensor("input_global_scale") {
            // compressed-tensors format: input_global_scale is a divisor, invert it
            let t = match vb.get_with_hints_dtype((1,), "input_global_scale", no_shard, DType::F32)
            {
                Ok(t) => t,
                Err(_) => {
                    vb.get_with_hints_dtype((), "input_global_scale", no_shard, DType::F32)?
                }
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

        let bias = if load_bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let weight_scale_swizzled = {
            let sm = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;
            if sm >= 100 {
                Some(attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                    &scales,
                )?)
            } else {
                None
            }
        };
        #[cfg(not(feature = "cuda"))]
        let weight_scale_swizzled: Option<Tensor> = None;

        Ok(Self {
            blocks,
            scales,
            global_scale,
            input_scale,
            bias,
            weight_scale_swizzled,
        })
    }
}

impl Module for LnNvfp4 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let result = attention_rs::nvfp4_linear::nvfp4_matmul(
            &x_2d,
            &self.blocks,
            &self.scales,
            self.global_scale,
            self.input_scale,
            self.bias.as_ref(),
            linear_is_prefill(),
            self.weight_scale_swizzled.as_ref(),
        )?;

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result.reshape(new_dims)
        } else {
            Ok(result)
        }
    }
}
