use crate::models::layers::VarBuilderX;
use candle_core::{DType, IndexOp, Result, Tensor, WithDType};
use candle_nn::{var_builder::Shard, Module};
use candle_nn::{Embedding, LayerNorm, RmsNorm};
use either::Either;

pub struct NormX {
    norm: Either<RmsNorm, LayerNorm>,
    /// When set, forward uses DeepSeek-V4 ATen-order RMSNorm CUDA kernel
    /// (`rms_norm_v4`) instead of Candle's generic reduction. V4's 86 HC
    /// updates amplify last-bit F32 differences from the mean reduction.
    v4_weight: Option<Tensor>,
    v4_eps: f32,
    dtype: DType,
}
impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(weight) = &self.v4_weight {
            return self.forward_v4(xs, weight);
        }
        let in_dtype = xs.dtype();
        if xs.dtype() != self.dtype {
            let converted = xs.to_dtype(self.dtype)?;
            let out = match &self.norm {
                Either::Left(norm) => norm.forward(&converted)?,
                Either::Right(norm) => norm.forward(&converted)?,
            };
            out.to_dtype(in_dtype)
        } else {
            let out = match &self.norm {
                Either::Left(norm) => norm.forward(xs)?,
                Either::Right(norm) => norm.forward(xs)?,
            };
            Ok(out)
        }
    }

    fn forward_v4(&self, xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let dims = xs.dims().to_vec();
        let dim = *dims
            .last()
            .ok_or_else(|| candle_core::Error::Msg("NormX V4: expected non-empty dims".into()))?;
        let rows: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        let rows = if dims.len() == 1 { 1 } else { rows.max(1) };
        // Kernel expects BF16 activations and F32 weights.
        let x_bf16 = if xs.dtype() == DType::BF16 {
            xs.clone()
        } else {
            xs.to_dtype(DType::BF16)?
        };
        let x_2d = if dims.len() == 2 && dims[0] == rows && dims[1] == dim {
            x_bf16
        } else {
            x_bf16.reshape((rows, dim))?
        };
        let weight_f32 = if weight.dtype() == DType::F32 {
            weight.clone()
        } else {
            weight.to_dtype(DType::F32)?
        };
        let out = attention_rs::deepseek_v4::rms_norm_v4(&x_2d, &weight_f32, dim, self.v4_eps)?;
        let out = if dims.len() == 2 && dims[0] == rows && dims[1] == dim {
            out
        } else {
            out.reshape(dims)?
        };
        if in_dtype == DType::BF16 {
            Ok(out)
        } else {
            out.to_dtype(in_dtype)
        }
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
) -> Result<NormX> {
    rms_norm_sharded(size, eps, vb, dtype, is_gemma, Shard::default())
}

/// DeepSeek-V4 RMSNorm with ATen (128,4) mean-reduction order.
pub fn rms_norm_v4(size: usize, eps: f64, vb: VarBuilderX, dtype: DType) -> Result<NormX> {
    rms_norm_v4_sharded(size, eps, vb, dtype, Shard::default())
}

pub fn rms_norm_v4_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    shard: Shard,
) -> Result<NormX> {
    let (weight, dtype) = match &vb.0 {
        Either::Left(vb) => {
            let ws = vb.get_with_hints(size, "weight", shard)?;
            if ws.dtype() != dtype {
                (ws.to_dtype(dtype)?, dtype)
            } else {
                (ws, dtype)
            }
        }
        Either::Right(vb) => (vb.get(size, "weight")?.dequantize(vb.device())?, DType::F32),
    };
    let weight_f32 = weight.to_dtype(DType::F32)?;
    Ok(NormX {
        // Keep a candle RmsNorm as unused fallback placeholder so Either stays populated.
        norm: Either::Left(RmsNorm::new(weight_f32.clone(), eps)),
        v4_weight: Some(weight_f32),
        v4_eps: eps as f32,
        dtype,
    })
}

pub fn rms_norm_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
    shard: Shard,
) -> Result<NormX> {
    let (weight, dtype) = match &vb.0 {
        Either::Left(vb) => {
            let ws = vb.get_with_hints(size, "weight", shard)?;
            if ws.dtype() != dtype {
                (ws.to_dtype(dtype)?, dtype)
            } else {
                (ws, dtype)
            }
        }
        Either::Right(vb) => (vb.get(size, "weight")?.dequantize(vb.device())?, DType::F32),
    };

    let weight = if is_gemma { (weight + 1.0)? } else { weight };
    Ok(NormX {
        norm: Either::Left(RmsNorm::new(weight, eps)),
        v4_weight: None,
        v4_eps: eps as f32,
        dtype,
    })
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<NormX> {
    let (weight, dtype) = match &vb.0 {
        Either::Left(vb) => (
            vb.get_with_hints(size, "weight", Shard::default())?
                .to_dtype(dtype)?,
            dtype,
        ),
        Either::Right(vb) => (vb.get(size, "weight")?.dequantize(vb.device())?, DType::F32),
    };
    if affine {
        let bias = match &vb.0 {
            Either::Left(vb) => vb.get(size, "bias")?.to_dtype(dtype)?,
            Either::Right(vb) => vb.get(size, "bias")?.dequantize(vb.device())?,
        };
        Ok(NormX {
            norm: Either::Right(LayerNorm::new(weight, bias, eps)),
            v4_weight: None,
            v4_eps: eps as f32,
            dtype,
        })
    } else {
        Ok(NormX {
            norm: Either::Right(LayerNorm::new_no_bias(weight, eps)),
            v4_weight: None,
            v4_eps: eps as f32,
            dtype,
        })
    }
}

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<(Embedding, usize)> {
    let (embeddings, vocab_size) = match &vb.0 {
        Either::Left(vb) => {
            assert!(
                vocab_size.is_some(),
                "vocab_size must be specified for safetensor models"
            );
            let vs = vocab_size.unwrap();
            if vb.contains_tensor("scales") {
                // MLX NVFP4: quantized embedding with U32 weights + U8 FP8 E4M3 scales.
                // Dequantize at load time since embeddings are looked up, not matmul'd.
                let emb = dequantize_mlx_nvfp4_embedding(vb, vs, hidden_size, dtype)?;
                (emb, vs)
            } else {
                (vb.get((vs, hidden_size), "weight")?.to_dtype(dtype)?, vs)
            }
        }
        Either::Right(vb) => {
            let weight = if vocab_size.is_some() {
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?
            } else {
                vb.get_no_shape("weight")?
            }
            .dequantize(vb.device())?;
            let vocab_size = vocab_size.unwrap_or(weight.dim(0)?);
            (weight, vocab_size)
        }
    };
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}

fn dequantize_mlx_nvfp4_embedding(
    vb: &candle_nn::var_builder::ShardedVarBuilder,
    vocab_size: usize,
    hidden_size: usize,
    dtype: DType,
) -> Result<Tensor> {
    use candle_nn::var_builder::Shard;
    let no_shard = Shard::default();
    let w_u32 = vb.get_with_hints_dtype(
        (vocab_size, hidden_size / 8),
        "weight",
        no_shard,
        DType::U32,
    )?;
    let scales = vb.get_with_hints_dtype(
        (vocab_size, hidden_size / 16),
        "scales",
        no_shard,
        DType::U8,
    )?;

    let out_dtype = match dtype {
        DType::F16 | DType::BF16 => dtype,
        _ => DType::BF16,
    };
    attention_rs::nvfp4_linear::mlx_dequant_embedding(
        &w_u32,
        &scales,
        vocab_size,
        hidden_size,
        out_dtype,
    )
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilderX,
    bias: bool,
) -> Result<candle_nn::Conv2d> {
    let (ws, bs) = match vb.0 {
        Either::Left(v) => {
            let ws = v.get(
                (
                    out_channels,
                    in_channels / cfg.groups,
                    kernel_size,
                    kernel_size,
                ),
                "weight",
            )?;
            let bs = if bias {
                Some(v.get(out_channels, "bias")?)
            } else {
                None
            };
            (ws, bs)
        }
        _ => {
            todo!()
        }
    };

    Ok(candle_nn::Conv2d::new(ws, bs, cfg))
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: candle_nn::Conv2d,
    conv2d_2: candle_nn::Conv2d,
}

impl Conv3dNoBias {
    pub fn from_conv2d_weights(
        w1: Tensor,
        w2: Tensor,
        cfg: candle_nn::Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            conv2d_1: candle_nn::Conv2d::new(w1, None, cfg),
            conv2d_2: candle_nn::Conv2d::new(w2, None, cfg),
        })
    }

    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilderX,
    ) -> Result<Self> {
        use candle_nn::Conv2dConfig;
        let expected_shape = (
            out_channels,
            in_channels / cfg.groups,
            kernel_sizes[0],
            kernel_sizes[1],
            kernel_sizes[2],
        );
        let ws = match vb.0 {
            Either::Left(v) => {
                match v.get(expected_shape, "weight") {
                    Ok(w) => w,
                    Err(_) => {
                        // MLX stores conv weights as (O, T, H, W, C) instead of (O, C, T, H, W).
                        let mlx_shape = (
                            out_channels,
                            kernel_sizes[0],
                            kernel_sizes[1],
                            kernel_sizes[2],
                            in_channels / cfg.groups,
                        );
                        let w = v.get(mlx_shape, "weight")?;
                        w.permute((0, 4, 1, 2, 3))?
                    }
                }
            }
            _ => {
                panic!("Unsupported quantized format for conv3d")
            }
        };

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        Ok(Self {
            conv2d_1: candle_nn::Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: candle_nn::Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }

    pub fn weight(&self) -> Result<Tensor> {
        let w1 = self.conv2d_1.weight().clone().unsqueeze(2)?;
        let w2 = self.conv2d_2.weight().clone().unsqueeze(2)?;
        Tensor::cat(&[w1, w2], 2)
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;

        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}

pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}
