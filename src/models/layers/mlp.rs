use crate::models::layers::distributed::{
    shard, Comm, MergedParallelColumnLinear, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear,
};
use crate::models::layers::{collect_key_map, VarBuilderX};
use crate::utils::config::QuantConfig;
use candle_core::{DType, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::{Activation, Module};
use std::rc::Rc;

/// NVFP4 gate/up merged into a single fused GEMM. Gate and up may have
/// different weight global scales; `row_scales` carries the per-row (half)
/// scale so the kernel dequantizes each half correctly without re-quant.
struct Nvfp4MergedWeights {
    blocks: Tensor,
    scales: Tensor,
    gate_gscale: f32,
    input_scale: f32,
    row_scales: Tensor,
    weight_scale_swizzled: Option<Tensor>,
    n: usize,
}

enum GateUpProjection {
    Separate {
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
    Nvfp4Merged(Nvfp4MergedWeights),
}

pub struct MLP {
    gate_up_proj: GateUpProjection,
    down_proj: TensorParallelRowLinear,
    activation: Activation,
    swiglu_limit: f32,
}

impl MLP {
    /// Merge two independently-loaded NVFP4 gate/up projections into a single
    /// fused GEMM (sglang-style). Byte-concatenates the FP4 blocks and E4M3
    /// block scales and carries the per-half weight global scales in
    /// `row_scales` so the kernel applies the correct scale to each half. No
    /// re-quantization, so the original NVFP4 weights are preserved exactly.
    fn merge_nvfp4_gate_up(
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
        gate_up_merged: bool,
    ) -> GateUpProjection {
        if gate_up_merged {
            return GateUpProjection::Separate { gate_proj, up_proj };
        }
        // Debug aid: force the separate gate/up path to build a precision
        // baseline against the fused NVFP4 merge. No effect unless set.
        if std::env::var("XINFER_DISABLE_NVFP4_GATEUP_MERGE").is_ok() {
            return GateUpProjection::Separate { gate_proj, up_proj };
        }
        match (gate_proj.as_nvfp4(), up_proj.as_nvfp4()) {
            (Some(g), Some(u)) => {
                let n = g.blocks.dim(0).unwrap_or(0);
                let same_shape = match (g.blocks.dims(), u.blocks.dims()) {
                    (gd, ud) => gd.len() == 2 && gd == ud,
                };
                if n == 0 || !same_shape {
                    return GateUpProjection::Separate { gate_proj, up_proj };
                }
                let blocks = match Tensor::cat(&[&g.blocks, &u.blocks], 0) {
                    Ok(b) => b,
                    Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                let scales = match Tensor::cat(&[&g.scales, &u.scales], 0) {
                    Ok(s) => s,
                    Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                let dev = blocks.device().clone();
                let (row_scales, swizzled) = match (
                    Tensor::full(g.global_scale, (n,), &dev),
                    Tensor::full(u.global_scale, (n,), &dev),
                ) {
                    (Ok(gr), Ok(ur)) => match Tensor::cat(&[&gr, &ur], 0) {
                        Ok(rs) => {
                            let swizzled = {
                                #[cfg(feature = "cuda")]
                                {
                                    let sm = match scales.device().as_cuda_device() {
                                        Ok(dev) => attention_rs::cuda_utils::sm_version(dev)
                                            .unwrap_or(0)
                                            as usize,
                                        Err(_) => 0,
                                    };
                                    if sm >= 100 {
                                        attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                                            &scales,
                                        )
                                        .ok()
                                    } else {
                                        None
                                    }
                                }
                                #[cfg(not(feature = "cuda"))]
                                None
                            };
                            (rs, swizzled)
                        }
                        Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                    },
                    _ => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                GateUpProjection::Nvfp4Merged(Nvfp4MergedWeights {
                    blocks,
                    scales,
                    gate_gscale: g.global_scale,
                    input_scale: g.input_scale.max(u.input_scale),
                    row_scales,
                    weight_scale_swizzled: swizzled,
                    n,
                })
            }
            _ => GateUpProjection::Separate { gate_proj, up_proj },
        }
    }

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
            candle_core::bail!("unexpected shard dim {} for {}", shard.dim, name);
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
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            );
        }

        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name,
            d0,
            d1,
            local,
            global_dim1
        );
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilderX,
        out_dim: usize,
        in_dim: usize,
        shard: Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.has_key("weight_scale") && !vb.has_key("weight_scale_inv") {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb
            .get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::F8E4M3)
            .or_else(|_| vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8))
        {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;
        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(scale) => scale,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(scale) => scale,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;
        Ok(Some((weight, weight_scale)))
    }

    fn try_load_packed_gate_up(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        hidden_size: usize,
        intermediate_size: usize,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        gate_up_merged: bool,
        dtype: DType,
    ) -> Result<Option<GateUpProjection>> {
        if vb.is_qvar_builder() || quant.is_some() {
            return Ok(None);
        }

        let is_fp8_quant = quant_cfg
            .as_ref()
            .map(|cfg| cfg.quant_method == "fp8")
            .unwrap_or(false);
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        let gate_shard = if gate_up_merged {
            shard(0, comm.rank(), comm.world_size() * 2)
        } else {
            shard(0, comm.rank(), comm.world_size())
        };
        let up_shard = if gate_up_merged {
            shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2)
        } else {
            shard(0, comm.rank(), comm.world_size())
        };

        if gate_up_merged {
            let gate_up_vb = vb.pp("gate_up_proj");
            if is_fp8_quant {
                let Some(block_size) = quant_cfg
                    .as_ref()
                    .and_then(|cfg| cfg.weight_block_size.clone())
                else {
                    candle_core::bail!(
                        "LnFp8: weight_block_size must be configured for packed gate_up"
                    );
                };
                if block_size.len() != 2 {
                    candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
                }
                let by = block_size[0];
                let total_out = intermediate_size * 2;
                let Some((gate_weight, gate_scale)) = Self::try_load_sharded_fp8_weight_scale(
                    &gate_up_vb,
                    total_out,
                    hidden_size,
                    gate_shard,
                    &block_size,
                )?
                else {
                    return Ok(None);
                };
                let Some((up_weight, up_scale)) = Self::try_load_sharded_fp8_weight_scale(
                    &gate_up_vb,
                    total_out,
                    hidden_size,
                    up_shard,
                    &block_size,
                )?
                else {
                    return Ok(None);
                };
                let local_gate = gate_weight.dim(0)?;
                let local_up = up_weight.dim(0)?;
                let gate_start = gate_shard.rank * local_gate;
                let up_start = up_shard.rank * local_up;
                if gate_start % by != 0 || up_start % by != 0 {
                    return Ok(None);
                }
                let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
                let packed_scale = Tensor::cat(&[&gate_scale, &up_scale], 0)?;
                #[cfg(feature = "cuda")]
                let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                    .unwrap_or(0) as usize;
                #[cfg(not(feature = "cuda"))]
                let sm_version = 0;
                let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                    packed_weight,
                    packed_scale,
                    None,
                    block_size,
                    sm_version,
                    vec![local_gate, local_up],
                )?;
                return Ok(Some(GateUpProjection::Packed(merged)));
            }

            if quant_cfg.is_some() {
                return Ok(None);
            }
            let total_out = intermediate_size * 2;
            let gate_weight = gate_up_vb.get_with_hints_dtype(
                (total_out, hidden_size),
                "weight",
                gate_shard,
                dtype,
            )?;
            let up_weight = gate_up_vb.get_with_hints_dtype(
                (total_out, hidden_size),
                "weight",
                up_shard,
                dtype,
            )?;
            let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
            let merged = MergedParallelColumnLinear::from_packed_local(
                packed_weight,
                None,
                vec![gate_weight.dim(0)?, up_weight.dim(0)?],
                &None,
            )?;
            return Ok(Some(GateUpProjection::Packed(merged)));
        }

        let gate_vb = vb.pp("gate_proj");
        let up_vb = vb.pp("up_proj");
        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                candle_core::bail!(
                    "LnFp8: weight_block_size must be configured for packed gate/up"
                );
            };
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }
            let by = block_size[0];
            let Some((gate_weight, gate_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &gate_vb,
                intermediate_size,
                hidden_size,
                gate_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((up_weight, up_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &up_vb,
                intermediate_size,
                hidden_size,
                up_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let local_gate = gate_weight.dim(0)?;
            let local_up = up_weight.dim(0)?;
            let gate_start = gate_shard.rank * local_gate;
            let up_start = up_shard.rank * local_up;
            if gate_start % by != 0 || up_start % by != 0 {
                return Ok(None);
            }
            let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
            let packed_scale = Tensor::cat(&[&gate_scale, &up_scale], 0)?;
            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;
            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;
            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                None,
                block_size,
                sm_version,
                vec![local_gate, local_up],
            )?;
            return Ok(Some(GateUpProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let gate_weight = gate_vb.get_with_hints_dtype(
            (intermediate_size, hidden_size),
            "weight",
            gate_shard,
            dtype,
        )?;
        let up_weight = up_vb.get_with_hints_dtype(
            (intermediate_size, hidden_size),
            "weight",
            up_shard,
            dtype,
        )?;
        let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            None,
            vec![gate_weight.dim(0)?, up_weight.dim(0)?],
            &None,
        )?;
        Ok(Some(GateUpProjection::Packed(merged)))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        hidden_size: usize,
        intermediate_size: usize,
        activation: &Activation,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        gate_up_merged: bool,
        dtype: DType,
        suffix: &str,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let key_map = collect_key_map(
            is_qvar_builder,
            [
                ("gate_proj", "ffn_gate"),
                ("up_proj", "ffn_up"),
                ("gate_up_proj", "ffn_up"),
                ("down_proj", "ffn_down"),
            ],
        );
        // Detect w1/w2/w3 naming (BF16 weight or FP8 weight+scale)
        let gate_key = if vb.pp("w1").has_key("weight") || vb.pp("w1").has_key("scale") {
            "w1"
        } else {
            "gate_proj"
        };
        let up_key = if vb.pp("w3").has_key("weight") || vb.pp("w3").has_key("scale") {
            "w3"
        } else {
            "up_proj"
        };
        let down_key = if vb.pp("w2").has_key("weight") || vb.pp("w2").has_key("scale") {
            "w2"
        } else {
            "down_proj"
        };
        let gate_up_proj = if let Some(packed) = Self::try_load_packed_gate_up(
            &vb,
            comm.clone(),
            hidden_size,
            intermediate_size,
            quant_cfg,
            quant,
            gate_up_merged,
            dtype,
        )? {
            packed
        } else {
            let gate_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                if gate_up_merged {
                    intermediate_size * 2
                } else {
                    intermediate_size
                },
                false,
                if is_qvar_builder {
                    vb.pp((key_map[if gate_up_merged {
                        "up_proj"
                    } else {
                        "gate_proj"
                    }]
                    .to_string()
                        + suffix)
                        .as_str())
                } else {
                    vb.pp(if gate_up_merged {
                        "gate_up_proj"
                    } else {
                        gate_key
                    })
                },
                if gate_up_merged {
                    shard(0, comm.rank(), comm.world_size() * 2)
                } else {
                    shard(0, comm.rank(), comm.world_size())
                },
                quant_cfg,
                quant,
                dtype,
            )?;

            let up_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                if gate_up_merged {
                    intermediate_size * 2
                } else {
                    intermediate_size
                },
                false,
                if is_qvar_builder {
                    vb.pp((key_map["up_proj"].to_string() + suffix).as_str())
                } else {
                    vb.pp(if gate_up_merged {
                        "gate_up_proj"
                    } else {
                        up_key
                    })
                },
                if gate_up_merged {
                    shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2)
                } else {
                    shard(0, comm.rank(), comm.world_size())
                },
                quant_cfg,
                quant,
                dtype,
            )?;
            Self::merge_nvfp4_gate_up(gate_proj, up_proj, gate_up_merged)
        };

        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_size,
            if is_qvar_builder {
                vb.pp((key_map["down_proj"].to_string() + suffix).as_str())
            } else {
                vb.pp(down_key)
            },
            comm.clone(),
            quant_cfg,
            quant,
            dtype,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            activation: activation.clone(),
            swiglu_limit: 0.0,
        })
    }

    pub fn with_swiglu_limit(mut self, limit: f32) -> Self {
        self.swiglu_limit = limit;
        self
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (gate, up) = match &self.gate_up_proj {
            GateUpProjection::Separate { gate_proj, up_proj } => {
                (gate_proj.forward(xs)?, up_proj.forward(xs)?)
            }
            GateUpProjection::Packed(gate_up_proj) => {
                let gate_up = gate_up_proj.forward(xs)?;
                if gate_up.len() != 2 {
                    candle_core::bail!(
                        "Expected 2 outputs from packed gate/up projection, got {}",
                        gate_up.len()
                    );
                }
                (gate_up[0].clone(), gate_up[1].clone())
            }
            GateUpProjection::Nvfp4Merged(m) => {
                let orig_dims = xs.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    xs.reshape((batch_size, features))?
                } else {
                    xs.clone()
                };
                let gate_up = attention_rs::nvfp4_linear::nvfp4_matmul(
                    &x_2d,
                    &m.blocks,
                    &m.scales,
                    m.gate_gscale,
                    m.input_scale,
                    None,
                    crate::models::layers::linear::linear_is_prefill(),
                    m.weight_scale_swizzled.as_ref(),
                    Some(&m.row_scales),
                )?;
                let leading: Vec<usize> = if orig_dims.len() > 2 {
                    orig_dims[..orig_dims.len() - 1].to_vec()
                } else {
                    vec![x_2d.dim(0)?]
                };
                let mut gate_shape = leading.clone();
                gate_shape.push(m.n);
                let gate = gate_up.narrow(1, 0, m.n)?.reshape(gate_shape.clone())?;
                let up = gate_up.narrow(1, m.n, m.n)?.reshape(gate_shape)?;
                (gate, up)
            }
        };
        let activated = if self.swiglu_limit > 0.0 && matches!(self.activation, Activation::Silu) {
            // DeepSeek V4's reference Expert promotes both projections to
            // FP32, clamps asymmetrically (gate <= limit, up in [-limit,
            // limit]), and only casts back immediately before w2.  Doing the
            // clamp and SiLU in BF16 changes routed/shared expert outputs
            // enough to perturb later HC and router values.
            let gate = gate.to_dtype(DType::F32)?.minimum(self.swiglu_limit)?;
            let up = up
                .to_dtype(DType::F32)?
                .clamp(-self.swiglu_limit, self.swiglu_limit)?;
            let activated = (candle_nn::ops::silu(&gate)? * up)?;
            activated.to_dtype(xs.dtype())?
        } else {
            (self.activation.forward(&gate)? * up)?
        };
        self.down_proj.forward(&activated)
    }
}

pub struct NaiveMLP {
    fc1: ReplicatedLinear,
    fc2: ReplicatedLinear,
    act: Activation,
}

impl NaiveMLP {
    pub fn new(
        vb: VarBuilderX,
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        names: &[&str],
        hidden_act: Activation,
        dtype: DType,
    ) -> Result<Self> {
        let fc1 = ReplicatedLinear::load_b(
            hidden_size,
            intermediate_size,
            bias,
            vb.pp(names[0]),
            &None,
            &None,
            dtype,
        )?;

        let fc2 = ReplicatedLinear::load_b(
            intermediate_size,
            hidden_size,
            bias,
            vb.pp(names[1]),
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            fc1,
            fc2,
            act: hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.fc1.forward(xs)?;
        let down = self.act.forward(&gate_up)?;
        self.fc2.forward(&down)
    }
}
