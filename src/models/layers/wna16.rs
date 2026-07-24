//! Linear layer (WNA16: GPTQ, AWQ)
use crate::models::layers::linear::shard;
use crate::utils::config::QuantConfig;
use crate::utils::gptq::{gptq_matmul, marlin_weight_repack};
use candle_core::{DType, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WNA16 {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub scales: Option<Tensor>,
    pub qzeros: Option<Tensor>,
    pub g_idx: Option<Tensor>,
    pub workspace: Option<Tensor>,
    group_size: i32,
    bits: i32,
    dtype: DType,
    is_awq: bool,
}

impl WNA16 {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shards: Shard,
        quant_config: &Option<QuantConfig>,
        bias: bool,
        dtype: DType,
        may_use_marlin: bool,
    ) -> Result<WNA16> {
        let mut shards = shards.clone();
        shards.dim = if shards.world_size < 2 || shards.dim == 1 {
            0
        } else {
            1
        };

        let ln = if let Some(cfg) = quant_config {
            let mut marlin_compatible = if (cfg.quant_method != "gptq" && cfg.quant_method != "awq")
                || (cfg.bits != 4 && cfg.bits != 8)
            {
                false
            } else {
                true
            };
            let marlin_format = cfg.checkpoint_format.is_some()
                && cfg.checkpoint_format.as_ref().unwrap() == "marlin";

            if !may_use_marlin {
                marlin_compatible = false;
            }
            let ws = vb.get_with_hints_dtype(
                if cfg.quant_method == "gptq" {
                    //quantized gptq (k/pack_factor, n) format
                    (
                        in_dim / (32 / cfg.bits) / if marlin_format { 2 } else { 1 },
                        out_dim * if marlin_format { 2 } else { 1 },
                    )
                } else {
                    //quantized awq (k, n/pack_factor) format
                    (
                        in_dim * if marlin_format { 2 } else { 1 },
                        out_dim / (32 / cfg.bits) / if marlin_format { 2 } else { 1 },
                    )
                },
                if marlin_format { "B" } else { "qweight" },
                shards,
                DType::U32,
            )?;

            let scale_and_zero_size = in_dim / (cfg.group_size as usize);
            // Marlin/GPTQ kernels consume scales in the activation dtype. Load
            // directly as `dtype` (do not force F16 then cast — that truncates
            // F16→BF16 on SM80+ and can discard F32 checkpoint precision when
            // upcasting is possible via a future F32 path).
            let scales = vb.get_with_hints_dtype(
                (scale_and_zero_size, out_dim),
                if marlin_format { "s" } else { "scales" },
                shards,
                dtype,
            )?;

            let in_dim_partition = if shards.dim == 0 {
                in_dim / shards.world_size
            } else {
                in_dim
            };

            let out_dim_partition = if shards.dim == 1 {
                out_dim / shards.world_size
            } else {
                out_dim
            };

            let bs = if bias {
                let bs = vb
                    .get_with_hints_dtype(
                        (out_dim,),
                        "bias",
                        shard(0, shards.rank, shards.world_size),
                        DType::F16,
                    )?
                    .to_dtype(dtype)?;
                Some(bs)
            } else {
                None
            };

            if marlin_format {
                let workspace = Tensor::zeros(out_dim_partition, DType::U32, vb.device())?;
                //marlin weight file
                Ok(WNA16 {
                    weight: ws,
                    bias: bs,
                    scales: Some(scales),
                    qzeros: None,
                    g_idx: None,
                    workspace: Some(workspace),
                    group_size: cfg.group_size,
                    bits: cfg.bits as i32,
                    dtype,
                    is_awq: cfg.quant_method == "awq",
                })
            } else {
                let qzeros = vb.get_with_hints_dtype(
                    (scale_and_zero_size, out_dim / (32 / cfg.bits)),
                    "qzeros",
                    shards,
                    DType::U32,
                )?;
                let g_idx = if cfg.quant_method == "gptq" {
                    let mut g_idx = vb.get_with_hints_dtype(
                        (in_dim,),
                        "g_idx",
                        Default::default(),
                        DType::U32,
                    )?;
                    g_idx = if shards.world_size > 1 {
                        let dim_size = g_idx.dims()[0];
                        let start = shards.rank * (dim_size / shards.world_size);
                        g_idx
                            .narrow(0, start, dim_size / shards.world_size)?
                            .contiguous()?
                    } else {
                        g_idx
                    };
                    Some(g_idx)
                } else {
                    None
                };

                if (cfg.sym.is_some() && !cfg.sym.unwrap())
                    || cfg.bits != 4
                    || !matches!(cfg.group_size, 64 | 128 | -1)
                    || (cfg.desc_act.is_some()
                        && cfg.desc_act.unwrap()
                        && cfg.quant_method == "gptq")
                {
                    //only model with 4-bit and desc_act==false can be repacked to marlin format
                    if cfg.quant_method == "marlin" {
                        crate::log_warn!("The current GPTQ model does no compatible with marlin format because one of the following conditions: !cfg.sym || cfg.bits != 4 || (cfg.group_size != 128 && cfg.group_size != -1) || (cfg.desc_act == true)");
                    }
                    //conventional gptq format
                    Ok(WNA16 {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx,
                        workspace: None,
                        group_size: cfg.group_size,
                        bits: cfg.bits as i32,
                        dtype,
                        is_awq: cfg.quant_method == "awq",
                    })
                } else {
                    //repack gptq format to marlin
                    fn get_scale_perms() -> (Vec<u32>, Vec<u32>) {
                        let mut scale_perm: Vec<u32> = Vec::new();
                        for i in 0..8 {
                            scale_perm.extend((0..8).map(|j| i + 8 * j));
                        }
                        let mut scale_perm_single: Vec<u32> = Vec::new();
                        for i in 0..4 {
                            scale_perm_single
                                .extend([0, 1, 8, 9, 16, 17, 24, 25].iter().map(|&j| 2 * i + j));
                        }
                        (scale_perm, scale_perm_single)
                    }

                    fn marlin_permute_scales(
                        s: &Tensor,
                        size_k: usize,
                        size_n: usize,
                        group_size: i32,
                        _num_bits: u32,
                    ) -> Result<Tensor> {
                        let (scale_perm, scale_perm_single) = get_scale_perms();
                        let s = if (group_size as usize) < size_k && group_size != -1 {
                            let s = s.reshape(((), scale_perm.len()))?;
                            let scale_perm_tensor =
                                Tensor::from_slice(&scale_perm, scale_perm.len(), s.device())?;
                            s.index_select(&scale_perm_tensor, 1)?
                        } else {
                            let s = s.reshape(((), scale_perm_single.len()))?;
                            let scale_perm_single_tensor = Tensor::from_slice(
                                &scale_perm_single,
                                scale_perm_single.len(),
                                s.device(),
                            )?;
                            s.index_select(&scale_perm_single_tensor, 1)?
                        };

                        let s = s.reshape(((), size_n))?.contiguous()?;
                        Ok(s)
                    }

                    let ws = if marlin_compatible {
                        marlin_weight_repack(&ws, cfg.bits as i32, cfg.quant_method != "gptq")?
                    } else {
                        ws
                    }; //repack to marlin format

                    let scales = if marlin_compatible {
                        marlin_permute_scales(
                            &scales,
                            in_dim_partition,
                            out_dim_partition,
                            cfg.group_size,
                            cfg.bits as u32,
                        )?
                    } else {
                        scales
                    };

                    let workspace = if marlin_compatible {
                        Some(Tensor::zeros(out_dim_partition, DType::U32, vb.device())?)
                    } else {
                        None
                    };
                    Ok(WNA16 {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx,
                        workspace,
                        group_size: cfg.group_size,
                        bits: cfg.bits as i32,
                        dtype,
                        is_awq: cfg.quant_method == "awq",
                    })
                }
            }
        } else {
            panic!("")
        };
        ln
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match (&self.scales, &self.qzeros, &self.g_idx, &self.workspace) {
            (Some(scale), qzeros, g_idx, workspace) => {
                let x = match *x.dims() {
                    [_, _, _] => gptq_matmul(
                        x,
                        &self.weight,
                        scale,
                        qzeros,
                        g_idx,
                        workspace,
                        self.bits,
                        self.group_size,
                        self.is_awq,
                    )?,
                    [seq_len, dim] => {
                        let x = x.reshape((1, seq_len, dim))?;
                        let o = gptq_matmul(
                            &x,
                            &self.weight,
                            scale,
                            qzeros,
                            g_idx,
                            workspace,
                            self.bits,
                            self.group_size,
                            self.is_awq,
                        )?;
                        o.reshape((seq_len, ()))?
                    }
                    _ => panic!("Invalid input format!"),
                };

                if let Some(bias) = &self.bias {
                    x.broadcast_add(bias)
                } else {
                    Ok(x)
                }
            }
            _ => {
                candle_core::bail!("Invalid arguments for gptq/awq matmul")
            }
        }
    }
}
