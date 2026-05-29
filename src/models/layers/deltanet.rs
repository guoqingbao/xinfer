// src/models/layers/deltanet.rs
// Shared Qwen3.5/Qwen3Next GatedDeltaNet linear-attention layer.

use crate::models::layers::distributed::{
    load_restored_gguf_column_linear, load_restored_gguf_merged_qkv_linear,
    load_restored_gguf_row_linear, shard, tensor_parallel_chunk, Comm, MergedParallelColumnLinear,
    TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::{collect_key_map, VarBuilderX};
use crate::utils::config::Config;
use crate::utils::gguf_helper::{
    restore_qwen35_a_log_from_gguf, restore_qwen35_qkv_weight, undo_tiled_v_heads_first_dim,
    undo_tiled_v_heads_last_dim,
};
use crate::utils::resolve_qwen3_hybrid_config;
use attention_rs::gdn;
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor};
use candle_nn::var_builder::Shard;
use std::rc::Rc;

enum GdnProjection {
    // Qwen3Next: in_proj_qkvz + in_proj_ba
    FusedQkvzBa {
        in_proj_qkvz: TensorParallelColumnLinear,
        in_proj_ba: TensorParallelColumnLinear,
    },
    // Qwen3.5: in_proj_qkv + in_proj_z + in_proj_ba + in_proj_a
    SplitQkvZaLegacy {
        in_proj_qkv: TensorParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
    // Qwen3.5 TP-safe split for packed in_proj_qkv [q|k|v].
    SplitQkvZaMerged {
        in_proj_qkv: MergedParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
}

pub struct GatedDeltaNet {
    projection: GdnProjection,
    out_proj: TensorParallelRowLinear,
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
    scale: f64,
    kernel_dtype: DType,
    /// The dtype that projection weights were loaded in (e.g. F16/BF16).
    /// Used to cast input before projection when norms output a different dtype (e.g. F32).
    projection_dtype: DType,
}

impl GatedDeltaNet {
    /// Check if a weight at the given VarBuilder path actually carries quantized data.
    /// Returns false when the weight is stored in its original dtype (BF16/F16/F32)
    /// even though the model-level quantization config is set.
    fn is_weight_quantized(vb: &VarBuilderX, quant_method: &str) -> bool {
        if vb.is_qvar_builder() {
            return false;
        }
        match quant_method {
            "fp8" => vb.has_key("weight_scale") || vb.has_key("weight_scale_inv"),
            "mxfp4" => vb.has_key("weight_packed") || vb.has_key("blocks"),
            "nvfp4" => {
                let has_packed = vb.has_key("weight_packed") || vb.has_key("blocks");
                let has_scale = vb.has_key("weight_scale") || vb.has_key("scales");
                let has_modelopt = vb.has_key("weight_scale_2") || vb.has_key("input_scale");
                (has_packed && has_scale) || (has_modelopt && has_scale)
            }
            "gptq" | "awq" => vb.has_key("qweight") || vb.has_key("B"),
            _ => true,
        }
    }

    /// Resolve effective quantization config for a specific weight.
    /// If the weight is not actually quantized, returns (None, None) so
    /// the loader falls back to the standard unquantized path.
    fn resolve_quant_for_weight(
        vb: &VarBuilderX,
        quantization_config: &Option<crate::utils::config::QuantConfig>,
        quant: &Option<String>,
    ) -> (Option<crate::utils::config::QuantConfig>, Option<String>) {
        if let Some(cfg) = quantization_config {
            if Self::is_weight_quantized(vb, &cfg.quant_method) {
                return (quantization_config.clone(), quant.clone());
            }
        }
        (None, None)
    }

    fn load_projection(
        vb: &VarBuilderX,
        hidden_size: usize,
        num_k_heads_global: usize,
        key_dim_global: usize,
        value_dim_global: usize,
        num_v_heads_global: usize,
        head_v_dim: usize,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_quantized: bool,
    ) -> Result<GdnProjection> {
        let (quantization_config, quant) = if is_quantized {
            (config.quantization_config.clone(), config.quant.clone())
        } else {
            (None, None)
        };
        let projection_pairs = [
            ("in_proj_qkv", "attn_qkv"),
            ("in_proj_z", "attn_gate"),
            ("in_proj_b", "ssm_beta"),
            ("in_proj_a", "ssm_alpha"),
        ];
        let projection_key_map = collect_key_map(vb.is_qvar_builder(), projection_pairs);

        // Qwen3Next format: fused qkvz + fused ba
        let projection_size_qkvz = key_dim_global * 2 + value_dim_global * 2;
        let projection_size_ba = num_v_heads_global * 2;

        let vb_qkvz = vb.pp("in_proj_qkvz");
        let (qc_qkvz, q_qkvz) =
            Self::resolve_quant_for_weight(&vb_qkvz, &quantization_config, &quant);
        let fused_qkvz = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_qkvz,
            false,
            vb_qkvz,
            comm.clone(),
            &qc_qkvz,
            &q_qkvz,
            dtype,
        );

        let vb_ba = vb.pp("in_proj_ba");
        let (qc_ba, q_ba) = Self::resolve_quant_for_weight(&vb_ba, &quantization_config, &quant);
        let fused_ba = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_ba,
            false,
            vb_ba,
            comm.clone(),
            &qc_ba,
            &q_ba,
            dtype,
        );

        if let (Ok(in_proj_qkvz), Ok(in_proj_ba)) = (fused_qkvz, fused_ba) {
            return Ok(GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            });
        };

        // Qwen3.5 format: split qkv, z, b, a
        let split_z = if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            load_restored_gguf_column_linear(
                vb,
                hidden_size,
                value_dim_global,
                projection_key_map["in_proj_z"],
                comm.clone(),
                DType::F32,
                |w| {
                    undo_tiled_v_heads_first_dim(
                        &w,
                        num_k_heads_global,
                        num_v_heads_global,
                        head_v_dim,
                    )
                },
            )
        } else {
            let vb_z = vb.pp(projection_key_map["in_proj_z"]);
            let (qc_z, q_z) = Self::resolve_quant_for_weight(&vb_z, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                value_dim_global,
                false,
                vb_z,
                comm.clone(),
                &qc_z,
                &q_z,
                dtype,
            )
        };

        let split_b = if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            load_restored_gguf_column_linear(
                vb,
                hidden_size,
                num_v_heads_global,
                projection_key_map["in_proj_b"],
                comm.clone(),
                DType::F32,
                |w| undo_tiled_v_heads_first_dim(&w, num_k_heads_global, num_v_heads_global, 1),
            )
        } else {
            let vb_b = vb.pp(projection_key_map["in_proj_b"]);
            let (qc_b, q_b) = Self::resolve_quant_for_weight(&vb_b, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                num_v_heads_global,
                false,
                vb_b,
                comm.clone(),
                &qc_b,
                &q_b,
                dtype,
            )
        };
        let split_a = if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            load_restored_gguf_column_linear(
                vb,
                hidden_size,
                num_v_heads_global,
                projection_key_map["in_proj_a"],
                comm.clone(),
                DType::F32,
                |w| undo_tiled_v_heads_first_dim(&w, num_k_heads_global, num_v_heads_global, 1),
            )
        } else {
            let vb_a = vb.pp(projection_key_map["in_proj_a"]);
            let (qc_a, q_a) = Self::resolve_quant_for_weight(&vb_a, &quantization_config, &quant);
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                num_v_heads_global,
                false,
                vb_a,
                comm.clone(),
                &qc_a,
                &q_a,
                dtype,
            )
        };

        if let (Ok(in_proj_z), Ok(in_proj_b), Ok(in_proj_a)) = (split_z, split_b, split_a) {
            if comm.world_size() > 1 {
                // TP-safe path for packed in_proj_qkv [q|k|v]:
                // shard each semantic chunk independently (q, k, v), not as one contiguous block.
                let split_qkv_merged =
                    if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
                        load_restored_gguf_merged_qkv_linear(
                            vb,
                            hidden_size,
                            key_dim_global,
                            value_dim_global,
                            num_k_heads_global,
                            num_v_heads_global,
                            head_v_dim,
                            projection_key_map["in_proj_qkv"],
                            comm.clone(),
                            DType::F32,
                        )
                    } else {
                        let vb_qkv = vb.pp(projection_key_map["in_proj_qkv"]);
                        let (qc_qkv, q_qkv) =
                            Self::resolve_quant_for_weight(&vb_qkv, &quantization_config, &quant);
                        MergedParallelColumnLinear::load_merged_chunks(
                            hidden_size,
                            key_dim_global * 2 + value_dim_global,
                            0,
                            vec![key_dim_global, key_dim_global, value_dim_global],
                            None,
                            vb_qkv,
                            comm.clone(),
                            &qc_qkv,
                            &q_qkv,
                            dtype,
                        )
                    };

                match split_qkv_merged {
                    Ok(in_proj_qkv) => {
                        return Ok(GdnProjection::SplitQkvZaMerged {
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                        });
                    }
                    Err(err) => {
                        if is_quantized && !vb.is_qvar_builder() {
                            candle_core::bail!(
                                "Unable to load TP-safe quantized Qwen3.5 split in_proj_qkv: {}",
                                err
                            );
                        }
                    }
                }
            }

            // Single GPU (or non-FP8 fallback): use legacy split loader.
            let split_qkv_legacy =
                if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
                    load_restored_gguf_column_linear(
                        vb,
                        hidden_size,
                        key_dim_global * 2 + value_dim_global,
                        projection_key_map["in_proj_qkv"],
                        comm.clone(),
                        DType::F32,
                        |w| {
                            restore_qwen35_qkv_weight(
                                &w,
                                key_dim_global,
                                num_k_heads_global,
                                num_v_heads_global,
                                head_v_dim,
                            )
                        },
                    )
                } else {
                    let vb_qkv = vb.pp(projection_key_map["in_proj_qkv"]);
                    let (qc_qkv, q_qkv) =
                        Self::resolve_quant_for_weight(&vb_qkv, &quantization_config, &quant);
                    TensorParallelColumnLinear::load_with_hints(
                        hidden_size,
                        key_dim_global * 2 + value_dim_global,
                        false,
                        vb_qkv,
                        comm.clone(),
                        &qc_qkv,
                        &q_qkv,
                        dtype,
                    )
                };

            if let Ok(in_proj_qkv) = split_qkv_legacy {
                return Ok(GdnProjection::SplitQkvZaLegacy {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                });
            }
        }

        candle_core::bail!("Unable to load Qwen3.5/Qwen3Next linear attention projection weights",)
    }

    fn fix_qwen3next_projection_order(
        &self,
        mixed_qkvz: &Tensor,
        mixed_ba: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let seq_len = mixed_qkvz.dim(0)?;
        let qkvz_group_dim =
            self.head_k_dim + self.head_k_dim + self.kv_group_size * self.head_v_dim * 2;
        let ba_group_dim = 2 * self.kv_group_size;

        let mixed_qkvz = mixed_qkvz.reshape((seq_len, self.num_k_heads, qkvz_group_dim))?;
        let mixed_ba = mixed_ba.reshape((seq_len, self.num_k_heads, ba_group_dim))?;

        let mut offset = 0usize;
        let query = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let key = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let value = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;
        offset += self.kv_group_size * self.head_v_dim;
        let z = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;

        let b = mixed_ba.narrow(2, 0, self.kv_group_size)?;
        let a = mixed_ba.narrow(2, self.kv_group_size, self.kv_group_size)?;

        Ok((
            query.reshape((seq_len, self.key_dim))?,
            key.reshape((seq_len, self.key_dim))?,
            value.reshape((seq_len, self.value_dim))?,
            z.reshape((seq_len, self.value_dim))?,
            b.reshape((seq_len, self.num_v_heads))?,
            a.reshape((seq_len, self.num_v_heads))?,
        ))
    }

    fn project_inputs(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // When norms output F32 (is_f16_mode / higher_precision) but projection weights
        // are in the original model dtype (F16/BF16), cast input to match weights.
        // GGUF (QLinear) handles this internally, but unquantized Linear requires matching dtypes.
        let xs = &if xs.dtype() != self.projection_dtype {
            xs.to_dtype(self.projection_dtype)?
        } else {
            xs.clone()
        };
        match &self.projection {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = in_proj_qkvz.forward(xs)?;
                let mixed_ba = in_proj_ba.forward(xs)?;
                self.fix_qwen3next_projection_order(&mixed_qkvz, &mixed_ba)
            }
            GdnProjection::SplitQkvZaLegacy {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = in_proj_qkv.forward(xs)?;
                let q = proj_qkv.narrow(1, 0, self.key_dim)?.contiguous()?;
                let k = proj_qkv
                    .narrow(1, self.key_dim, self.key_dim)?
                    .contiguous()?;
                let v = proj_qkv
                    .narrow(1, self.key_dim * 2, self.value_dim)?
                    .contiguous()?;
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
            GdnProjection::SplitQkvZaMerged {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let qkv = in_proj_qkv.forward(xs)?;
                if qkv.len() != 3 {
                    candle_core::bail!(
                        "Expected 3 chunks from merged in_proj_qkv, got {}",
                        qkv.len()
                    );
                }
                let q = qkv[0].clone();
                let k = qkv[1].clone();
                let v = qkv[2].clone();
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
        }
    }

    fn repeat_kv_heads(&self, x: Tensor) -> Result<Tensor> {
        if self.num_k_heads == self.num_v_heads {
            return Ok(x);
        }
        let (seq_len, _h, _d) = x.dims3()?;
        x.unsqueeze(2)?
            .broadcast_as((
                seq_len,
                self.num_k_heads,
                self.kv_group_size,
                self.head_k_dim,
            ))?
            .reshape((seq_len, self.num_v_heads, self.head_k_dim))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let hybrid = resolve_qwen3_hybrid_config(config);
        let world_size = comm.world_size();
        let rank = comm.rank();

        let num_v_heads_global = hybrid.num_v_heads;
        let num_k_heads_global = hybrid.num_k_heads;
        if num_v_heads_global % num_k_heads_global != 0 {
            candle_core::bail!(
                "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                num_v_heads_global,
                num_k_heads_global
            );
        }
        if num_v_heads_global % world_size != 0 || num_k_heads_global % world_size != 0 {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let is_quantized = config.quantization_config.is_some();
        let kernel_dtype = if vb.is_qvar_builder() || config.is_f16_mode {
            DType::F32
        } else {
            dtype
        };
        // GGUF (QLinear) casts input to F32 internally, so any input dtype works.
        // Quantized formats (LnFp8/LnNvfp4/LnMxfp4) and unquantized Linear both need
        // input in the model's weight dtype (F16/BF16).
        let projection_dtype = if vb.is_qvar_builder() {
            kernel_dtype
        } else {
            dtype
        };

        let num_v_heads = num_v_heads_global / world_size;
        let num_k_heads = num_k_heads_global / world_size;
        let head_k_dim = hybrid.key_head_dim;
        let head_v_dim = hybrid.value_head_dim;
        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let kv_group_size = num_v_heads / num_k_heads;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let conv_dim_global = key_dim_global * 2 + value_dim_global;

        // Learned GDN parameters
        let sd = shard(0, comm.rank(), comm.world_size());
        let gdn_pairs = [
            ("A_log", "ssm_a"),
            ("dt_bias", "ssm_dt.bias"),
            ("conv1d.weight", "ssm_conv1d.weight"),
            ("conv1d.bias", "ssm_conv1d.bias"),
            ("out_proj", "ssm_out"),
            ("norm.weight", "ssm_norm.weight"),
            ("norm.bias", "ssm_norm.bias"),
        ];
        let gdn_key_map = collect_key_map(vb.is_qvar_builder(), gdn_pairs);
        let a_log_loaded =
            vb.get_with_hints_dtype((num_v_heads_global,), gdn_key_map["A_log"], sd, DType::F32)?;
        let mut a_log = if vb.is_qvar_builder() {
            restore_qwen35_a_log_from_gguf(&a_log_loaded)?
        } else {
            a_log_loaded
        };
        let mut dt_bias = vb
            .get_with_hints_dtype(
                (num_v_heads_global,),
                gdn_key_map["dt_bias"],
                sd,
                DType::F32,
            )?
            .to_dtype(kernel_dtype)?;
        if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            a_log =
                undo_tiled_v_heads_first_dim(&a_log, num_k_heads_global, num_v_heads_global, 1)?;
            dt_bias =
                undo_tiled_v_heads_first_dim(&dt_bias, num_k_heads_global, num_v_heads_global, 1)?;
        }
        if vb.is_qvar_builder() {
            a_log = tensor_parallel_chunk(&a_log, 0, rank, world_size, gdn_key_map["A_log"])?;
            dt_bias = tensor_parallel_chunk(&dt_bias, 0, rank, world_size, gdn_key_map["dt_bias"])?;
        }

        let projection = Self::load_projection(
            &vb,
            hidden_size,
            num_k_heads_global,
            key_dim_global,
            value_dim_global,
            num_v_heads_global,
            head_v_dim,
            comm.clone(),
            config,
            dtype,
            is_quantized,
        )?;

        // Conv1D weights are stored global; slice rank-local q/k/v channel blocks.
        let conv_weight = if vb.is_qvar_builder() {
            vb.get_with_hints_dtype(
                (conv_dim_global, conv_kernel_size),
                gdn_key_map["conv1d.weight"],
                Default::default(),
                DType::F32,
            )?
            .unsqueeze(1)?
        } else {
            vb.get(
                (conv_dim_global, 1, conv_kernel_size),
                gdn_key_map["conv1d.weight"],
            )?
        };
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let q_w = conv_weight.narrow(0, q_start, key_dim)?;
        let k_w = conv_weight.narrow(0, k_start, key_dim)?;
        let mut v_w = conv_weight.narrow(0, key_dim_global * 2, value_dim_global)?;
        if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            v_w = undo_tiled_v_heads_first_dim(
                &v_w,
                num_k_heads_global,
                num_v_heads_global,
                head_v_dim,
            )?;
        }
        v_w = tensor_parallel_chunk(&v_w, 0, rank, world_size, "linear_attn.conv1d.weight[v]")?;
        let conv_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?.to_dtype(kernel_dtype)?;

        let conv_bias = vb.get((conv_dim_global,), gdn_key_map["conv1d.bias"]).ok();
        let conv_bias = if let Some(cb) = conv_bias {
            let q_b = cb.narrow(0, q_start, key_dim)?;
            let k_b = cb.narrow(0, k_start, key_dim)?;
            let mut v_b = cb.narrow(0, key_dim_global * 2, value_dim_global)?;
            if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
                v_b = undo_tiled_v_heads_first_dim(
                    &v_b,
                    num_k_heads_global,
                    num_v_heads_global,
                    head_v_dim,
                )?;
            }
            v_b = tensor_parallel_chunk(&v_b, 0, rank, world_size, "linear_attn.conv1d.bias[v]")?;
            Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0)?.to_dtype(kernel_dtype)?)
        } else {
            None
        };

        // Output projection
        let out_proj = if vb.is_qvar_builder() && num_k_heads_global != num_v_heads_global {
            load_restored_gguf_row_linear(
                &vb,
                value_dim_global,
                hidden_size,
                gdn_key_map["out_proj"],
                comm.clone(),
                DType::F32,
                |w| {
                    undo_tiled_v_heads_last_dim(
                        &w,
                        num_k_heads_global,
                        num_v_heads_global,
                        head_v_dim,
                    )
                },
            )?
        } else {
            let vb_out = vb.pp(gdn_key_map["out_proj"]);
            let (qc_out, q_out) = if is_quantized {
                Self::resolve_quant_for_weight(&vb_out, &config.quantization_config, &config.quant)
            } else {
                (None, None)
            };
            TensorParallelRowLinear::load_with_hints(
                value_dim_global,
                hidden_size,
                vb_out,
                comm.clone(),
                &qc_out,
                &q_out,
                dtype,
            )?
        };

        // GDN output norm (gated RMSNorm): both Qwen3.5 and Qwen3Next use per-head params.
        let gdn_norm_weight = vb
            .get_with_hints_dtype(
                (head_v_dim,),
                gdn_key_map["norm.weight"],
                Shard::default(),
                DType::F32,
            )
            .map_err(|err| {
                candle_core::Error::Msg(format!(
                    "Unable to load linear_attn.norm.weight as per-head [{head_v_dim}]: {err}"
                ))
            })?;
        let gdn_norm_bias = vb
            .get_with_hints_dtype(
                (head_v_dim,),
                gdn_key_map["norm.bias"],
                Shard::default(),
                DType::F32,
            )
            .ok();
        let scale = 1.0f64 / (head_k_dim as f64).sqrt();

        Ok(Self {
            projection,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            kv_group_size,
            gdn_layer_idx,
            rms_norm_eps: config.rms_norm_eps,
            scale,
            kernel_dtype,
            projection_dtype,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let slot_count = seq_slots.dim(0)?;
        if slot_count == 0 {
            candle_core::bail!("Linear attention requires non-empty sequence slots");
        }
        let original_dtype = xs.dtype();

        // For GGUF (qvar_builder), input is already F32 — project in F32.
        // For quantized models (FP8/NVFP4), projections need the model dtype (F16/BF16),
        // so we project first, THEN convert to kernel_dtype for GDN core ops.
        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;
        let (q, k, v, z, b, a) = self.project_inputs(xs)?;

        // Convert projection outputs to kernel_dtype for GDN core (conv1d, gating, recurrence).
        // For GGUF and F16 mode, kernel_dtype is F32 to avoid precision loss.
        let (q, k, v, z, b, a) = if q.dtype() != self.kernel_dtype {
            (
                q.to_dtype(self.kernel_dtype)?,
                k.to_dtype(self.kernel_dtype)?,
                v.to_dtype(self.kernel_dtype)?,
                z.to_dtype(self.kernel_dtype)?,
                b.to_dtype(self.kernel_dtype)?,
                a.to_dtype(self.kernel_dtype)?,
            )
        } else {
            (q, k, v, z, b, a)
        };
        let mixed_qkv = Tensor::cat(&[&q, &k, &v], 1)?;

        let (kv_conv, prefill_conv_state) = if is_prefill {
            let mut conv_state = mamba_cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let out = gdn::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                Some(cu_seqlens),
                true, // SiLU activation
            )?;
            (out, Some(conv_state))
        } else {
            if token_count != slot_count {
                candle_core::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    slot_count
                );
            }
            let out = gdn::causal_conv1d_update_slots(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                mamba_cache.conv_state_mut(self.gdn_layer_idx),
                seq_slots,
                true,
            )?;
            (out, None)
        };
        if let Some(conv_state) = prefill_conv_state {
            mamba_cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;
        }

        // Split convolved output back into q', k', v'
        let q_conv = kv_conv.narrow(1, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1, self.key_dim * 2, self.value_dim)?;

        // Fused GDN gating
        let (a_expanded, b_expanded) = (a.unsqueeze(0)?, b.unsqueeze(0)?); // [1, seq_len, num_heads]
        let (g, beta) =
            gdn::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let (g, beta) = (g.squeeze(0)?, beta.squeeze(0)?);

        let q = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = gdn::l2_norm_last_dim(&q, 1e-6)?;
        let k = gdn::l2_norm_last_dim(&k, 1e-6)?;
        let (q, k) = (self.repeat_kv_heads(q)?, self.repeat_kv_heads(k)?);

        let output = if is_prefill {
            // S1: Use batched varlen recurrence — one CUDA launch for all sequences
            let q_scaled = (&q * self.scale)?;

            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            // Get mutable reference to global state for in-place update (optimized prefill)
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);

            gdn::gated_delta_rule_recurrence_varlen(
                &q_scaled,
                &k,
                &v,
                &g,
                &beta,
                global_state,
                seq_slots,
                &cu_seqlens,
            )?
        } else {
            let batch = slot_count;
            let q_b = (q.reshape((batch, self.num_v_heads, self.head_k_dim))? * self.scale)?;
            let k_b = k.reshape((batch, self.num_v_heads, self.head_k_dim))?;
            let v_b = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b = g.reshape((batch, self.num_v_heads))?;
            let beta_b = beta.reshape((batch, self.num_v_heads))?;
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            gdn::gated_delta_rule_decode_slots(
                &q_b,
                &k_b,
                &v_b,
                &g_b,
                &beta_b,
                global_state,
                seq_slots,
            )?
        };

        // output: [seq_len, num_v_heads, head_v_dim] -> [seq_len, value_dim]
        let output = output.reshape((token_count, self.value_dim))?;

        // Gated RMSNorm: norm(output) * silu(z) via fused kernel
        let gated_output = gdn::gated_rmsnorm_silu_mul(
            &output,
            &z,
            &self.gdn_norm_weight,
            self.gdn_norm_bias.as_ref(),
            self.rms_norm_eps,
            self.head_v_dim,
        )?;

        // Output projection — cast to projection dtype for matmul, then restore original dtype
        let out = self
            .out_proj
            .forward(&gated_output.to_dtype(self.projection_dtype)?)?;
        if out.dtype() != original_dtype {
            out.to_dtype(original_dtype)
        } else {
            Ok(out)
        }
    }
}
