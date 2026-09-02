use crate::models::layers::distributed::{shard, Comm};
use crate::models::layers::linear::LinearX as Linear;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Result, Tensor};
use candle_nn::Module;
use std::rc::Rc;

/// Qwen4 Gated Residual (Hyper-Connection) read/write transform.
/// Reference: Qwen4ExpTextGatedResidual in HuggingFace modeling_qwen4_exp.py
pub struct Qwen4HyperConnection {
    hc_count: usize,
    hidden_size: usize,
    hc_norm_weight: Tensor,
    input_mix_weight_down: Linear,
    input_mix_weight_up: Linear,
    block_inject_weight: Option<Linear>,
    rms_norm_eps: f64,
}

pub struct Qwen4HyperConnectionState {
    pub hyper_input: Tensor,
    pub injection_weights: Option<Tensor>,
}

impl Qwen4HyperConnection {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        hc_count: usize,
        hidden_size: usize,
        hc_lowrank: usize,
        rms_norm_eps: f64,
        dtype: DType,
        use_combine: bool,
        prefix: &str,
    ) -> Result<Self> {
        let hc_hidden = hc_count * hidden_size;
        let hc_norm_weight = vb.get_with_hints_dtype(
            (hc_hidden,),
            &format!("{prefix}.hc_norm.weight"),
            shard(0, comm.rank(), comm.world_size()),
            dtype,
        )?;
        let input_mix_weight_down = Linear::new(
            vb.get_with_hints_dtype(
                (hc_lowrank, hc_hidden),
                &format!("{prefix}.input_mix_weight_down.weight"),
                shard(1, comm.rank(), comm.world_size()),
                dtype,
            )?,
            None,
            &None,
        )?;
        let input_mix_weight_up = Linear::new(
            vb.get_with_hints_dtype(
                (hc_hidden, hc_lowrank),
                &format!("{prefix}.input_mix_weight_up.weight"),
                shard(0, comm.rank(), comm.world_size()),
                dtype,
            )?,
            None,
            &None,
        )?;
        let block_inject_weight = if use_combine {
            Some(Linear::new(
                vb.get_with_hints_dtype(
                    (hc_count, hc_hidden),
                    &format!("{prefix}.block_inject_weight.weight"),
                    shard(0, comm.rank(), comm.world_size()),
                    dtype,
                )?,
                None,
                &None,
            )?)
        } else {
            None
        };
        Ok(Self {
            hc_count,
            hidden_size,
            hc_norm_weight,
            input_mix_weight_down,
            input_mix_weight_up,
            block_inject_weight,
            rms_norm_eps,
        })
    }

    /// Read: collapse hc branches to block input.
    pub fn read(&self, hyper_input: &Tensor) -> Result<(Tensor, Qwen4HyperConnectionState)> {
        #[cfg(feature = "cuda")]
        {
            if hyper_input.device().is_cuda() {
                let inject_w = self
                    .block_inject_weight
                    .as_ref()
                    .map(|l| l.dense_weight())
                    .transpose()?;
                let (mixed, inject, _scratch) = attention_rs::qwen4::hc_read(
                    hyper_input,
                    &self.hc_norm_weight,
                    self.input_mix_weight_down.dense_weight()?,
                    self.input_mix_weight_up.dense_weight()?,
                    inject_w.as_ref().map(|w| *w),
                    self.hc_count,
                    self.hidden_size,
                    self.input_mix_weight_down.dense_weight()?.dim(0)?,
                    self.rms_norm_eps as f32,
                )?;
                return Ok((
                    mixed,
                    Qwen4HyperConnectionState {
                        hyper_input: hyper_input.clone(),
                        injection_weights: inject,
                    },
                ));
            }
        }
        self.read_candle(hyper_input)
    }

    fn read_candle(&self, hyper_input: &Tensor) -> Result<(Tensor, Qwen4HyperConnectionState)> {
        let (seq_len, hc_hidden) = hyper_input.dims2()?;
        let hc = self.hc_count;
        let hidden = self.hidden_size;
        let x = hyper_input.reshape((seq_len, hc, hidden))?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
        let weight =
            (self.hc_norm_weight.to_dtype(normed.dtype())? + 1.0)?.reshape((1, hc, hidden))?;
        let normed = normed.broadcast_mul(&weight)?;
        let flat = normed.flatten_from(1)?;
        let mix_down = self.input_mix_weight_down.forward(&flat)?;
        let mix_down = candle_nn::ops::silu(&(mix_down / (hc as f64))?)?;
        let mix_up = candle_nn::ops::sigmoid(&self.input_mix_weight_up.forward(&mix_down)?)?;
        let mix_up = mix_up.reshape((seq_len, hc, hidden))?;
        let mixed = (mix_up * &normed)?.mean_keepdim(1)?.squeeze(1)?;
        let injection_weights = if let Some(w) = &self.block_inject_weight {
            let inj = candle_nn::ops::sigmoid(&w.forward(&flat)?)?;
            Some(((inj * 2.0)? / (hc as f64))?)
        } else {
            None
        };
        Ok((
            mixed,
            Qwen4HyperConnectionState {
                hyper_input: hyper_input.clone(),
                injection_weights,
            },
        ))
    }

    /// Write: inject block output back into hc branches.
    pub fn write(&self, block_out: &Tensor, state: &Qwen4HyperConnectionState) -> Result<Tensor> {
        let inject = state
            .injection_weights
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("hc write requires injection weights".into()))?;
        #[cfg(feature = "cuda")]
        {
            if block_out.device().is_cuda() {
                return attention_rs::qwen4::hc_write(
                    &state.hyper_input,
                    block_out,
                    inject,
                    self.hc_count,
                    self.hidden_size,
                );
            }
        }
        let seq_len = block_out.dim(0)?;
        let hc = self.hc_count;
        let hidden = self.hidden_size;
        let inject = inject.reshape((seq_len, hc, 1))?;
        let block = block_out.reshape((seq_len, 1, hidden))?;
        let injection = block.broadcast_mul(&inject)?.flatten_from(1)?;
        state.hyper_input.clone() + injection
    }
}
