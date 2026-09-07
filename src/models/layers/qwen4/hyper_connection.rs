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
        let _ = comm;
        let hc_hidden = hc_count * hidden_size;
        // HC weights are replicated on every TP rank. The HF tp_plan shards
        // input_mix_weight_down row-wise (split input), which would require an
        // all-reduce of the low-rank mix output; these matrices are tiny
        // (e.g. 320x10240), so replication is cheap and exact.
        let rep = shard(0, 0, 1);
        let hc_norm_weight = vb.get_with_hints_dtype(
            (hc_hidden,),
            &format!("{prefix}.hc_norm.weight"),
            rep,
            dtype,
        )?;
        let input_mix_weight_down = Linear::new(
            vb.get_with_hints_dtype(
                (hc_lowrank, hc_hidden),
                &format!("{prefix}.input_mix_weight_down.weight"),
                rep,
                dtype,
            )?,
            None,
            &None,
        )?;
        let input_mix_weight_up = Linear::new(
            vb.get_with_hints_dtype(
                (hc_hidden, hc_lowrank),
                &format!("{prefix}.input_mix_weight_up.weight"),
                rep,
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
                    rep,
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
        // NOTE: the fused CUDA kernel (attention_rs::qwen4::hc_read) launches
        // one block per token, so at decode batch=1 the two low-rank GEMVs
        // (~13MB of weights) are pulled through a single SM — measured
        // ~1.86ms/call vs ~0.1ms for the candle-op path below (cuBLAS uses
        // the whole GPU). Use the candle path until the kernel is
        // restructured with multi-block GEMV parallelism.
        self.read_candle(hyper_input)
    }

    fn read_candle(&self, hyper_input: &Tensor) -> Result<(Tensor, Qwen4HyperConnectionState)> {
        let (seq_len, _hc_hidden) = hyper_input.dims2()?;
        let hc = self.hc_count;
        let hidden = self.hidden_size;
        let in_dtype = hyper_input.dtype();
        // Reference computes the grouped RMSNorm in float32 with (1 + w) scale.
        let x = hyper_input
            .to_dtype(DType::F32)?
            .reshape((seq_len, hc, hidden))?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
        let weight = (self.hc_norm_weight.to_dtype(DType::F32)? + 1.0)?.reshape((1, hc, hidden))?;
        let normed = normed
            .broadcast_mul(&weight)?
            .to_dtype(in_dtype)?
            .reshape((seq_len, hc, hidden))?;
        let flat = normed.flatten_from(1)?;
        let mix_down = self.input_mix_weight_down.forward(&flat)?;
        let mix_down = candle_nn::ops::silu(&(mix_down / (hc as f64))?)?;
        let mix_up = candle_nn::ops::sigmoid(&self.input_mix_weight_up.forward(&mix_down)?)?;
        let mix_up = mix_up.reshape((seq_len, hc, hidden))?;
        let mixed = (mix_up * &normed)?.mean_keepdim(1)?.squeeze(1)?;
        // Reference: injection = 2 * sigmoid(W_inject @ normed / hc)
        let injection_weights = if let Some(w) = &self.block_inject_weight {
            let inj = candle_nn::ops::sigmoid(&(w.forward(&flat)? / (hc as f64))?)?;
            Some((inj * 2.0)?)
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
