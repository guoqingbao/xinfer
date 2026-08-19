use candle_core::{DType, Result, Tensor};
use candle_nn::{Activation, Module};

/// Gated activation used by SwiGLU-style MLPs and MoE projections.
///
/// `SwiGluOai` is selected from the raw model configuration rather than the
/// generic `Config` struct because it is a MiniMax M3-specific activation.
#[derive(Clone, Debug)]
pub enum GatedActivation {
    Standard(Activation),
    SwiGluOai { alpha: f32, beta: f32, limit: f32 },
}

impl GatedActivation {
    pub fn standard(activation: Activation) -> Self {
        Self::Standard(activation)
    }

    /// Resolve the activation without adding model-specific fields to the
    /// shared configuration structure. M3 keeps these values in the raw JSON
    /// stored by `Config::extra_config_json`.
    pub fn from_model_config(activation: Activation, extra_config_json: Option<&str>) -> Self {
        let Some(raw) = extra_config_json else {
            return Self::Standard(activation);
        };
        let Ok(root) = serde_json::from_str::<serde_json::Value>(raw) else {
            return Self::Standard(activation);
        };
        let is_m3 = root
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .and_then(|v| v.as_str())
            .is_some_and(|v| v.starts_with("MiniMaxM3"));
        if !is_m3 {
            return Self::Standard(activation);
        }

        let cfg = root.get("text_config").unwrap_or(&root);
        let alpha = cfg
            .get("swiglu_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.702) as f32;
        let beta = cfg
            .get("swiglu_beta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let limit = cfg
            .get("swiglu_limit")
            .and_then(|v| v.as_f64())
            .unwrap_or(7.0) as f32;
        Self::SwiGluOai { alpha, beta, limit }
    }

    pub fn forward_fused(&self, gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
        match self {
            Self::Standard(Activation::Silu) => {
                attention_rs::silu_and_mul::silu_and_mul(gate_up, half_dim)
            }
            Self::Standard(act) => {
                let gate = gate_up
                    .narrow(candle_core::D::Minus1, 0, half_dim)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(candle_core::D::Minus1, half_dim, half_dim)?
                    .contiguous()?;
                (up * gate.apply(act)?)?.contiguous()
            }
            Self::SwiGluOai { alpha, beta, limit } => {
                let gate = gate_up
                    .narrow(candle_core::D::Minus1, 0, half_dim)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(candle_core::D::Minus1, half_dim, half_dim)?
                    .contiguous()?;
                self.forward_separate_with_params(&gate, &up, *alpha, *beta, *limit)
            }
        }
    }

    pub fn forward_separate(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(act) => act.forward(gate)?.broadcast_mul(up),
            Self::SwiGluOai { alpha, beta, limit } => {
                self.forward_separate_with_params(gate, up, *alpha, *beta, *limit)
            }
        }
    }

    fn forward_separate_with_params(
        &self,
        gate: &Tensor,
        up: &Tensor,
        alpha: f32,
        beta: f32,
        limit: f32,
    ) -> Result<Tensor> {
        // Build scalar bounds from device tensors. Using Tensor::clamp or
        // scalar `+`/`*` operands would create CPU scalar tensors and enqueue
        // host-to-device copies during CUDA graph capture.
        let gate_limit = gate.ones_like()?.affine(0.0, limit as f64)?;
        let gate = gate.minimum(&gate_limit)?;
        let up_limit = up.ones_like()?.affine(0.0, limit as f64)?;
        let up_floor = up.ones_like()?.affine(0.0, -(limit as f64))?;
        let up = up.minimum(&up_limit)?.maximum(&up_floor)?;
        let gate_dtype = gate.dtype();
        let sigmoid =
            candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?.affine(alpha as f64, 0.0)?)?
                .to_dtype(gate_dtype)?;
        gate.broadcast_mul(&sigmoid)?
            .broadcast_mul(&up.affine(1.0, beta as f64)?)
    }
}
