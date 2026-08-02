//! Differential probe for attention.rs kernels that are shared by dense and
//! MoE models: rotary embedding, router top-k, and fused SiLU-and-mul.

use anyhow::{Context, Result};
use attention_rs::{fused_rope::FusedRope, silu_and_mul::silu_and_mul, topk};
use candle_core::{DType, Device, Tensor};
use std::io::Write;
use std::path::PathBuf;

fn record(file: &mut std::fs::File, name: &str, tensor: &Tensor) -> Result<()> {
    let cpu = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let values = cpu.flatten_all()?.to_vec1::<f32>()?;
    file.write_all(&(name.len() as u64).to_le_bytes())?;
    file.write_all(name.as_bytes())?;
    file.write_all(&(tensor.dims().len() as u64).to_le_bytes())?;
    for &dim in tensor.dims() {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }
    file.write_all(&(values.len() as u64).to_le_bytes())?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn random(device: &Device, shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(DType::BF16)?)
}

fn rope_case(
    file: &mut std::fs::File,
    device: &Device,
    tag: &str,
    interleaved: bool,
) -> Result<()> {
    let q = random(device, (11, 3, 8))?;
    let k = random(device, (11, 2, 8))?;
    let cos = Tensor::randn(0f32, 1f32, (64, 4), device)?.to_dtype(DType::BF16)?;
    let sin = Tensor::randn(0f32, 1f32, (64, 4), device)?.to_dtype(DType::BF16)?;
    let positions = Tensor::from_vec(vec![0i64, 7, 2, 31, 9, 16, 3, 63, 4, 12, 5], (11,), device)?;
    record(file, &format!("{tag}_q"), &q)?;
    record(file, &format!("{tag}_k"), &k)?;
    record(file, &format!("{tag}_cos"), &cos)?;
    record(file, &format!("{tag}_sin"), &sin)?;
    record(file, &format!("{tag}_positions"), &positions)?;
    let (q_out, k_out) = if interleaved {
        FusedRope::apply_rope_i(&q, &k, &cos, &sin, &positions)?
    } else {
        FusedRope::apply_rope(&q, &k, &cos, &sin, &positions)?
    };
    record(file, &format!("{tag}_q_out"), &q_out)?;
    record(file, &format!("{tag}_k_out"), &k_out)?;
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let output = std::env::var_os("XINFER_MISC_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_attention_misc_probe.bin"));
    let mut file = std::fs::File::create(&output)?;
    file.write_all(b"XINFMISC1")?;

    rope_case(&mut file, &device, "rope_noninterleaved", false)?;
    rope_case(&mut file, &device, "rope_interleaved", true)?;

    let gate_up = random(&device, (13, 2 * 24))?;
    let silu_out = silu_and_mul(&gate_up, 24)?;
    record(&mut file, "silu_gate_up", &gate_up)?;
    record(&mut file, "silu_out", &silu_out)?;
    let gate_up_f16 = Tensor::randn(0f32, 1f32, (13, 2 * 24), &device)?.to_dtype(DType::F16)?;
    let silu_out_f16 = silu_and_mul(&gate_up_f16, 24)?;
    record(&mut file, "silu_gate_up_f16", &gate_up_f16)?;
    record(&mut file, "silu_out_f16", &silu_out_f16)?;

    let logits = Tensor::randn(0f32, 1f32, (17, 32), &device)?;
    let scores = Tensor::randn(0f32, 1f32, (17, 32), &device)?;
    let (softmax_weights, softmax_indices) = topk::topk_softmax(&logits, 5)?;
    let (select_weights, select_indices) = topk::topk_select(&scores, 5)?;
    let sigmoid_logits = Tensor::randn(0f32, 1f32, (17, 32), &device)?;
    let sigmoid_bias = Tensor::randn(0f32, 0.1f32, (32,), &device)?;
    let (sigmoid_weights, sigmoid_indices) =
        topk::fused_sigmoid_topk(&sigmoid_logits, Some(&sigmoid_bias), 5)?;
    record(&mut file, "logits", &logits)?;
    record(&mut file, "softmax_weights", &softmax_weights)?;
    record(&mut file, "softmax_indices", &softmax_indices)?;
    record(&mut file, "scores", &scores)?;
    record(&mut file, "select_weights", &select_weights)?;
    record(&mut file, "select_indices", &select_indices)?;
    record(&mut file, "sigmoid_logits", &sigmoid_logits)?;
    record(&mut file, "sigmoid_bias", &sigmoid_bias)?;
    record(&mut file, "sigmoid_weights", &sigmoid_weights)?;
    record(&mut file, "sigmoid_indices", &sigmoid_indices)?;

    file.flush()?;
    println!("wrote {}", output.display());
    Ok(())
}
