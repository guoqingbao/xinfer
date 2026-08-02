//! Compare attention.rs FP8 GEMM dispatches with a PyTorch FP8 dequantized GEMM.

use anyhow::{Context, Result};
use attention_rs::fp8_linear::{fp8_matmul, fp8_matmul_fallback};
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

fn main() -> Result<()> {
    let dev = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let path = std::env::var_os("XINFER_FP8_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_fp8_probe.bin"));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(b"XINFFP81")?;

    let input = Tensor::randn(0f32, 1f32, (8, 128), &dev)?.to_dtype(DType::BF16)?;
    // Raw E4M3FN bytes; 0x38=1, 0xb8=-1, 0x40=2, 0xc0=-2 and nearby values.
    let fp8_values = (0..(128 * 128))
        .map(|i| [0x38u8, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x48, 0xc8][i % 8])
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(fp8_values, (128, 128), &dev)?;
    let scale = Tensor::ones((1, 1), DType::F32, &dev)?;
    let cutlass = fp8_matmul(&input, &weight, &scale, None, &[128, 128], true)?;
    let fallback = fp8_matmul_fallback(&input, &weight, &scale, &[128, 128])?;
    for (name, tensor) in [
        ("input", &input),
        ("weight", &weight),
        ("cutlass", &cutlass),
        ("fallback", &fallback),
    ] {
        record(&mut file, name, tensor)?;
    }
    file.flush()?;
    println!("wrote {}", path.display());
    Ok(())
}
