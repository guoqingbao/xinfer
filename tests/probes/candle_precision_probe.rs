//! Exercise Candle's CUDA PTX families and export their results for an
//! independent PyTorch comparison.  This deliberately uses only Candle's
//! public tensor operations; the golden implementation is in Python.

use anyhow::{Context, Result};
use candle_core::{quantized, DType, Device, Module, Tensor};
use candle_nn::ops::sigmoid;
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

fn record_bytes(file: &mut std::fs::File, name: &str, values: &[u8]) -> Result<()> {
    file.write_all(&(name.len() as u64).to_le_bytes())?;
    file.write_all(name.as_bytes())?;
    file.write_all(&1u64.to_le_bytes())?;
    file.write_all(&(values.len() as u64).to_le_bytes())?;
    file.write_all(&(values.len() as u64).to_le_bytes())?;
    for &value in values {
        file.write_all(&(value as f32).to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let output = std::env::var_os("XINFER_CANDLE_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_candle_probe.bin"));
    let mut file = std::fs::File::create(&output)?;
    file.write_all(b"XINFCAND1")?;

    // Keep the values away from singularities while still covering negative,
    // positive, subnormal-ish and large-magnitude branches.
    let x = Tensor::from_vec(
        vec![-8.0f32, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 16.0],
        (2, 5),
        &device,
    )?;
    let c20 = Tensor::new(20f32, &device)?.broadcast_as(x.shape())?;
    let c4 = Tensor::new(4f32, &device)?.broadcast_as(x.shape())?;
    let c025 = Tensor::new(0.25f32, &device)?.broadcast_as(x.shape())?;
    let a = Tensor::randn(0f32, 1f32, (7, 13), &device)?.to_dtype(DType::BF16)?;
    let b = Tensor::randn(0f32, 1f32, (7, 13), &device)?.to_dtype(DType::BF16)?;
    let bias = Tensor::randn(0f32, 1f32, (13,), &device)?.to_dtype(DType::BF16)?;
    let c025_a = Tensor::new(0.25f32, &device)?
        .broadcast_as(a.shape())?
        .to_dtype(DType::BF16)?;
    let mask = Tensor::from_vec(
        (0..91).map(|i| (i % 3 == 0) as u8).collect::<Vec<_>>(),
        (7, 13),
        &device,
    )?;
    let idx = Tensor::from_vec(vec![6u32, 2, 5, 0], (4,), &device)?;
    let gather_idx = Tensor::from_vec(
        vec![
            6u32, 2, 5, 0, 1, 3, 4, 6, 0, 2, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3,
        ],
        (7, 3),
        &device,
    )?;
    let conv1_x = Tensor::randn(0f32, 1f32, (2, 3, 17), &device)?.to_dtype(DType::BF16)?;
    let conv1_w = Tensor::randn(0f32, 1f32, (4, 3, 5), &device)?.to_dtype(DType::BF16)?;
    let conv2_x = Tensor::randn(0f32, 1f32, (1, 2, 8, 9), &device)?.to_dtype(DType::BF16)?;
    let conv2_w = Tensor::randn(0f32, 1f32, (3, 2, 3, 3), &device)?.to_dtype(DType::BF16)?;
    let image = Tensor::randn(0f32, 1f32, (1, 2, 8, 10), &device)?.to_dtype(DType::BF16)?;

    record(&mut file, "x", &x)?;
    record(&mut file, "a", &a)?;
    record(&mut file, "b", &b)?;
    record(&mut file, "bias", &bias)?;
    record(&mut file, "mask", &mask)?;
    record(&mut file, "idx", &idx)?;
    record(&mut file, "gather_idx", &gather_idx)?;
    record(&mut file, "conv1_x", &conv1_x)?;
    record(&mut file, "conv1_w", &conv1_w)?;
    record(&mut file, "conv2_x", &conv2_x)?;
    record(&mut file, "conv2_w", &conv2_w)?;
    record(&mut file, "image", &image)?;

    // unary.ptx: exercise every floating-point unary family exposed by
    // Candle's CUDA backend.
    for (name, value) in [
        ("neg", x.neg()?),
        ("recip", (&x + &c20)?.recip()?),
        ("exp", (&x / &c4)?.exp()?),
        ("log", (&x.abs()? + &c025)?.log()?),
        ("sin", x.sin()?),
        ("cos", x.cos()?),
        ("tanh", x.tanh()?),
        ("erf", x.erf()?),
        ("abs", x.abs()?),
        ("sqr", x.sqr()?),
        ("sqrt", (&x.abs()? + &c025)?.sqrt()?),
        ("gelu", x.gelu()?),
        ("gelu_erf", x.gelu_erf()?),
        ("relu", x.relu()?),
        ("elu", x.elu(1.0)?),
        ("silu", x.silu()?),
        ("sigmoid", sigmoid(&x)?),
    ] {
        record(&mut file, &format!("unary_{name}"), &value)?;
    }

    // binary.ptx, affine.ptx and ternary.ptx.
    for (name, value) in [
        ("add", (&a + &b)?),
        ("sub", (&a - &b)?),
        ("mul", (&a * &b)?),
        ("div", (&a / (&b.abs()? + &c025_a))?),
        ("maximum", a.maximum(&b)?),
        ("minimum", a.minimum(&b)?),
        ("broadcast_add", a.broadcast_add(&bias)?),
        ("broadcast_mul", a.broadcast_mul(&bias)?),
        ("where", mask.where_cond(&a, &b)?),
        ("affine", a.affine(1.75, -0.125)?),
    ] {
        record(&mut file, &format!("op_{name}"), &value)?;
    }

    // reduce.ptx.
    for (name, value) in [
        ("sum_all", a.sum_all()?),
        ("mean_all", a.mean_all()?),
        ("sum_dim1", a.sum(1)?),
        ("mean_dim1", a.mean(1)?),
        ("max_dim1", a.max(1)?),
        ("min_dim1", a.min(1)?),
        ("argmax_dim1", a.argmax(1)?),
        ("argmin_dim1", a.argmin(1)?),
        ("logsumexp_dim1", a.log_sum_exp(1)?),
    ] {
        record(&mut file, &format!("reduce_{name}"), &value)?;
    }

    // indexing.ptx and sort.ptx.
    record(&mut file, "index_select", &a.index_select(&idx, 0)?)?;
    record(&mut file, "gather", &a.gather(&gather_idx, 1)?)?;
    record(&mut file, "conv1d", &conv1_x.conv1d(&conv1_w, 1, 2, 1, 1)?)?;
    record(&mut file, "conv2d", &conv2_x.conv2d(&conv2_w, 1, 2, 1, 1)?)?;
    record(
        &mut file,
        "avg_pool2d",
        &image.avg_pool2d_with_stride((3, 3), (2, 2))?,
    )?;
    record(
        &mut file,
        "max_pool2d",
        &image.max_pool2d_with_stride((3, 3), (2, 2))?,
    )?;
    record(&mut file, "upsample2d", &image.upsample_nearest2d(11, 13)?)?;
    let (sorted, argsort) = a.sort_last_dim(true)?;
    record(&mut file, "sort_values", &sorted)?;
    record(&mut file, "sort_indices", &argsort)?;

    // cast.ptx and fill.ptx/copy2d are exercised by the conversions and the
    // contiguous materialization used by the exported tensors.
    record(&mut file, "cast_f32", &a.to_dtype(DType::F32)?)?;
    record(&mut file, "cast_f16", &a.to_dtype(DType::F16)?)?;
    record(&mut file, "cast_u8", &a.abs()?.to_dtype(DType::U8)?)?;
    record(
        &mut file,
        "zeros",
        &Tensor::zeros((7, 13), DType::BF16, &device)?,
    )?;
    record(
        &mut file,
        "ones",
        &Tensor::ones((7, 13), DType::BF16, &device)?,
    )?;

    // Quantized CUDA dequant/matmul families.  The CPU quantized tensor is
    // retained as the cross-backend reference; Python then computes the final
    // matmul with torch.matmul from the exported CPU-dequantized weights.
    let q_src = (0..(4 * 256))
        .map(|i| (i as f32 - 512.0) / 256.0)
        .collect::<Vec<_>>();
    let x_src = (0..(3 * 256))
        .map(|i| (i as f32 - 384.0) / 192.0)
        .collect::<Vec<_>>();
    let q_cpu = Tensor::from_vec(q_src, (4, 256), &Device::Cpu)?;
    let x_cpu = Tensor::from_vec(x_src, (3, 256), &Device::Cpu)?;
    let q_cuda = q_cpu.to_device(&device)?;
    let x_cuda = x_cpu.to_device(&device)?;
    for (tag, dtype) in [
        ("q4k", quantized::GgmlDType::Q4K),
        ("q6k", quantized::GgmlDType::Q6K),
    ] {
        let q_cpu_tensor = quantized::QTensor::quantize(&q_cpu, dtype)?;
        let q_cuda_tensor = quantized::QTensor::quantize_on_device(&q_cpu, dtype, &device)?;
        let q_native_tensor = quantized::QTensor::quantize(&q_cuda, dtype)?;
        let q_cpu_deq = q_cpu_tensor.dequantize(&Device::Cpu)?;
        let q_cuda_deq = q_cuda_tensor.dequantize(&device)?;
        let q_native_deq = q_native_tensor.dequantize(&device)?;
        let cpu_raw = q_cpu_tensor.data()?.into_owned();
        let cuda_raw = q_cuda_tensor.data()?.into_owned();
        let native_raw = q_native_tensor.data()?.into_owned();
        let y_cpu = quantized::QMatMul::from_qtensor(q_cpu_tensor)?.forward(&x_cpu)?;
        let y_cuda = quantized::QMatMul::from_qtensor(q_cuda_tensor)?.forward(&x_cuda)?;
        let y_native = quantized::QMatMul::from_qtensor(q_native_tensor)?.forward(&x_cuda)?;
        record(&mut file, &format!("{tag}_x"), &x_cpu)?;
        record(&mut file, &format!("{tag}_weight_source"), &q_cpu)?;
        record(&mut file, &format!("{tag}_weight_cpu_dequant"), &q_cpu_deq)?;
        record(
            &mut file,
            &format!("{tag}_weight_cuda_dequant"),
            &q_cuda_deq,
        )?;
        record(
            &mut file,
            &format!("{tag}_weight_native_dequant"),
            &q_native_deq,
        )?;
        record(&mut file, &format!("{tag}_matmul_cpu"), &y_cpu)?;
        record(&mut file, &format!("{tag}_matmul_cuda"), &y_cuda)?;
        record(&mut file, &format!("{tag}_matmul_native"), &y_native)?;
        record_bytes(&mut file, &format!("{tag}_cpu_raw"), &cpu_raw)?;
        record_bytes(&mut file, &format!("{tag}_cuda_raw"), &cuda_raw)?;
        record_bytes(&mut file, &format!("{tag}_native_raw"), &native_raw)?;
    }

    file.flush()?;
    println!("wrote {}", output.display());
    Ok(())
}
