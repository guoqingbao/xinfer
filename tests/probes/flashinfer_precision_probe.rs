//! Export deterministic Rust/attention.rs paged-prefill results for a Python
//! golden comparison. The reference implementation deliberately lives in
//! `tests/probes/compare_flashinfer_probe.py`, where it uses PyTorch FP32 and the
//! installed official FlashInfer Python package.

use anyhow::{Context, Result};
use attention_rs::flashinfer;
use candle_core::{DType, Device, Tensor};
use std::io::Write;
use std::path::PathBuf;

fn write_tensor(file: &mut std::fs::File, tensor: &Tensor) -> Result<()> {
    let cpu = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let values = cpu.flatten_all()?.to_vec1::<f32>()?;
    file.write_all(&(values.len() as u64).to_le_bytes())?;
    file.write_all(&(tensor.dims().len() as u64).to_le_bytes())?;
    for &dim in tensor.dims() {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn run_case(
    device: &Device,
    output_dir: &std::path::Path,
    case_name: &str,
    prefix_len: usize,
    append_len: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    fp8_kv: bool,
) -> Result<()> {
    let kv_len = prefix_len + append_len;
    let num_pages = kv_len.div_ceil(page_size);
    let dtype = DType::BF16;
    let cache_dtype = if fp8_kv { DType::U8 } else { dtype };
    let k_all =
        Tensor::randn(0f32, 1f32, (kv_len, num_kv_heads, head_dim), device)?.to_dtype(dtype)?;
    let v_all =
        Tensor::randn(0f32, 1f32, (kv_len, num_kv_heads, head_dim), device)?.to_dtype(dtype)?;
    let q =
        Tensor::randn(0f32, 1f32, (append_len, num_qo_heads, head_dim), device)?.to_dtype(dtype)?;

    let indices_host: Vec<u32> = (0..num_pages).map(|i| (i + 3) as u32).collect();
    let indptr_host = vec![0u32, num_pages as u32];
    let last_len_host = vec![((kv_len - 1) % page_size + 1) as u32];
    let indices = Tensor::from_vec(indices_host.clone(), (num_pages,), device)?;
    let indptr = Tensor::from_vec(indptr_host.clone(), (2,), device)?;
    let last_len = Tensor::from_vec(last_len_host.clone(), (1,), device)?;

    // FlashInfer NHD paged cache: [page, token, kv_head, head_dim].
    let mut k_cache = Tensor::zeros(
        (num_pages + 3, page_size, num_kv_heads, head_dim),
        cache_dtype,
        device,
    )?;
    let mut v_cache = Tensor::zeros(
        (num_pages + 3, page_size, num_kv_heads, head_dim),
        cache_dtype,
        device,
    )?;
    for page in 0..prefix_len.div_ceil(page_size) {
        let start = page * page_size;
        let len = (prefix_len - start).min(page_size);
        let k_page = k_all.narrow(0, start, len)?.unsqueeze(0)?;
        let v_page = v_all.narrow(0, start, len)?.unsqueeze(0)?;
        let (k_page, v_page) = if fp8_kv {
            let (k_page, _) = attention_rs::convert_to_fp8(&k_page, Some(1.0))?;
            let (v_page, _) = attention_rs::convert_to_fp8(&v_page, Some(1.0))?;
            (k_page, v_page)
        } else {
            (k_page, v_page)
        };
        let physical_page = indices_host[page] as usize;
        k_cache = k_cache.slice_assign(
            &[
                physical_page..physical_page + 1,
                0..len,
                0..num_kv_heads,
                0..head_dim,
            ],
            &k_page,
        )?;
        v_cache = v_cache.slice_assign(
            &[
                physical_page..physical_page + 1,
                0..len,
                0..num_kv_heads,
                0..head_dim,
            ],
            &v_page,
        )?;
    }

    let batch_indices = Tensor::zeros((append_len,), DType::U32, device)?;
    let positions = Tensor::from_vec(
        (prefix_len as u32..kv_len as u32).collect::<Vec<_>>(),
        (append_len,),
        device,
    )?;
    let scales = if fp8_kv {
        Some(Tensor::ones((num_kv_heads,), DType::F32, device)?)
    } else {
        None
    };
    flashinfer::append_kv_cache(
        &k_all.narrow(0, prefix_len, append_len)?,
        &v_all.narrow(0, prefix_len, append_len)?,
        &k_cache,
        &v_cache,
        scales.as_ref(),
        scales.as_ref(),
        &indices,
        &indptr,
        &last_len,
        Some(&batch_indices),
        Some(&positions),
    )?;

    let q_cu_seqlens_host = vec![0u32, append_len as u32];
    let plan = flashinfer::prefill_plan(
        device,
        &q_cu_seqlens_host,
        &indptr_host,
        &[kv_len as u32],
        append_len as u32,
        1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        None,
        Some(cache_dtype),
        false,
    )?;
    let q_cu_seqlens = Tensor::from_vec(q_cu_seqlens_host, (2,), device)?;
    let rust_output = flashinfer::prefill_with_plan(
        &q,
        &k_cache,
        &v_cache,
        scales.as_ref(),
        scales.as_ref(),
        &indices,
        &indptr,
        &last_len,
        &q_cu_seqlens,
        append_len as u32,
        page_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        None,
        None,
        &plan,
        false,
    )?;

    let path = output_dir.join(format!(
        "{}{}.bin",
        if fp8_kv { "fp8_" } else { "" },
        case_name
    ));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(b"XINFPROBE1")?;
    for value in [
        prefix_len as u64,
        append_len as u64,
        num_qo_heads as u64,
        num_kv_heads as u64,
        head_dim as u64,
        page_size as u64,
        num_pages as u64,
    ] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.write_all(&(indices_host.len() as u64).to_le_bytes())?;
    for index in indices_host {
        file.write_all(&index.to_le_bytes())?;
    }
    write_tensor(&mut file, &q)?;
    write_tensor(&mut file, &k_all)?;
    write_tensor(&mut file, &v_all)?;
    write_tensor(&mut file, &rust_output)?;
    file.flush()?;
    println!("wrote {}", path.display());
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let output_dir = std::env::var_os("XINFER_PROBE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_flashinfer_probe"));
    std::fs::create_dir_all(&output_dir)?;
    run_case(
        &device,
        &output_dir,
        "prefix0",
        0,
        1024,
        8,
        2,
        128,
        64,
        false,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix65",
        65,
        257,
        8,
        2,
        128,
        64,
        false,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix4097",
        4097,
        257,
        8,
        2,
        128,
        64,
        false,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix32769",
        32769,
        257,
        8,
        2,
        128,
        64,
        false,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix65536",
        65536,
        64,
        32,
        4,
        128,
        64,
        false,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix65",
        65,
        257,
        8,
        2,
        128,
        64,
        true,
    )?;
    run_case(
        &device,
        &output_dir,
        "prefix4097",
        4097,
        257,
        8,
        2,
        128,
        64,
        true,
    )?;
    Ok(())
}
