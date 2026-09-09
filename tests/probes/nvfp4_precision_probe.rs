//! Path-complete NVFP4 probe.
//!
//! The probe exports every input and intermediate observable from the
//! attention.rs public/FFI surface. The Python checker is the reference: it
//! does not use Candle or attention.rs for the golden GEMM.
//!
//! On SM70/SM75/SM90 this runs the software decode, software prefill, grouped
//! MoE, quantization-helper, and MLX paths. On SM100/SM120 it additionally
//! runs dense FlashInfer/CUTLASS and grouped MoE CUTLASS paths. Hardware cases
//! are omitted on older GPUs.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "cuda")]
use attention_rs::kernels::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Storage, Tensor};
#[cfg(feature = "cuda")]
use std::io::Write;
#[cfg(feature = "cuda")]
use std::path::PathBuf;

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn cuda_ptr(tensor: &Tensor, dtype: DType) -> Result<u64> {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(c) => Ok(match dtype {
            DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
            DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            DType::F32 => *c.as_cuda_slice::<f32>()?.device_ptr(),
            DType::U8 => *c.as_cuda_slice::<u8>()?.device_ptr(),
            DType::U32 => *c.as_cuda_slice::<u32>()?.device_ptr(),
            _ => anyhow::bail!("unsupported CUDA probe dtype {dtype:?}"),
        }),
        _ => anyhow::bail!("probe tensor is not CUDA resident"),
    }
}

#[cfg(feature = "cuda")]
fn f32_values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let v = ((i * 37 + salt * 19) % 257) as f32 - 128.0;
            if (i + salt) % 97 == 0 {
                0.0
            } else {
                v / 32.0
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn make_weight(device: &Device, e: usize, n: usize, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let codes = [
        0x0u8, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
    ];
    let weight = (0..e * n * (k / 2))
        .map(|i| codes[i % codes.len()] | (codes[(i + 5) % codes.len()] << 4))
        .collect::<Vec<_>>();
    let scale_codes = [0x38u8, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70];
    let scales = (0..e * n * (k / 16))
        .map(|i| scale_codes[(i * 3) % scale_codes.len()])
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_vec(weight, (e, n, k / 2), device)?,
        Tensor::from_vec(scales, (e, n, k / 16), device)?,
        Tensor::from_vec(
            (0..e).map(|i| 0.75 + i as f32 * 0.25).collect::<Vec<_>>(),
            (e,),
            device,
        )?,
    ))
}

#[cfg(feature = "cuda")]
fn run_dense(
    file: &mut std::fs::File,
    device: &Device,
    dtype: DType,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    weight: &Tensor,
    scales: &Tensor,
) -> Result<()> {
    let input = Tensor::from_vec(f32_values(m * k, m + n), (m, k), device)?.to_dtype(dtype)?;
    let w = weight.narrow(0, 0, 1)?.squeeze(0)?;
    let s = scales.narrow(0, 0, 1)?.squeeze(0)?;
    let out =
        attention_rs::nvfp4_linear::nvfp4_matmul(&input, &w, &s, 1.25, 1.0, None, m >= 32, None, None)?;
    record(file, &format!("{name}/input"), &input)?;
    record(file, &format!("{name}/weight_u8"), &w)?;
    record(file, &format!("{name}/weight_scale_u8"), &s)?;
    record(file, &format!("{name}/output"), &out)
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
fn run_direct_hardware_dense(
    file: &mut std::fs::File,
    device: &Device,
    dtype: DType,
    name: &str,
    weight: &Tensor,
    scales: &Tensor,
) -> Result<()> {
    let cuda = device.as_cuda_device()?;
    let m = 128usize;
    let weight = weight.narrow(0, 0, 1)?.squeeze(0)?;
    let scales = scales.narrow(0, 0, 1)?.squeeze(0)?;
    let n = weight.dims()[0];
    let k = weight.dims()[1] * 2;
    let ksc = k / 16;
    let kp = ksc.div_ceil(4) * 4;
    let input = Tensor::from_vec(f32_values(m * k, 91), (m, k), device)?.to_dtype(dtype)?;
    let packed = Tensor::zeros((m, k / 2), DType::U8, device)?;
    let act_scales = Tensor::zeros((m.div_ceil(128) * 128, ksc), DType::U8, device)?;
    let act_swizzled = Tensor::zeros((m.div_ceil(128) * 128, kp), DType::U8, device)?;
    let w_swizzled = attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(&scales)?;
    let alpha = Tensor::new(&[1.25f32], device)?;
    let stream = *cuda.cu_stream() as i64;
    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_quantize_activation_f16(
                cuda_ptr(&input, dtype)? as *const _,
                cuda_ptr(&packed, DType::U8)? as *mut _,
                cuda_ptr(&act_scales, DType::U8)? as *mut _,
                cuda_ptr(&act_swizzled, DType::U8)? as *mut _,
                1.0,
                m as i32,
                k as i32,
                m as i32,
                kp as i32,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_quantize_activation_bf16(
                cuda_ptr(&input, dtype)? as *const _,
                cuda_ptr(&packed, DType::U8)? as *mut _,
                cuda_ptr(&act_scales, DType::U8)? as *mut _,
                cuda_ptr(&act_swizzled, DType::U8)? as *mut _,
                1.0,
                m as i32,
                k as i32,
                m as i32,
                kp as i32,
                stream,
            ),
            _ => anyhow::bail!("hardware dense probe requires F16/BF16"),
        }
    }
    let (workspace, workspace_bytes) = attention_rs::workspace::get_cutlass_workspace(cuda, 0)?;
    for flashinfer in [false, true] {
        let output = Tensor::zeros((m, n), dtype, device)?;
        let global = cuda_ptr(&alpha, DType::F32)? as *const f32;
        let args = (
            cuda_ptr(&packed, DType::U8)? as *const _,
            cuda_ptr(&weight, DType::U8)? as *const _,
            cuda_ptr(&act_swizzled, DType::U8)? as *const _,
            cuda_ptr(&w_swizzled, DType::U8)? as *const _,
            global,
            cuda_ptr(&output, dtype)? as *mut _,
            m as i32,
            n as i32,
            k as i32,
            workspace,
            workspace_bytes as i64,
            stream,
        );
        unsafe {
            match (flashinfer, dtype) {
                (false, DType::F16) => ffi::nvfp4_cutlass_gemm_f16(
                    args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
                    args.10, args.11,
                ),
                (false, DType::BF16) => ffi::nvfp4_cutlass_gemm_bf16(
                    args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
                    args.10, args.11,
                ),
                (true, DType::F16) => ffi::flashinfer_nvfp4_cutlass_gemm_f16(
                    args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
                    args.10, args.11,
                ),
                (true, DType::BF16) => ffi::flashinfer_nvfp4_cutlass_gemm_bf16(
                    args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
                    args.10, args.11,
                ),
                _ => unreachable!(),
            }
        }
        // These are direct FFI launches rather than Candle CustomOps. Make
        // the probe observe completed kernel writes before exporting them;
        // production callers keep the same stream asynchronous.
        device.synchronize()?;
        let case = format!(
            "{name}_{}",
            if flashinfer { "flashinfer" } else { "cutlass" }
        );
        record(file, &format!("{case}/input"), &input)?;
        record(file, &format!("{case}/weight_u8"), &weight)?;
        record(file, &format!("{case}/weight_scale_u8"), &scales)?;
        record(file, &format!("{case}/act_packed_u8"), &packed)?;
        record(file, &format!("{case}/act_scale_u8"), &act_scales)?;
        record(
            file,
            &format!("{case}/act_scale_swizzled_u8"),
            &act_swizzled,
        )?;
        record(
            file,
            &format!("{case}/weight_scale_swizzled_u8"),
            &w_swizzled,
        )?;
        record(file, &format!("{case}/output"), &output)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_moe(
    file: &mut std::fs::File,
    device: &Device,
    dtype: DType,
    name: &str,
    is_prefill: bool,
) -> Result<()> {
    let (tokens, topk, experts, n, k) = (6usize, 2usize, 3usize, 64usize, 256usize);
    let input =
        Tensor::from_vec(f32_values(tokens * k, 17), (tokens, k), device)?.to_dtype(dtype)?;
    let (weights, scales, global_scales) = make_weight(device, experts, n, k)?;
    let input_scales = Tensor::from_vec(vec![1.0f32, 1.25, 0.875], (experts,), device)?;
    let indices = Tensor::from_vec(
        vec![0u32, 1, 1, 2, 2, 0, 0, 2, 1, 0, 2, 1],
        (tokens, topk),
        device,
    )?;
    let topk_weights = Tensor::from_vec(
        vec![
            0.7f32, 0.3, 0.6, 0.4, 0.8, 0.2, 0.55, 0.45, 0.65, 0.35, 0.75, 0.25,
        ],
        (tokens, topk),
        device,
    )?;
    let output = attention_rs::moe::moe_gemm_nvfp4(
        &input,
        &weights,
        &scales,
        &global_scales,
        Some(&input_scales),
        None,
        &indices,
        None,
        is_prefill,
        Some(&topk_weights),
        None,
        None,
    )?;
    for (suffix, tensor) in [
        ("input", &input),
        ("weight_u8", &weights),
        ("weight_scale_u8", &scales),
        ("weight_global_scale", &global_scales),
        ("input_scale", &input_scales),
        ("indices", &indices),
        ("topk_weights", &topk_weights),
        ("output", &output),
    ] {
        record(file, &format!("{name}/{suffix}"), tensor)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_auxiliary_helpers(
    file: &mut std::fs::File,
    device: &Device,
    dtype: DType,
    label: &str,
) -> Result<()> {
    let input = Tensor::from_vec(f32_values(6 * 256, 43), (6, 256), device)?.to_dtype(dtype)?;
    let (online_scale, online_inv) =
        attention_rs::nvfp4_linear::compute_online_input_scale(&input)?;
    let online = Tensor::new(&[online_scale, online_inv], device)?;
    let (_, rank3_scales, _) = make_weight(device, 3, 64, 256)?;
    let rank3_swizzled = attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(&rank3_scales)?;

    let sorted_ids = [0u32, 5, 6, 9, 1, 2, 8, 11, 3, 4, 7, 10];
    let expert_offsets = [0u32, 4, 8, 12];
    let sf_offsets = [0u32, 128, 256];
    let input_scale_invs = [1.0f32, 0.8, 1.0 / 0.875];
    let sorted_ids_t = Tensor::from_vec(sorted_ids.to_vec(), (12,), device)?;
    let expert_offsets_t = Tensor::from_vec(expert_offsets.to_vec(), (4,), device)?;
    let sf_offsets_t = Tensor::from_vec(sf_offsets.to_vec(), (3,), device)?;
    let input_scale_invs_t = Tensor::from_vec(input_scale_invs.to_vec(), (3,), device)?;
    let gathered = Tensor::zeros((12, 256), dtype, device)?;
    let packed = Tensor::zeros((12, 128), DType::U8, device)?;
    let grouped_swizzled = Tensor::zeros((384, 16), DType::U8, device)?;
    let stream = *device.as_cuda_device()?.cu_stream() as i64;

    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_moe_gather_f16(
                cuda_ptr(&input, dtype)? as *const _,
                cuda_ptr(&gathered, dtype)? as *mut _,
                cuda_ptr(&sorted_ids_t, DType::U32)? as *const i32,
                12,
                256,
                2,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_moe_gather_bf16(
                cuda_ptr(&input, dtype)? as *const _,
                cuda_ptr(&gathered, dtype)? as *mut _,
                cuda_ptr(&sorted_ids_t, DType::U32)? as *const i32,
                12,
                256,
                2,
                stream,
            ),
            _ => anyhow::bail!("unsupported auxiliary dtype"),
        }
        match dtype {
            DType::F16 => ffi::nvfp4_quantize_activation_grouped_f16(
                cuda_ptr(&gathered, dtype)? as *const _,
                cuda_ptr(&packed, DType::U8)? as *mut _,
                cuda_ptr(&grouped_swizzled, DType::U8)? as *mut _,
                cuda_ptr(&input_scale_invs_t, DType::F32)? as *const f32,
                cuda_ptr(&expert_offsets_t, DType::U32)? as *const i32,
                cuda_ptr(&sf_offsets_t, DType::U32)? as *const i32,
                12,
                3,
                256,
                16,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_quantize_activation_grouped_bf16(
                cuda_ptr(&gathered, dtype)? as *const _,
                cuda_ptr(&packed, DType::U8)? as *mut _,
                cuda_ptr(&grouped_swizzled, DType::U8)? as *mut _,
                cuda_ptr(&input_scale_invs_t, DType::F32)? as *const f32,
                cuda_ptr(&expert_offsets_t, DType::U32)? as *const i32,
                cuda_ptr(&sf_offsets_t, DType::U32)? as *const i32,
                12,
                3,
                256,
                16,
                stream,
            ),
            _ => anyhow::bail!("unsupported auxiliary dtype"),
        }
    }

    let scatter_in =
        Tensor::from_vec(f32_values(12 * 64, 47), (12, 64), device)?.to_dtype(dtype)?;
    let scatter_out = Tensor::zeros((12, 64), dtype, device)?;
    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_moe_scatter_f16(
                cuda_ptr(&scatter_in, dtype)? as *const _,
                cuda_ptr(&scatter_out, dtype)? as *mut _,
                cuda_ptr(&sorted_ids_t, DType::U32)? as *const i32,
                12,
                64,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_moe_scatter_bf16(
                cuda_ptr(&scatter_in, dtype)? as *const _,
                cuda_ptr(&scatter_out, dtype)? as *mut _,
                cuda_ptr(&sorted_ids_t, DType::U32)? as *const i32,
                12,
                64,
                stream,
            ),
            _ => anyhow::bail!("unsupported auxiliary dtype"),
        }
    }

    let global = Tensor::from_vec(vec![0.75f32, 1.0, 1.25], (3,), device)?;
    let input_scales = Tensor::from_vec(vec![1.0f32, 1.25, 0.875], (3,), device)?;
    let sf_out = Tensor::zeros((3,), DType::U32, device)?;
    let problem = Tensor::zeros((9,), DType::U32, device)?;
    let alphas = Tensor::zeros((3,), DType::F32, device)?;
    let inv_out = Tensor::zeros((3,), DType::F32, device)?;
    unsafe {
        ffi::nvfp4_moe_build_metadata(
            cuda_ptr(&expert_offsets_t, DType::U32)? as *const i32,
            cuda_ptr(&global, DType::F32)? as *const f32,
            cuda_ptr(&input_scales, DType::F32)? as *const f32,
            cuda_ptr(&sf_out, DType::U32)? as *mut i32,
            cuda_ptr(&problem, DType::U32)? as *mut i32,
            cuda_ptr(&alphas, DType::F32)? as *mut f32,
            cuda_ptr(&inv_out, DType::F32)? as *mut f32,
            3,
            64,
            256,
            stream,
        );
    }

    // All helper calls above are raw same-stream launches. Synchronization is
    // intentional here because the probe is reading device outputs to disk.
    device.synchronize()?;

    let prefix = format!("aux_{label}");
    for (suffix, tensor) in [
        ("input", &input),
        ("online_scale", &online),
        ("rank3_scale_u8", &rank3_scales),
        ("rank3_swizzled_u8", &rank3_swizzled),
        ("sorted_ids", &sorted_ids_t),
        ("expert_offsets", &expert_offsets_t),
        ("sf_offsets", &sf_offsets_t),
        ("input_scale_invs", &input_scale_invs_t),
        ("gathered", &gathered),
        ("grouped_packed_u8", &packed),
        ("grouped_swizzled_u8", &grouped_swizzled),
        ("scatter_input", &scatter_in),
        ("scatter_output", &scatter_out),
        ("metadata_sf_offsets", &sf_out),
        ("metadata_problem_sizes", &problem),
        ("metadata_alphas", &alphas),
        ("metadata_input_scale_invs", &inv_out),
    ] {
        record(file, &format!("{prefix}/{suffix}"), tensor)?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
fn run_hardware_moe(file: &mut std::fs::File, device: &Device) -> Result<()> {
    let (tokens, topk, experts, n, k) = (6usize, 2usize, 3usize, 128usize, 256usize);
    let input =
        Tensor::from_vec(f32_values(tokens * k, 29), (tokens, k), device)?.to_dtype(DType::BF16)?;
    let (weights, scales, global_scales) = make_weight(device, experts, n, k)?;
    let input_scales = Tensor::from_vec(vec![1.0f32, 1.25, 0.875], (experts,), device)?;
    let topk_weights = Tensor::from_vec(
        vec![
            0.7f32, 0.3, 0.6, 0.4, 0.8, 0.2, 0.55, 0.45, 0.65, 0.35, 0.75, 0.25,
        ],
        (tokens, topk),
        device,
    )?;
    let ids = [0u32, 1, 1, 2, 2, 0, 0, 2, 1, 0, 2, 1];
    let indices = Tensor::from_vec(ids.to_vec(), (tokens, topk), device)?;
    let mut routed = (0..ids.len())
        .map(|i| (ids[i], i as u32))
        .collect::<Vec<_>>();
    routed.sort_by_key(|x| (x.0, x.1));
    let sorted_tokens = Tensor::from_vec(
        routed.iter().map(|x| x.1).collect::<Vec<_>>(),
        (ids.len(),),
        device,
    )?;
    let sorted_experts = Tensor::from_vec(
        routed.iter().map(|x| x.0).collect::<Vec<_>>(),
        (ids.len(),),
        device,
    )?;
    let topk_option = Some(topk_weights.clone());
    let output = attention_rs::moe::moe_gemm_nvfp4_hardware(
        &input,
        &weights,
        &scales,
        &global_scales,
        Some(&input_scales),
        &topk_option,
        &sorted_tokens,
        &sorted_experts,
        topk,
        true,
        None,
    )?;
    device.synchronize()?;
    for (suffix, tensor) in [
        ("input", &input),
        ("weight_u8", &weights),
        ("weight_scale_u8", &scales),
        ("weight_global_scale", &global_scales),
        ("input_scale", &input_scales),
        ("indices", &indices),
        ("sorted_token_ids", &sorted_tokens),
        ("sorted_expert_ids", &sorted_experts),
        ("topk_weights", &topk_weights),
        ("output", &output),
    ] {
        record(file, &format!("moe_hardware_bf16/{suffix}"), tensor)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_mlx(file: &mut std::fs::File, device: &Device, include_bf16: bool) -> Result<()> {
    let weights = Tensor::from_vec(
        vec![
            0x01234567u32,
            0x89abcdef,
            0xfedcba98,
            0x76543210,
            0xdeadbeefu32,
            0x13579bdf,
            0x2468ace0,
            0x0badcafe,
        ],
        (2, 4),
        device,
    )?;
    let scales = Tensor::from_vec(vec![0x38u8, 0x48, 0x58, 0x68], (2, 2), device)?;
    let packed = attention_rs::nvfp4_linear::mlx_repack_u32_to_u8(&weights)?;
    let f16 =
        attention_rs::nvfp4_linear::mlx_dequant_embedding(&weights, &scales, 2, 32, DType::F16)?;
    for (suffix, tensor) in [
        ("weight_u32", &weights),
        ("scale_u8", &scales),
        ("repacked_u8", &packed),
        ("f16", &f16),
    ] {
        record(file, &format!("mlx/{suffix}"), tensor)?;
    }
    if include_bf16 {
        let bf16 = attention_rs::nvfp4_linear::mlx_dequant_embedding(
            &weights,
            &scales,
            2,
            32,
            DType::BF16,
        )?;
        record(file, "mlx/bf16", &bf16)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let output = std::env::var_os("XINFER_NVFP4_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_nvfp4_probe.bin"));
    let mut file = std::fs::File::create(&output)?;
    file.write_all(b"XINFNV4P2")?;
    let sm = device
        .as_cuda_device()
        .ok()
        .and_then(|d| attention_rs::cuda_utils::sm_version(d))
        .unwrap_or(0);
    let sm_tensor = Tensor::new(&[sm as f32], &device)?;
    record(&mut file, "meta/sm", &sm_tensor)?;

    let (weights, scales, _) = make_weight(&device, 1, 128, 256)?;
    let software_dtypes = if sm < 80 {
        vec![DType::F16]
    } else {
        vec![DType::F16, DType::BF16]
    };
    for dtype in &software_dtypes {
        let label = if *dtype == DType::F16 { "f16" } else { "bf16" };
        run_dense(
            &mut file,
            &device,
            *dtype,
            &format!("dense_{label}_decode_m1"),
            1,
            128,
            256,
            &weights,
            &scales,
        )?;
        run_dense(
            &mut file,
            &device,
            *dtype,
            &format!("dense_{label}_prefill_m32"),
            32,
            128,
            256,
            &weights,
            &scales,
        )?;
        run_dense(
            &mut file,
            &device,
            *dtype,
            &format!("dense_{label}_prefill_m128"),
            128,
            128,
            256,
            &weights,
            &scales,
        )?;
    }
    for dtype in &software_dtypes {
        let label = if *dtype == DType::F16 { "f16" } else { "bf16" };
        // The GEMM/MoE calls below are software CUDA kernels and are
        // intentionally run on SM70/SM75 as well. The auxiliary activation
        // quantizer/metadata FFI is the Blackwell hardware-preparation path;
        // it is only built when ENABLE_FP4 is enabled.
        if sm >= 100 {
            run_auxiliary_helpers(&mut file, &device, *dtype, label)?;
        }
        run_moe(
            &mut file,
            &device,
            *dtype,
            &format!("moe_{label}_decode_indexed"),
            false,
        )?;
        run_moe(
            &mut file,
            &device,
            *dtype,
            &format!("moe_{label}_prefill_wmma"),
            true,
        )?;
    }

    if sm >= 100 {
        #[cfg(all(feature = "flashinfer", feature = "cutlass"))]
        {
            for dtype in [DType::F16, DType::BF16] {
                let label = if dtype == DType::F16 { "f16" } else { "bf16" };
                run_direct_hardware_dense(
                    &mut file,
                    &device,
                    dtype,
                    &format!("dense_hw_{label}"),
                    &weights,
                    &scales,
                )?;
            }
            run_hardware_moe(&mut file, &device)?;
        }
        #[cfg(not(all(feature = "flashinfer", feature = "cutlass")))]
        eprintln!(
            "SM{sm}: hardware NVFP4 requires FlashInfer/CUTLASS; software paths were still tested"
        );
    } else {
        eprintln!("SM{sm}: skipping Blackwell dense CUTLASS/FlashInfer and grouped-MoE execution");
    }
    run_mlx(&mut file, &device, sm >= 80)?;
    file.flush()?;
    println!("wrote {} (SM{sm})", output.display());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    println!("SKIP NVFP4 probe: requires the cuda feature");
    Ok(())
}
