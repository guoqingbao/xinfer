//! Export attention.rs GDN/DeltaNet CUDA kernels for independent PyTorch
//! differential testing.  The Python side intentionally reimplements the
//! recurrences instead of using Candle or another Rust tensor operation.

use anyhow::{Context, Result};
use attention_rs::gdn;
use candle_core::{DType, Device, Tensor};
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

fn random<S: Into<candle_core::Shape>>(device: &Device, shape: S) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(DType::BF16)?)
}

fn record_recurrence(
    file: &mut std::fs::File,
    device: &Device,
    tag: &str,
    bh: usize,
    seq: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<()> {
    let q = random(device, (bh, seq, k_dim))?;
    let k = random(device, (bh, seq, k_dim))?;
    let v = random(device, (bh, seq, v_dim))?;
    let g = Tensor::randn(-1.0f32, 0.25f32, (bh, seq), device)?;
    let beta = sigmoid(&Tensor::randn(0.0f32, 0.2f32, (bh, seq), device)?)?;
    let state = Tensor::randn(0.0f32, 0.1f32, (bh, k_dim, v_dim), device)?;

    record(file, &format!("{tag}_q"), &q)?;
    record(file, &format!("{tag}_k"), &k)?;
    record(file, &format!("{tag}_v"), &v)?;
    record(file, &format!("{tag}_g"), &g)?;
    record(file, &format!("{tag}_beta"), &beta)?;
    record(file, &format!("{tag}_state_initial"), &state)?;
    let mut state_mut = state;
    let out = gdn::gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_mut)?;
    record(file, &format!("{tag}_out"), &out)?;
    record(file, &format!("{tag}_state_final"), &state_mut)?;
    Ok(())
}

fn record_conv(
    file: &mut std::fs::File,
    device: &Device,
    tag: &str,
    batch: usize,
    lengths: &[usize],
    d: usize,
    kernel: usize,
) -> Result<()> {
    let total: usize = lengths.iter().sum();
    let x = random(device, (total, d))?;
    let weight = random(device, (d, 1, kernel))?;
    let bias = random(device, (d,))?;
    let state = Tensor::randn(0.0f32, 0.1f32, (batch, d, kernel - 1), device)?;
    let cu = Tensor::from_vec(
        std::iter::once(0u32)
            .chain(lengths.iter().scan(0u32, |acc, &n| {
                *acc += n as u32;
                Some(*acc)
            }))
            .collect::<Vec<_>>(),
        (batch + 1,),
        device,
    )?;
    record(file, &format!("{tag}_x"), &x)?;
    record(file, &format!("{tag}_weight"), &weight)?;
    record(file, &format!("{tag}_bias"), &bias)?;
    record(file, &format!("{tag}_state_initial"), &state)?;
    record(file, &format!("{tag}_cu"), &cu)?;
    let mut state_mut = state;
    let out = gdn::causal_conv1d_fwd(
        &x,
        &weight,
        Some(&bias),
        &mut state_mut,
        None,
        Some(&cu),
        true,
    )?;
    record(file, &format!("{tag}_out"), &out)?;
    record(file, &format!("{tag}_state_final"), &state_mut)?;
    Ok(())
}

fn record_conv_slots(file: &mut std::fs::File, device: &Device) -> Result<()> {
    let batch = 3;
    let max_slots = 5;
    let d = 7;
    let kernel = 4;
    let x = random(device, (batch, d))?;
    let weight = random(device, (d, 1, kernel))?;
    let bias = random(device, (d,))?;
    let state = Tensor::randn(0.0f32, 0.1f32, (max_slots, d, kernel - 1), device)?;
    let slots = Tensor::from_vec(vec![4i64, 1, 3], (batch,), device)?;
    record(file, "conv_slots_x", &x)?;
    record(file, "conv_slots_weight", &weight)?;
    record(file, "conv_slots_bias", &bias)?;
    record(file, "conv_slots_state_initial", &state)?;
    record(file, "conv_slots_slots", &slots)?;
    let mut state_mut = state;
    let out =
        gdn::causal_conv1d_update_slots(&x, &weight, Some(&bias), &mut state_mut, &slots, true)?;
    record(file, "conv_slots_out", &out)?;
    record(file, "conv_slots_state_final", &state_mut)?;
    Ok(())
}

fn record_varlen(file: &mut std::fs::File, device: &Device) -> Result<()> {
    let lengths = [5usize, 8];
    let total = lengths.iter().sum::<usize>();
    let heads = 2;
    let k_dim = 64;
    let v_dim = 24;
    let q = random(device, (total, heads, k_dim))?;
    let k = random(device, (total, heads, k_dim))?;
    let v = random(device, (total, heads, v_dim))?;
    let g = Tensor::randn(-1.0f32, 0.25f32, (total, heads), device)?;
    let beta = sigmoid(&Tensor::randn(0.0f32, 0.2f32, (total, heads), device)?)?;
    let state = Tensor::randn(0.0f32, 0.1f32, (3, heads, k_dim, v_dim), device)?;
    let slots = Tensor::from_vec(vec![2i64, 0], (2,), device)?;
    let cu = Tensor::from_vec(vec![0u32, 5, 13], (3,), device)?;
    let snapshots = Tensor::zeros((total, heads, k_dim, v_dim), DType::F32, device)?;
    for (name, tensor) in [
        ("varlen_q", &q),
        ("varlen_k", &k),
        ("varlen_v", &v),
        ("varlen_g", &g),
        ("varlen_beta", &beta),
        ("varlen_state_initial", &state),
        ("varlen_slots", &slots),
        ("varlen_cu", &cu),
    ] {
        record(file, name, tensor)?;
    }
    let mut state_mut = state;
    let out = gdn::gated_delta_rule_recurrence_varlen(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &mut state_mut,
        &slots,
        &cu,
        Some(&snapshots),
    )?;
    record(file, "varlen_out", &out)?;
    record(file, "varlen_state_final", &state_mut)?;
    record(file, "varlen_snapshots", &snapshots)?;
    Ok(())
}

fn record_gqa(file: &mut std::fs::File, device: &Device) -> Result<()> {
    let lengths = [4usize, 7];
    let total = lengths.iter().sum::<usize>();
    let nk = 2;
    let nv = 4;
    let k_dim = 64;
    // FlashInfer's SM90 delta-rule kernel requires k_dim == v_dim. The
    // separate flat recurrence/decode cases above still cover sub-64 value
    // widths that exposed the shared-memory barrier bug.
    let v_dim = 64;
    // Match the model path: q/k are L2-normalized before the delta rule.
    let q = gdn::l2_norm_last_dim(&random(device, (total, nk, k_dim))?, 1e-6)?;
    let k = gdn::l2_norm_last_dim(&random(device, (total, nk, k_dim))?, 1e-6)?;
    let v = random(device, (total, nv, v_dim))?;
    let g = Tensor::randn(-1.0f32, 0.25f32, (total, nv), device)?;
    let beta = sigmoid(&Tensor::randn(0.0f32, 0.2f32, (total, nv), device)?)?;
    let state = Tensor::randn(0.0f32, 0.1f32, (4, nv, k_dim, v_dim), device)?;
    let slots = Tensor::from_vec(vec![3i64, 1], (2,), device)?;
    let cu = Tensor::from_vec(vec![0u32, 4, 11], (3,), device)?;
    let snapshots = Tensor::zeros((total, nv, k_dim, v_dim), DType::F32, device)?;
    for (name, tensor) in [
        ("gqa_q", &q),
        ("gqa_k", &k),
        ("gqa_v", &v),
        ("gqa_g", &g),
        ("gqa_beta", &beta),
        ("gqa_state_initial", &state),
        ("gqa_slots", &slots),
        ("gqa_cu", &cu),
    ] {
        record(file, name, tensor)?;
    }
    let mut state_mut = state.copy()?;
    let mut flash_state = state.copy()?;
    let out = gdn::gated_delta_rule_recurrence_varlen_gqa(
        &q,
        &k,
        &v,
        &g,
        &beta,
        &mut state_mut,
        &slots,
        &cu,
        0.7,
        Some(&snapshots),
    )?;
    record(file, "gqa_out", &out)?;
    record(file, "gqa_state_final", &state_mut)?;
    record(file, "gqa_snapshots", &snapshots)?;

    // The persistent SM90 FlashInfer path is opt-in in production and is not
    // part of the default precision suite. Enable it explicitly when auditing
    // that separate path.
    if std::env::var("XINFER_GDN_PROBE_INCLUDE_FLASHINFER").as_deref() == Ok("1") {
        let g_exp = g.exp()?;
        if let Some(flash_out) = gdn::gated_delta_rule_prefill_flashinfer_gqa(
            &q,
            &k,
            &v,
            &g_exp,
            &beta,
            &mut flash_state,
            &slots,
            &cu,
            0.7,
        )? {
            record(file, "gqa_flashinfer_out", &flash_out)?;
            record(file, "gqa_flashinfer_state_final", &flash_state)?;
        }
    }

    // Decode the same GQA state layout through the slot-indexed path.
    let qd = random(device, (2, nk, k_dim))?;
    let kd = random(device, (2, nk, k_dim))?;
    let vd = random(device, (2, nv, v_dim))?;
    let gd = Tensor::randn(-1.0f32, 0.25f32, (2, nv), device)?;
    let betad = sigmoid(&Tensor::randn(0.0f32, 0.2f32, (2, nv), device)?)?;
    let stated = Tensor::randn(0.0f32, 0.1f32, (4, nv, k_dim, v_dim), device)?;
    let slotsd = Tensor::from_vec(vec![3i64, 1], (2,), device)?;
    for (name, tensor) in [
        ("decode_q", &qd),
        ("decode_k", &kd),
        ("decode_v", &vd),
        ("decode_g", &gd),
        ("decode_beta", &betad),
        ("decode_state_initial", &stated),
        ("decode_slots", &slotsd),
    ] {
        record(file, name, tensor)?;
    }
    let mut stated_mut = stated;
    let outd = gdn::gated_delta_rule_decode_slots_gqa(
        &qd,
        &kd,
        &vd,
        &gd,
        &betad,
        &mut stated_mut,
        &slotsd,
        0.7,
    )?;
    record(file, "decode_out", &outd)?;
    record(file, "decode_state_final", &stated_mut)?;
    Ok(())
}

fn record_decode_flat(file: &mut std::fs::File, device: &Device) -> Result<()> {
    let batch = 2;
    let heads = 2;
    let k_dim = 64;
    let v_dim = 24;
    let q = random(device, (batch, heads, k_dim))?;
    let k = random(device, (batch, heads, k_dim))?;
    let v = random(device, (batch, heads, v_dim))?;
    let g = Tensor::randn(-1.0f32, 0.25f32, (batch, heads), device)?;
    let beta = sigmoid(&Tensor::randn(0.0f32, 0.2f32, (batch, heads), device)?)?;
    let state = Tensor::randn(0.0f32, 0.1f32, (4, heads, k_dim, v_dim), device)?;
    let slots = Tensor::from_vec(vec![3i64, 1], (batch,), device)?;
    for (name, tensor) in [
        ("flat_decode_q", &q),
        ("flat_decode_k", &k),
        ("flat_decode_v", &v),
        ("flat_decode_g", &g),
        ("flat_decode_beta", &beta),
        ("flat_decode_state_initial", &state),
        ("flat_decode_slots", &slots),
    ] {
        record(file, name, tensor)?;
    }
    let mut state_mut = state;
    let out = gdn::gated_delta_rule_decode_slots(&q, &k, &v, &g, &beta, &mut state_mut, &slots)?;
    record(file, "flat_decode_out", &out)?;
    record(file, "flat_decode_state_final", &state_mut)?;
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let output = std::env::var_os("XINFER_GDN_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_gdn_probe.bin"));
    let mut file = std::fs::File::create(&output)?;
    file.write_all(b"XINFGDN1")?;

    let heads = 6;
    let seq = 19;
    let a_log = Tensor::randn(-0.5f32, 0.4f32, (heads,), &device)?;
    let dt_bias = Tensor::randn(0.0f32, 1.0f32, (heads,), &device)?;
    let a = random(&device, (2, seq, heads))?;
    let b = random(&device, (2, seq, heads))?;
    let (g, beta) = gdn::fused_gdn_gating(&a_log, &a, &b, &dt_bias)?;
    for (name, tensor) in [
        ("gating_a_log", &a_log),
        ("gating_dt_bias", &dt_bias),
        ("gating_a", &a),
        ("gating_b", &b),
        ("gating_g", &g),
        ("gating_beta", &beta),
    ] {
        record(&mut file, name, tensor)?;
    }

    let norm_x = random(&device, (9, 32))?;
    let norm_z = random(&device, (9, 32))?;
    let norm_w_group = random(&device, (8,))?;
    let norm_b_group = random(&device, (8,))?;
    let norm_w_full = Tensor::randn(1.0f32, 0.1f32, (32,), &device)?;
    let norm_b_full = Tensor::randn(0.0f32, 0.1f32, (32,), &device)?;
    let norm_group = gdn::gated_rmsnorm_silu_mul(
        &norm_x,
        &norm_z,
        &norm_w_group,
        Some(&norm_b_group),
        1e-5,
        8,
    )?;
    let norm_full =
        gdn::gated_rmsnorm_silu_mul(&norm_x, &norm_z, &norm_w_full, Some(&norm_b_full), 1e-5, 8)?;
    let l2 = gdn::l2_norm_last_dim(&norm_x, 1e-6)?;
    for (name, tensor) in [
        ("norm_x", &norm_x),
        ("norm_z", &norm_z),
        ("norm_w_group", &norm_w_group),
        ("norm_b_group", &norm_b_group),
        ("norm_w_full", &norm_w_full),
        ("norm_b_full", &norm_b_full),
        ("norm_group", &norm_group),
        ("norm_full", &norm_full),
        ("l2_out", &l2),
    ] {
        record(&mut file, name, tensor)?;
    }

    record_conv(&mut file, &device, "conv", 2, &[7, 10], 9, 3)?;
    record_conv(&mut file, &device, "conv_k2", 2, &[4, 6], 5, 2)?;
    record_conv(&mut file, &device, "conv_k4", 2, &[4, 6], 5, 4)?;
    record_conv_slots(&mut file, &device)?;
    record_recurrence(&mut file, &device, "rec_k16", 2, 19, 16, 24)?;
    record_recurrence(&mut file, &device, "rec_k64", 1, 13, 64, 24)?;
    record_recurrence(&mut file, &device, "rec_k80", 1, 13, 80, 24)?;
    record_varlen(&mut file, &device)?;
    record_gqa(&mut file, &device)?;
    record_decode_flat(&mut file, &device)?;

    file.flush()?;
    println!("wrote {}", output.display());
    Ok(())
}
