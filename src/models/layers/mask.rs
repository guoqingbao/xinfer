use candle_core::{DType, Device, Tensor};

// Every backend with an in-kernel (flash-style) attention path is listed here.
// Those kernels apply causality and any sliding window themselves, so the
// dense quadratic mask is never read on these builds. On Metal this covers
// the causal prefill SDPA path, the paged prefill kernel and the decode
// kernels; `metal` enables `attention-rs/metal-flash` (see Cargo.toml).
#[cfg(any(
    feature = "flash",
    feature = "flashattn",
    feature = "flashinfer",
    feature = "metal"
))]
pub fn get_attention_causal_mask(
    _: &Device,
    _: DType,
    _: &Tensor,
    _: Vec<u32>,
    _: Option<usize>,
    _: bool,
) -> Option<Vec<Tensor>> {
    None
}

#[allow(unreachable_code)]
#[cfg(not(any(
    feature = "flash",
    feature = "flashattn",
    feature = "flashinfer",
    feature = "metal"
)))]
fn get_causal_mask_internal(
    device: &Device,
    dtype: DType,
    tgt_len: usize,
    sliding_window: Option<usize>,
) -> candle_core::Result<Tensor> {
    use attention_rs::mask::causal_mask;
    let mask = Tensor::zeros((tgt_len, tgt_len), dtype, device)?;
    let _ = causal_mask(&mask, sliding_window)?;
    mask.unsqueeze(0)?.unsqueeze(0)
}

#[cfg(not(any(
    feature = "flash",
    feature = "flashattn",
    feature = "flashinfer",
    feature = "metal"
)))]
pub fn get_attention_causal_mask(
    device: &Device,
    dtype: DType,
    _: &Tensor,
    seqlens: Vec<u32>,
    sliding_window: Option<usize>,
    is_prefill: bool,
) -> Option<Vec<Tensor>> {
    if !is_prefill {
        return None;
    }
    let mut offsets = vec![0u32];
    offsets.extend(seqlens.clone());
    let mut vec_mask = Vec::new();
    let mut start = 0;
    for (_, seq_offset) in seqlens.iter().enumerate() {
        let seq_len = seq_offset - start;
        let mask =
            get_causal_mask_internal(device, dtype, seq_len as usize, sliding_window).unwrap();
        vec_mask.push(mask);
        start = *seq_offset;
    }
    Some(vec_mask)
}
