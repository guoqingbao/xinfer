use candle_core::{DType, Device, Result, Tensor};

fn compressor_scratch_pair(device: &Device, head_dim: usize) -> Result<(Tensor, Tensor)> {
    Ok((
        Tensor::zeros((1, head_dim), DType::F32, device)?
            .transpose(0, 1)?
            .contiguous()?,
        Tensor::zeros((1, head_dim), DType::BF16, device)?
            .transpose(0, 1)?
            .contiguous()?,
    ))
}

/// Fixed-shape decode intermediates allocated once before CUDA graph capture.
/// All kernels write in-place into these buffers (no cudaMalloc during capture).
pub struct LayerDecodeBuffers {
    pub attn_out: Tensor,
    pub window_topk: Tensor,
    pub compress_topk: Tensor,
    pub concat_topk: Tensor,
    pub indexer_scores: Option<Tensor>,
    pub compressor_weighted: Option<Tensor>,
    pub compressor_out: Option<Tensor>,
    pub indexer_compressor_weighted: Option<Tensor>,
    pub indexer_compressor_out: Option<Tensor>,
}

impl LayerDecodeBuffers {
    pub fn new(
        device: &Device,
        num_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        compress_topk: usize,
        max_compressed_len: usize,
        compressor_head_dim: Option<usize>,
        indexer_head_dim: Option<usize>,
    ) -> Result<Self> {
        let compress_slots = compress_topk.max(1);
        let total_topk = (sliding_window + compress_topk).max(1);
        let attn_out = Tensor::zeros((1, num_heads, head_dim), DType::BF16, device)?;
        let window_topk = Tensor::zeros((1, sliding_window), DType::U32, device)?;
        let compress_topk_buf = Tensor::zeros((1, compress_slots), DType::U32, device)?;
        let concat_topk = Tensor::zeros((1, total_topk), DType::U32, device)?;
        let indexer_scores = indexer_head_dim
            .map(|_| Tensor::zeros(max_compressed_len.max(1), DType::F32, device))
            .transpose()?;
        let (compressor_weighted, compressor_out) = if let Some(hd) = compressor_head_dim {
            let (w, o) = compressor_scratch_pair(device, hd)?;
            (Some(w), Some(o))
        } else {
            (None, None)
        };
        let (indexer_compressor_weighted, indexer_compressor_out) =
            if let Some(hd) = indexer_head_dim {
                let (w, o) = compressor_scratch_pair(device, hd)?;
                (Some(w), Some(o))
            } else {
                (None, None)
            };
        Ok(Self {
            attn_out,
            window_topk,
            compress_topk: compress_topk_buf,
            concat_topk,
            indexer_scores,
            compressor_weighted,
            compressor_out,
            indexer_compressor_weighted,
            indexer_compressor_out,
        })
    }
}
