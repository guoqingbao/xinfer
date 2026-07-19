use crate::models::layers::distributed::shard;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Result, Tensor};

/// State saved from hc_pre to be consumed by hc_post
pub struct HcPreState {
    pub post: Tensor,
    pub comb: Tensor,
    pub seq_len: usize,
    pub hc: usize,
    pub hidden_dim: usize,
}

/// HC hidden states: [seq_len, hc, hidden_dim] in BF16
pub struct HcHiddenStates {
    pub data: Tensor,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub hc: usize,
}

/// Weights for a per-layer HC block (wraps either attention or FFN)
pub struct HcBlockWeights {
    pub hc_fn: Tensor,
    pub hc_scale: Tensor,
    pub hc_base: Tensor,
}

impl HcBlockWeights {
    pub fn load(vb: &VarBuilderX, prefix: &str, hc_mult: usize, hidden_dim: usize) -> Result<Self> {
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden_dim;
        let hc_fn = vb.get_with_hints_dtype(
            (mix_hc, hc_dim),
            &format!("{}_fn", prefix),
            shard(0, 0, 1),
            DType::F32,
        )?;
        let hc_scale = vb.get_with_hints_dtype(
            (3,),
            &format!("{}_scale", prefix),
            shard(0, 0, 1),
            DType::F32,
        )?;
        let hc_base = vb.get_with_hints_dtype(
            (mix_hc,),
            &format!("{}_base", prefix),
            shard(0, 0, 1),
            DType::F32,
        )?;
        Ok(Self {
            hc_fn,
            hc_scale,
            hc_base,
        })
    }
}

/// Weights for the final HC head (model-level collapse)
pub struct HcHeadWeights {
    pub hc_fn: Tensor,
    pub hc_scale: Tensor,
    pub hc_base: Tensor,
}

impl HcHeadWeights {
    pub fn load(vb: &VarBuilderX, hc_mult: usize, hidden_dim: usize) -> Result<Self> {
        let hc_dim = hc_mult * hidden_dim;
        let hc_fn =
            vb.get_with_hints_dtype((hc_mult, hc_dim), "hc_head_fn", shard(0, 0, 1), DType::F32)?;
        let hc_scale =
            vb.get_with_hints_dtype((1,), "hc_head_scale", shard(0, 0, 1), DType::F32)?;
        let hc_base =
            vb.get_with_hints_dtype((hc_mult,), "hc_head_base", shard(0, 0, 1), DType::F32)?;
        Ok(Self {
            hc_fn,
            hc_scale,
            hc_base,
        })
    }
}

/// Expand hidden states for HC: replicate hc_mult times via CUDA kernel.
/// Input: [seq_len, hidden_dim] → Output: HcHiddenStates with [seq_len, hc, hidden_dim]
pub fn hc_expand(xs: &Tensor, hc: usize) -> Result<HcHiddenStates> {
    let (seq_len, hidden_dim) = xs.dims2()?;
    let expanded = attention_rs::deepseek_v4::hc_expand(xs, hc)?;
    Ok(HcHiddenStates {
        data: expanded.reshape((seq_len, hc, hidden_dim))?,
        seq_len,
        hidden_dim,
        hc,
    })
}

/// HC pre: compute mixing via cuBLAS GEMM + fused sinkhorn + pre-output.
/// All operations stay on GPU, no CPU-side scalar extraction.
///
/// Returns: (branch_input: [seq_len, hidden_dim], HcPreState)
pub fn hc_pre(
    hc_hidden: &HcHiddenStates,
    hc_fn: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
) -> Result<(Tensor, HcPreState)> {
    let hc = hc_hidden.hc;
    let hidden_dim = hc_hidden.hidden_dim;
    let seq_len = hc_hidden.seq_len;
    let mix_hc = (2 + hc) * hc;

    // Flatten HC hidden states: [seq, hc, dim] -> [seq, hc*dim] for GEMM
    let x_flat = hc_hidden.data.reshape((seq_len, hc * hidden_dim))?;

    // cuBLAS GEMM + RMS scaling: produces mixes [seq, mix_hc]
    let mixes =
        attention_rs::deepseek_v4::hc_mixes(&x_flat, hc_fn, hc, hidden_dim, mix_hc, hc_eps)?;

    // Fused sinkhorn + sigmoid + pre-output (all GPU, hc=4)
    let (branch_input, post, comb) = attention_rs::deepseek_v4::hc_pre_from_mixes(
        &hc_hidden.data.reshape((seq_len, hc, hidden_dim))?,
        &mixes,
        hc_scale,
        hc_base,
        hc,
        hidden_dim,
        hc_sinkhorn_iters,
        hc_eps,
    )?;

    Ok((
        branch_input,
        HcPreState {
            post,
            comb,
            seq_len,
            hc,
            hidden_dim,
        },
    ))
}

/// HC pre with fused RMSNorm: sinkhorn + pre-output + layer norm in one kernel.
///
/// Returns: (normed_branch_input: [seq_len, hidden_dim], HcPreState)
pub fn hc_pre_norm(
    hc_hidden: &HcHiddenStates,
    hc_fn: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    norm_weight: &Tensor,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, HcPreState)> {
    let hc = hc_hidden.hc;
    let hidden_dim = hc_hidden.hidden_dim;
    let seq_len = hc_hidden.seq_len;
    let mix_hc = (2 + hc) * hc;

    let x_flat = hc_hidden.data.reshape((seq_len, hc * hidden_dim))?;
    let mixes =
        attention_rs::deepseek_v4::hc_mixes(&x_flat, hc_fn, hc, hidden_dim, mix_hc, hc_eps)?;

    let (normed_out, post, comb) = attention_rs::deepseek_v4::hc_pre_norm_from_mixes(
        &hc_hidden.data.reshape((seq_len, hc, hidden_dim))?,
        &mixes,
        hc_scale,
        hc_base,
        norm_weight,
        hc,
        hidden_dim,
        hc_sinkhorn_iters,
        hc_eps,
        norm_eps,
    )?;

    Ok((
        normed_out,
        HcPreState {
            post,
            comb,
            seq_len,
            hc,
            hidden_dim,
        },
    ))
}

/// HC post: merge branch output with HC residual via CUDA kernel.
///
/// `branch_out` may be BF16 or F32; residual HC state is BF16. Output HC
/// state is always BF16 so prefill/decode share one downstream dtype.
///
/// branch_out: [seq_len, hidden_dim]
/// residual: HcHiddenStates — HC hidden states before the branch
/// state: HcPreState — post and comb weights from hc_pre
///
/// Returns: HcHiddenStates with updated hidden states
pub fn hc_post(
    branch_out: &Tensor,
    residual: &HcHiddenStates,
    state: &HcPreState,
) -> Result<HcHiddenStates> {
    let hc = state.hc;
    let hidden_dim = residual.hidden_dim;
    let seq_len = state.seq_len;

    let result = attention_rs::deepseek_v4::hc_post(
        branch_out,
        &residual.data.reshape((seq_len, hc, hidden_dim))?,
        &state.post,
        &state.comb,
        hc,
        hidden_dim,
    )?;

    Ok(HcHiddenStates {
        data: result,
        seq_len,
        hidden_dim,
        hc,
    })
}

/// HC head: final collapse from HC hidden states to regular hidden states via CUDA kernels.
///
/// Uses cuBLAS GEMM for mix computation + sigmoid gating + weighted sum.
pub fn hc_head(
    hc_hidden: &HcHiddenStates,
    hc_fn: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
) -> Result<Tensor> {
    let hc = hc_hidden.hc;
    let hidden_dim = hc_hidden.hidden_dim;
    let seq_len = hc_hidden.seq_len;

    // hc_fn for head: [hc, hc*hidden_dim]
    // Compute mixes via cuBLAS GEMM + RMS scale
    let x_flat = hc_hidden.data.reshape((seq_len, hc * hidden_dim))?;
    let mixes = attention_rs::deepseek_v4::hc_mixes(&x_flat, hc_fn, hc, hidden_dim, hc, 1e-6)?;

    // Compute sigmoid pre gates
    let pre = attention_rs::deepseek_v4::hc_head_pre(&mixes, hc_scale, hc_base, hc, 1e-6)?;

    // Weighted sum: output[t,d] = sum_h(pre[t,h] * x[t,h,d])
    let x_3d = hc_hidden.data.reshape((seq_len, hc, hidden_dim))?;
    attention_rs::deepseek_v4::hc_pre_output(&x_3d, &pre, hc, hidden_dim)
}
