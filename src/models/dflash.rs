// DFlash draft model: a small, fully-replicated dense transformer that drafts future tokens by
// reading the target model's hidden states (at `target_layer_ids`) instead of re-reading the KV
// cache. Loaded separately from the primary model (its own safetensors/gguf checkpoint) and
// replicated on every tensor-parallel rank (no `comm`, no NCCL in the draft path).

use crate::models::layers::distributed::ReplicatedLinear;
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor, D};
#[cfg(feature = "cuda")]
use attention_rs::sort::ArgSortOp;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashConfig {
    pub mask_token_id: Option<u32>,
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub selector_rank: Option<usize>,
    #[serde(default)]
    pub selector_top_k: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashModelConfig {
    /// Checkpoint architecture tag: `"DFlashDraftModel"` (v1) vs `"DFlash2DraftModel"` (v2).
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub block_size: Option<usize>,
    pub num_target_layers: usize,
    #[serde(default)]
    pub dflash_config: Option<DFlashConfig>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

/// Newer checkpoints (e.g. DFlash2) nest the RoPE base under `rope_parameters`
/// instead of a top-level `rope_theta`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
}

impl DFlashModelConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.target_layer_ids.clone())
            .unwrap_or_else(|| {
                build_target_layer_ids(self.num_target_layers, self.num_hidden_layers)
            })
    }

    pub fn mask_token_id(&self) -> Option<u32> {
        self.dflash_config.as_ref().and_then(|c| c.mask_token_id)
    }

    /// Resolve the verification block width from either the top-level field or the nested
    /// `dflash_config` (DFlash2 checkpoints store it under `dflash_config`).
    pub fn effective_block_size(&self) -> Option<usize> {
        self.block_size
            .or_else(|| self.dflash_config.as_ref().and_then(|c| c.block_size))
    }

    /// Resolve the RoPE base from either the top-level field or `rope_parameters`
    /// (newer checkpoints nest `rope_theta` under `rope_parameters`).
    pub fn effective_rope_theta(&self) -> Option<f64> {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|rp| rp.rope_theta))
    }

    /// True when the checkpoint carries DFlash2 components (candidate selector and/or grouped
    /// dynamic convs). DFlash1 checkpoints have neither; this is the auto-detection signal for
    /// the kernel backend (the `architectures` field, when present, says "DFlash2DraftModel").
    pub fn has_v2_components(&self) -> bool {
        if self
            .architectures
            .as_ref()
            .and_then(|a| a.first())
            .is_some_and(|a| a.contains("DFlash2"))
        {
            return true;
        }
        self.dflash_config
            .as_ref()
            .is_some_and(|dc| {
                dc.selector_rank.is_some()
                    || dc.selector_top_k.is_some()
                    || (dc.conv_kernel_size.is_some() && dc.conv_group_size.is_some())
            })
    }
}

/// Default target-layer selection for the draft model's cross-attention context.
/// - 1 draft layer -> the middle target layer.
/// - else -> `num_draft_layers` ids evenly spaced in `[1, num_target_layers - 3]`.
pub fn build_target_layer_ids(num_target_layers: usize, num_draft_layers: usize) -> Vec<usize> {
    if num_draft_layers == 1 {
        return vec![num_target_layers / 2];
    }
    let start = 1usize;
    let end = num_target_layers.saturating_sub(3);
    let span = end.saturating_sub(start);
    (0..num_draft_layers)
        .map(|i| start + (i * span) / (num_draft_layers - 1))
        .collect()
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let half = last_dim / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let q_len = q.dim(2)?;
    let cos_len = cos.dim(2)?;

    let cos_q = if cos_len > q_len {
        cos.narrow(2, cos_len - q_len, q_len)?
    } else {
        cos.clone()
    };
    let sin_q = if cos_len > q_len {
        sin.narrow(2, cos_len - q_len, q_len)?
    } else {
        sin.clone()
    };

    let q_embed = (q.broadcast_mul(&cos_q)? + rotate_half(q)?.broadcast_mul(&sin_q)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}

/// Replicated (dense, non-TP-sharded) SiLU-Glu MLP. Weight names match a Qwen-style draft
/// checkpoint (`gate_proj`/`up_proj`/`down_proj`); the gguf aliases are resolved by the loader.
pub struct DFlashMLP {
    gate_proj: ReplicatedLinear,
    up_proj: ReplicatedLinear,
    down_proj: ReplicatedLinear,
}

impl DFlashMLP {
    pub fn new(vb: VarBuilderX, hidden_size: usize, intermediate_size: usize, dtype: DType) -> Result<Self> {
        let gate_proj = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            &None,
            &None,
            dtype,
        )?;
        let up_proj = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            &None,
            &None,
            dtype,
        )?;
        let down_proj = ReplicatedLinear::load_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let activated = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&activated)
    }
}

/// Grouped dynamic causal convolution (DFlash2 `attention_conv` / `mlp_conv`).
/// A `taps`-tap causal depthwise-grouped conv whose per-group kernels are a static base plus an
/// input-dependent projection. `prepare` convolves a sublayer's input and returns the kernel for
/// `finish` to convolve the sublayer's output; both come from one projection of the input.
pub struct GroupedDynamicCausalConv {
    base_kernel: Tensor, // [2, taps, hidden]
    kernel_projection: ReplicatedLinear, // hidden -> 2*taps*num_groups
    num_groups: usize,
    group_size: usize,
    taps: usize,
    block_size: usize,
}

impl GroupedDynamicCausalConv {
    pub fn new(
        vb: &VarBuilderX,
        hidden_size: usize,
        taps: usize,
        group_size: usize,
        block_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let num_groups = hidden_size / group_size;
        let base_kernel = vb
            .get((2, taps, hidden_size), "base_kernel")?
            .to_dtype(dtype)?;
        let kernel_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            2 * taps * num_groups,
            vb.pp("kernel_projection"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self {
            base_kernel,
            kernel_projection,
            num_groups,
            group_size,
            taps,
            block_size,
        })
    }

    fn convolve(&self, hidden: &Tensor, delta: &Tensor, side: usize) -> Result<Tensor> {
        // v2: fused CUDA kernel (block-aware causal reset). `delta` is the side-specific
        // [seq, taps, num_groups] half; the kernel wants the full [2, taps, hidden] base.
        let use_k = crate::utils::env::dflash_use_kernels();
        crate::log_info!(
            "[dflash-debug] conv: side={} seq={} taps={} block={} groups={} use_kernels={}",
            side, hidden.dim(0)?, self.taps, self.block_size, self.num_groups, use_k
        );
        if use_k {
            #[cfg(feature = "cuda")]
            {
                return attention_rs::topk::dflash_grouped_conv(
                    hidden,
                    delta,
                    &self.base_kernel,
                    self.block_size,
                    side,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (hidden, delta, side);
            }
        }
        self.convolve_candle(hidden, delta, side)
    }

    /// Portable candle conv with the same block-aware causal reset as the fused kernel:
    /// tap `t` contributes only where `(pos % block_size) >= t`.
    fn convolve_candle(&self, hidden: &Tensor, delta: &Tensor, side: usize) -> Result<Tensor> {
        let seq = hidden.dim(0)?;
        let blocks = hidden.reshape((seq, self.num_groups, self.group_size))?;
        let base_4d = self
            .base_kernel
            .narrow(0, side, 1)?
            .reshape((1, self.taps, self.num_groups, self.group_size))?;
        let delta_4d = delta.unsqueeze(3)?; // [seq, taps, num_groups, 1]
        let coefficients = base_4d.broadcast_add(&delta_4d)?; // [seq, taps, num_groups, group_size]
        let pos_mod = Tensor::from_vec(
            (0..seq).map(|p| (p % self.block_size) as i64).collect::<Vec<_>>(),
            (seq, 1, 1),
            hidden.device(),
        )?;
        let mut out = (coefficients
            .narrow(1, 0, 1)?
            .reshape((seq, self.num_groups, self.group_size))?
            * &blocks)?;
        for tap in 1..self.taps {
            let prev = blocks.narrow(0, 0, seq.saturating_sub(tap))?; // [seq-tap, ng, gs]
            let pad = Tensor::zeros(
                (tap, self.num_groups, self.group_size),
                hidden.dtype(),
                hidden.device(),
            )?;
            let shifted = Tensor::cat(&[&pad, &prev], 0)?; // [seq, ng, gs]
            let valid = pos_mod.ge(tap as i64)?.to_dtype(hidden.dtype())?; // [seq,1,1]
            let coeff_tap = coefficients
                .narrow(1, tap, 1)?
                .reshape((seq, self.num_groups, self.group_size))?;
            out = (out + (&coeff_tap * &shifted * &valid)?)?;
        }
        out.reshape((seq, self.num_groups * self.group_size))
    }

    /// Convolve the sublayer input; returns `(convolved_input, finish_delta)`.
    pub fn prepare(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = hidden.dim(0)?;
        let delta = self.kernel_projection.forward(hidden)?; // [seq, 2*taps*ng]
        let delta = delta.reshape((seq, 2, self.taps, self.num_groups))?;
        let prep_delta = delta.narrow(1, 0, 1)?.reshape((seq, self.taps, self.num_groups))?;
        let fin_delta = delta.narrow(1, 1, 1)?.reshape((seq, self.taps, self.num_groups))?;
        let convolved = self.convolve(hidden, &prep_delta, 0)?;
        Ok((convolved, fin_delta))
    }

    /// Convolve the sublayer output using the delta returned by `prepare`.
    pub fn finish(&self, hidden: &Tensor, fin_delta: &Tensor) -> Result<Tensor> {
        self.convolve(hidden, fin_delta, 1)
    }
}

/// DFlash2 candidate selector: scores KxK transitions between adjacent proposal slots via a
/// low-rank bilinear form over predecessor/successor codebooks, then walks the greedy path.
pub struct CandidateSelector {
    hidden_projection: ReplicatedLinear, // hidden -> rank
    predecessor_codebook: Tensor, // [vocab, rank]
    successor_codebook: Tensor, // [vocab, rank]
    top_k: usize,
}

impl CandidateSelector {
    pub fn new(
        vb: &VarBuilderX,
        hidden_size: usize,
        vocab_size: usize,
        rank: usize,
        top_k: usize,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            rank,
            vb.pp("hidden_projection"),
            &None,
            &None,
            dtype,
        )?;
        let predecessor_codebook = vb
            .get((vocab_size, rank), "predecessor_codebook")?
            .to_dtype(dtype)?;
        let successor_codebook = vb
            .get((vocab_size, rank), "successor_codebook")?
            .to_dtype(dtype)?;
        Ok(Self {
            hidden_projection,
            predecessor_codebook,
            successor_codebook,
            top_k,
        })
    }

    /// Greedy candidate walk. `logits`/`hidden` are `[n, *]` over the draft positions; `anchor`
    /// is the verified token preceding them. Returns one selected token per position.
    pub fn select(&self, logits: &Tensor, hidden: &Tensor, anchor: u32) -> Result<Vec<u32>> {
        let seq = logits.dim(0)?;
        let k = self.top_k;
        let device = logits.device();

        // Top-K candidate ids + their logit values per position.
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits.to_device(&Device::Cpu)?.sort_last_dim(false)?;
        let candidate_ids = asort.narrow(D::Minus1, 0, k)?.to_dtype(DType::I64)?; // [seq, k]
        let unary = sorted.narrow(D::Minus1, 0, k)?; // [seq, k]

        // Project hidden to the selector rank.
        let hidden_proj = self.hidden_projection.forward(hidden)?; // [seq, rank]

        // Predecessor ids: slot 0 -> anchor (k copies); slot l -> candidate_ids[l-1].
        let anchor_row = Tensor::full(anchor as i64, (1, k), device)?;
        let pred_tail = candidate_ids.narrow(0, 0, seq.saturating_sub(1))?; // [seq-1, k]
        let predecessor_ids = Tensor::cat(&[&anchor_row, &pred_tail], 0)?; // [seq, k]

        let predecessors = self.predecessor_codebook.index_select(&predecessor_ids, 0)?; // [seq, k, rank]
        let keys = self.successor_codebook.index_select(&candidate_ids, 0)?; // [seq, k, rank]

        // score[seq, p, c] = unary[seq, c] + sum_r (predecessors[seq, p, r] * hidden_proj[seq, r]) * keys[seq, c, r]
        let gated = (&predecessors * &hidden_proj.unsqueeze(1)?)?; // [seq, k, rank]
        let bilinear = gated.matmul(&keys.transpose(1, 2)?)?; // [seq, k_pred, k_cand]
        let unary = unary.to_dtype(bilinear.dtype())?;
        let scores = (&bilinear + &unary.unsqueeze(1)?)?; // [seq, k_pred, k_cand]

        // Greedy walk on the (tiny) CPU-side score lattice.
        let scores_cpu: Vec<Vec<Vec<f32>>> = scores
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec3()?;
        let cand_cpu: Vec<Vec<u32>> = candidate_ids
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .to_vec2()?;
        let mut path = Vec::with_capacity(seq);
        let mut prev_cand = 0usize;
        for l in 0..seq {
            let p = if l == 0 { 0 } else { prev_cand };
            let mut best_c = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for c in 0..k {
                let v = scores_cpu[l][p][c];
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            path.push(cand_cpu[l][best_c]);
            prev_cand = best_c;
        }
        Ok(path)
    }

    /// Grammar-gated candidate walk. `allow` is an optional per-position allow matrix
    /// `[seq, vocab]` u8 (1 = legal, 0 = illegal); `None` = unmasked.
    ///
    /// On the v2 (CUDA) backend the gate is applied *inside* the fused kernel
    /// (`dflash_select_candidates_masked`), so no host sync and no pre-masking. On the
    /// v1 / non-CUDA backend the logits are pre-masked (disallowed -> -inf) and the
    /// portable `select` walk runs, which naturally excludes the illegal candidates.
    pub fn select_masked(
        &self,
        logits: &Tensor,
        hidden: &Tensor,
        anchor: u32,
        allow: Option<&Tensor>,
    ) -> Result<Vec<u32>> {
        crate::log_info!(
            "[dflash-debug] selector.select_masked: top_k={} allow={} kernels={}",
            self.top_k,
            allow.is_some(),
            crate::utils::env::dflash_use_kernels()
        );
        if crate::utils::env::dflash_use_kernels() {
            #[cfg(feature = "cuda")]
            {
                let logits = logits.contiguous()?.to_dtype(DType::F32)?;
                let (unary, ids) = attention_rs::topk::topk_select(&logits, self.top_k)?;
                let hp = self.hidden_projection.forward(hidden)?.to_dtype(DType::F32)?;
                let anchor_t = Tensor::from_vec(vec![anchor], (1,), logits.device())?;
                // No grammar gate -> the original unmasked fused kernel (old interface).
                // Active gate -> the masked fused kernel (allow applied in-kernel).
                let selected = match allow {
                    None => attention_rs::topk::dflash_select_candidates(
                        &hp,
                        &unary,
                        &ids,
                        &self.predecessor_codebook,
                        &self.successor_codebook,
                        &anchor_t,
                    )?,
                    Some(a) => {
                        // The fused kernel expects an F32 allow matrix; our builders emit u8.
                        let a_f32 = a.to_dtype(DType::F32)?;
                        attention_rs::topk::dflash_select_candidates_masked(
                            &hp,
                            &unary,
                            &ids,
                            &self.predecessor_codebook,
                            &self.successor_codebook,
                            &anchor_t,
                            Some(&a_f32),
                        )?
                    }
                };
                return Ok(selected.to_vec1::<u32>()?);
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = allow;
            }
        }
        // Portable path: fold the allow gate into the logits, then run the existing walk.
        // The allow matrix is u8 (1 = legal, 0 = illegal); where_cond requires a u8/u32/i64
        // condition, so use it directly (no dtype cast) exactly like GuidedDecoding::mask_rows.
        let masked = match allow {
            Some(a) => {
                let neg_inf = Tensor::full(
                    f32::NEG_INFINITY,
                    logits.shape().clone(),
                    logits.device(),
                )?
                .to_dtype(logits.dtype())?;
                a.where_cond(logits, &neg_inf)?
            }
            None => logits.clone(),
        };
        self.select(&masked, hidden, anchor)
    }
}

pub struct DFlashAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    q_norm: NormX,
    k_norm: NormX,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl DFlashAttention {
    pub fn new(vb: VarBuilderX, config: &DFlashModelConfig, dtype: DType) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
            &None,
            &None,
            dtype,
        )?;
        let k_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            &None,
            &None,
            dtype,
        )?;
        let v_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            &None,
            &None,
            dtype,
        )?;
        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
            &None,
            &None,
            dtype,
        )?;

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("q_norm"),
            DType::F32,
            false,
        )?;
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("k_norm"),
            DType::F32,
            false,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
        })
    }

    /// Cross-attention: Q from the draft/noise tokens, K/V from `concat(target_hidden, draft)`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let q_len = hidden_states.dim(0)?;
        let ctx_len = target_hidden.dim(0)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((1, q_len, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = q.transpose(1, 2)?;

        let k_ctx = self.k_proj.forward(target_hidden)?;
        let k_noise = self.k_proj.forward(hidden_states)?;
        let v_ctx = self.v_proj.forward(target_hidden)?;
        let v_noise = self.v_proj.forward(hidden_states)?;

        let k = Tensor::cat(&[&k_ctx, &k_noise], 0)?;
        let v = Tensor::cat(&[&v_ctx, &v_noise], 0)?;
        let kv_len = ctx_len + q_len;
        let k = k.reshape((1, kv_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = k.transpose(1, 2)?;
        let v = v
            .reshape((1, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        let k = if num_kv_groups > 1 {
            k.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            k
        };
        let v = if num_kv_groups > 1 {
            v.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            v
        };

        let attn_weights = (q.matmul(&k.t()?)? * self.scaling)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct DFlashDecoderLayer {
    self_attn: DFlashAttention,
    mlp: DFlashMLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    attention_conv: Option<GroupedDynamicCausalConv>,
    mlp_conv: Option<GroupedDynamicCausalConv>,
}

impl DFlashDecoderLayer {
    pub fn new(vb: VarBuilderX, config: &DFlashModelConfig, dtype: DType) -> Result<Self> {
        let self_attn = DFlashAttention::new(vb.pp("self_attn"), config, dtype)?;
        let mlp = DFlashMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            dtype,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;

        // DFlash2 grouped dynamic convs (absent in v1 checkpoints).
        let conv_params = config
            .dflash_config
            .as_ref()
            .and_then(|dc| Some((dc.conv_kernel_size?, dc.conv_group_size?)));
        let block_size = config.effective_block_size().unwrap_or(1);
        let (attention_conv, mlp_conv) = match conv_params {
            Some((taps, group_size)) => {
                let a = GroupedDynamicCausalConv::new(
                    &vb.pp("attention_conv"),
                    config.hidden_size,
                    taps,
                    group_size,
                    block_size,
                    dtype,
                )?;
                let m = GroupedDynamicCausalConv::new(
                    &vb.pp("mlp_conv"),
                    config.hidden_size,
                    taps,
                    group_size,
                    block_size,
                    dtype,
                )?;
                (Some(a), Some(m))
            }
            None => (None, None),
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attention_conv,
            mlp_conv,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let (attn_in, attn_fin) = match &self.attention_conv {
            Some(conv) => {
                let (c, d) = conv.prepare(&normed)?;
                (c, Some(d))
            }
            None => (normed.clone(), None),
        };
        let attn_out = self.self_attn.forward(&attn_in, target_hidden, cos, sin)?;
        let attn_out = match (&self.attention_conv, attn_fin) {
            (Some(conv), Some(delta)) => conv.finish(&attn_out, &delta)?,
            _ => attn_out,
        };
        let hidden_states = (attn_out + residual)?;

        let residual = &hidden_states;
        let normed2 = self.post_attention_layernorm.forward(&hidden_states)?;
        let (mlp_in, mlp_fin) = match &self.mlp_conv {
            Some(conv) => {
                let (c, d) = conv.prepare(&normed2)?;
                (c, Some(d))
            }
            None => (normed2.clone(), None),
        };
        let mlp_out = self.mlp.forward(&mlp_in)?;
        let mlp_out = match (&self.mlp_conv, mlp_fin) {
            (Some(conv), Some(delta)) => conv.finish(&mlp_out, &delta)?,
            _ => mlp_out,
        };
        residual + mlp_out
    }
}

pub struct DFlashRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl DFlashRotaryEmbedding {
    pub fn new(
        config: &DFlashModelConfig,
        dtype: DType,
        device: &Device,
        yarn_factor: Option<f64>,
    ) -> Result<Self> {
        let head_dim = config.head_dim();
        let rope_theta = config.effective_rope_theta().unwrap_or(10000.0);
        let max_pos = config.max_position_embeddings;

        // When the backbone uses dynamic YARN scaling, the draft model's RoPE must be scaled by the
        // same factor (applied to the draft model's own rope_theta / max_position_embeddings) so its
        // positional encoding stays consistent with the target model at extended context lengths.
        // Reuse the exact YarnRotaryEmbedding math so the draft head scales identically to the model.
        if let Some(factor) = yarn_factor.filter(|f| *f > 1.0) {
            let (beta_fast, beta_slow, extrapolation_factor, attn_factor) =
                crate::utils::derive_yarn_parameters(factor);
            let yarn = crate::models::layers::rotary_emb::YarnRotaryEmbedding::new_yarn(
                dtype,
                device,
                rope_theta as f32,
                head_dim,
                max_pos,
                max_pos,
                beta_fast as f32,
                beta_slow as f32,
                attn_factor as f32,
                extrapolation_factor as f32,
                factor as f32,
            )?;
            // new_yarn returns half-width (table_len, head_dim/2) tables; DFlash uses the doubled
            // (interleaved) layout (table_len, head_dim).
            return Ok(Self {
                cos: Tensor::cat(&[&yarn.cos, &yarn.cos], D::Minus1)?,
                sin: Tensor::cat(&[&yarn.sin, &yarn.sin], D::Minus1)?,
            });
        }

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos_half = freqs.cos()?.to_dtype(dtype)?;
        let sin_half = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos: Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?,
            sin: Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?,
        })
    }

    pub fn get_cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        Ok((cos.unsqueeze(0)?, sin.unsqueeze(0)?))
    }
}

pub struct DFlashDraftModel {
    fc: ReplicatedLinear,
    hidden_norm: NormX,
    layers: Vec<DFlashDecoderLayer>,
    norm: NormX,
    rotary_emb: DFlashRotaryEmbedding,
    candidate_selector: Option<CandidateSelector>,
    pub config: DFlashModelConfig,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: Option<u32>,
    device: Device,
    dtype: DType,
}

impl DFlashDraftModel {
    /// `vb` is a (separately-loaded) var builder pointing at the DFlash checkpoint root.
    /// The model is fully replicated: no `comm`, no NCCL.
    pub fn new(
        vb: &VarBuilderX,
        config: &DFlashModelConfig,
        dtype: DType,
        device: &Device,
        yarn_factor: Option<f64>,
    ) -> Result<Self> {
        let target_layer_ids = config.target_layer_ids();
        let fc_in_dim = target_layer_ids.len() * config.hidden_size;

        let fc = ReplicatedLinear::load_no_bias(
            fc_in_dim,
            config.hidden_size,
            vb.pp("fc"),
            &None,
            &None,
            dtype,
        )?;

        let hidden_norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("hidden_norm"),
            DType::F32,
            false,
        )?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = DFlashDecoderLayer::new(vb.pp(&format!("layers.{}", i)), config, dtype)?;
            layers.push(layer);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm"),
            DType::F32,
            false,
        )?;

        let rotary_emb = DFlashRotaryEmbedding::new(config, dtype, device, yarn_factor)?;

        let candidate_selector = match (
            config.dflash_config.as_ref().and_then(|dc| dc.selector_rank),
            config.dflash_config.as_ref().and_then(|dc| dc.selector_top_k),
        ) {
            (Some(rank), Some(top_k)) => Some(CandidateSelector::new(
                &vb.pp("candidate_selector"),
                config.hidden_size,
                config.vocab_size,
                rank,
                top_k,
                dtype,
            )?),
            _ => None,
        };

        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm,
            rotary_emb,
            candidate_selector,
            target_layer_ids,
            block_size: config.effective_block_size().unwrap_or(0),
            mask_token_id: config.mask_token_id(),
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Select the per-`target_layer_ids` hidden states from the target model's collector
    /// (`collector[0]` = embedding, `collector[1..]` = selected layer outputs), concatenate on
    /// the last dim, project through `fc`, and norm. Result: one projected context vector per
    /// input token.
    pub fn extract_and_project_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        let selected: Vec<Tensor> = (0..self.target_layer_ids.len())
            .map(|i| all_hidden_states[i + 1].clone())
            .collect();
        let concatenated = Tensor::cat(&selected, D::Minus1)?.to_dtype(self.dtype)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    /// Run the draft transformer over `noise_embedding` (the `[last_token, MASK..MASK]` block)
    /// cross-attending to `target_hidden` (projected context). Returns the final normed hidden
    /// for every position (`ctx_len + noise_len`).
    pub fn forward(
        &self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let positions_flat = positions.flatten_all()?;
        let (cos, sin) = self.rotary_emb.get_cos_sin(&positions_flat)?;

        let mut hidden_states = noise_embedding.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, target_hidden, &cos, &sin)?;
        }

        self.norm.forward(&hidden_states)
    }

    /// The target lm_head logits over the `n` trailing (MASK) positions, plus those hiddens.
    pub fn draft_logits(
        &self,
        draft_hidden: &Tensor,
        n: usize,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let total_out = draft_hidden.dim(0)?;
        let h = draft_hidden.narrow(0, total_out - n, n)?;
        let logits = lm_head_fn(&h)?;
        Ok((logits, h))
    }

    /// Select draft tokens from pre-computed logits (no grammar gate): DFlash2 candidate-selector
    /// walk when available, else plain argmax. On the v2 (CUDA) backend this uses the fused
    /// unmasked kernel; on v1 / non-CUDA it uses the portable walk. Equivalent to
    /// `select_masked(..., allow = None)`.
    pub fn select_from_logits(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        anchor: u32,
    ) -> Result<Vec<u32>> {
        // Original unmasked interface: DFlash2 candidate-selector walk when available
        // (v2 fused kernel on CUDA, portable candle walk otherwise), else plain argmax.
        if let Some(selector) = &self.candidate_selector {
            crate::log_info!("[dflash-debug] model.select_from_logits: selector present -> selector walk (unmasked)");
            if let Ok(tokens) = selector.select_masked(logits, hidden_n, anchor, None) {
                return Ok(tokens);
            }
            crate::log_info!("[dflash-debug] model.select_from_logits: selector walk FAILED -> argmax fallback");
        } else {
            crate::log_info!("[dflash-debug] model.select_from_logits: no selector (v1) -> plain argmax");
        }
        Ok(logits.to_dtype(DType::F32)?.argmax(D::Minus1)?.to_vec1::<u32>()?)
    }

    /// Grammar-gated draft-token selection. `allow` is an optional per-position allow matrix
    /// `[n, vocab]` u8 (1 = legal, 0 = illegal); `None` = unguided. On the v2 (CUDA)
    /// backend the gate is applied inside the fused candidate-walk kernel; otherwise the
    /// logits are pre-masked (disallowed -> -inf) and the portable walk / argmax runs.
    pub fn select_tokens_masked(
        &self,
        logits: &Tensor,
        hidden_n: &Tensor,
        anchor: u32,
        allow: Option<&Tensor>,
    ) -> Result<Vec<u32>> {
        // No active grammar gate -> the original unmasked interface (v1 candle walk /
        // v2 fused kernel with allow=nullptr). The old path is preserved verbatim.
        if allow.is_none() {
            crate::log_info!("[dflash-debug] model.select_tokens_masked: allow=None -> unmasked (select_from_logits)");
            return self.select_from_logits(logits, hidden_n, anchor);
        }
        if let Some(selector) = &self.candidate_selector {
            crate::log_info!("[dflash-debug] model.select_tokens_masked: allow=Some + selector -> selector.select_masked");
            return selector.select_masked(logits, hidden_n, anchor, allow);
        }
        // No selector (v1 checkpoint): argmax, honoring the allow gate. The allow matrix is
        // u8 (1 = legal, 0 = illegal); where_cond requires a u8/u32/i64 condition, so use it
        // directly (no dtype cast) exactly like GuidedDecoding::mask_rows.
        crate::log_info!("[dflash-debug] model.select_tokens_masked: allow=Some, NO selector (v1) -> pre-mask argmax");
        let a = allow.unwrap();
        let neg_inf = Tensor::full(
            f32::NEG_INFINITY,
            logits.shape().clone(),
            logits.device(),
        )?
        .to_dtype(logits.dtype())?;
        let masked = a.where_cond(logits, &neg_inf)?;
        masked.to_dtype(DType::F32)?.argmax(D::Minus1)?.to_vec1::<u32>()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_target: usize, num_draft: usize) -> DFlashModelConfig {
        DFlashModelConfig {
            hidden_size: 64,
            num_hidden_layers: num_draft,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            head_dim: None,
            vocab_size: 1000,
            max_position_embeddings: 2048,
            rope_theta: None,
            attention_bias: None,
            block_size: Some(num_draft + 1),
            num_target_layers: num_target,
            dflash_config: None,
            hidden_act: None,
            layer_types: None,
            rope_parameters: None,
        }
    }

    #[test]
    fn target_layer_ids_single_draft_layer_is_midpoint() {
        assert_eq!(build_target_layer_ids(12, 1), vec![6]);
        assert_eq!(build_target_layer_ids(1, 1), vec![0]);
    }

    #[test]
    fn target_layer_ids_multi_draft_layer_spread() {
        // start=1, end=num_target-3, span=end-start; id_i = 1 + (i*span)/(n-1)
        assert_eq!(build_target_layer_ids(12, 3), vec![1, 5, 9]);
        assert_eq!(build_target_layer_ids(8, 2), vec![1, 4]);
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads() {
        let c = cfg(12, 1);
        assert_eq!(c.head_dim(), 64 / 4);
    }

    #[test]
    fn target_layer_ids_prefers_explicit_override() {
        let mut c = cfg(12, 1);
        c.dflash_config = Some(DFlashConfig {
            mask_token_id: Some(7),
            target_layer_ids: Some(vec![3, 7, 11]),
            block_size: None,
            conv_group_size: None,
            conv_kernel_size: None,
            selector_rank: None,
            selector_top_k: None,
        });
        assert_eq!(c.target_layer_ids(), vec![3, 7, 11]);
        assert_eq!(c.mask_token_id(), Some(7));
    }

    #[test]
    fn target_layer_ids_default_uses_formula() {
        let c = cfg(12, 3);
        assert_eq!(c.target_layer_ids(), vec![1, 5, 9]);
        assert_eq!(c.mask_token_id(), None);
    }

    #[test]
    fn dflash_rope_yarn_scales_table_and_matches_backbone() {
        let c = cfg(12, 1); // head_dim = 64/4 = 16, max_position_embeddings = 2048
        let dev = Device::Cpu;

        let plain = DFlashRotaryEmbedding::new(&c, DType::F32, &dev, None).unwrap();
        let yarn = DFlashRotaryEmbedding::new(&c, DType::F32, &dev, Some(4.0)).unwrap();

        // Plain RoPE table is max_position_embeddings long; YARN extends it by the factor.
        assert_eq!(plain.cos.dim(0).unwrap(), c.max_position_embeddings);
        assert_eq!(yarn.cos.dim(0).unwrap(), c.max_position_embeddings * 4);

        // At position 0, sin is zero for both; cos is 1.0 (plain) vs mscale (yarn).
        let zero = Tensor::from_vec(vec![0i64], (1,), &dev).unwrap();
        let (pc, ps) = plain.get_cos_sin(&zero).unwrap();
        let (yc, ys) = yarn.get_cos_sin(&zero).unwrap();
        let ps: Vec<f32> = ps.flatten_all().unwrap().to_vec1().unwrap();
        let ys: Vec<f32> = ys.flatten_all().unwrap().to_vec1().unwrap();
        assert!(ps.iter().all(|v| v.abs() < 1e-6));
        assert!(ys.iter().all(|v| v.abs() < 1e-6));

        let pc: Vec<f32> = pc.flatten_all().unwrap().to_vec1().unwrap();
        let yc: Vec<f32> = yc.flatten_all().unwrap().to_vec1().unwrap();
        assert!(pc.iter().all(|v| (*v - 1.0).abs() < 1e-5));
// YARN mscale = (0.1 * ln(factor) + 1) * attn_factor, with attn_factor = 1.0.
        let mscale = 0.1f32 * 4.0f32.ln() + 1.0;
        assert!(yc.iter().all(|v| (*v - mscale).abs() < 1e-4));
    }

    #[test]
    fn dflash2_config_schema_parses_nested_block_size_and_rope() {
        // Mirrors the z-lab DFlash2 checkpoint schema: block_size lives under dflash_config and
        // rope_theta under rope_parameters; neither is present at the top level.
        let json = r#"{
            "hidden_size": 5120,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "num_target_layers": 64,
            "dflash_config": { "block_size": 8, "mask_token_id": 248070, "target_layer_ids": [5, 19, 33, 47, 61] },
            "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" }
        }"#;
        let c: DFlashModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.effective_block_size(), Some(8));
        assert_eq!(c.effective_rope_theta(), Some(10000000.0));
        assert_eq!(c.target_layer_ids(), vec![5, 19, 33, 47, 61]);
    }

    #[test]
    fn top_level_block_size_still_resolves() {
        let c = cfg(12, 3); // block_size: Some(4)
        assert_eq!(c.effective_block_size(), Some(4));
    }
}