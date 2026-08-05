//! DeepSeek-V4 RoPE tables + per-layer sparse KV cache.
//!
//! Combined from former `v4_rope.rs` and `v4_sparse_cache.rs`.

use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use candle_core::{DType, Device, Result, Tensor};

#[derive(Clone)]
pub struct V4RopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
    pub max_seq_len: usize,
    pub rotary_dim: usize,
}

impl V4RopeTables {
    pub fn precompute(
        device: &Device,
        max_seq_len: usize,
        rotary_dim: usize,
        base: f64,
        original_seq_len: usize,
        yarn_factor: f64,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self> {
        if rotary_dim == 0 || rotary_dim % 2 != 0 {
            candle_core::bail!("V4 rotary_dim must be positive even, got {rotary_dim}");
        }
        if max_seq_len == 0 {
            candle_core::bail!("V4 rope max_seq_len must be positive");
        }

        let pairs = rotary_dim / 2;
        let mut inv_freq = Vec::with_capacity(pairs);
        let base = base as f32;
        for i in 0..pairs {
            let exponent = (2 * i) as f32 / rotary_dim as f32;
            inv_freq.push(1.0f32 / base.powf(exponent));
        }

        if original_seq_len > 0 {
            let find_correction_dim = |num_rotations: f32| -> f32 {
                rotary_dim as f32
                    * ((original_seq_len as f32) / (num_rotations * 2.0 * std::f32::consts::PI))
                        .ln()
                    / (2.0 * base.ln())
            };
            let low = find_correction_dim(beta_fast).floor().max(0.0);
            let mut high = find_correction_dim(beta_slow)
                .ceil()
                .min((rotary_dim - 1) as f32);
            if (high - low).abs() < f32::EPSILON {
                high += 0.001;
            }
            let factor = yarn_factor as f32;
            for (i, freq) in inv_freq.iter_mut().enumerate() {
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
            }
        }

        // [max_seq, pairs]: cos(t * inv_freq), sin(t * inv_freq)
        let mut cos_host = vec![0f32; max_seq_len * pairs];
        let mut sin_host = vec![0f32; max_seq_len * pairs];
        for t in 0..max_seq_len {
            for p in 0..pairs {
                let angle = (t as f32) * inv_freq[p];
                cos_host[t * pairs + p] = angle.cos();
                sin_host[t * pairs + p] = angle.sin();
            }
        }

        let cos = Tensor::from_vec(cos_host, (max_seq_len, pairs), device)?.to_dtype(DType::F32)?;
        let sin = Tensor::from_vec(sin_host, (max_seq_len, pairs), device)?.to_dtype(DType::F32)?;
        Ok(Self {
            cos,
            sin,
            max_seq_len,
            rotary_dim,
        })
    }

    /// Official Attention freqs selection: compress layers use YaRN + compress_rope_theta;
    /// SWA layers use base rope_theta with YaRN disabled.
    pub fn for_layer_kind(
        device: &Device,
        max_seq_len: usize,
        rotary_dim: usize,
        is_compress_layer: bool,
        rope_theta: f64,
        compress_rope_theta: f64,
        original_seq_len: usize,
        yarn_factor: f64,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self> {
        if is_compress_layer {
            Self::precompute(
                device,
                max_seq_len,
                rotary_dim,
                compress_rope_theta,
                original_seq_len,
                yarn_factor,
                beta_fast,
                beta_slow,
            )
        } else {
            Self::precompute(
                device,
                max_seq_len,
                rotary_dim,
                rope_theta,
                0,
                yarn_factor,
                beta_fast,
                beta_slow,
            )
        }
    }

    pub fn apply_inplace(&self, x: &Tensor, start_pos: usize, inverse: bool) -> Result<()> {
        attention_rs::deepseek_v4::apply_rope_hidden_inplace(
            x,
            &self.cos,
            &self.sin,
            start_pos,
            self.rotary_dim,
            inverse,
        )
    }

    /// CUDA-graph safe RoPE using GPU `positions` (optional per-token offset).
    pub fn apply_from_positions(
        &self,
        x: &Tensor,
        positions: &Tensor,
        position_offset: i64,
        inverse: bool,
    ) -> Result<()> {
        attention_rs::deepseek_v4::apply_rope_hidden_from_positions(
            x,
            &self.cos,
            &self.sin,
            positions,
            self.rotary_dim,
            position_offset,
            inverse,
        )
    }

    pub fn apply_strided_inplace(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<()> {
        attention_rs::deepseek_v4::apply_rope_hidden_strided_inplace(
            x,
            &self.cos,
            &self.sin,
            start_pos,
            position_stride,
            self.rotary_dim,
            inverse,
        )
    }
}

impl ApplyRotaryEmbedding for V4RopeTables {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let cos = self.cos.to_dtype(q.dtype())?;
        let sin = self.sin.to_dtype(q.dtype())?;
        attention_rs::fused_rope::FusedRope::apply_inplace(&q, &k, &cos, &sin, positions, false)?;
        Ok(Some((q, k)))
    }

    fn get_original_max_position_embeddings(&self) -> Option<usize> {
        None
    }

    fn get_llama_4_scaling_beta(&self) -> Option<f64> {
        None
    }
}

// ============================================================================
// Sparse KV cache
// ============================================================================

pub struct LayerSparseKvCache {
    /// `[slots, head_dim]` BF16 unified buffer.
    pub kv: Tensor,
    pub sliding_window: usize,
    pub compress_ratio: usize,
    pub compressed_slots: usize,
    pub head_dim: usize,
    /// How many compressed rows have been written (decode progress).
    pub compressed_len: usize,
}

impl LayerSparseKvCache {
    pub fn new(
        sliding_window: usize,
        compress_ratio: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let compressed_slots = if compress_ratio > 0 {
            max_seq_len / compress_ratio
        } else {
            0
        };
        let slots = sliding_window + compressed_slots;
        let kv = Tensor::zeros((slots, head_dim), DType::BF16, device)?;
        Ok(Self {
            kv,
            sliding_window,
            compress_ratio,
            compressed_slots,
            head_dim,
            compressed_len: 0,
        })
    }

    pub fn total_slots(&self) -> usize {
        self.sliding_window + self.compressed_slots
    }

    /// Clear decode progress without reallocating (CUDA graph pointer stability).
    pub fn reset(&mut self) -> Result<()> {
        self.kv.zero_()?;
        self.compressed_len = 0;
        Ok(())
    }

    /// Seed window ring from prefill token KV (official split copy when seq > window).
    pub fn seed_window_from_prefill(&mut self, token_kv: &Tensor) -> Result<()> {
        let seq_len = token_kv.dim(0)?;
        let win = self.sliding_window;
        if seq_len <= win {
            let token_kv = token_kv.contiguous()?;
            attention_rs::deepseek_v4::copy_contiguous_into(&self.kv, &token_kv, 0)?;
        } else {
            let cutoff = seq_len % win;
            let last = token_kv.narrow(0, seq_len - win, win)?;
            if cutoff == 0 {
                let last = last.contiguous()?;
                attention_rs::deepseek_v4::copy_contiguous_into(&self.kv, &last, 0)?;
            } else {
                let (part_a, part_b) = (
                    last.narrow(0, 0, win - cutoff)?.contiguous()?,
                    last.narrow(0, win - cutoff, cutoff)?.contiguous()?,
                );
                attention_rs::deepseek_v4::copy_contiguous_into(
                    &self.kv,
                    &part_a,
                    cutoff * self.head_dim,
                )?;
                attention_rs::deepseek_v4::copy_contiguous_into(&self.kv, &part_b, 0)?;
            }
        }
        Ok(())
    }

    /// Write compressed prefill block into `kv[window..]`.
    pub fn seed_compressed_from_prefill(&mut self, compressed: &Tensor) -> Result<()> {
        let n = compressed.dim(0)?;
        if n > self.compressed_slots {
            candle_core::bail!(
                "compressed prefill len {n} exceeds capacity {}",
                self.compressed_slots
            );
        }
        let win = self.sliding_window;
        let compressed = compressed.contiguous()?;
        attention_rs::deepseek_v4::copy_contiguous_into(
            &self.kv,
            &compressed,
            win * self.head_dim,
        )?;
        self.compressed_len = n;
        Ok(())
    }

    pub fn write_window_token(&mut self, token_kv: &Tensor, start_pos: usize) -> Result<()> {
        let slot = start_pos % self.sliding_window;
        let row = token_kv.reshape((1, self.head_dim))?.contiguous()?;
        attention_rs::deepseek_v4::copy_contiguous_into(&self.kv, &row, slot * self.head_dim)?;
        Ok(())
    }

    pub fn write_window_token_from_pos(
        &mut self,
        token_kv: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        let row = token_kv.reshape((1, self.head_dim))?.contiguous()?;
        attention_rs::deepseek_v4::write_kv_row_from_pos(
            &self.kv,
            &row,
            positions,
            self.sliding_window,
            self.head_dim,
        )
    }

    pub fn write_compressed_row(&mut self, compressed: &Tensor, row: usize) -> Result<()> {
        if row >= self.compressed_slots {
            candle_core::bail!(
                "compressed row {row} out of range {}",
                self.compressed_slots
            );
        }
        let win = self.sliding_window;
        let row_t = compressed.reshape((1, self.head_dim))?.contiguous()?;
        attention_rs::deepseek_v4::copy_contiguous_into(
            &self.kv,
            &row_t,
            (win + row) * self.head_dim,
        )?;
        self.compressed_len = self.compressed_len.max(row + 1);
        Ok(())
    }

    pub fn write_compressed_row_from_pos(
        &mut self,
        compressed: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        let ratio = self.compress_ratio.max(1);
        let row_t = compressed.reshape((1, self.head_dim))?.contiguous()?;
        attention_rs::deepseek_v4::write_compressed_row_from_pos(
            &self.kv,
            &row_t,
            positions,
            self.sliding_window,
            self.head_dim,
            ratio,
        )?;
        Ok(())
    }
}
