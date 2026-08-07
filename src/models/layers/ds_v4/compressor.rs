use super::rope_cache::V4RopeTables;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor};

/// Compressor weights for a single layer.
///
/// NonOverlap (ratio=128): wkv/wgate are [head_dim, hidden_dim], ape is [ratio, head_dim]
/// Overlap (ratio=4): wkv/wgate are [2*head_dim, hidden_dim], ape is [ratio, 2*head_dim]
pub struct CompressorWeights {
    pub wkv: Tensor,
    pub wgate: Tensor,
    pub ape: Tensor,
    pub norm: Tensor,
    pub ratio: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
}

impl CompressorWeights {
    pub fn load(
        vb: &VarBuilderX,
        prefix: &str,
        ratio: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Result<Option<Self>> {
        if ratio == 0 {
            return Ok(None);
        }

        let coff = if ratio == 4 { 2 } else { 1 };
        let out_dim = coff * head_dim;

        let wkv = vb.get((out_dim, hidden_dim), &format!("{prefix}.wkv.weight"))?;
        let wgate = vb.get((out_dim, hidden_dim), &format!("{prefix}.wgate.weight"))?;
        let ape = vb.get_with_hints_dtype(
            (ratio, out_dim),
            &format!("{prefix}.ape"),
            Default::default(),
            DType::F32,
        )?;
        let norm = vb.get((head_dim,), &format!("{prefix}.norm.weight"))?;

        Ok(Some(Self {
            wkv,
            wgate,
            ape,
            norm,
            ratio,
            head_dim,
            hidden_dim,
        }))
    }

    pub fn is_overlap(&self) -> bool {
        self.ratio == 4
    }

    /// Prefill: compress a full sequence into compressed KV tokens.
    /// Input: [seq_len, hidden_dim] BF16 (post-attn-norm hidden states)
    /// Returns: [compressed_len, head_dim] BF16
    ///
    /// After the CUDA epilogue (softmax pool + RMSNorm), applies **strided RoPE** on the
    /// pe dims with positions `0, ratio, 2*ratio, ...` (official Compressor.forward).
    /// When `rotate_fp4` is set (indexer compressor), also Hadamard + FP4 act quant.
    pub fn prefill(
        &self,
        x: &Tensor,
        seq_len: usize,
        rope: Option<&V4RopeTables>,
        rope_start: usize,
        rotate_fp4: bool,
    ) -> Result<Tensor> {
        let eps = 1e-6f32;
        let mut out = if self.is_overlap() {
            let (_weighted, out) = attention_rs::deepseek_v4::compressor_overlap_prefill(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                seq_len,
                self.hidden_dim,
                self.head_dim,
                eps,
            )?;
            out
        } else {
            let (_weighted, out) = attention_rs::deepseek_v4::compressor_nonoverlap_prefill(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                seq_len,
                self.hidden_dim,
                self.head_dim,
                self.ratio,
                eps,
            )?;
            out
        };

        out = out.contiguous()?;
        if let Some(rope) = rope {
            // Official: freqs_cis[:cutoff:ratio] — start at rope_start (usually 0), stride=ratio
            rope.apply_strided_inplace(&out, rope_start, self.ratio, false)?;
        }
        if rotate_fp4 {
            attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(&out, 1, self.head_dim)?;
        } else if let Some(rope) = rope {
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                &out,
                1,
                self.head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(out)
    }

    /// Continued-prefill: compress a chunk that appends at absolute `start`.
    ///
    /// Returns `(phase_a_rows, bulk_rows, bulk_start_row)`:
    /// - `phase_a_rows`: at most one emitted overlap head-boundary row
    /// - `bulk_rows`: remaining compressed rows from the cont epilogue
    /// - `bulk_start_row`: first compressed-row index of the bulk block
    pub fn prefill_continue(
        &self,
        x: &Tensor,
        state: &mut CompressorDecodeState,
        _positions: &Tensor,
        _weighted: &Tensor,
        _out: &Tensor,
        rope: Option<&V4RopeTables>,
        rotate_fp4: bool,
        start: usize,
    ) -> Result<(Vec<Tensor>, Option<Tensor>, usize)> {
        let seq_len = x.dim(0)?;
        if start == 0 {
            // Match the committed single-shot guard: nonoverlap ratio=128 (and
            // any ratio) must not call the epilogue when seq_len < ratio
            // (cudaErrorInvalidValue). Short prompts still seed decode state.
            if seq_len < self.ratio {
                self.seed_decode_state_after_prefill(x, state, seq_len)?;
                return Ok((Vec::new(), None, 0));
            }
            let bulk = self.prefill(x, seq_len, rope, 0, rotate_fp4)?;
            self.seed_decode_state_after_prefill(x, state, seq_len)?;
            let n = bulk.dim(0)?;
            return Ok((Vec::new(), if n > 0 { Some(bulk) } else { None }, 0));
        }

        let ratio = self.ratio;
        let eps = 1e-6f32;
        let mut phase_a_rows: Vec<Tensor> = Vec::new();
        let r = start % ratio;

        // Phase A (overlap only): finish the incomplete block straddling the
        // chunk boundary via the decode state machine.
        if self.is_overlap() && r > 0 {
            let n = (ratio - r).min(seq_len);
            for t in 0..n {
                let x_row = x.narrow(0, t, 1)?.contiguous()?;
                let abs_pos = start + t;
                let emitted = self.decode(&x_row, state, abs_pos, rope, rotate_fp4)?;
                if t == n - 1 {
                    if let Some(row) = emitted {
                        phase_a_rows.push(row);
                    }
                }
            }
        }

        let first_bulk_row = if self.is_overlap() {
            start / ratio + if r > 0 { 1 } else { 0 }
        } else {
            start / ratio
        };
        let end_abs = start + seq_len;
        let bulk_rows = end_abs / ratio - first_bulk_row;

        let bulk_out = if bulk_rows > 0 {
            let (_weighted, mut bulk) = if self.is_overlap() {
                attention_rs::deepseek_v4::compressor_overlap_prefill_cont(
                    x,
                    &self.wkv,
                    &self.wgate,
                    &self.ape,
                    &self.norm,
                    &state.kv_state,
                    &state.score_state,
                    seq_len,
                    start,
                    self.head_dim,
                    bulk_rows,
                    eps,
                )?
            } else {
                attention_rs::deepseek_v4::compressor_nonoverlap_prefill_cont(
                    x,
                    &self.wkv,
                    &self.wgate,
                    &self.ape,
                    &self.norm,
                    &state.kv_state,
                    &state.score_state,
                    seq_len,
                    start,
                    self.head_dim,
                    ratio,
                    bulk_rows,
                    eps,
                )?
            };
            bulk = bulk.contiguous()?;
            if let Some(rope) = rope {
                rope.apply_strided_inplace(&bulk, first_bulk_row * ratio, ratio, false)?;
            }
            if rotate_fp4 {
                attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(
                    &bulk,
                    1,
                    self.head_dim,
                )?;
            } else if let Some(rope) = rope {
                attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                    &bulk,
                    1,
                    self.head_dim,
                    rope.rotary_dim,
                    64,
                )?;
            }
            Some(bulk)
        } else {
            None
        };

        // Seed decode state for absolute end position. When `start % ratio == 0`,
        // chunk-relative remainder matches absolute; otherwise take the trailing
        // absolute-incomplete block from the end of this chunk.
        self.seed_decode_state_after_continue(x, state, start, seq_len)?;
        Ok((phase_a_rows, bulk_out, first_bulk_row))
    }

    /// Seed compressor decode state after a continued-prefill chunk ending at
    /// absolute position `start + seq_len`.
    fn seed_decode_state_after_continue(
        &self,
        x: &Tensor,
        state: &mut CompressorDecodeState,
        start: usize,
        seq_len: usize,
    ) -> Result<()> {
        let end_abs = start + seq_len;
        if end_abs == 0 || seq_len == 0 {
            return state.reset();
        }
        // When the chunk starts on a ratio boundary, absolute and chunk-relative
        // remainders coincide — reuse the standard seed.
        if start % self.ratio == 0 {
            return self.seed_decode_state_after_prefill(x, state, seq_len);
        }

        let ratio = self.ratio;
        let abs_remainder = end_abs % ratio;
        let out_dim = if self.is_overlap() {
            2 * self.head_dim
        } else {
            self.head_dim
        };
        let values = attention_rs::deepseek_v4::compressor_bf16_linear_f32(
            x,
            &self.wkv,
            seq_len,
            self.hidden_dim,
            out_dim,
        )?;
        let scores = attention_rs::deepseek_v4::compressor_bf16_linear_f32(
            x,
            &self.wgate,
            seq_len,
            self.hidden_dim,
            out_dim,
        )?;
        let ape = self.ape.contiguous()?;

        // Overlap old-half may live before `start`. Preserve it from the live
        // decode state (Phase A / prior chunk) before resetting.
        let mut saved_old_kv: Option<Tensor> = None;
        let mut saved_old_score: Option<Tensor> = None;
        if self.is_overlap() {
            let old_abs_start = if abs_remainder == 0 {
                end_abs.saturating_sub(ratio)
            } else {
                end_abs.saturating_sub(abs_remainder + ratio)
            };
            if old_abs_start < start && end_abs > old_abs_start {
                // First half of the current state holds the previous complete block
                // after Phase A emit+shift (or a prior boundary seed).
                saved_old_kv = Some(state.kv_state.narrow(0, 0, ratio)?.contiguous()?.copy()?);
                saved_old_score = Some(
                    state
                        .score_state
                        .narrow(0, 0, ratio)?
                        .contiguous()?
                        .copy()?,
                );
            }
        }

        state.reset()?;

        let offset = if self.is_overlap() { ratio } else { 0 };

        if self.is_overlap() {
            let old_abs_start = if abs_remainder == 0 {
                end_abs.saturating_sub(ratio)
            } else {
                end_abs.saturating_sub(abs_remainder + ratio)
            };
            if old_abs_start >= start && end_abs > old_abs_start {
                let local = old_abs_start - start;
                if local + ratio <= seq_len {
                    let kv_block = values.narrow(0, local, ratio)?.contiguous()?;
                    let score_block = (scores.narrow(0, local, ratio)? + &ape)?.contiguous()?;
                    attention_rs::deepseek_v4::copy_contiguous_into(&state.kv_state, &kv_block, 0)?;
                    attention_rs::deepseek_v4::copy_contiguous_into(
                        &state.score_state,
                        &score_block,
                        0,
                    )?;
                }
            } else if let (Some(kv), Some(score)) = (saved_old_kv, saved_old_score) {
                attention_rs::deepseek_v4::copy_contiguous_into(&state.kv_state, &kv, 0)?;
                attention_rs::deepseek_v4::copy_contiguous_into(&state.score_state, &score, 0)?;
            }
        }

        if abs_remainder > 0 {
            let rem_abs_start = end_abs - abs_remainder;
            if rem_abs_start >= start {
                let local = rem_abs_start - start;
                let kv_rem = values.narrow(0, local, abs_remainder)?.contiguous()?;
                let ape_rem = ape.narrow(0, 0, abs_remainder)?.contiguous()?;
                let score_rem =
                    (scores.narrow(0, local, abs_remainder)? + ape_rem)?.contiguous()?;
                let elem_off = offset * out_dim;
                attention_rs::deepseek_v4::copy_contiguous_into(
                    &state.kv_state,
                    &kv_rem,
                    elem_off,
                )?;
                attention_rs::deepseek_v4::copy_contiguous_into(
                    &state.score_state,
                    &score_rem,
                    elem_off,
                )?;
            }
        }
        Ok(())
    }

    /// Decode: process a single token, accumulate into state.
    /// Returns Some(compressed_kv) if a compressed token was emitted, None otherwise.
    pub fn decode(
        &self,
        x: &Tensor,
        state: &CompressorDecodeState,
        start_pos: usize,
        rope: Option<&V4RopeTables>,
        rotate_fp4: bool,
    ) -> Result<Option<Tensor>> {
        let eps = 1e-6f32;
        let should_compress = (start_pos + 1) % self.ratio == 0;

        let emitted = if self.is_overlap() {
            attention_rs::deepseek_v4::compressor_overlap_decode_at(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                &state.kv_state,
                &state.score_state,
                start_pos,
                self.hidden_dim,
                self.head_dim,
                0,
                eps,
            )?
        } else {
            attention_rs::deepseek_v4::compressor_nonoverlap_decode_at(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                &state.kv_state,
                &state.score_state,
                start_pos,
                self.hidden_dim,
                self.head_dim,
                self.ratio,
                0,
                eps,
            )?
        };

        if !should_compress {
            debug_assert!(emitted.is_none());
            return Ok(None);
        }
        let (_weighted, mut out) = emitted.expect("compressor kernel emits at ratio boundary");
        out = out.contiguous()?;
        if let Some(rope) = rope {
            // Official: freqs at `start_pos + 1 - ratio`
            let rope_pos = start_pos + 1 - self.ratio;
            rope.apply_inplace(&out, rope_pos, false)?;
        }
        if rotate_fp4 {
            attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(&out, 1, self.head_dim)?;
        } else if let Some(rope) = rope {
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                &out,
                1,
                self.head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(Some(out))
    }

    /// CUDA-graph decode path: fixed kernel topology, position from GPU buffer.
    pub fn decode_graph(
        &self,
        x: &Tensor,
        state: &CompressorDecodeState,
        positions: &Tensor,
        weighted: &Tensor,
        out: &Tensor,
        rope: Option<&V4RopeTables>,
        rotate_fp4: bool,
    ) -> Result<Tensor> {
        let eps = 1e-6f32;
        if self.is_overlap() {
            attention_rs::deepseek_v4::compressor_overlap_decode_at_graph(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                &state.kv_state,
                &state.score_state,
                positions,
                weighted,
                out,
                self.hidden_dim,
                self.head_dim,
                0,
                eps,
            )?;
        } else {
            attention_rs::deepseek_v4::compressor_nonoverlap_decode_at_graph(
                x,
                &self.wkv,
                &self.wgate,
                &self.ape,
                &self.norm,
                &state.kv_state,
                &state.score_state,
                positions,
                weighted,
                out,
                self.hidden_dim,
                self.head_dim,
                self.ratio,
                0,
                eps,
            )?;
        }
        let out = out.reshape((1, self.head_dim))?.contiguous()?;
        if let Some(rope) = rope {
            let rope_offset = 1_i64 - self.ratio as i64;
            rope.apply_from_positions(&out, positions, rope_offset, false)?;
        }
        if rotate_fp4 {
            attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(&out, 1, self.head_dim)?;
        } else if let Some(rope) = rope {
            attention_rs::deepseek_v4::fp8_act_quant_nope_bf16_inplace(
                &out,
                1,
                self.head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(out)
    }

    /// Seed decode accumulators to match official Compressor.forward prefill
    /// (`start_pos == 0`) state copy — NOT by replaying decode steps.
    ///
    /// Official layout after prefill:
    /// - Overlap: `kv_state[:ratio] = last full block`, then remainder into
    ///   `kv_state[ratio:ratio+remainder]` (second half stays 0 / -inf).
    /// - NonOverlap: remainder into `kv_state[:remainder]`.
    ///
    /// Decode-replay seeding leaves dirty second-half slots (shift does not
    /// clear them) and diverges when `seq_len % ratio == 0`.
    pub fn seed_decode_state_after_prefill(
        &self,
        x: &Tensor,
        state: &mut CompressorDecodeState,
        seq_len: usize,
    ) -> Result<()> {
        state.reset()?;

        if seq_len == 0 {
            return Ok(());
        }

        let out_dim = if self.is_overlap() {
            2 * self.head_dim
        } else {
            self.head_dim
        };
        let values = attention_rs::deepseek_v4::compressor_bf16_linear_f32(
            x,
            &self.wkv,
            seq_len,
            self.hidden_dim,
            out_dim,
        )?;
        let scores = attention_rs::deepseek_v4::compressor_bf16_linear_f32(
            x,
            &self.wgate,
            seq_len,
            self.hidden_dim,
            out_dim,
        )?;

        let ratio = self.ratio;
        let remainder = seq_len % ratio;
        let cutoff = seq_len - remainder;
        let offset = if self.is_overlap() { ratio } else { 0 };
        let ape = self.ape.contiguous()?;

        if self.is_overlap() && cutoff >= ratio {
            let kv_block = values.narrow(0, cutoff - ratio, ratio)?.contiguous()?;
            let score_block = (scores.narrow(0, cutoff - ratio, ratio)? + &ape)?.contiguous()?;
            attention_rs::deepseek_v4::copy_contiguous_into(&state.kv_state, &kv_block, 0)?;
            attention_rs::deepseek_v4::copy_contiguous_into(&state.score_state, &score_block, 0)?;
        }
        if remainder > 0 {
            let kv_rem = values.narrow(0, cutoff, remainder)?.contiguous()?;
            let ape_rem = ape.narrow(0, 0, remainder)?.contiguous()?;
            let score_rem = (scores.narrow(0, cutoff, remainder)? + ape_rem)?.contiguous()?;
            let elem_off = offset * out_dim;
            attention_rs::deepseek_v4::copy_contiguous_into(&state.kv_state, &kv_rem, elem_off)?;
            attention_rs::deepseek_v4::copy_contiguous_into(
                &state.score_state,
                &score_rem,
                elem_off,
            )?;
        }
        Ok(())
    }
}

/// Per-request compressor decode state (F32 accumulators on GPU).
///
/// NonOverlap: kv_state=[ratio, head_dim], score_state=[ratio, head_dim]
/// Overlap: kv_state=[2*ratio, 2*head_dim], score_state=[2*ratio, 2*head_dim]
pub struct CompressorDecodeState {
    pub kv_state: Tensor,
    pub score_state: Tensor,
    pub state_dim: usize,
    pub slots: usize,
}

impl CompressorDecodeState {
    pub fn new(ratio: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let (slots, state_dim) = if ratio == 4 {
            (2 * ratio, 2 * head_dim)
        } else {
            (ratio, head_dim)
        };

        let kv_state = Tensor::zeros((slots, state_dim), DType::F32, device)?.contiguous()?;
        let score_state =
            Tensor::full(f32::NEG_INFINITY, (slots, state_dim), device)?.contiguous()?;

        Ok(Self {
            kv_state,
            score_state,
            state_dim,
            slots,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.kv_state.zero_()?;
        let neg_inf = Tensor::full(
            f32::NEG_INFINITY,
            (self.slots, self.state_dim),
            self.score_state.device(),
        )?;
        self.score_state.copy_(&neg_inf, 0)?;
        Ok(())
    }
}

/// Per-layer compressor configuration derived from compress_ratios.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerCompressionType {
    Swa,
    Overlap,
    NonOverlap(usize),
}

impl LayerCompressionType {
    pub fn from_ratio(ratio: usize) -> Self {
        match ratio {
            0 => Self::Swa,
            4 => Self::Overlap,
            r => Self::NonOverlap(r),
        }
    }

    pub fn ratio(&self) -> usize {
        match self {
            Self::Swa => 0,
            Self::Overlap => 4,
            Self::NonOverlap(r) => *r,
        }
    }

    pub fn has_compressor(&self) -> bool {
        !matches!(self, Self::Swa)
    }

    pub fn has_indexer(&self) -> bool {
        matches!(self, Self::Overlap)
    }
}
