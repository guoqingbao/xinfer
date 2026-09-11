//! Qwen4-Exp PLE ("Engram" N-gram Embedding) layer.
//!
//! Port of the vLLM reference (`vllm/models/qwen4_exp/nvidia/ple_layer.py`,
//! `ops/ple.py`). A hashed n-gram lookup table (host-resident, memory-mapped
//! from the checkpoint shards) is injected into the hyper-connection stream
//! at the decoder layers listed in `ple_layer_ids` (1-based):
//!
//! ```text
//!   id[h]    = offset[h] + euclid_rem(xor_i(token[t-i] * mult[i]), size[h])
//!   emb      = dequant(table[ids]).flatten()                 # [T, ple_dim]
//!   key      = emb @ key_proj.T                              # [T, hc*hidden]
//!   value    = emb @ value_proj.T                            # [T, hidden]
//!   d        = dot(rmsnorm_g(key), rmsnorm_g(xs)) / sqrt(hidden)  (per group)
//!   g        = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
//!   gated    = g * value                       (value shared across groups)
//!   conv_in  = rmsnorm_g(gated) * (1 + norm_conv_w)
//!   delta    = gated + silu(depthwise_causal_conv(conv_in, k, dilation=ngram))
//!   xs      += delta
//! ```
//!
//! The table is a pure row gather (never matmul), so it stays on host memory
//! (mmap from the safetensors shards, served by the page cache) and only the
//! small projections/gate/conv run on GPU. Because the gather happens on the
//! CPU inside the forward pass, models with PLE are incompatible with CUDA
//! graph capture (call sites disable graphs when PLE is configured).

use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use memmap2::Mmap;
use rayon::prelude::*;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::models::layers::distributed::shard;
use crate::models::layers::VarBuilderX;
use crate::utils::Qwen4PleConfig;

// =============================================================================
// Hashing (port of vLLM Qwen4ExpNGramEmbedding)
// =============================================================================

const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;
const PLE_LAYER_PRIME: u64 = 10007;

fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn powmod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

fn is_prime_64(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if v % p == 0 {
            return v == p;
        }
    }
    let mut d = v - 1;
    let mut s = 0u64;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'bases: for a in [2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % v == 0 {
            continue;
        }
        let mut x = powmod(a, d, v);
        if x == 1 || x == v - 1 {
            continue;
        }
        for _ in 0..s.saturating_sub(1) {
            x = mulmod(x, x, v);
            if x == v - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count {
        let mut c = p + 1;
        if c <= 2 {
            p = 2;
            continue;
        }
        if c % 2 == 0 {
            c += 1;
        }
        while !is_prime_64(c) {
            c += 2;
        }
        p = c;
    }
    p
}

/// Deterministic per-layer hash multipliers (odd int64 values).
fn make_layer_multipliers(
    ngram_size: usize,
    unigram_vocab_size: u64,
    seed: u64,
    ple_dense_layer_id: usize,
) -> Vec<i64> {
    let max_multiplier = (i64::MAX as u64) / unigram_vocab_size.max(1);
    let half_bound = (max_multiplier / 2).max(1);
    let base_seed = seed.wrapping_add(PLE_LAYER_PRIME.wrapping_mul(ple_dense_layer_id as u64));
    (0..ngram_size)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

/// Per-head prime vocabulary sizes and cumulative global offsets.
fn make_vocab_layout(
    ngram_vocab_size_base: u64,
    ngram_heads: usize,
    ple_dense_layer_id: usize,
) -> (Vec<i64>, Vec<i64>) {
    let mut sizes = Vec::with_capacity(ngram_heads);
    let mut offsets = Vec::with_capacity(ngram_heads);
    let mut offset = 0u64;
    for local_head in 0..ngram_heads {
        let global_head = ple_dense_layer_id * ngram_heads + local_head;
        let size = nth_prime_after(ngram_vocab_size_base.saturating_sub(1), global_head + 1);
        sizes.push(size as i64);
        offsets.push(offset as i64);
        offset += size;
    }
    (sizes, offsets)
}

// =============================================================================
// Host-resident mmap n-gram table
// =============================================================================

#[derive(Clone, Copy, PartialEq, Debug)]
enum PleTableDtype {
    F8E4M3,
    BF16,
    F16,
    F32,
}

impl PleTableDtype {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "F8_E4M3" => Some(Self::F8E4M3),
            "BF16" => Some(Self::BF16),
            "F16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }

    fn elem_size(self) -> usize {
        match self {
            Self::F8E4M3 => 1,
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

fn build_fp8_e4m3_lut() -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        let sign = if i & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((i >> 3) & 0xF) as i32;
        let mant = (i & 0x7) as f32;
        *e = if exp == 0 {
            sign * (mant / 8.0) * 2f32.powi(-6)
        } else if exp == 15 && (i & 0x7) == 7 {
            f32::NAN // E4M3FN: no inf, S.1111.111 = NaN
        } else {
            sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
        };
    }
    lut
}

struct PleShard {
    data: PleData,
    data_offset: usize, // absolute byte offset of the tensor data in the file
    rows: usize,
}

/// The PLE shard backing store: an mmap (the page-cache served, the default) or a
/// direct heap read (the SM121 unified, where an mmap would compete with the
/// model + KV for the unified host/device memory pool).
#[derive(Clone)]
enum PleData {
    Mmap(Arc<Mmap>),
    Direct(Vec<u8>),
}

impl PleData {
    fn get(&self, range: std::ops::Range<usize>) -> &[u8] {
        match self {
            PleData::Mmap(m) => &m[range],
            PleData::Direct(d) => &d[range],
        }
    }
}

struct PleTable {
    shards: Vec<Option<PleShard>>,
    shard_rows: usize,
    head_dim: usize,
    dtype: PleTableDtype,
    scale: f32,
    fp8_lut: [f32; 256],
}

impl PleTable {
    /// Scan the model's safetensors files for `ngram_embedding.shard_*` tensors
    /// (and the optional global `weight_scale`) and memory-map them.
    fn scan(
        weight_files: &[PathBuf],
        tensor_prefix: &str,
        split_ngram_parts: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let shard_key_prefix = format!("{tensor_prefix}.shard_");
        let scale_key = format!("{tensor_prefix}.weight_scale");
        let mut shards: Vec<Option<PleShard>> = (0..split_ngram_parts).map(|_| None).collect();
        let mut scale: Option<f32> = None;
        let mut table_dtype: Option<PleTableDtype> = None;
        let mut shard_rows = 0usize;

        for path in weight_files {
            let mut file = std::fs::File::open(path)
                .map_err(|e| candle_core::Error::wrap(format!("open {path:?}: {e}")))?;
            let mut len_buf = [0u8; 8];
            file.read_exact(&mut len_buf)
                .map_err(candle_core::Error::wrap)?;
            let header_len = u64::from_le_bytes(len_buf) as usize;
            let mut header = vec![0u8; header_len];
            file.read_exact(&mut header)
                .map_err(candle_core::Error::wrap)?;
            let json: serde_json::Value =
                serde_json::from_slice(&header).map_err(candle_core::Error::wrap)?;
            let Some(obj) = json.as_object() else {
                continue;
            };
            if !obj
                .keys()
                .any(|k| k.starts_with(&shard_key_prefix) || k == &scale_key)
            {
                continue;
            }
            let data = if crate::utils::env::ple_no_mmap() {
                // the SM121 bypass: the direct heap read (the no mmap, the no page-cache
                // competition with the model + KV on the unified memory pool).
                PleData::Direct(std::fs::read(path).map_err(candle_core::Error::wrap)?)
            } else {
                // the default: the mmap (the page-cache served).
                PleData::Mmap(Arc::new(unsafe { Mmap::map(&file) }.map_err(candle_core::Error::wrap)?))
            };
            let data_base = 8 + header_len;
            for (key, meta) in obj {
                let offsets = meta["data_offsets"].as_array();
                let dtype_str = meta["dtype"].as_str().unwrap_or("");
                if key == &scale_key {
                    if let Some(offs) = offsets {
                        let begin = data_base + offs[0].as_u64().unwrap_or(0) as usize;
                        scale = Some(match dtype_str {
                            "F32" => f32::from_le_bytes(
                                data.get(begin..begin + 4)
                                    .try_into()
                                    .map_err(candle_core::Error::wrap)?,
                            ),
                            "BF16" => half::bf16::from_le_bytes(
                                data.get(begin..begin + 2)
                                    .try_into()
                                    .map_err(candle_core::Error::wrap)?,
                            )
                            .to_f32(),
                            other => {
                                candle_core::bail!("Unsupported PLE weight_scale dtype {other}")
                            }
                        });
                    }
                    continue;
                }
                let Some(rest) = key.strip_prefix(&shard_key_prefix) else {
                    continue;
                };
                let Some(idx_str) = rest.strip_suffix(".weight") else {
                    continue;
                };
                let Ok(shard_idx) = idx_str.parse::<usize>() else {
                    continue;
                };
                if shard_idx >= split_ngram_parts {
                    candle_core::bail!(
                        "PLE embedding shard index {shard_idx} exceeds split_ngram_parts={split_ngram_parts}"
                    );
                }
                let shape = meta["shape"]
                    .as_array()
                    .map(|s| {
                        s.iter()
                            .map(|v| v.as_u64().unwrap_or(0) as usize)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if shape.len() != 2 || shape[1] != head_dim {
                    candle_core::bail!(
                        "PLE shard {shard_idx} has unexpected shape {shape:?}, expected [_, {head_dim}]"
                    );
                }
                let dt = PleTableDtype::from_str(dtype_str).ok_or_else(|| {
                    candle_core::Error::wrap(format!("Unsupported PLE table dtype {dtype_str}"))
                })?;
                if let Some(existing) = table_dtype {
                    if existing != dt {
                        candle_core::bail!("PLE shards have mixed dtypes");
                    }
                } else {
                    table_dtype = Some(dt);
                }
                if shard_rows == 0 {
                    shard_rows = shape[0];
                } else if shape[0] != shard_rows && shard_idx != split_ngram_parts - 1 {
                    candle_core::bail!("PLE shards have inconsistent row counts");
                }
                let begin = data_base
                    + offsets
                        .and_then(|o| o.first())
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                shards[shard_idx] = Some(PleShard {
                    data: data.clone(),
                    data_offset: begin,
                    rows: shape[0],
                });
            }
        }

        let dtype = table_dtype.ok_or_else(|| {
            candle_core::Error::wrap(format!(
                "No PLE ngram_embedding shards found (prefix {tensor_prefix})"
            ))
        })?;
        let found = shards.iter().filter(|s| s.is_some()).count();
        if found != split_ngram_parts {
            candle_core::bail!(
                "PLE table incomplete: found {found}/{split_ngram_parts} shards (checkpoint download may be incomplete)"
            );
        }
        if dtype == PleTableDtype::F8E4M3 && scale.is_none() {
            candle_core::bail!("FP8 PLE table is missing its global ngram_embedding.weight_scale");
        }
        Ok(Self {
            shards,
            shard_rows,
            head_dim,
            dtype,
            scale: scale.unwrap_or(1.0),
            fp8_lut: build_fp8_e4m3_lut(),
        })
    }

    #[inline]
    fn read_row(&self, row_id: usize, dst: &mut [f32]) -> Result<()> {
        let shard_idx = row_id / self.shard_rows;
        let row = row_id % self.shard_rows;
        let shard = self
            .shards
            .get(shard_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                candle_core::Error::wrap(format!("PLE row {row_id} maps to missing shard"))
            })?;
        if row >= shard.rows {
            candle_core::bail!("PLE row {row_id} out of range for shard {shard_idx}");
        }
        let n = self.head_dim;
        let start = shard.data_offset + row * n * self.dtype.elem_size();
        match self.dtype {
            PleTableDtype::F8E4M3 => {
                let bytes = shard.data.get(start..start + n);
                for (d, &b) in dst.iter_mut().zip(bytes.iter()) {
                    *d = self.fp8_lut[b as usize] * self.scale;
                }
            }
            PleTableDtype::BF16 => {
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = start + i * 2;
                    *d = half::bf16::from_le_bytes(shard.data.get(o..o + 2).try_into().unwrap()).to_f32();
                }
            }
            PleTableDtype::F16 => {
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = start + i * 2;
                    *d = half::f16::from_le_bytes(shard.data.get(o..o + 2).try_into().unwrap()).to_f32();
                }
            }
            PleTableDtype::F32 => {
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = start + i * 4;
                    *d = f32::from_le_bytes(shard.data.get(o..o + 4).try_into().unwrap());
                }
            }
        }
        Ok(())
    }

    /// Gather rows → flat f32 vec of `row_ids.len() * head_dim`.
    fn gather(&self, row_ids: &[i64]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; row_ids.len() * self.head_dim];
        if row_ids.len() >= 4096 {
            // Prefill: parallel gather from the page cache.
            let err = std::sync::Mutex::new(None);
            out.par_chunks_mut(self.head_dim)
                .zip(row_ids.par_iter())
                .for_each(|(dst, &id)| {
                    if err.lock().unwrap().is_some() {
                        return;
                    }
                    if let Err(e) = self.read_row(id as usize, dst) {
                        *err.lock().unwrap() = Some(e);
                    }
                });
            if let Some(e) = err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            for (dst, &id) in out.chunks_mut(self.head_dim).zip(row_ids.iter()) {
                self.read_row(id as usize, dst)?;
            }
        }
        Ok(out)
    }
}

// =============================================================================
// PLE layer
// =============================================================================

struct PleState {
    conv_state: Tensor, // [capacity, state_len, C]
    ctx: Vec<Vec<i64>>, // last ngram_size-1 tokens per slot
    ctx_valid: Vec<bool>,
    capacity: usize,
}

pub struct Qwen4Ple {
    key_proj: Linear,
    value_proj: Linear,
    norm_key_w: Tensor,
    norm_query_w: Tensor,
    norm_conv_w: Tensor,
    conv_weight: Tensor, // [C, K] f32
    table: PleTable,
    multipliers: Vec<i64>,
    head_sizes: Vec<i64>,
    head_offsets: Vec<i64>,
    ngram_size: usize,
    heads_per_ngram: usize,
    ngram_heads: usize,
    embed_dim: usize,
    eos_token_id: i64,
    hc_count: usize,
    hidden_size: usize,
    rms_norm_eps: f64,
    dilation: usize,
    state_len: usize,
    state: Mutex<PleState>,
    device: Device,
    dtype: DType,
}

impl Qwen4Ple {
    /// `vb` is the decoder-layer VarBuilder (PLE weights live under `ple.*`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilderX,
        config: &crate::utils::config::Config,
        ple: &Qwen4PleConfig,
        ple_dense_layer_id: usize,
        hc_count: usize,
        dtype: DType,
    ) -> Result<Self> {
        if vb.is_qvar_builder() {
            candle_core::bail!("Qwen4 PLE (n-gram embedding) is not supported for GGUF models");
        }
        let device = vb.device();
        let hidden_size = config.hidden_size;
        let hc_hidden = hc_count * hidden_size;
        let ngram_heads = (ple.ngram_size - 1) * ple.heads_per_ngram;
        if ngram_heads == 0 || ple.ple_embed_dim % ngram_heads != 0 {
            candle_core::bail!(
                "ple_embed_dim ({}) must be divisible by ngram_heads ({ngram_heads})",
                ple.ple_embed_dim
            );
        }
        let head_dim = ple.ple_embed_dim / ngram_heads;
        let dilation = ple.ngram_size;
        let state_len = (ple.ple_conv_kernel_size - 1) * dilation;

        let key_w = vb.get_with_hints_dtype(
            (hc_hidden, ple.ple_embed_dim),
            "ple.key_proj.weight",
            shard(0, 0, 1),
            dtype,
        )?;
        let value_w = vb.get_with_hints_dtype(
            (hidden_size, ple.ple_embed_dim),
            "ple.value_proj.weight",
            shard(0, 0, 1),
            dtype,
        )?;
        let key_proj = Linear::new(key_w, None);
        let value_proj = Linear::new(value_w, None);

        let norm_key_w = vb
            .get_with_hints_dtype((hc_hidden,), "ple.norm_key.weight", shard(0, 0, 1), dtype)?
            .reshape((hc_count, hidden_size))?;
        let norm_query_w = vb
            .get_with_hints_dtype((hc_hidden,), "ple.norm_query.weight", shard(0, 0, 1), dtype)?
            .reshape((hc_count, hidden_size))?;
        let norm_conv_w = vb
            .get_with_hints_dtype((hc_hidden,), "ple.norm_conv.weight", shard(0, 0, 1), dtype)?
            .reshape((hc_count, hidden_size))?;
        let conv_weight = vb
            .get_with_hints_dtype(
                (hc_hidden, 1, ple.ple_conv_kernel_size),
                "ple.conv1d.weight",
                shard(0, 0, 1),
                dtype,
            )?
            .reshape((hc_hidden, ple.ple_conv_kernel_size))?
            .to_dtype(DType::F32)?;

        // Hash parameters: prefer the checkpoint's persistent buffers (exact),
        // fall back to the deterministic construction, cross-check when both
        // are available.
        let computed_multipliers = make_layer_multipliers(
            ple.ngram_size,
            config.vocab_size.unwrap_or(0) as u64,
            ple.seed,
            ple_dense_layer_id,
        );
        let (computed_sizes, computed_offsets) = make_vocab_layout(
            ple.ngram_vocab_size_base as u64,
            ngram_heads,
            ple_dense_layer_id,
        );
        let load_i64 = |name: &str, len: usize| -> Option<Vec<i64>> {
            if !vb.has_key(name) {
                return None;
            }
            vb.get_with_hints_dtype((len,), name, shard(0, 0, 1), DType::I64)
                .and_then(|t| t.to_vec1::<i64>())
                .ok()
        };
        let multipliers =
            load_i64("ple.ple_embedding.layer_multipliers", ple.ngram_size).inspect(|v| {
                if *v != computed_multipliers {
                    crate::log_error!(
                        "PLE layer_multipliers mismatch between checkpoint and computed layout"
                    );
                }
            });
        let head_sizes = load_i64("ple.ple_embedding.ngram_heads_vocab_sizes", ngram_heads)
            .inspect(|v| {
                if *v != computed_sizes {
                    crate::log_error!(
                        "PLE ngram_heads_vocab_sizes mismatch between checkpoint and computed layout"
                    );
                }
            });
        let head_offsets =
            load_i64("ple.ple_embedding.ngram_heads_offsets", ngram_heads).inspect(|v| {
                if *v != computed_offsets {
                    crate::log_error!(
                        "PLE ngram_heads_offsets mismatch between checkpoint and computed layout"
                    );
                }
            });
        let multipliers = multipliers.unwrap_or(computed_multipliers);
        let head_sizes = head_sizes.unwrap_or(computed_sizes);
        let head_offsets = head_offsets.unwrap_or(computed_offsets);

        let weight_files = vb.weight_paths().ok_or_else(|| {
            candle_core::Error::wrap("PLE requires safetensors weight file paths".to_string())
        })?;
        let table = PleTable::scan(
            &weight_files,
            &format!("{}.ple.ple_embedding.ngram_embedding", vb.module_path()),
            ple.split_ngram_parts,
            head_dim,
        )?;

        let eos_token_id = config
            .eos_token_id
            .as_ref()
            .map(|e| match e {
                crate::utils::config::EosTokenId::Single(v) => *v as i64,
                crate::utils::config::EosTokenId::Multiple(v) => {
                    v.first().copied().unwrap_or(0) as i64
                }
            })
            .unwrap_or(0);

        Ok(Self {
            key_proj,
            value_proj,
            norm_key_w,
            norm_query_w,
            norm_conv_w,
            conv_weight,
            table,
            multipliers,
            head_sizes,
            head_offsets,
            ngram_size: ple.ngram_size,
            heads_per_ngram: ple.heads_per_ngram,
            ngram_heads,
            embed_dim: ple.ple_embed_dim,
            eos_token_id,
            hc_count,
            hidden_size,
            rms_norm_eps: config.rms_norm_eps,
            dilation,
            state_len,
            state: Mutex::new(PleState {
                conv_state: Tensor::zeros((0, state_len, hc_hidden), dtype, &device)?,
                ctx: Vec::new(),
                ctx_valid: Vec::new(),
                capacity: 0,
            }),
            device,
            dtype,
        })
    }

    /// N-gram row ids for one request chunk, updating nothing.
    fn hash_chunk(&self, chunk: &[i64], ctx: &[i64], row_ids: &mut Vec<i64>) {
        let ngram = self.ngram_size;
        let hpn = self.heads_per_ngram;
        let eos = self.eos_token_id;
        let mut toks = vec![0i64; ngram];
        for (j, &tok) in chunk.iter().enumerate() {
            toks[0] = tok;
            let mut crossed = false;
            for shift in 1..ngram {
                let mut cand = if j >= shift {
                    chunk[j - shift]
                } else {
                    // ctx = [tok(start-(n-1)), ..., tok(start-1)], oldest first
                    let idx = ctx.len() as isize + j as isize - shift as isize;
                    ctx[idx as usize]
                };
                if crossed {
                    cand = eos;
                }
                if cand == eos {
                    crossed = true;
                }
                toks[shift] = cand;
            }
            for h in 0..self.ngram_heads {
                let order = h / hpn + 2;
                let mut mixed = toks[0].wrapping_mul(self.multipliers[0]);
                for (i, t) in toks.iter().enumerate().take(order).skip(1) {
                    mixed ^= t.wrapping_mul(self.multipliers[i]);
                }
                row_ids.push(mixed.rem_euclid(self.head_sizes[h]) + self.head_offsets[h]);
            }
        }
    }

    fn ensure_capacity(&self, st: &mut PleState, needed: usize) -> Result<()> {
        if needed <= st.capacity {
            return Ok(());
        }
        let new_cap = needed.max(32).max(st.capacity * 2);
        let c = self.hc_count * self.hidden_size;
        let new_state = Tensor::zeros((new_cap, self.state_len, c), self.dtype, &self.device)?;
        if st.capacity > 0 {
            new_state
                .narrow(0, 0, st.capacity)?
                .copy_(&st.conv_state, 0)?;
        }
        st.conv_state = new_state;
        st.ctx
            .resize(new_cap, vec![self.eos_token_id; self.ngram_size - 1]);
        st.ctx_valid.resize(new_cap, false);
        st.capacity = new_cap;
        Ok(())
    }

    /// Grouped RMSNorm with (1 + weight) scale, rounding to model dtype at the
    /// boundary (matches the reference kernel).
    fn grouped_norm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.rms_norm_eps)?.sqrt()?)?;
        normed
            .broadcast_mul(&(w.to_dtype(DType::F32)? + 1.0)?)?
            .to_dtype(self.dtype)
    }

    fn gate(&self, key: &Tensor, value: &Tensor, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = hidden.dim(0)?;
        let hc = self.hc_count;
        let h = self.hidden_size;
        let key = key.reshape((t, hc, h))?;
        let query = hidden.reshape((t, hc, h))?;

        let k_n = self.grouped_norm(&key, &self.norm_key_w)?;
        let q_n = self.grouped_norm(&query, &self.norm_query_w)?;
        // Reference rounding: products and the dot are rounded to model dtype.
        let products =
            (k_n.to_dtype(DType::F32)? * q_n.to_dtype(DType::F32)?)?.to_dtype(self.dtype)?;
        let dot = products
            .to_dtype(DType::F32)?
            .sum_keepdim(candle_core::D::Minus1)?
            .to_dtype(self.dtype)?; // [t, hc, 1]
        let d = (dot.to_dtype(DType::F32)? / (h as f64).sqrt())?.to_dtype(self.dtype)?;
        let df = d.to_dtype(DType::F32)?;
        let zeros = df.zeros_like()?;
        let sign = (df.gt(&zeros)?.to_dtype(DType::F32)? - df.lt(&zeros)?.to_dtype(DType::F32)?)?;
        let magnitude = df.abs()?.maximum(1e-6)?.sqrt()?.to_dtype(self.dtype)?;
        let g = candle_nn::ops::sigmoid(&(sign * magnitude.to_dtype(DType::F32)?)?)?
            .to_dtype(self.dtype)?; // [t, hc, 1]

        let v = value.reshape((t, 1, h))?.to_dtype(DType::F32)?;
        let gated = g
            .to_dtype(DType::F32)?
            .broadcast_mul(&v)?
            .to_dtype(self.dtype)?; // [t, hc, h]

        let gf = gated.to_dtype(DType::F32)?;
        let var = gf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = gf
            .broadcast_div(&(var + self.rms_norm_eps)?.sqrt()?)?
            .broadcast_mul(&(self.norm_conv_w.to_dtype(DType::F32)? + 1.0)?)?;
        let conv_input = normed.to_dtype(self.dtype)?.reshape((t, hc * h))?;
        Ok((gated.reshape((t, hc * h))?, conv_input))
    }

    /// Causal depthwise short conv (kernel K, dilation = ngram_size) with a
    /// per-slot persistent state of the last `state_len` inputs. Adds nothing;
    /// returns the conv output (SiLU-activated) to be added to `gated`.
    fn short_conv(
        &self,
        conv_input: &Tensor,
        req_lens: &[usize],
        slots: &[usize],
    ) -> Result<Tensor> {
        let c = self.hc_count * self.hidden_size;
        let k = self.conv_weight.dim(1)?;
        let dil = self.dilation;
        let state_len = self.state_len;
        let st = self
            .state
            .lock()
            .map_err(|e| candle_core::Error::wrap(format!("PLE state lock poisoned: {e}")))?;

        let mut outs = Vec::with_capacity(req_lens.len());
        let mut offset = 0usize;
        for (r, &len) in req_lens.iter().enumerate() {
            let slot = slots[r];
            let x = conv_input.narrow(0, offset, len)?; // [L, C]
            let state = st.conv_state.narrow(0, slot, 1)?.squeeze(0)?; // [state_len, C]
            let xfull = Tensor::cat(&[&state, &x], 0)?; // [state_len + L, C]
            let xfull = xfull.to_dtype(DType::F32)?;
            let mut acc: Option<Tensor> = None;
            for kk in 0..k {
                let tap = xfull.narrow(0, kk * dil, len)?; // [L, C]
                let w = self.conv_weight.narrow(1, kk, 1)?.reshape((c,))?; // [C]
                let term = tap.broadcast_mul(&w)?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => (a + term)?,
                });
            }
            let acc = acc.expect("PLE conv kernel size must be >= 1");
            // Reference rounds the accumulator to model dtype before SiLU.
            let conv = acc.to_dtype(self.dtype)?.to_dtype(DType::F32)?;
            let y = (conv.clone() * candle_nn::ops::sigmoid(&conv)?)?.to_dtype(self.dtype)?;
            outs.push(y);
            // Write back the last state_len inputs as the new state.
            let new_state = xfull.narrow(0, len, state_len)?.to_dtype(self.dtype)?;
            // NOTE: this candle fork's `copy_` takes the shape/offset from the
            // *destination* layout, so a narrowed dst with a nonzero start
            // offset would read the src out of bounds. Copy into the full
            // state tensor with an explicit element offset instead.
            st.conv_state
                .copy_(&new_state.unsqueeze(0)?, slot * state_len * c)?;
            offset += len;
        }
        Tensor::cat(&outs, 0)
    }

    /// Returns the PLE delta to add to the hyper-connection stream `xs`.
    pub fn forward(
        &self,
        xs: &Tensor,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let num_tokens = xs.dim(0)?;
        if num_tokens == 0 {
            return Tensor::zeros_like(xs);
        }
        if input_metadata.is_mtp_verify {
            candle_core::bail!(
                "Qwen4 PLE does not support MTP/DFlash verify steps yet; disable speculative decoding"
            );
        }

        let ids: Vec<i64> = match input_ids.dtype() {
            DType::I64 => input_ids.to_vec1::<i64>()?,
            DType::U32 => input_ids
                .to_vec1::<u32>()?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            DType::U8 => input_ids
                .to_vec1::<u8>()?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            dt => candle_core::bail!(
                "Qwen4 PLE expects raw token ids (got {dt:?}); embedded multimodal inputs are not supported with PLE"
            ),
        };
        let slots: Vec<usize> = seq_slots
            .to_vec1::<i64>()?
            .iter()
            .map(|&s| s as usize)
            .collect();
        let pos: Vec<i64> = match positions.dtype() {
            DType::I64 => positions.to_vec1::<i64>()?,
            DType::U32 => positions
                .to_vec1::<u32>()?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            dt => candle_core::bail!("Qwen4 PLE unexpected positions dtype {dt:?}"),
        };

        // Per-request token counts in this (flattened) forward pass.
        let req_lens: Vec<usize> = if input_metadata.is_prefill {
            // Prefill `seqlens` holds cumulative end offsets per request.
            let cum = input_metadata.seqlens.clone().unwrap_or_default();
            let mut lens = Vec::with_capacity(cum.len());
            let mut prev = 0u32;
            for c in &cum {
                lens.push((*c - prev) as usize);
                prev = *c;
            }
            lens
        } else {
            vec![1usize; slots.len()]
        };
        if req_lens.len() != slots.len() || req_lens.iter().sum::<usize>() != num_tokens {
            candle_core::bail!(
                "Qwen4 PLE batch layout mismatch: req_lens={req_lens:?} slots={} tokens={num_tokens}",
                slots.len()
            );
        }

        // Hash + gather on CPU (table is host-resident).
        let emb = {
            let mut st = self
                .state
                .lock()
                .map_err(|e| candle_core::Error::wrap(format!("PLE state lock poisoned: {e}")))?;
            let max_slot = slots.iter().copied().max().unwrap_or(0);
            self.ensure_capacity(&mut st, max_slot + 1)?;

            let mut row_ids = Vec::with_capacity(num_tokens * self.ngram_heads);
            let mut offset = 0usize;
            for (r, &len) in req_lens.iter().enumerate() {
                let slot = slots[r];
                let chunk = &ids[offset..offset + len];
                let fresh = pos[offset] == 0;
                let ctx = if fresh || !st.ctx_valid[slot] {
                    vec![self.eos_token_id; self.ngram_size - 1]
                } else {
                    st.ctx[slot].clone()
                };
                self.hash_chunk(chunk, &ctx, &mut row_ids);
                // Update the per-slot trailing context.
                let mut hist = ctx;
                hist.extend_from_slice(chunk);
                let keep = self.ngram_size - 1;
                st.ctx[slot] = hist[hist.len() - keep..].to_vec();
                st.ctx_valid[slot] = true;
                // A fresh sequence also resets the conv state.
                if fresh {
                    let c = self.hc_count * self.hidden_size;
                    let zeros = Tensor::zeros((1, self.state_len, c), self.dtype, &self.device)?;
                    // See the `copy_` note in `short_conv`: full-tensor dst +
                    // explicit element offset (narrowed dsts are unsupported).
                    st.conv_state.copy_(&zeros, slot * self.state_len * c)?;
                }
                offset += len;
            }
            self.table.gather(&row_ids)?
        };

        let emb = Tensor::from_vec(emb, (num_tokens, self.embed_dim), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let key = self.key_proj.forward(&emb)?; // [T, hc*hidden]
        let value = self.value_proj.forward(&emb)?; // [T, hidden]

        let (gated, conv_input) = self.gate(&key, &value, xs)?;
        let conv_out = self.short_conv(&conv_input, &req_lens, &slots)?;
        gated + conv_out
    }
}
