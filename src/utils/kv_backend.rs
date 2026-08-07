//! First-class KV cache backends: Flash (paged K/V), classical MLA, DeepSeek V4 hybrid.
//!
//! Engine owns the physical tensors for all three. Models consume the matching
//! variant; DeepSeek V4 writes through `slot_mapping` into hybrid pages (vLLM-style).

use crate::models::layers::ds_v4::V4HybridPagePool;
use candle_core::Tensor;
use parking_lot::Mutex;
use std::sync::Arc;

/// Which physical KV layout the engine allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheBackend {
    /// Standard paged `(K, V)` (FlashInfer / FlashAttention / Metal).
    Flash,
    /// Classical MLA `(ckv, kpe)` (DeepSeek V3 / GLM MoE DSA, etc.).
    Mla,
    /// DeepSeek V4 hybrid pages (SWA + compressed + residual + indexer).
    DeepSeekV4,
}

impl KvCacheBackend {
    pub fn is_mla(self) -> bool {
        matches!(self, Self::Mla)
    }

    pub fn is_deepseek_v4(self) -> bool {
        matches!(self, Self::DeepSeekV4)
    }

    pub fn is_flash(self) -> bool {
        matches!(self, Self::Flash)
    }
}

/// GPU-resident KV tensors owned by [`crate::core::runner::ModelRunner`].
///
/// DeepSeek V4 shares the same `Arc<Mutex<Option<V4HybridPagePool>>>` with the
/// model so layers and the engine see one pool.
pub enum GpuKvCache {
    Flash(Vec<(Tensor, Tensor)>),
    Mla(Vec<(Tensor, Tensor)>),
    DeepSeekV4(Arc<Mutex<Option<V4HybridPagePool>>>),
}

impl GpuKvCache {
    pub fn backend(&self) -> KvCacheBackend {
        match self {
            Self::Flash(_) => KvCacheBackend::Flash,
            Self::Mla(_) => KvCacheBackend::Mla,
            Self::DeepSeekV4(_) => KvCacheBackend::DeepSeekV4,
        }
    }

    /// Layer `(K,V)` or `(ckv,kpe)` pairs — Flash / MLA only.
    pub fn as_pairs(&self) -> Option<&Vec<(Tensor, Tensor)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4(_) => None,
        }
    }

    pub fn as_pairs_mut(&mut self) -> Option<&mut Vec<(Tensor, Tensor)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4(_) => None,
        }
    }

    pub fn as_v4_pool(&self) -> Option<&Arc<Mutex<Option<V4HybridPagePool>>>> {
        match self {
            Self::DeepSeekV4(p) => Some(p),
            _ => None,
        }
    }

    /// Number of attention layers represented (pairs len, or V4 layer count).
    pub fn num_layers(&self) -> usize {
        match self {
            Self::Flash(v) | Self::Mla(v) => v.len(),
            Self::DeepSeekV4(p) => p.lock().as_ref().map(|pool| pool.layers.len()).unwrap_or(0),
        }
    }
}

/// CPU swap mirror for Flash / MLA. V4 CPU swap is deferred.
pub enum CpuKvCache {
    Flash(Vec<(Tensor, Tensor)>),
    Mla(Vec<(Tensor, Tensor)>),
    DeepSeekV4,
}

impl CpuKvCache {
    pub fn as_pairs(&self) -> Option<&Vec<(Tensor, Tensor)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4 => None,
        }
    }

    pub fn as_pairs_mut(&mut self) -> Option<&mut Vec<(Tensor, Tensor)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4 => None,
        }
    }
}
