//! DeepSeek V4 hybrid KV cache specifications (vLLM-style).
//!
//! Logical unit is native token positions (256). Compressor residual is stored as
//! SWA-spec pages with `sliding_window = coff * compress_ratio` (packed `[kv|score]`
//! F32), matching vLLM `CompressorStateCache` / `SlidingWindowMLASpec`.

/// Native-token block size for V4 hybrid pools (vLLM logical unit).
pub const V4_NATIVE_BLOCK_SIZE: usize = 256;

/// Per-layer hybrid cache kinds driven by `compress_ratios[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4CacheKind {
    /// Sliding-window token KV (always present).
    Swa,
    /// Compressed KV pages at ratio 4 (overlap + indexer).
    C4,
    /// Compressed KV pages at ratio 128 (non-overlap, dense over pool).
    C128,
    /// Compressor residual registered as SWA with window = coff * ratio.
    CompressorResidual,
    /// Indexer compressed pages (ratio-4 layers only).
    IndexerCompressed,
    /// Indexer residual as SWA (ratio-4 layers only).
    IndexerResidual,
}

#[derive(Debug, Clone)]
pub struct V4LayerCacheSpec {
    pub layer_idx: usize,
    pub compress_ratio: usize,
    pub sliding_window: usize,
    pub head_dim: usize,
    pub index_head_dim: Option<usize>,
    /// Residual SWA window = coff * ratio (overlap coff=2 → 8 for C4; 128 for C128).
    pub residual_window: usize,
}

impl V4LayerCacheSpec {
    pub fn new(
        layer_idx: usize,
        compress_ratio: usize,
        sliding_window: usize,
        head_dim: usize,
        index_head_dim: Option<usize>,
        is_overlap: bool,
    ) -> Self {
        let coff = if is_overlap { 2 } else { 1 };
        let residual_window = if compress_ratio > 0 {
            coff * compress_ratio
        } else {
            0
        };
        Self {
            layer_idx,
            compress_ratio,
            sliding_window,
            head_dim,
            index_head_dim,
            residual_window,
        }
    }

    pub fn coff(&self) -> usize {
        if self.compress_ratio == 4 {
            2
        } else {
            1
        }
    }

    /// Width of one residual half (kv or score): `coff * head_dim`.
    pub fn residual_state_width(&self) -> usize {
        self.coff() * self.head_dim
    }

    /// Packed residual row dim: `[kv | score]` = `2 * state_width`.
    pub fn packed_residual_dim(&self) -> usize {
        2 * self.residual_state_width()
    }

    pub fn kinds(&self) -> Vec<V4CacheKind> {
        let mut kinds = vec![V4CacheKind::Swa];
        match self.compress_ratio {
            4 => {
                kinds.push(V4CacheKind::C4);
                kinds.push(V4CacheKind::CompressorResidual);
                if self.index_head_dim.is_some() {
                    kinds.push(V4CacheKind::IndexerCompressed);
                    kinds.push(V4CacheKind::IndexerResidual);
                }
            }
            128 => {
                kinds.push(V4CacheKind::C128);
                kinds.push(V4CacheKind::CompressorResidual);
            }
            _ => {}
        }
        kinds
    }

    /// Compressed entries stored per native block.
    pub fn compressed_entries_per_native_block(&self) -> usize {
        if self.compress_ratio == 0 {
            0
        } else {
            V4_NATIVE_BLOCK_SIZE / self.compress_ratio
        }
    }

    pub fn entries_per_page(&self, kind: V4CacheKind) -> usize {
        match kind {
            V4CacheKind::Swa => V4_NATIVE_BLOCK_SIZE,
            V4CacheKind::C4 | V4CacheKind::C128 | V4CacheKind::IndexerCompressed => {
                self.compressed_entries_per_native_block()
            }
            V4CacheKind::CompressorResidual | V4CacheKind::IndexerResidual => self.residual_window,
        }
    }

    pub fn page_row_bytes(&self, kind: V4CacheKind) -> usize {
        match kind {
            V4CacheKind::Swa | V4CacheKind::C4 | V4CacheKind::C128 => {
                // BF16 head_dim
                self.head_dim * 2
            }
            V4CacheKind::IndexerCompressed => {
                let dim = self.index_head_dim.unwrap_or(128);
                dim * 2
            }
            V4CacheKind::CompressorResidual => {
                // F32 packed [kv|score]
                self.packed_residual_dim() * 4
            }
            V4CacheKind::IndexerResidual => {
                let dim = self.index_head_dim.unwrap_or(128);
                let coff = 2usize; // indexer is always overlap (ratio 4)
                let packed = 2 * coff * dim;
                packed * 4
            }
        }
    }

    pub fn page_size_bytes(&self, kind: V4CacheKind) -> usize {
        self.entries_per_page(kind) * self.page_row_bytes(kind)
    }

    /// Bytes for one native page across all kinds on this layer.
    pub fn native_page_bytes(&self) -> usize {
        self.kinds()
            .into_iter()
            .map(|k| self.page_size_bytes(k))
            .sum()
    }
}

/// Build per-layer specs from V4 config compress ratios.
pub fn build_v4_cache_specs(
    compress_ratios: &[usize],
    sliding_window: usize,
    head_dim: usize,
    index_head_dim: usize,
) -> Vec<V4LayerCacheSpec> {
    compress_ratios
        .iter()
        .enumerate()
        .map(|(i, &ratio)| {
            let is_overlap = ratio == 4;
            let idx_dim = if ratio == 4 {
                Some(index_head_dim)
            } else {
                None
            };
            V4LayerCacheSpec::new(i, ratio, sliding_window, head_dim, idx_dim, is_overlap)
        })
        .collect()
}

/// Total bytes for one native block across all layers.
pub fn v4_bytes_per_native_page(specs: &[V4LayerCacheSpec]) -> usize {
    specs.iter().map(|s| s.native_page_bytes()).sum()
}

/// Align a token length down to the nearest native block boundary.
pub fn align_native_block(tokens: usize) -> usize {
    tokens / V4_NATIVE_BLOCK_SIZE * V4_NATIVE_BLOCK_SIZE
}
