//! Model-owned DeepSeek V4 hybrid page pool (vLLM-style).
//!
//! Pages are keyed by the engine `block_table` indices (native block = 256 tokens).
//! Compressor residual lives in residual pages — prefix reuse is block-table reuse,
//! not mamba-style side-store snapshots.

use super::hybrid_kv::{V4LayerCacheSpec, V4_NATIVE_BLOCK_SIZE};
use candle_core::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Per-layer hybrid page tensors for one V4 attention layer.
pub struct V4LayerPages {
    /// `[num_pages, 256, head_dim]` BF16 token KV (SWA ring is a view over these).
    pub swa: Tensor,
    /// `[num_pages, 256/ratio, head_dim]` BF16 compressed KV (None for SWA-only).
    pub compressed: Option<Tensor>,
    /// `[num_pages, residual_window, 2*state_width]` F32 packed `[kv|score]`.
    pub residual: Option<Tensor>,
    /// `[num_pages, 256/4, index_head_dim]` BF16 (C4 only).
    pub indexer_compressed: Option<Tensor>,
    /// `[num_pages, 8, 2*(2*index_head_dim)]` F32 (C4 only).
    pub indexer_residual: Option<Tensor>,
    pub compress_ratio: usize,
    pub head_dim: usize,
    pub residual_window: usize,
    pub index_head_dim: Option<usize>,
}

impl V4LayerPages {
    pub fn new(spec: &V4LayerCacheSpec, num_pages: usize, device: &Device) -> Result<Self> {
        let swa = Tensor::zeros(
            (num_pages, V4_NATIVE_BLOCK_SIZE, spec.head_dim),
            DType::BF16,
            device,
        )?;
        let compressed = if spec.compress_ratio > 0 {
            let rows = spec.compressed_entries_per_native_block();
            Some(Tensor::zeros(
                (num_pages, rows, spec.head_dim),
                DType::BF16,
                device,
            )?)
        } else {
            None
        };
        let residual = if spec.residual_window > 0 {
            Some(Tensor::zeros(
                (num_pages, spec.residual_window, spec.packed_residual_dim()),
                DType::F32,
                device,
            )?)
        } else {
            None
        };
        let (indexer_compressed, indexer_residual) = if let Some(idx_dim) = spec.index_head_dim {
            let ic = Tensor::zeros(
                (
                    num_pages,
                    spec.compressed_entries_per_native_block(),
                    idx_dim,
                ),
                DType::BF16,
                device,
            )?;
            let packed = 2 * 2 * idx_dim; // coff=2 for indexer
            let ir = Tensor::zeros(
                (num_pages, spec.residual_window, packed),
                DType::F32,
                device,
            )?;
            (Some(ic), Some(ir))
        } else {
            (None, None)
        };
        Ok(Self {
            swa,
            compressed,
            residual,
            indexer_compressed,
            indexer_residual,
            compress_ratio: spec.compress_ratio,
            head_dim: spec.head_dim,
            residual_window: spec.residual_window,
            index_head_dim: spec.index_head_dim,
        })
    }

    pub fn zero_page(&self, page: usize) -> Result<()> {
        let n = self.swa.dim(0)?;
        if page >= n {
            return Ok(());
        }
        self.swa.narrow(0, page, 1)?.zero_()?;
        if let Some(c) = &self.compressed {
            c.narrow(0, page, 1)?.zero_()?;
        }
        if let Some(r) = &self.residual {
            r.narrow(0, page, 1)?.zero_()?;
        }
        if let Some(c) = &self.indexer_compressed {
            c.narrow(0, page, 1)?.zero_()?;
        }
        if let Some(r) = &self.indexer_residual {
            r.narrow(0, page, 1)?.zero_()?;
        }
        Ok(())
    }
}

/// Global hybrid page pool shared across sequences via engine block IDs.
pub struct V4HybridPagePool {
    pub native_block_size: usize,
    pub num_pages: usize,
    pub layers: Vec<V4LayerPages>,
    /// Per physical page: residual was frozen at a native-block boundary.
    /// Prefix hits may only hydrate from pages marked frozen.
    residual_frozen: Vec<AtomicBool>,
}

impl V4HybridPagePool {
    pub fn new(specs: &[V4LayerCacheSpec], num_pages: usize, device: &Device) -> Result<Self> {
        if num_pages == 0 {
            candle_core::bail!("V4HybridPagePool requires num_pages > 0");
        }
        let mut layers = Vec::with_capacity(specs.len());
        for spec in specs {
            layers.push(V4LayerPages::new(spec, num_pages, device)?);
        }
        crate::log_info!(
            "DeepSeek V4 hybrid page pool: {} pages x {} layers (native_block={})",
            num_pages,
            layers.len(),
            V4_NATIVE_BLOCK_SIZE
        );
        let residual_frozen = (0..num_pages).map(|_| AtomicBool::new(false)).collect();
        Ok(Self {
            native_block_size: V4_NATIVE_BLOCK_SIZE,
            num_pages,
            layers,
            residual_frozen,
        })
    }

    /// True iff `native_len` is a native-block boundary and the physical page
    /// covering that boundary has a frozen residual (prefix-safe handoff).
    pub fn residual_frozen_at(&self, native_len: usize, block_table: &[u32]) -> bool {
        if native_len == 0 || native_len % self.native_block_size != 0 {
            return false;
        }
        let page_idx = (native_len - 1) / self.native_block_size;
        let Some(&page) = block_table.get(page_idx) else {
            return false;
        };
        self.residual_frozen
            .get(page as usize)
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Mark residual on the physical page covering `native_len - 1` as frozen.
    /// Only valid when `native_len` is a multiple of the native block size.
    pub fn mark_residual_frozen(&self, native_len: usize, block_table: &[u32]) {
        if native_len == 0 || native_len % self.native_block_size != 0 {
            return;
        }
        let page_idx = (native_len - 1) / self.native_block_size;
        let Some(&page) = block_table.get(page_idx) else {
            return;
        };
        if let Some(flag) = self.residual_frozen.get(page as usize) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Clear freeze flag when a physical page is recycled.
    pub fn clear_residual_frozen(&self, page: usize) {
        if let Some(flag) = self.residual_frozen.get(page) {
            flag.store(false, Ordering::Relaxed);
        }
    }

    /// Zero all layer tensors for a physical page (block recycle).
    pub fn zero_page(&self, page: usize) -> Result<()> {
        for layer in &self.layers {
            layer.zero_page(page)?;
        }
        Ok(())
    }

    pub fn layer(&self, idx: usize) -> Option<&V4LayerPages> {
        self.layers.get(idx)
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut V4LayerPages> {
        self.layers.get_mut(idx)
    }

    /// Write SWA token rows into a native page at absolute positions.
    /// `token_kv`: `[n, head_dim]` BF16; `abs_positions`: host i64 positions.
    pub fn write_swa_rows(
        &self,
        layer_idx: usize,
        token_kv: &Tensor,
        abs_positions: &[i64],
        block_table: &[u32],
    ) -> Result<()> {
        let Some(layer) = self.layer(layer_idx) else {
            return Ok(());
        };
        let n = token_kv.dim(0)?;
        if n != abs_positions.len() {
            candle_core::bail!(
                "write_swa_rows length mismatch: kv={} pos={}",
                n,
                abs_positions.len()
            );
        }
        let bs = self.native_block_size;
        let hd = layer.head_dim;
        // Contiguous page runs — typically ~ceil(n/256) copies instead of n.
        let mut i = 0;
        while i < n {
            let pos = abs_positions[i];
            if pos < 0 {
                i += 1;
                continue;
            }
            let pos = pos as usize;
            let page_idx = pos / bs;
            let row = pos % bs;
            let mut take = 1;
            while i + take < n {
                let p2 = abs_positions[i + take];
                if p2 < 0 || p2 as usize != pos + take {
                    break;
                }
                if (pos + take) / bs != page_idx {
                    break;
                }
                take += 1;
            }
            let Some(&page) = block_table.get(page_idx) else {
                i += take;
                continue;
            };
            let page = page as usize;
            if page >= self.num_pages {
                i += take;
                continue;
            }
            let src = token_kv.narrow(0, i, take)?.contiguous()?;
            let dst = layer.swa.narrow(0, page, 1)?.squeeze(0)?;
            attention_rs::deepseek_v4::copy_contiguous_into(&dst, &src, row * hd)?;
            i += take;
        }
        Ok(())
    }

    /// Write compressed rows into pages. `compressed`: `[n, head_dim]`,
    /// `first_row` is the absolute compressed-row index of the first row.
    pub fn write_compressed_rows(
        &self,
        layer_idx: usize,
        compressed: &Tensor,
        first_row: usize,
        block_table: &[u32],
    ) -> Result<()> {
        let Some(layer) = self.layer(layer_idx) else {
            return Ok(());
        };
        let Some(cache) = &layer.compressed else {
            return Ok(());
        };
        let ratio = layer.compress_ratio.max(1);
        let rows_per_page = V4_NATIVE_BLOCK_SIZE / ratio;
        let n = compressed.dim(0)?;
        let hd = layer.head_dim;
        let mut i = 0;
        while i < n {
            let abs_row = first_row + i;
            let page_idx = abs_row / rows_per_page;
            let row = abs_row % rows_per_page;
            let room = rows_per_page - row;
            let take = room.min(n - i);
            let Some(&page) = block_table.get(page_idx) else {
                i += take;
                continue;
            };
            let page = page as usize;
            if page >= self.num_pages {
                i += take;
                continue;
            }
            let src = compressed.narrow(0, i, take)?.contiguous()?;
            let dst = cache.narrow(0, page, 1)?.squeeze(0)?;
            attention_rs::deepseek_v4::copy_contiguous_into(&dst, &src, row * hd)?;
            i += take;
        }
        Ok(())
    }

    /// Pack compressor residual scratch into the residual page for the last
    /// native block covering `native_len` tokens.
    ///
    /// `kv_state` / `score_state`: `[slots, state_dim]` F32 (CompressorDecodeState layout).
    /// Overlap (ratio=4): slots=8, state_dim=2*head_dim → packed dim = 4*head_dim.
    /// Non-overlap: slots=ratio, state_dim=head_dim → packed dim = 2*head_dim.
    pub fn save_residual_from_state(
        &self,
        layer_idx: usize,
        kv_state: &Tensor,
        score_state: &Tensor,
        native_len: usize,
        block_table: &[u32],
        indexer: bool,
    ) -> Result<()> {
        let Some(layer) = self.layer(layer_idx) else {
            return Ok(());
        };
        let residual = if indexer {
            layer.indexer_residual.as_ref()
        } else {
            layer.residual.as_ref()
        };
        let Some(residual) = residual else {
            return Ok(());
        };
        if native_len == 0 || block_table.is_empty() {
            return Ok(());
        }
        let page_idx = (native_len - 1) / self.native_block_size;
        let Some(&page) = block_table.get(page_idx) else {
            return Ok(());
        };
        let page = page as usize;
        if page >= self.num_pages {
            return Ok(());
        }
        let win = layer.residual_window.min(kv_state.dim(0)?);
        if win == 0 {
            return Ok(());
        }
        let kv = kv_state.narrow(0, 0, win)?.contiguous()?;
        let score = score_state.narrow(0, 0, win)?.contiguous()?;
        let packed = Tensor::cat(&[&kv, &score], 1)?;
        let dst = residual.narrow(0, page, 1)?.squeeze(0)?;
        let packed_dim = packed.dim(1)?;
        let dst_dim = dst.dim(1)?;
        if packed_dim != dst_dim {
            crate::log_warn!(
                "V4 residual pack dim mismatch layer {} indexer={}: {} vs {}",
                layer_idx,
                indexer,
                packed_dim,
                dst_dim
            );
            return Ok(());
        }
        attention_rs::deepseek_v4::copy_contiguous_into(&dst, &packed, 0)?;
        Ok(())
    }

    /// After saving residual for `native_len`, update freeze flags for the
    /// physical page. Only exact native-block boundaries are prefix-safe.
    pub fn update_residual_freeze(&self, native_len: usize, block_table: &[u32]) {
        if native_len == 0 || block_table.is_empty() {
            return;
        }
        let page_idx = (native_len - 1) / self.native_block_size;
        let Some(&page) = block_table.get(page_idx) else {
            return;
        };
        let page = page as usize;
        if native_len % self.native_block_size == 0 {
            if let Some(flag) = self.residual_frozen.get(page) {
                flag.store(true, Ordering::Relaxed);
            }
        } else if let Some(flag) = self.residual_frozen.get(page) {
            // Mid-page residual is not a stable prefix handoff.
            flag.store(false, Ordering::Relaxed);
        }
    }

    /// Restore compressor residual scratch from the residual page covering
    /// `native_len` (used on prefix-cache hit / chunk handoff).
    pub fn load_residual_into_state(
        &self,
        layer_idx: usize,
        kv_state: &Tensor,
        score_state: &Tensor,
        native_len: usize,
        block_table: &[u32],
        indexer: bool,
    ) -> Result<()> {
        let Some(layer) = self.layer(layer_idx) else {
            return Ok(());
        };
        let residual = if indexer {
            layer.indexer_residual.as_ref()
        } else {
            layer.residual.as_ref()
        };
        let Some(residual) = residual else {
            return Ok(());
        };
        if native_len == 0 || block_table.is_empty() {
            return Ok(());
        }
        let page_idx = (native_len - 1) / self.native_block_size;
        let Some(&page) = block_table.get(page_idx) else {
            return Ok(());
        };
        let page = page as usize;
        if page >= self.num_pages {
            return Ok(());
        }
        let win = layer.residual_window.min(kv_state.dim(0)?);
        if win == 0 {
            return Ok(());
        }
        let src = residual.narrow(0, page, 1)?.squeeze(0)?;
        let state_dim = kv_state.dim(1)?;
        let packed_dim = src.dim(1)?;
        if packed_dim != 2 * state_dim {
            return Ok(());
        }
        let kv = src.narrow(1, 0, state_dim)?.contiguous()?;
        let score = src.narrow(1, state_dim, state_dim)?.contiguous()?;
        attention_rs::deepseek_v4::copy_contiguous_into(kv_state, &kv, 0)?;
        attention_rs::deepseek_v4::copy_contiguous_into(score_state, &score, 0)?;
        Ok(())
    }

    /// Gather SWA∪compressed from pages into a contiguous sparse buffer for attention.
    ///
    /// Layout matches `LayerSparseKvCache`: `[window + compressed_slots, head_dim]`.
    /// Window ring uses absolute positions modulo `sliding_window`.
    pub fn gather_sparse_into(
        &self,
        layer_idx: usize,
        sparse_kv: &Tensor,
        sliding_window: usize,
        compressed_len: usize,
        native_len: usize,
        block_table: &[u32],
    ) -> Result<()> {
        let Some(layer) = self.layer(layer_idx) else {
            return Ok(());
        };
        if native_len == 0 {
            return Ok(());
        }
        let hd = layer.head_dim;
        let bs = self.native_block_size;
        // Rebuild window ring from the last `sliding_window` tokens in page runs.
        let win_start = native_len.saturating_sub(sliding_window);
        let mut pos = win_start;
        while pos < native_len {
            let page_idx = pos / bs;
            let row = pos % bs;
            let room = bs - row;
            let take = room.min(native_len - pos);
            let Some(&page) = block_table.get(page_idx) else {
                pos += take;
                continue;
            };
            let page = page as usize;
            if page >= self.num_pages {
                pos += take;
                continue;
            }
            // Copy page rows into ring slots. Slots wrap; split at window boundary.
            let mut copied = 0;
            while copied < take {
                let abs = pos + copied;
                let slot = abs % sliding_window;
                let run = (sliding_window - slot).min(take - copied);
                let src = layer
                    .swa
                    .narrow(0, page, 1)?
                    .squeeze(0)?
                    .narrow(0, row + copied, run)?
                    .contiguous()?;
                attention_rs::deepseek_v4::copy_contiguous_into(sparse_kv, &src, slot * hd)?;
                copied += run;
            }
            pos += take;
        }
        // Copy compressed rows [0, compressed_len) in page runs.
        if let Some(cache) = &layer.compressed {
            let ratio = layer.compress_ratio.max(1);
            let rows_per_page = bs / ratio;
            let mut abs_row = 0;
            while abs_row < compressed_len {
                let page_idx = abs_row / rows_per_page;
                let row = abs_row % rows_per_page;
                let room = rows_per_page - row;
                let take = room.min(compressed_len - abs_row);
                let Some(&page) = block_table.get(page_idx) else {
                    abs_row += take;
                    continue;
                };
                let page = page as usize;
                if page >= self.num_pages {
                    abs_row += take;
                    continue;
                }
                let src = cache
                    .narrow(0, page, 1)?
                    .squeeze(0)?
                    .narrow(0, row, take)?
                    .contiguous()?;
                attention_rs::deepseek_v4::copy_contiguous_into(
                    sparse_kv,
                    &src,
                    (sliding_window + abs_row) * hd,
                )?;
                abs_row += take;
            }
        }
        Ok(())
    }
}
