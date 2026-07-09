// src/core/block_manager.rs
use super::kv_swap::{
    split_prefix_suffix, suffix_swap_in_pairs, suffix_swap_pairs, CpuOffloadEntry, SeqSwapState,
};
use super::prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixCacheUpdate};
use super::runner::RunnerType;
use super::sequence::{Sequence, SequenceStatus};
use crate::def_broadcast_message_to_runners;
use crate::runner::{receive_local, send_local, MessageType};
use crate::utils::env::{mamba_snapshot_block_stride_blocks, MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV};
use crate::utils::image::ImageData;
use candle_core::Result;
use interprocess::{local_socket::Stream as LocalStream, TryClone};
use parking_lot::RwLock;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Block {
    pub id: usize,
    pub ref_count: usize,
}

impl Block {
    pub fn reset(&mut self) {
        self.ref_count = 1;
    }
}

pub struct BlockManager {
    blocks: Vec<Block>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
    // CPU blocks (for swapping)
    cpu_blocks: Vec<Block>,
    free_cpu_block_ids: VecDeque<usize>,
    // mapping for swapped sequences: seq.id -> Vec<cpu_block_id>
    swapped_map: HashMap<usize, Vec<usize>>,
    block_size: usize,
    runners: Arc<RwLock<RunnerType>>,
    prefix_cache: Option<PrefixCache>,
    mamba_prefix_enabled: bool,
    mamba_snapshot_block_stride_blocks: usize,
    mamba_prefix_hashes_by_block: HashMap<usize, HashSet<u64>>,
    mamba_prefix_block_by_hash: HashMap<u64, usize>,
    valid_mamba_prefix_hashes: HashSet<u64>,
    /// Suffix-only CPU preempt state per sequence (prefix-cache mode).
    seq_swap_states: HashMap<usize, SeqSwapState>,
    /// CPU copies of evicted prefix-cache blocks keyed by trie hash.
    cpu_offload_entries: HashMap<u64, CpuOffloadEntry>,
    /// Trie hashes in LRU order (front = oldest) for CPU offload tier eviction.
    cpu_offload_lru: VecDeque<u64>,
}

impl BlockManager {
    pub fn new(
        runners: Arc<RwLock<RunnerType>>,
        num_blocks: usize,
        num_cpu_blocks: usize,
        block_size: usize,
        prefix_cache: PrefixCacheConfig,
        mamba_prefix_enabled: bool,
        mamba_snapshot_default_stride_blocks: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_block_ids = VecDeque::with_capacity(num_blocks);

        for i in 0..num_blocks {
            blocks.push(Block {
                id: i,
                ref_count: 0,
            });
            free_block_ids.push_back(i);
        }

        let mut cpu_blocks = Vec::with_capacity(num_cpu_blocks);
        let mut free_cpu_block_ids = VecDeque::with_capacity(num_cpu_blocks);
        for i in 0..num_cpu_blocks {
            cpu_blocks.push(Block {
                id: i,
                ref_count: 0,
            });
            free_cpu_block_ids.push_back(i);
        }

        let prefix_cache = if prefix_cache.enabled && prefix_cache.max_cached_blocks > 0 {
            Some(PrefixCache::new(block_size, prefix_cache))
        } else {
            None
        };
        let mamba_snapshot_block_stride_blocks =
            mamba_snapshot_block_stride_blocks(mamba_snapshot_default_stride_blocks);
        if mamba_prefix_enabled {
            crate::log_info!(
                "Hybrid mamba snapshot capture stride: {} block(s) ({} tokens), default follows prefill chunk size and can be overridden by {}.",
                mamba_snapshot_block_stride_blocks,
                mamba_snapshot_block_stride_blocks.saturating_mul(block_size),
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV
            );
        }

        Self {
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            cpu_blocks,
            free_cpu_block_ids,
            swapped_map: HashMap::new(),
            block_size,
            runners,
            prefix_cache,
            mamba_prefix_enabled,
            mamba_snapshot_block_stride_blocks,
            mamba_prefix_hashes_by_block: HashMap::new(),
            mamba_prefix_block_by_hash: HashMap::new(),
            valid_mamba_prefix_hashes: HashSet::new(),
            seq_swap_states: HashMap::new(),
            cpu_offload_entries: HashMap::new(),
            cpu_offload_lru: VecDeque::new(),
        }
    }

    fn block_ref_count(&self, block_id: usize) -> usize {
        self.blocks.get(block_id).map_or(0, |b| b.ref_count)
    }

    fn allocate_block(&mut self, block_id: usize) -> &mut Block {
        let block = &mut self.blocks[block_id];
        assert_eq!(block.ref_count, 0);
        block.reset();
        self.used_block_ids.insert(block_id);
        block
    }

    fn allocate_fresh(&mut self, seq: &mut Sequence) -> Result<()> {
        seq.num_cached_tokens = 0;
        seq.mamba_prefix_hash = None;
        for _ in 0..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        if let Some(last_block_id) = seq.block_table.last() {
            self.clear_blocks_guard(vec![*last_block_id], "allocate_fresh/last_block");
        }
        Ok(())
    }

    fn deallocate_block(&mut self, block_id: usize) {
        assert_eq!(self.blocks[block_id].ref_count, 0);
        if self.used_block_ids.remove(&block_id) {
            self.free_block_ids.push_back(block_id);
        }
    }

    /// Allocate a single free block and return its ID, or None if no blocks available.
    pub fn alloc_free_block(&mut self) -> Option<usize> {
        let block_id = self.free_block_ids.pop_front()?;
        self.allocate_block(block_id);
        Some(block_id)
    }

    fn image_prefix_seed(images: &ImageData) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        images.raw.hash(&mut hasher);
        images.shape.hash(&mut hasher);
        images.patches.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute the image seed and the block index at which it should be applied.
    /// Returns (None, None) when:
    ///  - no images are attached, or
    ///  - the model has no `image_token_id` (non-VL model where images don't
    ///    affect the KV cache), or
    ///  - no image placeholder tokens appear in the token sequence.
    ///
    /// When (Some(seed), Some(block)) is returned, blocks *before* `block` can
    /// be matched/inserted with the same hashes as the image-free case, while
    /// the seed is mixed in starting at `block`.
    fn image_seed_and_block(
        images: &ImageData,
        tokens: &[u32],
        block_size: usize,
    ) -> (Option<u64>, Option<usize>) {
        let Some(image_token_id) = images.image_token_id else {
            return (None, None);
        };
        let Some(first_pos) = tokens.iter().position(|&id| id == image_token_id) else {
            return (None, None);
        };
        let seed = Self::image_prefix_seed(images);
        (Some(seed), Some(first_pos / block_size))
    }

    pub fn required_blocks(&mut self, seq: &Sequence) -> usize {
        if self.prefix_cache.is_some() {
            let mut prefix_cache = self.prefix_cache.take().unwrap();
            let (seed, seed_block) = seq
                .images
                .as_ref()
                .map(|img| Self::image_seed_and_block(img, &seq.token_ids, self.block_size))
                .unwrap_or((None, None));
            let prefix_match =
                prefix_cache.match_prefix_with_seed(&seq.token_ids, seed, seed_block);
            let matched_blocks = self.resolve_mamba_matched_blocks(
                &prefix_cache,
                seq.id,
                self.adjusted_matched_blocks(seq.token_ids.len(), prefix_match.matched_blocks),
                prefix_match.last_hash,
            );
            let gpu_reattach = if self.mamba_prefix_enabled {
                0
            } else {
                self.count_gpu_trie_reattachable_blocks(
                    &prefix_cache,
                    &seq.token_ids,
                    seed,
                    seed_block,
                    matched_blocks,
                )
            };
            self.prefix_cache = Some(prefix_cache);
            let attachable = matched_blocks.saturating_add(gpu_reattach);
            seq.num_blocks().saturating_sub(attachable)
        } else {
            seq.num_blocks()
        }
    }

    /// Prefix blocks beyond mamba attach that still live in the GPU trie (no free pool cost).
    fn count_gpu_trie_reattachable_blocks(
        &self,
        prefix_cache: &PrefixCache,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
        mut matched_blocks: usize,
    ) -> usize {
        let full_blocks = tokens.len() / self.block_size;
        let mut gpu_reattach = 0usize;
        while matched_blocks < full_blocks {
            let Some(trie_hash) = prefix_cache.hash_for_blocks_with_seed(
                tokens,
                matched_blocks + 1,
                seed,
                seed_block,
            ) else {
                break;
            };
            if prefix_cache.block_id_for_hash(trie_hash).is_some() {
                matched_blocks += 1;
                gpu_reattach += 1;
                continue;
            }
            break;
        }
        gpu_reattach
    }

    fn tokens_for_full_block<'a>(
        &self,
        tokens: &'a [u32],
        full_blocks: usize,
    ) -> Option<&'a [u32]> {
        if full_blocks == 0 {
            return None;
        }
        let start = (full_blocks - 1).saturating_mul(self.block_size);
        let end = start.checked_add(self.block_size)?;
        tokens.get(start..end)
    }

    fn cpu_offload_entry_matches(
        &self,
        trie_hash: u64,
        tokens: &[u32],
        full_blocks: usize,
    ) -> bool {
        let Some(entry) = self.cpu_offload_entries.get(&trie_hash) else {
            return false;
        };
        let Some(block_tokens) = self.tokens_for_full_block(tokens, full_blocks) else {
            return false;
        };
        entry.content_hash == PrefixCache::content_fingerprint_for_tokens(block_tokens)
    }

    fn mamba_prefix_hash_usable(&mut self, seq_id: usize, trie_hash: u64) -> bool {
        if !self.mamba_prefix_enabled {
            return true;
        }
        if !self.valid_mamba_prefix_hashes.contains(&trie_hash) {
            return false;
        }
        match self.try_has_mamba_prefix_state(trie_hash) {
            Ok(true) => true,
            Ok(false) => {
                self.invalidate_mamba_prefix_hash(trie_hash);
                false
            }
            Err(e) => {
                crate::log_warn!(
                    "Failed to query mamba prefix state for seq {} hash {}: {}",
                    seq_id,
                    trie_hash,
                    e
                );
                false
            }
        }
    }

    fn prefix_hashes_for_blocks(
        &self,
        prefix_cache: &PrefixCache,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
        full_blocks: usize,
    ) -> HashSet<u64> {
        let mut hashes = HashSet::new();
        for i in 1..=full_blocks {
            if let Some(hash) = prefix_cache.hash_for_blocks_with_seed(tokens, i, seed, seed_block)
            {
                hashes.insert(hash);
            }
        }
        hashes
    }

    fn insert_promoted_prefix_path(
        &mut self,
        prefix_cache: &mut PrefixCache,
        seq: &Sequence,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
        full_blocks: usize,
    ) {
        if full_blocks == 0 || seq.block_table.len() < full_blocks {
            return;
        }
        let block_ids: Vec<usize> = seq.block_table[..full_blocks]
            .iter()
            .map(|&b| b as usize)
            .collect();
        let skip_block_ids: HashSet<usize> = block_ids.iter().copied().collect();
        let protected_hashes =
            self.prefix_hashes_for_blocks(prefix_cache, tokens, seed, seed_block, full_blocks);
        let PrefixCacheUpdate {
            inserted,
            evicted: _,
            evicted_detailed,
        } = prefix_cache.insert_prefix_with_seed_skip(
            &tokens[..full_blocks * self.block_size],
            &block_ids,
            seed,
            seed_block,
            &skip_block_ids,
        );
        for block_id in inserted {
            self.increment_block_ref(block_id);
        }
        self.apply_prefix_evictions(&evicted_detailed, &protected_hashes);
    }

    pub fn can_allocate(&mut self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= self.required_blocks(seq)
    }

    pub fn can_allocate_without_prefix(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        assert!(seq.block_table.is_empty());
        if self.prefix_cache.is_some() {
            let mut prefix_cache = self.prefix_cache.take().unwrap();
            let result = self.allocate_with_prefix(seq, &mut prefix_cache);
            self.prefix_cache = Some(prefix_cache);
            result
        } else {
            self.allocate_fresh(seq)
        }
    }

    pub fn allocate_without_prefix(&mut self, seq: &mut Sequence) -> Result<()> {
        assert!(seq.block_table.is_empty());
        self.allocate_fresh(seq)
    }

    pub fn deallocate(&mut self, seq: &Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            self.decrement_block_ref(block_id as usize);
        }
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let mut need_block: usize = 1;
        if seq.len() % self.block_size != 0 {
            need_block += 1;
        }
        self.free_block_ids.len() >= need_block
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<()> {
        let len_mod = seq.len() % self.block_size;
        if len_mod == 1 {
            //approaching next block
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        Ok(())
    }

    pub fn ensure_allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        let mut new_blocks = Vec::new();
        for i in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
            if i > seq.num_blocks() - 5 {
                new_blocks.push(block_id as u32);
            }
        }
        if !new_blocks.is_empty() {
            self.clear_blocks_guard(new_blocks, "ensure_allocate/new_blocks");
        }
        Ok(())
    }

    fn increment_block_ref(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        if block.ref_count == 0 {
            self.free_block_ids.retain(|&id| id != block_id);
            self.used_block_ids.insert(block_id);
        }
        block.ref_count += 1;
    }

    fn decrement_block_ref(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        block.ref_count = block.ref_count.saturating_sub(1);
        if block.ref_count == 0 {
            self.deallocate_block(block_id);
        }
    }

    fn adjusted_matched_blocks(&self, tokens_len: usize, matched_blocks: usize) -> usize {
        let full_blocks = tokens_len / self.block_size;
        if matched_blocks == full_blocks && tokens_len % self.block_size == 0 && matched_blocks > 0
        {
            matched_blocks - 1
        } else {
            matched_blocks
        }
    }

    fn resolve_mamba_matched_blocks(
        &mut self,
        prefix_cache: &PrefixCache,
        seq_id: usize,
        mut matched_blocks: usize,
        last_hash: Option<u64>,
    ) -> usize {
        if !self.mamba_prefix_enabled {
            return matched_blocks;
        }
        if matched_blocks == 0 {
            return 0;
        }
        let Some(hash) = last_hash else {
            return 0;
        };
        let hashes = prefix_cache.hashes_for_match(hash);
        if hashes.len() < matched_blocks {
            return 0;
        }
        while matched_blocks > 0 {
            let candidate_hash = hashes[matched_blocks - 1];
            if !self.valid_mamba_prefix_hashes.contains(&candidate_hash) {
                matched_blocks -= 1;
                continue;
            }
            match self.try_has_mamba_prefix_state(candidate_hash) {
                Ok(true) => break,
                Ok(false) => {
                    self.invalidate_mamba_prefix_hash(candidate_hash);
                    matched_blocks -= 1;
                }
                Err(e) => {
                    crate::log_warn!(
                        "Failed to query mamba prefix state for seq {}: {}",
                        seq_id,
                        e
                    );
                    return 0;
                }
            }
        }
        matched_blocks
    }

    fn allocate_with_prefix(
        &mut self,
        seq: &mut Sequence,
        prefix_cache: &mut PrefixCache,
    ) -> Result<()> {
        let tokens = &seq.token_ids;
        let token_ids = seq.token_ids.clone();
        let mut matched_blocks = 0usize;
        let mut raw_matched_blocks = 0usize;
        let mut last_hash = None;

        if prefix_cache.enabled() {
            let (seed, seed_block) = seq
                .images
                .as_ref()
                .map(|img| Self::image_seed_and_block(img, tokens, self.block_size))
                .unwrap_or((None, None));
            let prefix_match = prefix_cache.match_prefix_with_seed(tokens, seed, seed_block);
            last_hash = prefix_match.last_hash;
            raw_matched_blocks =
                self.adjusted_matched_blocks(tokens.len(), prefix_match.matched_blocks);
            matched_blocks = self.resolve_mamba_matched_blocks(
                prefix_cache,
                seq.id,
                raw_matched_blocks,
                last_hash,
            );
        }

        seq.mamba_prefix_hash = None;
        if self.mamba_prefix_enabled {
            if raw_matched_blocks > 0 && matched_blocks == 0 {
                crate::log_info!(
                    "Prefix cache mamba-state miss seq {} (raw {} blocks matched, but no compatible mamba snapshot)",
                    seq.id,
                    raw_matched_blocks
                );
            } else if raw_matched_blocks > matched_blocks {
                crate::log_info!(
                    "Prefix cache mamba-compatible partial hit seq {} (raw {} blocks, mamba {} blocks)",
                    seq.id,
                    raw_matched_blocks,
                    matched_blocks
                );
            }
            if matched_blocks > 0 {
                let mut matched_hash = None;
                if let Some(hash) = last_hash {
                    let hashes = prefix_cache.hashes_for_match(hash);
                    if hashes.len() >= matched_blocks {
                        matched_hash = Some(hashes[matched_blocks - 1]);
                    } else {
                        matched_blocks = 0;
                    }
                } else {
                    matched_blocks = 0;
                }
                seq.mamba_prefix_hash = matched_hash;
            }
        }

        let cached_tokens = matched_blocks * self.block_size;
        seq.mamba_prefix_warmup_tokens = None;
        if self.mamba_prefix_enabled && matched_blocks == 0 && raw_matched_blocks > 0 {
            let raw_cached_tokens = raw_matched_blocks * self.block_size;
            if raw_cached_tokens > cached_tokens && raw_cached_tokens < tokens.len() {
                seq.mamba_prefix_warmup_tokens = Some(raw_cached_tokens);
                crate::log_info!(
                    "Seq {}: scheduling mamba prefix warmup snapshot at {} cached tokens (raw {} blocks, no compatible mamba snapshot)",
                    seq.id,
                    raw_cached_tokens,
                    raw_matched_blocks
                );
            }
        }
        if matched_blocks > 0 {
            if let Some(hash) = last_hash {
                let mut cached_blocks = prefix_cache.blocks_for_match(hash);
                cached_blocks.truncate(matched_blocks);
                for block_id in cached_blocks {
                    self.increment_block_ref(block_id);
                    seq.block_table.push(block_id as u32);
                }
            }
        } else if prefix_cache.enabled()
            && tokens.len() >= self.block_size
            && raw_matched_blocks == 0
        {
            crate::log_info!(
                "Prefix cache miss seq {} ({} tokens, {} trie blocks cached globally)",
                seq.id,
                tokens.len(),
                prefix_cache.cached_blocks()
            );
        }

        let mamba_matched_blocks = matched_blocks;
        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, &token_ids, self.block_size))
            .unwrap_or((None, None));
        matched_blocks = self.extend_prefix_from_offload(
            prefix_cache,
            seq,
            &token_ids,
            seed,
            seed_block,
            matched_blocks,
        );

        let promoted_blocks = matched_blocks.saturating_sub(mamba_matched_blocks);
        let cached_tokens = matched_blocks * self.block_size;
        seq.num_cached_tokens = cached_tokens;
        if matched_blocks > 0 {
            crate::log_info!(
                "Prefix cache hit seq {} ({} cached tokens, {} blocks; raw {}, mamba {}, gpu_extend {})",
                seq.id,
                cached_tokens,
                matched_blocks,
                raw_matched_blocks,
                mamba_matched_blocks,
                promoted_blocks
            );
        }

        let mut new_blocks = Vec::new();
        for _ in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
            new_blocks.push(block_id as u32);
        }
        if !new_blocks.is_empty() {
            self.clear_blocks_guard(new_blocks, "allocate_with_prefix/new_blocks");
        }

        Ok(())
    }

    /// Extend prefix match by re-attaching GPU trie blocks and promoting CPU offloads.
    fn extend_prefix_from_offload(
        &mut self,
        prefix_cache: &mut PrefixCache,
        seq: &mut Sequence,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
        mut matched_blocks: usize,
    ) -> usize {
        let full_blocks = tokens.len() / self.block_size;
        let mut gpu_reattached = 0usize;
        let mut total_promoted = 0usize;
        while matched_blocks < full_blocks {
            let Some(trie_hash) = prefix_cache.hash_for_blocks_with_seed(
                tokens,
                matched_blocks + 1,
                seed,
                seed_block,
            ) else {
                break;
            };
            // Blocks still in the GPU trie (not offloaded) — re-attach without using CPU.
            if let Some(block_id) = prefix_cache.block_id_for_hash(trie_hash) {
                if !self.mamba_prefix_hash_usable(seq.id, trie_hash) {
                    break;
                }
                if let Some(entry) = self.take_cpu_offload_entry(trie_hash) {
                    self.free_cpu_block_ids.push_back(entry.cpu_block_id);
                }
                self.increment_block_ref(block_id);
                seq.block_table.push(block_id as u32);
                matched_blocks += 1;
                gpu_reattached += 1;
                if self.mamba_prefix_enabled {
                    seq.mamba_prefix_hash = Some(trie_hash);
                }
                continue;
            }
            if !self.cpu_offload_entry_matches(trie_hash, tokens, matched_blocks + 1)
                || !self.mamba_prefix_hash_usable(seq.id, trie_hash)
            {
                break;
            }
            // Batch-promote consecutive CPU-offloaded blocks.
            let mut promotions = Vec::new();
            let mut probe = matched_blocks;
            while probe < full_blocks {
                let Some(hash) =
                    prefix_cache.hash_for_blocks_with_seed(tokens, probe + 1, seed, seed_block)
                else {
                    break;
                };
                if prefix_cache.block_id_for_hash(hash).is_some() {
                    break;
                }
                if !self.cpu_offload_entry_matches(hash, tokens, probe + 1)
                    || !self.mamba_prefix_hash_usable(seq.id, hash)
                {
                    break;
                }
                let Some(gpu_block_id) = self.free_block_ids.pop_front() else {
                    break;
                };
                self.allocate_block(gpu_block_id);
                promotions.push((hash, gpu_block_id));
                probe += 1;
            }
            if promotions.is_empty() {
                break;
            }
            let gpu_blocks: Vec<u32> = promotions.iter().map(|(_, g)| *g as u32).collect();
            match self.promote_offloaded_prefix_batch(&promotions) {
                Ok(promoted) if promoted > 0 => {
                    for &(trie_hash, gpu_block_id) in promotions.iter().take(promoted) {
                        self.relocate_mamba_prefix_after_promote(trie_hash, gpu_block_id);
                    }
                    seq.block_table
                        .extend(gpu_blocks.iter().take(promoted).copied());
                    matched_blocks += promoted;
                    total_promoted += promoted;
                    if self.mamba_prefix_enabled {
                        if let Some(&(trie_hash, _)) = promotions.iter().take(promoted).last() {
                            seq.mamba_prefix_hash = Some(trie_hash);
                        }
                    }
                    self.insert_promoted_prefix_path(
                        prefix_cache,
                        seq,
                        tokens,
                        seed,
                        seed_block,
                        matched_blocks,
                    );
                    if promoted < promotions.len() {
                        for &gpu_block_id in gpu_blocks.iter().skip(promoted) {
                            self.decrement_block_ref(gpu_block_id as usize);
                        }
                        break;
                    }
                }
                _ => {
                    for (_, gpu_block_id) in promotions {
                        self.decrement_block_ref(gpu_block_id);
                    }
                    break;
                }
            }
        }
        if gpu_reattached > 0 || total_promoted > 0 {
            let mut detail = format!(
                "Extended prefix seq {}: {} GPU trie + {} CPU promoted ({} blocks cached)",
                seq.id, gpu_reattached, total_promoted, matched_blocks
            );
            if total_promoted == 0 && gpu_reattached > 0 {
                detail.push_str("; CPU tier unused — matched blocks still in GPU trie");
            }
            if matched_blocks < full_blocks {
                detail.push_str(&format!(
                    "; chain ends at block {}/{}",
                    matched_blocks, full_blocks
                ));
            }
            crate::log_info!("{}.", detail);
        }
        matched_blocks
    }

    pub fn capture_mamba_prefix_state(
        &mut self,
        seq: &Sequence,
        processed_tokens: usize,
    ) -> Option<u64> {
        if !self.mamba_prefix_enabled {
            return None;
        }
        let Some(prefix_cache) = self.prefix_cache.as_ref() else {
            return None;
        };
        if !prefix_cache.enabled() {
            return None;
        }
        let processed_tokens = processed_tokens.min(seq.token_ids.len());
        let full_blocks = processed_tokens / self.block_size;
        if full_blocks == 0 {
            return None;
        }
        // Keep prompt/prefill captures dense. During decode, capture only at
        // chunk-size boundaries plus the final response boundary so intermediate
        // decode blocks do not churn useful prompt snapshots.
        let final_decode_snapshot = !seq.output_ids.is_empty()
            && matches!(
                seq.status,
                SequenceStatus::Finished | SequenceStatus::Cached | SequenceStatus::FinishSwapped
            );
        if !seq.output_ids.is_empty()
            && !final_decode_snapshot
            && self.mamba_snapshot_block_stride_blocks > 1
            && full_blocks % self.mamba_snapshot_block_stride_blocks != 0
        {
            return None;
        }
        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, &seq.token_ids, self.block_size))
            .unwrap_or((None, None));
        let Some(hash) =
            prefix_cache.hash_for_blocks_with_seed(&seq.token_ids, full_blocks, seed, seed_block)
        else {
            return None;
        };
        let preserve_snapshot = seq.output_ids.is_empty();
        match self.try_capture_mamba_prefix_state(seq.id, hash, preserve_snapshot) {
            Ok(true) => {
                if let Some(&block_id_u32) = seq.block_table.get(full_blocks.saturating_sub(1)) {
                    let block_id = block_id_u32 as usize;
                    if let Some(old_block_id) =
                        self.mamba_prefix_block_by_hash.insert(hash, block_id)
                    {
                        if old_block_id != block_id {
                            if let Some(hashes) =
                                self.mamba_prefix_hashes_by_block.get_mut(&old_block_id)
                            {
                                hashes.remove(&hash);
                                if hashes.is_empty() {
                                    self.mamba_prefix_hashes_by_block.remove(&old_block_id);
                                }
                            }
                        }
                    }
                    self.valid_mamba_prefix_hashes.insert(hash);
                    self.mamba_prefix_hashes_by_block
                        .entry(block_id)
                        .or_default()
                        .insert(hash);
                }
                Some(hash)
            }
            Ok(false) => {
                if processed_tokens == seq.token_ids.len() {
                    crate::log_info!(
                        "Seq {}: mamba prefix snapshot capture returned false at {} tokens (hash {}).",
                        seq.id,
                        processed_tokens,
                        hash
                    );
                }
                None
            }
            Err(e) => {
                crate::log_warn!(
                    "Failed to capture mamba prefix state for seq {} hash {}: {}",
                    seq.id,
                    hash,
                    e
                );
                None
            }
        }
    }

    fn invalidate_mamba_prefix_hash(&mut self, hash: u64) {
        self.valid_mamba_prefix_hashes.remove(&hash);
        if let Some(block_id) = self.mamba_prefix_block_by_hash.remove(&hash) {
            if let Some(hashes) = self.mamba_prefix_hashes_by_block.get_mut(&block_id) {
                hashes.remove(&hash);
                if hashes.is_empty() {
                    self.mamba_prefix_hashes_by_block.remove(&block_id);
                }
            }
        }
        if let Err(e) = self.try_remove_mamba_prefix_state(hash) {
            crate::log_warn!(
                "Failed to remove invalidated mamba prefix snapshot hash {}: {}",
                hash,
                e
            );
        }
    }

    fn handle_mamba_prefix_evicted_blocks(&mut self, evicted_block_ids: &[usize]) {
        if !self.mamba_prefix_enabled || evicted_block_ids.is_empty() {
            return;
        }

        for &block_id in evicted_block_ids {
            if let Some(hashes) = self.mamba_prefix_hashes_by_block.remove(&block_id) {
                for hash in hashes {
                    self.valid_mamba_prefix_hashes.remove(&hash);
                    self.mamba_prefix_block_by_hash.remove(&hash);
                    if let Err(e) = self.try_remove_mamba_prefix_state(hash) {
                        crate::log_warn!(
                            "Failed to remove mamba prefix snapshot hash {} for evicted block {}: {}",
                            hash,
                            block_id,
                            e
                        );
                    }
                }
            }
        }
    }

    /// Keep mamba snapshot metadata aligned when an offloaded prefix block is promoted to a new GPU id.
    fn relocate_mamba_prefix_after_promote(&mut self, trie_hash: u64, new_block_id: usize) {
        if !self.mamba_prefix_enabled || !self.valid_mamba_prefix_hashes.contains(&trie_hash) {
            return;
        }
        if let Some(old_block_id) = self
            .mamba_prefix_block_by_hash
            .insert(trie_hash, new_block_id)
        {
            if old_block_id != new_block_id {
                if let Some(hashes) = self.mamba_prefix_hashes_by_block.get_mut(&old_block_id) {
                    hashes.remove(&trie_hash);
                    if hashes.is_empty() {
                        self.mamba_prefix_hashes_by_block.remove(&old_block_id);
                    }
                }
            }
        }
        self.mamba_prefix_hashes_by_block
            .entry(new_block_id)
            .or_default()
            .insert(trie_hash);
    }

    /// Evict prefix blocks from GPU: bulk offload where possible, invalidate mamba only on true drops.
    fn apply_prefix_evictions(
        &mut self,
        evicted: &[(usize, u64, u64)],
        protected_trie_hashes: &HashSet<u64>,
    ) -> usize {
        if evicted.is_empty() {
            return 0;
        }
        let offload_candidates: Vec<(usize, u64, u64)> = evicted
            .iter()
            .filter(|(block_id, _, _)| self.blocks[*block_id].ref_count == 1)
            .copied()
            .collect();
        let offloaded_gpu_ids =
            self.offload_prefix_blocks_batch(&offload_candidates, protected_trie_hashes);
        let offloaded_set: HashSet<usize> = offloaded_gpu_ids.into_iter().collect();

        for &(block_id, _, _) in evicted {
            self.decrement_block_ref(block_id);
        }

        let mamba_drop: Vec<usize> = evicted
            .iter()
            .map(|(block_id, _, _)| *block_id)
            .filter(|block_id| !offloaded_set.contains(block_id))
            .collect();
        self.handle_mamba_prefix_evicted_blocks(&mamba_drop);

        let offloaded = offloaded_set.len();
        if offloaded > 0 {
            crate::log_info!(
                "Offloaded {} of {} evicted prefix cache block(s) to CPU in one transfer.",
                offloaded,
                evicted.len()
            );
        } else if !offload_candidates.is_empty() {
            crate::log_warn!(
                "Evicted {} prefix cache block(s) without CPU offload (no free CPU blocks).",
                evicted.len()
            );
        }
        offloaded
    }

    pub fn cache_sequence(&mut self, seq: &Sequence) {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return;
        };
        if !prefix_cache.enabled() {
            return;
        }
        if matches!(
            seq.status,
            SequenceStatus::Swapped | SequenceStatus::FinishSwapped
        ) {
            return;
        }

        let tokens = &seq.token_ids;
        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return;
        }
        if seq.block_table.len() < full_blocks {
            return;
        }

        let blocks: Vec<usize> = seq
            .block_table
            .iter()
            .take(full_blocks)
            .map(|&id| id as usize)
            .collect();

        crate::log_info!(
            "Prefix cache insert seq {} ({} tokens, {} blocks)",
            seq.id,
            tokens.len(),
            full_blocks
        );

        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, tokens, self.block_size))
            .unwrap_or((None, None));
        let PrefixCacheUpdate {
            inserted,
            evicted: _,
            evicted_detailed,
        } = prefix_cache.insert_prefix_with_seed(tokens, &blocks, seed, seed_block);
        for block_id in inserted {
            self.increment_block_ref(block_id);
        }
        self.apply_prefix_evictions(&evicted_detailed, &HashSet::new());
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache
            .as_ref()
            .map_or(false, |cache| cache.enabled())
    }

    pub fn prefix_cache_blocks(&self) -> usize {
        self.prefix_cache
            .as_ref()
            .map_or(0, |cache| cache.cached_blocks())
    }

    /// Returns how many tokens of `seq` are already cached in the prefix cache.
    /// Used to decide whether to do local prefill vs transfer to PD server.
    pub fn get_prefix_cache_match_tokens(&mut self, seq: &Sequence) -> usize {
        if self.prefix_cache.is_none() {
            return 0;
        }
        let mut prefix_cache = self.prefix_cache.take().unwrap();
        if !prefix_cache.enabled() {
            self.prefix_cache = Some(prefix_cache);
            return 0;
        }
        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, &seq.token_ids, self.block_size))
            .unwrap_or((None, None));
        let prefix_match = prefix_cache.match_prefix_with_seed(&seq.token_ids, seed, seed_block);
        let matched_blocks = self.resolve_mamba_matched_blocks(
            &prefix_cache,
            seq.id,
            self.adjusted_matched_blocks(seq.token_ids.len(), prefix_match.matched_blocks),
            prefix_match.last_hash,
        );
        self.prefix_cache = Some(prefix_cache);
        matched_blocks * self.block_size
    }

    pub fn clear_prefix_cache(&mut self) {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return;
        };
        let evicted = prefix_cache.clear();
        let evicted_blocks = evicted.clone();
        for block_id in evicted {
            self.decrement_block_ref(block_id);
        }
        self.handle_mamba_prefix_evicted_blocks(&evicted_blocks);
    }

    /// GPU trie block ids on the longest prefix match for `seq` (admission eviction shield).
    pub fn prefix_protect_blocks_for_seq(&mut self, seq: &Sequence) -> HashSet<usize> {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return HashSet::new();
        };
        if !prefix_cache.enabled() {
            return HashSet::new();
        }
        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, &seq.token_ids, self.block_size))
            .unwrap_or((None, None));
        let prefix_match = prefix_cache.match_prefix_with_seed(&seq.token_ids, seed, seed_block);
        let Some(hash) = prefix_match.last_hash else {
            return HashSet::new();
        };
        prefix_cache.blocks_for_match(hash).into_iter().collect()
    }

    /// Trie hashes on the incoming request's full token prefix (CPU offload LRU shield).
    pub fn prefix_protect_trie_hashes_for_seq(&mut self, seq: &Sequence) -> HashSet<u64> {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return HashSet::new();
        };
        if !prefix_cache.enabled() {
            return HashSet::new();
        }
        let full_blocks = seq.token_ids.len() / self.block_size;
        if full_blocks == 0 {
            return HashSet::new();
        }
        let (seed, seed_block) = seq
            .images
            .as_ref()
            .map(|img| Self::image_seed_and_block(img, &seq.token_ids, self.block_size))
            .unwrap_or((None, None));
        let mut hashes = HashSet::new();
        for i in 1..=full_blocks {
            if let Some(hash) =
                prefix_cache.hash_for_blocks_with_seed(&seq.token_ids, i, seed, seed_block)
            {
                hashes.insert(hash);
            }
        }
        hashes
    }

    fn remove_cpu_offload_lru_hash(&mut self, trie_hash: u64) {
        if let Some(pos) = self.cpu_offload_lru.iter().position(|&h| h == trie_hash) {
            self.cpu_offload_lru.remove(pos);
        }
    }

    fn insert_cpu_offload_entry(&mut self, entry: CpuOffloadEntry) {
        let trie_hash = entry.trie_hash;
        self.cpu_offload_entries.insert(trie_hash, entry);
        self.cpu_offload_lru.push_back(trie_hash);
    }

    fn take_cpu_offload_entry(&mut self, trie_hash: u64) -> Option<CpuOffloadEntry> {
        self.remove_cpu_offload_lru_hash(trie_hash);
        self.cpu_offload_entries.remove(&trie_hash)
    }

    fn drop_cpu_offload_entry(&mut self, trie_hash: u64) -> Option<CpuOffloadEntry> {
        let entry = self.take_cpu_offload_entry(trie_hash)?;
        self.invalidate_mamba_prefix_hash(trie_hash);
        Some(entry)
    }

    /// Drop oldest CPU offload copies to free slots (never evicts `skip_trie_hashes`).
    fn evict_cpu_offload_lru(&mut self, count: usize, skip_trie_hashes: &HashSet<u64>) -> usize {
        let mut evicted = 0usize;
        let mut protected_pass = 0usize;
        while evicted < count {
            let Some(trie_hash) = self.cpu_offload_lru.pop_front() else {
                break;
            };
            if !self.cpu_offload_entries.contains_key(&trie_hash) {
                continue;
            }
            if skip_trie_hashes.contains(&trie_hash) {
                self.cpu_offload_lru.push_back(trie_hash);
                protected_pass += 1;
                if protected_pass >= self.cpu_offload_lru.len() {
                    break;
                }
                continue;
            }
            protected_pass = 0;
            if let Some(entry) = self.drop_cpu_offload_entry(trie_hash) {
                self.free_cpu_block_ids.push_back(entry.cpu_block_id);
                evicted += 1;
            }
        }
        evicted
    }

    fn ensure_cpu_offload_capacity(
        &mut self,
        blocks_needed: usize,
        skip_trie_hashes: &HashSet<u64>,
    ) -> usize {
        const MIN_CPU_BLOCKS_FOR_PREEMPT: usize = 8;
        let required_free = blocks_needed.saturating_add(MIN_CPU_BLOCKS_FOR_PREEMPT);
        if self.free_cpu_block_ids.len() >= required_free {
            return 0;
        }
        let deficit = required_free - self.free_cpu_block_ids.len();
        let evicted = self.evict_cpu_offload_lru(deficit, skip_trie_hashes);
        if evicted > 0 {
            crate::log_info!(
                "Evicted {} CPU offload block(s) via LRU to make room for {} new offload(s).",
                evicted,
                blocks_needed
            );
        }
        evicted
    }

    pub fn evict_prefix_cache_blocks(&mut self, num_blocks: usize) -> usize {
        self.evict_prefix_cache_blocks_skip(num_blocks, &HashSet::new(), &HashSet::new())
    }

    pub fn evict_prefix_cache_blocks_skip(
        &mut self,
        num_blocks: usize,
        skip_block_ids: &HashSet<usize>,
        protected_trie_hashes: &HashSet<u64>,
    ) -> usize {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return 0;
        };
        if num_blocks == 0 {
            return 0;
        }
        let evicted = prefix_cache.evict_blocks_detailed_skip(num_blocks, skip_block_ids);
        let evicted_count = evicted.len();
        if evicted_count == 0 {
            return 0;
        }
        self.apply_prefix_evictions(&evicted, protected_trie_hashes);
        evicted_count
    }

    /// Bulk GPU→CPU copy for evicted prefix-cache blocks (single runner IPC).
    /// Returns GPU block ids that were successfully offloaded.
    fn offload_prefix_blocks_batch(
        &mut self,
        blocks: &[(usize, u64, u64)],
        protected_trie_hashes: &HashSet<u64>,
    ) -> Vec<usize> {
        if blocks.is_empty() {
            return Vec::new();
        }
        let mut pending = Vec::new();
        for &(gpu_id, trie_hash, content_hash) in blocks {
            if self.cpu_offload_entries.contains_key(&trie_hash) {
                continue;
            }
            pending.push((gpu_id, trie_hash, content_hash));
        }
        // Reserve headroom for suffix preemption swaps (shares the same CPU pool).
        const MIN_CPU_BLOCKS_FOR_PREEMPT: usize = 8;
        if !pending.is_empty() {
            self.ensure_cpu_offload_capacity(pending.len(), protected_trie_hashes);
        }
        let max_offload = self
            .free_cpu_block_ids
            .len()
            .saturating_sub(MIN_CPU_BLOCKS_FOR_PREEMPT);
        let batch_len = pending.len().min(max_offload);
        if batch_len == 0 {
            return Vec::new();
        }
        pending.truncate(batch_len);

        let mut mapping = HashMap::with_capacity(batch_len);
        let mut staged = Vec::with_capacity(batch_len);
        for (gpu_id, trie_hash, content_hash) in pending {
            let cpu_bid = match self.free_cpu_block_ids.pop_front() {
                Some(id) => id,
                None => break,
            };
            mapping.insert(gpu_id, cpu_bid);
            staged.push((gpu_id, trie_hash, content_hash, cpu_bid));
        }
        if mapping.is_empty() {
            return Vec::new();
        }
        if self.try_swap_kvcache(mapping, false).is_err() {
            for (_, _, _, cpu_bid) in staged {
                self.free_cpu_block_ids.push_back(cpu_bid);
            }
            return Vec::new();
        }
        let mut offloaded_gpu_ids = Vec::with_capacity(staged.len());
        for (gpu_id, trie_hash, content_hash, cpu_bid) in staged {
            offloaded_gpu_ids.push(gpu_id);
            self.insert_cpu_offload_entry(CpuOffloadEntry {
                content_hash,
                trie_hash,
                cpu_block_id: cpu_bid,
            });
        }
        offloaded_gpu_ids
    }

    /// Promote offloaded prefix blocks back to GPU (single runner IPC).
    fn promote_offloaded_prefix_batch(&mut self, promotions: &[(u64, usize)]) -> Result<usize> {
        if promotions.is_empty() {
            return Ok(0);
        }
        let mut mapping = HashMap::with_capacity(promotions.len());
        let mut staged: Vec<CpuOffloadEntry> = Vec::with_capacity(promotions.len());
        for &(trie_hash, gpu_block_id) in promotions {
            let Some(entry) = self.take_cpu_offload_entry(trie_hash) else {
                continue;
            };
            mapping.insert(entry.cpu_block_id, gpu_block_id);
            staged.push(entry);
        }
        if mapping.is_empty() {
            return Ok(0);
        }
        let promoted = staged.len();
        if let Err(e) = self.try_swap_kvcache(mapping, true) {
            for entry in staged {
                self.insert_cpu_offload_entry(entry);
            }
            return Err(e);
        }
        for entry in staged {
            self.free_cpu_block_ids.push_back(entry.cpu_block_id);
        }
        Ok(promoted)
    }

    /// Promote one offloaded prefix block back to a freshly allocated GPU block.
    pub fn try_promote_offloaded_prefix(
        &mut self,
        trie_hash: u64,
        gpu_block_id: usize,
    ) -> Result<bool> {
        let promoted = self.promote_offloaded_prefix_batch(&[(trie_hash, gpu_block_id)])?;
        Ok(promoted > 0)
    }

    pub fn offloaded_prefix_blocks(&self) -> usize {
        self.cpu_offload_entries.len()
    }

    /// Whether suffix-only preemption is possible for this sequence.
    pub fn can_preempt_suffix(&self, seq: &Sequence) -> bool {
        #[cfg(feature = "cuda")]
        {
            let (_prefix_blocks, suffix_blocks) =
                split_prefix_suffix(seq, |id| self.block_ref_count(id));
            if suffix_blocks == 0 {
                return false;
            }
            self.free_cpu_block_ids.len() >= suffix_blocks
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = seq;
            false
        }
    }

    /// Detach shared prefix blocks and swap suffix KV to CPU (prefix-cache mode).
    pub fn preempt_sequence_suffix(&mut self, seq: &mut Sequence) -> Result<()> {
        if !self.prefix_cache_enabled() {
            return self.swap_out(seq);
        }
        let (prefix_blocks, suffix_blocks) =
            split_prefix_suffix(seq, |id| self.block_ref_count(id));
        if suffix_blocks == 0 {
            candle_core::bail!(
                "Seq {} has no exclusively-owned suffix blocks to preempt",
                seq.id
            );
        }
        if self.free_cpu_block_ids.len() < suffix_blocks {
            candle_core::bail!("Not enough CPU blocks for suffix preempt on seq {}", seq.id);
        }

        let mut cpu_ids = Vec::with_capacity(suffix_blocks);
        for _ in 0..suffix_blocks {
            let cpu_bid = self
                .free_cpu_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free CPU swap blocks"))?;
            cpu_ids.push(cpu_bid);
        }

        let pairs = suffix_swap_pairs(seq, prefix_blocks, &cpu_ids);
        let mapping: HashMap<usize, usize> = pairs.iter().copied().collect();
        self.try_swap_kvcache(mapping, false)?;

        // Detach prefix blocks (decrement refs; GPU KV stays in trie).
        for i in (0..prefix_blocks).rev() {
            let block_id = seq.block_table[i] as usize;
            self.decrement_block_ref(block_id);
            seq.block_table.remove(i);
        }

        // Free suffix GPU slots after CPU copy.
        for i in (0..suffix_blocks).rev() {
            let block_id = seq.block_table[i] as usize;
            self.decrement_block_ref(block_id);
            seq.block_table.remove(i);
        }

        seq.swapped_time = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize,
        );

        self.seq_swap_states.insert(
            seq.id,
            SeqSwapState {
                detached_prefix_blocks: prefix_blocks,
                preempted_cpu_blocks: cpu_ids,
                suffix_block_count: suffix_blocks,
            },
        );
        // Legacy map for has_cpu_swap checks during transition.
        if let Some(state) = self.seq_swap_states.get(&seq.id) {
            self.swapped_map
                .insert(seq.id, state.preempted_cpu_blocks.clone());
        }

        crate::log_warn!(
            "Preempted seq {} suffix ({} blocks); detached {} prefix block(s).",
            seq.id,
            suffix_blocks,
            prefix_blocks
        );
        Ok(())
    }

    /// Restore suffix KV from CPU after partial preempt (prefix re-attached separately).
    pub fn resume_sequence_suffix(&mut self, seq: &mut Sequence) -> Result<()> {
        if !self.prefix_cache_enabled() {
            return self.swap_in(seq);
        }
        let Some(state) = self.seq_swap_states.remove(&seq.id) else {
            return self.swap_in(seq);
        };
        self.swapped_map.remove(&seq.id);

        if state.preempted_cpu_blocks.len()
            > seq
                .block_table
                .len()
                .saturating_sub(state.detached_prefix_blocks)
        {
            self.seq_swap_states.insert(seq.id, state);
            candle_core::bail!(
                "Insufficient GPU suffix blocks allocated for seq {} resume",
                seq.id
            );
        }

        let pairs = suffix_swap_in_pairs(
            seq,
            state.detached_prefix_blocks,
            &state.preempted_cpu_blocks,
        );
        let mapping: HashMap<usize, usize> = pairs.iter().copied().collect();
        self.try_swap_kvcache(mapping, true)?;

        for cpu_bid in state.preempted_cpu_blocks {
            let cpu_block = &mut self.cpu_blocks[cpu_bid];
            cpu_block.ref_count = 0;
            self.free_cpu_block_ids.push_back(cpu_bid);
        }

        crate::log_warn!(
            "Resumed seq {} suffix ({} blocks).",
            seq.id,
            state.suffix_block_count
        );
        Ok(())
    }

    pub fn has_suffix_preempt(&self, seq_id: usize) -> bool {
        self.seq_swap_states.contains_key(&seq_id) || self.swapped_map.contains_key(&seq_id)
    }

    pub fn evict_prefix_cache_until_free(&mut self, min_free_blocks: usize) -> usize {
        self.evict_prefix_cache_until_free_skip(min_free_blocks, &HashSet::new(), &HashSet::new())
    }

    pub fn evict_prefix_cache_until_free_skip(
        &mut self,
        min_free_blocks: usize,
        skip_block_ids: &HashSet<usize>,
        protected_trie_hashes: &HashSet<u64>,
    ) -> usize {
        let mut total_evicted = 0;
        while self.free_block_ids.len() < min_free_blocks {
            let need = min_free_blocks.saturating_sub(self.free_block_ids.len());
            let evicted =
                self.evict_prefix_cache_blocks_skip(need, skip_block_ids, protected_trie_hashes);
            if evicted == 0 {
                break;
            }
            total_evicted += evicted;
        }
        total_evicted
    }

    fn clear_blocks_guard(&mut self, block_ids: Vec<u32>, context: &str) {
        let mut safe = Vec::new();
        for block_id in block_ids {
            let idx = block_id as usize;
            if idx >= self.blocks.len() {
                crate::log_error!(
                    "ClearBlocks guard: invalid block id {} in {}",
                    block_id,
                    context
                );
                continue;
            }
            let ref_count = self.blocks[idx].ref_count;
            if ref_count > 1 {
                crate::log_error!(
                    "ClearBlocks guard: block {} has ref_count {} in {}, skipping",
                    block_id,
                    ref_count,
                    context
                );
                continue;
            }
            safe.push(block_id);
        }
        if safe.is_empty() {
            return;
        }
        let _ = self.try_clear_blocks(safe);
    }

    pub fn get_num_total_blocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }

    pub fn get_cpu_swap_usage(&self) -> f32 {
        let total_cpu_blocks = self.cpu_blocks.len();
        (total_cpu_blocks - self.free_cpu_block_ids.len()) as f32 / total_cpu_blocks as f32
    }

    // def try_transfer_prefill
    def_broadcast_message_to_runners!(
        pub, // visibility
        try_transfer_prefill, // function name to create
        transfer_prefill, // thread-mode method name
        (seq: &Sequence), // arguments
        MessageType::TransferPrefill, // message to send
        (seq.clone()), // message payload (must clone)
        MessageType::TransferPrefillResponse, // response to match
        bool // inner return type
    );

    // def try_receive_prefill
    def_broadcast_message_to_runners!(
        pub, // visibility
        try_receive_prefill,
        try_receive_prefill,
        (available_tokens: usize),
        MessageType::ReceivePrefill,
        (available_tokens),
        MessageType::ReceivePrefillResponse,
        (bool, Option<Sequence>)
    );

    // def try_check_prefill_status
    def_broadcast_message_to_runners!(
        pub,
        try_check_prefill_status,
        check_prefill_status,
        (seq_id: usize),
        MessageType::CheckPrefillStatus,
        (seq_id),
        MessageType::CheckPrefillStatusResponse,
        bool
    );

    // def try_swap_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_swap_kvcache,
        swap_kvcache,
        (mappings: HashMap<usize, usize>, swap_in: bool),
        MessageType::KVCacheSwap,
        ((mappings.clone(), swap_in)),
        MessageType::KVCacheSwapResponse,
        bool
    );

    // def try_send_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_send_kvcache,
        send_kvcache,
        (seq: &Sequence, token: u32),
        MessageType::KvCacheSend,
        ((seq.clone(), token)),
        MessageType::KvCacheSendResponse,
        bool
    );

    // def try_receive_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_receive_kvcache,
        receive_kvcache,
        (seq: &Sequence),
        MessageType::KvCacheReceive,
        (seq.clone()),
        MessageType::KvCacheReceiveResponse,
        (bool, u32, usize, usize)
    );

    // After the client received prefill kvcache, it can call PD server to release
    // corresponding kvcache backup
    // def try_release_remote_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_release_remote_kvcache,
        release_remote_kvcache,
        (seq_id: usize),
        MessageType::KvCacheRelease,
        (seq_id),
        MessageType::KvCacheReleaseResponse,
        bool
    );

    // def try_check_kvcache_release
    def_broadcast_message_to_runners!(
        pub,
        try_check_kvcache_release,
        check_kvcache_release,
        (seq_id: usize),
        MessageType::CheckKvCacheRelease,
        (seq_id),
        MessageType::CheckKvCacheReleaseResponse,
        bool
    );

    // Capture and query mamba prefix states for hybrid models.
    def_broadcast_message_to_runners!(
        pub,
        try_capture_mamba_prefix_state,
        capture_mamba_prefix_state,
        (seq_id: usize, hash: u64, preserve: bool),
        MessageType::CaptureMambaPrefixState,
        ((seq_id, hash, preserve)),
        MessageType::CaptureMambaPrefixStateResponse,
        bool
    );
    def_broadcast_message_to_runners!(
        pub,
        try_has_mamba_prefix_state,
        has_mamba_prefix_state,
        (hash: u64),
        MessageType::HasMambaPrefixState,
        (hash),
        MessageType::HasMambaPrefixStateResponse,
        bool
    );
    def_broadcast_message_to_runners!(
        pub,
        try_remove_mamba_prefix_state,
        remove_mamba_prefix_state,
        (hash: u64),
        MessageType::RemoveMambaPrefixState,
        (hash),
        MessageType::RemoveMambaPrefixStateResponse,
        bool
    );
    // Zero specific blocks
    def_broadcast_message_to_runners!(
        pub,
        try_clear_blocks,
        clear_blocks,
        (block_ids: Vec<u32>),
        MessageType::ClearBlocks,
        (block_ids.clone()),
        MessageType::ClearBlocksResponse,
        bool
    );

    /// Can we swap-out `seq` (i.e., move its GPU blocks to CPU swap space)?
    #[allow(unused)]
    pub fn can_swap_out(&self, seq: &Sequence) -> bool {
        #[cfg(feature = "cuda")]
        {
            if self.prefix_cache_enabled() {
                return self.can_preempt_suffix(seq);
            }
            if seq
                .block_table
                .iter()
                .any(|&id| self.blocks[id as usize].ref_count > 1)
            {
                return false;
            }
            let needed = seq.num_blocks();
            self.free_cpu_block_ids.len() > needed
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = seq;
            false
        }
    }

    /// Can we swap-in `seq` (i.e., bring its blocks back from CPU to GPU)?
    #[allow(unused)]
    pub fn can_swap_in(&self, seq: &Sequence) -> bool {
        #[cfg(feature = "cuda")]
        {
            if let Some(state) = self.seq_swap_states.get(&seq.id) {
                return self.free_block_ids.len() > state.suffix_block_count;
            }
            self.free_block_ids.len() > seq.num_blocks()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = seq;
            false
        }
    }

    /// Swap out the GPU blocks of `seq` into CPU swap space.
    /// Caller need to deallocate the gpu blocks used in this seq
    pub fn swap_out(&mut self, seq: &mut Sequence) -> Result<()> {
        if self.prefix_cache_enabled() {
            return self.preempt_sequence_suffix(seq);
        }
        let num_blocks = seq.block_table.len();
        if self.free_cpu_block_ids.len() < num_blocks {
            candle_core::bail!("Not enough CPU swap blocks for seq {}", seq.id);
        }

        crate::log_warn!(
            "Swap out sequence {} ({} blocks) to CPU memory",
            seq.id,
            num_blocks,
        );

        // mapping GPU → CPU
        let mut mapping = std::collections::HashMap::new();
        let mut cpu_ids = Vec::with_capacity(num_blocks);

        for &gpu_bid_u32 in &seq.block_table {
            let gpu_bid = gpu_bid_u32 as usize;
            let cpu_bid = self
                .free_cpu_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free CPU swap blocks"))?;

            mapping.insert(gpu_bid, cpu_bid);
            cpu_ids.push(cpu_bid);
        }

        // Actual data copy GPU → CPU
        self.try_swap_kvcache(mapping.clone(), false)?;
        seq.swapped_time = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize,
        );

        // Update manager bookkeeping
        for (&gpu_bid, &cpu_bid) in &mapping {
            let gpu_block = &mut self.blocks[gpu_bid];
            let cpu_block = &mut self.cpu_blocks[cpu_bid];
            cpu_block.ref_count = gpu_block.ref_count;
        }

        self.swapped_map.insert(seq.id, cpu_ids);

        Ok(())
    }

    /// Need to preallocate new spaces for the seq before calling this function
    pub fn swap_in(&mut self, seq: &mut Sequence) -> Result<()> {
        if self.prefix_cache_enabled() && self.seq_swap_states.contains_key(&seq.id) {
            return self.resume_sequence_suffix(seq);
        }
        let cpu_ids = self
            .swapped_map
            .remove(&seq.id)
            .ok_or_else(|| candle_core::Error::msg("No CPU-swap entry for seq"))?;

        if cpu_ids.len() > seq.block_table.len() {
            // push back and free
            self.swapped_map.insert(seq.id, cpu_ids);
            self.free_cpu_swap_for_seq(seq.id);
            candle_core::bail!("Insufficient GPU blocks to swap in sequence {}", seq.id);
        }

        // mapping CPU → GPU (reverse)
        let mapping: std::collections::HashMap<usize, usize> = cpu_ids
            .iter()
            .enumerate()
            .map(|(i, &cpu_id)| (cpu_id, seq.block_table[i] as usize))
            .collect();

        // Actual data copy CPU → GPU
        self.try_swap_kvcache(mapping.clone(), true)?;

        // Free CPU blocks now that data is back on GPU
        for cpu_bid in cpu_ids {
            let cpu_block = &mut self.cpu_blocks[cpu_bid];
            cpu_block.ref_count = 0;
            self.free_cpu_block_ids.push_back(cpu_bid);
        }

        Ok(())
    }

    pub fn has_cpu_swap(&self, seq_id: usize) -> bool {
        self.swapped_map.contains_key(&seq_id)
    }

    /// Free CPU-side swap blocks for a seq (if any). Useful for aborts.
    pub fn free_cpu_swap_for_seq(&mut self, seq_id: usize) {
        self.seq_swap_states.remove(&seq_id);
        if let Some(cpu_ids) = self.swapped_map.remove(&seq_id) {
            for cpu_bid in cpu_ids {
                let cpu_block = &mut self.cpu_blocks[cpu_bid];
                cpu_block.ref_count = 0;
                self.free_cpu_block_ids.push_back(cpu_bid);
            }
        }
    }
}
