use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// Leave at least one token for prefill when the prompt ends on a block boundary.
pub(crate) fn max_reusable_blocks(tokens_len: usize, block_size: usize) -> usize {
    let full_blocks = tokens_len / block_size;
    if tokens_len % block_size == 0 {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_cached_blocks: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_cached_blocks: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefixMatch {
    pub matched_blocks: usize,
    pub last_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheUpdate {
    pub inserted: Vec<usize>,
    pub evicted: Vec<usize>,
    pub evicted_detailed: Vec<(usize, u64, u64)>,
}

#[derive(Clone)]
struct PrefixEntry {
    parent: Option<u64>,
    block_id: usize,
    children: usize,
    access_id: u64,
    content_hash: u64,
}

pub struct PrefixCache {
    block_size: usize,
    config: PrefixCacheConfig,
    entries: HashMap<u64, PrefixEntry>,
    leaf_set: HashSet<u64>,
    leaf_lru: VecDeque<(u64, u64)>,
    access_counter: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        Self {
            block_size,
            config,
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.max_cached_blocks > 0
    }

    pub fn cached_blocks(&self) -> usize {
        self.entries.len()
    }

    pub fn max_cached_blocks(&self) -> usize {
        self.config.max_cached_blocks
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        self.match_prefix_with_seed(tokens, None, None)
    }

    pub fn match_prefix_with_seed(
        &mut self,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
    ) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = 0u64;
        let mut last_hash = None;
        for (i, block_tokens) in tokens.chunks(self.block_size).take(full_blocks).enumerate() {
            if let Some(s) = seed {
                if seed_block.map_or(false, |sb| i == sb) {
                    parent_hash = Self::mix_seed(parent_hash, s);
                }
            }
            let hash = Self::hash_block(parent_hash, block_tokens);
            let fingerprint = Self::content_fingerprint(block_tokens);
            if let Some(entry) = self.entries.get(&hash) {
                if entry.content_hash != fingerprint {
                    break;
                }
                matched += 1;
                parent_hash = hash;
                last_hash = Some(hash);
                self.touch(hash);
            } else {
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
        }
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<usize> {
        let mut blocks = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => break,
            };
            blocks.push(entry.block_id);
            current = entry.parent;
        }
        blocks.reverse();
        blocks
    }

    pub fn hashes_for_match(&self, last_hash: u64) -> Vec<u64> {
        let mut hashes = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => break,
            };
            hashes.push(hash);
            current = entry.parent;
        }
        hashes.reverse();
        hashes
    }

    pub fn hash_for_blocks_with_seed(
        &self,
        tokens: &[u32],
        full_blocks: usize,
        seed: Option<u64>,
        seed_block: Option<usize>,
    ) -> Option<u64> {
        if full_blocks == 0 {
            return None;
        }
        let mut parent_hash = 0u64;
        let mut last_hash = None;
        for (i, block_tokens) in tokens.chunks(self.block_size).take(full_blocks).enumerate() {
            if let Some(s) = seed {
                if seed_block.map_or(false, |sb| i == sb) {
                    parent_hash = Self::mix_seed(parent_hash, s);
                }
            }
            let hash = Self::hash_block(parent_hash, block_tokens);
            parent_hash = hash;
            last_hash = Some(hash);
        }
        last_hash
    }

    /// Hash each full block once, preserving the same seed boundary as lookup.
    pub fn hashes_for_tokens_with_seed(
        &self,
        tokens: &[u32],
        seed: Option<u64>,
        seed_block: Option<usize>,
    ) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(tokens.len() / self.block_size);
        let mut parent_hash = 0u64;
        for (i, block_tokens) in tokens
            .chunks(self.block_size)
            .take(tokens.len() / self.block_size)
            .enumerate()
        {
            if let Some(s) = seed {
                if seed_block == Some(i) {
                    parent_hash = Self::mix_seed(parent_hash, s);
                }
            }
            parent_hash = Self::hash_block(parent_hash, block_tokens);
            hashes.push(parent_hash);
        }
        hashes
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) -> PrefixCacheUpdate {
        self.insert_prefix_with_seed(tokens, blocks, None, None)
    }

    pub fn insert_prefix_with_seed(
        &mut self,
        tokens: &[u32],
        blocks: &[usize],
        seed: Option<u64>,
        seed_block: Option<usize>,
    ) -> PrefixCacheUpdate {
        self.insert_prefix_with_seed_skip(tokens, blocks, seed, seed_block, &HashSet::new())
    }

    pub fn insert_prefix_with_seed_skip(
        &mut self,
        tokens: &[u32],
        blocks: &[usize],
        seed: Option<u64>,
        seed_block: Option<usize>,
        skip_block_ids: &HashSet<usize>,
    ) -> PrefixCacheUpdate {
        if !self.enabled() {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
                evicted_detailed: Vec::new(),
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        let max_blocks = std::cmp::min(full_blocks, blocks.len());
        if max_blocks == 0 {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
                evicted_detailed: Vec::new(),
            };
        }

        let mut inserted = Vec::new();
        let mut parent_hash: Option<u64> = None;
        for (i, (block_id, block_tokens)) in blocks
            .iter()
            .zip(tokens.chunks(self.block_size))
            .take(max_blocks)
            .enumerate()
        {
            let mut base = parent_hash.unwrap_or(0);
            if let Some(s) = seed {
                if seed_block.map_or(false, |sb| i == sb) {
                    base = Self::mix_seed(base, s);
                }
            }
            let hash = Self::hash_block(base, block_tokens);
            let fingerprint = Self::content_fingerprint(block_tokens);
            if self.entries.contains_key(&hash) {
                let content_match = self
                    .entries
                    .get(&hash)
                    .map_or(false, |e| e.content_hash == fingerprint);
                if !content_match {
                    break;
                }
                let access_id = self.next_access_id();
                self.entries.get_mut(&hash).unwrap().access_id = access_id;
                self.touch_leaf(hash);
            } else {
                if let Some(parent) = parent_hash {
                    if let Some(parent_entry) = self.entries.get_mut(&parent) {
                        if parent_entry.children == 0 {
                            self.leaf_set.remove(&parent);
                        }
                        parent_entry.children += 1;
                    }
                }
                let access_id = self.next_access_id();
                self.entries.insert(
                    hash,
                    PrefixEntry {
                        parent: parent_hash,
                        block_id: *block_id,
                        children: 0,
                        access_id,
                        content_hash: fingerprint,
                    },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access_id));
                inserted.push(*block_id);
            }
            parent_hash = Some(hash);
        }

        let excess = self
            .entries
            .len()
            .saturating_sub(self.config.max_cached_blocks);
        let evicted_detailed = if excess > 0 {
            self.evict_blocks_detailed_skip(excess, skip_block_ids)
        } else {
            Vec::new()
        };
        let evicted: Vec<usize> = evicted_detailed.iter().map(|(id, _, _)| *id).collect();

        PrefixCacheUpdate {
            inserted,
            evicted,
            evicted_detailed,
        }
    }

    pub fn evict_blocks(&mut self, num_blocks: usize) -> Vec<usize> {
        self.evict_blocks_detailed(num_blocks)
            .into_iter()
            .map(|(block_id, _, _)| block_id)
            .collect()
    }

    /// Evict LRU leaf blocks and return `(block_id, trie_hash, content_hash)`.
    pub fn evict_blocks_detailed(&mut self, num_blocks: usize) -> Vec<(usize, u64, u64)> {
        self.evict_blocks_detailed_skip(num_blocks, &HashSet::new())
    }

    /// Evict LRU leaves, skipping block ids that must be kept (e.g. incoming request prefix).
    pub fn evict_blocks_detailed_skip(
        &mut self,
        mut num_blocks: usize,
        skip_block_ids: &HashSet<usize>,
    ) -> Vec<(usize, u64, u64)> {
        let mut evicted = Vec::new();
        let mut protected_seen = HashSet::new();
        while num_blocks > 0 {
            let Some((hash, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let Some(entry) = self.entries.get(&hash) else {
                continue;
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }
            if skip_block_ids.contains(&entry.block_id) {
                if protected_seen.contains(&hash) {
                    break;
                }
                protected_seen.insert(hash);
                self.leaf_lru.push_back((hash, access_id));
                continue;
            }
            protected_seen.clear();
            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            evicted.push((entry.block_id, hash, entry.content_hash));
            num_blocks = num_blocks.saturating_sub(1);
            if let Some(parent) = entry.parent {
                if let Some(parent_entry) = self.entries.get_mut(&parent) {
                    if parent_entry.children > 0 {
                        parent_entry.children -= 1;
                    }
                    if parent_entry.children == 0 {
                        self.leaf_set.insert(parent);
                        self.leaf_lru.push_back((parent, parent_entry.access_id));
                    }
                }
            }
        }
        evicted
    }

    pub fn block_id_for_hash(&self, hash: u64) -> Option<usize> {
        self.entries.get(&hash).map(|e| e.block_id)
    }

    pub fn block_metadata(&self, block_id: usize) -> Option<(u64, u64)> {
        self.entries
            .values()
            .find(|e| e.block_id == block_id)
            .map(|e| (e.content_hash, e.parent.unwrap_or(0)))
    }

    pub fn entry_for_block(&self, block_id: usize) -> Option<(u64, u64)> {
        self.entries
            .iter()
            .find(|(_, e)| e.block_id == block_id)
            .map(|(hash, e)| (*hash, e.content_hash))
    }

    pub fn clear(&mut self) -> Vec<usize> {
        let blocks: Vec<usize> = self.entries.values().map(|entry| entry.block_id).collect();
        self.entries.clear();
        self.leaf_set.clear();
        self.leaf_lru.clear();
        blocks
    }

    fn touch(&mut self, hash: u64) {
        if self.entries.contains_key(&hash) {
            let access_id = self.next_access_id();
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.access_id = access_id;
            }
            self.touch_leaf(hash);
        }
    }

    fn touch_leaf(&mut self, hash: u64) {
        if self.leaf_set.contains(&hash) {
            if let Some(entry) = self.entries.get(&hash) {
                self.leaf_lru.push_back((hash, entry.access_id));
            }
        }
        self.compact_lru_if_needed();
    }

    fn compact_lru_if_needed(&mut self) {
        let threshold = self.entries.len().max(64) * 4;
        if self.leaf_lru.len() <= threshold {
            return;
        }
        self.leaf_lru.retain(|(hash, access_id)| {
            if !self.leaf_set.contains(hash) {
                return false;
            }
            match self.entries.get(hash) {
                Some(entry) => entry.access_id == *access_id,
                None => false,
            }
        });
    }

    fn next_access_id(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    fn content_fingerprint(tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        0xDEADBEEFu64.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    pub fn content_fingerprint_for_tokens(tokens: &[u32]) -> u64 {
        Self::content_fingerprint(tokens)
    }

    fn mix_seed(parent_hash: u64, seed: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{max_reusable_blocks, PrefixCache, PrefixCacheConfig};
    use std::collections::HashSet;

    #[test]
    fn retains_blocks_beyond_old_fraction_when_capacity_is_available() {
        let mut cache = PrefixCache::new(
            2,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 5,
            },
        );
        let update = cache.insert_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 11, 12, 13]);
        assert_eq!(cache.cached_blocks(), 4);
        assert!(update.evicted.is_empty());
    }

    #[test]
    fn protects_incoming_prefix_when_capacity_is_reached() {
        let mut cache = PrefixCache::new(
            2,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 3,
            },
        );
        cache.insert_prefix(&[1, 2, 3, 4], &[10, 11]);
        cache.insert_prefix(&[1, 2, 5, 6], &[10, 12]);
        let update = cache.insert_prefix_with_seed_skip(
            &[1, 2, 7, 8],
            &[10, 13],
            None,
            None,
            &HashSet::from([10, 13]),
        );
        assert_eq!(update.evicted, vec![11]);
        assert_eq!(cache.match_prefix(&[1, 2, 7, 8]).matched_blocks, 2);
    }

    #[test]
    fn explicit_capacity_can_evict_an_oversized_prefix_tail() {
        let mut cache = PrefixCache::new(
            2,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 2,
            },
        );
        let update = cache.insert_prefix(&[1, 2, 3, 4, 5, 6], &[10, 11, 12]);
        assert_eq!(update.evicted, vec![12]);
        assert_eq!(cache.cached_blocks(), 2);
    }

    #[test]
    fn incremental_hashes_match_seeded_lookup() {
        let cache = PrefixCache::new(
            2,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 5,
            },
        );
        let tokens = [1, 2, 3, 4, 5, 6, 7];
        let hashes = cache.hashes_for_tokens_with_seed(&tokens, Some(42), Some(1));
        assert_eq!(hashes.len(), 3);
        for (index, hash) in hashes.into_iter().enumerate() {
            assert_eq!(
                Some(hash),
                cache.hash_for_blocks_with_seed(&tokens, index + 1, Some(42), Some(1))
            );
        }
    }

    #[test]
    fn block_aligned_prompt_keeps_one_block_for_prefill() {
        assert_eq!(max_reusable_blocks(54_016, 64), 843);
        assert_eq!(max_reusable_blocks(54_017, 64), 844);
        assert_eq!(max_reusable_blocks(64, 64), 0);
    }
}
