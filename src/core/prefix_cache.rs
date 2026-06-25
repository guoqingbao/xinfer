use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

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
        if !self.enabled() {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        let max_blocks = std::cmp::min(full_blocks, blocks.len());
        if max_blocks == 0 {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
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
        let evicted = if excess > 0 {
            self.evict_blocks(excess)
        } else {
            Vec::new()
        };

        PrefixCacheUpdate { inserted, evicted }
    }

    pub fn evict_blocks(&mut self, mut num_blocks: usize) -> Vec<usize> {
        let mut evicted = Vec::new();
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
            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            evicted.push(entry.block_id);
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

    fn mix_seed(parent_hash: u64, seed: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixCacheConfig};

    #[test]
    fn prefix_cache_matches_full_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let update = cache.insert_prefix(&tokens, &blocks);
        assert!(update.evicted.is_empty());
        assert_eq!(update.inserted.len(), 2);

        let match_info = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(match_info.matched_blocks, 2);

        let matched_blocks = cache.blocks_for_match(match_info.last_hash.unwrap());
        assert_eq!(matched_blocks, blocks);
    }

    #[test]
    fn prefix_cache_evicts_leaf_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 1,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![21, 22];
        let update = cache.insert_prefix(&tokens, &blocks);
        assert_eq!(update.evicted.len(), 1);
        assert_eq!(update.evicted[0], 22);

        let match_info = cache.match_prefix(&tokens);
        assert_eq!(match_info.matched_blocks, 1);
    }
}
