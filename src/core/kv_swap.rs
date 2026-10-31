//! CPU-resident prefix KV block metadata.

/// CPU copy of a prefix block evicted from the GPU trie.
#[derive(Debug, Clone)]
pub struct CpuOffloadEntry {
    pub content_hash: u64,
    pub trie_hash: u64,
    pub cpu_block_id: usize,
}
