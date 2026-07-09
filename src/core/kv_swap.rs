//! Block-granular KV swap / offload helpers

use super::sequence::Sequence;

/// GPU block lifecycle state (logical; tracked via ref_count + side tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBlockState {
    Free,
    Active,
    SharedActive,
    PrefixCached,
    CpuOffloaded,
    CpuPreempted,
}

/// How a block in a sequence's `block_table` should be treated under swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockClass {
    /// Shared with prefix cache or other sequences — detach refs only.
    AttachedPrefix,
    /// Exclusively owned suffix — may be copied to CPU on preemption.
    SeqOwned,
}

/// Per-sequence suffix swap state (prefix blocks stay on GPU).
#[derive(Debug, Clone)]
pub struct SeqSwapState {
    pub detached_prefix_blocks: usize,
    /// CPU block ids holding suffix KV (in block_table order for suffix only).
    pub preempted_cpu_blocks: Vec<usize>,
    pub suffix_block_count: usize,
}

/// CPU-resident copy of an evicted prefix-cache block.
#[derive(Debug, Clone)]
pub struct CpuOffloadEntry {
    pub content_hash: u64,
    pub trie_hash: u64,
    pub cpu_block_id: usize,
}

/// Split `block_table` into prefix vs suffix ranges for partial preempt.
///
/// `prefix_blocks` is the count of leading blocks that are prefix-attached
/// (from `num_cached_tokens` / block alignment, capped by table length).
pub fn split_prefix_suffix(
    seq: &Sequence,
    block_ref_count: impl Fn(usize) -> usize,
) -> (usize, usize) {
    let table_len = seq.block_table.len();
    if table_len == 0 {
        return (0, 0);
    }
    let mut prefix_end = seq.num_cached_tokens / seq.block_size;
    prefix_end = prefix_end.min(table_len);
    // Extend prefix region for any shared block (ref_count > 1) after cached prefix.
    for i in prefix_end..table_len {
        let bid = seq.block_table[i] as usize;
        if block_ref_count(bid) > 1 {
            prefix_end = i + 1;
        } else {
            break;
        }
    }
    prefix_end = prefix_end.min(table_len);
    (prefix_end, table_len - prefix_end)
}

pub fn classify_block(_block_id: usize, ref_count: usize, is_prefix_region: bool) -> BlockClass {
    if is_prefix_region || ref_count > 1 {
        BlockClass::AttachedPrefix
    } else {
        BlockClass::SeqOwned
    }
}

/// Build GPU→CPU pairs for suffix-only swap-out.
pub fn suffix_swap_pairs(
    seq: &Sequence,
    prefix_blocks: usize,
    cpu_block_ids: &[usize],
) -> Vec<(usize, usize)> {
    seq.block_table
        .iter()
        .skip(prefix_blocks)
        .enumerate()
        .map(|(i, &gpu_id)| (gpu_id as usize, cpu_block_ids[i]))
        .collect()
}

/// Build CPU→GPU pairs for suffix-only swap-in.
pub fn suffix_swap_in_pairs(
    seq: &Sequence,
    prefix_blocks: usize,
    cpu_block_ids: &[usize],
) -> Vec<(usize, usize)> {
    cpu_block_ids
        .iter()
        .enumerate()
        .map(|(i, &cpu_id)| {
            let gpu_id = seq.block_table[prefix_blocks + i] as usize;
            (cpu_id, gpu_id)
        })
        .collect()
}