//! The `KvD` domain (the KV-cache domain). It owns the `BlockManager` (the block
//! pool, the prefix cache, the mamba slots, the tier ladder) + handles the
//! `Allocate`/`Free`/`TokenUpdate` messages (from the `SchedD` domain).
//!
//! This is the domain that replaces the `BlockManager` living inside the
//! `Scheduler`. The `KvD` actor owns the block manager + communicates with
//! the other domains via messages (the no shared locks).
//!
//! Ractor actors run as tokio tasks. The `BlockManager`'s `try_*` methods (the
//! PD transfer ops) are async (they drive the `FwdD` actor), so the `KvD`
//! handlers that call them are async too.

use ractor::ActorProcessingErr;
use ractor::{Actor, ActorRef, RpcReplyPort};

use crate::core::block_manager::BlockManager;
use crate::core::sequence::Sequence;

/// The KV-cache actor.
pub struct KvD;

/// The actor's state: the block manager (the block pool, the prefix cache, the
/// mamba slots, the tier ladder). Owned by this actor (the no shared state).
pub struct KvState {
    block_manager: BlockManager,
}

/// The `KvD` actor's message enum (the explicit form).
pub enum KvMessage {
    // The block queries (the request/response).
    RequiredBlocks(Sequence, RpcReplyPort<usize>),
    FreeBlocks(RpcReplyPort<usize>),
    TotalBlocks(RpcReplyPort<usize>),
    BlockSize(RpcReplyPort<usize>),
    CpuSwapUsage(RpcReplyPort<f32>),
    PrefixCacheBlocks(RpcReplyPort<usize>),
    PrefixCacheEnabled(RpcReplyPort<bool>),
    // The block allocation (the request/response).
    // Mutating methods return the modified Sequence (the block_table is written
    // inside the KvD actor; the caller rebinds the returned value).
    CanAllocate(Sequence, RpcReplyPort<bool>),
    Allocate(Sequence, RpcReplyPort<Result<Sequence, String>>),
    Deallocate(Sequence),
    CanAppend(Sequence, RpcReplyPort<bool>),
    MayAppend(Sequence, RpcReplyPort<Result<Sequence, String>>),
    EnsureAllocate(Sequence, RpcReplyPort<Result<Sequence, String>>),
    // The prefix cache (the request/response).
    CacheSequence(Sequence),
    EvictPrefixUntilFree(usize, RpcReplyPort<usize>),
    EvictPrefixBlocks(usize, RpcReplyPort<usize>),
    // The mamba prefix state (the request/response).
    CaptureMambaPrefix(Sequence, usize, RpcReplyPort<bool>),
    HasMambaPrefix(u64, RpcReplyPort<bool>),
    RemoveMambaPrefix(u64, RpcReplyPort<bool>),
    // The additional manager's remaining methods (the swap, the PD transfer, the prefix match).
    AllocFreeBlock(RpcReplyPort<Option<usize>>),
    CanAllocateWithoutPrefix(Sequence, RpcReplyPort<bool>),
    AllocateWithoutPrefix(Sequence, RpcReplyPort<Result<Sequence, String>>),
    GetPrefixCacheMatchTokens(Sequence, RpcReplyPort<usize>),
    ClearPrefixCache,
    TryTransferPrefill(Sequence, RpcReplyPort<bool>),
    TryReceivePrefill(usize, RpcReplyPort<(bool, Option<Sequence>)>),
    TryCheckPrefillStatus(usize, RpcReplyPort<bool>),
    TrySendKvcache(Sequence, u32, RpcReplyPort<bool>),
    TryReceiveKvcache(Sequence, RpcReplyPort<(bool, u32, usize, usize)>),
    TryReleaseRemoteKvcache(usize, RpcReplyPort<bool>),
    TryCheckKvcacheRelease(usize, RpcReplyPort<bool>),
    CanSwapOut(Sequence, RpcReplyPort<bool>),
    CanSwapIn(Sequence, RpcReplyPort<bool>),
    SwapInRequiredBlocks(Sequence, RpcReplyPort<usize>),
    HasPartialCpuSwap(usize, RpcReplyPort<bool>),
    RollbackPartialSwapInAllocation(Sequence, RpcReplyPort<Sequence>),
    SwapOut(Sequence, RpcReplyPort<Result<Sequence, String>>),
    SwapIn(Sequence, RpcReplyPort<Result<Sequence, String>>),
    HasCpuSwap(usize, RpcReplyPort<bool>),
    FreeCpuSwapForSeq(usize),
}

#[ractor::async_trait]
impl ractor::Actor for KvD {
    type Msg = KvMessage;
    type State = KvState;
    type Arguments = BlockManager;

    async fn pre_start(
        &self,
        _myself: ActorRef<KvMessage>,
        block_manager: BlockManager,
    ) -> Result<KvState, ActorProcessingErr> {
        Ok(KvState { block_manager })
    }

    async fn handle(
        &self,
        _myself: ActorRef<KvMessage>,
        message: KvMessage,
        state: &mut KvState,
    ) -> Result<(), ActorProcessingErr> {
        use KvMessage::*;
        let bm = &mut state.block_manager;
        match message {
            RequiredBlocks(seq, reply) => {
                let _ = reply.send(bm.required_blocks(&seq).await);
            }
            FreeBlocks(reply) => {
                let _ = reply.send(bm.get_num_free_blocks());
            }
            TotalBlocks(reply) => {
                let _ = reply.send(bm.get_num_total_blocks());
            }
            BlockSize(reply) => {
                let _ = reply.send(bm.get_block_size());
            }
            CpuSwapUsage(reply) => {
                let _ = reply.send(bm.get_cpu_swap_usage());
            }
            PrefixCacheBlocks(reply) => {
                let _ = reply.send(bm.prefix_cache_blocks());
            }
            PrefixCacheEnabled(reply) => {
                let _ = reply.send(bm.prefix_cache_enabled());
            }
            CanAllocate(seq, reply) => {
                let _ = reply.send(bm.can_allocate(&seq).await);
            }
            Allocate(seq, reply) => {
                let mut seq = seq;
                let result = bm.allocate(&mut seq).await.map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            Deallocate(seq) => {
                bm.deallocate(&seq);
            }
            CanAppend(seq, reply) => {
                let _ = reply.send(bm.can_append(&seq));
            }
            MayAppend(seq, reply) => {
                let mut seq = seq;
                let result = bm.may_append(&mut seq).map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            EnsureAllocate(seq, reply) => {
                let mut seq = seq;
                let result = bm.ensure_allocate(&mut seq).map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            CacheSequence(seq) => {
                bm.cache_sequence(&seq).await;
            }
            EvictPrefixUntilFree(min_free, reply) => {
                let _ = reply.send(bm.evict_prefix_cache_until_free(min_free).await);
            }
            EvictPrefixBlocks(n, reply) => {
                let _ = reply.send(bm.evict_prefix_cache_blocks(n).await);
            }
            CaptureMambaPrefix(seq, processed_tokens, reply) => {
                let _ = reply.send(bm.capture_mamba_prefix_state(&seq, processed_tokens).await.is_some());
            }
            HasMambaPrefix(hash, reply) => {
                let _ = reply.send(bm.try_has_mamba_prefix_state(hash).await.unwrap_or(false));
            }
            RemoveMambaPrefix(hash, reply) => {
                let _ = reply.send(bm.try_remove_mamba_prefix_state(hash).await.unwrap_or(false));
            }
            AllocFreeBlock(reply) => {
                let _ = reply.send(bm.alloc_free_block());
            }
            CanAllocateWithoutPrefix(seq, reply) => {
                let _ = reply.send(bm.can_allocate_without_prefix(&seq));
            }
            AllocateWithoutPrefix(seq, reply) => {
                let mut seq = seq;
                let result = bm.allocate_without_prefix(&mut seq).map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            GetPrefixCacheMatchTokens(seq, reply) => {
                let _ = reply.send(bm.get_prefix_cache_match_tokens(&seq).await);
            }
            ClearPrefixCache => {
                bm.clear_prefix_cache().await;
            }
            TryTransferPrefill(seq, reply) => {
                let _ = reply.send(bm.try_transfer_prefill(&seq).await.unwrap_or(false));
            }
            TryReceivePrefill(available_tokens, reply) => {
                let _ = reply.send(bm.try_receive_prefill(available_tokens).await.unwrap_or((false, None)));
            }
            TryCheckPrefillStatus(seq_id, reply) => {
                let _ = reply.send(bm.try_check_prefill_status(seq_id).await.unwrap_or(false));
            }
            TrySendKvcache(seq, token, reply) => {
                let _ = reply.send(bm.try_send_kvcache(&seq, token).await.unwrap_or(false));
            }
            TryReceiveKvcache(seq, reply) => {
                let _ = reply.send(bm.try_receive_kvcache(&seq).await.unwrap_or((false, 0, 0, 0)));
            }
            TryReleaseRemoteKvcache(seq_id, reply) => {
                let _ = reply.send(bm.try_release_remote_kvcache(seq_id).await.unwrap_or(false));
            }
            TryCheckKvcacheRelease(seq_id, reply) => {
                let _ = reply.send(bm.try_check_kvcache_release(seq_id).await.unwrap_or(false));
            }
            CanSwapOut(seq, reply) => {
                let _ = reply.send(bm.can_swap_out(&seq));
            }
            CanSwapIn(seq, reply) => {
                let _ = reply.send(bm.can_swap_in(&seq));
            }
            SwapInRequiredBlocks(seq, reply) => {
                let _ = reply.send(bm.swap_in_required_blocks(&seq));
            }
            HasPartialCpuSwap(seq_id, reply) => {
                let _ = reply.send(bm.has_partial_cpu_swap(seq_id));
            }
            RollbackPartialSwapInAllocation(seq, reply) => {
                let mut seq = seq;
                bm.rollback_partial_swap_in_allocation(&mut seq);
                let _ = reply.send(seq);
            }
            SwapOut(seq, reply) => {
                let mut seq = seq;
                let result = bm.swap_out(&mut seq).await.map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            SwapIn(seq, reply) => {
                let mut seq = seq;
                let result = bm.swap_in(&mut seq).await.map(|_| seq).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            HasCpuSwap(seq_id, reply) => {
                let _ = reply.send(bm.has_cpu_swap(seq_id));
            }
            FreeCpuSwapForSeq(seq_id) => {
                bm.free_cpu_swap_for_seq(seq_id);
            }
        }
        Ok(())
    }
}

/// Start the `KvD` actor (the KV-cache domain) on the tokio runtime. Returns
/// the `ActorRef` (the only handle other actors may hold).
pub async fn start_kv_actor(block_manager: BlockManager) -> ActorRef<KvMessage> {
    let (actor_ref, _) = KvD::spawn(None, KvD, block_manager)
        .await
        .expect("failed to start the KvD actor");
    actor_ref
}