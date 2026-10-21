//! The `SchedD` domain (the scheduler domain). It owns the `Scheduler` state (the
//! waiting/running/cached sequences, the block allocation) + handles the
//! `/Cancel` messages (from the `HttpD` domain) + drives the inference
//! loop (the `step` message).
//!
//! This is the domain that replaces the `Scheduler` shared struct. The `SchedD`
//! actor owns the scheduler state + communicates with the other domains via
//! messages (the no shared locks).
//!
//! Ractor actors run as tokio tasks. The `step` handler drives the full inference
//! loop (schedule -> forward -> postprocess -> delivery -> cleanup). The
//! `Scheduler` methods it calls are async (they drive the `KvD` actor's RPCs).

use ractor::ActorProcessingErr;
use ractor::rpc::CallResult;
use ractor::{Actor, ActorRef, RpcReplyPort};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::core::scheduler::Scheduler;
use crate::core::sequence::Sequence;
use crate::domain::kv::KvMessage;
use crate::domain::fwd::FwdMessage;
use crate::domain::http::HttpMessage;
use crate::domain::stream::StreamMessage;
use crate::domain::OwnedSeqs;
use crate::runner::MessageType;
use crate::runner::SpecSeqStatsData;

/// The scheduler actor.
pub struct SchedD;

/// The actor's state: the scheduler + the refs to the other actors.
pub struct SchedState {
    /// The scheduler state (the waiting/running/cached sequences, the block
    /// allocation). Owned by this actor (the no shared state).
    scheduler: Scheduler,
    /// The forward-pass actor (the sole model path).
    fwd: ActorRef<FwdMessage>,
    /// The stream delivery actor (the token delivery).
    stream: ActorRef<StreamMessage>,
    /// The HTTP ingestion actor (the request lifecycle, the prompt replay).
    /// Set after both actors are created (the circular dependency resolution).
    http: Mutex<Option<ActorRef<HttpMessage>>>,
    /// Whether the model uses the builtin MTP (the speculative decode).
mtp_enabled: bool,
    /// Whether the model uses DFlash (the speculative decode).
    dflash_enabled: bool,
    /// The per-seq speculative-decode stats captured at finish (the capture-
    /// before-drop: pulled from the runner in `fwd_finished` BEFORE the `Finished`
    /// message drops them runner-side, so the server's end-of-sequence report reads
    /// a surviving in-process copy). Keyed by seq id; dropped on `GetSpecStats`.
    spec_stats: Mutex<HashMap<usize, SpecSeqStatsData>>,
    /// The anti-loop detector (the per-seq bounded tail). Only active when
    /// XINFER_ANTI_LOOP is on (the zero-overhead default).
    loop_detector: crate::utils::loop_detect::LoopDetector,
    /// The per-seq anti-loop kicks (held until the periodicity clears).
    kick_store: crate::utils::loop_detect::KickStore,
    /// Per-sequence loop-defense stats (the guards + the durations, the admins' report).
    seq_loop_stats: std::collections::HashMap<usize, crate::utils::loop_detect::LoopSeqStats>,
    /// The transient per-seq guard-held-step counter (the duration tracking, not serialized).
    guard_step_counts: std::collections::HashMap<usize, usize>,
}

/// The `SchedD` actor's message enum (the explicit form).
pub enum SchedMessage {
    /// Add a new sequence (the HTTP ingestion). Returns the assigned seq id.
    Add(Sequence, RpcReplyPort<usize>),
    /// Cancel a sequence (the client disconnect).
    Cancel(usize),
    /// Run one engine-loop iteration: schedule a batch, drive the forward pass
    /// through the `FwdD` actor, postprocess + deliver tokens through the
    /// `StreamD` actor. Returns the number of sequences processed.
    Step(RpcReplyPort<usize>),
    /// Set the HTTP ingestion actor reference (the circular dependency resolution).
    SetHttp(ActorRef<HttpMessage>),
    // The scheduler's non-loop access (the request handling, the state management,
    // the block queries) routed through the actor proxy.
    GetRunning(usize, RpcReplyPort<Option<Sequence>>),
    GetSequences(Vec<usize>, RpcReplyPort<Vec<Sequence>>),
    GetNumCachedTokens(RpcReplyPort<usize>),
    GetNumCachedTokensForSeq(usize, RpcReplyPort<Option<usize>>),
    GetAvailableKvTokens(RpcReplyPort<usize>),
    GetTotalKvTokens(RpcReplyPort<usize>),
    GetCpuSwapUsage(RpcReplyPort<(f32, f32)>),
    GetSeqTokenUsage(usize, RpcReplyPort<Result<usize, String>>),
    FindSeqBySessionId(String, RpcReplyPort<Option<(usize, crate::core::sequence::SequenceStatus)>>),
    TakePendingRunnerReleases(RpcReplyPort<Vec<usize>>),
    KvCacheUsagePercent(RpcReplyPort<f32>),
    ClearFinished,
    ReleaseWaitings(RpcReplyPort<Vec<usize>>),
    EvictPrefixCacheBlocks(usize, RpcReplyPort<usize>),
    EvictPrefixUntilFree(usize, RpcReplyPort<usize>),
    BlockRequired(Sequence, RpcReplyPort<usize>),
    BlockFree(RpcReplyPort<usize>),
    BlockAllocate(Sequence, RpcReplyPort<Option<Sequence>>),
    BlockDeallocate(Sequence),
    PrintFreeBlocks,
    /// The end-of-sequence scheduling report (the cached tokens + the QoS stats).
GetSeqReport(usize, RpcReplyPort<Option<crate::core::qos::SeqReport>>),
    /// The per-seq speculative-decode stats captured at finish (the capture-
    /// before-drop: read from the `SchedD`-owned map, the no runner-side drop race).
    /// Returns `None` once taken (the map entry is removed on read).
    GetSpecStats(usize, RpcReplyPort<Option<SpecSeqStatsData>>),
    /// The active (running) sequence indices (for the periodic decode throughput log).
    GetActiveIndices(RpcReplyPort<Vec<usize>>),
    /// The cached per-seq loop-defense stats (for the server's end-of-sequence report).
    GetSeqLoopStats(usize, RpcReplyPort<Option<crate::utils::loop_detect::LoopSeqStats>>),
}

#[ractor::async_trait]
impl ractor::Actor for SchedD {
    type Msg = SchedMessage;
    type State = SchedState;
    type Arguments = (
        Scheduler,
        ActorRef<FwdMessage>,
        ActorRef<StreamMessage>,
        bool,
        bool,
    );

    async fn pre_start(
        &self,
        _myself: ActorRef<SchedMessage>,
        args: (
            Scheduler,
            ActorRef<FwdMessage>,
            ActorRef<StreamMessage>,
            bool,
            bool,
        ),
    ) -> Result<SchedState, ActorProcessingErr> {
        let (scheduler, fwd, stream, mtp_enabled, dflash_enabled) = args;
        Ok(SchedState {
            scheduler,
            fwd,
            stream,
            http: Mutex::new(None),
            mtp_enabled,
dflash_enabled,
            spec_stats: Mutex::new(HashMap::new()),
            loop_detector: crate::utils::loop_detect::LoopDetector::new(),
            kick_store: crate::utils::loop_detect::KickStore::new(),
            seq_loop_stats: std::collections::HashMap::new(),
            guard_step_counts: std::collections::HashMap::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<SchedMessage>,
        message: SchedMessage,
        state: &mut SchedState,
    ) -> Result<(), ActorProcessingErr> {
        use SchedMessage::*;
        match message {
            Add(seq, reply) => {
                let _ = reply.send(state.scheduler.add(seq));
            }
            Cancel(seq_id) => {
                state.scheduler.cancel(seq_id).await;
            }
            Step(reply) => {
                let processed = state.step().await;
                let _ = reply.send(processed);
            }
            SetHttp(http) => {
                *state.http.lock().unwrap() = Some(http);
            }
            GetRunning(idx, reply) => {
                let _ = reply.send(state.scheduler.get_running(idx).cloned());
            }
            GetSequences(ids, reply) => {
                let _ = reply.send(state.scheduler.get_sequences(&ids).into_iter().map(|s| s.clone()).collect());
            }
            GetNumCachedTokens(reply) => {
                let _ = reply.send(state.scheduler.get_num_cached_tokens().await);
            }
            GetNumCachedTokensForSeq(seq_id, reply) => {
                let _ = reply.send(state.scheduler.get_num_cached_tokens_for_seq(seq_id));
            }
            GetAvailableKvTokens(reply) => {
                let _ = reply.send(state.scheduler.get_available_kv_tokens().await);
            }
            GetTotalKvTokens(reply) => {
                let _ = reply.send(state.scheduler.get_total_kv_tokens().await);
            }
            GetCpuSwapUsage(reply) => {
                let _ = reply.send(state.scheduler.get_cpu_swap_usage().await);
            }
            GetSeqTokenUsage(seq_id, reply) => {
                let _ = reply.send(state.scheduler.get_seq_token_usage(seq_id).map_err(|e| e.to_string()));
            }
            FindSeqBySessionId(session_id, reply) => {
                let _ = reply.send(state.scheduler.find_seq_by_session_id(&session_id));
            }
            TakePendingRunnerReleases(reply) => {
                let _ = reply.send(state.scheduler.take_pending_runner_releases());
            }
            KvCacheUsagePercent(reply) => {
                let _ = reply.send(state.scheduler.kv_cache_usage_percent().await);
            }
            ClearFinished => {
                state.scheduler.clear_finished().await;
            }
            ReleaseWaitings(reply) => {
                let _ = reply.send(state.scheduler.release_waitings().await);
            }
            EvictPrefixCacheBlocks(n, reply) => {
                let _ = reply.send(state.scheduler.evict_prefix_cache_blocks(n).await);
            }
            EvictPrefixUntilFree(min_free_blocks, reply) => {
                let _ = reply.send(state.scheduler.evict_prefix_until_free(min_free_blocks).await);
            }
            BlockRequired(seq, reply) => {
                let val = match state.scheduler.kv.call(|r| KvMessage::RequiredBlocks(seq, r), None).await {
                    Ok(CallResult::Success(v)) => v,
                    _ => 0,
                };
                let _ = reply.send(val);
            }
            BlockFree(reply) => {
                let val = match state.scheduler.kv.call(|r| KvMessage::FreeBlocks(r), None).await {
                    Ok(CallResult::Success(v)) => v,
                    _ => 0,
                };
                let _ = reply.send(val);
            }
            BlockAllocate(seq, reply) => {
                let val = match state.scheduler.kv.call(|r| KvMessage::Allocate(seq, r), None).await {
                    Ok(CallResult::Success(Ok(s))) => Some(s),
                    _ => None,
                };
                let _ = reply.send(val);
            }
            BlockDeallocate(seq) => {
                let _ = state.scheduler.kv.send_message(KvMessage::Deallocate(seq));
            }
            PrintFreeBlocks => {
                state.scheduler.print_free_blocks().await;
            }
            GetSeqReport(seq_id, reply) => {
                let _ = reply.send(state.scheduler.get_seq_report(seq_id));
            }
GetActiveIndices(reply) => {
                let _ = reply.send(state.scheduler.get_active_indices());
}
            GetSpecStats(seq_id, reply) => {
                // Take the captured per-seq spec stats (the no runner-side
                // drop race). Removed on read so a seq's stats are surfaced once.
                let data = state.spec_stats.lock().unwrap().remove(&seq_id);
                let _ = reply.send(data);
            }
            GetSeqLoopStats(seq_id, reply) => {
                let _ = reply.send(state.seq_loop_stats.get(&seq_id).cloned());
            }
        }
        Ok(())
    }
}

impl SchedState {
    /// The one engine-loop cycle: schedule a batch, drive the forward pass
    /// through the `FwdD` actor, postprocess + deliver tokens through the
    /// `StreamD` actor. This replaces the old `start_engine` loop body.
    async fn step(&mut self) -> usize {
        // Phase 1: schedule (the scheduler state a batch).
        let (scheduled_ids, is_prefill) = match self.scheduler.schedule().await {
            Ok((ids, prefill)) if !ids.is_empty() => (ids, prefill),
            Ok(_) => {
                self.check_canceled(None).await;
                return 0;
            }
            Err(e) => {
                crate::log_error!("[SchedD] Schedule error: {e:?}");
                self.check_canceled(Some(format!("Schedule error: {e:?}"))).await;
                return 0;
            }
        };

        // Mamba prefix snapshot verification (prefill only).
        if is_prefill {
            let downgraded = self
                .scheduler
                .fallback_missing_mamba_prefix_snapshots(&scheduled_ids)
                .await
                .unwrap_or(0);
            if downgraded > 0 {
                crate::log_warn!(
                    "[SchedD] Prefill fallback: {downgraded} sequence(s) downgraded to full prefill"
                );
            }
        }

        // Phase 2: forward pass (through the FwdD actor, the sole model path).
        // The single-sequence check (the forward routing: the run_mtp_decode / the run_dflash_decode
        // only work for a single sequence; the batch is the run_mtp_decode_batch). The env-var
        // handling (the spec_parallel_draft / the dflash_parallel_slots) allows multi-sequence
        // drafting. Computed BEFORE the snapshot (the scheduled_ids.len() == the owned_seqs.len()).
        let parallel_ok = scheduled_ids.len() == 1 || crate::utils::env::spec_parallel_draft();
        let dflash_ok = parallel_ok
            && scheduled_ids.len() <= crate::utils::env::dflash_parallel_slots().max(1);
        let use_dflash = self.dflash_enabled && !is_prefill && dflash_ok;
        let use_mtp = self.mtp_enabled && !use_dflash && !is_prefill && parallel_ok;
        // Speculative fast-forward: append the full grammar-forced (ff) run after the
        // produced tokens (the MTP/DFlash draft, the spec-FF base, or the plain
        // decode). The ff is the deterministic continuation (the tool-call XML, the
        // forced literals), so it is appended regardless of which decode path
        // produced the base tokens. Gated on the XINFER_SPEC_FF env (default off)
        // + the decode step (not the prefill).
        let use_spec_ff = crate::utils::env::spec_ff() && !is_prefill;

// The single-writer step view (the Option 5, the no write-after-snapshot): the Scheduler does
        // ALL the writes (the speculative pre-alloc) + the freeze (the Sequence snapshot) in ONE
        // atomic call. No caller can interleave a write between the snapshot and the forward. The
        // gate is the origin/main pattern (the speculative decode ENABLED + the decode step, NOT
        // the single-sequence check).
        let owned_seqs = self.scheduler.prepare_step_view(&scheduled_ids, is_prefill).await;
        // Resolve the actual sequence IDs (the gate_commit + the delivery need the
        // sequence ID, not the running position).
        let seq_ids: Vec<usize> = owned_seqs.iter().map(|s| s.id).collect();

        // The MTP per-seq next-step K feedback (the runner-computed dynamic-K, the no the
        // scheduler guessing from an empty cross-process global). The no-MTP paths leave it
        // empty (the no K feedback, the prior behavior).
        let mut mtp_next_k: Vec<usize> = Vec::new();
        let forward_result: Result<Vec<Vec<u32>>, String> = if use_mtp {
            let sequences: Vec<crate::core::sequence::DecodeSequence> =
                owned_seqs.iter().map(crate::core::sequence::DecodeSequence::new).collect();
            match self.fwd_call_run_mtp(&sequences).await {
                Ok(step_outs) => {
                    // The capture the per-seq next-K (the runner-computed, the dynamic-K
                    // feedback) + extract the tokens (the no change to the postprocess/delivery).
                    mtp_next_k = step_outs.iter().map(|o| o.next_k).collect();
                    Ok(step_outs.into_iter().map(|o| o.tokens).collect())
                }
                Err(e) => Err(e),
            }
        } else if use_dflash {
            let sequences: Vec<crate::core::sequence::DecodeSequence> =
                owned_seqs.iter().map(crate::core::sequence::DecodeSequence::new).collect();
self.fwd_call_run_dflash(&sequences).await
        } else if use_spec_ff {
            let sequences: Vec<crate::core::sequence::DecodeSequence> =
                owned_seqs.iter().map(crate::core::sequence::DecodeSequence::new).collect();
            self.fwd_call_run_spec_ff(&sequences).await
        } else {
            let owned = if is_prefill {
                OwnedSeqs::Prefill(owned_seqs)
            } else {
                OwnedSeqs::Decode(
                    owned_seqs.iter().map(crate::core::sequence::DecodeSequence::new).collect(),
                )
            };
            self.fwd_call_run(&owned).await
        };

        // Phase 3: postprocess + stream delivery + cleanup.
        match forward_result {
            Ok(multi_output_ids) => {
                // The dynamic-K feedback: the runner-reported per-seq next-K (the
                // rolling prefix-curve knee, the no the scheduler guessing from an empty
                // cross-process global) is recorded into the scheduler's per-seq K map,
                // so the NEXT step's pre-alloc uses the runner's real signal. The no-MTP
                // paths leave mtp_next_k empty (the no K feedback, the prior behavior).
                if !mtp_next_k.is_empty() {
                    self.scheduler.record_spec_k_feedback(&scheduled_ids, &mtp_next_k);
                }

if crate::utils::env::actor_debug() {
                    crate::log_info!(
                        "[SchedD] forward ok: {} seq(s), is_prefill={}, outputs={:?}",
                        multi_output_ids.len(), is_prefill,
                        multi_output_ids.iter().map(|v| v.len()).collect::<Vec<_>>()
                    );
                }
                // Single matcher-gated ingress: commit each produced run to the FSM
                // and keep only the FSM-passing prefix, just before the sequence
                // append. `ff` selects the spec-FF ordered-queue writer (the ff
                // continuation is appended after the produced tokens, regardless
                // of which decode path produced them).
                let use_spec_ff = crate::utils::env::spec_ff() && !is_prefill;
                let gated = match self.fwd.call(|r| FwdMessage::GateCommit(seq_ids.clone(), multi_output_ids.clone(), use_spec_ff, r), None).await {
                    Ok(CallResult::Success(v)) => v,
                    _ => multi_output_ids.clone(), // the gate unavailable: pass through (the no FSM filtering)
                };
                if crate::utils::env::actor_debug() {
                    crate::log_info!(
                        "[SchedD] gate_commit: seq_ids={:?}, gated={:?}",
                        seq_ids, gated.iter().map(|v| v.len()).collect::<Vec<_>>()
                    );
                }
                // The anti-loop OBSERVE: check the accepted tokens for a closed loop.
                // Zero-overhead when XINFER_ANTI_LOOP is off (the default).
                if crate::utils::env::anti_loop_enabled() {
                    for (i, &seq_id) in seq_ids.iter().enumerate() {
                        if i < gated.len() {
                            self.observe_loop(seq_id, &gated[i]);
                        }
                    }
                }
                // Prefill: filter sequences still chunking from those done.
                // The SEAL: the append consumes the `gated` (the FSM-legal prefix from the
                // single `gate_commit`), never the raw `multi_output_ids`. When guidance is
                // inactive the `gated` == the raw (the fail-open), so this is a no-op for
                // unguided sequences.
                let (indices, output_by_running_index) = if is_prefill {
                    let (indices, finished_indices) =
                        self.scheduler.filter_prefill_finished(&scheduled_ids).await;
                    if indices.is_empty() {
                        self.check_canceled(None).await;
                        return 0;
                    }
                    let wrapped: Vec<Vec<u32>> = indices
                        .iter()
                        .map(|&output_idx| {
                            gated.get(output_idx).cloned().unwrap_or_default()
                        })
                        .collect();
                    self.scheduler.postprocess(&finished_indices, &wrapped).await;
                    let output_map = finished_indices
                        .iter()
                        .copied()
                        .zip(wrapped)
                        .collect::<std::collections::HashMap<_, _>>();
                    (finished_indices, output_map)
                } else {
                    self.scheduler.postprocess(&scheduled_ids, &gated).await;
                    let output_map = scheduled_ids
                        .iter()
                        .copied()
                        .zip(gated.iter().cloned())
                        .collect::<std::collections::HashMap<_, _>>();
                    (scheduled_ids, output_map)
                };

                // Per-sequence delivery loop.
                let mut prefill_batch: Vec<(usize, usize, f32, bool)> = Vec::new();
                for &idx in &indices {
                    let step_tokens = match output_by_running_index.get(&idx) {
                        Some(t) => t,
                        None => continue,
                    };
                    let Some(sq) = self.scheduler.get_running(idx) else { continue };
                    let seq_id = sq.id;
                    let s = sq;

                    if s.is_finished() {
                        // Deliver the final item (Done or Completion).
                        let request_type = self.stream_get_request_type(seq_id).await;
                        if let Some(request_type) = request_type {
                            let prompt_start_time = s.created_time();
                            let decode_finish_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as usize;
                            let decode_start_time = self
                                .stream_get_decode_start_time(seq_id)
                                .await
                                .unwrap_or(decode_finish_time);

                            // Tool-call-end: deliver the last token first.
                            if s.is_tool_call_end {
                                if request_type == crate::core::engine::RequestType::Stream {
                                    let decoded = self.stream_call_decode_tokens(seq_id, vec![s.last_token]).await;
                                    if let Some((_, text)) = decoded.first() {
                                        self.stream_deliver_token(seq_id, text.clone(), s.last_token).await;
                                    }
                                } else {
                                    self.stream_deliver_token_id(seq_id, s.last_token).await;
                                }
                            }

                            if request_type == crate::core::engine::RequestType::Stream {
                                let num_appended = s.output_ids.len()
                                    - self.stream_get_decode_length(seq_id).await;
                                let tokens_to_stream = &step_tokens[..num_appended.min(step_tokens.len())];
                                if !tokens_to_stream.is_empty() {
                                    let decoded = self.stream_call_decode_tokens(seq_id, tokens_to_stream.to_vec()).await;
                                    if decoded.is_empty() {
                                        for &tok in tokens_to_stream {
                                            self.stream_deliver_token(seq_id, String::new(), tok).await;
                                        }
                                    } else {
                                        for (tok, text) in decoded {
                                            self.stream_deliver_token(seq_id, text, tok).await;
                                        }
                                    }
                                }
                                self.stream_deliver_done(
                                    seq_id,
                                    crate::core::engine::StreamItem::Done((
                                        prompt_start_time,
                                        decode_start_time,
                                        decode_finish_time,
                                        s.output_ids.len(),
                                        s.stop_sequence.clone(),
                                    )),
                                ).await;
                            } else {
                                self.stream_deliver_done(
                                    seq_id,
                                    crate::core::engine::StreamItem::Completion((
                                        prompt_start_time,
                                        decode_start_time,
                                        decode_finish_time,
                                        s.output_ids.clone(),
                                        s.stop_sequence.clone(),
                                    )),
                                )
                                .await;
                            }

                            // Cleanup: remove delivery state + notify the runner + mark inactive.
                            self.stream_forget_sequence(seq_id).await;
                            if let Some(http) = self.http.lock().unwrap().clone() {
                                let _ = http.send_message(HttpMessage::RemoveActiveRequest(seq_id));
                            }
                            self.fwd_finished(seq_id).await;
                        }
                    } else {
                        // Not finished: track decode start, deliver new tokens.
                        if self.stream_get_decode_start_time(seq_id).await.is_none() {
                            let cur_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as usize;
                            self.stream_record_decode_start(seq_id, cur_time).await;
                            self.stream_set_decode_length(seq_id, 1).await;
                            let time_costs = cur_time - s.created_time();
                            if time_costs / 100 > 0 && s.len() > 0 {
                                prefill_batch.push((
                                    seq_id,
                                    s.len(),
                                    time_costs as f32 / 1000f32,
                                    s.num_cached_tokens > 0,
                                ));
                            }
                        }
                        self.stream_set_decode_length(seq_id, s.output_len()).await;

                        // Compute the tokens to deliver this step.
                        let mut token_ids: Vec<u32> = if step_tokens.len() > 1 {
                            step_tokens.clone()
                        } else {
                            vec![s.last_token]
                        };
                        // Prompt replay: prepend the replay tokens.
                        let http_ref = self.http.lock().unwrap().clone();
                        if let Some(http) = http_ref {
                            if let Some(mut replay_ids) = self.http_take_prompt_replay(&http, seq_id).await {
                                replay_ids.extend(token_ids);
                                token_ids = replay_ids;
                            }
                        }

                        let request_type = self.stream_get_request_type(seq_id).await;
                        if let Some(request_type) = request_type {
                            if request_type == crate::core::engine::RequestType::Stream {
                                let decoded = self.stream_call_decode_tokens(seq_id, token_ids.clone()).await;
                                if decoded.is_empty() {
                                    for &token_id in &token_ids {
                                        self.stream_deliver_token(seq_id, String::new(), token_id).await;
                                    }
                                } else {
                                    for (token_id, text) in decoded {
                                        self.stream_deliver_token(seq_id, text, token_id).await;
                                    }
                                }
                            } else {
                                for &token_id in &token_ids {
                                    self.stream_deliver_token_id(seq_id, token_id).await;
                                }
                            }
                        }
                    }
                }
// Prefill batch log.
                if !prefill_batch.is_empty() {
                    let total_tokens: usize = prefill_batch.iter().map(|(_, t, _, _)| *t).sum();
                    let max_time: f32 = prefill_batch.iter().map(|(_, _, t, _)| *t).fold(0.0f32, f32::max);
                    if max_time > 0.0 {
                        crate::log_info!(
                            "[SchedD] Prefilling {} seq(s): {} total tokens in {:.2}s ({:.2} tokens/s)",
                            prefill_batch.len(),
                            total_tokens,
                            max_time,
                            total_tokens as f32 / max_time,
                        );
                    }
                }

                // Cleanup: remove finished sequences from running/waiting.
                self.scheduler.clear_finished().await;

                // KV pressure handling.
                if indices.is_empty()
                    && self.scheduler.kv_cache_usage_percent().await > crate::core::scheduler::KVCACHE_SWAP_THRESHOLD + 0.01f32
                {
                    self.check_canceled(Some("KV cache pressure: dropping oldest request".to_string())).await;
                } else {
                    self.check_canceled(None).await;
                }

                indices.len()
            }
            Err(e) => {
                crate::log_error!("[SchedD] Forward error: {e}");
                self.check_canceled(Some(format!("Forward error: {e}"))).await;
                0
            }
        }
    }

    // The FwdD actor RPC calls (the async Ractor calls).
    async fn fwd_call_run(&self, owned: &OwnedSeqs) -> Result<Vec<Vec<u32>>, String> {
        match owned {
            OwnedSeqs::Prefill(s) => {
                let msg = MessageType::RunPrefill((s.clone(), true));
                self.fwd_call_round_trip(msg).await
            }
            OwnedSeqs::Decode(d) => {
                let msg = MessageType::RunDecode((d.clone(), false));
                self.fwd_call_round_trip(msg).await
            }
        }
    }

    async fn fwd_call_run_mtp(&self, sequences: &[crate::core::sequence::DecodeSequence]) -> Result<Vec<crate::runner::SpecStepOut>, String> {
        // Use the FwdD actor's RunMtp RPC (which correctly handles the
        // RunResponseMTP response, the per-seq tokens + the next-step dynamic K),
        // not the generic ForwardRaw escape hatch (which only handles the plain
        // RunResponse).
        match self.fwd.call(|r| FwdMessage::RunMtp(sequences.to_vec(), r), None).await {
            Ok(CallResult::Success(result)) => result,
            _ => Err("IPC round_trip failed (timeout or actor stopped)".into()),
        }
    }

    async fn fwd_call_run_dflash(&self, sequences: &[crate::core::sequence::DecodeSequence]) -> Result<Vec<Vec<u32>>, String> {
        // Use the FwdD actor's RunDflash RPC (which correctly handles the
        // RunResponseDFlash response), not the generic ForwardRaw escape hatch
        // (which only handles the plain RunResponse).
        match self.fwd.call(|r| FwdMessage::RunDflash(sequences.to_vec(), r), None).await {
            Ok(CallResult::Success(result)) => result,
            _ => Err("IPC round_trip failed (timeout or actor stopped)".into()),
        }
    }

async fn fwd_call_run_spec_ff(&self, sequences: &[crate::core::sequence::DecodeSequence]) -> Result<Vec<Vec<u32>>, String> {
        // Use the FwdD actor's RunSpecFf RPC (which correctly handles the
        // RunResponseSpecFF response), not the generic ForwardRaw escape hatch
        // (which only handles the plain RunResponse).
        match self.fwd.call(|r| FwdMessage::RunSpecFf(sequences.to_vec(), r), None).await {
            Ok(CallResult::Success(result)) => result,
            _ => Err("IPC round_trip failed (timeout or actor stopped)".into()),
        }
    }

    async fn fwd_call_round_trip(&self, msg: MessageType) -> Result<Vec<Vec<u32>>, String> {
        let resp = match self.fwd.call(|r| FwdMessage::ForwardRaw(msg, r), None).await {
            Ok(CallResult::Success(m)) => Some(m),
            _ => None,
        };
        match resp {
            Some(MessageType::RunResponse(ids)) if !ids.is_empty() => {
                Ok(ids.into_iter().map(|t| vec![t]).collect())
            }
            Some(MessageType::RunResponse(_)) => Err("runner returned an empty response".into()),
            Some(MessageType::Error(e)) => Err(e),
            Some(other) => Err(format!("unexpected IPC response: {other:?}")),
            None => Err("IPC round_trip failed (timeout or actor stopped)".into()),
        }
    }

    async fn fwd_finished(&self, seq_id: usize) {
        // The capture-before-drop: pull the per-seq spec stats from the runner
        // BEFORE the `Finished` message drops them (the runner's `FinishDecode` ->
        // `spec_seq_report` removes the entry). Without this the server's end-of-
        // sequence report (the lazy `notify_runner_finished` fetch) reads an already-
        // dropped entry + the "[Seq N] <mechanism> Speculation" line is skipped.
        let captured = match self
            .fwd
            .call(|r| FwdMessage::FetchSpecSeqStats(seq_id, r), None)
            .await
        {
            Ok(CallResult::Success(data)) if !data.mechanism.is_empty() => Some(data),
            _ => None,
        };
        if let Some(data) = captured {
            self.spec_stats.lock().unwrap().insert(seq_id, data);
        }
        let _ = self.fwd.send_message(FwdMessage::Finished(seq_id));
    }

    // The StreamD actor RPC calls (the async Ractor calls).
    async fn stream_get_request_type(&self, seq_id: usize) -> Option<crate::core::engine::RequestType> {
        match self.stream.call(|r| StreamMessage::GetRequestType(seq_id, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => None,
        }
    }
    async fn stream_get_decode_start_time(&self, seq_id: usize) -> Option<usize> {
        match self.stream.call(|r| StreamMessage::GetDecodeStartTime(seq_id, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => None,
        }
    }
    async fn stream_get_decode_length(&self, seq_id: usize) -> usize {
        match self.stream.call(|r| StreamMessage::GetDecodeLength(seq_id, r), None).await {
            Ok(CallResult::Success(v)) => v.unwrap_or(0),
            _ => 0,
        }
    }
    async fn stream_call_decode_tokens(&self, seq_id: usize, tokens: Vec<u32>) -> Vec<(u32, String)> {
        match self.stream.call(|r| StreamMessage::DecodeTokens(seq_id, tokens, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => Vec::new(),
        }
    }
    async fn stream_deliver_token(&self, seq_id: usize, text: String, token: u32) {
        let _ = self.stream.send_message(StreamMessage::DeliverToken(seq_id, text, token));
    }
    async fn stream_deliver_token_id(&self, seq_id: usize, token: u32) {
        let _ = self.stream.send_message(StreamMessage::DeliverTokenId(seq_id, token));
    }
    async fn stream_deliver_done(&self, seq_id: usize, item: crate::core::engine::StreamItem) {
        let _ = self.stream.send_message(StreamMessage::DeliverDone(seq_id, item));
    }
    async fn stream_forget_sequence(&self, seq_id: usize) {
        let _ = self.stream.send_message(StreamMessage::ForgetSequence(seq_id));
    }
    async fn stream_record_decode_start(&self, seq_id: usize, time: usize) {
        let _ = self.stream.send_message(StreamMessage::RecordDecodeStart(seq_id, time));
    }
    async fn stream_set_decode_length(&self, seq_id: usize, n: usize) {
        let _ = self.stream.send_message(StreamMessage::SetDecodeLength(seq_id, n));
    }

    /// Process pending cancellations (the client loop's `check_canceled`).
    async fn check_canceled(&mut self, reason: Option<String>) {
        let cancelled = {
            let http_ref = self.http.lock().unwrap().clone();
            match http_ref {
                Some(http) => self.http_take_cancellations(&http).await,
                None => Vec::new(),
            }
        };
        if !cancelled.is_empty() {
            for seq_id in &cancelled {
                self.scheduler.cancel(*seq_id).await;
                self.fwd_finished(*seq_id).await;
                if let Some(http) = self.http.lock().unwrap().clone() {
                    let _ = http.send_message(HttpMessage::RemoveActiveRequest(*seq_id));
                }
                self.stream_forget_sequence(*seq_id).await;
                let http_ref = self.http.lock().unwrap().clone();
                if let Some(http) = http_ref {
                    let _ = self.http_take_prompt_replay(&http, *seq_id).await;
                }
                if let Some(r) = &reason {
                    let request_type = self.stream_get_request_type(*seq_id).await;
                    if let Some(rt) = request_type {
                        if rt == crate::core::engine::RequestType::Stream {
                            self.stream_deliver_done(*seq_id, crate::core::engine::StreamItem::Error(r.clone())).await;
                        }
                    }
                }
            }
            self.scheduler.clear_finished().await;
        }
        // Drain pending runner releases.
        let releases = self.scheduler.take_pending_runner_releases();
        for seq_id in releases {
            self.fwd_finished(seq_id).await;
        }
    }

    /// The anti-loop OBSERVE: after the accepted tokens are known, check for a closed
    /// loop. On a hit, apply the kick (the mitigation) + record the per-seq stats + emit
    /// the [llg] event log. On a miss (the loop cleared), remove the kick + log it.
    /// Zero-overhead when XINFER_ANTI_LOOP is off (the default).
    fn observe_loop(&mut self, seq_id: usize, accepted: &[u32]) {
        use crate::utils::loop_detect::LoopSeqStats;
        let probe_depth = crate::utils::env::repetition_probe_depth();
        let was_active = self.kick_store.is_active(seq_id);
        // The guard duration tracking: while a kick is held, count the steps.
        if was_active {
            *self.guard_step_counts.entry(seq_id).or_insert(0) += 1;
            if let Some(stats) = self.seq_loop_stats.get_mut(&seq_id) {
                stats.guard_steps += 1;
            }
        }
        match self.loop_detector.observe(seq_id, accepted, probe_depth) {
            Some(span) => {
                self.kick_store.apply(seq_id, &span);
                let kick = self.kick_store.get(seq_id).expect("just applied");
                let stats = self.seq_loop_stats.entry(seq_id).or_insert_with(LoopSeqStats::default);
                stats.guards += 1;
                if !stats.periods.contains(&span.period) {
                    stats.periods.push(span.period);
                }
                crate::log_warn!(
                    "[llg] [Seq {}] loop detected: period={} unit={:?} -> mitigation applied (exclude {} tokens)",
                    seq_id, span.period, kick.unit, kick.unit.len()
                );
                // Bridge the kick to the runner (the sampling constraint) via the FwdD actor.
                // The FULL unit (the no single forbid token) -- the whole
                // period-P cycle is excluded at the trigger position to break the loop.
                let _ = self.fwd.send_message(FwdMessage::SetLoopKick(seq_id, kick.unit.clone(), kick.seed.clone(), span.period));
            }
            None => {
                if was_active {
                    self.kick_store.remove(seq_id);
                    crate::log_info!("[llg] [Seq {}] loop cleared; kick removed", seq_id);
                    let _ = self.fwd.send_message(FwdMessage::ClearLoopKick(seq_id));
                }
            }
        }
    }

    async fn http_take_cancellations(&self, http: &ActorRef<HttpMessage>) -> Vec<usize> {
        match http.call(|r| HttpMessage::TakeCancellations(r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => Vec::new(),
        }
    }
    async fn http_take_prompt_replay(&self, http: &ActorRef<HttpMessage>, seq_id: usize) -> Option<Vec<u32>> {
        match http.call(|r| HttpMessage::TakePromptReplay(seq_id, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => None,
        }
    }
}

/// Start the `SchedD` actor (the scheduler domain + the engine loop) on the
/// tokio runtime. Returns the `ActorRef` (the only handle other actors may hold).
pub async fn start_sched_actor(
    scheduler: Scheduler,
    fwd: ActorRef<FwdMessage>,
    stream: ActorRef<StreamMessage>,
    mtp_enabled: bool,
    dflash_enabled: bool,
) -> ActorRef<SchedMessage> {
    let (actor_ref, _) = SchedD::spawn(None, SchedD, (scheduler, fwd, stream, mtp_enabled, dflash_enabled))
        .await
        .expect("failed to start the SchedD actor");
    actor_ref
}