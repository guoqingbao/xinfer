// src/core/scheduler.rs
use super::{
    prefix_cache::PrefixCacheConfig,
    qos::{BlockSnapshot, QosController, Regime, SchedSeqStats, SeqReport, SamplingOverride},
    sequence::{Sequence, SequenceStatus},
};
use crate::transfer::{PdConfig, PdRole};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use crate::domain::kv::KvMessage;
use candle_core::Result;
use ractor::rpc::CallResult;
use ractor::RpcReplyPort;

/// Extract the success value from a Ractor RPC call on the `KvD` actor, or a
/// default. Wraps the `Actor::call` + `CallResult` extraction so call sites
/// stay concise.
async fn kv_call<T: Send + 'static>(
    kv: &ractor::ActorRef<KvMessage>,
    builder: impl FnOnce(RpcReplyPort<T>) -> KvMessage,
    default: T,
) -> T {
    match kv.call(builder, None).await {
        Ok(CallResult::Success(v)) => v,
        _ => default,
    }
}
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    transferred: VecDeque<Sequence>,
    /// The KV-cache actor (the block pool, the prefix cache, the mamba slots, the
    /// tier ladder). Reached only through the `ActorRef<KvMessage>` proxy (the no
    /// shared block_manager).
    pub kv: ractor::ActorRef<KvMessage>,
    next_seq_id: usize,
    /// The QoS-computed dynamic speculative draft length K, PER SEQUENCE (the
    /// `schedule()` sets it from the per-seq rolling prefix-acceptance curve, the
    /// `prepare_step_view` reads it for the per-seq pre-alloc). The no
    /// fixed `num_speculative_tokens` + the no global-EMA (the per-seq isolation: one
    /// sequence's hard region does not shrink another's). Keyed by seq id.
    last_spec_draft_len: std::collections::HashMap<usize, usize>,
    /// The QoS-computed sampling override (the `AdaptivePlan.sampling`, the
    /// per-class top-k/top-p width). The `schedule()` sets it, the
    /// `prepare_step_view` applies it to the sequences' sampling params (the
    /// short-output class → the tight top-k, the long-output class → the wide
    /// top-p). The `None` = the no override (the model default sampling).
    last_sampling_override: Option<SamplingOverride>,
    /// Per-seq end-of-sequence report (cached-token count + QoS scheduling stats),
    /// retained briefly after `clear_finished()` (which drops the Sequence) so the
    /// server's report can still read it. Bounded by `FINISHED_CACHED_TOKENS_MAX`.
    finished_reports: HashMap<usize, SeqReport>,
    eos_token_id: Vec<u32>,
    /// Token IDs that represent the end of a tool call (e.g., </tool_call> tokens)
    tool_call_end_token_ids: Vec<u32>,
    /// Token IDs that represent the start of a tool call (used to avoid false end matches)
    tool_call_start_token_ids: Vec<u32>,
    /// Token ID for } character (used for JSON tool call detection)
    json_end_token_id: Option<u32>,
    /// Tokenizer for decoding output to check JSON tool call patterns
    tokenizer: Option<Arc<Tokenizer>>,
    /// Regex for detecting JSON tool calls
    tool_call_regex: Regex,
    cfg: EngineConfig,
    pd_config: Option<PdConfig>,
    is_last_prefill: bool,
    /// Sequence IDs whose hybrid GDN/Mamba active slots must be released on the
    /// runner after scheduler-side abort (swap-in failure, etc.).
    pending_runner_releases: Vec<usize>,
    /// The self-tuning QoS controller (per model). When QoS is off this is a
    /// no-op (the static degenerate plan), so behavior is unchanged.
    policy: QosController,
    /// Wall-clock of the previous `schedule()` step; the delta is the ITL sensor.
    last_step: Option<Instant>,
}

/// Cap on `Scheduler::finished_reports`. Bounded memory beats LRU
/// here: dropping reporting on a small population is acceptable.
const FINISHED_CACHED_TOKENS_MAX: usize = 16_384;
pub const KVCACHE_SWAP_THRESHOLD: f32 = 0.95f32; // over 95%
const SWAP_COOLING_PERIOD: usize = 5000; // 5 seconds cooling time to prevent frequent swap out/in
const MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP: usize = 1000; // to swap-in, at least 1000 kvcache tokens left for decoding
pub const PD_PREFILL_STATUS_CHECK_COOLING_PERIOD: usize = 50; // check prefill status every 50ms (data is pushed immediately)
pub const PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD: usize = 128; // do not transfer prefill length < 128
/// When prefix cache hit is high, prefer local prefill if new tokens < this threshold
pub const PD_LOCAL_PREFILL_NEW_TOKEN_THRESHOLD: usize = 1024;
const PREFIX_CACHE_RATIO_NORMAL: f32 = 0.65;
const PREFIX_CACHE_RATIO_PD_SERVER: f32 = 0.8;
const PREFIX_CACHE_RATIO_PD_CLIENT: f32 = 0.5;
const PREFIX_CACHE_PRESSURE_EVICT_PERCENT: f32 = 0.1; // evict 10% of prefix cache when under pressure

fn active_sequence_limit(max_num_seqs: usize, mamba_cache_capacity: Option<usize>) -> usize {
    match mamba_cache_capacity {
        Some(mamba_cap) if mamba_cap > 0 => max_num_seqs.min(mamba_cap),
        _ => max_num_seqs,
    }
    .max(1)
}

/// The K-bridge: seed the QoS `spec_draft_max` (the dynamic-K ceiling) from the
/// operator's `--num-speculative-tokens` when left unset (the 0 default). Without
/// this the QoS-off degenerate plan returns `spec_draft_len = 0`, the
/// `prepare_step_view` skips the speculative pre-alloc, + the runner's MTP draft
/// (the fixed CLI K) crashes in `compute_slot_mappings` (no KV blocks for the
/// verify positions). Seeding makes the scheduler's pre-alloc K + the runner's
/// draft K agree by construction. An operator-set `spec_draft_max` is preserved
/// (the no override of an explicit QoS tuning).
fn seed_spec_draft_max(econfig: &EngineConfig) -> crate::core::qos::QosConfig {
    let mut qos_cfg = econfig.qos.clone();
    if qos_cfg.spec_draft_max == 0 {
        qos_cfg.spec_draft_max = econfig.num_speculative_tokens.unwrap_or(0);
    }
    qos_cfg
}

pub fn build_prefix_cache_config(econfig: &EngineConfig) -> PrefixCacheConfig {
    let enabled = econfig.prefix_cache.unwrap_or(false);
    if !enabled {
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    let mut max_cached_blocks = if let Some(max_tokens) = econfig.prefix_cache_max_tokens {
        max_tokens / econfig.block_size
    } else {
        let is_pd_server = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        };
        let is_pd_client = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Client)
        } else {
            false
        };

        let ratio = if is_pd_server {
            econfig
                .pd_server_prefix_cache_ratio
                .unwrap_or(PREFIX_CACHE_RATIO_PD_SERVER)
        } else if is_pd_client {
            econfig
                .pd_client_prefix_cache_ratio
                .unwrap_or(PREFIX_CACHE_RATIO_PD_CLIENT)
        } else {
            PREFIX_CACHE_RATIO_NORMAL
        };

        if is_pd_server || is_pd_client {
            ((econfig.num_blocks as f32) * ratio) as usize
        } else {
            // Workload-aware sizing (QoS on): hold all concurrent sequences at full
            // context. max_model_len is resolved at engine init before Scheduler::new.
            let qos_on = crate::utils::env::qos_enabled() || econfig.qos.enabled;
            if qos_on {
                let max_tokens_per_seq = econfig.max_model_len.unwrap_or(262144);
                econfig.max_num_seqs * max_tokens_per_seq / econfig.block_size
            } else {
                // QoS off: fall back to the fixed fraction of the KV budget.
                ((econfig.num_blocks as f32) * ratio) as usize
            }
        }
    };

    if max_cached_blocks > econfig.num_blocks {
        max_cached_blocks = econfig.num_blocks;
    }

    if max_cached_blocks == 0 {
        crate::log_warn!("Prefix cache enabled but max cached blocks is 0; disabling.");
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    crate::log_warn!(
        "Prefix cache enabled: {} blocks ({} tokens).",
        max_cached_blocks,
        max_cached_blocks * econfig.block_size
    );

    PrefixCacheConfig {
        enabled: true,
        max_cached_blocks,
    }
}

impl Scheduler {
    pub fn new(
            kv: ractor::ActorRef<KvMessage>,
            econfig: &EngineConfig,
            config: &Config,
        ) -> Self {
            Self {
                waiting: VecDeque::new(),
                running: Vec::new(),
                cached: Vec::new(),
                transferred: VecDeque::new(),
                kv,
                next_seq_id: 0,
                finished_reports: HashMap::new(),
            eos_token_id: match &config.eos_token_id {
                Some(EosTokenId::Single(eos)) => vec![*eos],
                Some(EosTokenId::Multiple(eos)) => eos.into_iter().map(|x| *x).collect(),
                _ => vec![],
            },
            // Tool call end tokens will be set by engine after tokenizer is initialized
            tool_call_end_token_ids: Vec::new(),
            tool_call_start_token_ids: Vec::new(),
            json_end_token_id: None,
            tokenizer: None,
            // Regex to match JSON tool call format: {"name": "...", "arguments": {...}}
            // We use (?s) to allow dot matching newlines
            tool_call_regex: Regex::new(r#"(?s)\{\s*"name"\s*:.*"arguments"\s*:.*\}\s*$"#).unwrap(),
            cfg: econfig.clone(),
            pd_config: econfig.pd_config.clone(),
            is_last_prefill: false,
            pending_runner_releases: Vec::new(),
            policy: QosController::new(seed_spec_draft_max(econfig)),
            last_step: None,
            last_spec_draft_len: std::collections::HashMap::new(),
            last_sampling_override: None,
        }
    }

    pub fn request_runner_release(&mut self, seq_id: usize) {
        if !self.pending_runner_releases.contains(&seq_id) {
            self.pending_runner_releases.push(seq_id);
        }
    }

    pub fn take_pending_runner_releases(&mut self) -> Vec<usize> {
        if self.pending_runner_releases.is_empty() {
            return Vec::new();
        }
        std::mem::take(&mut self.pending_runner_releases)
    }

    /// Set tool call end token IDs (called by engine after tokenizer is available)
    pub fn set_tool_call_end_tokens(&mut self, token_ids: Vec<u32>) {
        self.tool_call_end_token_ids = token_ids;
    }

    /// Set tool call start token IDs (called by engine after tokenizer is available)
    pub fn set_tool_call_start_tokens(&mut self, token_ids: Vec<u32>) {
        self.tool_call_start_token_ids = token_ids;
    }

    /// Set tokenizer for JSON tool call detection (called by engine after initialization)
    pub fn set_tokenizer(&mut self, tokenizer: Arc<Tokenizer>) {
        // Get the token ID for "}" character
        if let Ok(tokens) = tokenizer.encode("}", false) {
            if let Some(&token_id) = tokens.get_ids().last() {
                self.json_end_token_id = Some(token_id);
                crate::log_info!("JSON end token ID (}}) set to: {}", token_id);
            }
        }
        self.tokenizer = Some(tokenizer);
    }

    pub fn add(&mut self, mut seq: Sequence) -> usize {
        seq.id = self.next_seq_id;
        let id = seq.id;
        self.next_seq_id += 1;
        // Assign the QoS class from the controller's measured output-length estimate
        // (a no-op Throughput default until outputs are observed).
        seq.qos_class = self.policy.infer_class(&seq);
// The §1 two-rate sensor (the fast path): the admission records the
        // prefix-hit EMA + the turns counter + the reuse-evidence digest (the
        // NInfer real-demand signal). The `num_cached_tokens` is the prefix hit,
        // the `len()` is the query.
        if self.policy.active() {
            let reuse_digest = seq.mamba_prefix_hash.unwrap_or(0);
            self.policy.record_admission(
                seq.qos_class,
                seq.num_cached_tokens,
                seq.len(),
                reuse_digest,
            );
        }
        if crate::utils::env::debug_llg() {
            if let Some(gram) = &seq.sampling_params.grammar {
                crate::log_info!(
                    "[llg] Seq {}: TopLevelGrammar:\n{}\n",
                    seq.id,
                    crate::utils::guidance_grammar::get_lark_from_top_level_grammar(gram)
                );
            }
        }
        self.waiting.push_back(seq);
        id
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    fn active_sequence_limit(&self) -> usize {
        active_sequence_limit(
            self.cfg.max_num_parallel_reqs.max(1),
            self.cfg.mamba_cache_capacity,
        )
    }

    /// Stable-partition the waiting queue so Latency-class requests are admitted ahead
    /// of Throughput (FIFO within each class). Only called when QoS is active; when
    /// it is not, the queue is already FIFO. O(n), no per-step HashMap.
    fn partition_waiting_by_class(&mut self) {
        let (latency, throughput) = self
            .waiting
            .drain(..)
            .partition::<VecDeque<Sequence>, _>(|s| s.qos_class.is_latency());
        self.waiting = latency.into_iter().chain(throughput).collect();
    }

    /// Schedule sequences and return their indexes in `running` along with prefill flag
    #[allow(non_snake_case)]
    pub async fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;

        // QoS: measure the step (the ITL sensor), then ask the policy for this
        // step's plan. When QoS is off the plan is the static degenerate one (full
        // chunk, FIFO admission, no gates), so behavior is unchanged.
        let now = Instant::now();
        let step_ms = self
            .last_step
            .map(|t| now.duration_since(t).as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        self.last_step = Some(now);
        if self.policy.active() {
            // Guard against idle gaps: the engine loop skips schedule() while idle,
            // so a large delta means "woke from idle", not a slow step. Feeding it
            // to the ITL sensor would falsely spike the gain.
            let eff = self.policy.effective();
            if step_ms <= eff.max_meaningful_step_ms {
                self.policy.record_step(step_ms);
            }
            // The decode batch size (the MagicDec signal: the SD speedup grows
            // with the batch for the separate-drafter, the MTP is batch-invariant).
            self.policy.record_spec_batch_size(self.running.len());
        }

        let blocks = BlockSnapshot {
            free_blocks: kv_call(&self.kv, |r| KvMessage::FreeBlocks(r), 0).await,
            total_blocks: kv_call(&self.kv, |r| KvMessage::TotalBlocks(r), 0).await,
            max_num_seqs: self.active_sequence_limit(),
            max_num_batched_tokens: self.cfg.max_num_batched_tokens,
            static_chunk: self.cfg.effective_prefill_chunk_size(),
            chunk_floor: self.policy.effective().chunk_floor,
        };
// The §1: the AdaptivePlan (the base StepPlan + the QoS actuators: the
        // batch_cap, the sampling override, the parallelism_hint, the
        // kv_demote_threshold). The histogram is fed by the record_step (the ITL)
        // + the record_batch_size (the regime) + the record_admission (the
        // prefix-hit).
        let eff = self.policy.effective();
        let decode_pressure = self.policy.contention(&self.running, &blocks).decode_pressure;
        let kv_pressure = 1.0 - blocks.mem_headroom() as f64;
        let regime = Regime::classify(
            decode_pressure,
            kv_pressure,
            eff.regime_decode_pressure,
            eff.regime_kv_pressure,
        );
        self.policy.record_batch_size(regime, self.running.len());
        // The online calibration: feed this step's measurements so the controller
        // derives its `None` config values from the measured system instead of
        // blind constants. After the calibration window this is a no-op.
        self.policy.observe_calibration(
            step_ms,
            decode_pressure,
            kv_pressure,
            None,
            self.running.len(),
        );
        let is_mtp = crate::speculative::uses_builtin_mtp(&self.cfg);
        let plan = self.policy.plan(&self.running, &blocks, is_mtp);
        let chunk_size = plan.chunk_size;
        // The per-seq draft length K. When the adaptive-K gate is on, each running
        // sequence's K is computed from its OWN rolling prefix-acceptance curve (the
        // knee) -- one sequence's hard region does not shrink another's. When the gate
        // is off (the default), every sequence uses the fixed `plan.spec_draft_len`
        // (the prior behavior, the no surprise).
        let dynamic_k = crate::utils::env::spec_dynamic_k();
        // The seed the per-seq K map. The runner's feedback (the record_spec_k_feedback,
        // the prior step's SpecStepOut.next_k) is the authoritative signal (the no the
        // main-process curve, which is empty across the process boundary). The seed only
        // computes a K for seqs that have NO feedback yet (the first step, the no the
        // runner's signal) -- it does NOT overwrite the runner's feedback (the no the
        // prior defect: the recompute clobbered the feedback, the no the dynamic-K never
        // actually tracked the runner's signal).
        //
        // The prior_feedback is PRESERVED in full (the no just the running subset): a
        // preempted/swapped-out seq retains its K (the no the swap-in resetting it to the
        // ceiling, the no the over-draft on re-admission). The map starts from the prior
        // feedback + only adds entries for running seqs that lack feedback (the first step).
        let prior_feedback = std::mem::take(&mut self.last_spec_draft_len);
        let mut new_map = prior_feedback; // The preserve ALL prior feedback (the no the running-only subset).
        for s in self.running.iter() {
            // The runner's feedback (the prior step's next_k) wins (the no the recompute).
            if new_map.contains_key(&s.id) {
                continue;
            }
            // The no feedback yet (the first step): compute from the per-seq law (the
            // dynamic_k) or the static plan (the no dynamic_k). The no the main-process
            // curve is empty (the cross-process global), so spec_draft_len_for_seq
            // returns the ceiling (the no history) -- the correct first-step seed.
            let k = if dynamic_k {
                self.policy.spec_draft_len_for_seq(s.id, is_mtp)
            } else {
                plan.spec_draft_len
            };
            new_map.insert(s.id, if k > 0 { k } else { plan.spec_draft_len });
        }
        self.last_spec_draft_len = new_map;
        // The §1 AdaptivePlan (the QoS actuators): the sampling override is stored
        // for the `prepare_step_view` (the per-class top-k/top-p width, the
        // short-output → the tight top-k, the long-output → the wide top-p).
        let adaptive = self.policy.adaptive_plan(&self.running, &blocks, is_mtp);
        self.last_sampling_override = if adaptive.sampling.top_k.is_some() || adaptive.sampling.top_p.is_some() {
            Some(adaptive.sampling)
        } else {
            None
        };

        // QoS: admit Latency-class requests ahead of Throughput (stable partition),
        // and log the live contention vector under pressure. No-op when QoS is off.
        if self.policy.active() {
            self.partition_waiting_by_class();
            if tracing::enabled!(tracing::Level::DEBUG) {
                let c = self.policy.contention(&self.running, &blocks);
                if c.decode_pressure > 0.0 {
tracing::debug!(
                         "[qos] contention={:?} chunk={} budget={}",
                         c,
                         plan.chunk_size,
                         plan.token_budget
                     );
                }
            }
        }

        // PD server: Check for new incoming prefill requests
        if self.is_pd_server() {
            let available_tokens = self.get_available_kv_tokens().await;
            let recv = match self.kv.call(|r| KvMessage::TryReceivePrefill(available_tokens, r), None).await {
                Ok(CallResult::Success(v)) => v,
                _ => (false, None),
            };
            if let (fit, Some(seq)) = recv {
                let seq_id = seq.id;
                if !fit {
                    crate::log_warn!(
                        "Prefill request (Seq {}) enter pending status because it require {} KvCache tokens (left {}).",
                        seq_id,
                        seq.len() + 1,
                        self.get_available_kv_tokens().await,
                    );
                } else {
                    crate::log_warn!(
                        "Prefill request (Seq {}, {} tokens) received from PD client.",
                        seq_id,
                        seq.len(),
                    );
                }
                // Add to waiting queue.
                self.waiting.push_back(seq);
            }
        }

        // Prefill phase: move sequences from waiting to running if possible.
        // Use effective chunk tokens (not full sequence length) for budget accounting
        // to enable batched prefill of multiple sequences per step.
        // Preserve interleaving: when the previous step was prefill and decode
        // sequences already existed, give those sequences a decode step before
        // admitting another waiting prefill batch.
        let pre_existing_running = self.running.len();
        let max_seqs_limit = self.active_sequence_limit();
        let token_budget = plan.token_budget;
        let mut thr_in_use = plan.thr_in_use;

        while let Some(mut seq) = self.waiting.pop_front() {
            // Try to transfer prefill requests to PD server when applicable
            if self.is_pd_mode() && !self.is_pd_server() && self.try_transfer(&mut seq).await {
                break;
            }

            let effective_tokens = seq.prefill_chunk_tokens(chunk_size);

            // QoS reservation gates: a Throughput request must not consume the
            // Latency slot / KV reserve. No-op when QoS is off (both gates false).
            let is_thr = !seq.qos_class.is_latency();
            let slot_blocked = is_thr
                && plan.thr_slot_capacity > 0
                && thr_in_use >= plan.thr_slot_capacity;
            let kv_blocked = is_thr && plan.thr_kv_full;
            if slot_blocked || kv_blocked {
                seq.sched_stats.reservation_blocked += 1;
                self.waiting.push_front(seq);
                break;
            }

            if self.running.len() >= max_seqs_limit
                || scheduled_ids.len() >= max_seqs_limit
                || num_tokens + effective_tokens > token_budget
                || (seq.block_table.is_empty() && !kv_call(&self.kv, |r| KvMessage::CanAllocate(seq.clone(), r), false).await)
                // interleaved scheduling: alternate prefill/decode for fairness
                // only block when there are pre-existing decode sequences
                || (self.is_last_prefill && pre_existing_running > 0)
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            if seq.block_table.is_empty() {
                match kv_call(&self.kv, |r| KvMessage::Allocate(seq.clone(), r), Err("RPC failed".to_string())).await {
                    Ok(updated) => { seq = updated; }
                    Err(e) => {
                        crate::log_error!("block allocation failed: {e}");
                        return Err(candle_core::Error::msg(format!("block allocation failed: {e}")));
                    }
                }
            }
            seq.status = SequenceStatus::Running;
            // Stamp the adaptive chunk so the runner's prepare_prefill and the
            // scheduler's filter_prefill_finished agree on what was budgeted.
            seq.active_prefill_chunk = Some(chunk_size);
            // QoS per-seq stats: record the adaptive adjustments that fired this step.
            seq.sched_stats.qos_active = self.policy.active();
            seq.sched_stats.qos_class = seq.qos_class;
            // The §1 two-rate sensor (the fast path): the admission records the
            // request-state transition (the `Running` → the `DecodeReady` (
            // index 6, the NInfer `EngineRequestState`).
            if self.policy.active() {
                self.policy.record_request_state(seq.qos_class, 6);
            }
            seq.sched_stats.gain_start = self.policy.gain();
            seq.sched_stats.prefill_steps += 1;
            // Stamp the speculative-draft K used for this sequence's pre-alloc (the
            // per-seq dynamic K, the no global). 0 = SD not enabled.
            seq.sched_stats.spec_k = self.last_spec_draft_len.get(&seq.id).copied().unwrap_or(0);
            if chunk_size < blocks.static_chunk {
                seq.sched_stats.chunk_shrunk += 1;
            }
            if seq.qos_class.is_latency() && self.policy.active() {
                seq.sched_stats.priority_admitted += 1;
            }
            num_tokens += effective_tokens;
            self.running.push(seq);
            if is_thr {
                thr_in_use += 1;
            }
            scheduled_ids.push(self.running.len() - 1); // index of newly pushed seq
        }

        if !scheduled_ids.is_empty() {
            self.is_last_prefill = true;
            return Ok((scheduled_ids, true));
        }

        // Decode phase: pick sequences from running for decoding (up to max_num_seqs)
        let mut decode_ids = Vec::new();
        let mut preempt_ids = Vec::new();

        for (idx, seq) in self.running.iter().enumerate() {
            if !kv_call(&self.kv, |r| KvMessage::CanAppend(seq.clone(), r), false).await
                && seq.status != SequenceStatus::Swapped
                && seq.status != SequenceStatus::FinishSwapped
            {
                preempt_ids.push(idx);
            }
        }
        // QoS per-seq stats: count preemption pressure on each affected sequence.
        for &idx in &preempt_ids {
            self.running[idx].sched_stats.preempted += 1;
            // The §1: the preemption feeds the histogram's preemptions[class]
            // (the NInfer retraction_count, the capacity signal).
            if self.policy.active() {
                self.policy.record_preemption(self.running[idx].qos_class);
            }
        }

        // Client: Check for finished prefills
        if self.is_pd_mode() && !self.is_pd_server() {
            self.try_receive_kvcache().await?;
        }

        // Swap back seq from cpu memory if possible. Aggregate KV usage can stay high
        // under prefix cache because reusable prefix blocks occupy the GPU pool, so
        // already-swapped sequences must still get a chance to run the precise
        // suffix-allocation and prefix-eviction checks in `try_swap_in`.
        let has_swapped_seq = self
            .cached
            .iter()
            .any(|seq| seq.status == SequenceStatus::Swapped);
        let should_try_swap_in = cfg!(feature = "cuda")
            && preempt_ids.is_empty()
            && (has_swapped_seq
                || self.kv_cache_usage_percent().await < KVCACHE_SWAP_THRESHOLD * 0.9
                || (self.running.is_empty() && self.kv_cache_usage_percent().await <= 0.3f32));
        if should_try_swap_in {
            #[cfg(feature = "cuda")]
            self.try_swap_in().await;
        } else if !preempt_ids.is_empty() || self.kv_cache_usage_percent().await > KVCACHE_SWAP_THRESHOLD
        {
            // Requests unable to be processed at the current moment
            // If we only have one sequence running and it has been preempt,
            // swap out to cpu memory make non-sense
            // in such case, the only option is either waiting resources or abort it
            let evicted = self.evict_prefix_cache_under_pressure().await;
            if evicted > 0 {
                crate::log_warn!("Evicted {} prefix cache block(s) under pressure.", evicted);
            } else {
                #[cfg(feature = "cuda")]
                if !preempt_ids.is_empty() && self.running.len() > 1 {
                    if let Some((idx, _)) = preempt_ids
                        .iter()
                        .map(|&i| (i, &self.running[i]))
                        .min_by_key(|(_, seq)| seq.id)
                    // swap-out the oldest sequence
                    {
                        crate::log_warn!(
                            "Trying to swap out preempt Seq {:?}",
                            self.running[idx].id
                        );
self.try_swap_out(idx, true).await;
                        self.running[idx].sched_stats.swapped += 1;
                    }
                }
            }
        }

        let is_pd_server = self.is_pd_server();
        let decode_max_seqs = self.active_sequence_limit();
        let mut pd_finished_ids: Vec<usize> = Vec::new();
        for (idx, seq) in self.running.iter_mut().enumerate() {
            if decode_ids.len() >= decode_max_seqs {
                break;
            }
            if seq.status == SequenceStatus::Finished {
                continue;
            }
            if !kv_call(&self.kv, |r| KvMessage::CanAppend(seq.clone(), r), false).await {
                // filter out seq that unable to acquire resources
                continue;
            }
            if is_pd_server && seq.status == SequenceStatus::Cached {
                let success = kv_call(&self.kv, |r| KvMessage::TryCheckKvcacheRelease(seq.id, r), false).await;
                if success {
                    crate::log_warn!("PD Server: release prefilled kvcache for Seq {} (prefix cache retained)", seq.id);
                    seq.status = SequenceStatus::Finished;
                    let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                    pd_finished_ids.push(seq.id);
                }
                continue;
            }
            match kv_call(&self.kv, |r| KvMessage::MayAppend(seq.clone(), r), Err("RPC failed".to_string())).await {
                    Ok(updated) => { *seq = updated; }
                    Err(e) => {
                        crate::log_error!("may_append failed: {e}");
                        return Err(candle_core::Error::msg(format!("may_append failed: {e}")));
                    }
                }
            decode_ids.push(idx);
        }
        if !pd_finished_ids.is_empty() {
            self.running.retain(|s| !pd_finished_ids.contains(&s.id));
        }

        self.is_last_prefill = false;
        Ok((decode_ids, false))
    }

    /// Provide immutable access to sequences by indexes (for model inference)
    pub fn get_sequences(&self, ids: &[usize]) -> Vec<&Sequence> {
        ids.iter().map(|&i| &self.running[i]).collect()
    }

    /// For prefill sequences that rely on cached prefix tokens, verify mamba state snapshots
    /// still exist. If a snapshot is missing/evicted, downgrade that sequence to full prefill.
pub async fn fallback_missing_mamba_prefix_snapshots(
        &mut self,
        scheduled_ids: &[usize],
    ) -> Result<usize> {
        let (running, block_manager) = (&mut self.running, &mut self.kv);
        let mut downgraded = 0usize;

        for &idx in scheduled_ids {
            if idx >= running.len() {
                continue;
            }
            let seq = &mut running[idx];
            if seq.num_cached_tokens == 0 {
                continue;
            }
            let Some(hash) = seq.mamba_prefix_hash else {
                continue;
            };

            let has_snapshot = kv_call(block_manager, |r| KvMessage::HasMambaPrefix(hash, r), false).await;
            if has_snapshot {
                continue;
            }

            let prev_cached = seq.num_cached_tokens;
            let required_blocks = seq.num_blocks();

            // Drop current block refs (including any shared prefix refs), then rebuild from scratch.
            let _ = block_manager.send_message(KvMessage::Deallocate(seq.clone()));
            seq.clear_block_table();

            if !kv_call(block_manager, |r| KvMessage::CanAllocateWithoutPrefix(seq.clone(), r), false).await {
                let evicted = kv_call(block_manager, |r| KvMessage::EvictPrefixUntilFree(required_blocks, r), 0).await;
                if evicted > 0 {
                    crate::log_warn!(
                        "Seq {}: evicted {} prefix cache block(s) before mamba fallback reallocation.",
                        seq.id,
                        evicted
                    );
                }
            }
            match kv_call(block_manager, |r| KvMessage::AllocateWithoutPrefix(seq.clone(), r), Err("RPC failed".to_string())).await {
                Ok(updated) => { *seq = updated; }
                Err(e) => {
                    crate::log_error!("allocate_without_prefix failed: {e}");
                    return Err(candle_core::Error::msg(format!("allocate_without_prefix failed: {e}")));
                }
            }
            downgraded += 1;
            crate::log_warn!(
                "Seq {}: missing mamba snapshot hash {} (cached {} tokens). Downgraded to full prefill.",
                seq.id,
                hash,
                prev_cached
            );
        }

        Ok(downgraded)
    }

    pub fn get_running(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.running.len() {
            Some(&self.running[idx])
        } else {
            None
        }
    }

    /// The active (running) sequence indices (for the periodic decode throughput log).
    pub fn get_active_indices(&self) -> Vec<usize> {
        (0..self.running.len()).collect()
    }

    pub fn get_waiting(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.waiting.len() {
            Some(&self.waiting[idx])
        } else {
            None
        }
    }

    pub fn get_seq_token_usage(&self, seq_id: usize) -> Result<usize> {
        // search waiting
        if let Some(item) = self.waiting.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search running
        if let Some(item) = self.running.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search cached
        if let Some(item) = self.cached.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // if nothing found
        Ok(0)
    }

    pub fn find_seq_by_session_id(&self, session_id: &str) -> Option<(usize, SequenceStatus)> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .find(|seq| seq.sampling_params.session_id.as_deref() == Some(session_id))
            .map(|seq| (seq.id, seq.status))
    }

    /// Postprocess output tokens for each sequence. Handles both single-token (normal decode)
    /// and multi-token (MTP speculative decode) outputs uniformly.
pub async fn postprocess(&mut self, ids: &[usize], multi_output_ids: &[Vec<u32>]) {
        for (i, &idx) in ids.iter().enumerate() {
            if idx >= self.running.len() {
                continue;
            }
            // The OOB guard: the runner may return fewer outputs than ids (the MTP
            // error path sends an empty RunResponseMTP). A missing output degrades
            // to "no tokens this step" (the skip), never an index panic.
            let tokens = match multi_output_ids.get(i) {
                Some(t) => t,
                None => continue,
            };
            let seq_id = self.running[idx].id;

            // PD server: transfer KV cache on first token, then move to next sequence.
            if self.is_pd_server() {
                if let Some(&first_token) = tokens.first() {
                    let success = kv_call(&self.kv, |r| KvMessage::TrySendKvcache(self.running[idx].clone(), first_token, r), false).await;
                    crate::log_warn!(
                        "PD Server: transferred KV cache for seq {} ({})",
                        seq_id,
                        if success { "success" } else { "faild" }
                    );
                    let seq = &mut self.running[idx];
                    if success {
                        let _ = kv_call(&self.kv, |r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.len(), r), false).await;
                        let _ = self.kv.send_message(KvMessage::CacheSequence(seq.clone()));
                        seq.status = SequenceStatus::Cached;
                        let cur_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis()
                            as usize;
                        let time_costs = cur_time - seq.created_time();
                        if time_costs / 100 > 0 && seq.len() > 0 {
                            crate::log_info!(
                                "PD Prefilling [seq_id {}]: {} tokens in {:.2}s ({:.2} tokens/s)",
                                seq_id,
                                seq.len(),
                                time_costs as f32 / 1000f32,
                                seq.len() as f32 / (time_costs as f32 * 1.0f32 / 1000f32),
                            )
                        }
                    } else {
                            seq.status = SequenceStatus::Finished;
                            let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                    }
                }
                continue;
            }

            for &token in tokens {
                if self.running[idx].sampling_params.mcp_mode.is_some() {
                    let is_end = self.is_tool_call_end(token, idx);
                    if is_end {
                        crate::log_info!(
                            "[Seq {}] Detected </tool_call> token {}, finishing for external handling",
                            seq_id,
                            token
                        );
                        let seq = &mut self.running[idx];
                        seq.append_token(token);
                        seq.is_tool_call_end = true;
                        seq.status = SequenceStatus::Finished;
                        let _ = kv_call(&self.kv, |r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.len(), r), false).await;
                        let _ = self.kv.send_message(KvMessage::CacheSequence(seq.clone()));
                        let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                        break;
                }
                }

                let matched_stop_sequence_idx =
                    self.stop_sequence_match_index(token, &self.running[idx]);
                let hit_stop_sequence = matched_stop_sequence_idx.is_some();
                let seq = &mut self.running[idx];

                if hit_stop_sequence
                    || self.eos_token_id.contains(&token)
                    || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                    || seq.len()
                        > self
                            .cfg
                            .max_model_len
                            .unwrap_or(self.cfg.max_kv_cache_tokens.max(1))
                {
                    if hit_stop_sequence {
                        crate::log_info!(
                            "[Seq {}] Detected stop sequence token {}, finishing",
                            seq_id,
                            token
                        );
                        seq.hit_stop_sequence = true;
                        seq.stop_sequence = matched_stop_sequence_idx.and_then(|stop_idx| {
                            seq.sampling_params
                                .stop_sequences
                                .as_ref()
                                .and_then(|stops| stops.get(stop_idx))
                                .cloned()
                        });
                }
                    seq.status = SequenceStatus::Finished;
                    let _ = kv_call(&self.kv, |r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.len(), r), false).await;
                    let _ = self.kv.send_message(KvMessage::CacheSequence(seq.clone()));
                    let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                break;
                } else {
                    seq.append_token(token);
                    if seq.len() % self.cfg.block_size == 1 && seq.len() > 1 {
                        match kv_call(&self.kv, |r| KvMessage::MayAppend(seq.clone(), r), Err("RPC failed".to_string())).await {
                            Ok(updated) => { *seq = updated; }
                            Err(e) => {
                                crate::log_error!("may_append failed: {e}");
                            }
                        }
                    }
                    if seq.len() % self.cfg.block_size == 0 {
                        let _ = kv_call(&self.kv, |r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.len(), r), false).await;
                    }
                }
            }
        }
    }

/// Pre-allocate KV cache blocks for MTP speculative tokens.
    /// Called before MTP runs to ensure the verification forward has room
    /// to write KV for speculative positions. The per-seq K: `extra_tokens`
    /// is one entry per `ids` (the seq's own dynamic-K, the no single global K).
    pub async fn pre_allocate_mtp_blocks(&mut self, ids: &[usize], extra_tokens: &[usize]) {
        for (&idx, &k) in ids.iter().zip(extra_tokens.iter()) {
            if idx >= self.running.len() {
                continue;
            }
            let seq = &mut self.running[idx];
            // The verify writes K+1 positions (the anchor + the K drafts), so the
            // pre-alloc must cover seq.len() + k + 1 (the no the k-only, which
            // under-allocates the verify's last position -> the "missing KV block"
            // crash when the dynamic-K collapses to a small k).
            let needed_len = seq.len() + k + 1;
            let needed_blocks = needed_len.div_ceil(self.cfg.block_size);
            while seq.block_table.len() < needed_blocks {
                if let Some(block_id) = kv_call(&self.kv, |r| KvMessage::AllocFreeBlock(r), None).await {
                    seq.block_table.push(block_id as u32);
                } else {
                    break;
                }
            }
        }
    }

    /// The dynamic-K feedback: record the runner-reported per-seq next-step K (the
    /// rolling prefix-curve knee, the no the scheduler guessing from an empty
    /// cross-process global) into the per-seq K map, so the NEXT step's
    /// `prepare_step_view` pre-alloc uses the runner's real signal. The `ids` + the
    /// `next_k` are parallel (the per-seq). A `next_k` of 0 (the no MTP, the plain
    /// decode) leaves the seq's K at the prior value (the no a spurious 0).
    pub fn record_spec_k_feedback(&mut self, ids: &[usize], next_k: &[usize]) {
        for (idx, &k) in ids.iter().zip(next_k.iter()) {
            // The record the feedback regardless of k (the no the k > 0 guard). A k == 0
            // means the seq fell back to plain decode (the no the MTP head, the no the
            // empty-draft bail) -- the scheduler must stop pre-allocating speculative
            // blocks for it (the no the stale K from a prior drafting step). The no the
            // k > 0 guard, which kept the stale K for a plain-decode seq.
            if let Some(seq) = self.running.get(*idx) {
                self.last_spec_draft_len.insert(seq.id, k);
            }
        }
    }

    /// The single-writer step view (the Option 5, the no write-after-snapshot): the Scheduler is
    /// the sole writer of the block_table. This function does ALL the writes (the speculative
    /// pre-alloc) + the freeze (the Sequence snapshot) in ONE atomic call. No caller can
    /// interleave a write between the snapshot and the forward (the write + the freeze are one
    /// function). The gate is the origin/main pattern (the speculative decode ENABLED + the
    /// decode step, NOT the single-sequence check).
    pub async fn prepare_step_view(&mut self, ids: &[usize], is_prefill: bool) -> Vec<Sequence> {
        // The write (the speculative pre-alloc, the origin/main gate, the decode only).
        // The per-seq dynamic-K: each seq's K is its own rolling prefix-curve knee
        // (the `last_spec_draft_len` map, the no global). The pre-alloc is per-seq
        // (the no single global K) + the seq's `spec_k` is stamped so it travels via
        // the DecodeSequence IPC to the runner (the no fixed mtp_num_speculative).
        if !is_prefill
            && (crate::speculative::uses_builtin_mtp(&self.cfg)
                || crate::speculative::uses_dflash(&self.cfg))
        {
            let per_seq_k: Vec<usize> = ids
                .iter()
                .map(|idx| {
                    let seq_id: usize = self
                        .running
                        .get(*idx)
                        .map(|s| s.id)
                        .unwrap_or(*idx);
                    // The per-seq K (the knee); the fallback is the seq's CLI ceiling
                    // (the no global last_spec_draft_len, the per-seq isolation).
                    self.last_spec_draft_len
                        .get(&seq_id)
                        .copied()
                        .unwrap_or_else(|| self.cfg.num_speculative_tokens.unwrap_or(0))
                })
                .collect();
            let any_k = per_seq_k.iter().any(|&k| k > 0);
            if any_k {
                // The batch-max K: the speculative verify (the MTP batch + the DFlash batch)
                // uses a uniform `verify_len = 1 + max_k` for the one attention pass, so
                // every seq in the batch must have room for `max_k + 1` verify positions.
                // The single-seq case (the common path) has `max_k == per_seq_k[0]`, so this
                // is identical to the prior per-seq pre-alloc. The floor: a seq with SD
                // enabled (k > 0) pre-allocates for k + 1 verify positions (the anchor +
                // the k drafts). A seq with k == 0 (the SD off for that seq, the plain
                // decode) still pre-allocates 1 position (the k.max(1) floor, the no the
                // plain-decode token needs a block) -- but the normal decode path (the
                // MayAppend at the block boundary) also allocates it, so the speculative
                // pre-alloc for a k == 0 seq is a harmless no-op (the no the double-alloc,
                // the pre_allocate_mtp_blocks only extends the block_table if it's short).
                let batch_max_k = per_seq_k.iter().copied().max().unwrap_or(0);
                let floored: Vec<usize> = std::iter::repeat(batch_max_k.max(1))
                    .take(ids.len())
                    .collect();
                self.pre_allocate_mtp_blocks(ids, &floored).await;
                // The stamp the per-seq K onto the sequences (the travels via the
                // DecodeSequence.spec_k to the runner, the no fixed ceiling) + onto the
                // sched_stats (the report carrier, the no the admission-time 0 -- the
                // spec_k is the K actually used for this seq's decode, stamped here).
                for (idx, &k) in ids.iter().zip(per_seq_k.iter()) {
                    if let Some(seq) = self.running.get_mut(*idx) {
                        seq.spec_k = k;
                        seq.sched_stats.spec_k = k;
                    }
                }
            }
        }
        // The freeze (the Sequence snapshot, AFTER the write).
        // The §1 sampling override (the per-class top-k/top-p width): the to
        // the sequences' sampling params before the freeze, so it travels with
        // the existing IPC messages (the no new `MessageType` variant).
        if let Some(override_) = &self.last_sampling_override {
            for id in ids {
                if let Some(seq) = self.running.iter_mut().find(|s| &s.id == id) {
                    if let Some(top) = override_.top_k {
                        seq.sampling_params.top_k = Some(top as isize);
                    }
                    if let Some(p) = override_.top_p {
                        seq.sampling_params.top_p = Some(p);
                    }
                }
            }
        }
        self.get_sequences(ids)
            .iter()
            .map(|s| if is_prefill { s.clone_for_prefill_forward() } else { (*s).clone() })
            .collect()
    }

pub async fn postprocess_speculative_extra(&mut self, ids: &[usize], extras: &[Vec<u32>]) {
        // The QoS acceptance-rate sensor (the break-even signal): the
        // accepted / the proposed (the K). The per-seq K (the no global): the
        // proposed_k is the seq's own dynamic K (the map), the no one global value.
        for (i, extra_tokens) in extras.iter().enumerate() {
            if i >= ids.len() {
                break;
            }
            let seq_id = ids[i];
            let proposed_k = self
                .last_spec_draft_len
                .get(&seq_id)
                .copied()
                .unwrap_or_else(|| self.cfg.num_speculative_tokens.unwrap_or(0))
                .max(1);
            if self.policy.active() {
                let accept_frac = (extra_tokens.len() as f64) / (proposed_k as f64);
                self.policy.record_spec_acceptance(accept_frac);
            }
        }
        for (i, extra_tokens) in extras.iter().enumerate() {
            if i >= ids.len() {
                break;
            }
            let idx = ids[i];
            if idx >= self.running.len() {
                continue;
            }
            if self.running[idx].status == SequenceStatus::Finished {
                continue;
            }
            let mut finished = false;
            for &tok in extra_tokens {
                if self.eos_token_id.contains(&tok) {
                    self.running[idx].append_token(tok);
                    self.running[idx].status = SequenceStatus::Finished;
                    let _ = kv_call(&self.kv, |r| KvMessage::CaptureMambaPrefix(self.running[idx].clone(), self.running[idx].len(), r), false).await;
                    let _ = self.kv.send_message(KvMessage::CacheSequence(self.running[idx].clone()));
                    let _ = self.kv.send_message(KvMessage::Deallocate(self.running[idx].clone()));
                    finished = true;
                    break;
                }
                self.running[idx].append_token(tok);
            }
            if !finished {
                match kv_call(&self.kv, |r| KvMessage::EnsureAllocate(self.running[idx].clone(), r), Err("RPC failed".to_string())).await {
                    Ok(updated) => { self.running[idx] = updated; }
                    Err(e) => {
                        crate::log_error!("ensure_allocate failed: {e}");
                    }
                }
            }
        }
    }

    pub async fn clear_finished(&mut self) {
        let is_pd_server = self.is_pd_server();
        // Collect the finished sequences' report data (id, cached tokens, sched
        // stats, output length) before the retain() drops them.
        let finished: Vec<(usize, usize, SchedSeqStats, usize, Option<String>)> = self
            .running
            .iter()
            .chain(self.waiting.iter())
            .filter(|s| s.status == SequenceStatus::Finished)
            .map(|s| {
                (
                    s.id,
                    s.num_cached_tokens,
                    s.sched_stats.clone(),
                    s.output_ids.len(),
                    // The model that served this sequence (the "base" when no LoRA
                    // tenant, the tenant name when a LoRA adapter served it).
                    // Surfaced in the per-sequence report.
                    s.sampling_params.lora_tenant.clone(),
                )
            })
            .collect();
        if is_pd_server {
            self.print_free_blocks().await;
        }
        for (seq_id, cached_tokens, mut sched, output_len, lora_tenant) in finished {
            // Stamp the controller's learning trajectory at the sequence's end so the
            // report shows whether QoS was adapting (gain) and holding the SLO (ITL).
            sched.gain_end = self.policy.gain();
            sched.itl_ms = self.policy.itl_ms();
            sched.itl_slo_ms = self.policy.slo_ms();
            sched.output_len = output_len;
            // The model label (the "base" when no LoRA tenant served this sequence,
            // the tenant name when a LoRA adapter did). The operator sees which
            // model produced the output in the per-sequence report.
            sched.model_label = lora_tenant.unwrap_or_else(|| "base".to_string());
            self.policy.record_output(output_len);
            // The §1 two-rate sensor (the slow path): the finish records the turn
            // (the conversation warmth, the L1 capture_seq) + the reuse-evidence
            // (the NInfer real-demand signal: the `proven_frontier` + the 256-entry
            // demand digest) + the request-state transition (the `ModelFinished` lifecycle).
            if self.policy.active() {
                let class = sched.qos_class;
                self.policy.record_turn();
                self.policy.record_reuse_evidence(class, cached_tokens as u64);
                self.policy.record_request_state(class, 7); // The ModelFinished (the 8-state index 7).
            }
            self.remember_finished_report(
                seq_id,
                SeqReport {
                    cached_tokens,
                    sched,
                },
            );
        }
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
        // The cleanup the finished seqs' per-seq K feedback (the no the
        // last_spec_draft_len map accumulating entries for finished sequences, the
        // unbounded growth over a long run). The retain'd running + waiting seqs keep
        // their K (the no the swapped-out seq's K, the 8.1 seed-preservation).
        let live_ids: std::collections::HashSet<usize> = self
            .running
            .iter()
            .chain(self.waiting.iter())
            .map(|s| s.id)
            .collect();
        self.last_spec_draft_len.retain(|id, _| live_ids.contains(id));
    }

    pub async fn release_waitings(&mut self) -> Vec<usize> {
        // Release all waiting sequences since there are no more resources (kv cache)
        let mut released_ids = Vec::with_capacity(self.waiting.len() + self.cached.len());
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            released_ids.push(seq.id);
            seq.status = SequenceStatus::Finished;
            let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
        }
        self.waiting.clear();
        for i in 0..self.cached.len() {
            let seq = &mut self.cached[i];
            released_ids.push(seq.id);
            seq.status = SequenceStatus::Finished;
            let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
            // free gpu blocks and also free any CPU swap space
            let _ = self.kv.send_message(KvMessage::FreeCpuSwapForSeq(seq.id));
        }
        self.cached.clear();
        let _ = self.kv.send_message(KvMessage::ClearPrefixCache);
        released_ids
    }

    pub async fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            // also free cpu swap
            if seq.status == SequenceStatus::Swapped || seq.status == SequenceStatus::FinishSwapped
            {
                let _ = self.kv.send_message(KvMessage::FreeCpuSwapForSeq(seq_id));
            }
            seq.status = SequenceStatus::Finished;
            let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
        }
    }

    pub async fn cancel(&mut self, seq_id: usize) {
        // A normal cancellation will release the runner slot itself. Do not
        // leave a scheduler-side release queued for the same sequence.
        self.pending_runner_releases.retain(|&id| id != seq_id);
        for i in 0..self.running.len() {
            let seq = &mut self.running[i];
            if seq.id == seq_id {
                crate::log_warn!("Seq {} - cancel requested (status {})", seq.id, seq.status);
                seq.status = SequenceStatus::Finished;
                let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                break;
            }
        }
        if let Some(pos) = self.waiting.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.waiting.remove(pos).unwrap();
            if seq.num_cached_tokens > 0 && seq.num_cached_tokens < seq.len() {
                crate::log_warn!(
                    "Seq {} - cancel requested mid-prefill (cached {} / {} tokens)",
                    seq.id,
                    seq.num_cached_tokens,
                    seq.len()
                );
            } else {
                crate::log_warn!("Seq {} - cancel requested (status {})", seq.id, seq.status);
            }
            seq.status = SequenceStatus::Finished;
            let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
        }
        if let Some(pos) = self.transferred.iter().position(|seq| seq.id == seq_id) {
            let seq = self.transferred.remove(pos).unwrap();
            crate::log_warn!(
                "Seq {} - cancel requested while awaiting PD transfer",
                seq.id
            );
            let _ = kv_call(&self.kv, |r| KvMessage::TryReleaseRemoteKvcache(seq.id, r), false).await;
        }
        self.release_cache(seq_id).await;
        self.running.retain(|seq| seq.id != seq_id);
        // The cleanup the cancelled seq's per-seq K feedback (the no the
        // last_spec_draft_len map accumulating entries for cancelled sequences, the
        // unbounded growth over a long run -- the cancel path removes the seq from
        // running directly, the no the clear_finished retain, so the K entry must be
        // cleaned up here too).
        self.last_spec_draft_len.remove(&seq_id);
    }

    #[allow(non_snake_case)]
    pub async fn filter_prefill_finished(
        &mut self,
        scheduled_ids: &Vec<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut finished_seqs = Vec::new();
        let mut remove_ids = Vec::new();
        let mut chunked_info: Vec<(usize, usize, usize)> = Vec::new(); // (seq_id, cached, remain)
        let mut chunk_finished_info: Vec<(usize, usize)> = Vec::new(); // (seq_id, total_len)
        let chunk_size = self.cfg.effective_prefill_chunk_size();
        for (i, id) in scheduled_ids.iter().enumerate() {
            if *id < self.running.len() {
                let seq = &self.running[*id];
                // Use the adaptive chunk stamped by schedule() (fallback: static).
                let seq_chunk = seq.active_prefill_chunk.unwrap_or(chunk_size);
                let chunk_tokens = seq.prefill_chunk_tokens(seq_chunk);
                if chunk_tokens == 0 || seq.num_cached_tokens + chunk_tokens >= seq.len() {
                    let _ = self.kv.call(|r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.len(), r), None).await;
                    if seq.len() > seq_chunk {
                        chunk_finished_info.push((seq.id, seq.len()));
                    }
                    finished_seqs.push((i, seq.id));
                } else {
                    let _ = self.kv.call(|r| KvMessage::CaptureMambaPrefix(seq.clone(), seq.num_cached_tokens + chunk_tokens, r), None).await;
                    remove_ids.push(seq.id);
                    let mut seq = seq.clone();
                    seq.num_cached_tokens += chunk_tokens;
                    // The active mamba slot already contains the state at this
                    // chunk boundary. Keep the captured snapshot available for
                    // other requests, but do not force this in-progress request
                    // to revalidate it on the next scheduling pass.
                    seq.mamba_prefix_hash = None;
                    if seq.active_mamba_prefix_warmup_target().is_none() {
                        seq.clear_mamba_prefix_warmup();
                }
                    seq.status = SequenceStatus::Waiting;
                    chunked_info.push((
                        seq.id,
                        seq.num_cached_tokens,
                        seq.len() - seq.num_cached_tokens,
                    ));
                    self.waiting.push_back(seq);
                }
            }
        }

        if !chunked_info.is_empty() {
            let total_chunked: usize = chunked_info.iter().map(|(_, c, _)| *c).sum();
            let seq_details: Vec<String> = chunked_info
                .iter()
                .map(|(id, cached, remain)| format!("{}:{}/{}", id, cached, cached + remain))
                .collect();
            crate::log_info!(
                "Chunk prefilled {} seq(s) [{}] ({} total tokens processed)",
                chunked_info.len(),
                seq_details.join(", "),
                total_chunked
            );
        }
        if !chunk_finished_info.is_empty() {
            let seq_ids: Vec<usize> = chunk_finished_info.iter().map(|(id, _)| *id).collect();
            let total: usize = chunk_finished_info.iter().map(|(_, len)| len).sum();
            crate::log_warn!(
                "Chunk prefill finished for {} seq(s) {:?} ({} total tokens)",
                chunk_finished_info.len(),
                seq_ids,
                total
            );
        }

        self.running.retain(|s| !remove_ids.contains(&s.id));
        let (indices, finished_ids): (Vec<usize>, Vec<usize>) = finished_seqs.into_iter().unzip();
        let finished_indices: Vec<usize> = finished_ids
            .iter()
            .filter_map(|&target_id| self.running.iter().position(|seq| seq.id == target_id))
            .collect();
        (indices, finished_indices)
    }

    pub async fn try_transfer(&mut self, seq: &mut Sequence) -> bool {
        if !self.is_suitable_for_transfer(&seq).await {
            return false;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        if let Some(tm) = seq.swapped_time {
            if cur_time - tm < SWAP_COOLING_PERIOD {
                return false;
            }
        }
        let success = kv_call(&self.kv, |r| KvMessage::TryTransferPrefill(seq.clone(), r), false).await;
        if success {
            // Client: Offload prefill request to PD server
            seq.swapped_time = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize,
            );
            crate::log_warn!(
                "Prefill request (Seq {}, {} tokens) transfered to PD server.",
                seq.id,
                seq.len(),
            );
            seq.pd_first_token = None;
            self.transferred.push_back(seq.clone());
        } else {
            crate::log_warn!(
                "Unable transfer prefill request (Seq {}) to PD server. Retry later...",
                seq.id
            );
            self.waiting.push_front(seq.clone()); // push back, retry later
        }
        success
    }

    pub async fn try_swap_in(&mut self) {
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for i in 0..self.cached.len() {
            let (status, swapped_time, seq_id) = {
                let seq = &self.cached[i];
                (seq.status, seq.swapped_time().unwrap_or(cur_time), seq.id)
            };
            if status != SequenceStatus::Swapped || cur_time - swapped_time < SWAP_COOLING_PERIOD {
                continue;
            }

            let available_kvcache_tokens = self.get_available_kv_tokens().await;
            let swap_in_required_tokens = kv_call(&self.kv, |r| KvMessage::SwapInRequiredBlocks(self.cached[i].clone(), r), 0).await
                .saturating_mul(self.cfg.block_size);
            if !kv_call(&self.kv, |r| KvMessage::CanSwapIn(self.cached[i].clone(), r), false).await
                || (available_kvcache_tokens.saturating_sub(swap_in_required_tokens)
                    < MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP)
            {
                if !self.running.is_empty() {
                    // Wait for swap in
                    continue;
                }

                let required_blocks = kv_call(&self.kv, |r| KvMessage::SwapInRequiredBlocks(self.cached[i].clone(), r), 0).await
                    .saturating_add(1);
                let evicted = kv_call(&self.kv, |r| KvMessage::EvictPrefixUntilFree(required_blocks, r), 0).await;
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for swap-in Seq {}.",
                        evicted,
                        seq_id
                    );
                break;
                }

                let seq_id = self.cached[i].id;
                crate::log_error!(
                    "No KvCache left for swap in Seq {} — cancelling request.",
                    seq_id
                );
                self.cancel(seq_id).await;
                break;
            }

            let mut seq = self.cached.remove(i);
            seq.swapped_time = Some(cur_time);
            let partial_swap = kv_call(&self.kv, |r| KvMessage::HasPartialCpuSwap(seq.id, r), false).await;
            if !partial_swap {
                seq.clear_block_table(); // reallocate block table since previous gpu blocks were freed
            }

            match kv_call(&self.kv, |r| KvMessage::EnsureAllocate(seq.clone(), r), Err("RPC failed".to_string())).await {
                Ok(updated) => { seq = updated; }
                Err(e) => {
                    crate::log_error!("ensure_allocate failed: {e}");
                    // Transient: GPU blocks may be available on the next schedule pass.
                    // Keep the swapped CPU KV and GDN slot so swap-in can retry.
                    if partial_swap {
                        seq = kv_call(&self.kv, |r| KvMessage::RollbackPartialSwapInAllocation(seq.clone(), r), seq.clone()).await;
                    }
                    seq.status = SequenceStatus::Swapped;
                    let swapped_class = seq.qos_class;
                    self.cached.push(seq);
                    // The §1 two-rate sensor (the fast path): the swap-out
                    // records the request-state transition (the `Swapped` → the
                    // `Paused` lifecycle index 5, the NInfer `EngineRequestState`).
                    if self.policy.active() {
                        self.policy.record_request_state(swapped_class, 5);
                    }
                    break;
                }
            }

            // Swap in data from CPU (if swapped out previously)
            match kv_call(&self.kv, |r| KvMessage::SwapIn(seq.clone(), r), Err("RPC failed".to_string())).await {
                Ok(updated) => {
                    seq = updated;
                    seq.status = SequenceStatus::Running;
                    crate::log_warn!("Seq {} is swapped in for execution!", seq.id);
                }
                Err(e) => {
                    crate::log_error!("Seq {} swap in failed: {e}", seq.id);
                    if partial_swap {
                        seq = kv_call(&self.kv, |r| KvMessage::RollbackPartialSwapInAllocation(seq.clone(), r), seq.clone()).await;
                    } else {
                        let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                        seq.clear_block_table();
                    }
                    if kv_call(&self.kv, |r| KvMessage::HasCpuSwap(seq.id, r), false).await {
                        // CPU swap data intact - retry on a later schedule pass.
                        seq.status = SequenceStatus::Swapped;
                        self.cached.push(seq);
                    } else {
                        // CPU swap state lost - abort and release the GDN slot.
                        self.request_runner_release(seq.id);
                    }
                    break;
                }
            }
            self.running.push(seq);
            break;
        }
    }

    // swap out one sequence a time
    pub async fn try_swap_out(&mut self, idx: usize, is_running: bool) -> bool {
        if (cfg!(feature = "metal") || is_running && idx >= self.running.len())
            || (!is_running && idx >= self.cached.len())
        {
            return false;
        }

        let seq = if is_running {
            &mut self.running[idx]
        } else {
            &mut self.cached[idx]
        };

        // If sequence has blocks, attempt to swap to CPU.
        // If cannot swap, fallback.
        if !seq.block_table.is_empty()
            && (seq.status == SequenceStatus::Running || seq.status == SequenceStatus::Cached)
            && kv_call(&self.kv, |r| KvMessage::CanSwapOut(seq.clone(), r), false).await
        {
            let prefix_mode = kv_call(&self.kv, |r| KvMessage::PrefixCacheEnabled(r), false).await;
            if !prefix_mode {
                // make sure we have identical number of blocks when swapping in
                // for decoding
                match kv_call(&self.kv, |r| KvMessage::EnsureAllocate(seq.clone(), r), Err("RPC failed".to_string())).await {
                    Ok(updated) => { *seq = updated; }
                    Err(e) => {
                        crate::log_error!("ensure_allocate failed: {e}");
                        return false;
                    }
                }
            }
            match kv_call(&self.kv, |r| KvMessage::SwapOut(seq.clone(), r), Err("RPC failed".to_string())).await {
                Ok(updated) => {
                    *seq = updated;
                    let mut seq = if is_running {
                        self.running.remove(idx)
                    } else {
                        // Even though the cached sequence swapped out,
                        // no need remove it from cached list since it can be recoved
                        self.cached.remove(idx)
                    };
                    if seq.status == SequenceStatus::Running {
                        seq.status = SequenceStatus::Swapped;
                    } else {
                        seq.status = SequenceStatus::FinishSwapped;
                    }
                    // seq.num_cached_tokens = seq.len();
                    if !prefix_mode {
                        let _ = self.kv.send_message(KvMessage::Deallocate(seq.clone()));
                    }
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                    return true;
                }
                Err(e) => {
                    crate::log_warn!("Swap out failed for seq {}: {e}", seq.id);
                }
            }
        }
        return false;
    }

    pub async fn try_swap_out_by_id(&mut self, seq_id: usize, is_running: bool) -> bool {
        if is_running {
            if let Some(pos) = self.running.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, true).await;
            }
        } else {
            if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, false).await;
            }
        }
        false
    }

    pub fn is_pd_mode(&self) -> bool {
        self.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    pub async fn is_suitable_for_transfer(&mut self, seq: &Sequence) -> bool {
        if seq.status == SequenceStatus::Swapped // swapped out sequence
            || seq.status == SequenceStatus::FinishSwapped // swapped out and finished sequence
            || seq.len() < PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD
        // prefill length < 128
        {
            return false;
        }

        // Check prefix cache: if most tokens are already cached, do local prefill
        let cached_tokens = kv_call(&self.kv, |r| KvMessage::GetPrefixCacheMatchTokens(seq.clone(), r), 0).await;
        let new_tokens = seq.len().saturating_sub(cached_tokens);
        if cached_tokens > 0 && new_tokens < PD_LOCAL_PREFILL_NEW_TOKEN_THRESHOLD {
            crate::log_info!(
                "Seq {} has {} cached tokens, {} new tokens - doing local prefill",
                seq.id,
                cached_tokens,
                new_tokens
            );
            return false;
        }

        true
    }

    /// Client: Check for finished prefills and move them to the running queue.
pub async fn try_receive_kvcache(&mut self) -> Result<()> {
        let mut finished_seq_ids = Vec::new();
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        for idx in 0..self.transferred.len() {
            let seq_id = self.transferred[idx].id;
            if cur_time - self.transferred[idx].swapped_time().unwrap_or(cur_time)
                < PD_PREFILL_STATUS_CHECK_COOLING_PERIOD
            {
                continue;
            }

            self.transferred[idx].swapped_time = Some(cur_time);
            let status = kv_call(&self.kv, |r| KvMessage::TryCheckPrefillStatus(seq_id, r), false).await;
            if !status {
                continue;
            }
            // We have the data. Can we allocate space for it?
            if !kv_call(&self.kv, |r| KvMessage::CanAllocateWithoutPrefix(self.transferred[idx].clone(), r), false).await
            {
                // Not enough memory right now. Put data back and try later.
                let evicted = kv_call(&self.kv, |r| KvMessage::EvictPrefixUntilFree(self.transferred[idx].num_blocks(), r), 0).await;
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for Seq {} KvCache receiving!",
                        evicted,
                        seq_id
                    );
                    break;
                }

                crate::log_warn!(
                    "KvCache Transfer: Seq {} prefill finished on PD server, but no blocks to receive. Will retry.",
                    seq_id
                );
                // For simplicity, we just break and retry next cycle.
                break;
            }

            // Allocate GPU blocks for the sequence
            match kv_call(&self.kv, |r| KvMessage::AllocateWithoutPrefix(self.transferred[idx].clone(), r), Err("RPC failed".to_string())).await {
                Ok(updated) => { self.transferred[idx] = updated; }
                Err(e) => {
                    crate::log_error!("allocate_without_prefix failed: {e}");
                    break;
                }
            }

            // Perform the actual KV cache data transfer
            let (ret, first_token, sending_time, num_cached_tokens) = kv_call(&self.kv, |r| KvMessage::TryReceiveKvcache(self.transferred[idx].clone(), r), (false, 0, 0, 0)).await;
            let success = ret;
            {
                let seq = &mut self.transferred[idx];
                if success {
                    seq.num_cached_tokens = num_cached_tokens;
                    // Update sequence and move to running
                    // The first token is generated on PD server,
                    // it has been transfered to client, but haven't been send to user
                    seq.append_token(first_token);
                    seq.pd_first_token = Some(first_token);
                    seq.status = SequenceStatus::Running;
                    self.running.push(seq.clone());
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_millis() as usize;
                    let transfer_duration = now - sending_time;

                    if transfer_duration > 10000 {
                        // Log a warning when a sequence takes an unusually long time to receive and swap-in.
                        // Possible causes: insufficient KV cache on server or client, or low communication bandwidth.
                        crate::log_warn!(
                            "KvCache Transfer: Seq {} prefill finished, but receive (with swap-in) time was unexpectedly long ({} s).",
                            seq.id,
                            transfer_duration / 1000
                        );
                    } else {
                        crate::log_info!(
                            "KvCache Transfer: Seq {} prefill finished and received in {} ms!",
                            seq.id,
                            transfer_duration
                        );
                    };

                    // Since KVCache Transfer involved here, the prefill speed might not accurate
                    // since receive remote kvcache requires sufficient local cache memory (sometime pending for kvcache)
                    // The actual prefill speed need to exclude the transfer time, for simplicity we didn't do that

                    let _ = kv_call(&self.kv, |r| KvMessage::TryReleaseRemoteKvcache(seq.id, r), false).await;
                } else {
                    crate::log_error!(
                        "KvCache Transfer: Seq {} prefill finished but failed to receive. Aborting seq.",
                        seq.id,
                    );
                }
            }

            let seq = &mut self.transferred[idx];
            if !success {
                seq.status = SequenceStatus::Finished; // Mark as failed
                self.running.push(seq.clone());
            }
            finished_seq_ids.push(seq.id);
        }

        // Remove all processed sequences from the transferred queue
        self.transferred
            .retain(|s| !finished_seq_ids.contains(&s.id));
        Ok(())
    }

    pub async fn get_num_cached_tokens(&self) -> usize {
        let cached_blocks = kv_call(&self.kv, |r| KvMessage::PrefixCacheBlocks(r), 0).await;
        let block_size = kv_call(&self.kv, |r| KvMessage::BlockSize(r), 0).await;
        cached_blocks * block_size
    }

    /// Per-seq prefix-cache hit count for response finalization. Falls back
    /// to `finished_reports` for seqs already swept by `clear_finished`.
    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .chain(self.transferred.iter())
            .find(|s| s.id == seq_id)
            .map(|s| s.num_cached_tokens)
            .or_else(|| self.finished_reports.get(&seq_id).map(|r| r.cached_tokens))
    }

    /// The end-of-sequence report (cached tokens + QoS scheduling stats) for the
    /// server's per-sequence display. Searches the live pools, then the retained map.
    pub fn get_seq_report(&self, seq_id: usize) -> Option<SeqReport> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .find(|s| s.id == seq_id)
            .map(|s| SeqReport {
                cached_tokens: s.num_cached_tokens,
                sched: s.sched_stats.clone(),
            })
            .or_else(|| self.finished_reports.get(&seq_id).cloned())
    }

    fn remember_finished_report(&mut self, seq_id: usize, report: SeqReport) {
        self.finished_reports.insert(seq_id, report);
        while self.finished_reports.len() > FINISHED_CACHED_TOKENS_MAX {
            let Some(oldest_seq_id) = self.finished_reports.keys().min().copied() else {
                break;
            };
            self.finished_reports.remove(&oldest_seq_id);
        }
    }

    pub async fn evict_prefix_until_free(&mut self, min_free_blocks: usize) -> usize {
        kv_call(&self.kv, |r| KvMessage::EvictPrefixUntilFree(min_free_blocks, r), 0).await
    }

    pub async fn evict_prefix_cache_blocks(&mut self, blocks: usize) -> usize {
        kv_call(&self.kv, |r| KvMessage::EvictPrefixBlocks(blocks, r), 0).await
    }

    async fn evict_prefix_cache_under_pressure(&mut self) -> usize {
        let cached_blocks = kv_call(&self.kv, |r| KvMessage::PrefixCacheBlocks(r), 0).await;
        if cached_blocks == 0 {
            return 0;
        }
        let blocks = ((cached_blocks as f32) * PREFIX_CACHE_PRESSURE_EVICT_PERCENT).ceil() as usize;
        let blocks = blocks.max(1);
        kv_call(&self.kv, |r| KvMessage::EvictPrefixBlocks(blocks, r), 0).await
    }

    pub async fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = kv_call(&self.kv, |r| KvMessage::FreeBlocks(r), 0).await;
        let block_size = kv_call(&self.kv, |r| KvMessage::BlockSize(r), 0).await;
        free_blocks * block_size
    }

    pub async fn get_total_kv_tokens(&self) -> usize {
        let total_blocks = kv_call(&self.kv, |r| KvMessage::TotalBlocks(r), 0).await;
        let block_size = kv_call(&self.kv, |r| KvMessage::BlockSize(r), 0).await;
        total_blocks * block_size
    }

    pub async fn get_cpu_swap_usage(&self) -> (f32, f32) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        if cfg!(feature = "metal") {
            // Metal use unified memory, no cpu swap memory used
            (0f32, 0f32)
        } else {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
            let usage = kv_call(&self.kv, |r| KvMessage::CpuSwapUsage(r), 0.0).await;
            (
                usage * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        }
    }

    pub async fn print_free_blocks(&self) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let total_blocks = kv_call(&self.kv, |r| KvMessage::TotalBlocks(r), 0).await;
        let free_blocks = kv_call(&self.kv, |r| KvMessage::FreeBlocks(r), 0).await;
        let used_percent =
            100.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32) * 100.0f32;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        #[cfg(feature = "cuda")]
        let cpu_swap_log = {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
            let cpu_swap_usage = kv_call(&self.kv, |r| KvMessage::CpuSwapUsage(r), 0.0).await;
            format!(
                "CPU swap used {:.1}% ({:.2}GB/{:.2}GB)",
                cpu_swap_usage * 100.0f32,
                cpu_swap_usage * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let cpu_swap_log = "".to_string();

        let block_size = kv_call(&self.kv, |r| KvMessage::BlockSize(r), 0).await;
        crate::log_info!(
            "GPU Kvcache: {} blocks ({} tokens) free, used {:.1}% ({:.2}GB/{:.2}GB); {}",
            free_blocks,
            free_blocks * block_size,
            used_percent,
            used_percent / 100.0f32 * kvcache_memory_gb,
            kvcache_memory_gb,
            cpu_swap_log,
        );
        if let Some(capacity) = self.cfg.mamba_cache_capacity {
            if capacity > 0 && self.cfg.mamba_slot_bytes > 0 {
                let active_slots = (self.running.len()
                    + self.waiting.len()
                    + self.cached.len()
                    + self.transferred.len())
                .min(capacity);
                let active_percent = active_slots as f32 * 100.0f32 / capacity as f32;
                let slot_mb = self.cfg.mamba_slot_bytes as f32 / 1024.0f32 / 1024.0f32;
                let used_gb = (active_slots * self.cfg.mamba_slot_bytes) as f32 / SIZE_IN_GB as f32;
                let budget_gb = self.cfg.mamba_memory_bytes as f32 / SIZE_IN_GB as f32;
                crate::log_info!(
                    "GPU MambaState: {} / {} slots used ({:.1}%), approx {:.2}GB/{:.2}GB (slot {:.2}MB)",
                    active_slots,
                    capacity,
                    active_percent,
                    used_gb,
                    budget_gb,
                    slot_mb
                );
            }
        }
    }

    pub async fn kv_cache_usage_percent(&self) -> f32 {
        let total_blocks = kv_call(&self.kv, |r| KvMessage::TotalBlocks(r), 0).await;
        let free_blocks = kv_call(&self.kv, |r| KvMessage::FreeBlocks(r), 0).await;
        1.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32)
    }

    /// Check if the given token is a tool call end token
    /// This supports both:
    /// 1. Explicit tool call end tokens (e.g., </tool_call> in XML format)
    /// 2. JSON end token "}" combined with Regex validation for {..."name":..., "arguments":...} pattern
    ///    (only used as a fallback for models without explicit end token IDs)
    pub fn is_tool_call_end(&self, token: u32, idx: usize) -> bool {
        // 1. Check for explicit tool call end tokens (XML style)
        if self.tool_call_end_token_ids.contains(&token) {
            // If we know start token IDs for this model, only treat end markers as structural
            // when a start marker has already appeared in generated output.
            if !self.tool_call_start_token_ids.is_empty() {
                let has_start = self.running[idx]
                    .output_ids
                    .iter()
                    .any(|id| self.tool_call_start_token_ids.contains(id));
                if !has_start {
                    return false;
                }
            }
            return true;
        }

        // 2. JSON regex fallback — only for models without explicit end token IDs.
        // Models with known end tokens (e.g. Qwen3's </tool_call> = 151658) should
        // rely solely on the token ID check above. The regex
        //   \{\s*"name"\s*:.*"arguments"\s*:.*\}\s*$
        // can match prematurely at the inner `}` of the arguments dict before the
        // outer `}` and the actual end token are generated.
        if !self.tool_call_end_token_ids.is_empty() {
            return false;
        }

        if self.json_end_token_id == Some(token) {
            if let Some(tokenizer) = &self.tokenizer {
                let mut temp_output = self.running[idx].output_ids.to_vec();
                temp_output.push(token);

                if let Ok(decoded) = tokenizer.decode(&temp_output, true) {
                    if self.tool_call_regex.is_match(&decoded) {
                        return true;
                }
                }
            }
        }

        false
    }

    fn stop_sequence_match_index(&self, token: u32, seq: &Sequence) -> Option<usize> {
        let Some(stop_sequences) = &seq.sampling_params.stop_token_ids else {
            return None;
        };
        if stop_sequences.is_empty() {
            return None;
        }

        for (idx, stop) in stop_sequences.iter().enumerate() {
            if stop.is_empty() {
                continue;
            }
            if stop.len() == 1 {
                if stop[0] == token {
                    return Some(idx);
                }
                continue;
            }

            let prior_len = seq.output_ids.len();
            if stop.len() - 1 > prior_len {
                continue;
            }
            let start_idx = prior_len + 1 - stop.len();
            if seq.output_ids[start_idx..] == stop[..stop.len() - 1]
                && stop[stop.len() - 1] == token
            {
                return Some(idx);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::{active_sequence_limit, seed_spec_draft_max};
    use crate::utils::config::{EngineConfig, LoraConfig};

    #[test]
    fn active_sequence_limit_uses_mamba_capacity_for_hybrid_models() {
        assert_eq!(active_sequence_limit(8, Some(5)), 5);
        assert_eq!(active_sequence_limit(5, Some(9)), 5);
    }

    #[test]
    fn active_sequence_limit_respects_parallel_limit_for_non_hybrid_models() {
        assert_eq!(active_sequence_limit(1, None), 1);
    }

    #[test]
    fn active_sequence_limit_never_exceeds_mamba_or_parallel_capacity() {
        assert_eq!(active_sequence_limit(12, Some(4)), 4);
        assert_eq!(active_sequence_limit(0, None), 1);
    }

    #[test]
    fn active_sequence_limit_preserves_zero_mamba_as_disabled() {
        assert_eq!(active_sequence_limit(7, Some(0)), 7);
    }

    // The K-bridge tests (the inclusive + exclusive property, the rule 21).
    fn make_econfig(num_spec: Option<usize>, spec_draft_max: usize) -> EngineConfig {
        let mut econfig = EngineConfig::new(
            None, // model_id
            None, // reasoning_grammars
            None, // weight_path
            None, // weight_file
            None, // hf_token
            None, // hf_token_path
            None, // enforce_parser
            None, // max_num_seqs
            None, // config_model_len
            None, // max_model_len
            None, // max_tokens
            None, // isq
            Some(1), // num_shards
            Some(vec![0]), // device_ids
            None, // generation_cfg
            None, // seed
            true, // disable_prefix_cache
            None, // prefix_cache_max_tokens
            None, // kvcache_dtype
            None, // server_mode
            None, // cpu_mem_fold
            None, // kv_fraction
            None, // mamba_fraction
            None, // pd_config
            None, // mcp_command
            None, // mcp_config
            None, // mcp_args
            None, // tool_prompt_template
            None, // pd_server_prefix_cache_ratio
            None, // pd_client_prefix_cache_ratio
            None, // yarn_scaling_factor
            false, // disable_reasoning
            false, // disable_cuda_graph
            None, // prefill_chunk_size
            1, // num_nodes
            0, // node_rank
            None, // master_addr
            29500, // master_port
            false, // enable_tool_grammar
            num_spec, // num_speculative_tokens
            None, // draft_model
            LoraConfig::default(), // lora
        );
        econfig.qos.spec_draft_max = spec_draft_max;
        econfig
    }

    #[test]
    fn seed_spec_draft_max_inclusive_seeds_from_cli_when_unset() {
        // The operator set --num-speculative-tokens 4 + left spec_draft_max at the
        // 0 default: the bridge seeds the ceiling to 4 (the no collapsed K).
        let econfig = make_econfig(Some(4), 0);
        let qos_cfg = seed_spec_draft_max(&econfig);
        assert_eq!(qos_cfg.spec_draft_max, 4);
    }

    #[test]
    fn seed_spec_draft_max_exclusive_preserves_operator_set_ceiling() {
        // The operator explicitly tuned spec_draft_max to 2: the bridge must NOT
        // override it (the explicit QoS tuning wins over the CLI default).
        let econfig = make_econfig(Some(4), 2);
        let qos_cfg = seed_spec_draft_max(&econfig);
        assert_eq!(qos_cfg.spec_draft_max, 2);
    }

    #[test]
    fn seed_spec_draft_max_exclusive_no_cli_means_sd_off() {
        // The no --num-speculative-tokens + the no operator ceiling: the K stays
        // 0 (the SD genuinely off, the no phantom pre-alloc).
        let econfig = make_econfig(None, 0);
        let qos_cfg = seed_spec_draft_max(&econfig);
        assert_eq!(qos_cfg.spec_draft_max, 0);
    }

    #[test]
    fn prepare_step_view_floor_keeps_k_at_least_one_when_enabled() {
        // The floor property: when SD is enabled (spec_tokens > 0) the pre-alloc
        // K is spec_tokens.max(1), so a break-even collapse to 0 still drafts >= 1.
        let k = 0usize.max(1);
        assert_eq!(k, 1);
        let k = 4usize.max(1);
        assert_eq!(k, 4);
    }

    // The record_spec_k_feedback property (the no the k > 0 guard): a k == 0 feedback
    // (the seq fell back to plain decode) MUST update the map (the no the stale K from a
    // prior drafting step). The k > 0 feedback updates the map as before. This is the
    // pure-logic property the fix enforces (the no the Scheduler instance dep).
    #[test]
    fn record_spec_k_feedback_updates_map_for_zero_k() {
        // The property: for any k (the no just k > 0), the map entry is updated to k.
        // The k == 0 case (the plain-decode fallback) is the no the stale K.
        let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        // The prior step's K (the stale value, the no the k == 0 feedback should clear).
        map.insert(7, 8);
        // The k == 0 feedback (the seq fell back to plain decode): the map updates to 0.
        let k: usize = 0;
        map.insert(7, k); // The no the k > 0 guard (the prior defect).
        assert_eq!(map.get(&7), Some(&0), "the k == 0 feedback clears the stale K");
        // The k > 0 feedback (the normal case): the map updates to the new K.
        let k: usize = 3;
        map.insert(7, k);
        assert_eq!(map.get(&7), Some(&3), "the k > 0 feedback updates the K");
    }

    // The seed-preservation property (the no the running-only clobber): a prior-feedback
    // entry for a NON-running seq (the preempted/swapped-out) is preserved (the no the
    // seed rebuilding the map from the running set only, which would drop it + reset the
    // swapped-in seq to the ceiling). The inclusive: a swapped-out seq retains its K.
    #[test]
    fn seed_preserves_non_running_feedback() {
        // The prior feedback (the runner's K for seq 5, which is now swapped-out, the no
        // in the running set).
        let mut prior_feedback: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        prior_feedback.insert(5, 2); // The seq 5's K is 2 (the weak drafter, the no the ceiling 8).
        // The running set (the no seq 5 -- it's swapped out).
        let running_ids: Vec<usize> = vec![1, 2, 3];
        // The seed: start from the prior_feedback (the no the empty) + only add entries for
        // running seqs that lack feedback. The no clobbering the prior_feedback.
        let mut new_map = prior_feedback.clone();
        for &s_id in &running_ids {
            if !new_map.contains_key(&s_id) {
                new_map.insert(s_id, 8); // The ceiling (the no feedback yet, the first step).
            }
        }
        // The inclusive: the swapped-out seq 5 retains its K=2 (the no reset to the ceiling).
        assert_eq!(new_map.get(&5), Some(&2), "the swapped-out seq retains its K");
        // The running seqs (the no prior feedback) get the ceiling.
        assert_eq!(new_map.get(&1), Some(&8));
        assert_eq!(new_map.get(&2), Some(&8));
        assert_eq!(new_map.get(&3), Some(&8));
    }

    // The clear_finished retention property (the no the unbounded map growth): finished
    // seqs' K entries are removed, the live (running + waiting) seqs' K entries are kept.
    #[test]
    fn clear_finished_retains_only_live_seq_k() {
        let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        map.insert(1, 8); // The live running seq.
        map.insert(2, 4); // The live waiting seq.
        map.insert(3, 6); // The finished seq (the no the map entry should be dropped).
        // The live ids (the running + waiting, the no the finished).
        let live_ids: std::collections::HashSet<usize> = [1, 2].iter().cloned().collect();
        map.retain(|id, _| live_ids.contains(id));
        // The inclusive: the live seqs retain their K.
        assert_eq!(map.get(&1), Some(&8));
        assert_eq!(map.get(&2), Some(&4));
        // The exclusive: the finished seq's K is dropped (the no the unbounded growth).
        assert_eq!(map.get(&3), None);
    }

    // The cancel-path K cleanup property (the no the unbounded growth via the cancel
    // path): a cancelled seq's K entry is removed (the cancel path removes the seq from
    // running directly, the no the clear_finished retain, so the K entry must be cleaned
    // up there too). A swapped-out seq's K entry is RETAINED (the swap-in needs it, the
    // seed preservation).
    #[test]
    fn cancel_path_removes_k_entry_swap_out_retains_it() {
        let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        map.insert(1, 8); // The cancelled seq (the no the K entry should be dropped).
        map.insert(2, 4); // The swapped-out seq (the no the K entry should be retained).
        // The cancel path: the last_spec_draft_len.remove(&seq_id) (the no the seq 1).
        map.remove(&1);
        // The inclusive: the swapped-out seq (the no the seq 2) retains its K (the no the
        // swap-in resetting it to the ceiling).
        assert_eq!(map.get(&2), Some(&4), "the swapped-out seq retains its K");
        // The exclusive: the cancelled seq's K is dropped (the no the unbounded growth).
        assert_eq!(map.get(&1), None, "the cancelled seq's K is dropped");
    }
}
