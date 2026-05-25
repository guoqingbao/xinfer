// src/core/scheduler.rs
use super::runner::RunnerType;
use super::{
    block_manager::BlockManager,
    prefix_cache::PrefixCacheConfig,
    sequence::{Sequence, SequenceStatus},
};
use crate::transfer::{PdConfig, PdRole};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use candle_core::Result;
use parking_lot::RwLock;
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    transferred: VecDeque<Sequence>,
    pub block_manager: BlockManager,
    next_seq_id: usize,
    /// Per-seq cached-token count retained briefly after `clear_finished()`
    /// so response finalization can still read it. Bounded by
    /// `FINISHED_CACHED_TOKENS_MAX`.
    finished_cached_tokens: HashMap<usize, usize>,
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
}

const MIN_NUM_SCHEDULED_REQS: usize = 5;
/// Cap on `Scheduler::finished_cached_tokens`. Bounded memory beats LRU
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

fn build_prefix_cache_config(econfig: &EngineConfig) -> PrefixCacheConfig {
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

        ((econfig.num_blocks as f32) * ratio) as usize
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
    pub fn new(runners: Arc<RwLock<RunnerType>>, econfig: &EngineConfig, config: &Config) -> Self {
        let prefix_cache_cfg = build_prefix_cache_config(econfig);
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cached: Vec::new(),
            transferred: VecDeque::new(),
            block_manager: BlockManager::new(
                runners,
                econfig.num_blocks,
                (econfig.cpu_mem_fold.unwrap_or(0.2f32) * econfig.num_blocks as f32) as usize,
                econfig.block_size,
                prefix_cache_cfg,
                config
                    .architectures
                    .as_ref()
                    .and_then(|arches| arches.first())
                    .map(|arch| crate::utils::is_qwen3_hybrid_arch_name(arch))
                    .unwrap_or(false),
            ),
            next_seq_id: 0,
            finished_cached_tokens: HashMap::new(),
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
        }
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
        self.waiting.push_back(seq);
        id
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Schedule sequences and return their indexes in `running` along with prefill flag
    #[allow(non_snake_case)]
    pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;
        let chunk_size = self.cfg.effective_prefill_chunk_size();

        // PD server: Check for new incoming prefill requests
        if self.is_pd_server() {
            if let Ok((fit, Some(seq))) = self
                .block_manager
                .try_receive_prefill(self.get_available_kv_tokens())
            {
                let seq_id = seq.id;
                if !fit {
                    crate::log_warn!(
                        "Prefill request (Seq {}) enter pending status because it require {} KvCache tokens (left {}).",
                        seq_id,
                        seq.len() + 1,
                        self.get_available_kv_tokens(),
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

        // Prefill phase: move sequences from waiting to running if possible
        // Use effective chunk tokens (not full sequence length) for budget accounting
        // to enable batched prefill of multiple sequences per step.
        // Capture running count before this batch so the interleave check only
        // considers pre-existing decode sequences, not ones added in this loop.
        let pre_existing_running = self.running.len();

        // For hybrid mamba models, cap total concurrent sequences to the mamba
        // cache capacity so the runner never hits "no free slots" at forward time.
        let max_seqs_limit = if let Some(mamba_cap) = self.cfg.mamba_cache_capacity {
            if mamba_cap > 0 {
                mamba_cap
            } else {
                std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
            }
        } else {
            std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
        };

        while let Some(mut seq) = self.waiting.pop_front() {
            // Try to transfer prefill requests to PD server when applicable
            if self.is_pd_mode() && !self.is_pd_server() && self.try_transfer(&mut seq) {
                break;
            }

            let effective_tokens = std::cmp::min(chunk_size, seq.len() - seq.num_cached_tokens);

            if self.running.len() >= max_seqs_limit
                || scheduled_ids.len() >= std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
                || num_tokens + effective_tokens >= self.cfg.max_num_batched_tokens - 1
                || (seq.block_table.is_empty() && !self.block_manager.can_allocate(&seq))
                // interleaved scheduling: alternate prefill/decode for fairness
                // only block when there are pre-existing decode sequences
                || (self.is_last_prefill && pre_existing_running > 0)
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            if seq.block_table.is_empty() {
                self.block_manager.allocate(&mut seq)?;
            }
            seq.status = SequenceStatus::Running;
            num_tokens += effective_tokens;
            self.running.push(seq);
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
            if !self.block_manager.can_append(seq)
                && seq.status != SequenceStatus::Swapped
                && seq.status != SequenceStatus::FinishSwapped
            {
                preempt_ids.push(idx);
            }
        }

        // Client: Check for finished prefills
        if self.is_pd_mode() && !self.is_pd_server() {
            self.try_receive_kvcache()?;
        }

        // Swap back seq from cpu memory if possible. Prefix-cache eviction is
        // backend-neutral, while CPU swap is CUDA-only.
        let should_try_swap_in = cfg!(feature = "cuda")
            && preempt_ids.is_empty()
            && (self.kv_cache_usage_percent() < KVCACHE_SWAP_THRESHOLD * 0.9
                || (self.running.is_empty() && self.kv_cache_usage_percent() <= 0.3f32));
        if should_try_swap_in {
            #[cfg(feature = "cuda")]
            self.try_swap_in();
        } else if !preempt_ids.is_empty() || self.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD
        {
            // Requests unable to be processed at the current moment
            // If we only have one sequence running and it has been preempt,
            // swap out to cpu memory make non-sense
            // in such case, the only option is either waiting resources or abort it
            let evicted = self.evict_prefix_cache_under_pressure();
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
                        self.try_swap_out(idx, true);
                    }
                }
            }
        }

        let is_pd_server = self.is_pd_server();
        let decode_max_seqs = if let Some(mamba_cap) = self.cfg.mamba_cache_capacity {
            if mamba_cap > 0 {
                std::cmp::min(
                    mamba_cap,
                    std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS),
                )
            } else {
                std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
            }
        } else {
            std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
        };
        let mut pd_finished_ids: Vec<usize> = Vec::new();
        for (idx, seq) in self.running.iter_mut().enumerate() {
            if decode_ids.len() >= decode_max_seqs {
                break;
            }
            if seq.status == SequenceStatus::Finished {
                continue;
            }
            if !self.block_manager.can_append(&seq) {
                // filter out seq that unable to acquire resources
                continue;
            }
            if is_pd_server && seq.status == SequenceStatus::Cached {
                if let Ok(success) = self.block_manager.try_check_kvcache_release(seq.id) {
                    if success {
                        crate::log_warn!("PD Server: release prefilled kvcache for Seq {} (prefix cache retained)", seq.id);
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(seq);
                        pd_finished_ids.push(seq.id);
                    }
                }
                continue;
            }
            self.block_manager.may_append(seq)?;
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
    pub fn fallback_missing_mamba_prefix_snapshots(
        &mut self,
        scheduled_ids: &[usize],
    ) -> Result<usize> {
        let (running, block_manager) = (&mut self.running, &mut self.block_manager);
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

            let has_snapshot = match block_manager.try_has_mamba_prefix_state(hash) {
                Ok(v) => v,
                Err(e) => {
                    crate::log_warn!(
                        "Seq {}: failed to query mamba prefix snapshot hash {}: {}. Falling back to full prefill.",
                        seq.id,
                        hash,
                        e
                    );
                    false
                }
            };
            if has_snapshot {
                continue;
            }

            let prev_cached = seq.num_cached_tokens;
            let required_blocks = seq.num_blocks();

            // Drop current block refs (including any shared prefix refs), then rebuild from scratch.
            block_manager.deallocate(seq);
            seq.clear_block_table();

            if !block_manager.can_allocate_without_prefix(seq) {
                let evicted = block_manager.evict_prefix_cache_until_free(required_blocks);
                if evicted > 0 {
                    crate::log_warn!(
                        "Seq {}: evicted {} prefix cache block(s) before mamba fallback reallocation.",
                        seq.id,
                        evicted
                    );
                }
            }
            block_manager.allocate_without_prefix(seq)?;
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

    /// Postprocess output tokens and modify sequences by indexes
    pub fn postprocess(&mut self, ids: &[usize], output_ids: &[u32]) {
        for (i, &idx) in ids.iter().enumerate() {
            // Sequence may swapped out
            if idx >= self.running.len() {
                continue;
            }
            let seq_id = self.running[idx].id;
            let token = output_ids[i];

            // Since all reqeusts in PD server are prefill request, we need to finish and transfer
            // the kvcache in the first postprocess for each request.
            if self.is_pd_server() {
                match self
                    .block_manager
                    .try_send_kvcache(&self.running[idx], token)
                {
                    Ok(success) => {
                        crate::log_warn!(
                            "PD Server: transferred KV cache for seq {} ({})",
                            seq_id,
                            if success { "success" } else { "faild" }
                        );
                        let seq = &mut self.running[idx];
                        if success {
                            // Insert into prefix cache so future requests can benefit
                            // LRU eviction will handle memory pressure automatically
                            self.block_manager
                                .capture_mamba_prefix_state(seq, seq.len());
                            self.block_manager.cache_sequence(seq);
                            // Maintain resources until client asks to release or cache eviction
                            seq.status = SequenceStatus::Cached;
                            let cur_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis() as usize;
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
                            // release resources immediately if failed
                            seq.status = SequenceStatus::Finished;
                            self.block_manager.deallocate(seq);
                        }
                    }
                    Err(e) => {
                        crate::log_error!(
                            "PD Server: failed to transfer KV cache for seq {}: {}",
                            seq_id,
                            e
                        );
                        let seq = &mut self.running[idx];
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(seq);
                    }
                }

                continue; // Go to next sequence in postprocess
            }

            // Check for tool call end token BEFORE checking EOS
            // When tools are enabled (mcp_mode.is_some()), finish stream at </tool_call>
            if self.running[idx].sampling_params.mcp_mode.is_some() {
                // Check if this is a tool call end (supports both XML </tool_call> and JSON } patterns)
                // We check BEFORE borrowing seq mutably
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
                    // External tool mode: finish stream so client can handle tool calls
                    seq.status = SequenceStatus::Finished;
                    self.block_manager
                        .capture_mamba_prefix_state(seq, seq.len());
                    self.block_manager.cache_sequence(seq);
                    self.block_manager.deallocate(seq);
                    continue;
                }
            }

            let matched_stop_sequence_idx =
                self.stop_sequence_match_index(token, &self.running[idx]);
            let hit_stop_sequence = matched_stop_sequence_idx.is_some();
            let seq = &mut self.running[idx];

            if hit_stop_sequence
                || self.eos_token_id.contains(&token)
                || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                || seq.len() > self.cfg.max_num_batched_tokens
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
                self.block_manager
                    .capture_mamba_prefix_state(seq, seq.len());
                self.block_manager.cache_sequence(seq);
                self.block_manager.deallocate(seq);
            } else {
                seq.append_token(token);
                if seq.len() % self.cfg.block_size == 0 {
                    self.block_manager
                        .capture_mamba_prefix_state(seq, seq.len());
                }
            }
        }
    }

    pub fn clear_finished(&mut self) {
        let is_pd_server = self.is_pd_server();
        let mut finished_counts = Vec::new();
        for seq in &self.running {
            if seq.status == SequenceStatus::Finished {
                if is_pd_server {
                    self.print_free_blocks();
                }
                finished_counts.push((seq.id, seq.num_cached_tokens));
            }
        }
        for seq in &self.waiting {
            if seq.status == SequenceStatus::Finished {
                finished_counts.push((seq.id, seq.num_cached_tokens));
            }
        }
        for (seq_id, num_cached_tokens) in finished_counts {
            self.remember_finished_cached_tokens(seq_id, num_cached_tokens);
        }
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }

    pub fn release_waitings(&mut self) {
        // Release all waiting sequences since there are no more resources (kv cache)
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
        }
        self.waiting.clear();
        for i in 0..self.cached.len() {
            let seq = &mut self.cached[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
            // free gpu blocks and also free any CPU swap space
            self.block_manager.free_cpu_swap_for_seq(seq.id);
        }
        self.cached.clear();
        self.block_manager.clear_prefix_cache();
    }

    pub fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            // also free cpu swap
            if seq.status == SequenceStatus::Swapped || seq.status == SequenceStatus::FinishSwapped
            {
                self.block_manager.free_cpu_swap_for_seq(seq_id);
            }
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(&seq);
        }
    }

    pub fn cancel(&mut self, seq_id: usize) {
        for i in 0..self.running.len() {
            let seq = &mut self.running[i];
            if seq.id == seq_id {
                crate::log_warn!("Seq {} - cancel requested (status {})", seq.id, seq.status);
                seq.status = SequenceStatus::Finished;
                self.block_manager.deallocate(seq);
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
            self.block_manager.deallocate(&seq);
        }
        if let Some(pos) = self.transferred.iter().position(|seq| seq.id == seq_id) {
            let seq = self.transferred.remove(pos).unwrap();
            crate::log_warn!(
                "Seq {} - cancel requested while awaiting PD transfer",
                seq.id
            );
            let _ = self.block_manager.try_release_remote_kvcache(seq.id);
        }
        self.release_cache(seq_id);
        self.running.retain(|seq| seq.id != seq_id);
    }

    #[allow(non_snake_case)]
    pub fn filter_prefill_finished(
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
                if seq.len() < chunk_size || seq.num_cached_tokens + chunk_size >= seq.len() {
                    self.block_manager
                        .capture_mamba_prefix_state(seq, seq.len());
                    if seq.len() > chunk_size {
                        chunk_finished_info.push((seq.id, seq.len()));
                    }
                    finished_seqs.push((i, seq.id));
                } else {
                    self.block_manager
                        .capture_mamba_prefix_state(seq, seq.num_cached_tokens + chunk_size);
                    remove_ids.push(seq.id);
                    let mut seq = seq.clone();
                    seq.num_cached_tokens += chunk_size;
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

    pub fn try_transfer(&mut self, seq: &mut Sequence) -> bool {
        if !self.is_suitable_for_transfer(&seq) {
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
        if let Ok(success) = self.block_manager.try_transfer_prefill(&seq) {
            // Client: Offload prefill request to PD server
            seq.swapped_time = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize,
            );
            if success {
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
        } else {
            false
        }
    }

    pub fn try_swap_in(&mut self) {
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for i in 0..self.cached.len() {
            let (status, swapped_time, seq_id, seq_len) = {
                let seq = &self.cached[i];
                (
                    seq.status,
                    seq.swapped_time().unwrap_or(cur_time),
                    seq.id,
                    seq.len(),
                )
            };
            if status != SequenceStatus::Swapped || cur_time - swapped_time < SWAP_COOLING_PERIOD {
                continue;
            }

            let available_kvcache_tokens = self.get_available_kv_tokens();
            if !self.block_manager.can_swap_in(&self.cached[i])
                || (available_kvcache_tokens - seq_len < MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP)
            {
                if !self.running.is_empty() {
                    // Wait for swap in
                    continue;
                }

                let required_blocks = self.cached[i].num_blocks().saturating_add(1);
                let evicted = self
                    .block_manager
                    .evict_prefix_cache_until_free(required_blocks);
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for swap-in Seq {}.",
                        evicted,
                        seq_id
                    );
                    break;
                }

                let mut seq = self.cached.remove(i);
                seq.status = SequenceStatus::Finished;
                crate::log_error!("No KvCache left for swap in Seq {}!", seq.id);
                self.running.push(seq);
                break;
            }

            let mut seq = self.cached.remove(i);
            seq.swapped_time = Some(cur_time);
            seq.clear_block_table(); //reallocate block table since previous gpu blocks were freed

            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                continue;
            }

            // Swap in data from CPU (if swapped out previously)
            match self.block_manager.swap_in(&mut seq) {
                Ok(_) => {
                    seq.status = SequenceStatus::Running;
                    crate::log_warn!("Seq {} is swapped in for execution!", seq.id);
                }
                Err(e) => {
                    seq.status = SequenceStatus::Finished;
                    crate::log_error!("Seq {} swap in failed: {:?}!", seq.id, e);
                }
            }
            self.running.push(seq);
            break;
        }
    }

    // swap out one sequence a time
    pub fn try_swap_out(&mut self, idx: usize, is_running: bool) -> bool {
        if (cfg!(feature = "metal") || is_running && idx >= self.running.len())
            || (!is_running && idx >= self.cached.len())
        {
            return false;
        }

        let mut seq = if is_running {
            &mut self.running[idx]
        } else {
            &mut self.cached[idx]
        };

        // If sequence has blocks, attempt to swap to CPU.
        // If cannot swap, fallback.
        if !seq.block_table.is_empty()
            && (seq.status == SequenceStatus::Running || seq.status == SequenceStatus::Cached)
            && self.block_manager.can_swap_out(&seq)
        {
            // make sure we have identical number of blocks when swapping in
            // for decoding
            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                return false;
            }
            match self.block_manager.swap_out(&mut seq) {
                Ok(_) => {
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
                    self.block_manager.deallocate(&seq);
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                    return true;
                }
                Err(e) => {
                    crate::log_warn!("Swap out failed for seq {}: {:?}", seq.id, e);
                }
            }
        }
        return false;
    }

    pub fn try_swap_out_by_id(&mut self, seq_id: usize, is_running: bool) -> bool {
        if is_running {
            if let Some(pos) = self.running.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, true);
            }
        } else {
            if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, false);
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

    pub fn is_suitable_for_transfer(&mut self, seq: &Sequence) -> bool {
        if seq.status == SequenceStatus::Swapped // swapped out sequence
            || seq.status == SequenceStatus::FinishSwapped // swapped out and finished sequence
            || seq.len() < PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD
        // prefill length < 128
        {
            return false;
        }

        // Check prefix cache: if most tokens are already cached, do local prefill
        let cached_tokens = self.block_manager.get_prefix_cache_match_tokens(seq);
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
    pub fn try_receive_kvcache(&mut self) -> Result<()> {
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
            let status = self.block_manager.try_check_prefill_status(seq_id);
            if status.is_err() || !status.unwrap_or(false) {
                continue;
            }
            // We have the data. Can we allocate space for it?
            if !self
                .block_manager
                .can_allocate_without_prefix(&self.transferred[idx])
            {
                // Not enough memory right now. Put data back and try later.
                let evicted = self
                    .block_manager
                    .evict_prefix_cache_until_free(self.transferred[idx].num_blocks());
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
            self.block_manager
                .allocate_without_prefix(&mut self.transferred[idx])?;

            // Perform the actual KV cache data transfer
            let mut success = false;
            match self
                .block_manager
                .try_receive_kvcache(&self.transferred[idx])
            {
                Ok((ret, first_token, sending_time, num_cached_tokens)) => {
                    let seq = &mut self.transferred[idx];
                    success = ret;
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

                        self.block_manager.try_release_remote_kvcache(seq.id)?;
                    } else {
                        crate::log_error!(
                            "KvCache Transfer: Seq {} prefill finished but failed to receive. Aborting seq.",
                            seq.id,
                        );
                    }
                }
                Err(e) => {
                    crate::log_error!(
                        "KvCache Transfer: Failed to receive KV cache for seq {}: {}. Aborting seq.",
                        seq_id, e
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

    pub fn get_num_cached_tokens(&self) -> usize {
        self.block_manager.prefix_cache_blocks() * self.block_manager.get_block_size()
    }

    /// Per-seq prefix-cache hit count for response finalization. Falls back
    /// to `finished_cached_tokens` for seqs already swept by `clear_finished`.
    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .chain(self.transferred.iter())
            .find(|s| s.id == seq_id)
            .map(|s| s.num_cached_tokens)
            .or_else(|| self.finished_cached_tokens.get(&seq_id).copied())
    }

    fn remember_finished_cached_tokens(&mut self, seq_id: usize, num_cached_tokens: usize) {
        self.finished_cached_tokens
            .insert(seq_id, num_cached_tokens);
        while self.finished_cached_tokens.len() > FINISHED_CACHED_TOKENS_MAX {
            let Some(oldest_seq_id) = self.finished_cached_tokens.keys().min().copied() else {
                break;
            };
            self.finished_cached_tokens.remove(&oldest_seq_id);
        }
    }

    pub fn evict_prefix_cache_until_free(&mut self, min_free_blocks: usize) -> usize {
        self.block_manager
            .evict_prefix_cache_until_free(min_free_blocks)
    }

    pub fn evict_prefix_cache_blocks(&mut self, blocks: usize) -> usize {
        self.block_manager.evict_prefix_cache_blocks(blocks)
    }

    fn evict_prefix_cache_under_pressure(&mut self) -> usize {
        let cached_blocks = self.block_manager.prefix_cache_blocks();
        if cached_blocks == 0 {
            return 0;
        }
        let blocks = ((cached_blocks as f32) * PREFIX_CACHE_PRESSURE_EVICT_PERCENT).ceil() as usize;
        let blocks = blocks.max(1);
        self.block_manager.evict_prefix_cache_blocks(blocks)
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_manager.get_num_free_blocks();
        free_blocks * self.block_manager.get_block_size()
    }

    pub fn get_total_kv_tokens(&self) -> usize {
        let total_blocks = self.block_manager.get_num_total_blocks();
        total_blocks * self.block_manager.get_block_size()
    }

    pub fn get_cpu_swap_usage(&self) -> (f32, f32) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        if cfg!(feature = "metal") {
            // Metal use unified memory, no cpu swap memory used
            (0f32, 0f32)
        } else {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.2f32);
            (
                self.block_manager.get_cpu_swap_usage() * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        }
    }

    pub fn print_free_blocks(&self) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        let used_percent =
            100.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32) * 100.0f32;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        #[cfg(feature = "cuda")]
        let cpu_swap_log = {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.2f32);
            format!(
                "CPU swap used {:.1}% ({:.2}GB/{:.2}GB)",
                self.block_manager.get_cpu_swap_usage() * 100.0f32,
                self.block_manager.get_cpu_swap_usage() * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let cpu_swap_log = "".to_string();

        crate::log_info!(
            "GPU Kvcache: {} blocks ({} tokens) free, used {:.1}% ({:.2}GB/{:.2}GB); {}",
            free_blocks,
            free_blocks * self.block_manager.get_block_size(),
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

    pub fn kv_cache_usage_percent(&self) -> f32 {
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
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
