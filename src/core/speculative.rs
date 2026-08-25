// Common speculative-decoding machinery: the `Drafter` trait, the shared verify/accept/rollback/
// emit core, and cross-mechanism statistics. Mechanism-specific drafters (MTP, DFlash, FF) live in
// their own modules (`mtp.rs`, `dflash.rs`, `fftokens.rs`) and implement `Drafter`; the grammar
// firewall and hybrid-state rollback are handled once, here, for every mechanism.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use candle_core::{Result, Tensor};

use crate::core::mtp::{verify_draft_masked, MtpSeqInfo, MtpVerifyResult};
use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;

/// A speculative-drafting proposal: the anchor token plus the candidate block to verify.
pub struct Proposal {
    /// The anchor token emitted this step (freshly sampled).
    pub anchor: u32,
    /// Candidate tokens to verify after the anchor.
    pub tokens: Vec<u32>,
}

/// A drafting mechanism. `propose` produces the candidate block (mechanism-specific); the shared
/// core (`run_spec_decode`) verifies it against the target, applies the grammar firewall, rolls
/// back hybrid state, emits, and records statistics.
pub trait Drafter {
    /// Stable name for stats/logging (e.g. "mtp", "dflash", "ff").
    fn name(&self) -> &'static str;

    /// Mechanism-specific anchor step: target forward + sample the anchor (FSM-committed).
    /// Returns `(anchor_token, optional hidden context for draft)`.
    fn anchor(&self, runner: &ModelRunner, seqs: Seqs, seq: &MtpSeqInfo) -> Result<(u32, Option<Tensor>)>;

    /// Speculative drafts for the positions after the anchor.
    fn draft(
        &self,
        runner: &ModelRunner,
        seq: &MtpSeqInfo,
        anchor: u32,
        hidden: &Option<Tensor>,
    ) -> Result<Vec<u32>>;

    /// Target layers to collect hidden states at during the verify forward (empty for MTP/FF).
    fn verify_target_layers(&self) -> &[usize] {
        &[]
    }

    /// Post-verify hook (e.g. DFlash refreshes its context window). Default no-op.
    fn on_verified(
        &self,
        _runner: &ModelRunner,
        _seq: &MtpSeqInfo,
        _proposal: &Proposal,
        _vhidden: &[Tensor],
        _accepted: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Run the verify forward. Default: eager target forward collecting hidden states at
    /// `verify_target_layers()`. MTP overrides this to use its CUDA-graph replay when captured.
    fn verify_forward(
        &self,
        runner: &ModelRunner,
        verify_ids: &Tensor,
        verify_positions: &Tensor,
        kv_pairs: Option<&Vec<(Tensor, Tensor)>>,
        metadata: &attention_rs::InputMetadata,
_verify_len: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        runner.spec_verify_forward(
            verify_ids,
            verify_positions,
            kv_pairs,
            metadata,
            self.verify_target_layers(),
        )
    }
}

// ---------------------------------------------------------------------------
// Cross-mechanism statistics
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
struct SpecCounters {
    mechanism: String,
    steps: usize,
    proposed: usize,
    accepted: usize,
    rejected: usize,
    grammar_bound: usize,
    target_bound: usize,
    ff_continuations: usize,
}

impl SpecCounters {
    fn add(&mut self, res: &MtpVerifyResult) {
        self.steps += 1;
        self.proposed += res.num_proposed;
        self.accepted += res.num_accepted;
        self.rejected += res.num_proposed.saturating_sub(res.num_accepted);
        if res.grammar_prefix < res.target_prefix {
            self.grammar_bound += 1;
        } else if res.target_prefix < res.grammar_prefix {
            self.target_bound += 1;
        }
        if res.continuation_is_ff {
            self.ff_continuations += 1;
        }
    }

    fn summary(&self, label: &str) -> String {
        let rate = if self.proposed > 0 {
            self.accepted as f64 / self.proposed as f64 * 100.0
        } else {
            0.0
        };
        let avg = if self.steps > 0 {
            (self.accepted + 2 * self.steps) as f64 / self.steps as f64
        } else {
            1.0
        };
        format!(
"{} steps={} proposed={} accepted={} rejected={} rate={:.1}% avg_tok/step={:.2} grammar_bound={} target_bound={} ff_continuations={}",
            label,
            self.steps,
            self.proposed,
            self.accepted,
            self.rejected,
            rate,
            avg,
            self.grammar_bound,
            self.target_bound,
            self.ff_continuations
        )
    }
}

/// Per-sequence window, reported (and dropped) when the sequence finishes.
static SPEC_SEQ_STATS: LazyLock<Mutex<HashMap<usize, SpecCounters>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record one speculative step into the per-seq window.
pub fn spec_stats_update(name: &str, seq_id: usize, res: &MtpVerifyResult) {
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    let c = map.entry(seq_id).or_default();
    if c.mechanism.is_empty() {
        c.mechanism = name.to_string();
    }
    c.add(res);
}

/// Report + drop the per-sequence window (at the sequence's end). None if empty.
pub fn spec_seq_report(seq_id: usize) -> Option<String> {
    let mut map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    let c = map.remove(&seq_id)?;
    if c.steps == 0 {
        return None;
    }
    Some(c.summary(&format!("seq {}", seq_id)))
}

/// Look up a sequence's speculative stats (without removing them) for cross-process reporting.
pub fn spec_seq_stats_data(seq_id: usize) -> crate::runner::SpecSeqStatsData {
    let map = SPEC_SEQ_STATS.lock().expect("spec seq stats mutex poisoned");
    map.get(&seq_id).map(|c| crate::runner::SpecSeqStatsData {
        mechanism: c.mechanism.clone(),
        steps: c.steps,
        proposed: c.proposed,
        accepted: c.accepted,
        rejected: c.rejected,
        grammar_bound: c.grammar_bound,
        target_bound: c.target_bound,
        ff_continuations: c.ff_continuations,
    }).unwrap_or_default()
}

fn build_seq_infos(seqs: &Seqs) -> (usize, Vec<MtpSeqInfo>) {
    match seqs {
        Seqs::SeqRefs(s) => {
            let infos: Vec<MtpSeqInfo> = s
                .iter()
                .map(|seq| MtpSeqInfo {
                    id: seq.id,
                    len: seq.len(),
                    block_table: seq.block_table.clone(),
                })
                .collect();
            (s.len(), infos)
        }
        Seqs::DecodeVec(d) => {
            let infos: Vec<MtpSeqInfo> = d
                .iter()
                .map(|ds| MtpSeqInfo {
                    id: ds.id,
                    len: ds.len,
                    block_table: ds.block_tables.clone(),
                })
                .collect();
            (d.len(), infos)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared speculative-decode core
// ---------------------------------------------------------------------------

impl ModelRunner {
    /// Run one speculative-decode step for a single sequence using `drafter`:
    /// propose -> verify (target forward) -> accept (grammar firewall) -> rollback -> emit -> stats.
    /// Falls back to plain decode when the batch is > 1.
    pub fn run_spec_decode(&self, seqs: Seqs, drafter: &dyn Drafter) -> Result<Vec<Vec<u32>>> {
        let (batch_size, seq_infos) = build_seq_infos(&seqs);
        if batch_size != 1 {
            let output = self.run(seqs, false)?;
            return Ok(output.into_iter().map(|t| vec![t]).collect());
        }
        let seq_info = &seq_infos[0];
        let seq_id = seq_info.id;

        // 0. Terminal FF: if the grammar forces the entire remaining output (e.g. a fixed
        //    "yes"), emit it directly -- no model forward, no anchor, no verify.
        // 1. Anchor (mechanism-specific forward + sample + FSM commit).
        let (anchor_token, hidden) = drafter.anchor(self, seqs, seq_info)?;

        // 2. Speculative drafts (grammar-masked via the fused mask when guided).
        let model_drafts = drafter.draft(self, seq_info, anchor_token, &hidden)?;
        if model_drafts.is_empty() {
            return Ok(vec![vec![anchor_token]]);
        }

        // 3. Assemble the proposal: [model_drafts...].
        let proposal = Proposal {
            anchor: anchor_token,
            tokens: model_drafts.clone(),
        };

        // 5. Build the verify block [anchor, tokens...] and check it fits the pre-allocated KV.
        let verify_tokens: Vec<u32> =
            std::iter::once(proposal.anchor).chain(proposal.tokens.iter().copied()).collect();
        let q_len = verify_tokens.len();
        let block_size = self.block_size();
        let needed_pages = (seq_info.len + q_len).div_ceil(block_size);
        if needed_pages > seq_info.block_table.len() {
            return Ok(vec![vec![proposal.anchor]]);
        }
        let slot_mappings = self.compute_slot_mappings(seq_info, q_len, block_size, drafter.name())?;
        let verify_ids = Tensor::from_vec(verify_tokens.clone(), (q_len,), self.device())?;
        let verify_positions = Tensor::from_vec(
            (0..q_len).map(|i| (seq_info.len + i) as i64).collect::<Vec<_>>(),
            (q_len,),
            self.device(),
        )?;
        let verify_metadata = self.build_mtp_metadata(seq_info, &slot_mappings[..q_len], q_len)?;

        // 6. Verify forward (prefill-style, is_mtp_verify) collecting hidden states at the
        //    drafter's target layers.
        let _prefill_guard = set_linear_is_prefill(true);
        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (vlogits, vhidden) = drafter.verify_forward(
            self,
            &verify_ids,
            &verify_positions,
            kv_pairs,
            &verify_metadata,
            q_len,
        )?;
        drop(kv_cache);
        drop(_prefill_guard);

        // 7. Accept (grammar firewall) + mechanism post-verify hook.
        let res = verify_draft_masked(
            &vlogits,
            &proposal.tokens,
            &self.guided_decoding,
            seq_id,
        )?;
        drafter.on_verified(self, seq_info, &proposal, &vhidden, res.num_accepted)?;

        // 8. Roll back hybrid (Mamba/GDN) state to the accepted boundary on partial rejection.
        if res.num_accepted < res.num_proposed {
            let keep_tokens = 1 + res.num_accepted;
            self.mtp_rollback_mamba(seq_id, keep_tokens)?;
        }

        // 9. Emit [anchor, accepted..., continuation].
        let mut result_tokens = Vec::with_capacity(2 + res.num_accepted);
        result_tokens.push(proposal.anchor);
        result_tokens.extend_from_slice(&res.accepted_tokens);
        result_tokens.push(res.continuation_token);

        // 10. Stats: accumulate per-step + per-sequence counters (reported at the sequence's end).
        spec_stats_update(drafter.name(), seq_id, &res);

        Ok(vec![result_tokens])
    }

    /// The shared verify forward: target model over the verify block, collecting hidden states at
    /// `target_layers` (empty for MTP/FF). Returns `(logits, hidden_collector)`.
    fn spec_verify_forward(
        &self,
        verify_ids: &Tensor,
        verify_positions: &Tensor,
        kv_pairs: Option<&Vec<(Tensor, Tensor)>>,
        verify_metadata: &attention_rs::InputMetadata,
        target_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        match self.model() {
            Model::Qwen3(m) => m.forward_with_hidden_states(
                verify_ids,
                verify_positions,
                kv_pairs,
                verify_metadata,
                false,
                target_layers,
            ),
            Model::Qwen3MoE(m) => m.forward_with_hidden_states(
                verify_ids,
                verify_positions,
                kv_pairs,
                verify_metadata,
                false,
                target_layers,
            ),
            Model::Qwen3_5(m) => m.forward_with_hidden_states(
                verify_ids,
                verify_positions,
                kv_pairs,
                verify_metadata,
                false,
                target_layers,
            ),
            Model::Qwen3_5MoE(m) => m.forward_with_hidden_states(
                verify_ids,
                verify_positions,
                kv_pairs,
                verify_metadata,
                false,
                target_layers,
            ),
            Model::Qwen3VL(m) => m.forward_with_hidden_states(
                verify_ids,
                verify_positions,
                kv_pairs,
                verify_metadata,
                false,
                target_layers,
            ),
            _ => candle_core::bail!("speculative verify requires a supported model type"),
        }
    }
}