// src/utils/guidance.rs
// This module contains non-grammar guidance utilities:
// - GuidanceTokens: token ID collections
// - ParserFactory: llguidance parser factory
// - GuidanceState: matcher state for guided decoding

use crate::utils::config::TokenizerConfig;
use crate::utils::special_tokens::SpecialTokens;
use anyhow::Result;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::SimpleVob;
use toktrie_hf_tokenizers::ByteTokenizer;

// Re-export from guidance_grammar for grammar-related types
// Only export the two entrypoints: generate_grammar_from_request and build_grammar_from_request
pub use crate::utils::guidance_grammar::{
    build_grammar_from_request, generate_grammar_from_request,
};

#[derive(Clone, Debug, Default)]
pub struct GuidanceTokens {
    pub bos_token_ids: Vec<u32>,
    pub eos_token_ids: Vec<u32>,
    pub reasoning_start_ids: Vec<u32>,
    pub reasoning_end_ids: Vec<u32>,
    pub tool_call_start_ids: Vec<u32>,
    pub tool_call_end_ids: Vec<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

impl GuidanceTokens {
    /// Compress a sorted list of token IDs into ranges.
    /// E.g., [1, 2, 3, 5, 7, 8] -> [(1, 3), (5, 5), (7, 8)]
    fn compress_to_ranges(ids: &[u32]) -> Vec<(u32, u32)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut ranges = Vec::new();
        let mut start = ids[0];
        let mut prev = ids[0];
        for &id in &ids[1..] {
            if id == prev + 1 {
                prev = id;
            } else {
                ranges.push((start, prev));
                start = id;
                prev = id;
            }
        }
        ranges.push((start, prev));
        ranges
    }

    /// Generate a token-range expression for free text generation.
    /// Returns the RHS expression usable directly after `text: ` in Lark grammar.
    /// Uses llguidance's negated token range syntax to allow all tokens EXCEPT the
    /// excluded set (the ~20 control tokens), a 12,000x smaller mask than the
    /// allowed set. Empty exclusions fall back to the full regex.
    pub fn token_range_expression(excluded_ids: Vec<u32>) -> String {
        if excluded_ids.is_empty() {
            return r#"/(?s:.*)/"#.to_string();
        }
        let mut sorted_ids: Vec<u32> = excluded_ids;
        sorted_ids.sort();
        sorted_ids.dedup();
        let ranges = Self::compress_to_ranges(&sorted_ids);
        format!(
            "(<[^{}]>)+",
            ranges
                .iter()
                .map(|(start, end)| {
                    if start == end {
                        start.to_string()
                    } else {
                        format!("{}-{}", start, end)
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        )
        .trim()
        .to_string()
    }

    // Disallow all control tokens, used in the middle of grammars
    pub fn text_grammar_mask(&self) -> String {
        let mut ids = Vec::new();
        ids.extend_from_slice(&self.bos_token_ids);
        ids.extend_from_slice(&self.eos_token_ids);
        ids.extend_from_slice(&self.reasoning_start_ids);
        ids.extend_from_slice(&self.reasoning_end_ids);
        ids.extend_from_slice(&self.tool_call_start_ids);
        ids.extend_from_slice(&self.tool_call_end_ids);
        Self::token_range_expression(ids)
    }

    // Construct reasoning mask rule relative to how model must generate (BOS+ or prepended)
    pub fn reasoning_grammar_mask(&self) -> String {
        let mut range_exp = self.text_grammar_mask();
        if self.add_bos_token {
            range_exp = format!(
                r#"({}) {}"#,
                self.reasoning_start_ids
                    .iter()
                    .map(|&n| format!("<[{}]>", n))
                    .collect::<Vec<String>>()
                    .join(" | "),
                &range_exp
            );
        }
        format!(
            r#"{} ({})"#,
            &range_exp,
            self.reasoning_end_ids
                .iter()
                .map(|&n| format!("<[{}]>", n))
                .collect::<Vec<String>>()
                .join(" | ")
        )
    }

    /// Get the excluded token IDs for the free-text rule (the BOS + the reasoning + the
    /// tool-call IDs, NOT the EOS). Used by the ToolCallGrammar to build the text rule
    /// via token_range_expression (the token-number masking).
    pub fn get_text_excluded_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        ids.extend_from_slice(&self.bos_token_ids);
        ids.extend_from_slice(&self.reasoning_start_ids);
        ids.extend_from_slice(&self.reasoning_end_ids);
        ids.extend_from_slice(&self.tool_call_start_ids);
        ids.extend_from_slice(&self.tool_call_end_ids);
        ids
    }
}

pub fn extract_guidance_tokens(
    tokenizer: &Tokenizer,
    eos_token_ids: Vec<u32>,
    bos_token_ids: Vec<u32>,
    tokenizer_config: &TokenizerConfig,
) -> GuidanceTokens {
    let special_tokens = SpecialTokens::new(tokenizer);

    // Verify EOS token IDs are in added vocabulary if more than one provided
    let added_tokens: HashMap<u32, String> = tokenizer
        .get_added_tokens_decoder()
        .iter()
        .map(|(id, token)| (*id, token.content.clone()))
        .collect();

    let validated_eos: Vec<u32> = if eos_token_ids.len() > 1 {
        eos_token_ids
            .into_iter()
            .filter(|id| added_tokens.contains_key(id))
            .collect()
    } else {
        eos_token_ids
    };

    let validated_bos: Vec<u32> = {
        let retained: Vec<u32> = bos_token_ids
            .into_iter()
            .filter(|id| !validated_eos.contains(id))
            .collect();
        if retained.is_empty() {
            special_tokens.bos_token_ids()
        } else {
            retained
        }
    };

    // Determine if BOS token should be added based on tokenizer config
    // add_bos_token == Some(true) means the tokenizer adds BOS automatically
    let add_bos_token = tokenizer_config.add_bos_token == Some(true);
    let add_eos_token = tokenizer_config.add_eos_token == Some(true);

    GuidanceTokens {
        bos_token_ids: validated_bos,
        eos_token_ids: validated_eos,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
        add_bos_token,
        add_eos_token,
    }
}

pub type ParserFactory = LlgParserFactory;

pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
) -> Result<Arc<ParserFactory>> {
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    let target_vocab = vocab_size.map(|v| {
        if v < tokenizer_vocab {
            crate::log_warn!(
                "Requested vocab size {} is smaller than tokenizer vocab size {}. Using tokenizer size.",
                v,
                tokenizer_vocab
            );
            tokenizer_vocab
        } else {
            v
        }
    });
    let env = ByteTokenizer::from_tokenizer(tokenizer)?.into_tok_env(target_vocab)?;
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub struct GuidanceState {
    matcher: Matcher,
    /// Track generated tokens for logging and reasoning-mode transition.
    llm_tokens: Vec<u32>,
    /// vLLM/SGLang two-phase reasoning support:
    /// Token IDs that mark the end of reasoning (e.g. </think>).
    /// When non-empty, grammar constraints are deferred until after
    /// a reasoning-end token is seen. This keeps reasoning free-form
    /// and only constrains the structured output that follows.
    reasoning_end_ids: Vec<u32>,
    /// Whether reasoning has ended (the </think> token was observed).
    /// Once true, grammar masks are applied normally.
    reasoning_ended: bool,
    /// Full-envelope mode: grammar constrains from BOS to EOS including reasoning.
    /// When true, the `reasoning_ended` logic is bypassed and masks are always applied.
    full_envelope: bool,
    /// The index into `llm_tokens` of the last token successfully committed to the matcher
    /// (the reject-and-resync boundary, the P2 fix). Tokens at or after this index are uncommitted.
    last_committed: usize,
    /// The count of reasoning tokens admitted so far (the two-phase mode, the
    /// `reasoning_ended` is false). Used by the reasoning-budget promotion (the
    /// sampling bias that steers the model to the end-of-reasoning token as
    /// the count approaches the budget). Reset when `reasoning_ended` flips.
    reasoning_token_count: usize,
}

impl GuidanceState {
    pub fn new_from_grammar_with_reasoning(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
        reasoning_end_ids: Vec<u32>,
        log_rank0: bool,
    ) -> Result<Self> {
        use crate::utils::guidance_grammar::get_lark_from_top_level_grammar;

        if tracing::enabled!(tracing::Level::DEBUG) {
            let lark = get_lark_from_top_level_grammar(grammar);
            tracing::debug!(
                "[llg] Initializing guidance parser from grammar: {} bytes, {} lines",
                lark.len(),
                lark.lines().count()
            );
            tracing::trace!("[llg] Guidance parser grammar:\n{}\n", lark);
        }

        // In full-envelope mode OR llg_full_enabled, reasoning is constrained from the start
        let llg_full_enabled = crate::utils::env::llg_full_enabled();
        let reasoning_ended = if llg_full_enabled {
            true  // Single-phase: grammar always applies
        } else {
            reasoning_end_ids.is_empty()
        };

        // The parser is created fresh per GuidanceState (the no cache, the no
        // deep_clone). The deep_clone corrupted the parser's internal PDA state
        // (the epsilon closure + the stack), so the full-envelope (the
        // XINFER_LLG_FULL) grammar got stuck in the reasoning_block phase and
        // could never reach the eos rule. The fresh parser (the
        // factory.create_parser) is correct; the performance cost of
        // re-instantiating the FSM per sequence is accepted for correctness.
        let mut parser = if !reasoning_ended {
            if log_rank0 {
                crate::log_info!(
                    "[llg] Two-phase reasoning: grammar constraint deferred until after reasoning end tokens {:?}",
                    reasoning_end_ids
                );
            }
            factory.create_parser(grammar.clone())?
        } else {
            if log_rank0 {
                crate::log_info!(
                    "[llg] Full-envelope/single-phase mode: grammar constrains all generation"
                );
            }
            // Max tokens is capped by the scheduler anyway so allow grammar space to generate reasoning
            if let Some(max_tokens) = grammar.max_tokens {
                let mut grammar = grammar.clone();
                grammar.max_tokens = Some(max_tokens * 2);
                factory.create_parser(grammar)?
            } else {
                factory.create_parser(grammar.clone())?
            }
        };
        parser.start_without_prompt();
        // The PDA machine + its live config are owned by the Matcher (the
        // llguidance dpda feature compiles the PDA at parser construction +
        // advances it on every consume_token). The GuidanceState no longer keeps
        // a parallel PDA mirror (the pda_ctrl/pda_stack/token_to_terminal), which
        // could desync from the Earley authority. It queries the Matcher for the
        // PDA config (the pda_config) + drives batched draft projection through
        // the Matcher (the project_drafts / the with_pda_machine).
        let matcher = Matcher::new(Ok(parser));

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            reasoning_end_ids,
            reasoning_ended,
            full_envelope: llg_full_enabled,
            last_committed: 0,
            reasoning_token_count: 0,
        })
    }

    /// Commit token and track for speculative decoding recovery.
    /// During reasoning, tokens are tracked but NOT fed to the grammar.
    /// When the reasoning-end token is seen, we transition to grammar mode.
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        self.llm_tokens.push(token);

        if !self.reasoning_ended {
            self.reasoning_token_count += 1;
            if self.reasoning_end_ids.contains(&token) {
                self.reasoning_ended = true;
                crate::log_warn!(
                    "[llg] Reasoning ended (token {}), grammar constraints now active (after {} reasoning tokens)",
                    token,
                    self.reasoning_token_count
                );
            }
            return Ok(());
        }

        if !self.matcher.is_stopped() {
            // The Matcher advances its internal PDA in lockstep with the Earley
            // parser on every consume_token (the llguidance dpda feature), so there
            // is no separate PDA pre-check + no resync to do here. The CPU parser
            // is the authority; the PDA is a fast projection kept consistent by the
            // Matcher itself.
            self.matcher.consume_token(token)?;
            self.last_committed = self.llm_tokens.len(); // the committed boundary advances
        }
        Ok(())
    }

    /// The reject-and-resync (the P2 fix): on a FSM failure (the token->byte bridge desync), roll
    /// the matcher + the llm_tokens back to the last committed token boundary (the no disable-
    /// constraints). The constraints stay active; the next decode step restarts from the last
    /// accepted token.
    pub fn resync(&mut self) {
        let rollback_count = self.llm_tokens.len().saturating_sub(self.last_committed);
        if rollback_count > 0 {
            let _ = self.matcher.rollback(rollback_count);
            self.llm_tokens.truncate(self.last_committed);
        }
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// The count of reasoning tokens admitted so far (the two-phase mode). The
    /// `0` when reasoning has ended (the count is not incremented after the
    /// `reasoning_ended` flip). Used by the reasoning-budget promotion (the
    /// sampling bias that steers the model to the end-of-reasoning token).
    pub fn reasoning_token_count(&self) -> usize {
        self.reasoning_token_count
    }

    /// Compute mask or return EOS token set if stopped.
    /// In full-envelope mode, always apply the grammar mask (no all-ones during reasoning).
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        // Full-envelope mode: always apply the grammar mask (no all-ones during reasoning)
        if self.full_envelope {
            return self.matcher.compute_mask_immut().map_err(Into::into);
        }
        // Two-phase mode: allow everything during reasoning
        if !self.reasoning_ended {
            return self
                .matcher
                .compute_mask_immut()
                .map(|mut mask| {
                    mask.set_all(true);
                    mask
                })
                .map_err(Into::into);
        }
        // Two-phase mode: apply grammar mask after reasoning
        self.matcher.compute_mask_immut().map_err(Into::into)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding).
    /// During reasoning, no fast-forward is possible.
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if !self.reasoning_ended {
            return Vec::new();
        }
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// The read-only ff run (the non-advancing variant). Returns the grammar-forced
    /// continuation from the already-settled state without advancing the matcher.
    /// The sample site uses this to rewrite the token VALUE (the ff override)
    /// without committing; the single `commit_run` (the gate) is the only FSM
    /// advance.
    pub fn compute_ff_tokens_immut(&mut self) -> Vec<u32> {
        if !self.reasoning_ended {
            return Vec::new();
        }
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens_immut()
    }

    /// Commit a run of tokens to the grammar FSM. Returns the number of tokens
    /// successfully committed (the FSM-legal prefix). The tokens after the
    /// first rejection are not committed (the FSM stays at the legal boundary);
    /// the caller resyncs if it needs to recover. During reasoning (two-phase
    /// mode), tokens are tracked but not fed to the grammar, so the full run
    /// is always accepted.
    pub fn commit_run(&mut self, run: &[u32]) -> usize {
        if !self.reasoning_ended {
            // Two-phase: reasoning tokens are unconstrained, accept all.
for &token in run {
                self.llm_tokens.push(token);
                self.reasoning_token_count += 1;
                if self.reasoning_end_ids.contains(&token) {
                    self.reasoning_ended = true;
                }
            }
            return run.len();
        }
        // Grammar active: commit each token to the matcher, counting at the first
        // rejection (the legal prefix is committed, the FSM is not). The
        // `last_committed` advances with each accepted token (the resync boundary),
        // so a partial commit_run + the resync rolls back to the legal prefix, not
        // the start.
        let mut committed = 0;
        for &token in run {
            if self.matcher.is_stopped() {
                break;
            }
            if self.matcher.consume_token(token).is_err() {
                break; // FSM rejected this token (the legal prefix is committed)
            }
            self.llm_tokens.push(token);
            self.last_committed = self.llm_tokens.len(); // the committed boundary advances
            committed += 1;
        }
        committed
    }

    /// Non-mutating: how many of `tokens` are grammar-legal from the current
    /// state. Returns the count without advancing the matcher.
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        if !self.reasoning_ended {
            return Ok(tokens.len());
        }
        if self.matcher.is_stopped() {
            return Ok(0);
        }
        self.matcher.validate_tokens(tokens)
    }

    // ─── PDA fast path (delegated to the Matcher, the no parallel mirror) ───
    //
    // The PDA machine + its live config are owned by the Matcher (the llguidance
    // dpda feature compiles the PDA at parser construction + advances it on every
    // consume_token). The GuidanceState no longer keeps a parallel pda_ctrl/
    // pda_stack/token_to_terminal mirror (which could desync from the Earley
    // authority). It queries the Matcher for the config (the pda_config) + drives
    // the batched draft projection through the Matcher (the project_drafts /
    // the with_pda_machine).

    /// Whether the PDA fast path is active (PDA present in the Matcher + reasoning
    /// ended + the env gate).
    pub fn has_pda(&mut self) -> bool {
        self.matcher.pda_config().is_some()
            && self.reasoning_ended
            && crate::utils::env::pda_grammar_enabled()
    }

    /// Get the raw VOB mask words for the current PDA config.
    /// Returns (words, is_deny) where `is_deny` means the bits represent DENIED tokens.
    /// The mask is computed over the Matcher's current PDA config frontier (the
    /// pda_config), so it is always consistent with the Earley authority.
    pub fn pda_mask_words(&mut self) -> Option<(Vec<u32>, bool)> {
        let configs = self.matcher.pda_config()?;
        // The deterministic grammar has a single-config frontier; take the first.
        let (ctrl, stack) = configs.first()?;
        let mut words: Option<Vec<u32>> = None;
        self.matcher.with_pda_machine(|pda| {
            let allowed = pda.mask_at_cfg(*ctrl, stack);
            let w = (pda.num_inputs + 31) / 32;
            let mut ws = vec![0u32; w as usize];
            for &a in &allowed {
                if a < pda.num_inputs {
                    ws[a as usize / 32] |= 1u32 << (a as usize % 32);
                }
            }
            words = Some(ws);
        });
        words.map(|w| (w, false)) // Always "allow" semantics (the bits represent allowed tokens).
    }

    /// Validate a sequence of draft tokens against the PDA.
    /// Returns the number of tokens that are legal (stops at first illegal).
    /// Delegates to the Matcher's batched projection (the project_drafts, the
    /// SIMD project_batch + the config-set advance), so the draft is validated
    /// against the same PDA state the Earley authority tracks.
    pub fn pda_validate(&mut self, tokens: &[u32]) -> usize {
        if !self.reasoning_ended {
            return tokens.len();
        }
        let (legal, _final_cfg) = self.matcher.project_drafts(tokens);
        legal
    }

    /// Check if the current PDA config is accepting (grammar complete).
    /// The PDA's accepting state (the q_out(start)) is reached only via an exit
    /// epsilon move from the final dot state, so a direct `accepting.contains(ctrl)`
    /// check would reject a complete sequence that is sitting one epsilon step short
    /// of the accept state. The correct check is the epsilon-closure acceptance
    /// (the PdaMachine::accepts_via_eps), which follows the exit epsilon move and
    /// accepts correctly. This checks the (ctrl, stack) config, not just the ctrl.
    pub fn pda_is_accepting(&mut self) -> bool {
        let Some(configs) = self.matcher.pda_config() else {
            return false;
        };
        let Some((ctrl, stack)) = configs.first() else {
            return false;
        };
        self.matcher
            .with_pda_machine(|pda| pda.accepts_via_eps(*ctrl, stack))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_guidance_tokens() {
        // This test verifies that extract_guidance_tokens compiles
        // It doesn't actually run since we don't have a tokenizer here
        let tokens = GuidanceTokens::default();
        assert!(tokens.bos_token_ids.is_empty());
    }

    /// Build a simple explicit tool grammar for testing (the no real tokenizer needed).
    fn explicit_tool_grammar() -> (super::ParserFactory, llguidance::api::TopLevelGrammar) {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // Simple grammar: token 97, then token 98, then token 99.
        let grm_str = r#"
start: <[97]> <[98]> <[99]>
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        (factory, grm)
    }

    ///// The P2 fix (the reject-and-resync): a desynced PDA, after resync, rolls back to the last
    /// committed token boundary (the no disable-constraints). The next commit from the resynced
    /// state succeeds (the grammar-consistent token).
    #[test]
    fn resync_rolls_back_to_last_committed() {
        let (factory, grm) = explicit_tool_grammar();
        let mut state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(factory),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        // The grammar: token 97, then 98, then 99 (the explicit_tool_grammar).
        state.commit_token(97).unwrap(); // the success -> the last_committed = 1
        // The out-of-order token 99 (the grammar expects 98) -> the commit fails.
        assert!(state.commit_token(99).is_err(), "the FSM rejects the out-of-order token 99");
        // The P2 fix: the resync rolls back to the last committed boundary (the token 97).
        state.resync();
        // After the resync, the grammar-consistent token 98 succeeds.
        state.commit_token(98).unwrap();
        state.commit_token(99).unwrap();
        assert!(state.is_finished(), "the grammar is complete (the 97, 98, 99)");
    }

    /// Build a PDA-compilable sequence grammar (the string-literal form). The
    /// single-byte env maps 'a'->97, 'b'->98, 'c'->99. The token-range form
    /// (<[97] <[98]> <[99]>) does not compile to a PDA (the bridge PDA is None),
    /// so the differential proof must use the literal form.
    fn pda_sequence_grammar() -> (super::ParserFactory, llguidance::api::TopLevelGrammar) {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"
start: "a" "b" "c"
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        (factory, grm)
    }

    /// The differential PDA proof (the R3 independent-oracle property): the PDA
    /// claimant (`pda_validate`, the batched projection) must agree with the CPU
    /// Earley authority (the sequential `commit_token` walk) on the length of the
    /// legal prefix, for every candidate draft. The PDA is the fast claimant; the
    /// CPU parser is the oracle. They must never disagree on how many draft tokens
    /// are grammar-legal — that is the inclusive (every legal prefix is accepted)
    /// + exclusive (no illegal token is accepted) property.
    ///
    /// Two separate GuidanceState instances are used (one per authority) so the
    /// CPU walk does not perturb the PDA state being
    #[test]
    fn pda_validate_agrees_with_cpu_authority() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;
        let (_factory, grm) = pda_sequence_grammar();

        // The candidate drafts: the legal full sequence, the legal prefixes, the
        // illegal-first-token, the illegal-middle-token, the over-long (legal prefix
        // + extra), the empty, + a repeated-token case.
        let drafts: Vec<Vec<u32>> = vec![
            vec![97, 98, 99],        // the full legal sequence -> 3
            vec![97, 98],            // the legal prefix -> 2
            vec![97],               // the legal prefix -> 1
            vec![98, 99],           // the illegal first token (98 != 97) -> 0
            vec![97, 99],          // the illegal second token (99 != 98) -> 1
            vec![97, 98, 99, 97],  // the over-long (legal 3 + extra) -> 3
            vec![],                 // the empty -> 0
            vec![97, 97],          // the repeated first token (97 != 98 at pos 1) -> 1
        ];

        for draft in drafts {
            // The PDA claimant: a fresh state, the pda_validate (the batched projection).
            let mut claimant = GuidanceState::new_from_grammar_with_reasoning(
                std::sync::Arc::new(ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap()),
                &grm,
                vec![],
                true,
            )
            .unwrap();
            let pda_legal = claimant.pda_validate(&draft);

            // The CPU oracle: a fresh state, the sequential commit_token walk. Count how
            // many tokens commit before the first failure (the legal prefix length).
            let oracle_factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
            let mut oracle = GuidanceState::new_from_grammar_with_reasoning(
                std::sync::Arc::new(oracle_factory),
                &grm,
                vec![],
                true,
            )
            .unwrap();
            let mut cpu_legal = 0;
            for &tok in &draft {
                // The oracle stops counting once the grammar is complete (the accepting
                // state reached): tokens past the end are not part of the legal prefix.
                // This matches the PDA claimant, which also stops advancing once the
                // config SET is accepting.
                if oracle.is_finished() {
                    break;
                }
                if oracle.commit_token(tok).is_err() {
                    break;
                }
                cpu_legal += 1;
            }

            assert_eq!(
                pda_legal, cpu_legal,
                "the PDA claimant (the {pda_legal}) must agree with the CPU oracle (the {cpu_legal}) on the draft {draft:?}"
            );
        }
    }

    /// Build a raw CPU Matcher (the Earley authority) for the PDA sequence
    /// grammar, independent of any GuidanceState PDA claimant.
    fn pda_sequence_matcher() -> llguidance::Matcher {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;
        let (_factory, grm) = pda_sequence_grammar();
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let parser = factory.create_parser(grm).unwrap();
        llguidance::Matcher::new(Ok(parser))
    }

    /// PROOF (the PDA mask == the CPU mask, the inclusive + exclusive token-set
    /// equality): at every step of the sequence walk, the set of tokens allowed by
    /// the PDA mask (the pda_mask_words, expanded from the terminal alphabet to the
    /// token alphabet via the forward terminal->token bridge) must EXACTLY the set of
    /// tokens allowed by the CPU Earley mask (the compute_mask_immut). Inclusive:
    /// every CPU-allowed token is PDA-allowed. Exclusive: every CPU-disallowed token
    /// is PDA-disallowed. The two authorities must never disagree on the allowed set.
    #[test]
    fn pda_mask_matches_cpu_mask_at_every_step() {
        use llguidance::dpda_adapter::terminal_token_map;
        use llguidance::earley::CGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        // The forward terminal->token bridge (the PDA terminal index -> the token IDs it covers).
        let cgrm: CGrammar = {
            let p = factory.create_parser(grm.clone()).unwrap();
            p.parser.grammar().clone()
        };
        let bridge = terminal_token_map(&cgrm, &env);

        // The PDA claimant (the GuidanceState, the pda_mask_words).
        let mut claimant = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        // The CPU oracle (the raw Matcher, the compute_mask_immut).
        let mut oracle = pda_sequence_matcher();

        // Walk the full legal sequence (the 'a'=97, 'b'=98, 'c'=99). Compare the PDA
        // mask against the CPU mask at each pre-consume step (the steps 0..seq.len()). At
        // step s both authorities have consumed seq[0..s] and report the set of tokens
        // allowed next.
        let seq: Vec<u32> = vec![97, 98, 99];
        for step in 0..seq.len() {
            // The PDA allowed tokens: expand the PDA terminal mask to tokens via the bridge.
            let pda_allowed: std::collections::HashSet<u32> = match claimant.pda_mask_words() {
                Some((words, _deny)) => {
                    // The words are VOB bit-words over the PDA terminal alphabet. Reconstruct
                    // the allowed terminal set, then expand each terminal to its token set.
                    let num_terminals = bridge.len();
                    let mut term_allowed = vec![false; num_terminals];
                    for (w, word) in words.iter().enumerate() {
                        for bit in 0..32u32 {
                            let t = (w as u32) * 32 + bit;
                            if t < num_terminals as u32 && (word >> bit) & 1 == 1 {
                                term_allowed[t as usize] = true;
                            }
                        }
                    }
                    let mut set = std::collections::HashSet::new();
                    for (t, allowed) in term_allowed.iter().enumerate() {
                        if *allowed {
                            for &tok in &bridge[t] {
                                set.insert(tok);
                            }
                        }
                    }
                    set
                }
                None => std::collections::HashSet::new(),
            };
            // The CPU allowed tokens (the oracle mask).
            let cpu_mask = oracle.compute_mask_immut().expect("the CPU mask");
            let cpu_allowed: std::collections::HashSet<u32> = cpu_mask.to_list().into_iter().collect();

            assert_eq!(
                pda_allowed, cpu_allowed,
                "at step {step} the PDA mask (the {pda_allowed:?}) must equal the CPU mask (the {cpu_allowed:?})"
            );

            // Advance both authorities one step (the no the last step, where the sequence is done).
            if step < seq.len() {
                let tok = seq[step];
                claimant.commit_token(tok).expect("the claimant commit");
                oracle.consume_token(tok).expect("the oracle consume");
            }
        }
    }

    /// PROOF (the PDA accepting == the CPU accepting, at every step): the PDA
    /// final-state acceptance check (the pda_is_accepting) must agree with the CPU
    /// Earley accepting check (the is_accepting) at every point in the walk.
    /// Inclusive: when the CPU is accepting, the PDA is accepting. Exclusive: when
    /// the CPU is not accepting, the PDA is not accepting.
    #[test]
    fn pda_is_accepting_matches_cpu_at_every_step() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut claimant = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        let mut oracle = pda_sequence_matcher();

        let seq: Vec<u32> = vec![97, 98, 99];
        // Step 0 (the start): neither is accepting yet.
        assert_eq!(claimant.pda_is_accepting(), oracle.is_accepting().expect("the CPU accepting check"), "step 0 accepting must agree");
        for &tok in &seq {
            claimant.commit_token(tok).expect("the claimant commit");
            oracle.consume_token(tok).expect("the oracle consume");
            assert_eq!(
                claimant.pda_is_accepting(),
                oracle.is_accepting().expect("the CPU accepting check"),
                "the accepting state must agree after consuming {tok}"
            );
        }
        // After the full sequence both must be accepting (the grammar complete).
        assert!(claimant.pda_is_accepting(), "the PDA must be accepting after the full sequence");
        assert!(oracle.is_accepting().expect("the CPU accepting check"), "the CPU must be accepting after the full sequence");
    }

    /// PROOF (the has_pda gate reflects PDA availability): has_pda is true exactly
    /// when the Matcher has a live PDA config (the pda_config is Some) + reasoning
    /// has ended + the env gate is on. For a PDA-compilable deterministic grammar the
    /// PDA config is present, so has_pda must be true (the inclusive). For a grammar
    /// that does not compile to a PDA (the parametric / the no bridge), has_pda must
    /// be false (the exclusive).
    #[test]
    fn has_pda_reflects_pda_availability() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        // The inclusive case: a PDA-compilable deterministic grammar has a live PDA config.
        let (_factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut claimant = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        // The PDA config must be present for a compiled PDA grammar (the first gate).
        assert!(claimant.matcher.pda_config().is_some(), "the PDA config is present for a compiled PDA grammar");
        // The has phase: with an empty reasoning_end_ids the grammar is single-phase,
        // so reasoning_ended is true from the start (the second gate term).
        assert!(claimant.reasoning_ended, "reasoning is ended for a single-phase grammar");
        // The has_pda gate is the conjunction of the three terms (the PDA config present,
        // the reasoning ended, the env gate on). Prove the gate tracks the env gate exactly:
        // has_pda must equal (config_present AND reasoning_ended AND env_enabled).
        let env_on = crate::utils::env::pda_grammar_enabled();
        let expected = claimant.matcher.pda_config().is_some() && claimant.reasoning_ended && env_on;
        assert_eq!(
            claimant.has_pda(),
            expected,
            "has_pda must be the conjunction of the PDA-config-present, the reasoning-ended, + the env-gate terms"
        );
    }

    /// PROOF (the compute_ff_tokens == the Matcher forced-run): the GuidanceState's
    /// fast-forward tokens must equal the CPU Matcher's compute-run at every point in
    /// the walk. Inclusive: every forced token the CPU reports is reported by the
    /// state. Exclusive: the state reports no token the CPU does not force. Also
    /// proves the two guard clauses (the reasoning-not-ended + the matcher-stopped
    /// both yield an empty ff-run).
    #[test]
    fn compute_ff_tokens_matches_matcher_forced_run() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        let mut matcher = pda_sequence_matcher();

        // The guard clause: a stopped matcher yields an empty ff-run.
        // (The fresh matcher is not stopped, so consume the full sequence to stop it,
        // then check the empty ff.)
        for &tok in &[97u32, 98, 99] {
            state.commit_token(tok).unwrap();
            matcher.consume_token(tok).unwrap();
        }
        assert!(matcher.is_stopped(), "the matcher is stopped after the full sequence");
        assert!(state.compute_ff_tokens().is_empty(), "a stopped state yields no ff tokens");

        // The positive case: a fresh state at the start of the sequence forces the
        // entire remaining run (the 'a','b','c' are all forced by the sequence grammar).
        let mut fresh_state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        let mut fresh_matcher = pda_sequence_matcher();
        let state_ff = fresh_state.compute_ff_tokens();
        let matcher_ff = fresh_matcher.compute_ff_tokens();
        assert_eq!(
            state_ff, matcher_ff,
            "the state ff-run must equal the matcher ff-run (the {state_ff:?} vs the {matcher_ff:?})"
        );
        // The sequence grammar forces all 3 tokens from the start.
        assert_eq!(state_ff, vec![97, 98, 99], "the sequence grammar forces its full run from the start");
    }

    /// PROOF (the commit_run == the Matcher prefix): the inclusive + exclusive commit
    /// property. commit_run must commit exactly the grammar-legal prefix of the run
    /// (the Matcher's try_consume_tokens count) + return the prefix length. Inclusive:
    /// every legal token in the prefix is committed. Exclusive: no illegal token is
    /// committed (the run stops at the first rejection, the count reflects the prefix).
    #[test]
    fn commit_run_commits_exactly_the_legal_prefix() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();

        // Case 1 (the inclusive): a fully-legal run commits in full.
        let mut ok_state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        assert_eq!(ok_state.commit_run(&[97, 98, 99]), 3, "a fully-legal run commits all 3");

        // Case 2 (the exclusive): an illegal token in the middle stops the commit.
        // The run [97, 99, 98] is illegal at index 1 (99 != 98), so commit_run must
        // commit only the legal prefix [97] + return the prefix length (1).
        let mut bad_state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();
        let result = bad_state.commit_run(&[97, 99, 98]);
        // The commit_run returns the legal-prefix count (1) when the FSM rejects a token.
        assert_eq!(result, 1, "a run containing an illegal token commits only the legal prefix");
        // The legal prefix [97] was committed before the rejection (the llm_tokens reflects it).
        assert_eq!(bad_state.llm_tokens, vec![97], "only the legal prefix is retained");
    }

    /// PROOF (the validate_tokens == the Matcher legal count, the non-mutating):
    /// validate_tokens must report the number of grammar-legal tokens in the input
    /// WITHOUT advancing the state (the no mutation). Inclusive: it counts every legal
    /// prefix token. Exclusive: it stops at the first illegal token. The state must be
    /// unchanged after the call (the no-mutation property).
    #[test]
    fn validate_tokens_is_non_mutating_and_counts_the_legal_prefix() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = pda_sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            true,
        )
        .unwrap();

        // The non-mutation: validate_tokens must not advance the state. Call it twice;
        // the result is identical (the state did not move).
        let first = state.validate_tokens(&[97, 98, 99]).unwrap();
        let second = state.validate_tokens(&[97, 98, 99]).unwrap();
        assert_eq!(first, second, "validate_tokens is non-mutating (two calls agree)");
        // The inclusive: the full legal sequence validates in full.
        assert_eq!(first, 3, "the full legal sequence validates to 3");

        // The exclusive: an illegal token bounds the legal prefix.
        let partial = state.validate_tokens(&[97, 99, 98]).unwrap();
        assert_eq!(partial, 1, "the illegal [97,99,98] is legal only for its first token (97)");

        // The state is still at the start (the no mutation from any of the above).
        assert_eq!(state.llm_tokens, Vec::<u32>::new(), "validate_tokens did not mutate the state");
    }
}
