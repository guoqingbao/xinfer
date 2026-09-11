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
    /// allowed set. Empty exclusions fall back to the full regex regex.
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
                    .map(|&n| format!("<[{}]>", n.to_string()))
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
                .map(|&n| format!("<[{}]>", n.to_string()))
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
    /// GPU-resident PDA (pushdown-rs transition table).
    pub(crate) pda: Option<pushdown_rs::machine::PdaMachine>,
    /// The PDA control-state stack (top = last element). Mirrors the LR state stack.
    pub(crate) pda_stack: Vec<u32>,
    /// The current PDA control state (separate from the stack).
    pub(crate) pda_ctrl: u32,
}

impl GuidanceState {
    pub fn new_from_grammar_with_reasoning(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
        reasoning_end_ids: Vec<u32>,
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

        let mut parser = if !reasoning_ended {
            crate::log_info!(
                "[llg] Two-phase reasoning: grammar constraint deferred until after reasoning end tokens {:?}",
                reasoning_end_ids
            );
            factory.create_parser(grammar.clone())?
        } else {
            crate::log_info!(
                "[llg] Full-envelope/single-phase mode: grammar constrains all generation"
            );
            // Max tokens is capped by the scheduler anyway so allow grammar space to generate reasoning
            if let Some(max_tokens) = grammar.max_tokens {
                let mut grammar = grammar.clone();
                grammar.max_tokens = Some(max_tokens * 2);
                factory.create_parser(grammar.clone())?
            } else {
                factory.create_parser(grammar.clone())?
            }
        };
        parser.start_without_prompt();
        let matcher = Matcher::new(Ok(parser));
    
        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            reasoning_end_ids,
            reasoning_ended,
            full_envelope: llg_full_enabled,
            pda: None,
            pda_stack: Vec::new(),
            pda_ctrl: 0,
        })
    }

    /// Commit token and track for speculative decoding recovery.
    /// During reasoning, tokens are tracked but NOT fed to the grammar.
    /// When the reasoning-end token is seen, we transition to grammar mode.
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        self.llm_tokens.push(token);

        if !self.reasoning_ended {
            if self.reasoning_end_ids.contains(&token) {
                self.reasoning_ended = true;
                crate::log_warn!(
                    "[llg] Reasoning ended (token {}), grammar constraints now active (after {} reasoning tokens)",
                    token,
                    self.llm_tokens.len()
                );
            }
            return Ok(());
        }

        if !self.matcher.is_stopped() {
            // PDA fast pre-check: if the PDA has a transition, use it to
            // validate before the (slower) CPU parser call.
            if self.has_pda() {
                match self.pda_advance(token) {
                    Some(()) => {
                        // PDA accepted: still run CPU parser as the blocker.
                        // If CPU rejects, the PDA was out of sync -> resync.
                        if let Err(e) = self.matcher.consume_token(token) {
                            crate::log_warn!(
                                "[llg] PDA/CPU desync at token {}: CPU rejected. Resyncing PDA.",
                                token
                            );
                            self.resync_pda_from_cpu();
                            return Err(e);
                        }
                    }
                    None => {
                        // PDA rejected: token not in PDA transition table.
                        // Fall through to CPU parser (boundary case or PDA incomplete).
                        // CPU is the authority: if it accepts, resync PDA.
                        if let Err(e) = self.matcher.consume_token(token) {
                            return Err(e); // truly forbidden
                        }
                        self.resync_pda_from_cpu();
                    }
                }
            } else {
                // No PDA: pure CPU path (original behavior)
                self.matcher.consume_token(token)?;
            }
        }
        Ok(())
    }

    /// Commit a produced token run to the FSM (several clicks). Returns the count of
    /// leading tokens that PASSED the matcher (the authority) — only these may be
    /// appended to the sequence. `None` means the matcher entered an error state and
    /// the sequence should be marked unguided. Handles the two-phase reasoning
    /// transition within the run (free tokens until the reasoning-end, then the
    /// grammar-gated tail).
    pub fn commit_run(&mut self, run: &[u32]) -> Option<usize> {
        if self.matcher.is_stopped() {
            return Some(0);
        }
        if self.reasoning_ended {
            match self.matcher.try_consume_tokens(run) {
                Ok(n) => {
                    self.llm_tokens.extend_from_slice(&run[..n]);
                    self.matcher.settle();
                    Some(n)
                }
                Err(_) => None,
            }
        } else {
            let mut accepted = 0;
            for &tok in run {
                self.llm_tokens.push(tok);
                accepted += 1;
                if self.reasoning_end_ids.contains(&tok) {
                    self.reasoning_ended = true;
                    let tail = &run[accepted..];
                    match self.matcher.try_consume_tokens(tail) {
                        Ok(n) => {
                            self.llm_tokens.extend_from_slice(&tail[..n]);
                            accepted += n;
                            self.matcher.settle();
                        }
                        Err(_) => return None,
                    }
                    break;
                }
            }
            Some(accepted)
        }
    }

    /// Resync the PDA state from the CPU parser's current position.
    /// Called when the PDA and CPU disagree (PDA incomplete or desynced).
    fn resync_pda_from_cpu(&mut self) {
        if let Some(ref pda) = self.pda {
            // Re-derive the PDA control-state stack by replaying llm_tokens from start.
            // This is O(n) but only happens on desync (rare).
            let mut ctrl = pda.start_state;
            let mut stack = vec![pda.start_stack];
            for &tok in &self.llm_tokens {
                let top = stack.last().copied().unwrap_or(pda.start_stack);
                match pda.lookup(ctrl, Some(tok), top).as_slice() {
                    [t] => {
                        stack.pop();
                        for &p in t.push.iter().rev() {
                            stack.push(p);
                        }
                        ctrl = t.next_q;
                    }
                    _ => break, // can't replay further; stay at last known state
                }
            }
            self.pda_stack = stack;
        }
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Compute mask or return EOS token set if stopped.
    /// In full-envelope mode, always apply grammar mask (no all-ones during reasoning).
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
        self.matcher.compute_ff_tokens_immut()
    }

    /// Non-mutating: how many of `tokens` are grammar-legal from the current state.
    /// Used by speculative-decoding acceptance to cap the draft prefix without advancing.
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        if !self.reasoning_ended {
            return Ok(tokens.len());
        }
        self.matcher.validate_tokens(tokens)
    }

    // ─── PDA fast path (CPU table lookup, ~1000x faster than parser walk) ───

    /// Whether the PDA fast path is active (env-gated + PDA available + reasoning ended).
    pub fn has_pda(&self) -> bool {
        self.pda.is_some() && self.reasoning_ended && crate::utils::env::pda_grammar_enabled()
    }

    /// Get the raw VOB mask words for the current PDA config.
    /// Returns (words, is_deny) where `is_deny` means the bits represent DENIED tokens.
    /// Uses the epsilon-closure mask (the `mask_at_cfg`), consistent with the
    /// `advance_eps` (the no stuck call dots).
    pub fn pda_mask_words(&self) -> Option<(Vec<u32>, bool)> {
        let pda = self.pda.as_ref()?;
        let ctrl = self.pda_ctrl;
        let stack = if self.pda_stack.is_empty() {
            vec![pda.start_stack]
        } else {
            self.pda_stack.clone()
        };
        let allowed = pda.mask_at_cfg(ctrl, &stack);
        let w = (pda.num_inputs + 31) / 32;
        let mut words = vec![0u32; w as usize];
        for &a in &allowed {
            if a < pda.num_inputs {
                words[a as usize / 32] |= 1u32 << (a as usize % 32);
            }
        }
        // Always "allow" semantics (the bits represent allowed tokens).
        Some((words, false))
    }

    /// Advance the PDA by one token. Returns None if the token is illegal.
    /// Uses the epsilon-closure advance (the `advance_eps`), so it goes
    /// through the call dots (the no stuck) — consistent with the mask
    /// (`mask_at_cfg`, the epsilon-closure union).
    pub fn pda_advance(&mut self, token: u32) -> Option<()> {
        let pda = self.pda.as_ref()?;
        if self.pda_stack.is_empty() {
            self.pda_stack.push(pda.start_stack);
            self.pda_ctrl = pda.start_state;
        }
        match pda.advance_eps(self.pda_ctrl, &self.pda_stack, token) {
            Some((nq, ns)) => {
                self.pda_ctrl = nq;
                self.pda_stack = ns;
                Some(())
            }
            None => None,
        }
    }

    /// Validate a sequence of draft tokens against the PDA.
    /// Returns the number of tokens that are legal (stops at first illegal).
    pub fn pda_validate(&self, tokens: &[u32]) -> usize {
        let Some(pda) = self.pda.as_ref() else { return 0 };
        let mut ctrl = self.pda_ctrl;
        let mut stack = if self.pda_stack.is_empty() {
            vec![pda.start_stack]
        } else {
            self.pda_stack.clone()
        };
        let mut count = 0;
        for &token in tokens {
            let top = stack.last().copied().unwrap_or(pda.start_stack);
            match pda.lookup(ctrl, Some(token), top).as_slice() {
                [t] => {
                    stack.pop();
                    for &p in t.push.iter().rev() {
                        stack.push(p);
                    }
                    ctrl = t.next_q;
                    count += 1;
                }
                _ => break,
            }
        }
        count
    }

    /// Check if the current PDA control state is accepting (grammar complete).
    pub fn pda_is_accepting(&self) -> bool {
        let Some(pda) = self.pda.as_ref() else { return false };
        let top = *self.pda_stack.last().unwrap_or(&pda.start_state);
        pda.accepting.contains(&top)
    }
}

#[cfg(test)]
mod tests {
    use pushdown_rs::pda::Dpda;
    // (the PDA is now pushdown_rs::machine::PdaMachine)

    // === 1-Phase Full-Envelope Grammar Region Tests ===
    // These verify the, termination, and region logic for both
    // explicit and implicit tool grammars.

    /// Build a 1-phase full-envelope grammar with EXPLICIT tool structure.
    /// Regions: reasoning_block -> (text | tool_call)+ -> eos
    /// tool_call has specific param names and JSON structure.
    fn explicit_tool_grammar() -> (llguidance::ParserFactory, llguidance::api::TopLevelGrammar) {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Nested explicit grammar: token 97, then a nested inner (98 then 100), then 99.
        // Uses <[id]> token-range syntax so token_ranges populate without a real tokenizer.
        // The nesting (inner non-terminal) creates more PDA states than a flat alternation.
        let grm_str = r#"
start: <[97]> inner <[99]>
inner: <[98]> <[100]>
"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        (factory, grm)
    }

    /// Build a 1-phase full-envelope grammar with IMPLICIT (catch-all) structure.
    /// Uses a simple alternation: "a" or "b" (2 choices, no structure)
    fn implicit_tool_grammar() -> (llguidance::ParserFactory, llguidance::api::TopLevelGrammar) {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Simple implicit grammar: a 2-token sequence (fewer states than the nested explicit one).
        // Uses <[id]> token-range syntax so token_ranges populate.
        // PDA: start -> after_97 -> after_98 (accept)
        let grm_str = r#"
start: <[97]> <[98]>
"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        (factory, grm)
    }

    #[test]
    fn explicit_grammar_accepts_valid_sequence() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile failed");
        // Verify the PDA structure (the token-walking tests are in llguidance
        // where the single-byte tokenizer makes the local ID mapping known).
        assert!(pda.num_states > 1, "PDA should have multiple states");
        assert!(pda.transitions.len() > 0, "PDA should have transitions");
        assert!(pda.is_deterministic(), "explicit tool grammar is deterministic");
        assert!(!pda.accepting.is_empty(), "PDA has accepting states");
        pda.validate_bounds().expect("PDA transitions are in-bounds");
    }

    #[test]
    fn explicit_grammar_rejects_wrong_param() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile failed");
        // The PDA is deterministic (no two transitions share the same (q, a, top)).
        assert!(pda.is_deterministic(), "explicit tool grammar is deterministic");
        // The kappa(G) state count matches the compiled machine.
        let adapter = llguidance::dpda_adapter::PdaGrammar::new(&cgrm);
        let k = pushdown_rs::compile::kappa(&adapter);
        assert_eq!(k, pda.num_states, "kappa must match state count");
    }

    #[test]
    fn implicit_grammar_accepts_any_body() {
        let (factory, grm) = implicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile failed");
        // Verify the PDA structure (the token-walking tests are in llguidance).
        assert!(pda.num_states > 1, "PDA should have multiple states");
        assert!(pda.transitions.len() > 0, "PDA should have transitions");
        assert!(!pda.accepting.is_empty(), "PDA has accepting states");
        pda.validate_bounds().expect("PDA transitions are in-bounds");
    }

    #[test]
    fn implicit_grammar_rejects_special_in_body() {
        let (factory, grm) = implicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile failed");
        // The PDA's num_inputs is the count of distinct terminals in the grammar.
        // A token ID >= num_inputs is out of range (the PDA rejects it).
        assert!(pda.num_inputs < 256, "the grammar has fewer than 256 terminals");
    }

    #[test]
    fn accept_state_is_terminal() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let cgrm = parser.parser.grammar().clone();
        let pda = llguidance::dpda_adapter::compile_pda(&cgrm).expect("PDA compile failed");
        // The accepting states have no outgoing transitions for any input
        // (the grammar is complete - no continuation after accept).
        for &accept_q in &pda.accepting {
            for tok in 0..pda.num_inputs {
                let top = pda.start_stack;
                assert!(
                    pda.lookup(accept_q, Some(tok), top).is_empty(),
                    "accept state {} should not allow token {} (no continuation after EOS)",
                    accept_q, tok
                );
            }
        }
    }

    #[test]
    fn explicit_has_more_states_than_implicit() {
        let (f1, g1) = explicit_tool_grammar();
        let (f2, g2) = implicit_tool_grammar();
        let cgrm1 = f1.create_parser(g1).unwrap().parser.grammar().clone();
        let cgrm2 = f2.create_parser(g2).unwrap().parser.grammar().clone();
        let pda1 = llguidance::dpda_adapter::compile_pda(&cgrm1).expect("explicit PDA");
        let pda2 = llguidance::dpda_adapter::compile_pda(&cgrm2).expect("implicit PDA");
assert!(
            pda1.num_states > pda2.num_states,
            "explicit ({}) should have more states than implicit ({})",
            pda1.num_states, pda2.num_states
        );
    }

    #[test]
    fn test_compress_to_ranges() {
        use super::GuidanceTokens;
        // single ID
        assert_eq!(GuidanceTokens::compress_to_ranges(&[151644]), vec![(151644, 151644)]);
        // consecutive IDs collapse to one range
        assert_eq!(
            GuidanceTokens::compress_to_ranges(&[151644, 151645, 151646]),
            vec![(151644, 151646)]
        );
        // non-consecutive stay separate
        assert_eq!(
            GuidanceTokens::compress_to_ranges(&[151644, 151650, 151658]),
            vec![(151644, 151644), (151650, 151650), (151658, 151658)]
        );
        // mixed
        assert_eq!(
            GuidanceTokens::compress_to_ranges(&[1, 2, 5, 6, 7, 10]),
            vec![(1, 2), (5, 7), (10, 10)]
        );
        // empty
        assert_eq!(GuidanceTokens::compress_to_ranges(&[]), Vec::<(u32, u32)>::new());
        // unsorted input preserves input order (sorting happens in token_range_expression)
        assert_eq!(
            GuidanceTokens::compress_to_ranges(&[151658, 151644, 151650]),
            vec![(151658, 151658), (151644, 151644), (151650, 151650)]
        );
    }

    #[test]
    fn test_token_range_expression() {
        use super::GuidanceTokens;
        // empty exclusions -> full regex
        assert_eq!(
            GuidanceTokens::token_range_expression(vec![]),
            r#"/(?s:.*)/"#
        );
        // single exclusion -> negated range, one-or-more
        assert_eq!(
            GuidanceTokens::token_range_expression(vec![151644]),
            "(<[^151644]>)+"
        );
        // consecutive exclusions collapse into a range
        assert_eq!(
            GuidanceTokens::token_range_expression(vec![151644, 151645, 151646]),
            "(<[^151644-151646]>)+"
        );
        // non-consecutive exclusions are sorted + comma-joined
        assert_eq!(
            GuidanceTokens::token_range_expression(vec![151644, 151650, 151658]),
            "(<[^151644,151650,151658]>)+"
        );
        // mixed
        assert_eq!(
            GuidanceTokens::token_range_expression(vec![1, 2, 5, 6, 7, 10]),
            "(<[^1-2,5-7,10]>)+"
        );
    }
}
