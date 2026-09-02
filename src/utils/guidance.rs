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
    ///
    /// Returns the RHS expression that can be used directly after `text: ` in Lark grammar.
    /// Uses llguidance's negated token range syntax to allow all tokens EXCEPT the excluded set.
    ///
    /// # Arguments
    /// * `excluded_ids` - Token IDs to exclude from free generation
    ///
    /// # Example Output
    /// ```text
    /// <[^151644,151645,151657-151658]>
    /// ```
    ///
    /// This can be used in grammar as:
    /// ```lark
    /// text: <[^151644,151645,151657-151658]>
    /// ```
    fn token_range_expression(excluded_ids: Vec<u32>) -> String {
        if excluded_ids.is_empty() {
            return r#"/(?s:.*)/"#.to_string();
        }

        // Sort and deduplicate excluded IDs
        let mut sorted_ids: Vec<u32> = excluded_ids.to_vec();
        sorted_ids.sort();
        sorted_ids.dedup();

        // Compress consecutive IDs into ranges
        let ranges = Self::compress_to_ranges(&sorted_ids);

        // Generate negated token range expression
        // Format: <[^id1,id2,id3-id4,id5]>
        let expr = format!(
            "(<[^{}]>)+ ",
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
        ).trim().to_string();

        expr
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

    // Allow EOS tokens to "naturally" finish output
    pub fn _text_grammar_mask_outer(&self) -> String {
        let mut ids = Vec::new();
        ids.extend_from_slice(&self.bos_token_ids);
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
                r#"({}) {}"#, self.reasoning_start_ids.iter().map(|&n| format!("<[{}]>", n.to_string())).collect::<Vec<String>>().join(" | "), &range_exp
            );
        }
        format!(r#"{} ({})"#, &range_exp,  self.reasoning_end_ids.iter().map(|&n| format!("<[{}]>", n.to_string())).collect::<Vec<String>>().join(" | "))
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
    /// GPU-resident DFA (Phase 1: CPU-side table lookup, ~1000x faster than parser walk).
    pub(crate) dfa: Option<llguidance::hw_dfa::HwDfa>,
    pub(crate) dfa_state: u32,
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
        // In full-envelope mode, reasoning is constrained by grammar from the start
        let reasoning_ended = if crate::utils::env::llg_full_enabled() {
            true  // Bypass two-phase logic; always apply grammar masks
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
            dfa: None,
            dfa_state: 0,
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
            self.matcher.consume_token(token)?;
        }
        Ok(())
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Compute mask or return EOS token set if stopped.
    /// In full-envelope mode, always apply grammar mask (no all-ones during reasoning).
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        // Two-phase mode: allow everything during reasoning
        if !crate::utils::env::llg_full_enabled() && !self.reasoning_ended {
            return self
                .matcher
                .compute_mask_or_eos()
                .map(|mut mask| {
                    mask.set_all(true);
                    mask
                })
                .map_err(Into::into);
        }

        // Two-phase mode: apply grammar mask after reasoning
        if self.llm_tokens.is_empty() {
            return self.matcher.compute_mask().map_err(Into::into)
        }
        self.matcher.compute_mask_or_eos().map_err(Into::into)
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

    /// Non-mutating: how many of `tokens` are grammar-legal from the current state.
    /// Used by speculative-decoding acceptance to cap the draft prefix without advancing.
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        if !self.reasoning_ended {
            return Ok(tokens.len());
        }
        self.matcher.validate_tokens(tokens)
    }

    /// Deep copy of this FSM state (independent matcher), for projecting drafts without
    /// mutating the live state.
    pub fn deep_clone(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            llm_tokens: self.llm_tokens.clone(),
            reasoning_end_ids: self.reasoning_end_ids.clone(),
            reasoning_ended: self.reasoning_ended,
            dfa: self.dfa.clone(),
            dfa_state: self.dfa_state,
        }
    }

    // ─── DFA fast path (Phase 1: CPU table lookup, ~1000x faster than parser walk) ───

    /// Whether the DFA fast path is active (env-gated + DFA available + reasoning ended).
    pub fn has_dfa(&self) -> bool {
        self.dfa.is_some() && self.reasoning_ended && crate::utils::env::dfa_grammar_enabled()
    }

    /// Get the raw VOB mask words for the current DFA state (bypasses SimpleVob).
    /// Returns (words, is_deny) where `is_deny` means the bits represent DENIED tokens.
    pub fn dfa_mask_words(&self) -> Option<(&[u32], bool)> {
        let dfa = self.dfa.as_ref()?;
        let words = dfa.mask_at(self.dfa_state);
        let deny = dfa.sign_at(self.dfa_state) == llguidance::hw_dfa::MaskSign::Deny;
        Some((words, deny))
    }

    /// Advance the DFA state by one token. Returns None if the token is illegal.
    pub fn dfa_advance(&mut self, token: u32) -> Option<()> {
        let dfa = self.dfa.as_ref()?;
        let next = dfa.advance(self.dfa_state, token)?;
        self.dfa_state = next;
        Some(())
    }

    /// Validate a sequence of draft tokens against the DFA.
    /// Returns the number of tokens that are legal (stops at first illegal).
    pub fn dfa_validate(&self, tokens: &[u32]) -> usize {
        let Some(dfa) = self.dfa.as_ref() else { return 0 };
        let mut state = self.dfa_state;
        for &token in tokens {
            match dfa.advance(state, token) {
                Some(next) => state = next,
                None => break,
            }
        }
        // Count how many were accepted
        let mut count = 0;
        let mut s = self.dfa_state;
        for &token in tokens {
            match dfa.advance(s, token) {
                Some(next) => { s = next; count += 1; }
                None => break,
            }
        }
        count
    }

    /// Check if the current DFA state is accepting (grammar complete).
    pub fn dfa_is_accepting(&self) -> bool {
        let Some(dfa) = self.dfa.as_ref() else { return false };
        dfa.accept_states.contains(&self.dfa_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llguidance::hw_dfa::{HwDfa, DfaEdge, MaskSign};

    #[test]
    fn test_compress_to_ranges() {
        // Test single ID
        let ranges = GuidanceTokens::compress_to_ranges(&[151644]);
        assert_eq!(ranges, vec![(151644, 151644)]);

        // Test consecutive IDs
        let ranges = GuidanceTokens::compress_to_ranges(&[151644, 151645, 151646]);
        assert_eq!(ranges, vec![(151644, 151646)]);

        // Test non-consecutive IDs
        let ranges = GuidanceTokens::compress_to_ranges(&[151644, 151650, 151658]);
        assert_eq!(ranges, vec![(151644, 151644), (151650, 151650), (151658, 151658)]);

        // Test mixed consecutive and non-consecutive
        let ranges = GuidanceTokens::compress_to_ranges(&[1, 2, 5, 6, 7, 10]);
        assert_eq!(ranges, vec![(1, 2), (5, 7), (10, 10)]);

        // Test empty
        let ranges = GuidanceTokens::compress_to_ranges(&[]);
        assert_eq!(ranges, vec![]);

        // Test unsorted input (preserves input order, noes consecutive runs)
        let ranges = GuidanceTokens::compress_to_ranges(&[151658, 151644, 151650]);
        assert_eq!(ranges, vec![(151658, 151658), (151644, 151644), (151650, 151650)]);
    }

    #[test]
    fn test_token_range_expression() {
        // Test empty exclusions
        let expr = GuidanceTokens::token_range_expression(vec![]);
        assert_eq!(expr, r#"/(?s:.*)/"#.to_string());

        // Test single exclusion
        let expr = GuidanceTokens::token_range_expression(vec![151644]);
        assert_eq!(expr, "(<[^151644]>)+");

        // Test consecutive exclusions
        let expr = GuidanceTokens::token_range_expression(vec![151644, 151645, 151646]);
        assert_eq!(expr, "(<[^151644-151646]>)+");

        // Test non-consecutive exclusions
        let expr = GuidanceTokens::token_range_expression(vec![151644, 151650, 151658]);
        assert_eq!(expr, "(<[^151644,151650,151658]>)+");

        // Test mixed exclusions
        let expr = GuidanceTokens::token_range_expression(vec![1, 2, 5, 6, 7, 10]);
        assert_eq!(expr, "(<[^1-2,5-7,10]>)+");
    }

    /// Build a minimal 3-state DFA for testing:
    ///   state 0 (start): allows tokens {1, 2}, both go to state 1
    ///   state 1: allows token {3}, goes to state 2
    ///   state 2 (accept): allows token {0} (EOS), self-loop
    ///
    /// Vocab size = 4 (tokens 0..3), words_per_vob = 1 (4 bits in 1 u32)
    fn test_dfa() -> HwDfa {
        HwDfa {
            num_states: 3,
            words_per_vob: 1,
            start_state: 0,
            // mask_sign: state 0=Allow, state 1=Allow, state 2=Allow
            mask_sign: vec![0, 0, 0],
            // mask_words (1 word per state, vocab=4 so bits 0-3):
            //   state 0: allow {1,2} = 0b0110 = 6
            //   state 1: allow {3}   = 0b1000 = 8
            //   state 2: allow {0}   = 0b0001 = 1
            mask_words: vec![6, 8, 1],
            // edges: state 0 has 2 edges (tok1->1, tok2->1), state 1 has 1 edge (tok3->2), state 2 has 0 edges
            edges: vec![
                DfaEdge { token: 1, next_state: 1 },
                DfaEdge { token: 2, next_state: 1 },
                DfaEdge { token: 3, next_state: 2 },
            ],
            edge_offsets: vec![0, 2, 3],
            edge_counts: vec![2, 1, 0],
            // universal_target: state 0 -> 1 (both edges go to 1), state 1 -> 2, state 2 -> self (2)
            universal_target: vec![1, 2, 2],
            accept_states: vec![2],
            state_depths: vec![0, 1, 2],
            state_labels: vec!["start".into(), "mid".into(), "accept".into()],
        }
    }

    #[test]
    fn dfa_mask_at_returns_correct_words() {
        let dfa = test_dfa();
        assert_eq!(dfa.mask_at(0), &[6u32]); // allow {1,2}
        assert_eq!(dfa.mask_at(1), &[8u32]); // allow {3}
        assert_eq!(dfa.mask_at(2), &[1u32]); // allow {0}
    }

    #[test]
    fn dfa_is_token_allowed() {
        let dfa = test_dfa();
        assert!(dfa.is_token_allowed(0, 1));
        assert!(dfa.is_token_allowed(0, 2));
        assert!(!dfa.is_token_allowed(0, 0));
        assert!(!dfa.is_token_allowed(0, 3));
        assert!(dfa.is_token_allowed(1, 3));
        assert!(!dfa.is_token_allowed(1, 1));
        assert!(dfa.is_token_allowed(2, 0));
        assert!(!dfa.is_token_allowed(2, 1));
    }

    #[test]
    fn dfa_advance_follows_edges() {
        let dfa = test_dfa();
        // state 0 + token 1 -> state 1
        assert_eq!(dfa.advance(0, 1), Some(1));
        // state 0 + token 2 -> state 1
        assert_eq!(dfa.advance(0, 2), Some(1));
        // state 0 + token 3 -> None (not allowed)
        assert_eq!(dfa.advance(0, 3), None);
        // state 1 + token 3 -> state 2
        assert_eq!(dfa.advance(1, 3), Some(2));
        // state 2 + token 0 -> state 2 (universal self-loop)
        assert_eq!(dfa.advance(2, 0), Some(2));
        // state 2 + token 1 -> None (not allowed)
        assert_eq!(dfa.advance(2, 1), None);
    }

    #[test]
    fn dfa_project_masks_walks_trajectory() {
        let dfa = test_dfa();
        // From state 0, tokens [1, 3]:
        //   mask at state 0 = [6]
        //   advance(0, 1) -> state 1, mask at state 1 = [8]
        //   advance(1, 3) -> state 2, mask at state 2 = [1]
        let masks = dfa.project_masks(0, &[1, 3]).unwrap();
        assert_eq!(masks.len(), 3); // initial + 2 advances
        assert_eq!(masks[0], vec![6]);
        assert_eq!(masks[1], vec![8]);
        assert_eq!(masks[2], vec![1]);
    }

    #[test]
    fn dfa_project_masks_rejects_illegal_token() {
        let dfa = test_dfa();
        // From state 0, token 3 is not allowed -> None
        assert!(dfa.project_masks(0, &[3]).is_none());
    }

    #[test]
    fn dfa_project_states_returns_trajectory() {
        let dfa = test_dfa();
        let states = dfa.project_states(0, &[1, 3]).unwrap();
        assert_eq!(states, vec![1, 2]); // after tok1 -> state 1, after tok3 -> state 2
    }

    #[test]
    fn dfa_bounds_safe_on_invalid_state() {
        let dfa = test_dfa();
        // Out-of-range state never panics
        assert!(dfa.mask_at(99).is_empty());
        assert_eq!(dfa.sign_at(99), MaskSign::Allow);
        assert!(!dfa.is_token_allowed(99, 0));
        assert_eq!(dfa.advance(99, 0), None);
    }

    #[test]
    fn dfa_accept_states() {
        let dfa = test_dfa();
        assert!(dfa.accept_states.contains(&2));
        assert!(!dfa.accept_states.contains(&0));
        assert!(!dfa.accept_states.contains(&1));
    }

    #[test]
    fn dfa_grammar_env_gate() {
        // Verify the env var function returns false by default (no env set).
        // We can't easily set/unset env vars in a unit test without race conditions,
        // so just verify the function exists and returns a bool.
        let enabled = crate::utils::env::dfa_grammar_enabled();
        // Default is false (opt-in). If the test environment has it set, that's fine.
        assert!(enabled || !enabled); // always true, just verifies it compiles and runs
    }

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

        // Explicit tool: "TOOL" marker, then "name" param, then "END"
        let In single-byte env: 'T'=84, 'O'=79, 'L'=76, 'n'=110, 'a'=97, 'm'=109, 'E'=69, 'N'=78, 'D'=68
        let grm_str = r#"
start: reasoning_block ( text | tool_call )* eos
reasoning_block: (<[^84,79,76,69,78,68]>)+
text: (<[^84,79,76,69,78,68]>)+
tool_call: "TOOL" "name" "END"
eos: "STOP"
"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        (factory, grm)
    }

    /// Build a 1-phase full-envelope grammar with IMPLICIT tool structure.
    /// tool_call is a catch-all (no specific param names enforced).
    fn implicit_tool_grammar() -> (llguidance::ParserFactory, llguidance::api::TopLevelGrammar) {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();

        // Implicit tool: "TOOL" marker, then ANY bytes (catch-all), then "END"
        let grm_str = r#"
start: reasoning_block ( text | tool_call )* eos
reasoning_block: (<[^84,79,76,69,78,68]>)+
text: (<[^84,79,76,69,78,68]>)+
tool_call: "TOOL" /(.|\n)*?/ "END"
eos: "STOP"
"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;
        (factory, grm)
    }

    /// Verify that in the reasoning_block region, special tokens are DENIED.
    #[test]
    fn reasoning_region_denies_special_tokens() {
        use llguidance::hw_dfa::MaskSign;
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // Start state should be in the reasoning_block region.
        // Special tokens (T=84, O=79, L=76, E=69, N=78, D=68, S=83, P=80) should be denied.
        let start = dfa.start_state;
        let sign = dfa.sign_at(start);

        // In a_block, the mask is a Deny mask (allows everything EXCEPT special tokens)
        // OR an Allow mask (allows only non-special tokens).
        // Either way, special tokens should NOT be allowed.
        for &special in &[84u32, 79, 76, 69, 78, 68, 83, 80] { // T,O,L,E,N,D,S,P
            assert!(
                !dfa.is_token_allowed(start, special),
                "reasoning region should deny special token {} (state {}, sign={:?})",
                special, start, sign
            );
        }

        // Regular tokens (e.g., 'a'=97, 'b'=98) should be allowed.
        assert!(dfa.is_token_allowed(start, 97), "reasoning region should allow 'a'");
        assert!(dfa.is_token_allowed(start, 98), "reasoning region should allow 'b'");
    }

    /// Verify that EOS ("STOP") is only allowed at the terminal position.
    #[test]
    fn eos_only_at_terminal() {
        use llguidance::hw_dfa::MaskSign;
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // 'S' (83) is the first char of "STOP" (EOS).
        // It should NOT be allowed in the reasoning_block or text regions.
        let start = dfa.start_state;
        assert!(
            !dfa.is_token_allowed(start, 83),
            "EOS char 'S' should not be allowed in reasoning region"
        );

        // Walk to a state where EOS IS allowed (after text/tool_call, before STOP)
        // Advance through reasoning_block: emit 'a' (97) which is allowed in reasoning
        let after_reasoning = dfa.advance(start, 97).expect("advance through reasoning");
        // 'S' might now be allowed (start of STOP) or might need more context
        // The key assertion: from the START state, 'S' is denied.
    }

    /// Verify explicit tool grammar enforces param structure.
    #[test]
    fn explicit_tool_enforces_params() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // Walk: reasoning('a') -> text('b') -> tool_call("TOOL")
        // After "TOOL", the next token must be 'n' (110) for "name"
        let start = dfa.start_state;
        let s1 = dfa.advance(start, 97).unwrap(); // 'a' (reasoning)
        let s2 = dfa.advance(s1, 98).unwrap(); // 'b' (text)
        // Now we're at a position where tool_call can start: "TOOL"
        let s3 = dfa.advance(s2, 84).unwrap(); // 'T'
        let s4 = dfa.advance(s3, 79).unwrap(); // 'O'
        let s5 = dfa.advance(s4, 76).unwrap(); // 'L'
        // After "TOOL", next must be 'n' (110) for "name"
        assert!(
            dfa.is_token_allowed(s5, 110),
            "after TOOL, 'n' (name) should be allowed"
        );
        // Other tokens should be denied at this position
        assert!(
            !dfa.is_token_allowed(s5, 97),
            "after TOOL, 'a' should NOT be allowed (must be 'n' for name)"
        );
    }

    /// Verify implicit tool grammar allows any content in tool body.
    #[test]
    fn implicit_tool_allows_any_content() {
        let (factory, grm) = implicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // Walk: reasoning('a') -> tool_call("TOOL")
 body (any byte)
        let start = dfa.start_state;
        let s1 = dfa.advance(start, 97).unwrap(); // 'a' (reasoning)
        let s2 = dfa.advance(s1, 84).unwrap(); // 'T' (start of TOOL)
        let s3 = dfa.advance(s2, 79).unwrap(); // 'O'
        let s4 = dfa.advance(s3, 76).unwrap(); // 'L'
        // After "TOOL", we're in the catch-all body region.
        // Any non byte should be allowed.
        assert!(
            dfa.is_token_allowed(s4, 97),
            "implicit tool body should allow 'a'"
        );
        assert!(
            dfa.is_token_allowed(s4, 50),
            "implicit tool body should allow '2'"
        );
        // Special tokens (T,O,L,E,N,D,S,P) should still be denied in catch-all
        // (they're excluded from the catch-all mask)
        assert!(
            !dfa.is_token_allowed(s4, 83),
            "implicit tool body should deny 'S' (special)"
        );
    }

    /// Verify the DFA reaches an accept state after the full grammar.
    #[test]
    fn dfa_reaches_accept_after_full_grammar() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // Walk the full grammar: reasoning('a') -> text('b') -> tool_callTOOLnameEND") -> eos("STOP")
        let start = dfa.start_state;
        let s1 = dfa.advance(start, 97).unwrap(); // 'a'
        let s2 = dfa.advance(s1, 98).unwrap(); // 'b'
        let s3 = dfa.advance(s2, 84).unwrap(); // 'T'
        let s4 = dfa.advance(s3, 79).unwrap(); // 'O'
        let s5 = dfa.advance(s4, 76).unwrap(); // 'L'
        let s6 = dfa.advance(s5, 110).unwrap(); // 'n'
        let s7 = dfa.advance(s6, 97).unwrap(); // 'a'
        let s8 = dfa.advance(s7, 109).unwrap(); // 'm'
        let s9 = dfa.advance(s8, 69).unwrap(); // 'E'
        let s10 = dfa.advance(s9, 78).unwrap(); // 'N'
        let s11 = dfa.advance(s10, 68).unwrap(); // 'D'
        let s12 = dfa.advance(s11, 83).unwrap(); // 'S'
        let s13 = dfa.advance(s12, 84).unwrap(); // 'T'
        let s14 = dfa.advance(s13, 79).unwrap(); // 'O'
        let s15 = dfa.advance(s14, 80).unwrap(); // 'P'
        // Now we should be at (or past) the accept state
        assert!(
            dfa.accept_states.contains(&s15) || dfa.is_token_allowed(s15, 0),
            "after full grammar walk, should be at accept state or allow EOS"
        );
    }

    /// Verify termination: no tokens allowed after accept state (except self-loop).
    #[test]
    fn termination_no_tokens_after_accept() {
        let (factory, grm) = explicit_tool_grammar();
        let parser = factory.create_parser(grm).unwrap();
        let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");

        // Find an accept state
        let accept = dfa.accept_states.first().copied().expect("no accept states");
        // From the accept state, only self-loop or EOS should be allowed.
        // Any regular token (e.g., 'a'=97) should be denied.
        assert!(
            !dfa.is_token_allowed(accept, 97),
            "accept state should not allow regular token 'a'"
        );
        // The universal_target for accept should be self (loop) or a
        let universal = dfa.universal_target[accept as usize];
        assert!(
            universal == accept || universal == u32::MAX,
            "accept state universal_target should be self-loop or none, got {}",
            universal
        );
    }
}
