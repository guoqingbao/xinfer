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
    reasoning_ended: bool
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Test unsorted input (should be sorted internally)
        let ranges = GuidanceTokens::compress_to_ranges(&[151658, 151644, 151650]);
        assert_eq!(ranges, vec![(151644, 151644), (151650, 151650), (151658, 151658)]);
    }

    #[test]
    fn test_token_range_expression() {
        // Test empty exclusions
        let expr = GuidanceTokens::token_range_expression(vec![]);
        assert_eq!(expr, r#"/(?s:.*)/"#.to_string());

        // Test single exclusion
        let expr = GuidanceTokens::token_range_expression(vec![151644]);
        assert_eq!(expr, "<[^151644]> ");

        // Test consecutive exclusions
        let expr = GuidanceTokens::token_range_expression(vec![151644, 151645, 151646]);
        assert_eq!(expr, "<[^151644-151646]> ");

        // Test non-consecutive exclusions
        let expr = GuidanceTokens::token_range_expression(vec![151644, 151650, 151658]);
        assert_eq!(expr, "<[^151644,151650,151658]> ");

        // Test mixed exclusions
        let expr = GuidanceTokens::token_range_expression(vec![1, 2, 5, 6, 7, 10]);
        assert_eq!(expr, "<[^1-2,5-7,10]> ");
    }
}
