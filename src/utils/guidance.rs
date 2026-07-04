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

    GuidanceTokens {
        bos_token_ids: validated_bos,
        eos_token_ids: validated_eos,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
        add_bos_token,
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

        let mut grammar = grammar.clone();
        if let Some(max_tokens) = grammar.max_tokens {
            let bos_len = 1;
            let eos_len = 1;
            grammar.max_tokens = Some(max_tokens + bos_len + eos_len);
        };
        let parser = factory.create_parser(grammar)?;
        let matcher = Matcher::new(Ok(parser));

        let reasoning_ended = reasoning_end_ids.is_empty();

        if !reasoning_ended {
            crate::log_info!(
                "[llg] Two-phase reasoning: grammar deferred until after reasoning end tokens {:?}",
                reasoning_end_ids
            );
        }

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
    /// During reasoning, returns an all-ones mask (allow everything).
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        if !self.reasoning_ended {
            return self
                .matcher
                .compute_mask_or_eos()
                .map(|mut mask| {
                    mask.set_all(true);
                    mask
                })
                .map_err(Into::into);
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
    fn test_extract_guidance_tokens() {
        // This test verifies that extract_guidance_tokens compiles
        // It doesn't actually run since we don't have a tokenizer here
        let tokens = GuidanceTokens::default();
        assert!(tokens.bos_token_ids.is_empty());
    }
}
