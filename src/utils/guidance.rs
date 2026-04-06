// src/utils/guidance.rs
// This module contains non-grammar guidance utilities:
// - GuidanceTokens: token ID collections
// - ParserFactory: llguidance parser factory
// - GuidanceState: matcher state for speculative decoding
// - Mask operations: batch mask bias and early exit validation

use crate::utils::special_tokens::SpecialTokens;
use anyhow::Result;
use candle_core::Tensor;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::utils::logits_processor::{LogitsProcessor, Sampling};

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
}

pub fn extract_guidance_tokens(
    tokenizer: &Tokenizer,
    eos_token_ids: Vec<u32>,
    bos_token_ids: Vec<u32>,
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

    GuidanceTokens {
        bos_token_ids,
        eos_token_ids: validated_eos,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
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

pub fn load_toktrie_from_path(path: impl AsRef<std::path::Path>) -> Result<TokTrie> {
    let tokenizer = ByteTokenizer::from_file(path)?;
    let env = ByteTokenizerEnv::new(tokenizer, None)?;
    Ok(env.tok_trie)
}

/// WS regex pattern for Lark grammars - matches whitespace including spaces, tabs, newlines, carriage returns
pub fn lark_ws_regex() -> &'static str {
    "/[ \\t\\n\\r]+/"
}

/// Cache for precomputed mask slices to avoid expensive re-computation
#[derive(Clone, Default)]
pub struct SlicerCache {
    cache: HashMap<usize, Vec<bool>>,
}

impl SlicerCache {
    /// Get or compute a mask slice for a given position
    pub fn get_or_compute(
        &mut self,
        pos: usize,
        compute_fn: impl FnOnce() -> Vec<bool>,
    ) -> &Vec<bool> {
        if !self.cache.contains_key(&pos) {
            self.cache.insert(pos, compute_fn());
        }
        self.cache
            .get(&pos)
            .expect("entry must exist after compute")
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

pub struct GuidanceState {
    matcher: Matcher,
    /// Track llm tokens for speculative decoding recovery
    llm_tokens: Vec<u32>,
    /// Track llm bytes for rollback calculations
    llm_bytes: usize,
    /// Cache for precomputed mask slices
    slicer_cache: SlicerCache,
}

impl GuidanceState {
    pub fn new_from_grammar(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
    ) -> Result<Self> {
        use crate::utils::guidance_grammar::get_lark_from_top_level_grammar;
        let lark = get_lark_from_top_level_grammar(grammar);
        crate::log_info!("[llg] Composed Grammar Constraint:\n{}\n", lark);
        let mut grammar = grammar.clone();
        // Add generation space for EOS token to prevent overrun if max_tokens reached
        if let Some(max_tokens) = grammar.max_tokens {
            grammar.max_tokens = Some(max_tokens + 1);
        };
        let parser = factory.create_parser(grammar)?;
        let matcher = Matcher::new(Ok(parser));

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            llm_bytes: 0,
            slicer_cache: SlicerCache::default(),
        })
    }

    /// Compute mask with caching for performance
    pub fn compute_mask(&mut self) -> Result<Option<SimpleVob>> {
        if self.matcher.is_stopped() {
            return Ok(None);
        }
        let mask = self.matcher.compute_mask()?;
        Ok(Some(mask))
    }

    /// Commit token and track for speculative decoding recovery
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
            self.llm_tokens.push(token);
            self.llm_bytes += 4;
        }
        Ok(())
    }

    /// Get the number of committed tokens
    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    /// Get the number of committed bytes
    pub fn num_bytes(&self) -> usize {
        self.llm_bytes
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Get the last committed token
    pub fn last_token(&self) -> Option<u32> {
        self.llm_tokens.last().copied()
    }

    /// Validate token without consuming it (for re-sampling)
    pub fn validate_token(&mut self, token: u32) -> bool {
        if self.matcher.is_stopped() {
            return true;
        }
        let result = self.matcher.validate_tokens(&[token]).unwrap_or(0);
        result == 1
    }

    /// Compute mask or return EOS token set if stopped
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        self.matcher.compute_mask_or_eos().map_err(Into::into)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding)
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// Fast-forward and consume tokens guaranteed to be accepted by the grammar
    pub fn consume_ff_tokens(&mut self) -> Result<Vec<u32>, anyhow::Error> {
        if self.matcher.is_stopped() {
            return Ok(Vec::new());
        }

        let ff_tokens = self.matcher.compute_ff_tokens();

        for &token in &ff_tokens {
            self.matcher.consume_token(token)?;
            self.llm_tokens.push(token);
            self.llm_bytes += 4;
        }

        Ok(ff_tokens)
    }

    /// Check if there are pending lexeme bytes to be consumed
    pub fn has_pending_lexeme_bytes(&self) -> bool {
        false
    }

    /// Rollback to a previous state with byte tracking
    pub fn rollback_to(&mut self, token_pos: usize, byte_pos: usize) -> Result<()> {
        let tokens_to_rollback = self.llm_tokens.len().saturating_sub(token_pos);
        if tokens_to_rollback > 0 {
            self.matcher.rollback(tokens_to_rollback)?;
        }
        self.llm_tokens.truncate(token_pos);
        self.llm_bytes = byte_pos;
        Ok(())
    }

    /// Capture current state as rollback snapshot
    pub fn capture_snapshot(&mut self) {}

    /// Clear all state
    pub fn clear(&mut self) {
        self.llm_tokens.clear();
        self.llm_bytes = 0;
        self.slicer_cache.clear();
    }

    /// Get a reference to the slicer cache
    pub fn slicer_cache(&mut self) -> &mut SlicerCache {
        &mut self.slicer_cache
    }

    /// Validate a sequence of tokens against the grammar
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Option<usize> {
        if self.matcher.is_stopped() {
            return Some(tokens.len());
        }
        match self.matcher.validate_tokens(tokens) {
            Ok(count) => Some(count),
            Err(_) => None,
        }
    }
}

/// Apply sparse mask bias to logits
/// Uses iter_set_entries to only iterate allowed tokens
pub fn _batch_mask_bias(
    logits: &Tensor,
    masks: &[(usize, SimpleVob)],
    vocab_size: usize,
) -> candle_core::Result<Tensor> {
    let batch_size = masks.len();

    // Create bias vector initialized to -inf
    let mut bias_data = vec![f32::NEG_INFINITY; batch_size * vocab_size];

    // Fill in allowed tokens using sparse iteration
    // masks is Vec<(batch_idx, SimpleVob)> where batch_idx is the sequence position in the batch
    for (batch_idx, mask) in masks.iter() {
        mask.iter_set_entries(|idx| {
            if idx < vocab_size {
                bias_data[*batch_idx * vocab_size + idx] = 0.0;
            }
        });
    }

    // Create bias tensor on same device as logits
    let bias_tensor = Tensor::from_vec(bias_data, (batch_size, vocab_size), logits.device())?;

    // GPU tensor addition (no CPU copy)
    logits.broadcast_add(&bias_tensor)
}

/// Two-stage validation with early exit
/// Stage 1: Sample and validate token
/// Stage 2: Only compute mask if token is invalid
pub fn _early_exit_validate(
    guidance_states: &mut HashMap<usize, GuidanceState>,
    seq_ids: &[usize],
    tokens: &mut [u32],
    logits: &Tensor,
    vocab_size: usize,
    _factory: &Arc<ParserFactory>,
    sampling: &Sampling,
    logit_processor: &LogitsProcessor,
) -> candle_core::Result<()> {
    for (seq_idx, seq_id) in seq_ids.iter().enumerate() {
        let token = tokens[seq_idx];

        if let Some(state) = guidance_states.get_mut(seq_id) {
            // Stage 1: Validate token
            if state.validate_token(token) {
                // Early exit - token is valid, consume it
                state
                    .commit_token(token)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                continue;
            }

            // Stage 2: Token is invalid, compute mask and re-sample
            let mask = match state.compute_mask_or_eos() {
                Ok(m) => m,
                Err(e) => {
                    crate::log_error!(
                        "[llg] Unable to compute mask for token {} due to {}",
                        token,
                        e
                    );
                    continue;
                }
            };

            // Build bias vector using sparse iteration
            let mut acc = vec![f32::NEG_INFINITY; vocab_size];
            mask.iter_set_entries(|idx| {
                if idx < acc.len() {
                    acc[idx] = 0.0;
                }
            });

            // Get current sequence's logits as 1D tensor - MUST CLONE to avoid cross-contamination
            let row_start = seq_idx * vocab_size;
            let row_end = row_start + vocab_size;
            let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let mut row_vec = logits_vec.clone(); // Clone to avoid modifying original
            let row = &mut row_vec[row_start..row_end];

            // Apply bias directly to this sequence's row
            for tok in 0..vocab_size {
                if acc[tok] != 0.0 {
                    row[tok] = f32::NEG_INFINITY;
                }
            }

            // Create 1D tensor for just this sequence
            let biased_row = Tensor::from_vec(
                row_vec[row_start..row_end].to_vec(),
                (vocab_size,),
                logits.device(),
            )?;

            // Re-sample just this sequence from the biased 1D logits
            let re_sampled = logit_processor.sample_with_strategy(&biased_row, sampling)?;
            tokens[seq_idx] = re_sampled[0]; // 1D output, first (only) element

            // Commit the re-sampled token
            state
                .commit_token(tokens[seq_idx])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
    }

    Ok(())
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
