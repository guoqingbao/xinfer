// src/utils/special_tokens.rs
use std::collections::HashMap;
use tokenizers::Tokenizer;

const BOS_TOKEN_STRINGS: &[&str] = &[
    "<s>", "<|im_start|>",
    "<start_of_turn>", "<|beginning_of_sentence|>",
    "<bos>",
    // Llama4 encapsulates the role between header IDs:
    // <|start_header_id|>user<|end_header_id|>\n\n
    "<|start_header_id|>", "<|end_header_id|>",
    "<|turn>"

];

const REASONING_TOKEN_PAIRS: &[(&str, &str)] = &[
    ("<thinking>", "</thinking>"),
    ("[THINK]", "[/THINK]"),
    ("<|thinking|>", "<|/thinking|>"),
    ("<reasoning>", "</reasoning>"),
    ("<internal>", "</internal>"),
    ("<reflection>", "</reflection>"),
    ("<|think|>", "<|/think|>"),
    ("<thought>", "</thought>"),
    ("<|channel>", "<channel|>"),
    // Leave the most common selection for last - fallthrough
    ("<think>", "</think>"),
];

const TOOL_CALL_TOKEN_PAIRS: &[(&str, &str)] = &[
    ("<tool_call>", "</tool_call>"),
    ("<start_function_call>", "<end_function_call>"),
    ("<|tool_call>", "<tool_call|>"),
    ("[TOOL_CALLS]", "]"),
    ("<minimax:tool_call>", "</minimax:tool_call>")
];

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    reasoning_start_ids: Vec<u32>,
    reasoning_end_ids: Vec<u32>,
    tool_call_start_ids: Vec<u32>,
    tool_call_end_ids: Vec<u32>,
    bos_token_ids: Vec<u32>
}

impl SpecialTokens {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let mut reasoning_start_ids = Vec::new();
        let mut reasoning_end_ids = Vec::new();
        let mut tool_call_start_ids = Vec::new();
        let mut tool_call_end_ids = Vec::new();
        let mut bos_token_ids = Vec::new();

        // Build lookup maps from added tokens
        let mut added_start_map: HashMap<String, Vec<u32>> = HashMap::new();
        let mut added_end_map: HashMap<String, Vec<u32>> = HashMap::new();
        
        for (id, token) in tokenizer.get_added_tokens_decoder().iter() {
            let content = token.content.as_str();
            
            // Collect BOS tokens
            for &bos_str in BOS_TOKEN_STRINGS.iter() {
                if content == bos_str {
                    bos_token_ids.push(*id)
                }
            }

            // Build added token maps for pairs
            for &(start, end) in REASONING_TOKEN_PAIRS.iter() {
                if content == start {
                    added_start_map.entry(start.to_string()).or_default().push(*id);
                }
                if content == end {
                    added_end_map.entry(end.to_string()).or_default().push(*id);
                }
            }
            
            for &(start, end) in TOOL_CALL_TOKEN_PAIRS.iter() {
                if content == start {
                    added_start_map.entry(start.to_string()).or_default().push(*id);
                }
                if content == end {
                    added_end_map.entry(end.to_string()).or_default().push(*id);
                }
            }
        }
        
        // Process reasoning token pairs with fallback to common vocab
        for &(start, end) in REASONING_TOKEN_PAIRS.iter() {
            process_pair(
                start,
                end,
                &added_start_map,
                &added_end_map,
                &mut reasoning_start_ids,
                &mut reasoning_end_ids,
                |s, e| {
                    let vocab = tokenizer.get_vocab(true);
                    vocab.get(s).cloned().zip(vocab.get(e).cloned())
                },
                "reasoning",
            );
        }
        
        // Process tool call token pairs with fallback to common vocab
        for &(start, end) in TOOL_CALL_TOKEN_PAIRS.iter() {
            process_pair(
                start,
                end,
                &added_start_map,
                &added_end_map,
                &mut tool_call_start_ids,
                &mut tool_call_end_ids,
                |s, e| {
                    let vocab = tokenizer.get_vocab(true);
                    vocab.get(s).cloned().zip(vocab.get(e).cloned())
                },
                "tool_call",
            );
        }

        sort_and_dedup(&mut reasoning_start_ids);
        sort_and_dedup(&mut reasoning_end_ids);
        sort_and_dedup(&mut tool_call_start_ids);
        sort_and_dedup(&mut tool_call_end_ids);

        Self {
            reasoning_start_ids,
            reasoning_end_ids,
            tool_call_start_ids,
            tool_call_end_ids,
            bos_token_ids
        }
    }

    pub fn bos_token_ids(&self) -> Vec<u32> {
        self.bos_token_ids.clone()
    }

    pub fn reasoning_start_ids(&self) -> Vec<u32> {
        self.reasoning_start_ids.clone()
    }

    pub fn reasoning_end_ids(&self) -> Vec<u32> {
        self.reasoning_end_ids.clone()
    }

    pub fn tool_call_start_ids(&self) -> Vec<u32> {
        self.tool_call_start_ids.clone()
    }

    pub fn tool_call_end_ids(&self) -> Vec<u32> {
        self.tool_call_end_ids.clone()
    }
}

/// Helper function to sort and deduplicate token IDs
fn sort_and_dedup(ids: &mut Vec<u32>) {
    ids.sort_unstable();
    ids.dedup();
}

/// Process a token pair, searching added vocab first, then falling back to common vocab
/// Only adds both tokens if the pair is complete (no stragglers)
fn process_pair<F>(
    start_str: &str,
    end_str: &str,
    added_start_map: &HashMap<String, Vec<u32>>,
    added_end_map: &HashMap<String, Vec<u32>>,
    start_ids: &mut Vec<u32>,
    end_ids: &mut Vec<u32>,
    fallback_fn: F,
    pair_type: &str,
) where
    F: FnOnce(&str, &str) -> Option<(u32, u32)>,
{
    // First: check if both tokens exist in added vocabulary
    let start_in_added = added_start_map.contains_key(start_str);
    let end_in_added = added_end_map.contains_key(end_str);

    if start_in_added && end_in_added {
        // Both found in added tokens - collect them
        if let Some(ids) = added_start_map.get(start_str) {
            start_ids.extend(ids);
        }
        if let Some(ids) = added_end_map.get(end_str) {
            end_ids.extend(ids);
        }
        return;
    }

    // Fallback: check common vocabulary
    if let Some((start_id, end_id)) = fallback_fn(start_str, end_str) {
        crate::log_warn!(
            "[{}] Pair '{}' + '{}' not found in added vocabulary, falling back to common vocab with IDs: {} + {}",
            pair_type,
            start_str,
            end_str,
            start_id,
            end_id
        );
        start_ids.push(start_id);
        end_ids.push(end_id);
    }
}

#[cfg(test)]
mod tests {
    use super::SpecialTokens;
    use tokenizers::{models::bpe::BPE, AddedToken, Tokenizer};

    #[test]
    fn special_tokens_collects_only_exact_single_token_candidates() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<think>", true),
            AddedToken::from("</think>", true),
            AddedToken::from("<|assistant|>", true),
        ]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        assert_eq!(special_tokens.reasoning_start_ids().len(), 1);
        assert_eq!(special_tokens.reasoning_end_ids().len(), 1);
    }

    #[test]
    fn special_tokens_pair_not_found_in_added_fallback_to_common_vocab() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        // Add only start token to added vocabulary
        tokenizer.add_special_tokens(&[AddedToken::from("<thinking>", true)]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        // Should not find anything since only one token of the pair is in added vocab
        assert_eq!(special_tokens.reasoning_start_ids().len(), 0);
        assert_eq!(special_tokens.reasoning_end_ids().len(), 0);
    }

    #[test]
    fn special_tokens_pair_fallback_with_both_tokens_in_common_vocab() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<thinking>", true),
            AddedToken::from("</thinking>", true),
        ]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        // Both tokens found in added vocabulary
        assert_eq!(special_tokens.reasoning_start_ids().len(), 1);
        assert_eq!(special_tokens.reasoning_end_ids().len(), 1);
    }

    #[test]
    fn special_tokens_skips_partial_pair() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        // Only add start tag, not end tag
        tokenizer.add_special_tokens(&[AddedToken::from("<thinking>", true)]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        // Should skip the pair entirely - no stragglers
        assert_eq!(special_tokens.reasoning_start_ids().len(), 0);
        assert_eq!(special_tokens.reasoning_end_ids().len(), 0);
    }

    #[test]
    fn special_tokens_tool_call_pair_handling() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|tool_call>", true),
            AddedToken::from("<tool_call|>", true),
        ]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        // Both tokens found in added vocabulary
        assert_eq!(special_tokens.tool_call_start_ids().len(), 1);
        assert_eq!(special_tokens.tool_call_end_ids().len(), 1);
    }
}
