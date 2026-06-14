//! Integration test: `LLMEngine::preprocess` is byte-identical to the
//! canonical `apply_chat_template + tokenizer.encode_fast` path that
//! `add_request_` (and through it, `generate_sync` / `generate_stream`)
//! has used in the engine.
//!
//! Without this guard, future tweaks to either method could silently drift
//! and the bench would be the only signal — too coarse to catch a
//! tokenize-mismatch bug.
//!
//! This is a real engine spin-up; it downloads `Qwen/Qwen3-0.6B` if not
//! cached and loads it on the device set by `XINFER_TEST_DEVICE`
//! (default `0`). It's gated behind `XINFER_INTEGRATION_TEST=1` so the
//! ordinary `cargo test` flow doesn't pull a model.
//!
//! Run with:
//!   XINFER_INTEGRATION_TEST=1 cargo test --release \
//!     --no-default-features --features metal --test preprocess_parity
//!
//! (Swap `--features metal` for `--features "cuda nccl"` on Linux/CUDA.)

use candle_core::DType;

use xinfer::core::engine::LLMEngine;
use xinfer::utils::chat_template::Message;
use xinfer::utils::config::{EngineConfig, SamplingParams};

fn integration_enabled() -> bool {
    std::env::var("XINFER_INTEGRATION_TEST")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn build_econfig() -> EngineConfig {
    let device_id: usize = std::env::var("XINFER_TEST_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    EngineConfig::new(
        Some("Qwen/Qwen3-0.6B".to_string()), // model_id
        None,                                // weight_path
        None,                                // weight_file
        None,                                // hf_token
        None,                                // hf_token_path
        None,                                // enforce_parser
        Some(1),                             // max_num_seqs — minimal
        None,                                // config_model_len
        Some(2048),                          // max_model_len — small
        Some(8),                             // max_tokens
        None,                                // isq
        Some(1),                             // num_shards
        Some(vec![device_id]),               // device_ids
        None,                                // generation_cfg
        Some(0),                             // seed
        false,                               // disable_prefix_cache
        None,                                // prefix_cache_max_tokens
        None,                                // kvcache_dtype
        Some(false),                         // server_mode — drive engine directly
        None,                                // cpu_mem_fold
        None,                                // kv_fraction
        None,                                // mamba_fraction
        None,                                // pd_config
        None,                                // mcp_command
        None,                                // mcp_config
        None,                                // mcp_args
        None,                                // tool_prompt_template
        None,                                // pd_server_prefix_cache_ratio
        None,                                // pd_client_prefix_cache_ratio
        None,                                // yarn_scaling_factor
        true,                                // disable_reasoning — deterministic prompt
        false,                               // disable_cuda_graph
        None,                                // prefill_chunk_size — default
    )
}

fn sample_inputs() -> (Vec<SamplingParams>, Vec<Vec<Message>>) {
    // Three deliberately varied conversations to exercise (a) trivial
    // single-user, (b) multi-turn user/assistant alternation, and (c)
    // long content that should produce many tokens.
    let params = vec![
        SamplingParams::new_with_max_tokens(8),
        SamplingParams::new_with_max_tokens(8),
        SamplingParams::new_with_max_tokens(8),
    ];

    let long =
        "the rapid evolution of large language model inference engines on heterogeneous hardware. "
            .repeat(20);

    let message_list = vec![
        vec![Message::new(
            "user".to_string(),
            "What is the capital of France?".to_string(),
            0,
        )],
        vec![
            Message::new("user".to_string(), "Tell me about the moon.".to_string(), 0),
            Message::new(
                "assistant".to_string(),
                "The moon is Earth's natural satellite.".to_string(),
                0,
            ),
            Message::new("user".to_string(), "How far away is it?".to_string(), 0),
        ],
        vec![Message::new("user".to_string(), long, 0)],
    ];

    (params, message_list)
}

#[test]
fn preprocess_is_byte_identical_to_apply_chat_template_plus_tokenize() {
    if !integration_enabled() {
        eprintln!(
            "Skipping: set XINFER_INTEGRATION_TEST=1 to enable (downloads Qwen3-0.6B, ~1.2GB)"
        );
        return;
    }
    eprintln!("[integration] preprocess_parity: spinning up LLMEngine");

    let econfig = build_econfig();
    let engine = LLMEngine::new(&econfig, DType::BF16).expect("LLMEngine::new failed");
    let engine = engine.read();

    let (params, message_list) = sample_inputs();
    let tools: Vec<xinfer::tools::Tool> = vec![];

    // ----- Path A: new preprocess() -----
    let preprocessed = engine
        .preprocess(&params, &message_list, &tools, false)
        .expect("preprocess failed");

    assert_eq!(
        preprocessed.len(),
        params.len(),
        "preprocess returned a different count than input"
    );

    // ----- Path B: canonical apply_chat_template + tokenizer.encode_fast -----
    for (idx, (param, messages)) in params.iter().zip(message_list.iter()).enumerate() {
        let (expected_prompt, expected_image_idx) =
            engine.apply_chat_template(param, messages, &tools, false);
        let expected_tokens = engine
            .tokenizer
            .encode_fast(expected_prompt.as_str(), true)
            .expect("expected tokenize failed");
        let expected_token_ids: Vec<u32> = expected_tokens.get_ids().to_vec();

        let pp = &preprocessed[idx];

        assert_eq!(
            pp.prompt, expected_prompt,
            "rendered prompt mismatch on row {}",
            idx
        );
        assert_eq!(
            pp.image_idx, expected_image_idx,
            "image_idx mismatch on row {}",
            idx
        );
        assert_eq!(
            pp.token_ids, expected_token_ids,
            "token_ids mismatch on row {} — preprocess refactor is no longer byte-identical",
            idx
        );
        assert!(
            !pp.token_ids.is_empty(),
            "row {} produced empty token_ids",
            idx
        );
    }
}

#[test]
fn preprocess_rejects_size_mismatch() {
    if !integration_enabled() {
        eprintln!("Skipping: set XINFER_INTEGRATION_TEST=1 to enable");
        return;
    }

    let econfig = build_econfig();
    let engine = LLMEngine::new(&econfig, DType::BF16).expect("LLMEngine::new failed");
    let engine = engine.read();

    let params = vec![SamplingParams::new_with_max_tokens(8)];
    let message_list: Vec<Vec<Message>> = vec![]; // intentionally mismatched
    let tools: Vec<xinfer::tools::Tool> = vec![];

    let res = engine.preprocess(&params, &message_list, &tools, false);
    assert!(
        res.is_err(),
        "preprocess should reject mismatched params/message_list lengths"
    );
}
