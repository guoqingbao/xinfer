// src/server/server.rs
use super::logger::ChatCompletionLogger;
use super::{
    build_guided_decoding_grammar, build_messages_and_images, collect_openai_constraint_grammar,
    normalize_reasoning_controls,
    streaming::{ChatResponse, Streamer, StreamingStatus},
    ChatResponder, DetokenizeRequest, DetokenizeResponse, EmbeddingRequest, EmbeddingResponse,
    EncodingFormat, TokenizeInput, TokenizeRequest, TokenizeResponse,
};
use super::{
    ChatChoice, ChatChoiceChunk, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, CompletionTokensDetails, Delta,
    EmbeddingData, EmbeddingOutput, EmbeddingUsage, ErrorMsg, PromptTokensDetails, ServerData,
    Usage, UsageQuery, UsageResponse,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::server::parser::{BufferedFinalizeResult, StreamResult, StreamToolParser};
use crate::tools::helpers::{
    build_invalid_tool_call_feedback, build_tool_schema_map, filter_tool_calls, log_tool_calls,
    resolve_tools, retain_tool_calls_forced_name, strict_tool_call_validation_enabled,
};
use crate::tools::{ToolChoice, ToolChoiceMode};
use crate::utils::config::SamplingParams;
use crate::utils::guidance::ReasoningEffort;
use axum::{
    extract::{Json, Query, State},
    response::{sse::KeepAlive, Sse},
};
use base64::Engine;
use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::task;
use uuid::Uuid;

/// Helper struct to manage streaming response chunks
/// Provides clean API for sending tokens, errors, and status notifications
struct StreamingContext {
    seq_id: usize,
    model_id: String,
    created: u64,
    response_tx: flume::Sender<ChatResponse>,
}

/// Routes streaming tokens to either `content` or `reasoning_content` in SSE
/// chunks based on reasoning marker state from the tool parser.
///
/// When `XINFER_STREAM_AS_REASONING_CONTENT` is enabled (default), reasoning
/// markers (`<think>`, `</think>`, etc.) are stripped from the stream and the
/// inner text is emitted as `delta.reasoning_content` instead of `delta.content`.
///
/// The routing decision uses the tool parser's `in_reasoning()` state which
/// correctly handles split tokens (e.g. `<thin` + `k>`). The router only
/// strips complete markers from the text to avoid sending them to the client.
struct ReasoningContentRouter {
    enabled: bool,
}

impl ReasoningContentRouter {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Route a content token to the appropriate SSE field.
    /// `parser_in_reasoning` should be the tool parser's `in_reasoning()` state
    /// captured BEFORE `process_token` was called for this token.
    /// Returns false if the client disconnected.
    fn send(&self, text: &str, parser_in_reasoning: bool, ctx: &StreamingContext) -> bool {
        if text.is_empty() {
            return true;
        }
        if !self.enabled {
            return ctx.send_token(text);
        }

        let stripped = strip_reasoning_markers(text);
        if stripped.is_empty() {
            return true;
        }

        if parser_in_reasoning {
            ctx.send_reasoning_token(&stripped)
        } else {
            ctx.send_token(&stripped)
        }
    }
}

/// Strip all reasoning start/end markers from a text fragment.
/// Unlike `strip_reasoning_blocks` which removes matched pairs, this removes
/// individual marker occurrences so they are never sent to the client.
fn strip_reasoning_markers(text: &str) -> String {
    let mut result = text.to_string();
    for &(open, close) in crate::server::parser::reasoning_markers() {
        result = result.replace(open, "");
        result = result.replace(close, "");
    }
    result
}

const EMPTY_TOOL_RESULT_ACK: &str = "Tool executed successfully with no textual output.";

fn extract_text_from_content(content: Option<&super::MessageContentType>) -> String {
    match content {
        Some(super::MessageContentType::PureText(text)) => text.clone(),
        Some(super::MessageContentType::Single(item)) => match item {
            super::MessageContent::Text { text } => text.clone(),
            _ => String::new(),
        },
        Some(super::MessageContentType::Multi(items)) => items
            .iter()
            .filter_map(|item| match item {
                super::MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => String::new(),
    }
}

fn normalize_empty_openai_tool_results(messages: &mut [ChatMessage]) {
    for msg in messages {
        if msg.role != "tool" {
            continue;
        }

        let is_empty = extract_text_from_content(msg.content.as_ref())
            .trim()
            .is_empty();
        if is_empty {
            msg.content = Some(super::MessageContentType::PureText(
                EMPTY_TOOL_RESULT_ACK.to_string(),
            ));
        }
    }
}

fn validate_openai_tool_messages(messages: &[ChatMessage]) -> Result<(), String> {
    let mut assistant_tool_call_ids: HashSet<String> = HashSet::new();
    let mut tool_result_ids_seen: HashSet<String> = HashSet::new();
    let mut pending_tool_results: Option<HashSet<String>> = None;

    for (idx, msg) in messages.iter().enumerate() {
        if let Some(expected_results) = pending_tool_results.as_mut() {
            if msg.role != "tool" {
                let mut pending_ids = expected_results.iter().cloned().collect::<Vec<_>>();
                pending_ids.sort();
                return Err(format!(
                    "messages[{idx}] must be role=tool to answer pending assistant tool_calls {:?}",
                    pending_ids
                ));
            }
            if msg.tool_calls.is_some() {
                return Err(format!(
                    "messages[{idx}] role=tool must not include tool_calls"
                ));
            }

            let call_id = msg.tool_call_id.as_deref().unwrap_or("").trim();
            if call_id.is_empty() {
                return Err(format!(
                    "messages[{idx}] role=tool requires a non-empty tool_call_id"
                ));
            }
            if !tool_result_ids_seen.insert(call_id.to_string()) {
                return Err(format!(
                    "messages[{idx}] role=tool has duplicate tool_call_id '{}'",
                    call_id
                ));
            }
            if !expected_results.remove(call_id) {
                let mut pending_ids = expected_results.iter().cloned().collect::<Vec<_>>();
                pending_ids.sort();
                return Err(format!(
                    "messages[{idx}] role=tool references unexpected tool_call_id '{}'. pending ids: {:?}",
                    call_id, pending_ids
                ));
            }

            let text = extract_text_from_content(msg.content.as_ref());
            if text.trim().is_empty() {
                return Err(format!(
                    "messages[{idx}] role=tool requires non-empty content"
                ));
            }
            if expected_results.is_empty() {
                pending_tool_results = None;
            }
            continue;
        }

        match msg.role.as_str() {
            "assistant" => {
                if let Some(tool_calls) = &msg.tool_calls {
                    if tool_calls.is_empty() {
                        continue;
                    }
                    let mut expected_results = HashSet::new();
                    for (tool_idx, call) in tool_calls.iter().enumerate() {
                        let call_id = call.id.trim();
                        if call_id.is_empty() {
                            return Err(format!(
                                "messages[{idx}] assistant tool_calls[{tool_idx}] requires a non-empty id"
                            ));
                        }
                        if !expected_results.insert(call_id.to_string()) {
                            return Err(format!(
                                "messages[{idx}] assistant tool_call id '{}' is duplicated",
                                call_id
                            ));
                        }
                        if !assistant_tool_call_ids.insert(call_id.to_string()) {
                            return Err(format!(
                                "messages[{idx}] assistant tool_call id '{}' is duplicated",
                                call_id
                            ));
                        }
                    }
                    pending_tool_results = Some(expected_results);
                }
            }
            "tool" => {
                let call_id = msg.tool_call_id.as_deref().unwrap_or("").trim();
                if !call_id.is_empty() && tool_result_ids_seen.contains(call_id) {
                    return Err(format!(
                        "messages[{idx}] role=tool has duplicate tool_call_id '{}'",
                        call_id
                    ));
                }
                return Err(format!(
                    "messages[{idx}] role=tool has no preceding assistant tool_calls to answer"
                ));
            }
            _ => {}
        }
    }

    if let Some(pending) = pending_tool_results {
        let mut pending_ids = pending.into_iter().collect::<Vec<_>>();
        pending_ids.sort();
        return Err(format!(
            "Missing role=tool results for assistant tool_call ids: {:?}",
            pending_ids
        ));
    }

    Ok(())
}

impl StreamingContext {
    fn new(
        seq_id: usize,
        model_id: String,
        created: u64,
        response_tx: flume::Sender<ChatResponse>,
    ) -> Self {
        Self {
            seq_id,
            model_id,
            created,
            response_tx,
        }
    }

    /// Send a content token chunk. Returns false if client disconnected.
    fn send_token(&self, token: &str) -> bool {
        let chunk = ChatCompletionChunk {
            id: format!("seq-{}", self.seq_id),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(token.to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        };
        self.response_tx
            .try_send(ChatResponse::Chunk(chunk))
            .is_ok()
    }

    /// Send a reasoning_content token chunk. Returns false if client disconnected.
    fn send_reasoning_token(&self, token: &str) -> bool {
        let chunk = ChatCompletionChunk {
            id: format!("seq-{}", self.seq_id),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: Some(token.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        };
        self.response_tx
            .try_send(ChatResponse::Chunk(chunk))
            .is_ok()
    }

    /// Send initial assistant role delta chunk for OpenAI streaming compatibility.
    fn send_role_start(&self) -> bool {
        let chunk = ChatCompletionChunk {
            id: format!("seq-{}", self.seq_id),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        };
        self.response_tx
            .try_send(ChatResponse::Chunk(chunk))
            .is_ok()
    }
}

#[utoipa::path(
    post,
    tag = "xinfer",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completion(
    State(data): State<Arc<ServerData>>,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    // Create logger for this request (None if XINFER_CHAT_LOGGER not set to true)
    let logger = ChatCompletionLogger::new();
    if let Some(ref l) = logger {
        l.log_request(&request);
    }

    let use_stream = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(true);
    let tool_buffer_timeout = Duration::from_secs(
        env::var("XINFER_TOOL_BUFFER_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(600),
    );

    let model_id = request.model.clone().unwrap_or("default".to_string());
    let max_tokens = request
        .max_tokens
        .unwrap_or(data.econfig.max_tokens.unwrap_or(16384));

    let mut params = SamplingParams::new_with_max_tokens(max_tokens);
    params.temperature = request.temperature;
    params.top_k = request.top_k;
    params.top_p = request.top_p;
    params.frequency_penalty = request.frequency_penalty;
    params.presence_penalty = request.presence_penalty;
    params.session_id = request.session_id.clone();
    params.thinking = request.thinking.clone();
    params.stop_sequences = request.stop.clone();
    params.reasoning_effort = request
        .reasoning_effort
        .clone()
        .map(ReasoningEffort::from_str);
    let (img_cfg, model_type, tool_config, engine_config, guidance_tokens) = {
        let e = data.engine.read();
        (
            e.img_cfg.clone(),
            e.model_type.clone(),
            e.tool_config.clone(),
            e.econfig.clone(),
            e.guidance_tokens.clone(),
        )
    };
    let constraint_grammar = match collect_openai_constraint_grammar(&request) {
        Ok(grammar) => grammar,
        Err(err) => return ChatResponder::ValidationError(err.to_string()),
    };
    if constraint_grammar.is_some() {
        normalize_reasoning_controls(&mut params, &guidance_tokens);
    }

    let mcp_tools = data
        .mcp_manager
        .as_ref()
        .map(|manager| manager.cached_tools())
        .unwrap_or_default();
    let mut resolved_tools = resolve_tools(request.tools.as_deref(), &mcp_tools);
    let mut forced_tool_name: Option<String> = None;
    let mut tool_choice_required = false;

    // Set tool mode for streaming tool call handling:
    // - None: No tools, ignore </tool_call> detection
    // - Some(true): Tools enabled, finish stream at </tool_call> for external handling
    match request.tool_choice.as_ref() {
        Some(ToolChoice::Mode(ToolChoiceMode::None)) => {
            resolved_tools.clear();
        }
        Some(ToolChoice::Function { function, .. }) => {
            tool_choice_required = true;
            forced_tool_name = Some(function.name.clone());
        }
        Some(ToolChoice::Mode(ToolChoiceMode::Required)) => {
            tool_choice_required = true;
        }
        Some(ToolChoice::Mode(ToolChoiceMode::Auto)) | None => {}
    }

    if tool_choice_required && resolved_tools.is_empty() {
        return ChatResponder::ValidationError(
            "tool_choice requires at least one tool but none were provided".to_string(),
        );
    }

    if let Some(name) = forced_tool_name.clone() {
        let selected = resolved_tools
            .iter()
            .find(|tool| tool.function.name == name)
            .cloned();
        match selected {
            Some(tool) => {
                resolved_tools = vec![tool];
            }
            None => {
                return ChatResponder::ValidationError(format!(
                    "tool_choice requires tool '{}' but it was not provided",
                    name
                ));
            }
        }
    }

    let tool_schemas = Arc::new(build_tool_schema_map(&resolved_tools));
    let has_tools = !resolved_tools.is_empty();
    params.mcp_mode = if has_tools { Some(true) } else { None };

    if has_tools {
        crate::log_warn!("Tools enabled for request");
    }

    let mut chat_messages = request.messages.clone();
    normalize_empty_openai_tool_results(&mut chat_messages);
    if let Err(err) = validate_openai_tool_messages(&chat_messages) {
        return ChatResponder::ValidationError(err);
    }
    let parser_model_id =
        super::resolve_engine_model_id(&engine_config).unwrap_or_else(|| model_id.clone());
    let enforce_parser = engine_config.enforce_parser.clone();

    let (messages, image_data) = match build_messages_and_images(&chat_messages, img_cfg.as_ref()) {
        Ok(output) => output,
        Err(e) => {
            crate::log_error!("Image processing failed: {:?}", e);
            return ChatResponder::InternalError(format!("Internal server error {:?}", e));
        }
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    {
        let engine = data.engine.read();
        params.grammar = build_guided_decoding_grammar(
            &engine.guidance_tokens,
            constraint_grammar,
            max_tokens,
            params.reasoning_effort.clone(),
        );
    }

    if use_stream {
        let session_id = params.session_id.clone();
        if let Some(sid) = session_id {
            crate::log_warn!("Stream request has session_id {sid}");
        }
        let preprocessed = {
            let e = data.engine.read();
            match e.preprocess(
                std::slice::from_ref(&params),
                std::slice::from_ref(&messages),
                &resolved_tools,
                false,
            ) {
                Ok(mut p) => p.pop().expect("preprocess returned 0 items for 1 input"),
                Err(e) => {
                    crate::log_error!("Stream preprocess failed: {:?}", e);
                    return ChatResponder::ValidationError(format!(
                        "Stream preprocess failed: {:?}",
                        e
                    ));
                }
            }
        };
        let (seq_id, prompt_length, prefilled_reasoning_end, stream) = {
            let mut e = data.engine.write();
            match e.generate_stream(preprocessed, image_data, &logger) {
                Ok((seq_id, prompt_length, prefilled_reasoning_end, stream)) => {
                    (seq_id, prompt_length, prefilled_reasoning_end, stream)
                }
                Err(e) => {
                    crate::log_error!("Stream generation failed: {:?}", e);
                    return ChatResponder::ValidationError(format!(
                        "Stream generation failed: {:?}",
                        e
                    ));
                }
            }
        };

        let stream = stream;
        let (response_tx, client_rx) = flume::unbounded();
        let (disconnect_tx, mut disconnect_rx) = watch::channel(false);

        // Clone data needed for the async task
        let engine_clone = data.engine.clone();
        let _img_cfg_clone = img_cfg.clone();

        let tool_config = tool_config.clone();
        let mut tool_parser = StreamToolParser::new_with_config(
            &model_type,
            parser_model_id.clone(),
            tool_config,
            resolved_tools.clone(),
            enforce_parser.clone(),
        );
        tool_parser.set_initial_reasoning_end_marker(prefilled_reasoning_end.clone());
        tool_parser.set_detect_tools_in_reasoning(crate::utils::env::stream_as_reasoning_content());
        let forced_tool_name = forced_tool_name.clone();
        let stream_tool_schemas = tool_schemas.clone();
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let stream_logger = logger.clone();

        task::spawn(async move {
            #[allow(unused_assignments)]
            let mut decode_start_time = 0u64;
            let mut total_decoded_tokens = 0usize;
            let mut full_decoded_text = String::new();
            let mut pending_tool_calls: Vec<crate::tools::ToolCall> = Vec::new();
            let mut suppressed_tool_markup: String = String::new();
            let mut buffering_since: Option<Instant> = None;
            let mut buffering_cancel_requested = false;
            let mut buffering_warned = false;

            // Create streaming context for clean helper methods
            let stream_ctx =
                StreamingContext::new(seq_id, model_id.to_string(), created, response_tx.clone());
            if !stream_ctx.send_role_start() {
                let mut e = engine_clone.write();
                e.cancel(seq_id);
                let _ = response_tx.try_send(ChatResponse::Done);
                return;
            }

            // Initialize the stream tool parser (handles all tool call detection internally)
            let mut tool_parser = tool_parser;
            let should_parse_tools = has_tools.clone();

            let reasoning_router =
                ReasoningContentRouter::new(crate::utils::env::stream_as_reasoning_content());

            let mut current_stream = stream;
            let current_seq_id = seq_id;

            loop {
                let item = tokio::select! {
                    item = current_stream.recv() => item,
                    res = disconnect_rx.changed() => {
                        if res.is_err() {
                            break;
                        }
                        if *disconnect_rx.borrow() {
                            crate::log_warn!(
                                "[Seq {}] Stream client disconnected during prefill/stream",
                                current_seq_id
                            );
                            let mut e = engine_clone.write();
                            e.cancel(current_seq_id);
                            break;
                        }
                        continue;
                    }
                };

                let item = match item {
                    Some(item) => item,
                    None => break,
                };

                match item {
                    StreamItem::Token(token, token_id) => {
                        if decode_start_time == 0 {
                            decode_start_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                        }

                        // Accumulate raw token text so reasoning blocks can
                        // be tokenized at finalization for `reasoning_tokens`.
                        // Cheap: text is already small per token.
                        full_decoded_text.push_str(&token);

                        // Use StreamToolParser for all tool call detection and buffering
                        if should_parse_tools {
                            match tool_parser.process_token(token_id, &token).await {
                                StreamResult::Content(text) => {
                                    buffering_since = None;
                                    buffering_cancel_requested = false;
                                    buffering_warned = false;
                                    if text.is_empty() {
                                        continue;
                                    }
                                    if tool_parser.contains_tool_markup(&text) {
                                        suppressed_tool_markup.push_str(&text);
                                        crate::log_warn!(
                                            "[Seq {}] Suppressing {} tool-markup chars pending final tool parsing",
                                            current_seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    if !pending_tool_calls.is_empty() {
                                        if text.trim().is_empty() {
                                            continue;
                                        }
                                        crate::log_warn!(
                                            "[Seq {}] Dropping {} trailing text chars after tool call emission",
                                            current_seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    // Send content to client (routed to reasoning_content if inside <think>)
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&text);
                                    }
                                    // Capture reasoning state before token processing for routing
                                    let was_in_reasoning = tool_parser.in_reasoning();
                                    if !reasoning_router.send(&text, was_in_reasoning, &stream_ctx)
                                    {
                                        crate::log_error!(
                                            "[Seq {}] Stream send error (disconnected)",
                                            current_seq_id
                                        );
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        break;
                                    }
                                }
                                StreamResult::Buffering => {
                                    // Parser is buffering, don't send anything to client yet.
                                    if buffering_since.is_none() {
                                        buffering_since = Some(Instant::now());
                                        buffering_warned = false;
                                    }
                                    if tool_parser.take_buffer_parse_activity() {
                                        buffering_since = Some(Instant::now());
                                        buffering_cancel_requested = false;
                                        buffering_warned = false;
                                    }
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&token);
                                    }
                                    if !buffering_warned
                                        && buffering_since.is_some_and(|since| {
                                            since.elapsed() >= Duration::from_secs(120)
                                        })
                                    {
                                        crate::log_warn!(
                                            "[Seq {}] Tool call buffering exceeded 120s; still waiting for completion",
                                            current_seq_id
                                        );
                                        buffering_warned = true;
                                    }
                                    if !buffering_cancel_requested
                                        && !tool_buffer_timeout.is_zero()
                                        && buffering_since.is_some_and(|since| {
                                            since.elapsed() >= tool_buffer_timeout
                                        })
                                    {
                                        crate::log_warn!(
                                            "[Seq {}] Tool buffering exceeded {:?}, cancelling sequence for EOS finalization",
                                            current_seq_id,
                                            tool_buffer_timeout
                                        );
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        buffering_cancel_requested = true;
                                    }
                                }
                                StreamResult::FlushBuffer(text) => {
                                    buffering_since = None;
                                    buffering_cancel_requested = false;
                                    buffering_warned = false;
                                    if text.is_empty() {
                                        continue;
                                    }
                                    if tool_parser.contains_tool_markup(&text) {
                                        suppressed_tool_markup.push_str(&text);
                                        crate::log_warn!(
                                            "[Seq {}] Suppressing {} buffered tool-markup chars pending final tool parsing",
                                            current_seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    if !pending_tool_calls.is_empty() {
                                        if text.trim().is_empty() {
                                            continue;
                                        }
                                        crate::log_warn!(
                                            "[Seq {}] Dropping {} buffered chars after tool call emission",
                                            current_seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    let safe_text =
                                        tool_parser.sanitize_tool_markup_for_display(&text);
                                    if safe_text != text {
                                        crate::log_warn!(
                                            "[Seq {}] Sanitized leaked tool markup in flushed text",
                                            current_seq_id
                                        );
                                    }
                                    // False positive - flush buffered content as text
                                    crate::log_info!(
                                        "[Seq {}] Flushing {} chars (false positive)",
                                        current_seq_id,
                                        safe_text.len()
                                    );
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&safe_text);
                                    }
                                    if !reasoning_router.send(&safe_text, false, &stream_ctx) {
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        break;
                                    }
                                }
                                StreamResult::ToolCalls(tools) => {
                                    buffering_since = None;
                                    buffering_cancel_requested = false;
                                    buffering_warned = false;
                                    pending_tool_calls.extend(tools);
                                }
                            }
                        } else {
                            // No tool parsing - stream directly. Advance the
                            // reasoning state machine so `in_reasoning()` is
                            // still accurate on the next iteration and the
                            // SSE router can split `<think>…</think>` into
                            // `delta.reasoning_content` for non-tools chats.
                            tool_parser.advance_reasoning_state(&token);
                            let was_in_reasoning = tool_parser.in_reasoning();
                            if token.is_empty() {
                                continue;
                            }
                            if let Some(ref l) = stream_logger {
                                l.log_stream_token(&token);
                            }
                            if !reasoning_router.send(&token, was_in_reasoning, &stream_ctx) {
                                crate::log_error!(
                                    "[Seq {}] Stream send error (disconnected)",
                                    current_seq_id
                                );
                                let mut e = engine_clone.write();
                                e.cancel(current_seq_id);
                                break;
                            }
                        }
                    }
                    StreamItem::Done((
                        prompt_start_time,
                        decode_start_time_done,
                        decode_finish_time,
                        final_decoded_length,
                        _stop_sequence,
                    )) => {
                        total_decoded_tokens += final_decoded_length;

                        // Flush any buffered content at end of stream
                        if should_parse_tools {
                            if let Some(finalized) =
                                tool_parser.finalize_buffered_tool_calls().await
                            {
                                match finalized {
                                    BufferedFinalizeResult::ToolCalls(calls) => {
                                        pending_tool_calls.extend(calls);
                                    }
                                    BufferedFinalizeResult::FlushBuffer(buffer) => {
                                        if !buffer.is_empty() {
                                            if tool_parser.contains_tool_markup(&buffer) {
                                                suppressed_tool_markup.push_str(&buffer);
                                                crate::log_warn!(
                                                    "[Seq {}] Suppressing {} buffered tool-markup chars at stream end",
                                                    current_seq_id,
                                                    buffer.len()
                                                );
                                            } else if !pending_tool_calls.is_empty() {
                                                crate::log_warn!(
                                                    "[Seq {}] Dropping {} buffered chars because tool calls were already parsed",
                                                    current_seq_id,
                                                    buffer.len()
                                                );
                                            } else {
                                                let safe_buffer = tool_parser
                                                    .sanitize_tool_markup_for_display(&buffer);
                                                if safe_buffer != buffer {
                                                    crate::log_warn!(
                                                        "[Seq {}] Sanitized leaked tool markup in partial buffer",
                                                        current_seq_id
                                                    );
                                                }
                                                crate::log_warn!(
                                                    "[Seq {}] Tool parse partial, flushing {} chars",
                                                    current_seq_id,
                                                    safe_buffer.len()
                                                );
                                                stream_ctx.send_token(&safe_buffer);
                                            }
                                        }
                                    }
                                }
                            }
                            if pending_tool_calls.is_empty() {
                                let accumulated = tool_parser.accumulated_output().to_string();
                                let reparsed =
                                    tool_parser.parse_complete_with_fallback(&accumulated).await;
                                if !reparsed.is_empty() {
                                    crate::log_warn!(
                                        "[Seq {}] Recovered {} tool call(s) from full-output fallback parse",
                                        current_seq_id,
                                        reparsed.len()
                                    );
                                    pending_tool_calls.extend(reparsed);
                                } else {
                                    let stripped =
                                        tool_parser.accumulated_output_without_reasoning();
                                    if stripped != accumulated && !stripped.trim().is_empty() {
                                        let reparsed_stripped = tool_parser
                                            .parse_complete_with_fallback(&stripped)
                                            .await;
                                        if !reparsed_stripped.is_empty() {
                                            crate::log_warn!(
                                                "[Seq {}] Recovered {} tool call(s) from reasoning-stripped fallback parse",
                                                current_seq_id,
                                                reparsed_stripped.len()
                                            );
                                            pending_tool_calls.extend(reparsed_stripped);
                                        }
                                    }
                                }
                            }
                            if pending_tool_calls.is_empty() && !suppressed_tool_markup.is_empty() {
                                let safe_suppressed = tool_parser
                                    .sanitize_tool_markup_for_display(&suppressed_tool_markup);
                                crate::log_warn!(
                                    "[Seq {}] Releasing {} suppressed tool-markup chars as sanitized text (no tool calls recovered)",
                                    current_seq_id,
                                    safe_suppressed.len()
                                );
                                if let Some(ref l) = stream_logger {
                                    l.log_stream_token(&safe_suppressed);
                                }
                                if !stream_ctx.send_token(&safe_suppressed) {
                                    let mut e = engine_clone.write();
                                    e.cancel(current_seq_id);
                                    break;
                                }
                            } else if !pending_tool_calls.is_empty()
                                && !suppressed_tool_markup.is_empty()
                            {
                                crate::log_warn!(
                                    "[Seq {}] Dropping {} suppressed tool-markup chars because tool calls were recovered",
                                    current_seq_id,
                                    suppressed_tool_markup.len()
                                );
                            }
                        }

                        let dropped = retain_tool_calls_forced_name(
                            &mut pending_tool_calls,
                            forced_tool_name.as_deref(),
                        );
                        if dropped > 0 {
                            crate::log_warn!(
                                "[Seq {}] Dropped {} tool call(s) that did not match forced tool_choice",
                                current_seq_id,
                                dropped
                            );
                        }

                        let (validated_calls, invalid_calls) =
                            filter_tool_calls(&pending_tool_calls, stream_tool_schemas.as_ref());

                        if !invalid_calls.is_empty() {
                            crate::log_error!(
                                "[Seq {}] Found {} invalid tool call(s)",
                                current_seq_id,
                                invalid_calls.len()
                            );
                            log_tool_calls("Invalid", &invalid_calls);
                            if let Some(ref l) = logger {
                                l.log_tool_calls("Invalid", &invalid_calls);
                            }
                        }
                        let invalid_feedback = build_invalid_tool_call_feedback(
                            &invalid_calls,
                            stream_tool_schemas.as_ref(),
                            forced_tool_name.as_deref(),
                        );

                        let (valid_calls, invalid_feedback) = if !invalid_calls.is_empty()
                            && !strict_tool_call_validation_enabled()
                        {
                            crate::log_error!("Invalid tool call feedback {:?}", invalid_feedback);
                            (pending_tool_calls, None)
                        } else {
                            (validated_calls, invalid_feedback)
                        };

                        let tool_calls: Option<Vec<crate::server::PublicToolCall>> =
                            if valid_calls.is_empty() {
                                None
                            } else {
                                log_tool_calls("Valid", &valid_calls);
                                Some(
                                    valid_calls
                                        .into_iter()
                                        .enumerate()
                                        .map(|(i, tc)| crate::server::PublicToolCall {
                                            index: Some(i),
                                            id: tc.id,
                                            type_: tc.tool_type,
                                            function: tc.function,
                                        })
                                        .collect(),
                                )
                            };
                        let has_any_tool_calls = tool_calls.is_some();
                        if let Some(ref streamed_tool_calls) = tool_calls {
                            let tool_chunk = ChatCompletionChunk {
                                id: format!("seq-{}", current_seq_id),
                                object: "chat.completion.chunk",
                                created,
                                model: model_id.to_string(),
                                choices: vec![ChatChoiceChunk {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(streamed_tool_calls.clone()),
                                    },
                                    finish_reason: None,
                                    error: None,
                                }],
                                usage: None,
                            };
                            let _ = response_tx.try_send(ChatResponse::Chunk(tool_chunk));
                        }
                        if !has_any_tool_calls {
                            if let Some(feedback) = invalid_feedback {
                                if let Some(ref l) = stream_logger {
                                    l.log_stream_token(&feedback);
                                }
                                if !stream_ctx.send_token(&feedback) {
                                    crate::log_error!(
                                        "[Seq {}] Stream send error (disconnected)",
                                        current_seq_id
                                    );
                                    let mut e = engine_clone.write();
                                    e.cancel(current_seq_id);
                                    break;
                                }
                            }
                        }
                        if tool_choice_required && !has_any_tool_calls {
                            crate::log_warn!(
                                "[Seq {}] Tool choice required but no tool calls were produced",
                                current_seq_id
                            );
                        }
                        // Send final chunk
                        let final_chunk = ChatCompletionChunk {
                            id: format!("seq-{}", current_seq_id),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.to_string(),
                            choices: vec![ChatChoiceChunk {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: if has_any_tool_calls {
                                    Some("tool_calls".to_string())
                                } else if total_decoded_tokens >= max_tokens {
                                    Some("length".to_string())
                                } else {
                                    Some("stop".to_string())
                                },
                                error: None,
                            }],
                            usage: include_usage.then_some({
                                let engine = engine_clone.read();
                                let cached = engine
                                    .get_num_cached_tokens_for_seq(current_seq_id)
                                    .unwrap_or(0);
                                let reasoning_tokens =
                                    crate::utils::chat_template::extract_reasoning_content(
                                        &full_decoded_text,
                                    )
                                    .and_then(|(r, _)| {
                                        engine.tokenizer.encode(r.as_str(), false).ok()
                                    })
                                    .map(|enc| enc.get_ids().len())
                                    .unwrap_or(0);
                                Usage {
                                    prompt_tokens: prompt_length,
                                    completion_tokens: total_decoded_tokens,
                                    total_tokens: prompt_length + total_decoded_tokens,
                                    prompt_tokens_details: (cached > 0).then_some(
                                        PromptTokensDetails {
                                            cached_tokens: cached,
                                        },
                                    ),
                                    completion_tokens_details: (reasoning_tokens > 0)
                                        .then_some(CompletionTokensDetails { reasoning_tokens }),
                                }
                            }),
                        };

                        if has_any_tool_calls {
                            crate::log_info!(
                                "Final chunk emitted after tool-call delta chunk(s): {:?}",
                                final_chunk
                            );
                        }
                        if let Some(ref l) = stream_logger {
                            l.log_stream_end(&final_chunk);
                        }
                        let _ = response_tx.try_send(ChatResponse::Chunk(final_chunk));

                        // Performance metrics
                        let prompt_time_taken = if decode_start_time_done > prompt_start_time {
                            (decode_start_time_done - prompt_start_time) as f32 / 1000.0
                        } else {
                            0.0
                        };
                        let decode_time_taken = if decode_finish_time > decode_start_time_done {
                            (decode_finish_time - decode_start_time_done) as f32 / 1000.0
                        } else {
                            0.0
                        };

                        crate::log_warn!("--- Performance Metrics ---");
                        if prompt_time_taken > 0.0 {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt: {} tokens in {:.2}s ({:.2} t/s)",
                                current_seq_id,
                                prompt_length,
                                prompt_time_taken,
                                prompt_length as f32 / prompt_time_taken.max(0.001)
                            );
                        } else {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt: {} tokens (cached)",
                                current_seq_id,
                                prompt_length
                            );
                        }
                        crate::log_info!(
                            "[Seq {}] ⏱️ Decoded: {} tokens in {:.2}s ({:.2} t/s)",
                            current_seq_id,
                            total_decoded_tokens,
                            decode_time_taken,
                            total_decoded_tokens as f32 / decode_time_taken.max(0.001)
                        );

                        break;
                    }
                    StreamItem::Error(e) => {
                        crate::log_error!("[Seq {}] Stream error: {}", current_seq_id, e);
                        let error_chunk = ChatCompletionChunk {
                            id: format!("seq-{}", current_seq_id),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.to_string(),
                            choices: vec![ChatChoiceChunk {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                error: Some(vec![ErrorMsg { message: Some(e) }]),
                            }],
                            usage: None,
                        };

                        let _ = response_tx.try_send(ChatResponse::Chunk(error_chunk));
                        break;
                    }
                    _ => {}
                }
            }

            let _ = response_tx.try_send(ChatResponse::Done);
        });

        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx: client_rx,
                status: StreamingStatus::Uninitialized,
                disconnect_tx: Some(disconnect_tx),
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        // Non-streaming
        let current_params = params.clone();
        let mut total_prompt_tokens = 0;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_time_taken = 0f32;
        let mut total_decoded_time_taken = 0f32;
        let mut choices = Vec::new();
        let tokenizer = {
            let e = data.engine.read();
            Arc::new(e.tokenizer.clone())
        };

        crate::log_info!(
            "Received completion request with {} messages",
            messages.len()
        );
        let preprocessed = {
            let e = data.engine.read();
            match e.preprocess(
                std::slice::from_ref(&current_params),
                std::slice::from_ref(&messages),
                &resolved_tools,
                false,
            ) {
                Ok(p) => p,
                Err(e) => {
                    crate::log_error!("Preprocess failed: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            }
        };
        let receivers = {
            let mut e = data.engine.write();
            match e.generate_sync(preprocessed, image_data, &logger) {
                Ok(receivers) => receivers,
                Err(e) => {
                    crate::log_error!("Completion generation failed: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            }
        };
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let results =
            match LLMEngine::collect_sync_results(receivers, tokenizer.clone(), logger.clone())
                .await
            {
                Ok(results) => results,
                Err(e) => {
                    crate::log_error!("Failed to collect completion results: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            };

        // Per-seq cached counts and decode_output snapshots read after the
        // loop, summed/scanned into single figures for the response Usage.
        let mut sync_seq_ids: Vec<usize> = Vec::with_capacity(results.len());
        let mut sync_decode_outputs: Vec<String> = Vec::with_capacity(results.len());
        for output in results {
            sync_seq_ids.push(output.seq_id);
            sync_decode_outputs.push(output.decode_output.clone());
            total_prompt_tokens += output.prompt_length;
            total_decoded_tokens += output.decoded_length;
            let prompt_time_taken =
                (output.decode_start_time - output.prompt_start_time) as f32 / 1000.0;
            let decode_time_taken =
                (output.decode_finish_time - output.decode_start_time) as f32 / 1000.0;
            total_prompt_time_taken += prompt_time_taken;
            total_decoded_time_taken += decode_time_taken;

            // Parse tool calls from the model output if tools were provided
            let (content, tool_calls) = if has_tools {
                let tool_parser = StreamToolParser::new_with_config(
                    &model_type,
                    parser_model_id.clone(),
                    tool_config.clone(),
                    resolved_tools.clone(),
                    enforce_parser.clone(),
                );
                let mut parsed_calls = tool_parser
                    .parse_complete_with_fallback(&output.decode_output)
                    .await;
                if parsed_calls.is_empty() {
                    let stripped =
                        crate::server::parser::strip_reasoning_blocks(&output.decode_output);
                    if stripped != output.decode_output && !stripped.trim().is_empty() {
                        parsed_calls = tool_parser.parse_complete_with_fallback(&stripped).await;
                        if !parsed_calls.is_empty() {
                            crate::log_warn!(
                                "Recovered {} tool call(s) from reasoning-stripped fallback parse",
                                parsed_calls.len()
                            );
                        }
                    }
                }
                let dropped =
                    retain_tool_calls_forced_name(&mut parsed_calls, forced_tool_name.as_deref());
                if dropped > 0 {
                    crate::log_warn!(
                        "Dropped {} tool call(s) that did not match forced tool_choice",
                        dropped
                    );
                }
                let (validated_calls, invalid_calls) =
                    filter_tool_calls(&parsed_calls, tool_schemas.as_ref());

                if !invalid_calls.is_empty() {
                    crate::log_warn!("Found {} invalid tool call(s)", invalid_calls.len());
                    log_tool_calls("Invalid", &invalid_calls);
                }
                let invalid_feedback = build_invalid_tool_call_feedback(
                    &invalid_calls,
                    tool_schemas.as_ref(),
                    forced_tool_name.as_deref(),
                );

                let valid_calls = validated_calls;
                if valid_calls.is_empty() {
                    if tool_choice_required {
                        crate::log_warn!("Tool choice required but no tool calls were produced");
                    }
                    let fallback_text = if let Some(feedback) = invalid_feedback {
                        feedback
                    } else {
                        if tool_parser.contains_tool_markup(&output.decode_output) {
                            tool_parser.sanitize_tool_markup_for_display(&output.decode_output)
                        } else {
                            output.decode_output.clone()
                        }
                    };
                    (Some(fallback_text), None)
                } else {
                    log_tool_calls("Valid", &valid_calls);
                    let public_calls = valid_calls
                        .into_iter()
                        .map(|tc| crate::server::PublicToolCall {
                            index: None,
                            id: tc.id,
                            type_: tc.tool_type,
                            function: tc.function,
                        })
                        .collect();
                    (None, Some(public_calls))
                }
            } else {
                (Some(output.decode_output), None)
            };

            // For external tool calls (not MCP), return to client
            let has_tool_calls = tool_calls.is_some();
            let (content, reasoning_content) = if crate::utils::env::stream_as_reasoning_content() {
                match content {
                    Some(text) => {
                        match crate::utils::chat_template::extract_reasoning_content(&text) {
                            Some((reasoning, remaining)) => {
                                let c = if remaining.is_empty() {
                                    None
                                } else {
                                    Some(remaining)
                                };
                                (c, Some(reasoning))
                            }
                            None => (Some(text), None),
                        }
                    }
                    None => (None, None),
                }
            } else {
                (content, None)
            };
            choices.push(ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content,
                    reasoning_content,
                    tool_calls,
                },
                finish_reason: if has_tool_calls {
                    Some("tool_calls".to_string())
                } else {
                    Some("stop".to_string())
                },
            });
        }

        crate::log_warn!("--- Performance Metrics ---");
        crate::log_info!(
            "[{} requests] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
            choices.len(),
            total_prompt_tokens,
            total_prompt_time_taken,
            total_prompt_tokens as f32 / total_prompt_time_taken.max(0.001)
        );
        crate::log_info!(
            "[{} requests] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
            choices.len(),
            total_decoded_tokens,
            total_decoded_time_taken,
            total_decoded_tokens as f32 / total_decoded_time_taken.max(0.001)
        );

        let (cached_tokens_total, reasoning_tokens_total): (usize, usize) = {
            let engine = data.engine.read();
            let cached: usize = sync_seq_ids
                .iter()
                .filter_map(|sid| engine.get_num_cached_tokens_for_seq(*sid))
                .sum();
            // Reasoning text is extracted directly from each choice's
            // `decode_output` rather than from `message.reasoning_content`,
            // because the latter is only populated on the env-gated tools
            // path. Counting from the raw text keeps the figure correct
            // for plain chat completions too.
            let reasoning: usize = sync_decode_outputs
                .iter()
                .filter_map(|text| {
                    crate::utils::chat_template::extract_reasoning_content(text).map(|(r, _)| r)
                })
                .filter_map(|text| engine.tokenizer.encode(text.as_str(), false).ok())
                .map(|enc| enc.get_ids().len())
                .sum();
            (cached, reasoning)
        };

        let response = ChatCompletionResponse {
            id: "cmpl-".to_string() + &Uuid::new_v4().to_string()[..8],
            object: "chat.completion",
            created,
            model: model_id.to_string(),
            choices,
            usage: {
                Usage {
                    prompt_tokens: total_prompt_tokens,
                    completion_tokens: total_decoded_tokens,
                    total_tokens: total_prompt_tokens + total_decoded_tokens,
                    prompt_tokens_details: (cached_tokens_total > 0).then_some(
                        PromptTokensDetails {
                            cached_tokens: cached_tokens_total,
                        },
                    ),
                    completion_tokens_details: (reasoning_tokens_total > 0).then_some(
                        CompletionTokensDetails {
                            reasoning_tokens: reasoning_tokens_total,
                        },
                    ),
                }
            },
        };

        if let Some(ref l) = logger {
            l.log_response(&response);
        }
        ChatResponder::Completion(response)
    }
}

#[utoipa::path(
    post,
    tag = "xinfer",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings"))
)]
pub async fn create_embeddings(
    State(data): State<Arc<ServerData>>,
    request: Json<EmbeddingRequest>,
) -> ChatResponder {
    let EmbeddingRequest {
        model,
        input,
        encoding_format,
        embedding_type,
    } = request.0;
    let inputs = input.into_vec();
    if inputs.is_empty() {
        return ChatResponder::ValidationError("input cannot be empty".to_string());
    }

    let model_name = model.unwrap_or_else(|| "default".to_string());

    let mut engine = data.engine.write();
    let (vectors, prompt_tokens) = match engine.embed(&inputs, embedding_type.clone()) {
        Ok(res) => res,
        Err(e) => return ChatResponder::ModelError(format!("Embedding generation failed: {e:?}")),
    };

    crate::log_warn!(
        "Finished with {} embedding vectors and {} prompt tokens",
        vectors.len(),
        prompt_tokens
    );
    let data: Vec<EmbeddingData> = vectors
        .into_iter()
        .enumerate()
        .map(|(idx, vec)| {
            let embedding = match encoding_format {
                EncodingFormat::Float => EmbeddingOutput::Vector(vec),
                EncodingFormat::Base64 => {
                    let bytes = bytemuck::cast_slice::<f32, u8>(&vec);
                    EmbeddingOutput::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
            };
            EmbeddingData {
                object: "embedding",
                embedding,
                index: idx,
            }
        })
        .collect();

    ChatResponder::Embedding(EmbeddingResponse {
        object: "list",
        data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
}

#[utoipa::path(
    get,
    tag = "xinfer",
    path = "/v1/usage",
    request_body = UsageQuery,
    responses((status = 200, description = "Token Usage Request"))
)]
pub async fn get_usage(
    State(state): State<Arc<ServerData>>,
    Query(request): Query<UsageQuery>,
) -> ChatResponder {
    let engine = state.engine.read();
    let stats = match engine.get_usage_stats(request.session_id.clone()) {
        Ok(s) => s,
        Err(e) => {
            return ChatResponder::InternalError(format!("Failed to obtain usage status {:?}", e));
        }
    };

    ChatResponder::Usage(UsageResponse {
        token_used: stats.token_used,
        max_model_len: stats.max_model_len,
        used_kvcache_tokens: stats.used_kvcache_tokens,
        total_kv_cache_tokens: stats.total_kv_cache_tokens,
        swap_used: stats.swap_used,
        total_swap_memory: stats.total_swap_memory,
        session_status: stats.session_status,
    })
}

#[utoipa::path(
    post,
    tag = "xinfer",
    path = "/tokenize",
    request_body = TokenizeRequest,
    responses((status = 200, description = "Tokenize text or messages"))
)]
pub async fn tokenize(
    State(data): State<Arc<ServerData>>,
    request: Json<TokenizeRequest>,
) -> ChatResponder {
    let add_special_tokens = request.add_special_tokens.unwrap_or(true);

    // Get text to tokenize based on input type
    let (text, input_type) = match &request.0.input {
        TokenizeInput::Text { prompt } => (prompt.clone(), "text"),
        TokenizeInput::Messages { messages } => {
            // For messages, we need to apply chat template
            // First convert to internal Message format
            let img_cfg = {
                let e = data.engine.read();
                e.img_cfg.clone()
            };
            let (converted_messages, _) =
                match build_messages_and_images(messages, img_cfg.as_ref()) {
                    Ok(result) => result,
                    Err(e) => {
                        return ChatResponder::ValidationError(format!(
                            "Message processing failed: {:?}",
                            e
                        ));
                    }
                };

            // Apply chat template using engine's template
            let engine = data.engine.read();
            let mut template = engine.get_chat_template();
            template.set_messages(&converted_messages);
            let prompt = match template.apply_chat_template(&Vec::new(), false) {
                Ok(prompt) => prompt,
                Err(e) => {
                    return ChatResponder::InternalError(format!(
                        "Failed to apply chat template: {:?}",
                        e
                    ));
                }
            };
            (prompt, "messages")
        }
    };

    let input_chars = text.len();

    // Get tokenizer and tokenize
    let tokenizer = {
        let e = data.engine.read();
        e.tokenizer.clone()
    };

    let encoding = match tokenizer.encode(text.as_str(), add_special_tokens) {
        Ok(enc) => enc,
        Err(e) => {
            return ChatResponder::InternalError(format!("Tokenization failed: {:?}", e));
        }
    };

    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let count = tokens.len();

    crate::log_info!(
        "[Tokenize] input_type={}, input_chars={}, output_tokens={}",
        input_type,
        input_chars,
        count
    );

    ChatResponder::Tokenize(TokenizeResponse {
        tokens,
        count,
        max_model_len: data.econfig.max_model_len,
    })
}

#[utoipa::path(
    post,
    tag = "xinfer",
    path = "/detokenize",
    request_body = DetokenizeRequest,
    responses((status = 200, description = "Detokenize tokens to text"))
)]
pub async fn detokenize(
    State(data): State<Arc<ServerData>>,
    request: Json<DetokenizeRequest>,
) -> ChatResponder {
    let skip_special_tokens = request.skip_special_tokens.unwrap_or(true);

    let tokenizer = {
        let e = data.engine.read();
        e.tokenizer.clone()
    };

    let input_tokens = request.tokens.len();

    let prompt = match tokenizer.decode(&request.tokens, skip_special_tokens) {
        Ok(text) => text,
        Err(e) => {
            return ChatResponder::InternalError(format!("Detokenization failed: {:?}", e));
        }
    };

    crate::log_info!(
        "[Detokenize] input_tokens={}, output_chars={}",
        input_tokens,
        prompt.len()
    );

    ChatResponder::Detokenize(DetokenizeResponse { prompt })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_ctx() -> (StreamingContext, flume::Receiver<ChatResponse>) {
        let (tx, rx) = flume::unbounded();
        let ctx = StreamingContext::new(1, "test-model".to_string(), 0, tx);
        (ctx, rx)
    }

    fn collect_deltas(rx: &flume::Receiver<ChatResponse>) -> Vec<(Option<String>, Option<String>)> {
        let mut deltas = Vec::new();
        while let Ok(resp) = rx.try_recv() {
            if let ChatResponse::Chunk(chunk) = resp {
                for choice in &chunk.choices {
                    deltas.push((
                        choice.delta.content.clone(),
                        choice.delta.reasoning_content.clone(),
                    ));
                }
            }
        }
        deltas
    }

    #[test]
    fn reasoning_router_disabled_sends_all_as_content() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(false);
        assert!(router.send("<think>hello</think>world", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].0.as_deref(), Some("<think>hello</think>world"));
        assert_eq!(deltas[0].1, None);
    }

    #[test]
    fn reasoning_router_disabled_no_tools_sends_reasoning_as_content() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(false);
        assert!(router.send("<think>", true, &ctx));
        assert!(router.send("reasoning text", true, &ctx));
        assert!(router.send("</think>", false, &ctx));
        assert!(router.send("main content", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 4);
        for d in &deltas {
            assert!(
                d.1.is_none(),
                "reasoning_content must be None when router is disabled"
            );
        }
        assert_eq!(deltas[0].0.as_deref(), Some("<think>"));
        assert_eq!(deltas[1].0.as_deref(), Some("reasoning text"));
        assert_eq!(deltas[2].0.as_deref(), Some("</think>"));
        assert_eq!(deltas[3].0.as_deref(), Some("main content"));
    }

    #[test]
    fn reasoning_router_strips_markers_and_routes_reasoning() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        // Token "<think>hello" arrives while parser says in_reasoning=true
        assert!(router.send("<think>hello", true, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].0, None);
        assert_eq!(deltas[0].1.as_deref(), Some("hello"));
    }

    #[test]
    fn reasoning_router_strips_end_marker_routes_content() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        // Token "</think>world" arrives while parser says in_reasoning=false (already transitioned)
        assert!(router.send("</think>world", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].0.as_deref(), Some("world"));
        assert_eq!(deltas[0].1, None);
    }

    #[test]
    fn reasoning_router_handles_split_tokens() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        // Simulating: <think> | reasoning part | </think> | content part
        // Parser state: false→true for <think>, true for reasoning, true→false for </think>, false for content
        assert!(router.send("<think>", true, &ctx)); // marker only, stripped
        assert!(router.send("reasoning part", true, &ctx));
        assert!(router.send("</think>", false, &ctx)); // marker only, stripped
        assert!(router.send("content part", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].0, None);
        assert_eq!(deltas[0].1.as_deref(), Some("reasoning part"));
        assert_eq!(deltas[1].0.as_deref(), Some("content part"));
        assert_eq!(deltas[1].1, None);
    }

    #[test]
    fn reasoning_router_handles_prefilled_reasoning() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        // Prefilled: prompt ends with <think>, parser starts in_reasoning=true
        assert!(router.send("I'm thinking", true, &ctx));
        assert!(router.send("</think>", false, &ctx));
        assert!(router.send("answer", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].0, None);
        assert_eq!(deltas[0].1.as_deref(), Some("I'm thinking"));
        assert_eq!(deltas[1].0.as_deref(), Some("answer"));
        assert_eq!(deltas[1].1, None);
    }

    #[test]
    fn reasoning_router_handles_qwen_markers() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        assert!(router.send("<|think|>reasoning", true, &ctx));
        assert!(router.send("<|/think|>answer", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].0, None);
        assert_eq!(deltas[0].1.as_deref(), Some("reasoning"));
        assert_eq!(deltas[1].0.as_deref(), Some("answer"));
        assert_eq!(deltas[1].1, None);
    }

    #[test]
    fn reasoning_router_empty_text_is_noop() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        assert!(router.send("", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert!(deltas.is_empty());
    }

    #[test]
    fn reasoning_router_marker_only_token_sends_nothing() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        assert!(router.send("<think>", true, &ctx));
        assert!(router.send("</think>", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert!(
            deltas.is_empty(),
            "Pure marker tokens should produce no output"
        );
    }

    #[test]
    fn reasoning_router_plain_content_no_markers() {
        let (ctx, rx) = make_test_ctx();
        let router = ReasoningContentRouter::new(true);
        assert!(router.send("hello world", false, &ctx));
        let deltas = collect_deltas(&rx);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].0.as_deref(), Some("hello world"));
        assert_eq!(deltas[0].1, None);
    }

    #[test]
    fn strip_reasoning_markers_removes_all_marker_types() {
        assert_eq!(strip_reasoning_markers("<think>hello</think>"), "hello");
        assert_eq!(strip_reasoning_markers("<|think|>hi<|/think|>"), "hi");
        assert_eq!(strip_reasoning_markers("[THINK]hi[/THINK]"), "hi");
        assert_eq!(strip_reasoning_markers("<thought>hi</thought>"), "hi");
        assert_eq!(strip_reasoning_markers("no markers"), "no markers");
        assert_eq!(strip_reasoning_markers("<think>"), "");
        assert_eq!(strip_reasoning_markers("</think>world"), "world");
    }

    #[test]
    fn validates_openai_tool_messages_with_known_tool_call_id() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![crate::tools::new_tool_call("call_1", "lookup", "{}")]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage::tool_result("call_1", "{\"ok\":true}"),
        ];

        assert!(validate_openai_tool_messages(&messages).is_ok());
    }

    #[test]
    fn rejects_openai_tool_message_with_unknown_tool_call_id() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![crate::tools::new_tool_call("call_1", "lookup", "{}")]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage::tool_result("call_unknown", "{\"ok\":true}"),
        ];

        let err = validate_openai_tool_messages(&messages).unwrap_err();
        assert!(err.contains("unexpected tool_call_id"));
    }

    #[test]
    fn rejects_duplicate_openai_tool_result_ids() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![crate::tools::new_tool_call("call_1", "lookup", "{}")]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage::tool_result("call_1", "{\"ok\":true}"),
            ChatMessage::tool_result("call_1", "{\"ok\":false}"),
        ];

        let err = validate_openai_tool_messages(&messages).unwrap_err();
        assert!(err.contains("duplicate tool_call_id"));
    }

    #[test]
    fn rejects_non_adjacent_openai_tool_result_response() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![crate::tools::new_tool_call("call_1", "lookup", "{}")]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage::text("user", "let us skip the tool result"),
        ];

        let err = validate_openai_tool_messages(&messages).unwrap_err();
        assert!(err.contains("must be role=tool"));
    }

    #[test]
    fn validates_openai_multiple_tool_results_in_order() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![
                    crate::tools::new_tool_call("call_1", "lookup", "{}"),
                    crate::tools::new_tool_call("call_2", "lookup", "{}"),
                ]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage::tool_result("call_1", "{\"ok\":true}"),
            ChatMessage::tool_result("call_2", "{\"ok\":true}"),
            ChatMessage::text("assistant", "done"),
        ];

        assert!(validate_openai_tool_messages(&messages).is_ok());
    }
}
