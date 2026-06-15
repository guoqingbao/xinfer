use super::{
    build_messages_and_images, ChatMessage, ImageUrlContent, MessageContent, MessageContentType,
    ServerData,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::server::logger::ChatCompletionLogger;
use crate::server::parser::{BufferedFinalizeResult, StreamResult, StreamToolParser};
use crate::tools::helpers::{
    build_invalid_tool_call_feedback, build_tool_schema_map, filter_tool_calls,
    retain_tool_calls_forced_name, strict_tool_call_validation_enabled,
};
use crate::tools::{Tool, ToolCall, ToolChoice};
use crate::utils::config::SamplingParams;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use flume::{Receiver, TrySendError};
use futures::Stream;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    env,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::task;
use tokio::time;
use uuid::Uuid;

const STABLE_CLAUDE_SETTINGS_FILENAME: &str =
    "claude-settings-00000000-0000-0000-0000-000000000000.json";
const STABLE_ANTHROPIC_CCH: &str = "00000";

static CLAUDE_SETTINGS_UUID_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"claude-settings-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.json",
    )
    .expect("valid claude-settings UUID regex")
});

static ANTHROPIC_BILLING_CCH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(x-anthropic-billing-header:[^\n]*\bcch=)[^;\s]+")
        .expect("valid anthropic billing cch regex")
});

const CLAUDE_REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
    ("<|channel>", "<channel|>"),
];
const SYNTHETIC_THINKING_SIGNATURE_PREFIX: &str = "xinfer-thinking-v1:";

fn strip_nested_reasoning_markers(text: &str) -> String {
    let mut result = text.to_string();
    for &(open, close) in CLAUDE_REASONING_MARKERS {
        result = result.replace(open, "");
        result = result.replace(close, "");
    }
    result
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ClaudeContent {
    Text(String),
    Blocks(Vec<ClaudeContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ClaudeContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ClaudeImageSource },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: ClaudeToolResultContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ClaudeImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ClaudeToolResultContent {
    Text(String),
    Blocks(Vec<ClaudeContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: ClaudeContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ClaudeSystem {
    Text(String),
    Blocks(Vec<ClaudeContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "input_schema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ClaudeToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
    #[serde(rename = "none")]
    None,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeMessageRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(default)]
    pub system: Option<ClaudeSystem>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(default)]
    pub tool_choice: Option<ClaudeToolChoice>,
    #[serde(default)]
    pub thinking: Option<ClaudeThinking>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ClaudeThinking {
    Bool(bool),
    Config(ClaudeThinkingConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeThinkingConfig {
    #[serde(rename = "type")]
    pub mode: String,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeTokenCountRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(default)]
    pub system: Option<ClaudeSystem>,
    #[serde(default)]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct ClaudeTokenCountResponse {
    pub input_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ClaudeContentBlockOut>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: ClaudeUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ClaudeContentBlockOut {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Serialize)]
pub struct ClaudeUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub message: ClaudeMessageResponse,
}

#[derive(Debug, Serialize)]
pub struct ClaudeContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
    pub content_block: ClaudeContentBlockOut,
}

#[derive(Debug, Serialize)]
pub struct ClaudeContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
    pub delta: ClaudeContentDelta,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ClaudeContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        #[serde(rename = "partial_json")]
        partial_json: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SyntheticThinkingSignature {
    version: u8,
    suffix_placeholder: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaudeThinkingBlock {
    thinking: String,
    signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedClaudeAssistantOutput {
    thinking_blocks: Vec<ClaudeThinkingBlock>,
    text: String,
}

#[derive(Debug, Serialize)]
pub struct ClaudeContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: ClaudeMessageDelta,
    pub usage: ClaudeUsageDelta,
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ClaudeUsageDelta {
    pub output_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ClaudeErrorResponse {
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub error: ClaudeErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ClaudeErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

pub enum ClaudeResponder {
    Streamer(Sse<ClaudeStreamer>),
    Message(ClaudeMessageResponse),
    TokenCount(ClaudeTokenCountResponse),
    Error(ClaudeErrorResponse, StatusCode),
}

impl IntoResponse for ClaudeResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ClaudeResponder::Streamer(s) => s.into_response(),
            ClaudeResponder::Message(m) => Json(m).into_response(),
            ClaudeResponder::TokenCount(c) => Json(c).into_response(),
            ClaudeResponder::Error(err, status) => {
                let mut resp = Json(err).into_response();
                *resp.status_mut() = status;
                resp
            }
        }
    }
}

#[derive(PartialEq)]
enum ClaudeStreamingStatus {
    Uninitialized,
    Started,
    Interrupted,
    Stopped,
}

enum ClaudeStreamItem {
    Event(Event),
    Done,
}

pub struct ClaudeStreamer {
    rx: Receiver<ClaudeStreamItem>,
    status: ClaudeStreamingStatus,
}

impl Stream for ClaudeStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == ClaudeStreamingStatus::Stopped {
            return Poll::Ready(None);
        }
        match self.rx.try_recv() {
            Ok(item) => match item {
                ClaudeStreamItem::Event(event) => {
                    if self.status != ClaudeStreamingStatus::Started {
                        self.status = ClaudeStreamingStatus::Started;
                    }
                    Poll::Ready(Some(Ok(event)))
                }
                ClaudeStreamItem::Done => {
                    self.status = ClaudeStreamingStatus::Stopped;
                    Poll::Ready(None)
                }
            },
            Err(err) => {
                if self.status == ClaudeStreamingStatus::Started
                    && err == flume::TryRecvError::Disconnected
                {
                    self.status = ClaudeStreamingStatus::Interrupted;
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

#[derive(Debug)]
enum StreamSendError {
    Full,
    Disconnected,
}

const FINAL_EVENT_TIMEOUT_MS: u64 = 500;

struct ClaudeStreamingContext {
    seq_id: usize,
    response_tx: flume::Sender<ClaudeStreamItem>,
}

impl ClaudeStreamingContext {
    fn new(seq_id: usize, response_tx: flume::Sender<ClaudeStreamItem>) -> Self {
        Self {
            seq_id,
            response_tx,
        }
    }

    fn send_event(&self, event: Event) -> Result<(), StreamSendError> {
        match self.response_tx.try_send(ClaudeStreamItem::Event(event)) {
            Ok(_) => Ok(()),
            Err(TrySendError::Full(_)) => Err(StreamSendError::Full),
            Err(TrySendError::Disconnected(_)) => Err(StreamSendError::Disconnected),
        }
    }

    fn send_json_event<T: Serialize>(&self, name: &str, data: &T) -> Result<(), StreamSendError> {
        match Event::default().event(name).json_data(data) {
            Ok(event) => self.send_event(event),
            Err(err) => {
                crate::log_error!(
                    "[Seq {}] Failed to serialize {} event: {:?}",
                    self.seq_id,
                    name,
                    err
                );
                Err(StreamSendError::Disconnected)
            }
        }
    }
}

fn tool_choice_to_openai(choice: &Option<ClaudeToolChoice>) -> Option<ToolChoice> {
    match choice {
        Some(ClaudeToolChoice::Auto) => Some(ToolChoice::auto()),
        Some(ClaudeToolChoice::Any) => Some(ToolChoice::required()),
        Some(ClaudeToolChoice::None) => Some(ToolChoice::none()),
        Some(ClaudeToolChoice::Tool { name }) => Some(ToolChoice::function(name.clone())),
        None => None,
    }
}

fn encode_synthetic_thinking_signature(suffix_placeholder: &str) -> String {
    let payload = SyntheticThinkingSignature {
        version: 1,
        suffix_placeholder: suffix_newlines_to_spaces(suffix_placeholder),
    };
    let json = serde_json::to_vec(&payload).unwrap_or_default();
    format!(
        "{}{}",
        SYNTHETIC_THINKING_SIGNATURE_PREFIX,
        URL_SAFE_NO_PAD.encode(json)
    )
}

fn decode_synthetic_thinking_signature(signature: &str) -> Option<SyntheticThinkingSignature> {
    let encoded = signature.strip_prefix(SYNTHETIC_THINKING_SIGNATURE_PREFIX)?;
    let bytes = URL_SAFE_NO_PAD.decode(encoded).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn replay_text_for_thinking_block(thinking: &str, signature: Option<&str>) -> String {
    let thinking = normalize_suffix_thinking_text(thinking);
    let suffix_placeholder = signature
        .and_then(decode_synthetic_thinking_signature)
        .map(|payload| payload.suffix_placeholder)
        .unwrap_or_else(|| " ".to_string());
    format!(" {}{}", thinking, suffix_placeholder)
}

fn suffix_newlines_to_spaces(text: &str) -> String {
    text.chars()
        .map(|ch| if ch == '\n' || ch == '\r' { ' ' } else { ch })
        .collect()
}

fn normalize_suffix_thinking_text(thinking: &str) -> String {
    if thinking.trim().is_empty() {
        suffix_newlines_to_spaces(thinking)
    } else {
        thinking.to_string()
    }
}

fn find_reasoning_start(text: &str) -> Option<(usize, &'static str, &'static str)> {
    CLAUDE_REASONING_MARKERS
        .iter()
        .filter_map(|(start, end)| text.find(start).map(|idx| (idx, *start, *end)))
        .min_by_key(|(idx, _, _)| *idx)
}

fn find_reasoning_end(text: &str) -> Option<(usize, &'static str)> {
    CLAUDE_REASONING_MARKERS
        .iter()
        .filter_map(|(_, end)| text.find(end).map(|idx| (idx, *end)))
        .min_by_key(|(idx, _)| *idx)
}

fn append_reasoning_segment(
    rendered: &mut String,
    thinking_blocks: &mut Vec<ClaudeThinkingBlock>,
    thinking: &str,
    trailing_ws: &str,
) {
    let suffix_placeholder = format!(" {}", trailing_ws);
    let normalized_thinking = normalize_suffix_thinking_text(thinking);
    let signature = encode_synthetic_thinking_signature(&suffix_placeholder);
    if normalized_thinking.trim().is_empty() {
        rendered.push_str(&replay_text_for_thinking_block(
            &normalized_thinking,
            Some(signature.as_str()),
        ));
    } else {
        thinking_blocks.push(ClaudeThinkingBlock {
            thinking: normalized_thinking,
            signature,
        });
    }
}

fn parse_claude_assistant_output(text: &str) -> ParsedClaudeAssistantOutput {
    let mut remaining = text;
    let mut rendered = String::new();
    let mut thinking_blocks = Vec::new();

    while !remaining.is_empty() {
        let next_start = find_reasoning_start(remaining);
        let next_end = find_reasoning_end(remaining);
        if let Some((end_idx, end_marker)) = next_end {
            let start_before_end = next_start
                .as_ref()
                .is_some_and(|(start_idx, _, _)| *start_idx < end_idx);
            if !start_before_end {
                let after_end = &remaining[end_idx + end_marker.len()..];
                let trailing_ws_len = after_end
                    .char_indices()
                    .take_while(|(_, ch)| ch.is_whitespace())
                    .map(|(idx, ch)| idx + ch.len_utf8())
                    .last()
                    .unwrap_or(0);
                append_reasoning_segment(
                    &mut rendered,
                    &mut thinking_blocks,
                    &remaining[..end_idx],
                    &after_end[..trailing_ws_len],
                );
                remaining = &after_end[trailing_ws_len..];
                continue;
            }
        }

        let Some((start_idx, start_marker, end_marker)) = next_start else {
            rendered.push_str(remaining);
            break;
        };

        rendered.push_str(&remaining[..start_idx]);
        let after_start = &remaining[start_idx + start_marker.len()..];
        let Some(end_idx) = after_start.find(end_marker) else {
            rendered.push_str(&remaining[start_idx..]);
            break;
        };

        let after_end = &after_start[end_idx + end_marker.len()..];
        let trailing_ws_len = after_end
            .char_indices()
            .take_while(|(_, ch)| ch.is_whitespace())
            .map(|(idx, ch)| idx + ch.len_utf8())
            .last()
            .unwrap_or(0);
        append_reasoning_segment(
            &mut rendered,
            &mut thinking_blocks,
            &after_start[..end_idx],
            &after_end[..trailing_ws_len],
        );
        remaining = &after_end[trailing_ws_len..];
    }

    ParsedClaudeAssistantOutput {
        thinking_blocks,
        text: rendered,
    }
}

fn normalize_claude_volatile_text(text: &mut String) -> usize {
    let mut occurrences = CLAUDE_SETTINGS_UUID_RE.find_iter(text.as_str()).count();
    if occurrences > 0 {
        *text = CLAUDE_SETTINGS_UUID_RE
            .replace_all(text.as_str(), STABLE_CLAUDE_SETTINGS_FILENAME)
            .into_owned();
    }

    let cch_occurrences = ANTHROPIC_BILLING_CCH_RE
        .captures_iter(text.as_str())
        .count();
    if cch_occurrences > 0 {
        *text = ANTHROPIC_BILLING_CCH_RE
            .replace_all(text.as_str(), format!("${{1}}{STABLE_ANTHROPIC_CCH}"))
            .into_owned();
    }
    occurrences += cch_occurrences;
    occurrences
}

fn normalize_json_strings(value: &mut Value) -> usize {
    match value {
        Value::String(text) => normalize_claude_volatile_text(text),
        Value::Array(items) => items.iter_mut().map(normalize_json_strings).sum(),
        Value::Object(map) => map.values_mut().map(normalize_json_strings).sum(),
        _ => 0,
    }
}

fn normalize_tool_result_content(content: &mut ClaudeToolResultContent) -> usize {
    match content {
        ClaudeToolResultContent::Text(text) => normalize_claude_volatile_text(text),
        ClaudeToolResultContent::Blocks(blocks) => normalize_content_blocks(blocks),
    }
}

fn normalize_content_blocks(blocks: &mut [ClaudeContentBlock]) -> usize {
    let mut normalized = 0usize;
    for block in blocks {
        normalized += match block {
            ClaudeContentBlock::Text { text } => normalize_claude_volatile_text(text),
            ClaudeContentBlock::Thinking { thinking, .. } => {
                normalize_claude_volatile_text(thinking)
            }
            ClaudeContentBlock::RedactedThinking { data } => normalize_claude_volatile_text(data),
            ClaudeContentBlock::ToolUse { input, .. } => normalize_json_strings(input),
            ClaudeContentBlock::ToolResult { content, .. } => {
                normalize_tool_result_content(content)
            }
            ClaudeContentBlock::Image { .. } => 0,
        };
    }
    normalized
}

fn normalize_claude_content(content: &mut ClaudeContent) -> usize {
    match content {
        ClaudeContent::Text(text) => normalize_claude_volatile_text(text),
        ClaudeContent::Blocks(blocks) => normalize_content_blocks(blocks),
    }
}

fn normalize_claude_system(system: &mut ClaudeSystem) -> usize {
    match system {
        ClaudeSystem::Text(text) => normalize_claude_volatile_text(text),
        ClaudeSystem::Blocks(blocks) => normalize_content_blocks(blocks),
    }
}

fn normalize_claude_request_for_prefix_cache(request: &mut ClaudeMessageRequest) -> usize {
    let mut normalized = 0usize;
    if let Some(system) = request.system.as_mut() {
        normalized += normalize_claude_system(system);
    }
    for message in &mut request.messages {
        normalized += normalize_claude_content(&mut message.content);
    }
    if let Some(tools) = request.tools.as_mut() {
        for tool in tools {
            if let Some(description) = tool.description.as_mut() {
                normalized += normalize_claude_volatile_text(description);
            }
            normalized += normalize_json_strings(&mut tool.input_schema);
        }
    }
    normalized
}

fn claude_tools_to_tools(tools: &[ClaudeTool]) -> Vec<Tool> {
    tools
        .iter()
        .map(|tool| {
            let description = tool.description.clone().unwrap_or_default();
            crate::tools::function_tool(&tool.name, description)
                .parameters_schema(tool.input_schema.clone())
                .build()
        })
        .collect()
}

fn system_to_chat_message(system: &ClaudeSystem) -> Result<ChatMessage, String> {
    let items = match system {
        ClaudeSystem::Text(text) => {
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![MessageContent::Text { text: text.clone() }]
            }
        }
        ClaudeSystem::Blocks(blocks) => blocks_to_message_content(blocks, true)?,
    };

    let content = build_message_content_type(items).ok_or_else(|| {
        "system content must include at least one text or image block".to_string()
    })?;

    Ok(ChatMessage {
        role: "system".to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

fn blocks_to_message_content(
    blocks: &[ClaudeContentBlock],
    allow_images: bool,
) -> Result<Vec<MessageContent>, String> {
    let mut items = Vec::new();
    for block in blocks {
        match block {
            ClaudeContentBlock::Text { text } => {
                if !text.trim().is_empty() {
                    items.push(MessageContent::Text { text: text.clone() });
                }
            }
            ClaudeContentBlock::Thinking { .. } => {
                return Err("thinking blocks are not valid in plain content".to_string())
            }
            ClaudeContentBlock::RedactedThinking { .. } => {
                return Err("redacted_thinking blocks are not valid in plain content".to_string())
            }
            ClaudeContentBlock::Image { source } => {
                if !allow_images {
                    return Err("image blocks are not supported here".to_string());
                }
                match source {
                    ClaudeImageSource::Base64 { media_type, data } => {
                        let base64 = format!("data:{};base64,{}", media_type, data);
                        items.push(MessageContent::ImageBase64 {
                            image_base64: base64,
                        });
                    }
                    ClaudeImageSource::Url { url } => {
                        items.push(MessageContent::ImageUrl {
                            image_url: ImageUrlContent::Url(url.clone()),
                        });
                    }
                }
            }
            ClaudeContentBlock::ToolUse { .. } => {
                return Err("tool_use blocks are not valid in plain content".to_string())
            }
            ClaudeContentBlock::ToolResult { .. } => {
                return Err("tool_result blocks are not valid in plain content".to_string())
            }
        }
    }
    Ok(items)
}

fn build_message_content_type(items: Vec<MessageContent>) -> Option<MessageContentType> {
    if items.is_empty() {
        return None;
    }
    if items.len() == 1 {
        Some(MessageContentType::Single(items[0].clone()))
    } else {
        Some(MessageContentType::Multi(items))
    }
}

fn push_text_content(items: &mut Vec<MessageContent>, text: String) {
    if text.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(MessageContent::Text { text: existing }) => existing.push_str(&text),
        _ => items.push(MessageContent::Text { text }),
    }
}

fn tool_result_content_to_text(content: &ClaudeToolResultContent) -> Result<String, String> {
    match content {
        ClaudeToolResultContent::Text(text) => Ok(text.clone()),
        ClaudeToolResultContent::Blocks(blocks) => {
            let mut combined = String::new();
            for block in blocks {
                match block {
                    ClaudeContentBlock::Text { text } => {
                        if !combined.is_empty() {
                            combined.push(' ');
                        }
                        combined.push_str(text);
                    }
                    ClaudeContentBlock::Thinking { thinking, .. } => {
                        if !combined.is_empty() {
                            combined.push(' ');
                        }
                        combined.push_str(thinking);
                    }
                    _ => {
                        return Err(
                            "only text blocks are supported inside tool_result content".to_string()
                        )
                    }
                }
            }
            Ok(combined)
        }
    }
}

fn flush_content_message(
    out: &mut Vec<ChatMessage>,
    role: &str,
    items: &mut Vec<MessageContent>,
    reasoning_content: Option<String>,
) {
    let content = build_message_content_type(std::mem::take(items));
    if content.is_some() || reasoning_content.is_some() {
        out.push(ChatMessage {
            role: role.to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content,
        });
    }
}

fn flush_tool_call_message(out: &mut Vec<ChatMessage>, calls: &mut Vec<ToolCall>) {
    if !calls.is_empty() {
        out.push(ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(std::mem::take(calls)),
            tool_call_id: None,
            reasoning_content: None,
        });
    }
}

fn validate_claude_tool_result_protocol(messages: &[ClaudeMessage]) -> Result<(), String> {
    let mut known_tool_use_ids: HashSet<String> = HashSet::new();
    let mut awaiting_tool_results: Option<HashSet<String>> = None;

    for (idx, message) in messages.iter().enumerate() {
        let role = message.role.as_str();
        if role != "user" && role != "assistant" {
            return Err(format!(
                "unsupported role at messages[{idx}]: {}",
                message.role
            ));
        }
        let mut consumed_expected_results = false;

        if let Some(expected) = awaiting_tool_results.take() {
            if role != "user" {
                return Err(format!(
                    "messages[{idx}] must be a user message with tool_result blocks after assistant tool_use"
                ));
            }

            let ClaudeContent::Blocks(blocks) = &message.content else {
                return Err(format!(
                    "messages[{idx}] must provide tool_result blocks (plain text is not valid here)"
                ));
            };

            let mut provided: HashSet<String> = HashSet::new();
            let mut seen_non_tool_result = false;
            for (block_idx, block) in blocks.iter().enumerate() {
                match block {
                    ClaudeContentBlock::ToolResult { tool_use_id, .. } => {
                        if seen_non_tool_result {
                            return Err(format!(
                                "messages[{idx}].content[{block_idx}] tool_result blocks must appear before text/image blocks"
                            ));
                        }
                        let id = tool_use_id.trim();
                        if id.is_empty() {
                            return Err(format!(
                                "messages[{idx}].content[{block_idx}] tool_result requires non-empty tool_use_id"
                            ));
                        }
                        if !provided.insert(id.to_string()) {
                            return Err(format!(
                                "messages[{idx}] contains duplicate tool_result for tool_use_id '{}'",
                                id
                            ));
                        }
                    }
                    _ => seen_non_tool_result = true,
                }
            }

            if provided.is_empty() {
                return Err(format!(
                    "messages[{idx}] must start with tool_result blocks for pending tool_use ids"
                ));
            }

            if provided != expected {
                let mut expected_ids = expected.into_iter().collect::<Vec<_>>();
                expected_ids.sort();
                let mut provided_ids = provided.into_iter().collect::<Vec<_>>();
                provided_ids.sort();
                return Err(format!(
                    "messages[{idx}] tool_result ids do not match pending assistant tool_use ids. expected={:?}, provided={:?}",
                    expected_ids, provided_ids
                ));
            }
            consumed_expected_results = true;
        }

        let ClaudeContent::Blocks(blocks) = &message.content else {
            continue;
        };

        let mut message_tool_use_ids: HashSet<String> = HashSet::new();
        let mut has_tool_use = false;
        let mut has_tool_result = false;
        let mut seen_non_tool_result = false;

        for (block_idx, block) in blocks.iter().enumerate() {
            match block {
                ClaudeContentBlock::ToolUse { id, .. } => {
                    if role != "assistant" {
                        return Err(format!(
                            "messages[{idx}].content[{block_idx}] tool_use blocks must be in assistant messages"
                        ));
                    }
                    let call_id = id.trim();
                    if call_id.is_empty() {
                        return Err(format!(
                            "messages[{idx}].content[{block_idx}] tool_use requires non-empty id"
                        ));
                    }
                    if !known_tool_use_ids.insert(call_id.to_string()) {
                        return Err(format!(
                            "messages[{idx}] duplicates tool_use id '{}' from a prior message",
                            call_id
                        ));
                    }
                    message_tool_use_ids.insert(call_id.to_string());
                    has_tool_use = true;
                }
                ClaudeContentBlock::ToolResult { tool_use_id, .. } => {
                    if role != "user" {
                        return Err(format!(
                            "messages[{idx}].content[{block_idx}] tool_result blocks must be in user messages"
                        ));
                    }
                    if seen_non_tool_result {
                        return Err(format!(
                            "messages[{idx}].content[{block_idx}] tool_result blocks must appear before text/image blocks"
                        ));
                    }
                    let result_id = tool_use_id.trim();
                    if result_id.is_empty() {
                        return Err(format!(
                            "messages[{idx}].content[{block_idx}] tool_result requires non-empty tool_use_id"
                        ));
                    }
                    has_tool_result = true;
                }
                _ => {
                    if role == "user" {
                        seen_non_tool_result = true;
                    }
                }
            }
        }

        if has_tool_use {
            if !message_tool_use_ids.is_empty() {
                awaiting_tool_results = Some(message_tool_use_ids);
            }
        } else if has_tool_result && !consumed_expected_results {
            return Err(format!(
                "messages[{idx}] contains tool_result blocks without a preceding assistant tool_use message"
            ));
        }
    }

    if let Some(pending) = awaiting_tool_results {
        let mut ids = pending.into_iter().collect::<Vec<_>>();
        ids.sort();
        return Err(format!(
            "Missing tool_result response for assistant tool_use ids: {:?}",
            ids
        ));
    }

    Ok(())
}

fn convert_claude_message(message: &ClaudeMessage) -> Result<Vec<ChatMessage>, String> {
    let role = message.role.as_str();
    if role != "user" && role != "assistant" {
        return Err(format!("unsupported role: {}", message.role));
    }

    match &message.content {
        ClaudeContent::Text(text) => {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![ChatMessage::text(role, text.clone())]);
        }
        ClaudeContent::Blocks(blocks) => {
            let mut out = Vec::new();
            let mut content_items: Vec<MessageContent> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut thinking_content: Option<String> = None;

            for block in blocks {
                match block {
                    ClaudeContentBlock::Text { text } => {
                        if !tool_calls.is_empty() {
                            flush_tool_call_message(&mut out, &mut tool_calls);
                        }
                        if !text.is_empty() {
                            push_text_content(&mut content_items, text.clone());
                        }
                    }
                    ClaudeContentBlock::Thinking {
                        thinking,
                        signature: _,
                    } => {
                        if role != "assistant" {
                            return Err("thinking blocks must be in assistant messages".to_string());
                        }
                        if !tool_calls.is_empty() {
                            flush_tool_call_message(&mut out, &mut tool_calls);
                        }
                        let cleaned = strip_nested_reasoning_markers(thinking);
                        if !cleaned.trim().is_empty() {
                            match &mut thinking_content {
                                Some(existing) => {
                                    existing.push('\n');
                                    existing.push_str(cleaned.trim());
                                }
                                None => {
                                    thinking_content = Some(cleaned.trim().to_string());
                                }
                            }
                        }
                    }
                    ClaudeContentBlock::RedactedThinking { .. } => {
                        if role != "assistant" {
                            return Err("redacted_thinking blocks must be in assistant messages"
                                .to_string());
                        }
                        if !tool_calls.is_empty() {
                            flush_tool_call_message(&mut out, &mut tool_calls);
                        }
                    }
                    ClaudeContentBlock::Image { source } => {
                        if !tool_calls.is_empty() {
                            flush_tool_call_message(&mut out, &mut tool_calls);
                        }
                        match source {
                            ClaudeImageSource::Base64 { media_type, data } => {
                                let base64 = format!("data:{};base64,{}", media_type, data);
                                content_items.push(MessageContent::ImageBase64 {
                                    image_base64: base64,
                                });
                            }
                            ClaudeImageSource::Url { url } => {
                                content_items.push(MessageContent::ImageUrl {
                                    image_url: ImageUrlContent::Url(url.clone()),
                                });
                            }
                        }
                    }
                    ClaudeContentBlock::ToolUse { id, name, input } => {
                        if role != "assistant" {
                            return Err("tool_use blocks must be in assistant messages".to_string());
                        }
                        flush_content_message(
                            &mut out,
                            role,
                            &mut content_items,
                            thinking_content.take(),
                        );
                        let args = serde_json::to_string(input).map_err(|err| err.to_string())?;
                        tool_calls.push(crate::tools::new_tool_call(
                            id.clone(),
                            name.clone(),
                            args,
                        ));
                    }
                    ClaudeContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        if role != "user" {
                            return Err("tool_result blocks must be in user messages".to_string());
                        }
                        flush_content_message(&mut out, role, &mut content_items, None);
                        flush_tool_call_message(&mut out, &mut tool_calls);
                        let raw_text = tool_result_content_to_text(content)?;
                        let is_error = is_error.unwrap_or(false);
                        let text = if raw_text.trim().is_empty() {
                            if is_error {
                                "<tool_use_error>Tool returned an error with no message.</tool_use_error>"
                                    .to_string()
                            } else {
                                "Tool executed successfully with no textual output.".to_string()
                            }
                        } else if is_error && !raw_text.contains("<tool_use_error>") {
                            format!("<tool_use_error>{}</tool_use_error>", raw_text)
                        } else {
                            raw_text
                        };

                        out.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(MessageContentType::PureText(text)),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                            reasoning_content: None,
                        });
                    }
                }
            }

            flush_content_message(&mut out, role, &mut content_items, thinking_content.take());
            flush_tool_call_message(&mut out, &mut tool_calls);
            Ok(out)
        }
    }
}

fn build_chat_messages(request: &ClaudeMessageRequest) -> Result<Vec<ChatMessage>, String> {
    validate_claude_tool_result_protocol(&request.messages)?;

    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(system_to_chat_message(system)?);
    }
    for message in &request.messages {
        messages.extend(convert_claude_message(message)?);
    }
    if messages.is_empty() {
        return Err("messages cannot be empty".to_string());
    }
    Ok(messages)
}

fn stop_reason_from_decoding(
    has_tool_calls: bool,
    decoded_tokens: usize,
    max_tokens: usize,
    stop_sequence: Option<&str>,
) -> String {
    if has_tool_calls {
        "tool_use".to_string()
    } else if decoded_tokens >= max_tokens {
        "max_tokens".to_string()
    } else if stop_sequence.is_some() {
        "stop_sequence".to_string()
    } else {
        "end_turn".to_string()
    }
}

fn tool_calls_to_blocks(tool_calls: &[ToolCall]) -> Vec<ClaudeContentBlockOut> {
    tool_calls
        .iter()
        .map(|call| {
            let args_str = call.function.arguments.as_deref().unwrap_or("{}");
            let input = serde_json::from_str(args_str).unwrap_or_else(|_| {
                crate::log_warn!(
                    "Failed to parse tool arguments for '{}'",
                    call.function.name
                );
                Value::Null
            });
            ClaudeContentBlockOut::ToolUse {
                id: call.id.clone(),
                name: call.function.name.clone(),
                input,
            }
        })
        .collect()
}

fn send_text_with_start(
    stream_ctx: &ClaudeStreamingContext,
    text_block_started: &mut bool,
    text_block_index: usize,
    text: &str,
) -> Result<(), StreamSendError> {
    if !*text_block_started {
        let start_block = ClaudeContentBlockStartEvent {
            event_type: "content_block_start",
            index: text_block_index,
            content_block: ClaudeContentBlockOut::Text {
                text: String::new(),
            },
        };
        stream_ctx.send_json_event("content_block_start", &start_block)?;
        *text_block_started = true;
    }
    send_text_delta(stream_ctx, text_block_index, text)
}

fn send_text_delta(
    stream_ctx: &ClaudeStreamingContext,
    index: usize,
    text: &str,
) -> Result<(), StreamSendError> {
    let delta = ClaudeContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ClaudeContentDelta::TextDelta {
            text: text.to_string(),
        },
    };
    stream_ctx.send_json_event("content_block_delta", &delta)
}

fn send_tool_use_block(
    stream_ctx: &ClaudeStreamingContext,
    index: usize,
    call: &ToolCall,
) -> Result<(), StreamSendError> {
    let start_payload = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": call.id.clone(),
            "name": call.function.name.clone(),
            "input": {}
        }
    });
    stream_ctx.send_json_event("content_block_start", &start_payload)?;

    let input_json = call.function.arguments.clone().unwrap_or_default();

    let empty_delta = ClaudeContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ClaudeContentDelta::InputJsonDelta {
            partial_json: String::new(),
        },
    };
    stream_ctx.send_json_event("content_block_delta", &empty_delta)?;

    let delta = ClaudeContentBlockDeltaEvent {
        event_type: "content_block_delta",
        index,
        delta: ClaudeContentDelta::InputJsonDelta {
            partial_json: input_json,
        },
    };
    stream_ctx.send_json_event("content_block_delta", &delta)?;

    let stop = ClaudeContentBlockStopEvent {
        event_type: "content_block_stop",
        index,
    };
    stream_ctx.send_json_event("content_block_stop", &stop)?;
    Ok(())
}

fn longest_partial_marker_suffix(text: &str, markers: &[&str]) -> usize {
    let mut longest = 0usize;
    for marker in markers {
        let max_len = marker.len().saturating_sub(1).min(text.len());
        for len in 1..=max_len {
            if text.ends_with(&marker[..len]) {
                longest = longest.max(len);
            }
        }
    }
    longest
}

#[derive(Debug)]
enum ClaudeThinkingStreamMode {
    Text,
    ThinkingCandidate {
        end_marker: &'static str,
        buffered: String,
    },
    Thinking {
        index: usize,
        end_marker: &'static str,
    },
    PendingThinkingClose {
        index: usize,
        trailing_ws: String,
    },
    PendingEmptyThinking {
        placeholder: String,
    },
}

#[derive(Debug)]
struct ClaudeThinkingStreamEmitter {
    next_block_index: usize,
    open_text_block: Option<usize>,
    parse_buffer: String,
    mode: ClaudeThinkingStreamMode,
}

impl ClaudeThinkingStreamEmitter {
    fn new() -> Self {
        Self {
            next_block_index: 0,
            open_text_block: None,
            parse_buffer: String::new(),
            mode: ClaudeThinkingStreamMode::Text,
        }
    }

    fn open_text_block_index(&self) -> Option<usize> {
        self.open_text_block
    }

    fn ensure_text_block(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
    ) -> Result<usize, StreamSendError> {
        if let Some(index) = self.open_text_block {
            return Ok(index);
        }
        let index = self.next_block_index;
        let start_block = ClaudeContentBlockStartEvent {
            event_type: "content_block_start",
            index,
            content_block: ClaudeContentBlockOut::Text {
                text: String::new(),
            },
        };
        stream_ctx.send_json_event("content_block_start", &start_block)?;
        self.open_text_block = Some(index);
        self.next_block_index += 1;
        Ok(index)
    }

    fn emit_text(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
        logger: Option<&ChatCompletionLogger>,
        text: &str,
    ) -> Result<(), StreamSendError> {
        if text.is_empty() {
            return Ok(());
        }
        if let Some(logger) = logger {
            logger.log_stream_token(text);
        }
        let index = self.ensure_text_block(stream_ctx)?;
        send_text_delta(stream_ctx, index, text)
    }

    fn close_text_block(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
    ) -> Result<(), StreamSendError> {
        let Some(index) = self.open_text_block.take() else {
            return Ok(());
        };
        let stop_event = ClaudeContentBlockStopEvent {
            event_type: "content_block_stop",
            index,
        };
        stream_ctx.send_json_event("content_block_stop", &stop_event)
    }

    fn start_thinking_candidate(&mut self, end_marker: &'static str) {
        self.mode = ClaudeThinkingStreamMode::ThinkingCandidate {
            end_marker,
            buffered: String::new(),
        };
    }

    fn start_thinking_block(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
    ) -> Result<usize, StreamSendError> {
        self.close_text_block(stream_ctx)?;
        let index = self.next_block_index;
        let start_block = ClaudeContentBlockStartEvent {
            event_type: "content_block_start",
            index,
            content_block: ClaudeContentBlockOut::Thinking {
                thinking: String::new(),
                signature: None,
            },
        };
        stream_ctx.send_json_event("content_block_start", &start_block)?;
        self.next_block_index += 1;
        Ok(index)
    }

    fn emit_thinking_delta(
        stream_ctx: &ClaudeStreamingContext,
        index: usize,
        text: &str,
    ) -> Result<(), StreamSendError> {
        let cleaned = strip_nested_reasoning_markers(text);
        if cleaned.is_empty() {
            return Ok(());
        }
        let delta = ClaudeContentBlockDeltaEvent {
            event_type: "content_block_delta",
            index,
            delta: ClaudeContentDelta::ThinkingDelta { thinking: cleaned },
        };
        stream_ctx.send_json_event("content_block_delta", &delta)
    }

    fn close_thinking_block(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
        index: usize,
        suffix_placeholder: &str,
    ) -> Result<(), StreamSendError> {
        let signature = encode_synthetic_thinking_signature(suffix_placeholder);
        let delta = ClaudeContentBlockDeltaEvent {
            event_type: "content_block_delta",
            index,
            delta: ClaudeContentDelta::SignatureDelta { signature },
        };
        stream_ctx.send_json_event("content_block_delta", &delta)?;
        let stop = ClaudeContentBlockStopEvent {
            event_type: "content_block_stop",
            index,
        };
        stream_ctx.send_json_event("content_block_stop", &stop)?;
        self.mode = ClaudeThinkingStreamMode::Text;
        Ok(())
    }

    fn push_chunk(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
        logger: Option<&ChatCompletionLogger>,
        text: &str,
    ) -> Result<(), StreamSendError> {
        if text.is_empty() {
            return Ok(());
        }
        self.parse_buffer.push_str(text);
        self.drain(stream_ctx, logger, false)
    }

    fn finish(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
        logger: Option<&ChatCompletionLogger>,
    ) -> Result<usize, StreamSendError> {
        self.drain(stream_ctx, logger, true)?;
        self.close_text_block(stream_ctx)?;
        Ok(self.next_block_index)
    }

    fn drain(
        &mut self,
        stream_ctx: &ClaudeStreamingContext,
        logger: Option<&ChatCompletionLogger>,
        finalize: bool,
    ) -> Result<(), StreamSendError> {
        loop {
            let mode = std::mem::replace(&mut self.mode, ClaudeThinkingStreamMode::Text);
            match mode {
                ClaudeThinkingStreamMode::Text => {
                    let next_start = find_reasoning_start(&self.parse_buffer);
                    let next_end = find_reasoning_end(&self.parse_buffer);
                    let Some((start_idx, start_marker, end_marker)) = next_start else {
                        if let Some((end_idx, end_marker)) = next_end {
                            let buffered = self.parse_buffer[..end_idx].to_string();
                            self.parse_buffer.drain(..end_idx);
                            self.mode = ClaudeThinkingStreamMode::ThinkingCandidate {
                                end_marker,
                                buffered,
                            };
                            continue;
                        }
                        let reserve = if finalize {
                            0
                        } else {
                            let markers: Vec<&str> = CLAUDE_REASONING_MARKERS
                                .iter()
                                .map(|(start, _)| *start)
                                .collect();
                            longest_partial_marker_suffix(&self.parse_buffer, &markers)
                        };
                        let emit_len = self.parse_buffer.len().saturating_sub(reserve);
                        if emit_len == 0 {
                            self.mode = ClaudeThinkingStreamMode::Text;
                            return Ok(());
                        }
                        let emit_text = self.parse_buffer[..emit_len].to_string();
                        self.parse_buffer.drain(..emit_len);
                        self.emit_text(stream_ctx, logger, &emit_text)?;
                        self.mode = ClaudeThinkingStreamMode::Text;
                        continue;
                    };

                    if let Some((end_idx, end_marker_only)) = next_end {
                        if end_idx < start_idx {
                            let buffered = self.parse_buffer[..end_idx].to_string();
                            self.parse_buffer.drain(..end_idx);
                            self.mode = ClaudeThinkingStreamMode::ThinkingCandidate {
                                end_marker: end_marker_only,
                                buffered,
                            };
                            continue;
                        }
                    }

                    if start_idx > 0 {
                        let emit_text = self.parse_buffer[..start_idx].to_string();
                        self.parse_buffer.drain(..start_idx);
                        self.emit_text(stream_ctx, logger, &emit_text)?;
                        self.mode = ClaudeThinkingStreamMode::Text;
                        continue;
                    }

                    if self.parse_buffer.len() < start_marker.len() {
                        self.mode = ClaudeThinkingStreamMode::Text;
                        return Ok(());
                    }

                    self.parse_buffer.drain(..start_marker.len());
                    self.start_thinking_candidate(end_marker);
                }
                ClaudeThinkingStreamMode::ThinkingCandidate {
                    end_marker,
                    mut buffered,
                } => {
                    if let Some(end_idx) = self.parse_buffer.find(end_marker) {
                        let delta_text = self.parse_buffer[..end_idx].to_string();
                        self.parse_buffer.drain(..end_idx + end_marker.len());
                        buffered.push_str(&delta_text);
                        let cleaned = strip_nested_reasoning_markers(&buffered);
                        if cleaned.trim().is_empty() {
                            self.mode = ClaudeThinkingStreamMode::PendingEmptyThinking {
                                placeholder: format!(
                                    " {}",
                                    normalize_suffix_thinking_text(&cleaned)
                                ),
                            };
                        } else {
                            let index = self.start_thinking_block(stream_ctx)?;
                            Self::emit_thinking_delta(stream_ctx, index, &cleaned)?;
                            self.mode = ClaudeThinkingStreamMode::PendingThinkingClose {
                                index,
                                trailing_ws: String::new(),
                            };
                        }
                        continue;
                    }

                    let reserve = if finalize {
                        0
                    } else {
                        longest_partial_marker_suffix(&self.parse_buffer, &[end_marker])
                    };
                    let emit_len = self.parse_buffer.len().saturating_sub(reserve);
                    if emit_len == 0 {
                        if finalize {
                            let remaining = std::mem::take(&mut self.parse_buffer);
                            buffered.push_str(&remaining);
                            let cleaned = strip_nested_reasoning_markers(&buffered);
                            if cleaned.trim().is_empty() {
                                let placeholder =
                                    format!(" {}", normalize_suffix_thinking_text(&cleaned));
                                self.emit_text(stream_ctx, logger, &placeholder)?;
                                self.mode = ClaudeThinkingStreamMode::Text;
                            } else {
                                let index = self.start_thinking_block(stream_ctx)?;
                                Self::emit_thinking_delta(stream_ctx, index, &cleaned)?;
                                self.close_thinking_block(stream_ctx, index, " ")?;
                            }
                            continue;
                        }
                        self.mode = ClaudeThinkingStreamMode::ThinkingCandidate {
                            end_marker,
                            buffered,
                        };
                        return Ok(());
                    }
                    let delta_text = self.parse_buffer[..emit_len].to_string();
                    self.parse_buffer.drain(..emit_len);
                    buffered.push_str(&delta_text);
                    let cleaned = strip_nested_reasoning_markers(&buffered);
                    if !cleaned.trim().is_empty() {
                        let index = self.start_thinking_block(stream_ctx)?;
                        Self::emit_thinking_delta(stream_ctx, index, &cleaned)?;
                        self.mode = ClaudeThinkingStreamMode::Thinking { index, end_marker };
                    } else {
                        self.mode = ClaudeThinkingStreamMode::ThinkingCandidate {
                            end_marker,
                            buffered,
                        };
                    }
                }
                ClaudeThinkingStreamMode::Thinking { index, end_marker } => {
                    if let Some(end_idx) = self.parse_buffer.find(end_marker) {
                        let delta_text = self.parse_buffer[..end_idx].to_string();
                        self.parse_buffer.drain(..end_idx + end_marker.len());
                        Self::emit_thinking_delta(stream_ctx, index, &delta_text)?;
                        self.mode = ClaudeThinkingStreamMode::PendingThinkingClose {
                            index,
                            trailing_ws: String::new(),
                        };
                        continue;
                    }

                    let reserve = if finalize {
                        0
                    } else {
                        longest_partial_marker_suffix(&self.parse_buffer, &[end_marker])
                    };
                    let emit_len = self.parse_buffer.len().saturating_sub(reserve);
                    if emit_len == 0 {
                        if finalize {
                            let remaining = std::mem::take(&mut self.parse_buffer);
                            Self::emit_thinking_delta(stream_ctx, index, &remaining)?;
                            self.close_thinking_block(stream_ctx, index, " ")?;
                            continue;
                        }
                        self.mode = ClaudeThinkingStreamMode::Thinking { index, end_marker };
                        return Ok(());
                    }
                    let delta_text = self.parse_buffer[..emit_len].to_string();
                    self.parse_buffer.drain(..emit_len);
                    Self::emit_thinking_delta(stream_ctx, index, &delta_text)?;
                    self.mode = ClaudeThinkingStreamMode::Thinking { index, end_marker };
                }
                ClaudeThinkingStreamMode::PendingThinkingClose {
                    index,
                    mut trailing_ws,
                } => {
                    let ws_len = self
                        .parse_buffer
                        .char_indices()
                        .take_while(|(_, ch)| ch.is_whitespace())
                        .map(|(idx, ch)| idx + ch.len_utf8())
                        .last()
                        .unwrap_or(0);
                    if ws_len > 0 {
                        trailing_ws.push_str(&self.parse_buffer[..ws_len]);
                        self.parse_buffer.drain(..ws_len);
                    }
                    if !self.parse_buffer.is_empty() || finalize {
                        let suffix_placeholder = format!(" {}", trailing_ws);
                        self.close_thinking_block(stream_ctx, index, &suffix_placeholder)?;
                        continue;
                    }
                    self.mode =
                        ClaudeThinkingStreamMode::PendingThinkingClose { index, trailing_ws };
                    return Ok(());
                }
                ClaudeThinkingStreamMode::PendingEmptyThinking { mut placeholder } => {
                    let ws_len = self
                        .parse_buffer
                        .char_indices()
                        .take_while(|(_, ch)| ch.is_whitespace())
                        .map(|(idx, ch)| idx + ch.len_utf8())
                        .last()
                        .unwrap_or(0);
                    if ws_len > 0 {
                        placeholder
                            .push_str(&suffix_newlines_to_spaces(&self.parse_buffer[..ws_len]));
                        self.parse_buffer.drain(..ws_len);
                    }
                    if !self.parse_buffer.is_empty() || finalize {
                        let text = std::mem::take(&mut placeholder);
                        self.emit_text(stream_ctx, logger, &text)?;
                        self.mode = ClaudeThinkingStreamMode::Text;
                        continue;
                    }
                    self.mode = ClaudeThinkingStreamMode::PendingEmptyThinking { placeholder };
                    return Ok(());
                }
            }
        }
    }
}

async fn send_json_event_with_timeout<T: Serialize>(
    seq_id: usize,
    response_tx: &flume::Sender<ClaudeStreamItem>,
    name: &str,
    data: &T,
    timeout: Duration,
) -> bool {
    let event = match Event::default().event(name).json_data(data) {
        Ok(event) => event,
        Err(err) => {
            crate::log_error!(
                "[Seq {}] Failed to serialize {} event: {:?}",
                seq_id,
                name,
                err
            );
            return false;
        }
    };

    match time::timeout(
        timeout,
        response_tx.send_async(ClaudeStreamItem::Event(event)),
    )
    .await
    {
        Ok(Ok(_)) => true,
        Ok(Err(err)) => {
            crate::log_warn!(
                "[Seq {}] Failed to send {} after backpressure: {:?}",
                seq_id,
                name,
                err
            );
            false
        }
        Err(_) => {
            crate::log_warn!(
                "[Seq {}] Timed out sending {} after backpressure",
                seq_id,
                name
            );
            false
        }
    }
}

async fn send_done_with_timeout(
    seq_id: usize,
    response_tx: &flume::Sender<ClaudeStreamItem>,
    timeout: Duration,
) -> bool {
    match time::timeout(timeout, response_tx.send_async(ClaudeStreamItem::Done)).await {
        Ok(Ok(_)) => true,
        Ok(Err(err)) => {
            crate::log_warn!(
                "[Seq {}] Failed to send stream done after backpressure: {:?}",
                seq_id,
                err
            );
            false
        }
        Err(_) => {
            crate::log_warn!(
                "[Seq {}] Timed out sending stream done after backpressure",
                seq_id
            );
            false
        }
    }
}

async fn finalize_stream_on_backpressure(
    seq_id: usize,
    response_tx: &flume::Sender<ClaudeStreamItem>,
    text_block_open: bool,
    text_block_index: usize,
    total_decoded_tokens: usize,
    include_message_delta: bool,
) {
    let timeout = Duration::from_millis(FINAL_EVENT_TIMEOUT_MS);

    if text_block_open {
        let stop_event = ClaudeContentBlockStopEvent {
            event_type: "content_block_stop",
            index: text_block_index,
        };
        let _ = send_json_event_with_timeout(
            seq_id,
            response_tx,
            "content_block_stop",
            &stop_event,
            timeout,
        )
        .await;
    }

    if include_message_delta {
        let message_delta = ClaudeMessageDeltaEvent {
            event_type: "message_delta",
            delta: ClaudeMessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: ClaudeUsageDelta {
                output_tokens: total_decoded_tokens,
            },
        };
        let _ = send_json_event_with_timeout(
            seq_id,
            response_tx,
            "message_delta",
            &message_delta,
            timeout,
        )
        .await;
    }

    let message_stop = ClaudeMessageStopEvent {
        event_type: "message_stop",
    };
    let _ =
        send_json_event_with_timeout(seq_id, response_tx, "message_stop", &message_stop, timeout)
            .await;
    let _ = send_done_with_timeout(seq_id, response_tx, timeout).await;
}

async fn handle_stream_send_error(
    err: StreamSendError,
    seq_id: usize,
    response_tx: &flume::Sender<ClaudeStreamItem>,
    text_block_open: bool,
    text_block_index: usize,
    total_decoded_tokens: usize,
    include_message_delta: bool,
) {
    match err {
        StreamSendError::Full => {
            crate::log_warn!(
                "[Seq {}] SSE buffer full; closing stream with stop/done",
                seq_id
            );
            finalize_stream_on_backpressure(
                seq_id,
                response_tx,
                text_block_open,
                text_block_index,
                total_decoded_tokens,
                include_message_delta,
            )
            .await;
        }
        StreamSendError::Disconnected => {
            crate::log_warn!("[Seq {}] SSE client disconnected", seq_id);
        }
    }
}

fn log_tool_calls(label: &str, seq_id: usize, tool_calls: &[ToolCall]) {
    if tool_calls.is_empty() {
        return;
    }
    let summary = tool_calls
        .iter()
        .map(|call| {
            let args = call
                .function
                .arguments
                .as_deref()
                .unwrap_or("")
                .replace('\n', " ");
            let truncated = if args.len() > 160 {
                let snippet: String = args.chars().take(160).collect();
                format!("{}...", snippet)
            } else {
                args
            };
            format!("{}(args={})", call.function.name, truncated)
        })
        .collect::<Vec<_>>()
        .join(", ");
    crate::log_info!("[Seq {}] {} tool call(s): {}", seq_id, label, summary);
}

fn log_performance_metrics(
    seq_id: usize,
    prompt_length: usize,
    total_decoded_tokens: usize,
    prompt_start_time: usize,
    decode_start_time_done: usize,
    decode_finish_time: usize,
) {
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

    crate::log_warn!("--- Claude Performance Metrics ---");
    if prompt_time_taken > 0.0 {
        crate::log_info!(
            "[Seq {}] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
            seq_id,
            prompt_length,
            prompt_time_taken,
            prompt_length as f32 / prompt_time_taken.max(0.001)
        );
    } else {
        crate::log_info!(
            "[Seq {}] ⏱️ Prompt tokens: {} (cached context)",
            seq_id,
            prompt_length
        );
    }
    crate::log_info!(
        "[Seq {}] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
        seq_id,
        total_decoded_tokens,
        decode_time_taken,
        total_decoded_tokens as f32 / decode_time_taken.max(0.001)
    );
}

fn thinking_to_bool(thinking: &Option<ClaudeThinking>) -> Option<bool> {
    match thinking {
        Some(ClaudeThinking::Bool(value)) => Some(*value),
        Some(ClaudeThinking::Config(config)) => {
            if config.budget_tokens.is_some() {
                crate::log_warn!("Anthropic thinking budget_tokens provided but ignored");
            }
            match config.mode.as_str() {
                "enabled" | "adaptive" => Some(true),
                "disabled" => Some(false),
                other => {
                    crate::log_warn!("Anthropic thinking mode '{}' not recognized", other);
                    None
                }
            }
        }
        None => None,
    }
}

pub async fn messages(
    State(data): State<Arc<ServerData>>,
    request: Json<ClaudeMessageRequest>,
) -> ClaudeResponder {
    // Create logger for this request (None if XINFER_CHAT_LOGGER not set to true)
    let logger = ChatCompletionLogger::new_claude();
    if let Some(ref l) = logger {
        l.log_raw_request(&*request);
    }
    let mut request = request.0;
    let normalized = normalize_claude_request_for_prefix_cache(&mut request);
    if normalized > 0 {
        crate::log_info!(
            "Normalized {} volatile Claude prompt token(s) for prefix cache stability.",
            normalized
        );
    }

    let chat_messages = match build_chat_messages(&request) {
        Ok(messages) => messages,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "invalid_request_error".to_string(),
                        message: err,
                    },
                },
                StatusCode::UNPROCESSABLE_ENTITY,
            );
        }
    };

    let model_id = if request.model.trim().is_empty() {
        "default".to_string()
    } else {
        request.model.clone()
    };
    let max_tokens = request
        .max_tokens
        .unwrap_or(data.econfig.max_tokens.unwrap_or(16384));
    let use_stream = request.stream.unwrap_or(false);
    let tool_buffer_timeout = Duration::from_secs(
        env::var("XINFER_TOOL_BUFFER_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(600),
    );

    let anthropic_thinking = thinking_to_bool(&request.thinking);
    let anthropic_thinking_enabled = anthropic_thinking == Some(true);
    let mut params = SamplingParams::new_with_max_tokens(max_tokens);
    params.temperature = request.temperature;
    params.top_k = request.top_k.map(|v| v as isize);
    params.top_p = request.top_p;
    params.thinking = anthropic_thinking;
    if let Some(stop_sequences) = &request.stop_sequences {
        if !stop_sequences.is_empty() {
            params.stop_sequences = Some(stop_sequences.clone());
        }
    }

    let request_tools = request.tools.as_deref().unwrap_or_default();
    let mcp_tools = data
        .mcp_manager
        .as_ref()
        .map(|manager| manager.cached_tools())
        .unwrap_or_default();
    let converted_tools = claude_tools_to_tools(request_tools);
    let mut resolved_tools = if !converted_tools.is_empty() {
        converted_tools
    } else {
        mcp_tools.clone()
    };
    let mut forced_tool_name: Option<String> = None;
    let mut tool_choice_required = false;

    match request.tool_choice.as_ref() {
        Some(ClaudeToolChoice::None) => {
            resolved_tools.clear();
        }
        Some(ClaudeToolChoice::Tool { name }) => {
            tool_choice_required = true;
            forced_tool_name = Some(name.clone());
        }
        Some(ClaudeToolChoice::Any) => {
            tool_choice_required = true;
        }
        Some(ClaudeToolChoice::Auto) | None => {}
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
                return ClaudeResponder::Error(
                    ClaudeErrorResponse {
                        response_type: "error",
                        error: ClaudeErrorBody {
                            error_type: "invalid_request_error".to_string(),
                            message: format!(
                                "tool_choice requires tool '{}' but it was not provided",
                                name
                            ),
                        },
                    },
                    StatusCode::UNPROCESSABLE_ENTITY,
                );
            }
        }
    }

    if tool_choice_required && resolved_tools.is_empty() {
        return ClaudeResponder::Error(
            ClaudeErrorResponse {
                response_type: "error",
                error: ClaudeErrorBody {
                    error_type: "invalid_request_error".to_string(),
                    message: "tool_choice requires at least one tool but none were provided"
                        .to_string(),
                },
            },
            StatusCode::UNPROCESSABLE_ENTITY,
        );
    }
    let use_claude_thinking_blocks = anthropic_thinking_enabled;

    let tool_schemas = Arc::new(build_tool_schema_map(&resolved_tools));
    params.mcp_mode = if !resolved_tools.is_empty() {
        Some(true)
    } else {
        None
    };
    let _tool_choice = tool_choice_to_openai(&request.tool_choice);

    let (model_type, tool_config, engine_config) = {
        let e = data.engine.read();
        (
            e.model_type.clone(),
            e.tool_config.clone(),
            e.econfig.clone(),
        )
    };
    let parser_model_id =
        super::resolve_engine_model_id(&engine_config).unwrap_or_else(|| model_id.clone());
    let enforce_parser = engine_config.enforce_parser.clone();

    let img_cfg = {
        let e = data.engine.read();
        e.img_cfg.clone()
    };

    let (messages, image_data) = match build_messages_and_images(&chat_messages, img_cfg.as_ref()) {
        Ok(output) => output,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "invalid_request_error".to_string(),
                        message: format!("Message processing failed: {err:?}"),
                    },
                },
                StatusCode::UNPROCESSABLE_ENTITY,
            );
        }
    };

    if use_stream {
        let preprocessed = {
            let e = data.engine.read();
            match e.preprocess(
                std::slice::from_ref(&params),
                std::slice::from_ref(&messages),
                &resolved_tools,
                false,
            ) {
                Ok(mut p) => p.pop().expect("preprocess returned 0 items for 1 input"),
                Err(err) => {
                    return ClaudeResponder::Error(
                        ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "invalid_request_error".to_string(),
                                message: format!("Stream preprocess failed: {err:?}"),
                            },
                        },
                        StatusCode::UNPROCESSABLE_ENTITY,
                    );
                }
            }
        };
        let (seq_id, prompt_length, prefilled_reasoning_end, stream) = {
            let mut e = data.engine.write();
            match e.generate_stream(preprocessed, image_data, &logger) {
                Ok((seq_id, prompt_length, prefilled_reasoning_end, stream)) => {
                    (seq_id, prompt_length, prefilled_reasoning_end, stream)
                }
                Err(err) => {
                    return ClaudeResponder::Error(
                        ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "invalid_request_error".to_string(),
                                message: format!("Stream generation failed: {err:?}"),
                            },
                        },
                        StatusCode::UNPROCESSABLE_ENTITY,
                    );
                }
            }
        };

        let buffer_size = env::var("CLAUDE_SSE_BUFFER")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(256);
        let (response_tx, client_rx) = flume::bounded(buffer_size);
        let engine_clone = data.engine.clone();
        let stream_model_id = model_id.clone();
        let stream_parser_model_id = parser_model_id.clone();
        let stream_model_type = model_type.clone();
        let stream_tool_config = tool_config.clone();
        let stream_tool_schemas = tool_schemas.clone();
        let forced_tool_name = forced_tool_name.clone();
        let stream_tools = resolved_tools.clone();
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let stream_logger = logger.clone();

        task::spawn(async move {
            struct StreamGuard {
                done_tx: tokio::sync::watch::Sender<bool>,
            }

            impl Drop for StreamGuard {
                fn drop(&mut self) {
                    let _ = self.done_tx.send(true);
                }
            }

            let (done_tx, mut done_rx) = tokio::sync::watch::channel(false);
            let _guard = StreamGuard {
                done_tx: done_tx.clone(),
            };

            let keep_alive_interval = Duration::from_millis(
                env::var("KEEP_ALIVE_INTERVAL")
                    .map(|val| val.parse::<u64>().unwrap_or(1000))
                    .unwrap_or(1000),
            );

            let keepalive_tx = response_tx.clone();
            let keepalive_engine = engine_clone.clone();
            tokio::spawn(async move {
                let mut ticker = time::interval(keep_alive_interval);
                loop {
                    tokio::select! {
                        _ = ticker.tick() => {
                            if let Err(err) = keepalive_tx.try_send(
                                ClaudeStreamItem::Event(Event::default().comment("keep-alive"))
                            ) {
                                match err {
                                    TrySendError::Full(_) => {
                                        crate::log_warn!(
                                            "[Seq {}] SSE buffer full during keepalive",
                                            seq_id
                                        );
                                    }
                                    TrySendError::Disconnected(_) => {
                                        crate::log_warn!(
                                            "[Seq {}] SSE client disconnected during keepalive",
                                            seq_id
                                        );
                                    }
                                }
                                let mut e = keepalive_engine.write();
                                e.cancel(seq_id);
                                break;
                            }
                        }
                        _ = done_rx.changed() => {
                            if *done_rx.borrow() {
                                break;
                            }
                        }
                    }
                }
            });

            let message_id = format!("msg_{}", Uuid::new_v4().simple());
            let stream_ctx = ClaudeStreamingContext::new(seq_id, response_tx.clone());
            let mut total_decoded_tokens = 0usize;
            let mut stream_finished = false;
            let idle_timeout = Duration::from_millis(
                env::var("CLAUDE_STREAM_IDLE_TIMEOUT_MS")
                    .map(|val| val.parse::<u64>().unwrap_or(300000))
                    .unwrap_or(300000),
            );
            let idle_sleep = time::sleep(idle_timeout);
            tokio::pin!(idle_sleep);
            let mut stream_started = false;

            let message_start = ClaudeMessageStartEvent {
                event_type: "message_start",
                message: ClaudeMessageResponse {
                    id: message_id.clone(),
                    response_type: "message",
                    role: "assistant",
                    content: Vec::new(),
                    model: stream_model_id.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: ClaudeUsage {
                        input_tokens: prompt_length,
                        output_tokens: 0,
                    },
                },
            };

            if let Err(err) = stream_ctx.send_json_event("message_start", &message_start) {
                crate::log_warn!("[Seq {}] Failed to send message_start: {:?}", seq_id, err);
                let mut e = engine_clone.write();
                e.cancel(seq_id);
                return;
            }

            let mut text_block_started = false;
            let text_block_index = 0usize;
            let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
            let mut suppressed_tool_markup: String = String::new();
            let mut buffering_since: Option<Instant> = None;
            let mut buffering_cancel_requested = false;
            let mut buffering_warned = false;
            let mut tool_parser = StreamToolParser::new_with_config(
                &stream_model_type,
                stream_parser_model_id.clone(),
                stream_tool_config,
                stream_tools.clone(),
                enforce_parser.clone(),
            );
            tool_parser.set_initial_reasoning_end_marker(prefilled_reasoning_end.clone());
            let should_parse_tools = !stream_tools.is_empty();
            let use_claude_thinking_stream = anthropic_thinking_enabled;
            let mut thinking_stream =
                use_claude_thinking_stream.then(ClaudeThinkingStreamEmitter::new);
            if use_claude_thinking_stream && should_parse_tools {
                tool_parser.set_detect_tools_in_reasoning(true);
            }

            let mut current_stream = stream;
            'stream: loop {
                let item = tokio::select! {
                    item = current_stream.recv() => item,
                    _ = &mut idle_sleep => {
                        if stream_started {
                            crate::log_warn!(
                                "[Seq {}] Stream idle timeout reached, cancelling request",
                                seq_id
                            );
                            let mut e = engine_clone.write();
                            e.cancel(seq_id);
                            break;
                        }
                        idle_sleep.as_mut().reset(time::Instant::now() + idle_timeout);
                        continue;
                    }
                };

                let item = match item {
                    Some(item) => item,
                    None => break,
                };

                stream_started = true;
                idle_sleep
                    .as_mut()
                    .reset(time::Instant::now() + idle_timeout);

                match item {
                    StreamItem::Token(token, token_id) => {
                        total_decoded_tokens += 1;

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
                                            seq_id,
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
                                            seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    let send_result = if use_claude_thinking_stream {
                                        thinking_stream
                                            .as_mut()
                                            .expect("thinking stream emitter")
                                            .push_chunk(
                                                &stream_ctx,
                                                stream_logger.as_ref().map(|logger| &**logger),
                                                &text,
                                            )
                                    } else {
                                        if let Some(ref l) = stream_logger {
                                            l.log_stream_token(&text);
                                        }
                                        send_text_with_start(
                                            &stream_ctx,
                                            &mut text_block_started,
                                            text_block_index,
                                            &text,
                                        )
                                    };
                                    if let Err(err) = send_result {
                                        let text_status = thinking_stream
                                            .as_ref()
                                            .and_then(|emitter| emitter.open_text_block_index());
                                        handle_stream_send_error(
                                            err,
                                            seq_id,
                                            &response_tx,
                                            text_status.is_some() || text_block_started,
                                            text_status.unwrap_or(text_block_index),
                                            total_decoded_tokens,
                                            true,
                                        )
                                        .await;
                                        let mut e = engine_clone.write();
                                        e.cancel(seq_id);
                                        stream_finished = true;
                                        break 'stream;
                                    }
                                }
                                StreamResult::Buffering => {
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
                                            seq_id
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
                                            seq_id,
                                            tool_buffer_timeout
                                        );
                                        let mut e = engine_clone.write();
                                        e.cancel(seq_id);
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
                                            seq_id,
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
                                            seq_id,
                                            text.len()
                                        );
                                        continue;
                                    }
                                    let safe_text =
                                        tool_parser.sanitize_tool_markup_for_display(&text);
                                    if !use_claude_thinking_stream && safe_text != text {
                                        crate::log_warn!(
                                            "[Seq {}] Sanitized leaked tool markup in flushed text",
                                            seq_id
                                        );
                                    }
                                    let send_result = if use_claude_thinking_stream {
                                        thinking_stream
                                            .as_mut()
                                            .expect("thinking stream emitter")
                                            .push_chunk(
                                                &stream_ctx,
                                                stream_logger.as_ref().map(|logger| &**logger),
                                                &safe_text,
                                            )
                                    } else {
                                        if let Some(ref l) = stream_logger {
                                            l.log_stream_token(&safe_text);
                                        }
                                        send_text_with_start(
                                            &stream_ctx,
                                            &mut text_block_started,
                                            text_block_index,
                                            &safe_text,
                                        )
                                    };
                                    if let Err(err) = send_result {
                                        let text_status = thinking_stream
                                            .as_ref()
                                            .and_then(|emitter| emitter.open_text_block_index());
                                        handle_stream_send_error(
                                            err,
                                            seq_id,
                                            &response_tx,
                                            text_status.is_some() || text_block_started,
                                            text_status.unwrap_or(text_block_index),
                                            total_decoded_tokens,
                                            true,
                                        )
                                        .await;
                                        let mut e = engine_clone.write();
                                        e.cancel(seq_id);
                                        stream_finished = true;
                                        break 'stream;
                                    }
                                }
                                StreamResult::ToolCalls(calls) => {
                                    buffering_since = None;
                                    buffering_cancel_requested = false;
                                    buffering_warned = false;
                                    pending_tool_calls.extend(calls);
                                }
                            }
                        } else if !token.is_empty() {
                            let send_result = if use_claude_thinking_stream {
                                thinking_stream
                                    .as_mut()
                                    .expect("thinking stream emitter")
                                    .push_chunk(
                                        &stream_ctx,
                                        stream_logger.as_ref().map(|logger| &**logger),
                                        &token,
                                    )
                            } else {
                                if let Some(ref l) = stream_logger {
                                    l.log_stream_token(&token);
                                }
                                send_text_with_start(
                                    &stream_ctx,
                                    &mut text_block_started,
                                    text_block_index,
                                    &token,
                                )
                            };
                            if let Err(err) = send_result {
                                let text_status = thinking_stream
                                    .as_ref()
                                    .and_then(|emitter| emitter.open_text_block_index());
                                handle_stream_send_error(
                                    err,
                                    seq_id,
                                    &response_tx,
                                    text_status.is_some() || text_block_started,
                                    text_status.unwrap_or(text_block_index),
                                    total_decoded_tokens,
                                    true,
                                )
                                .await;
                                let mut e = engine_clone.write();
                                e.cancel(seq_id);
                                stream_finished = true;
                                break 'stream;
                            }
                        }
                    }
                    StreamItem::Done((
                        prompt_start_time,
                        decode_start_time,
                        decode_finish_time,
                        final_decoded_length,
                        stop_sequence,
                    )) => {
                        total_decoded_tokens = final_decoded_length;

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
                                                    seq_id,
                                                    buffer.len()
                                                );
                                            } else if !pending_tool_calls.is_empty() {
                                                crate::log_warn!(
                                                    "[Seq {}] Dropping {} buffered chars because tool calls were already parsed",
                                                    seq_id,
                                                    buffer.len()
                                                );
                                            } else {
                                                let safe_buffer = tool_parser
                                                    .sanitize_tool_markup_for_display(&buffer);
                                                if !use_claude_thinking_stream
                                                    && safe_buffer != buffer
                                                {
                                                    crate::log_warn!(
                                                        "[Seq {}] Sanitized leaked tool markup in partial buffer",
                                                        seq_id
                                                    );
                                                }
                                                if use_claude_thinking_stream {
                                                    let _ = thinking_stream
                                                        .as_mut()
                                                        .expect("thinking stream emitter")
                                                        .push_chunk(
                                                            &stream_ctx,
                                                            stream_logger
                                                                .as_ref()
                                                                .map(|logger| &**logger),
                                                            &safe_buffer,
                                                        );
                                                } else {
                                                    let _ = send_text_with_start(
                                                        &stream_ctx,
                                                        &mut text_block_started,
                                                        text_block_index,
                                                        &safe_buffer,
                                                    );
                                                }
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
                                        seq_id,
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
                                                seq_id,
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
                                    seq_id,
                                    safe_suppressed.len()
                                );
                                if use_claude_thinking_stream {
                                    let _ = thinking_stream
                                        .as_mut()
                                        .expect("thinking stream emitter")
                                        .push_chunk(
                                            &stream_ctx,
                                            stream_logger.as_ref().map(|logger| &**logger),
                                            &safe_suppressed,
                                        );
                                } else {
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&safe_suppressed);
                                    }
                                    let _ = send_text_with_start(
                                        &stream_ctx,
                                        &mut text_block_started,
                                        text_block_index,
                                        &safe_suppressed,
                                    );
                                }
                            } else if !pending_tool_calls.is_empty()
                                && !suppressed_tool_markup.is_empty()
                            {
                                crate::log_warn!(
                                    "[Seq {}] Dropping {} suppressed tool-markup chars because tool calls were recovered",
                                    seq_id,
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
                                seq_id,
                                dropped
                            );
                        }

                        let (tool_calls, has_tool_calls, invalid_feedback) = if pending_tool_calls
                            .is_empty()
                        {
                            (Vec::new(), false, None)
                        } else {
                            let (validated_calls, invalid) = filter_tool_calls(
                                &pending_tool_calls,
                                stream_tool_schemas.as_ref(),
                            );
                            if !invalid.is_empty() {
                                crate::log_error!(
                                    "[Seq {}] Found {} invalid tool call(s): {:?}",
                                    seq_id,
                                    invalid.len(),
                                    invalid
                                );
                            }

                            if !invalid.is_empty() {
                                log_tool_calls("Invalid", seq_id, &invalid);
                                if let Some(ref l) = stream_logger {
                                    l.log_tool_calls("Invalid", &invalid);
                                }
                            }
                            let invalid_feedback = build_invalid_tool_call_feedback(
                                &invalid,
                                stream_tool_schemas.as_ref(),
                                forced_tool_name.as_deref(),
                            );

                            let (final_tool_calls, invalid_feedback) =
                                if !invalid.is_empty() && !strict_tool_call_validation_enabled() {
                                    crate::log_error!(
                                        "Invalid tool call feedback {:?}",
                                        invalid_feedback
                                    );
                                    (pending_tool_calls, None)
                                } else {
                                    (validated_calls, invalid_feedback)
                                };

                            if final_tool_calls.is_empty() {
                                (Vec::new(), false, invalid_feedback)
                            } else {
                                log_tool_calls("Valid", seq_id, &final_tool_calls);
                                if let Some(ref l) = stream_logger {
                                    l.log_tool_calls("Valid", &final_tool_calls);
                                }
                                (final_tool_calls, true, invalid_feedback)
                            }
                        };

                        if tool_choice_required && !has_tool_calls {
                            if let Some(ref name) = forced_tool_name {
                                crate::log_warn!(
                                    "[Seq {}] Tool choice required '{}' but no tool calls were produced",
                                    seq_id,
                                    name
                                );
                            } else {
                                crate::log_warn!(
                                    "[Seq {}] Tool choice required but no tool calls were produced",
                                    seq_id
                                );
                            }
                        }

                        let stop_reason = stop_reason_from_decoding(
                            has_tool_calls,
                            total_decoded_tokens,
                            max_tokens,
                            stop_sequence.as_deref(),
                        );

                        if !has_tool_calls {
                            if let Some(feedback) = invalid_feedback.as_deref() {
                                let send_result = if use_claude_thinking_stream {
                                    thinking_stream
                                        .as_mut()
                                        .expect("thinking stream emitter")
                                        .push_chunk(
                                            &stream_ctx,
                                            stream_logger.as_ref().map(|logger| &**logger),
                                            feedback,
                                        )
                                } else {
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(feedback);
                                    }
                                    send_text_with_start(
                                        &stream_ctx,
                                        &mut text_block_started,
                                        text_block_index,
                                        feedback,
                                    )
                                };
                                if let Err(err) = send_result {
                                    let text_status = thinking_stream
                                        .as_ref()
                                        .and_then(|emitter| emitter.open_text_block_index());
                                    handle_stream_send_error(
                                        err,
                                        seq_id,
                                        &response_tx,
                                        text_status.is_some() || text_block_started,
                                        text_status.unwrap_or(text_block_index),
                                        total_decoded_tokens,
                                        true,
                                    )
                                    .await;
                                    let mut e = engine_clone.write();
                                    e.cancel(seq_id);
                                    stream_finished = true;
                                    break 'stream;
                                }
                            }
                        }

                        let next_block_index = if use_claude_thinking_stream {
                            match thinking_stream
                                .as_mut()
                                .expect("thinking stream emitter")
                                .finish(&stream_ctx, stream_logger.as_ref().map(|logger| &**logger))
                            {
                                Ok(index) => index,
                                Err(err) => {
                                    let text_status = thinking_stream
                                        .as_ref()
                                        .and_then(|emitter| emitter.open_text_block_index());
                                    handle_stream_send_error(
                                        err,
                                        seq_id,
                                        &response_tx,
                                        text_status.is_some(),
                                        text_status.unwrap_or(text_block_index),
                                        total_decoded_tokens,
                                        true,
                                    )
                                    .await;
                                    let mut e = engine_clone.write();
                                    e.cancel(seq_id);
                                    stream_finished = true;
                                    break 'stream;
                                }
                            }
                        } else {
                            let mut next_block_index = 0usize;
                            if text_block_started {
                                let stop_event = ClaudeContentBlockStopEvent {
                                    event_type: "content_block_stop",
                                    index: text_block_index,
                                };
                                if let Err(err) =
                                    stream_ctx.send_json_event("content_block_stop", &stop_event)
                                {
                                    handle_stream_send_error(
                                        err,
                                        seq_id,
                                        &response_tx,
                                        text_block_started,
                                        text_block_index,
                                        total_decoded_tokens,
                                        true,
                                    )
                                    .await;
                                    let mut e = engine_clone.write();
                                    e.cancel(seq_id);
                                    stream_finished = true;
                                    break 'stream;
                                }
                                text_block_started = false;
                                next_block_index = text_block_index + 1;
                            }
                            next_block_index
                        };

                        if next_block_index == 0 && !has_tool_calls {
                            let start_block = ClaudeContentBlockStartEvent {
                                event_type: "content_block_start",
                                index: 0,
                                content_block: ClaudeContentBlockOut::Text {
                                    text: String::new(),
                                },
                            };
                            let _ = stream_ctx.send_json_event("content_block_start", &start_block);
                            let stop_event = ClaudeContentBlockStopEvent {
                                event_type: "content_block_stop",
                                index: 0,
                            };
                            let _ = stream_ctx.send_json_event("content_block_stop", &stop_event);
                        }

                        if has_tool_calls {
                            let tool_blocks = tool_calls_to_blocks(&tool_calls);
                            crate::log_info!("[Seq {}] Tool use blocks: {:?}", seq_id, tool_blocks);
                            for (idx, call) in tool_calls.iter().enumerate() {
                                if let Err(err) =
                                    send_tool_use_block(&stream_ctx, next_block_index + idx, call)
                                {
                                    handle_stream_send_error(
                                        err,
                                        seq_id,
                                        &response_tx,
                                        text_block_started,
                                        text_block_index,
                                        total_decoded_tokens,
                                        true,
                                    )
                                    .await;
                                    let mut e = engine_clone.write();
                                    e.cancel(seq_id);
                                    stream_finished = true;
                                    break 'stream;
                                }
                            }
                        }

                        log_performance_metrics(
                            seq_id,
                            prompt_length,
                            total_decoded_tokens,
                            prompt_start_time,
                            decode_start_time,
                            decode_finish_time,
                        );

                        let message_delta = ClaudeMessageDeltaEvent {
                            event_type: "message_delta",
                            delta: ClaudeMessageDelta {
                                stop_reason: Some(stop_reason),
                                stop_sequence: stop_sequence.clone(),
                            },
                            usage: ClaudeUsageDelta {
                                output_tokens: total_decoded_tokens,
                            },
                        };
                        let message_stop = ClaudeMessageStopEvent {
                            event_type: "message_stop",
                        };
                        if let Err(err) =
                            stream_ctx.send_json_event("message_delta", &message_delta)
                        {
                            handle_stream_send_error(
                                err,
                                seq_id,
                                &response_tx,
                                text_block_started,
                                text_block_index,
                                total_decoded_tokens,
                                true,
                            )
                            .await;
                            let mut e = engine_clone.write();
                            e.cancel(seq_id);
                            stream_finished = true;
                            break 'stream;
                        }
                        if let Err(err) = stream_ctx.send_json_event("message_stop", &message_stop)
                        {
                            handle_stream_send_error(
                                err,
                                seq_id,
                                &response_tx,
                                text_block_started,
                                text_block_index,
                                total_decoded_tokens,
                                false,
                            )
                            .await;
                            let mut e = engine_clone.write();
                            e.cancel(seq_id);
                            stream_finished = true;
                            break 'stream;
                        }
                        let _ = send_done_with_timeout(
                            seq_id,
                            &response_tx,
                            Duration::from_millis(FINAL_EVENT_TIMEOUT_MS),
                        )
                        .await;
                        stream_finished = true;
                        break 'stream;
                    }
                    StreamItem::Error(err) => {
                        let error = ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "server_error".to_string(),
                                message: err,
                            },
                        };
                        let _ = send_json_event_with_timeout(
                            seq_id,
                            &response_tx,
                            "error",
                            &error,
                            Duration::from_millis(FINAL_EVENT_TIMEOUT_MS),
                        )
                        .await;
                        let _ = send_done_with_timeout(
                            seq_id,
                            &response_tx,
                            Duration::from_millis(FINAL_EVENT_TIMEOUT_MS),
                        )
                        .await;
                        stream_finished = true;
                        break;
                    }
                    _ => {}
                }
            }

            if !stream_finished {
                let message_stop = ClaudeMessageStopEvent {
                    event_type: "message_stop",
                };
                let _ = send_json_event_with_timeout(
                    seq_id,
                    &response_tx,
                    "message_stop",
                    &message_stop,
                    Duration::from_millis(FINAL_EVENT_TIMEOUT_MS),
                )
                .await;
                let _ = send_done_with_timeout(
                    seq_id,
                    &response_tx,
                    Duration::from_millis(FINAL_EVENT_TIMEOUT_MS),
                )
                .await;
            }
        });

        ClaudeResponder::Streamer(
            Sse::new(ClaudeStreamer {
                rx: client_rx,
                status: ClaudeStreamingStatus::Uninitialized,
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive"),
            ),
        )
    } else {
        let tokenizer = {
            let e = data.engine.read();
            Arc::new(e.tokenizer.clone())
        };

        let preprocessed = {
            let e = data.engine.read();
            match e.preprocess(
                std::slice::from_ref(&params),
                std::slice::from_ref(&messages),
                &resolved_tools,
                false,
            ) {
                Ok(p) => p,
                Err(err) => {
                    return ClaudeResponder::Error(
                        ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "server_error".to_string(),
                                message: format!("Preprocess failed: {err:?}"),
                            },
                        },
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
            }
        };
        let receivers = {
            let mut e = data.engine.write();
            match e.generate_sync(preprocessed, image_data, &logger) {
                Ok(receivers) => receivers,
                Err(err) => {
                    return ClaudeResponder::Error(
                        ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "server_error".to_string(),
                                message: format!("Completion generation failed: {err:?}"),
                            },
                        },
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
            }
        };
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let results =
            match LLMEngine::collect_sync_results(receivers, tokenizer, logger.clone()).await {
                Ok(results) => results,
                Err(err) => {
                    return ClaudeResponder::Error(
                        ClaudeErrorResponse {
                            response_type: "error",
                            error: ClaudeErrorBody {
                                error_type: "server_error".to_string(),
                                message: format!("Failed to collect results: {err:?}"),
                            },
                        },
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
            };

        let output = match results.into_iter().next() {
            Some(output) => output,
            None => {
                return ClaudeResponder::Error(
                    ClaudeErrorResponse {
                        response_type: "error",
                        error: ClaudeErrorBody {
                            error_type: "server_error".to_string(),
                            message: "No output returned".to_string(),
                        },
                    },
                    StatusCode::INTERNAL_SERVER_ERROR,
                );
            }
        };

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
        let dropped = retain_tool_calls_forced_name(&mut parsed_calls, forced_tool_name.as_deref());
        if dropped > 0 {
            crate::log_warn!(
                "[Seq {}] Dropped {} tool call(s) that did not match forced tool_choice",
                output.seq_id,
                dropped
            );
        }
        let (validated_calls, invalid_calls) =
            filter_tool_calls(&parsed_calls, tool_schemas.as_ref());
        if !invalid_calls.is_empty() {
            crate::log_error!("Found {} invalid tool call(s)", invalid_calls.len());
        }
        let invalid_feedback = build_invalid_tool_call_feedback(
            &invalid_calls,
            tool_schemas.as_ref(),
            forced_tool_name.as_deref(),
        );

        let valid_calls = validated_calls;

        if !valid_calls.is_empty() {
            log_tool_calls("Valid", output.seq_id, &valid_calls);
            if let Some(ref l) = logger {
                l.log_tool_calls("Valid", &valid_calls);
            }
        }
        let has_tool_calls = !valid_calls.is_empty();
        if tool_choice_required && !has_tool_calls {
            if let Some(ref name) = forced_tool_name {
                crate::log_warn!(
                    "[Seq {}] Tool choice required '{}' but no tool calls were produced",
                    output.seq_id,
                    name
                );
            } else {
                crate::log_warn!(
                    "[Seq {}] Tool choice required but no tool calls were produced",
                    output.seq_id
                );
            }
        }
        let content = if has_tool_calls {
            if use_claude_thinking_blocks {
                let parsed = parse_claude_assistant_output(&output.decode_output);
                let mut blocks = parsed
                    .thinking_blocks
                    .into_iter()
                    .map(|block| ClaudeContentBlockOut::Thinking {
                        thinking: block.thinking,
                        signature: Some(block.signature),
                    })
                    .collect::<Vec<_>>();
                if !parsed.text.is_empty() {
                    let safe_text = tool_parser.sanitize_tool_markup_for_display(&parsed.text);
                    if !safe_text.is_empty() {
                        blocks.push(ClaudeContentBlockOut::Text { text: safe_text });
                    }
                }
                blocks.extend(tool_calls_to_blocks(&valid_calls));
                blocks
            } else {
                let safe_text = tool_parser.sanitize_tool_markup_for_display(&output.decode_output);
                let mut blocks = Vec::new();
                if !safe_text.is_empty() {
                    blocks.push(ClaudeContentBlockOut::Text { text: safe_text });
                }
                blocks.extend(tool_calls_to_blocks(&valid_calls));
                blocks
            }
        } else {
            let safe_text = if let Some(feedback) = invalid_feedback {
                feedback
            } else if tool_parser.contains_tool_markup(&output.decode_output) {
                tool_parser.sanitize_tool_markup_for_display(&output.decode_output)
            } else {
                output.decode_output.clone()
            };
            if use_claude_thinking_blocks {
                let parsed = parse_claude_assistant_output(&safe_text);
                let mut blocks = parsed
                    .thinking_blocks
                    .into_iter()
                    .map(|block| ClaudeContentBlockOut::Thinking {
                        thinking: block.thinking,
                        signature: Some(block.signature),
                    })
                    .collect::<Vec<_>>();
                if !parsed.text.is_empty() {
                    blocks.push(ClaudeContentBlockOut::Text { text: parsed.text });
                }
                if blocks.is_empty() {
                    vec![ClaudeContentBlockOut::Text {
                        text: String::new(),
                    }]
                } else {
                    blocks
                }
            } else {
                vec![ClaudeContentBlockOut::Text { text: safe_text }]
            }
        };

        let response = ClaudeMessageResponse {
            id: format!("msg_{}", Uuid::new_v4().simple()),
            response_type: "message",
            role: "assistant",
            content,
            model: model_id,
            stop_reason: Some(stop_reason_from_decoding(
                has_tool_calls,
                output.decoded_length,
                max_tokens,
                output.stop_sequence.as_deref(),
            )),
            stop_sequence: output.stop_sequence.clone(),
            usage: ClaudeUsage {
                input_tokens: output.prompt_length,
                output_tokens: output.decoded_length,
            },
        };

        log_performance_metrics(
            output.seq_id,
            output.prompt_length,
            output.decoded_length,
            output.prompt_start_time,
            output.decode_start_time,
            output.decode_finish_time,
        );

        if let Some(ref l) = logger {
            l.log_raw_response(&response);
        }
        ClaudeResponder::Message(response)
    }
}

pub async fn count_tokens(
    State(data): State<Arc<ServerData>>,
    request: Json<ClaudeTokenCountRequest>,
) -> ClaudeResponder {
    let request = request.0;
    let mut message_request = ClaudeMessageRequest {
        model: request.model.clone(),
        messages: request.messages.clone(),
        system: request.system.clone(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        top_k: None,
        stream: None,
        stop_sequences: None,
        tools: request.tools.clone(),
        tool_choice: None,
        thinking: None,
        extra: request.extra.clone(),
    };
    let normalized = normalize_claude_request_for_prefix_cache(&mut message_request);
    if normalized > 0 {
        crate::log_info!(
            "Normalized {} volatile Claude prompt token(s) for prefix cache stability (count_tokens).",
            normalized
        );
    }

    let chat_messages = match build_chat_messages(&message_request) {
        Ok(messages) => messages,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "invalid_request_error".to_string(),
                        message: err,
                    },
                },
                StatusCode::UNPROCESSABLE_ENTITY,
            );
        }
    };

    let img_cfg = {
        let e = data.engine.read();
        e.img_cfg.clone()
    };
    let (messages, _) = match build_messages_and_images(&chat_messages, img_cfg.as_ref()) {
        Ok(output) => output,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "invalid_request_error".to_string(),
                        message: format!("Message processing failed: {err:?}"),
                    },
                },
                StatusCode::UNPROCESSABLE_ENTITY,
            );
        }
    };

    let engine = data.engine.read();
    let mut template = engine.get_chat_template();
    template.set_messages(&messages);
    let prompt = match template.apply_chat_template(&Vec::new(), false) {
        Ok(prompt) => prompt,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "server_error".to_string(),
                        message: format!("Failed to apply chat template: {err:?}"),
                    },
                },
                StatusCode::INTERNAL_SERVER_ERROR,
            );
        }
    };

    let tokenizer = engine.tokenizer.clone();
    let encoding = match tokenizer.encode(prompt.as_str(), true) {
        Ok(encoding) => encoding,
        Err(err) => {
            return ClaudeResponder::Error(
                ClaudeErrorResponse {
                    response_type: "error",
                    error: ClaudeErrorBody {
                        error_type: "server_error".to_string(),
                        message: format!("Tokenization failed: {err:?}"),
                    },
                },
                StatusCode::INTERNAL_SERVER_ERROR,
            );
        }
    };

    ClaudeResponder::TokenCount(ClaudeTokenCountResponse {
        input_tokens: encoding.get_ids().len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::schema::SchemaBuilder;
    use serde_json::json;

    #[test]
    fn converts_text_messages() {
        let request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: Some(ClaudeSystem::Text("system".to_string())),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            extra: HashMap::new(),
        };

        let messages = build_chat_messages(&request).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn converts_tool_use_and_result_blocks() {
        let request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![
                ClaudeMessage {
                    role: "assistant".to_string(),
                    content: ClaudeContent::Blocks(vec![
                        ClaudeContentBlock::Text {
                            text: "run tool".to_string(),
                        },
                        ClaudeContentBlock::ToolUse {
                            id: "call_1".to_string(),
                            name: "get_weather".to_string(),
                            input: json!({"city": "tokyo"}),
                        },
                    ]),
                },
                ClaudeMessage {
                    role: "user".to_string(),
                    content: ClaudeContent::Blocks(vec![ClaudeContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: ClaudeToolResultContent::Text("ok".to_string()),
                        is_error: None,
                    }]),
                },
            ],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            extra: HashMap::new(),
        };

        let converted = build_chat_messages(&request).unwrap();
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "assistant");
        assert_eq!(converted[1].role, "assistant");
        assert_eq!(converted[2].role, "tool");
        let tool_calls = converted[1]
            .tool_calls
            .clone()
            .expect("assistant tool_calls expected");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn rejects_assistant_tool_result_blocks() {
        let request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![ClaudeMessage {
                role: "assistant".to_string(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: ClaudeToolResultContent::Text("bad".to_string()),
                    is_error: None,
                }]),
            }],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            extra: HashMap::new(),
        };

        let err = build_chat_messages(&request).unwrap_err();
        assert!(err.contains("tool_result blocks must be in user messages"));
    }

    #[test]
    fn rejects_non_adjacent_tool_result_response() {
        let request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![
                ClaudeMessage {
                    role: "assistant".to_string(),
                    content: ClaudeContent::Blocks(vec![ClaudeContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        input: json!({"city": "tokyo"}),
                    }]),
                },
                ClaudeMessage {
                    role: "user".to_string(),
                    content: ClaudeContent::Text("plain text first".to_string()),
                },
            ],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            extra: HashMap::new(),
        };

        let err = build_chat_messages(&request).unwrap_err();
        assert!(err.contains("must provide tool_result blocks"));
    }

    #[test]
    fn preserves_empty_success_tool_result_as_ack() {
        let blocks = vec![ClaudeContentBlock::ToolResult {
            tool_use_id: "call_1".to_string(),
            content: ClaudeToolResultContent::Text(String::new()),
            is_error: Some(false),
        }];

        let message = ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(blocks),
        };

        let converted = convert_claude_message(&message).unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_1"));
        let text = match converted[0].content.as_ref() {
            Some(MessageContentType::PureText(text)) => text.clone(),
            _ => String::new(),
        };
        assert_eq!(text, "Tool executed successfully with no textual output.");
    }

    #[test]
    fn wraps_tool_result_when_is_error_true() {
        let blocks = vec![ClaudeContentBlock::ToolResult {
            tool_use_id: "call_1".to_string(),
            content: ClaudeToolResultContent::Text("boom".to_string()),
            is_error: Some(true),
        }];

        let message = ClaudeMessage {
            role: "user".to_string(),
            content: ClaudeContent::Blocks(blocks),
        };

        let converted = convert_claude_message(&message).unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        let text = match converted[0].content.as_ref() {
            Some(MessageContentType::PureText(text)) => text.clone(),
            _ => String::new(),
        };
        assert_eq!(text, "<tool_use_error>boom</tool_use_error>");
    }

    #[test]
    fn accepts_thinking_config() {
        let request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text("hello".to_string()),
            }],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: Some(ClaudeThinking::Config(ClaudeThinkingConfig {
                mode: "enabled".to_string(),
                budget_tokens: Some(128),
            })),
            extra: HashMap::new(),
        };

        let enabled = thinking_to_bool(&request.thinking);
        assert_eq!(enabled, Some(true));
    }

    #[test]
    fn accepts_adaptive_thinking_config() {
        let thinking = Some(ClaudeThinking::Config(ClaudeThinkingConfig {
            mode: "adaptive".to_string(),
            budget_tokens: None,
        }));
        assert_eq!(thinking_to_bool(&thinking), Some(true));
    }

    #[test]
    fn converts_assistant_thinking_blocks_to_reasoning_content() {
        let signature = encode_synthetic_thinking_signature(" \n\n");
        let message = ClaudeMessage {
            role: "assistant".to_string(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::Thinking {
                    thinking: "\nplan\n".to_string(),
                    signature: Some(signature),
                },
                ClaudeContentBlock::Text {
                    text: "done".to_string(),
                },
            ]),
        };

        let converted = convert_claude_message(&message).unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(
            converted[0].reasoning_content.as_deref(),
            Some("plan"),
            "thinking block content should be set as reasoning_content"
        );
        let text = crate::server::extract_text_content(converted[0].content.as_ref().unwrap());
        assert_eq!(text, "done", "text block should be the regular content");
    }

    #[test]
    fn parses_assistant_output_into_thinking_blocks_and_text() {
        let parsed = parse_claude_assistant_output("<think>\nabc\n</think>\n\nResult text");
        assert_eq!(
            parsed.thinking_blocks,
            vec![ClaudeThinkingBlock {
                thinking: "\nabc\n".to_string(),
                signature: encode_synthetic_thinking_signature(" \n\n"),
            }]
        );
        assert_eq!(parsed.text, "Result text");
        let replay = replay_text_for_thinking_block(
            &parsed.thinking_blocks[0].thinking,
            Some(parsed.thinking_blocks[0].signature.as_str()),
        );
        assert_eq!(replay, " \nabc\n   ");
    }

    #[test]
    fn normalizes_newlines_in_empty_suffix_thinking_block() {
        let parsed = parse_claude_assistant_output("<think>\n\n</think>\n\n");
        assert!(parsed.thinking_blocks.is_empty());
        assert_eq!(parsed.text, "      ");
    }

    #[test]
    fn normalizes_prefilled_reasoning_close_suffix_to_placeholders() {
        let parsed = parse_claude_assistant_output("  </think>\n\n");
        assert!(parsed.thinking_blocks.is_empty());
        assert_eq!(parsed.text, "      ");
    }

    #[test]
    fn converts_tools_to_openai_format() {
        let tool = ClaudeTool {
            name: "lookup".to_string(),
            description: Some("Lookup data".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "q": { "type": "string" }
                },
                "required": ["q"]
            }),
        };

        let tools = claude_tools_to_tools(&[tool]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "lookup");
    }

    #[test]
    fn filters_invalid_tool_calls() {
        let schema = SchemaBuilder::object()
            .string_prop("path", "Path to list", true)
            .build();
        let tools = vec![crate::tools::function_tool("list_files", "List files")
            .parameters_schema(schema)
            .build()];
        let schemas = build_tool_schema_map(&tools);
        let valid_call = crate::tools::new_tool_call("call_1", "list_files", r#"{"path": "."}"#);
        let invalid_call = crate::tools::new_tool_call("call_2", "list_files", r#"{"dir": "."}"#);
        let (valid, invalid) = filter_tool_calls(&[valid_call, invalid_call], &schemas);

        assert_eq!(valid.len(), 1);
        assert_eq!(invalid.len(), 1);
        assert_eq!(valid[0].function.name, "list_files");
    }

    #[test]
    fn normalizes_claude_settings_uuid_for_prefix_cache_stability() {
        let mut request = ClaudeMessageRequest {
            model: "default".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Blocks(vec![
                    ClaudeContentBlock::Text {
                        text: "open /tmp/claude-settings-20fc829c-5ec4-43a6-93f9-62663f5517b1.json"
                            .to_string(),
                    },
                    ClaudeContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "Read".to_string(),
                        input: json!({
                            "file_path": "/tmp/claude-settings-90a7db64-23ee-477b-b8f3-e97eb1ba1d1f.json"
                        }),
                    },
                    ClaudeContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: ClaudeToolResultContent::Text(
                            "result from /tmp/claude-settings-d6d53ae3-d79a-4df0-adde-80d87aa3e471.json"
                                .to_string(),
                        ),
                        is_error: None,
                    },
                ]),
            }],
            system: Some(ClaudeSystem::Text(
                "x-anthropic-billing-header: cc_version=2.1.52.7f8; cc_entrypoint=cli; cch=abcde; sandbox deny /tmp/claude-settings-aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb.json"
                    .to_string(),
            )),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: Some(vec![ClaudeTool {
                name: "Bash".to_string(),
                description: Some(
                    "denyWithinAllow /tmp/claude-settings-ffffffff-ffff-ffff-ffff-ffffffffffff.json"
                        .to_string(),
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "restriction": {
                            "type": "string",
                            "description": "Use /tmp/claude-settings-12345678-1234-1234-1234-123456789abc.json"
                        }
                    }
                }),
            }]),
            tool_choice: None,
            thinking: None,
            extra: HashMap::new(),
        };

        let normalized = normalize_claude_request_for_prefix_cache(&mut request);
        assert_eq!(normalized, 7);

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains(STABLE_CLAUDE_SETTINGS_FILENAME));
        assert!(serialized.contains("cch=00000"));
        assert!(!serialized.contains("cch=abcde"));
        assert!(!serialized.contains("20fc829c-5ec4-43a6-93f9-62663f5517b1"));
        assert!(!serialized.contains("90a7db64-23ee-477b-b8f3-e97eb1ba1d1f"));
        assert!(!serialized.contains("d6d53ae3-d79a-4df0-adde-80d87aa3e471"));
        assert!(!serialized.contains("aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"));
        assert!(!serialized.contains("ffffffff-ffff-ffff-ffff-ffffffffffff"));
        assert!(!serialized.contains("12345678-1234-1234-1234-123456789abc"));
    }

    #[test]
    fn converts_thinking_block_with_tool_use() {
        let messages = convert_claude_message(&ClaudeMessage {
            role: "assistant".to_string(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::Thinking {
                    thinking: "I should use the tool".to_string(),
                    signature: None,
                },
                ClaudeContentBlock::Text {
                    text: "Let me check.".to_string(),
                },
                ClaudeContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "Read".to_string(),
                    input: json!({"file_path": "/tmp/test.txt"}),
                },
            ]),
        })
        .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "assistant");
        assert!(messages[0].content.is_some());
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        let calls = messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "Read");
    }

    #[test]
    fn converts_text_block_with_think_markers_and_tool_use() {
        let messages = convert_claude_message(&ClaudeMessage {
            role: "assistant".to_string(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::Text {
                    text: "<think>\nI should use the tool\n</think>\n\nLet me check.".to_string(),
                },
                ClaudeContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "Read".to_string(),
                    input: json!({"file_path": "/tmp/test.txt"}),
                },
            ]),
        })
        .unwrap();
        assert_eq!(messages.len(), 2);
        let text = crate::server::extract_text_content(messages[0].content.as_ref().unwrap());
        assert!(
            text.contains("<think>"),
            "text block should preserve <think> markers for chat template processing"
        );
    }

    #[test]
    fn parse_output_extracts_thinking_with_tool_markup() {
        let output = "<think>\nReasoning here\n</think>\n\nSome text\n<tool_call>\n<function=Read>\n<parameter=file_path>/tmp/test.txt</parameter>\n</function>";
        let parsed = parse_claude_assistant_output(output);
        assert_eq!(parsed.thinking_blocks.len(), 1);
        assert!(
            parsed.thinking_blocks[0]
                .thinking
                .contains("Reasoning here"),
            "thinking block should contain the reasoning text"
        );
        assert!(
            parsed.text.contains("<tool_call>"),
            "tool markup should remain in text for separate parsing"
        );
    }

    #[test]
    fn stop_reason_tool_use_when_has_tool_calls() {
        assert_eq!(stop_reason_from_decoding(true, 100, 1000, None), "tool_use");
    }

    #[test]
    fn stop_reason_end_turn_when_no_tool_calls() {
        assert_eq!(
            stop_reason_from_decoding(false, 100, 1000, None),
            "end_turn"
        );
    }

    #[test]
    fn stop_reason_max_tokens() {
        assert_eq!(
            stop_reason_from_decoding(false, 1000, 1000, None),
            "max_tokens"
        );
    }

    #[test]
    fn stop_reason_stop_sequence() {
        assert_eq!(
            stop_reason_from_decoding(false, 100, 1000, Some("\n")),
            "stop_sequence"
        );
    }

    #[test]
    fn replay_text_for_thinking_preserves_content() {
        let replay = replay_text_for_thinking_block("I need to think", None);
        assert!(
            replay.contains("I need to think"),
            "replay text should contain the thinking content"
        );
        assert!(
            !replay.contains("<think>"),
            "replay text should not contain <think> markers"
        );
    }

    #[test]
    fn tool_calls_to_blocks_produces_valid_tool_use() {
        let calls = vec![crate::tools::new_tool_call(
            "call_1".to_string(),
            "get_weather".to_string(),
            r#"{"city":"tokyo"}"#.to_string(),
        )];
        let blocks = tool_calls_to_blocks(&calls);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ClaudeContentBlockOut::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "tokyo");
            }
            _ => panic!("expected ToolUse block"),
        }
    }
}
