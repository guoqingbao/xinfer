// src/server/mod.rs
use clap::Parser;
use llguidance::api::TopLevelGrammar;
use serde::{Deserialize, Serialize};
pub mod claude_server;
pub mod logger;
pub mod parser;
pub mod server;
pub mod streaming;
use crate::core::engine::LLMEngine;
use crate::server::streaming::Streamer;
use crate::tools::schema::{schema_to_tools, ToolGrammarBuilder};
use crate::transfer::PdRole;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::guidance::{compose_grammars, GuidanceTokens, TopLevelGrammarExt};
use crate::utils::image::{
    compute_tokens_per_image, get_tensor_raw_data, load_image_from_base64, load_image_from_url,
    ImageData, ImageProcessConfig, ImageProcessTrait, IMAGE_PLACEHOLDER,
};
use crate::utils::reasoning::ReasoningEffort;
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use candle_core::{Result, Tensor};
use colored::*;
use local_ip_address::local_ip;
use parking_lot::RwLock;
use rustchatui::start_ui_server;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn deserialize_stop_sequences<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<StopSequences>::deserialize(deserializer)?;
    Ok(value.map(|stop| match stop {
        StopSequences::One(single) => vec![single],
        StopSequences::Many(many) => many,
    }))
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>,
    #[serde(
        default,
        alias = "stop_sequences",
        deserialize_with = "deserialize_stop_sequences"
    )]
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub session_id: Option<String>,
    /// Tools available for the model to call
    #[serde(default)]
    pub tools: Option<Vec<crate::tools::Tool>>,
    /// How the model should choose which tool to call
    #[serde(default)]
    pub tool_choice: Option<crate::tools::ToolChoice>,
    /// OpenAI-style response format for structured outputs
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Extra body for OpenAI-compatible clients (e.g. structured_outputs)
    #[serde(default)]
    pub extra_body: Option<ExtraBody>,
    /// Direct structured_outputs for convenience (parsed from extra_body if not present)
    #[serde(default, alias = "structured_outputs")]
    pub structured_outputs: Option<StructuredOutputs>,
    /// Legacy constraint field for llguidance (llg-new.diff pattern)
    /// Use constraint_type to specify grammar format: "regex", "lark", "json_schema"
    #[serde(alias = "grammar", default)]
    pub constraint: Option<String>,
    /// Type of constraint for legacy constraint field
    #[serde(default)]
    pub constraint_type: Option<String>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    /// Values: "none", "low", "medium", "high"
    #[serde(default, alias = "reasoning")]
    pub reasoning_effort: Option<String>,
}

pub fn resolve_engine_model_id(econfig: &EngineConfig) -> Option<String> {
    if let Some(model_id) = &econfig.model_id {
        if !model_id.trim().is_empty() {
            return Some(model_id.clone());
        }
    }

    if let Some(weight_path) = &econfig.weight_path {
        let trimmed = weight_path.trim_end_matches(['/', '\\']);
        let path = Path::new(trimmed);
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        if let Some(component) = path.components().last() {
            let name = component.as_os_str().to_string_lossy().to_string();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }

    if let Some(weight_file) = &econfig.weight_file {
        let path = Path::new(weight_file);
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    None
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct StructuredOutputs {
    #[serde(default)]
    pub choice: Option<Vec<String>>,
    #[serde(default)]
    pub regex: Option<String>,
    #[serde(default)]
    pub json: Option<serde_json::Value>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub structural_tag: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ResponseFormatJsonSchema {
    #[serde(default)]
    pub name: Option<String>,
    pub schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<ResponseFormatJsonSchema>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ExtraBody {
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputs>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// TopLevelGrammar conversion functions
// Client grammars are composed alongside TEXT and optional reasoning grammars.

pub fn grammar_fragment_from_structured_outputs(
    structured: &StructuredOutputs,
) -> Result<Option<llguidance::api::TopLevelGrammar>> {
    let mut selected: Option<llguidance::api::TopLevelGrammar> = None;
    let mut constraint_count = 0;

    if let Some(choice) = &structured.choice {
        if !choice.is_empty() {
            constraint_count += 1;
            if constraint_count > 1 {
                crate::log_error!("[llg] Multiple constraints specified - structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag");
                return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let choice_gram = crate::tools::schema::build_choice_lark_grammar(choice)
                .map_err(|e| candle_core::Error::msg(e))?;
            selected = Some(choice_gram);
        }
    }

    if let Some(regex) = &structured.regex {
        constraint_count += 1;
        if constraint_count > 1 {
            crate::log_error!("[llg] Multiple constraints specified - structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag");
            return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
        }
        let regex_gram = TopLevelGrammarExt::from_regex_ascii(regex);
        selected = Some(regex_gram);
    }

    if let Some(schema) = &structured.json {
        constraint_count += 1;
        if constraint_count > 1 {
            crate::log_error!("[llg] Multiple constraints specified - structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag");
            return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
        }
        let schema = crate::tools::schema::sanitize_schema_for_llguidance(schema);
        let json_gram = TopLevelGrammarExt::from_json_schema_utf8(schema)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        selected = Some(json_gram);
    }

    if let Some(grammar) = &structured.grammar {
        constraint_count += 1;
        if constraint_count > 1 {
            crate::log_error!("[llg] Multiple constraints specified - structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag");
            return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
        }
        let lark_gram = TopLevelGrammarExt::from_lark_utf8(grammar);
        selected = Some(lark_gram);
    }

    if let Some(tag) = &structured.structural_tag {
        constraint_count += 1;
        if constraint_count > 1 {
            crate::log_error!("[llg] Multiple constraints specified - structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag");
            return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
        }
        let (start, end, schema) = crate::tools::schema::parse_structural_tag(tag)
            .map_err(|e| candle_core::Error::msg(e))?;
        let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
        // Convert schema Value to Vec<Tool> for build_json_tool_lark_grammar
        let tools = schema_to_tools(&schema);
        // structural_tag uses text-based matching, pass None for token IDs
        let tool_gram = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag(&start)
            .end_tag(&end)
            .start_is_special(false)
            .end_is_special(false)
            .build_json();
        selected = Some(tool_gram);
    }

    if selected.is_none() {
        crate::log_error!("[llg] No constraint specified in structured_outputs - must set exactly one of choice, regex, json, grammar, or structural_tag");
        return Err(candle_core::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
    }

    Ok(selected)
}

pub fn grammar_fragment_from_response_format(
    response_format: &ResponseFormat,
) -> Result<Option<llguidance::api::TopLevelGrammar>> {
    match response_format.format_type.as_str() {
        "json_schema" => {
            let Some(schema) = response_format.json_schema.as_ref() else {
                crate::log_error!(
                    "[llg] response_format.json_schema is required for type=json_schema"
                );
                return Err(candle_core::Error::msg(
                    "response_format.json_schema is required",
                ));
            };
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema.schema);
            let json_gram = TopLevelGrammarExt::from_json_schema_utf8(schema)
                .map_err(|e| candle_core::Error::msg(e.to_string()))?;
            Ok(Some(json_gram))
        }
        "json_object" => {
            let json_gram = TopLevelGrammarExt::from_json_schema_utf8(json!({
                "type": "object"
            }))
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
            Ok(Some(json_gram))
        }
        other => {
            crate::log_error!(
                "[llg] Unsupported response_format type '{}'; only 'json_schema' and 'json_object' are supported",
                other
            );
            Err(candle_core::Error::msg(format!(
                "Unsupported response_format type '{}'; only 'json_schema' and 'json_object' are supported",
                other
            )))
        }
    }
}

fn structured_outputs_kind(structured: &StructuredOutputs) -> &'static str {
    if structured
        .choice
        .as_ref()
        .is_some_and(|choice| !choice.is_empty())
    {
        "choice"
    } else if structured.regex.is_some() {
        "regex"
    } else if structured.json.is_some() {
        "json"
    } else if structured.grammar.is_some() {
        "grammar"
    } else if structured.structural_tag.is_some() {
        "structural_tag"
    } else {
        "unknown"
    }
}

pub fn collect_openai_constraint_grammar(
    request: &ChatCompletionRequest,
) -> Result<Option<TopLevelGrammar>> {
    let mut selected: Option<TopLevelGrammar> = None;

    let mut try_set = |grammar: TopLevelGrammar, source: &str, kind: &str| -> Result<()> {
        if selected.is_some() {
            return Err(candle_core::Error::msg(
                "only one of structured_outputs, response_format, or constraint may be set",
            ));
        }
        selected = Some(grammar);
        crate::log_info!(
            "[llg] Request constraint selected: source={} type={}",
            source,
            kind
        );
        Ok(())
    };

    if let Some(structured) = request.structured_outputs.as_ref().or_else(|| {
        request
            .extra_body
            .as_ref()
            .and_then(|body| body.structured_outputs.as_ref())
    }) {
        if let Some(grammar) = grammar_fragment_from_structured_outputs(structured)? {
            try_set(
                grammar,
                "structured_outputs",
                structured_outputs_kind(structured),
            )?;
        }
    }

    if let Some(response_format) = request.response_format.as_ref() {
        if let Some(grammar) = grammar_fragment_from_response_format(response_format)? {
            try_set(
                grammar,
                "response_format",
                response_format.format_type.as_str(),
            )?;
        }
    }

    if let Some(grammar_str) = request.constraint.as_ref() {
        let constraint_type = request.constraint_type.as_deref().unwrap_or("regex");
        let grammar = match constraint_type {
            "regex" => TopLevelGrammarExt::from_regex_ascii(grammar_str),
            "lark" => TopLevelGrammarExt::from_lark_utf8(grammar_str),
            "json_schema" | "json" => {
                let value: serde_json::Value =
                    serde_json::from_str(grammar_str).map_err(candle_core::Error::wrap)?;
                let value = crate::tools::schema::sanitize_schema_for_llguidance(&value);
                TopLevelGrammarExt::from_json_schema_utf8(value)
                    .map_err(candle_core::Error::wrap)?
            }
            other => {
                return Err(candle_core::Error::msg(format!(
                    "unknown constraint_type '{}'",
                    other
                )));
            }
        };
        try_set(grammar, "constraint", constraint_type)?;
    }

    Ok(selected)
}

pub fn build_guided_decoding_grammar(
    guidance_tokens: &GuidanceTokens,
    constraint_grammar: Option<TopLevelGrammar>,
    max_tokens: usize,
    reasoning_effort: Option<ReasoningEffort>,
) -> Option<TopLevelGrammar> {
    if constraint_grammar.is_none() {
        return None;
    }

    crate::log_info!(
        "[llg] Guided decoding enabled: constraint={} max_tokens={} reasoning={}",
        true,
        max_tokens,
        reasoning_effort
            .as_ref()
            .map(|effort| format!("{effort:?}"))
            .unwrap_or_else(|| "none".to_string())
    );

    Some(compose_grammars(
        constraint_grammar.into_iter().collect(),
        Some(max_tokens),
        guidance_tokens,
        reasoning_effort,
    ))
}

pub fn normalize_reasoning_controls(params: &mut SamplingParams, guidance_tokens: &GuidanceTokens) {
    let reasoning_enabled = params
        .reasoning_effort
        .as_ref()
        .is_some_and(|effort| *effort != ReasoningEffort::None);
    if !reasoning_enabled {
        return;
    }

    let has_reasoning_tokens = !guidance_tokens.reasoning_start_ids.is_empty()
        && !guidance_tokens.reasoning_end_ids.is_empty();
    if !has_reasoning_tokens {
        crate::log_warn!(
            "[llg] reasoning_effort requested but current model/tokenizer does not expose reasoning tokens; disabling reasoning grammar"
        );
        params.reasoning_effort = None;
        return;
    }

    params.thinking = Some(true);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStrategy {
    Mean,
    Last,
}

impl Default for EmbeddingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum ImageUrlContent {
    Url(String),
    Object {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

impl ImageUrlContent {
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) => url,
            Self::Object { url, .. } => url,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MessageContent {
    // pure text (classic chat format)
    #[serde(alias = "input_text", alias = "text")]
    Text { text: String },

    // URL image: "image_url": "https://..."
    #[serde(alias = "image_url")]
    ImageUrl { image_url: ImageUrlContent },

    // Base64 format: "data:image/jpeg;base64,xxxxx"
    #[serde(alias = "image_base64")]
    ImageBase64 { image_base64: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum MessageContentType {
    PureText(String),
    Single(MessageContent),
    Multi(Vec<MessageContent>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContentType>,
    /// Tool calls made by the assistant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
    /// Tool call ID when role is "tool"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning/thinking content for the assistant turn (used by some clients)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// Create a simple text message
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(MessageContentType::PureText(content.into())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// Create an assistant message with tool calls
    pub fn with_tool_calls(tool_calls: Vec<crate::tools::ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// Create a tool result message
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContentType::PureText(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
        }
    }
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: Option<String>,
}

/// Public tool call structure with correct serialization fields
#[derive(Serialize, Debug, Clone)]
pub struct PublicToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: crate::tools::FunctionCall,
}

/// Message in the response (may contain tool calls)
#[derive(Serialize)]
pub struct ChatResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<PublicToolCall>>,
}

#[derive(Serialize, Debug)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceChunk>,
    pub usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
pub struct ErrorMsg {
    pub message: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatChoiceChunk {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    pub error: Option<Vec<ErrorMsg>>,
}

#[derive(Serialize, Debug)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<PublicToolCall>>,
}

#[derive(Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: EmbeddingOutput,
    pub index: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum EmbeddingOutput {
    Vector(Vec<f32>),
    Base64(String),
}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default)]
    pub embedding_type: EmbeddingStrategy,
}

// === Tokenize API ===

/// Input for tokenize request - either plain text or chat messages
#[derive(Deserialize)]
#[serde(untagged)]
pub enum TokenizeInput {
    /// Chat messages input (will apply chat template)
    Messages { messages: Vec<ChatMessage> },
    /// Plain text input
    Text { prompt: String },
}

/// Request body for /tokenize endpoint
#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: Option<String>,
    #[serde(flatten)]
    pub input: TokenizeInput,
    /// Whether to add special tokens (default: true)
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
}

/// Response from /tokenize endpoint
#[derive(Serialize)]
pub struct TokenizeResponse {
    /// List of token IDs
    pub tokens: Vec<u32>,
    /// Number of tokens
    pub count: usize,
    /// Maximum model context length (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

// === Detokenize API ===

/// Request body for /detokenize endpoint
#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    /// Token IDs to decode
    pub tokens: Vec<u32>,
    /// Whether to skip special tokens in output (default: true)
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

/// Response from /detokenize endpoint
#[derive(Serialize)]
pub struct DetokenizeResponse {
    /// Decoded text
    pub prompt: String,
}

#[derive(Deserialize)]
pub struct UsageQuery {
    pub session_id: Option<String>,
    // pub user_id: Option<String>,
    // pub detail: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct UsageResponse {
    pub token_used: usize,
    pub max_model_len: usize,
    pub used_kvcache_tokens: usize,
    pub total_kv_cache_tokens: usize,
    pub swap_used: f32,
    pub total_swap_memory: f32,
    pub session_status: String,
}

pub struct ServerData {
    pub engine: Arc<RwLock<LLMEngine>>,
    pub econfig: EngineConfig,
    pub mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
}

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

pub enum ChatResponder {
    Streamer(Sse<Streamer>),
    Completion(ChatCompletionResponse),
    Usage(UsageResponse),
    Embedding(EmbeddingResponse),
    Tokenize(TokenizeResponse),
    Detokenize(DetokenizeResponse),
    ModelError(String),
    InternalError(String),
    ValidationError(String),
}

impl IntoResponse for ChatResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatResponder::Streamer(s) => s.into_response(),
            ChatResponder::Completion(s) => Json(s).into_response(),
            ChatResponder::Usage(s) => Json(s).into_response(),
            ChatResponder::Embedding(s) => Json(s).into_response(),
            ChatResponder::Tokenize(s) => Json(s).into_response(),
            ChatResponder::Detokenize(s) => Json(s).into_response(),
            ChatResponder::InternalError(e) => {
                JsonError::new(e).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatResponder::ValidationError(e) => {
                JsonError::new(e).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatResponder::ModelError(msg) => {
                JsonError::new(msg).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Maximum number of concurrent sequences to allow, default 1 for interactive chat
    #[arg(long, default_value_t = 1)]
    pub max_num_seqs: usize,

    /// Size of a block
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long = "m")]
    pub model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long = "w")]
    pub weight_path: Option<String>,

    /// gguf file path or gguf file name when model_id is given
    #[arg(long = "f")]
    pub weight_file: Option<String>,

    /// Enforce a specific tool-call parser (e.g., qwen, qwen_coder, json)
    #[arg(long, default_value = None)]
    pub enforce_parser: Option<String>,

    pub hf_token: Option<String>,

    pub hf_token_path: Option<String>,

    #[arg(long)]
    pub dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    pub cpu: bool,

    #[arg(long = "d", value_delimiter = ',')]
    pub device_ids: Option<Vec<usize>>,

    #[arg(long, default_value_t = false)]
    pub log: bool,

    #[arg(long, value_delimiter = '|')]
    pub prompts: Option<Vec<String>>,

    // in-site quantization, e.g. q4_k, q2_k, q8_0, etc.
    // if not provided, it will not perform in-situ quantization for the original model
    // do not use this option if you are using a gguf file
    #[arg(long, default_value = None)]
    pub isq: Option<String>,

    #[arg(long = "i", default_value_t = false)]
    pub interactive: bool,

    /// max tokens for each request
    #[arg(long, default_value_t = 16384)]
    pub max_tokens: usize,

    /// for batch performance test
    #[arg(long, default_value = None)]
    pub batch: Option<usize>,

    #[arg(long, default_value = None)]
    pub temperature: Option<f32>,

    #[arg(long, default_value = None)]
    pub top_k: Option<isize>,

    #[arg(long, default_value = None)]
    pub top_p: Option<f32>,

    #[arg(long, default_value = None)]
    pub frequency_penalty: Option<f32>,

    #[arg(long, default_value = None)]
    pub presence_penalty: Option<f32>,

    #[arg(long, default_value = None)]
    pub seed: Option<u64>, //seed for reproduce the results

    #[arg(long)]
    pub tool_prompt: Option<String>,

    /// Disable prefix caching (enabled by default)
    #[arg(long, default_value_t = false)]
    pub disable_prefix_cache: bool,

    /// Max cached prefix size in tokens (rounded down to block size).
    #[arg(long, default_value = None)]
    pub prefix_cache_max_tokens: Option<usize>,

    #[arg(long, default_value_t = false)]
    pub server: bool, //server mode

    #[arg(long)]
    pub port: Option<usize>,

    #[arg(long, default_value = None, help = "KV cache dtype: auto (default, uses model dtype), fp8, turbo8, turbo4, turbo3")]
    pub kvcache_dtype: Option<String>,

    // After model loading, the percentage of the remaining gpu memory for kvcache
    #[arg(long, default_value = None)]
    pub kv_fraction: Option<f32>,

    // For hybrid mamba models, percentage of cache budget reserved for mamba states
    #[arg(long, default_value = None)]
    pub mamba_fraction: Option<f32>,

    #[arg(long, default_value = None)]
    pub cpu_mem_fold: Option<f32>, //the percentage of cpu vs. gpu kvcache size

    #[arg(long, default_value_t = false)]
    pub pd_server: bool, //PD server mode

    #[arg(long, default_value_t = false)]
    pub pd_client: bool, //PD client mode

    #[arg(long)]
    pub pd_url: Option<String>, //Url for PD server mode (server in remote)

    #[arg(long, default_value_t = false)]
    pub ui_server: bool, //Start the web chat

    /// MCP server command to spawn for tool discovery and calls
    #[arg(long, default_value = None)]
    pub mcp_command: Option<String>,

    /// MCP config file path for multi-server setups
    #[arg(long, default_value = None)]
    pub mcp_config: Option<String>,

    /// MCP server arguments (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = None)]
    pub mcp_args: Option<Vec<String>>,

    /// YARN RoPE scaling factor (explicit override, no auto-calculation)
    #[arg(long, default_value = None)]
    pub yarn_scaling_factor: Option<f64>,

    /// Disable reasoning/thinking by default when requests do not pass
    /// `thinking` / `enable_thinking`.
    #[arg(long, default_value_t = false)]
    pub disable_reasoning: bool,

    /// Disable CUDA graph capture (enabled by default when compiled with cuda feature)
    #[arg(long, default_value_t = false)]
    pub disable_cuda_graph: bool,

    /// Base prefill chunk size in tokens. Rounded to 1k, clamped to 1k..32k.
    /// Metal uses half of this value after rounding.
    #[arg(long, default_value_t = crate::utils::config::DEFAULT_PREFILL_CHUNK_SIZE)]
    pub prefill_chunk_size: usize,
}

/// Result of executing tool calls via MCP
#[allow(dead_code)]
pub struct ToolExecutionResult {
    /// Messages to add for follow-up (assistant tool_calls + tool results)
    followup_messages: Vec<ChatMessage>,
    /// The tool calls that were executed
    tool_calls: Vec<crate::tools::ToolCall>,
}

/// Default timeout for individual tool calls (60 seconds)
const TOOL_CALL_TIMEOUT_SECS: u64 = 60;

/// Execute tool calls via MCP manager and return messages for follow-up generation
/// Each tool call has a timeout of TOOL_CALL_TIMEOUT_SECS seconds
pub async fn execute_mcp_tool_calls_async(
    tool_calls: Vec<crate::tools::ToolCall>,
    mcp_manager: std::sync::Arc<crate::mcp::McpClientManager>,
    base_messages: Vec<ChatMessage>,
) -> ToolExecutionResult {
    let mut followup_messages = base_messages.clone();
    followup_messages.push(ChatMessage::with_tool_calls(tool_calls.clone()));

    for call in &tool_calls {
        let args_str = call.function.arguments.as_deref().unwrap_or("{}");
        let args_value: serde_json::Value =
            serde_json::from_str(args_str).unwrap_or_else(|_| serde_json::json!({"raw": args_str}));
        let args_map = args_value
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<HashMap<String, serde_json::Value>>();

        let call_name = call.function.name.clone();
        let call_id = call.id.clone();
        let mcp_manager_clone = mcp_manager.clone();
        crate::log_info!(
            "Executing tool call: {} with args {:?}",
            call_name,
            args_map
        );

        let start = std::time::Instant::now();

        // Execute tool call with timeout using spawn_blocking
        let timeout_duration = std::time::Duration::from_secs(TOOL_CALL_TIMEOUT_SECS);
        let tool_result = match tokio::time::timeout(
            timeout_duration,
            tokio::task::spawn_blocking(move || mcp_manager_clone.call_tool(&call_name, args_map)),
        )
        .await
        {
            Ok(Ok(Ok(result))) => {
                // Success: spawn_blocking succeeded, call_tool succeeded
                let elapsed = start.elapsed();
                crate::log_info!(
                    "Tool '{}' completed in {:.2}s",
                    call.function.name,
                    elapsed.as_secs_f32()
                );
                let mut content = result
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        crate::mcp::ToolContent::Text { text } => Some(text.clone()),
                        crate::mcp::ToolContent::Resource { text, .. } => text.clone(),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if content.trim().is_empty() {
                    content = "Tool executed successfully with no textual output.".to_string();
                }
                ChatMessage::tool_result(call_id, content)
            }
            Ok(Ok(Err(err))) => {
                // Tool execution failed
                let elapsed = start.elapsed();
                crate::log_error!(
                    "Tool '{}' failed after {:.2}s: {:?}",
                    call.function.name,
                    elapsed.as_secs_f32(),
                    err
                );
                ChatMessage::tool_result(
                    call_id,
                    format!("<tool_use_error>Tool execution failed: {err}</tool_use_error>"),
                )
            }
            Ok(Err(join_err)) => {
                // spawn_blocking panicked
                crate::log_error!(
                    "Tool '{}' task panicked: {:?}",
                    call.function.name,
                    join_err
                );
                ChatMessage::tool_result(
                    call_id,
                    format!("<tool_use_error>Tool execution panicked: {join_err}</tool_use_error>"),
                )
            }
            Err(_timeout_err) => {
                // Timeout occurred
                crate::log_error!(
                    "Tool '{}' timed out after {}s",
                    call.function.name,
                    TOOL_CALL_TIMEOUT_SECS
                );
                ChatMessage::tool_result(
                    call_id,
                    format!(
                        "<tool_use_error>Tool execution timed out after {}s</tool_use_error>",
                        TOOL_CALL_TIMEOUT_SECS
                    ),
                )
            }
        };
        followup_messages.push(tool_result);
    }

    ToolExecutionResult {
        followup_messages,
        tool_calls,
    }
}

pub fn convert_chat_message(
    msg: &ChatMessage,
    processor: &mut Option<Box<dyn ImageProcessTrait + Send>>,
    images_tensors: &mut Vec<(Tensor, Vec<(usize, usize)>)>,
) -> Result<Message> {
    let role = msg.role.clone();
    let mut prompt = String::new();
    let mut images = Vec::new();

    // Keep assistant tool-call turns structured so chat templates can render proper
    // function-calling transcripts (same as vLLM/OpenAI style history).
    if role == "assistant" {
        if let Some(tool_calls) = &msg.tool_calls {
            let mut content = String::new();
            if let Some(existing) = &msg.content {
                content = extract_text_content(existing);
            }
            let template_calls = tool_calls
                .iter()
                .map(to_template_tool_call)
                .collect::<Vec<_>>();
            return Ok(Message {
                role,
                content,
                num_images: 0,
                tool_calls: Some(template_calls),
                tool_call_id: None,
                reasoning_content: msg.reasoning_content.clone(),
            });
        }
    }

    // Handle tool result messages specially
    if role == "tool" {
        let content = msg
            .content
            .as_ref()
            .map(extract_text_content)
            .unwrap_or_default()
            .trim()
            .to_owned();
        return Ok(Message {
            role,
            content,
            num_images: 0,
            tool_calls: None,
            tool_call_id: msg.tool_call_id.clone(),
            reasoning_content: None,
        });
    }

    // Normal message handling
    if let Some(content) = &msg.content {
        match content {
            MessageContentType::PureText(text) => {
                prompt.push_str(text);
            }
            MessageContentType::Single(item) => {
                append_message_item(item, &mut prompt, &mut images)?;
                prompt.push(' '); // keep spacing readable
            }
            MessageContentType::Multi(items) => {
                for item in items {
                    append_message_item(item, &mut prompt, &mut images)?;
                    prompt.push(' '); // keep spacing readable
                }
            }
        }
    }

    if !images.is_empty() && processor.is_some() {
        if let Some(processor) = processor.as_mut() {
            let (images_tensor, image_sizes) = processor.process_inputs(&mut prompt, &images)?;
            images_tensors.push((images_tensor, image_sizes));
        }
    }

    let mut message = Message::new(role.clone(), prompt.trim().to_owned(), images.len());
    if role == "assistant" {
        message.reasoning_content = msg.reasoning_content.clone();
    }
    Ok(message)
}

fn extract_text_content(content: &MessageContentType) -> String {
    match content {
        MessageContentType::PureText(text) => text.clone(),
        MessageContentType::Single(item) => match item {
            MessageContent::Text { text } => text.clone(),
            _ => String::new(),
        },
        MessageContentType::Multi(items) => items
            .iter()
            .filter_map(|item| match item {
                MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn to_template_tool_call(call: &crate::tools::ToolCall) -> serde_json::Value {
    let args = parse_template_tool_arguments(call.function.arguments.as_deref());

    serde_json::json!({
        "id": call.id.clone(),
        "type": call.tool_type.clone(),
        "function": {
            "name": call.function.name.clone(),
            "arguments": args
        }
    })
}

fn parse_template_tool_arguments(arguments: Option<&str>) -> serde_json::Value {
    let Some(raw) = arguments.map(str::trim).filter(|s| !s.is_empty()) else {
        return serde_json::json!({});
    };

    match serde_json::from_str::<serde_json::Value>(raw).ok() {
        Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        // Some clients double-encode arguments as a JSON string. Handle that
        // shape so chat templates expecting a mapping still receive one.
        Some(serde_json::Value::String(inner)) => {
            match serde_json::from_str::<serde_json::Value>(inner.trim()).ok() {
                Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
                _ => serde_json::json!({}),
            }
        }
        _ => serde_json::json!({}),
    }
}

fn append_message_item(
    item: &MessageContent,
    prompt: &mut String,
    images: &mut Vec<image::DynamicImage>,
) -> Result<()> {
    match item {
        MessageContent::Text { text } => {
            prompt.push_str(text);
        }
        MessageContent::ImageUrl { image_url } => {
            let url = image_url.url();
            let img = if url.starts_with("data:") {
                let img = load_image_from_base64(url)?;
                crate::log_info!("Chat image decoded: {} x {}", img.width(), img.height());
                img
            } else {
                let img = load_image_from_url(url)?;
                crate::log_info!("Chat image downloaded: {} x {}", img.width(), img.height());
                img
            };
            prompt.push_str(&IMAGE_PLACEHOLDER);
            images.push(img);
        }
        MessageContent::ImageBase64 { image_base64 } => {
            let img = load_image_from_base64(image_base64)?;
            crate::log_info!("Chat image decoded: {} x {}", img.width(), img.height());
            prompt.push_str(&IMAGE_PLACEHOLDER);
            images.push(img);
        }
    }
    Ok(())
}

pub fn build_messages_and_images(
    messages: &[ChatMessage],
    img_cfg: Option<&ImageProcessConfig>,
) -> Result<(Vec<Message>, Option<ImageData>)> {
    use crate::models::qwen3_vl::input::Qwen3VLImageProcessor;
    use crate::utils::config::ModelType;
    use crate::utils::image::ImageProcessor;

    let mut processor: Option<Box<dyn ImageProcessTrait + Send>> = if let Some(cfg) = img_cfg {
        if matches!(cfg.model_type, ModelType::Qwen3VL) {
            Some(Box::new(Qwen3VLImageProcessor::default(cfg)))
        } else {
            Some(Box::new(ImageProcessor::new(cfg)))
        }
    } else {
        None
    };

    let mut images: Vec<(Tensor, Vec<(usize, usize)>)> = vec![];

    let messages: Vec<Message> = messages
        .iter()
        .map(|m| convert_chat_message(m, &mut processor, &mut images))
        .collect::<Result<Vec<_>>>()?;

    let image_data = if !images.is_empty() && img_cfg.is_some() {
        let mut image_sizes = Vec::new();
        let mut image_tensors = Vec::new();
        for (t, s) in &images {
            image_tensors.push(t);
            image_sizes.extend(s);
        }
        let images_tensor = Tensor::cat(&image_tensors, 0)?;
        let (images_raw, images_shape) = get_tensor_raw_data(&images_tensor)?;
        crate::log_info!(
            "{} images detected in the chat message, combined image shape {:?}",
            images_shape[0],
            images_shape
        );
        let cfg = img_cfg.unwrap();
        let tokens_per_image = compute_tokens_per_image(cfg, &image_sizes);
        Some(ImageData {
            raw: images_raw,
            shape: images_shape,
            patches: image_sizes,
            image_idx: 0,
            image_token_offset: 0,
            tokens_per_image,
            image_token_id: cfg.image_token_id,
        })
    } else {
        None
    };

    Ok((messages, image_data))
}

pub async fn run_server(
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
    port: usize,
    with_ui_server: bool,
) -> Result<()> {
    use axum::extract::DefaultBodyLimit;
    let (has_vision, model_name, resolved_max_model_len) = {
        let e = engine.read();
        e.get_model_info()
    };
    let has_vision = Arc::new(has_vision);
    let exposed_model_id = resolve_engine_model_id(&econfig).unwrap_or_else(|| model_name.clone());
    let exposed_max_model_len = resolved_max_model_len;

    let is_pd_server = if let Some(cfg) = &econfig.pd_config {
        matches!(cfg.role, PdRole::Server)
    } else {
        false
    };

    let mcp_manager_config = if let Some(path) = &econfig.mcp_config {
        match crate::mcp::manager::McpManagerConfig::from_file(path) {
            Ok(cfg) => Some(cfg),
            Err(err) => {
                crate::log_error!("Failed to load MCP config file: {:?}", err);
                None
            }
        }
    } else if let Some(command) = econfig.mcp_command.clone() {
        Some(crate::mcp::manager::McpManagerConfig::from_single(
            crate::mcp::manager::McpToolConfig::new(
                command,
                econfig.mcp_args.clone().unwrap_or_default(),
            ),
        ))
    } else {
        None
    };

    let mcp_manager = if let Some(cfg) = mcp_manager_config {
        match crate::mcp::McpClientManager::new(cfg) {
            Ok(manager) => Some(Arc::new(manager)),
            Err(err) => {
                crate::log_error!("Failed to start MCP client manager: {:?}", err);
                None
            }
        }
    } else {
        None
    };

    let server_data = ServerData {
        engine,
        econfig,
        mcp_manager,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route(
            "/v1/models",
            get(move || async move {
                let m = if *has_vision {
                    vec!["text", "image"]
                } else {
                    vec!["text", "embedding"]
                };
                Json(json!({
                    "object": "list",
                    "data": [
                        {
                            "id": exposed_model_id,
                            "object": "model",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as i64,
                            "owned_by": "xinfer",
                            "permission": [],
                            "modalities": m,
                            "max_model_len": exposed_max_model_len,
                        }
                    ]
                }))
            }),
        )
        .route("/v1/chat/completions", post(server::chat_completion))
        .route("/v1/messages", post(claude_server::messages))
        .route(
            "/v1/messages/count_tokens",
            post(claude_server::count_tokens),
        )
        .route("/v1/embeddings", post(server::create_embeddings))
        .route("/v1/usage", get(server::get_usage))
        .route("/tokenize", post(server::tokenize))
        .route("/detokenize", post(server::detokenize))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB body size limit
        .layer(cors)
        .with_state(Arc::new(server_data));

    let addr = if is_pd_server {
        crate::log_warn!("🚀 PD server started, waiting for prefill request(s)...",);
        format!("0.0.0.0:{}", 0)
    } else {
        let ip = local_ip().unwrap_or("127.0.0.1".parse().unwrap());
        let local_url = format!("http://localhost:{port}/v1/");
        let lan_url = format!("http://{ip}:{port}/v1/");

        let api_server_url = format!(
            "🧠 API server running at:\n   -  {} (Local Access) \n   -  {} (Remote Access, IP may not correct under Docker)",
            local_url, lan_url
        );
        println!("{}", api_server_url.cyan());

        println!(
            "{}",
            format!("📡 Supported endpoints (OpenAI/Claude):").yellow()
        );
        println!("{}", format!("   - POST /v1/chat/completions").yellow());
        println!("{}", format!("   - POST /v1/messages").yellow());
        println!(
            "{}",
            format!("   - POST /v1/messages/count_tokens").yellow()
        );
        println!("{}", format!("   - POST /v1/embeddings").yellow());
        println!("{}", format!("   - GET  /v1/models").yellow());
        println!("{}", format!("   - GET  /v1/usage").yellow());
        println!("{}", format!("   - POST /tokenize").yellow());
        println!("{}", format!("   - POST /detokenize").yellow());
        println!("");
        println!(
            "🛑 {}",
            format!("EXIT: Ctrl+C to quit. If unresponsive: Ctrl+P → Ctrl+Q (last resort).")
                .bold()
                .red()
        );
        println!("");
        format!("0.0.0.0:{}", port)
    };

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let mut tasks = Vec::new();
    tasks.push(tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("API server error: {e:?}");
        }
    }));

    if with_ui_server {
        tasks.push(tokio::spawn(async move {
            start_ui_server((port + 1) as u16, Some(port as u16), None, None)
                .await
                .unwrap();
        }));
    }

    futures::future::try_join_all(tasks)
        .await
        .map_err(candle_core::Error::wrap)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_without_images() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContentType::PureText("hello world".to_string())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let (converted, images) = build_messages_and_images(&messages, None).unwrap();

        assert!(images.is_none());
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content, "hello world");
        assert_eq!(converted[0].num_images, 0);
    }

    #[test]
    fn test_chat_message_helpers() {
        let text_msg = ChatMessage::text("user", "Hello!");
        assert_eq!(text_msg.role, "user");
        assert!(text_msg.content.is_some());

        let tool_result = ChatMessage::tool_result("call_123", r#"{"result": 42}"#);
        assert_eq!(tool_result.role, "tool");
        assert_eq!(tool_result.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn preserves_assistant_tool_calls_in_template_message() {
        let tool_call =
            crate::tools::new_tool_call("call_1", "Read", r#"{"file_path":"ReadMe.md"}"#);
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![tool_call]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let mut processor = None;
        let mut images = Vec::new();
        let converted = convert_chat_message(&msg, &mut processor, &mut images).unwrap();

        assert_eq!(converted.role, "assistant");
        assert_eq!(converted.content, "");
        assert!(converted.tool_calls.is_some());
        let calls = converted.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "Read");
        assert!(calls[0]["function"]["arguments"].is_object());
        assert_eq!(calls[0]["function"]["arguments"]["file_path"], "ReadMe.md");
    }

    #[test]
    fn preserves_tool_result_metadata_in_template_message() {
        let msg = ChatMessage::tool_result("call_1", "{\"ok\":true}");
        let mut processor = None;
        let mut images = Vec::new();
        let converted = convert_chat_message(&msg, &mut processor, &mut images).unwrap();

        assert_eq!(converted.role, "tool");
        assert_eq!(converted.content, "{\"ok\":true}");
        assert_eq!(converted.tool_call_id, Some("call_1".to_string()));
    }

    #[test]
    fn test_tokenize_request_text_parsing() {
        let json = r#"{"prompt": "Hello, world!"}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        match request.input {
            TokenizeInput::Text { prompt } => assert_eq!(prompt, "Hello, world!"),
            _ => panic!("Expected TokenizeInput::Text"),
        }
    }

    #[test]
    fn test_tokenize_request_messages_parsing() {
        let json = r#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        match request.input {
            TokenizeInput::Messages { messages } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, "user");
            }
            _ => panic!("Expected TokenizeInput::Messages"),
        }
    }

    #[test]
    fn test_tokenize_request_with_options() {
        let json = r#"{"prompt": "test", "add_special_tokens": false}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.add_special_tokens, Some(false));
    }

    #[test]
    fn test_detokenize_request_parsing() {
        let json = r#"{"tokens": [1, 2, 3, 4]}"#;
        let request: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_detokenize_request_with_options() {
        let json = r#"{"tokens": [1, 2], "skip_special_tokens": false}"#;
        let request: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.skip_special_tokens, Some(false));
    }

    #[test]
    fn test_tokenize_response_serialization() {
        let response = TokenizeResponse {
            tokens: vec![1, 2, 3],
            count: 3,
            max_model_len: Some(4096),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"count\":3"));
        assert!(json.contains("\"tokens\":[1,2,3]"));
    }

    #[test]
    fn test_detokenize_response_serialization() {
        let response = DetokenizeResponse {
            prompt: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"prompt\":\"Hello, world!\""));
    }

    #[test]
    fn test_chat_completion_tool_choice_required_parsing() {
        let json = r#"{"messages": [{"role":"user","content":"hi"}], "tool_choice": "required"}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(
            request.tool_choice,
            Some(crate::tools::ToolChoice::Mode(
                crate::tools::ToolChoiceMode::Required
            ))
        ));
    }

    #[test]
    fn test_chat_completion_stop_parsing() {
        let json = r#"{"messages":[{"role":"user","content":"hi"}],"stop":["END"]}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
    }

    #[test]
    fn test_chat_completion_stop_string_parsing() {
        let json = r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
    }

    #[test]
    fn test_chat_completion_stream_options_parsing() {
        let json = r#"{
            "messages":[{"role":"user","content":"hi"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        }"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stream_options.map(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn test_collect_openai_constraint_grammar_rejects_multiple_sources() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages":[{"role":"user","content":"hi"}],
                "structured_outputs":{"choice":["a"]},
                "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}
            }"#,
        )
        .unwrap();
        let result = collect_openai_constraint_grammar(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_choice() {
        let so = StructuredOutputs {
            choice: Some(vec!["option1".to_string(), "option2".to_string()]),
            regex: None,
            json: None,
            grammar: None,
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_json() {
        let so = StructuredOutputs {
            choice: None,
            regex: None,
            json: Some(serde_json::json!({"type": "object", "properties": {}})),
            grammar: None,
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_regex() {
        let so = StructuredOutputs {
            choice: None,
            regex: Some("^[a-z]+$".to_string()),
            json: None,
            grammar: None,
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_grammar() {
        let so = StructuredOutputs {
            choice: None,
            regex: None,
            json: None,
            // Grammar without start: - that's managed by ComposedGrammar
            grammar: Some("'hello' 'world'".to_string()),
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_empty() {
        let so = StructuredOutputs {
            choice: None,
            regex: None,
            json: None,
            grammar: None,
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_fragment_from_structured_outputs_too_many() {
        let so = StructuredOutputs {
            choice: Some(vec!["a".to_string()]),
            regex: Some("b".to_string()),
            json: None,
            grammar: None,
            structural_tag: None,
        };
        let result = grammar_fragment_from_structured_outputs(&so);
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_fragment_from_response_format_json_schema() {
        let rf = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(ResponseFormatJsonSchema {
                name: None,
                schema: serde_json::json!({"type": "object", "properties": {}}),
            }),
        };
        let result = grammar_fragment_from_response_format(&rf);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_response_format_json_object() {
        let rf = ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        };
        let result = grammar_fragment_from_response_format(&rf);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_grammar_fragment_from_response_format_missing_json_schema() {
        let rf = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: None,
        };
        let result = grammar_fragment_from_response_format(&rf);
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_fragment_from_response_format_unsupported_type() {
        let rf = ResponseFormat {
            format_type: "unsupported".to_string(),
            json_schema: None,
        };
        let result = grammar_fragment_from_response_format(&rf);
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_fragment_from_response_format_json_schema_composed() {
        // Test that json_schema grammars pass through ComposedGrammar
        let rf = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(ResponseFormatJsonSchema {
                name: None,
                schema: serde_json::json!({"type": "object", "properties": {"test": {"type": "string"}}}),
            }),
        };
        let result = grammar_fragment_from_response_format(&rf);
        assert!(result.is_ok());
        // The grammar was created via ComposedGrammar - just verify it's Some
        let grammar = result.unwrap();
        assert!(grammar.is_some());
    }

    #[test]
    fn test_build_guided_decoding_grammar_reasoning_only() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: vec![2],
            reasoning_start_ids: vec![101],
            reasoning_end_ids: vec![102],
        };

        let grammar =
            build_guided_decoding_grammar(&guidance_tokens, None, 64, Some(ReasoningEffort::Low));

        assert!(
            grammar.is_none(),
            "reasoning-only requests must not build guided decoding without a constraint"
        );
    }

    #[test]
    fn test_build_guided_decoding_grammar_reasoning_with_choice_constraint() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: vec![2],
            reasoning_start_ids: vec![101],
            reasoning_end_ids: vec![102],
        };
        let constraint =
            TopLevelGrammar::from_lark_utf8(r#"start: "positive" | "negative" | "neutral""#);

        let grammar = build_guided_decoding_grammar(
            &guidance_tokens,
            Some(constraint),
            64,
            Some(ReasoningEffort::Low),
        )
        .expect("reasoning + choice guided grammar should be built");

        let lark = crate::utils::guidance::get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark.contains("start: reasoning_block"),
            "start rule should sequence reasoning first: {lark}"
        );
        assert!(
            lark.contains(r#""positive" | "negative" | "neutral""#),
            "choice constraint should remain grouped: {lark}"
        );
        assert!(
            lark.contains("\nreasoning_block:"),
            "reasoning rule definition should remain on its own line: {lark}"
        );
        assert!(
            lark.contains("\nthinkgram:"),
            "reasoning helper rules should remain intact: {lark}"
        );
    }

    #[test]
    fn test_build_guided_decoding_grammar_json_schema_constraint_with_reasoning() {
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: vec![2],
            reasoning_start_ids: vec![101],
            reasoning_end_ids: vec![102],
        };
        let constraint = TopLevelGrammar::from_json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "label": {"type": "string"}
            },
            "required": ["label"],
            "additionalProperties": false
        }));

        let grammar = build_guided_decoding_grammar(
            &guidance_tokens,
            Some(constraint),
            64,
            Some(ReasoningEffort::Low),
        )
        .expect("reasoning + json-schema guided grammar should be built");

        let lark = crate::utils::guidance::get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark.contains("@reasoning @inner"),
            "wrapper should reference reasoning and inner subgrammars: {lark}"
        );
        assert!(
            !lark.contains("none have lark_grammar"),
            "wrapper must not stringify non-lark grammars: {lark}"
        );
        assert!(
            grammar
                .grammars
                .iter()
                .any(|g| g.name.as_deref() == Some("inner") && g.json_schema.is_some()),
            "json-schema constraint should be preserved as nested grammar"
        );
    }

    #[test]
    fn test_normalize_reasoning_controls_enables_thinking() {
        let mut params = SamplingParams::new_with_max_tokens(32);
        params.thinking = Some(false);
        params.reasoning_effort = Some(ReasoningEffort::Low);
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: vec![2],
            reasoning_start_ids: vec![101],
            reasoning_end_ids: vec![102],
        };

        normalize_reasoning_controls(&mut params, &guidance_tokens);

        assert_eq!(params.thinking, Some(true));
        assert_eq!(params.reasoning_effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn test_normalize_reasoning_controls_disables_unsupported_reasoning() {
        let mut params = SamplingParams::new_with_max_tokens(32);
        params.thinking = Some(false);
        params.reasoning_effort = Some(ReasoningEffort::High);
        let guidance_tokens = GuidanceTokens {
            eos_token_ids: vec![2],
            reasoning_start_ids: Vec::new(),
            reasoning_end_ids: Vec::new(),
        };

        normalize_reasoning_controls(&mut params, &guidance_tokens);

        assert_eq!(params.thinking, Some(false));
        assert_eq!(params.reasoning_effort, None);
    }

    #[test]
    fn usage_omits_token_details_when_none() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        let value: serde_json::Value = serde_json::to_value(&usage).expect("serialize usage");
        let object = value.as_object().expect("usage is a JSON object");

        assert_eq!(object.get("prompt_tokens"), Some(&serde_json::json!(100)));
        assert_eq!(
            object.get("completion_tokens"),
            Some(&serde_json::json!(50))
        );
        assert_eq!(object.get("total_tokens"), Some(&serde_json::json!(150)));
        assert!(
            !object.contains_key("prompt_tokens_details"),
            "prompt_tokens_details should be omitted when None, got: {value}"
        );
        assert!(
            !object.contains_key("completion_tokens_details"),
            "completion_tokens_details should be omitted when None, got: {value}"
        );
    }

    #[test]
    fn usage_includes_prompt_tokens_details_when_some() {
        let usage = Usage {
            prompt_tokens: 200,
            completion_tokens: 64,
            total_tokens: 264,
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 128 }),
            completion_tokens_details: None,
        };

        let value: serde_json::Value = serde_json::to_value(&usage).expect("serialize usage");
        assert_eq!(
            value
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(|v| v.as_u64()),
            Some(128),
            "cached_tokens should round-trip under prompt_tokens_details, got: {value}",
        );
    }

    #[test]
    fn usage_includes_completion_tokens_details_when_some() {
        let usage = Usage {
            prompt_tokens: 64,
            completion_tokens: 256,
            total_tokens: 320,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: 192,
            }),
        };

        let value: serde_json::Value = serde_json::to_value(&usage).expect("serialize usage");
        assert_eq!(
            value
                .pointer("/completion_tokens_details/reasoning_tokens")
                .and_then(|v| v.as_u64()),
            Some(192),
            "reasoning_tokens should round-trip under completion_tokens_details, got: {value}",
        );
    }
}
