// src/server/logger.rs
//! Chat completion request/response logger.
//! Enable by setting environment variable XINFER_CHAT_LOGGER=true
//!
//! Supports both OpenAI API and Claude API server logging.

use super::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::tools::ToolCall;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Server type for distinguishing log entries
#[derive(Debug, Clone, Copy)]
pub enum ServerType {
    OpenAI,
    Claude,
}

impl std::fmt::Display for ServerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerType::OpenAI => write!(f, "OpenAI"),
            ServerType::Claude => write!(f, "Claude"),
        }
    }
}

/// Check if chat logging is enabled via environment variable
pub fn is_logging_enabled() -> bool {
    std::env::var("XINFER_CHAT_LOGGER")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

/// Helper struct to log chat completion requests and responses to files.
/// Each call creates a new file with timestamp in the "log" folder.
pub struct ChatCompletionLogger {
    file_path: String,
    server_type: ServerType,
}

impl ChatCompletionLogger {
    /// Create a new logger for OpenAI API. Returns None if logging is disabled.
    pub fn new() -> Option<Arc<Self>> {
        Self::with_server_type(ServerType::OpenAI)
    }

    /// Create a new logger for Claude API. Returns None if logging is disabled.
    pub fn new_claude() -> Option<Arc<Self>> {
        Self::with_server_type(ServerType::Claude)
    }

    /// Create a new logger with specified server type. Returns None if logging is disabled.
    pub fn with_server_type(server_type: ServerType) -> Option<Arc<Self>> {
        if !is_logging_enabled() {
            return None;
        }

        // Ensure log directory exists
        let log_dir = Path::new("log");
        if !log_dir.exists() {
            let _ = fs::create_dir_all(log_dir);
        }

        // Generate timestamp-based filename using std time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();
        let millis = now.subsec_millis();
        let prefix = match server_type {
            ServerType::OpenAI => "openai",
            ServerType::Claude => "claude",
        };
        let file_path = format!("log/{}_{}_{:03}.log", prefix, secs, millis);

        crate::log_info!(
            "[{}] Chat logging enabled, writing to: {}",
            prefix.to_uppercase(),
            file_path
        );

        Some(Arc::new(Self {
            file_path,
            server_type,
        }))
    }

    fn write(&self, content: &str) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            let _ = file.write_all(content.as_bytes());
        }
    }

    pub fn log_request(&self, request: &ChatCompletionRequest) {
        if let Ok(json) = serde_json::to_string_pretty(request) {
            let content = format!("=== {} REQUEST ===\n{}\n\n", self.server_type, json);
            self.write(&content);
        }
    }

    /// Log a raw request (for Claude API or other formats)
    pub fn log_raw_request<T: serde::Serialize>(&self, request: &T) {
        if let Ok(json) = serde_json::to_string_pretty(request) {
            let content = format!("=== {} REQUEST ===\n{}\n\n", self.server_type, json);
            self.write(&content);
        }
    }

    pub fn log_stream_token(&self, token: &str) {
        self.write(token);
    }

    /// Log raw tool call body before parsing
    pub fn log_raw_tool_body(&self, raw: &str) {
        let content = format!("\n\n=== RAW TOOL BODY ===\n{}\n", raw);
        self.write(&content);
    }

    /// Log parsed tool calls with a label (valid/invalid)
    pub fn log_tool_calls(&self, label: &str, tool_calls: &[ToolCall]) {
        if tool_calls.is_empty() {
            return;
        }
        if let Ok(json) = serde_json::to_string_pretty(tool_calls) {
            let content = format!(
                "\n=== {} TOOL CALLS ({}) ===\n{}\n",
                label.to_uppercase(),
                tool_calls.len(),
                json
            );
            self.write(&content);
        }
    }

    pub fn log_stream_end(&self, final_chunk: &ChatCompletionChunk) {
        if let Ok(json) = serde_json::to_string_pretty(final_chunk) {
            let content = format!("\n\n=== FINAL CHUNK ===\n{}\n", json);
            self.write(&content);
        }
    }

    /// Log final chunk with tool calls (for streaming)
    pub fn log_final_tool_chunk(&self, final_chunk: &ChatCompletionChunk) {
        self.log_stream_end(final_chunk);
    }

    pub fn log_response(&self, response: &ChatCompletionResponse) {
        if let Ok(json) = serde_json::to_string_pretty(response) {
            let content = format!("=== {} RESPONSE ===\n{}\n", self.server_type, json);
            self.write(&content);
        }
    }

    /// Log a raw response (for Claude API or other formats)
    pub fn log_raw_response<T: serde::Serialize>(&self, response: &T) {
        if let Ok(json) = serde_json::to_string_pretty(response) {
            let content = format!("=== {} RESPONSE ===\n{}\n", self.server_type, json);
            self.write(&content);
        }
    }

    /// Log an error
    pub fn log_error(&self, error: &str) {
        let content = format!("\n=== ERROR ===\n{}\n", error);
        self.write(&content);
    }

    /// Log the prompt after chat template application
    pub fn log_prompt(&self, prompt: &str) {
        let content = format!("\n=== PROMPT (after chat template) ===\n{}\n", prompt);
        self.write(&content);
    }

    pub fn log_start_response(&self) {
        let content = format!("=== {} MODEL RESPONSE ===\n", self.server_type);
        self.write(&content);
    }
}
