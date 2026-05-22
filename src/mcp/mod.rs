// src/mcp/mod.rs
//! Model Context Protocol (MCP) support for xInfer
//!
//! MCP enables AI applications to connect with external data sources and tools
//! through a standardized protocol.
//!
//! Supports both local (stdio) and remote (HTTP/SSE) MCP servers.

pub mod client;
pub mod manager;
pub mod server;
pub mod transport;
pub mod types;

pub use client::McpClient;
pub use manager::{
    McpClientManager, McpManagerConfig, McpServerDefinition, McpToolConfig, McpTransportType,
};
pub use server::McpServer;
pub use transport::HttpTransport;
pub use types::*;
