# MCP Integration and Tool Calling

xInfer supports **Model Context Protocol (MCP)** integration, allowing you to easily expose tools from MCP servers to your LLM.

**Important:** xInfer follows the standard OpenAI tool calling specification. This means the server **only handles tool definitions and prompt injection**. It does **not** execute tools internally. When a model calls a tool, the generation stops, and the tool call details are returned to the client. The client is responsible for executing the tool and submitting the results back to the server.

## Overview

### What is MCP?

The **Model Context Protocol (MCP)** is a standardized protocol for connecting LLMs to external tools and services. xInfer supports:

- **Stdio transport**: Connect to local MCP servers via command-line processes
- **HTTP/SSE transport**: Connect to remote MCP servers via HTTP
- **Multi-server support**: Configure multiple MCP servers with automatic tool name prefixing
- **Automatic tool injection**: Tool definitions from configured MCP servers are automatically injected into the model's system prompt

---

## Tool Calling Workflow

1. **Configuration**: You configure MCP servers in xInfer (via CLI or `mcp.json`).
2. **Injection**: xInfer fetches tool definitions from these servers and appends them to the system prompt of your request.
3. **Generation**: The model generates a tool call.
4. **Completion**: The stream (or request) finishes with `finish_reason="tool_calls"`.
5. **Execution (Client-side)**: Your client code receives the tool call, executes the corresponding function (which you must implement or bridge to an MCP client), and sends the result back in a new request.

### Example Flow

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

# Request triggers tool injection from configured MCP servers
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List files in current directory"}],
    stream=True
)

# ... Handle stream ...
# If model calls a tool, response will contain tool_calls
```

---

## Configuration Options

### CLI Options

| Option | Description |
|--------|-------------|
| `--mcp-command` | Path to single MCP server executable |
| `--mcp-args` | Comma-separated arguments for MCP server |
| `--mcp-config` | Path to JSON config file for multiple servers |

### Config File Format

#### Local Servers (Stdio Transport)

```json
{
    "mcpServers": {
        "server_name": {
            "command": "path/to/executable",
            "args": ["arg1", "arg2"],
            "env": {
                "ENV_VAR": "value"
            }
        }
    }
}
```

#### Remote Servers (HTTP/SSE Transport)

```json
{
    "mcpServers": {
        "server_name": {
            "url": "https://mcp.example.com/api/",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "Accept": "text/event-stream"
            }
        }
    }
}
```

> **Note:** Each server must have either `command` (local) or `url` (remote), not both. You can mix local and remote servers in the same config file.

### Tool Name Prefixing

When using multiple MCP servers, tool names are prefixed with the server name to avoid conflicts:

- Server `filesystem` with tool `list_directory` → `filesystem_list_directory`
- Server `github` with tool `search_repos` → `github_search_repos`

---

## Tool Call Parsing Architecture

xInfer uses a dual-strategy approach for detecting tool calls in streaming responses.

### Detection Strategy

1. **Token ID-based detection** (preferred): For models with dedicated tool call tokens
   - Uses special token IDs defined in the tokenizer
   - More robust and prevents false positives from text that looks like tool calls

2. **Text-based detection** (fallback): For models without special tokens
   - Uses pattern matching for start/end tag strings
   - Applied when token IDs are not available

### Model Support Matrix

| Model Family | Detection Method | Start Token | End Token |
|--------------|-----------------|-------------|-----------|
| Qwen 2.5/3   | Token ID + Text | `<tool_call>` (151657) | `</tool_call>` (151658) |
| Llama 3/3.1  | Token ID + Text | `<\|python_tag\|>` (128010) | `<\|eom_id\|>` (128008) |
| Mistral      | Token ID + Text | `[TOOL_CALLS]` (9) | `]` |
| Gemma        | Text only       | `<start_function_call>` | `<end_function_call>` |
| Phi4, GLM4   | Text only       | `<tool_call>` | `</tool_call>` |

### Buffering Logic

The streaming tool parser uses a state machine:

1. **Normal State**: Tokens stream through. Check for start trigger.
2. **Buffering State**: On start detection, accumulate tokens silently.
3. **End Detection**: When end token/text is detected:
   - Parse buffer into structured tool calls
   - If parsing succeeds: return `finish_reason: "tool_calls"`
   - If parsing fails: flush buffered content as normal text

### Content Filtering

Tool calls are NOT detected inside:
- **Reasoning blocks** (`<think>...</think>`, `<|think|>...<|/think|>`)
- **Code blocks** (` ``` `)

This prevents false positives from examples or documentation.

---

## Behavior Matrix

| Has Request Tools | MCP Configured | Behavior |
|-------------------|----------------|----------|
| ❌ | ❌ | Normal generation |
| ✅ | ❌ | User tools used. Stream finishes at tool call. |
| ❌ | ✅ | MCP tools injected. Stream finishes at tool call. |
| ✅ | ✅ | Both User and MCP tools available. Stream finishes at tool call. |
