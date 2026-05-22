## Tool Call Parsing

This project uses the model-specific parsers for parsing tool calls for both
streaming and non-streaming responses. The goal is to keep parsing logic
consistent across models while remaining robust to partial output and format
differences.

### Parser selection

Parser selection follows this order:

1. `--enforce-parser` (if provided and valid).
2. Model-based heuristics (model type + model id).
3. Fallback to `passthrough`.

If you pass an invalid name to `--enforce-parser`, the server returns an error
and includes the list of valid parser names.

Available parser names:

- passthrough
- json
- mistral
- qwen
- qwen_coder
- pythonic
- llama
- deepseek
- glm45_moe
- glm47_moe
- step3
- kimik2
- minimax_m2

### Streaming parsing

Streaming requests use incremental parsing logic. Each incoming token is appended to an internal buffer and fed into the parser. The parser emits streaming tool call fragments, which are accumulated into full tool calls.

When an end marker is detected (token id or `</tool_call>` tag), the stream parser:

1. Flushes any unstreamed arguments from the external parser.
2. Builds tool calls from the accumulated fragments.
3. If no tool calls were produced, falls back to `parse_complete_with_fallback`
   on the buffered content.

If parsing still fails, the buffered content is emitted as normal text so the
client does not lose output.

### Non-streaming parsing

Non-streaming requests reuse the same stream parser and call
`parse_complete_with_fallback`. This keeps parser selection and fallback logic
identical between streaming and non-streaming paths.

### Enforcing a parser

CLI (Rust server):

```
--enforce-parser qwen_coder
```

Python server example (`server.py` or `xinfer.server`):

```
--enforce-parser qwen_coder
```

### Environment Variables

- `XINFER_STRICT_TOOL_CALL`:
  - `1` or `true`: Strict validation. Dropping invalid tool calls (calls that do not match the schema) effectively preventing them from being sent to the client. The server logs a warning for dropped calls.
  - `0` or `false` (default): Lenient validation. Invalid tool calls are kept and sent to the client, but a warning is logged by the server. This allows models to output "hallucinated" or malformed calls if desired.

