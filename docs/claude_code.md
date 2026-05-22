# Claude Code + xInfer (Anthropic-compatible endpoint)

This guide connects Claude Code directly to xInfer using the built-in
Anthropic-compatible `/v1/messages` API. No proxy required.

```
Claude Code -> xInfer (Anthropic-compatible)
```

## 1) Start xInfer on port 8000

```bash
# Rust
xinfer --m Qwen/Qwen3-Coder-Next-FP8 --server --d 0,1

# Different model
xinfer --m Qwen/Qwen3.5-27B-FP8 --d 0 --server

# Python
python3 -m xinfer.server --m Qwen/Qwen3-Coder-Next-FP8 --d 0,1
```

## 2) Configure Claude Code

Install claude code

```shell
npm install -g @anthropic-ai/claude-code
```

or
```shell
curl -fsSL https://claude.ai/install.sh | bash
```

Export config

```shell
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="sk-dummy"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

Or make it permanent

Set `~/.claude/settings.json` (or copy from `example/claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_MODEL": "default",
    "ANTHROPIC_SMALL_FAST_MODEL": "default",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

## 3) Run Claude Code

run claude code

```shell
claude
```

or verify with a direct request (optional)

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Hello from Claude Code"}
    ]
  }'
```

## Notes

- Streaming uses server-sent events (SSE) on `/v1/messages` with `stream: true`.
- Token counting is available at `POST /v1/messages/count_tokens`.
- Embeddings are not part of the Anthropic API and are not exposed here.

### Trouble shooting

1. Use the chat logger to monitor detailed interactions between Claude Code and xInfer.

```shell
# Log into files (in folder ./log)
export XINFER_CHAT_LOGGER=1
```