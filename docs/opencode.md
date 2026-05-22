# OpenCode + xInfer (OpenAI-compatible endpoint)

This guide connects OpenCode directly to xInfer using the built-in
OpenAI-compatible `/v1/chat_completions` API. No proxy required.

```
OpenCode -> xInfer (OpenAI-compatible)
```

## 1) Start xInfer on port 8000

```bash
# Rust
xinfer --m Qwen/Qwen3.5-35B-A3B-FP8 --server --d 0

# Different model
xinfer --m Qwen/Qwen3.5-27B-FP8 --d 0 --server

# Python
python3 -m xinfer.server --m Qwen/Qwen3-Coder-Next-FP8 --d 0,1
```

## 2) Configure OpenCode

Install opencode (CLI)

```shell
curl -fsSL https://opencode.ai/install | bash
# Or install with npm
npm i -g opencode-ai
```

Export config into `~/.config/opencode/config.json`


```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "xinfer": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "xInfer Local",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3-coder": {
          "name": "Qwen3 Coder"
        }
      }
    }
  },
  "model": "xinfer/qwen3-coder"
}
```

Install Desktop OpenCode (optional)

```shell
visit https://opencode.ai/download
```

Connect to provider -> custom -> base URL (http://localhost:8000/v1) -> Empty key


## 3) Run OpenCode

run opencode (CLI)

```shell
opencode
```

Or, run OpenCode desktop (choose configured custom provider)

### Trouble shooting

1. Use the chat logger to monitor detailed interactions between OpenCode and xInfer.

```shell
# Log into files (in folder ./log)
export XINFER_CHAT_LOGGER=1
```