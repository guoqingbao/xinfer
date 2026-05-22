# xbot + xInfer

This guide connects `xbot` directly to `xInfer` through the built-in OpenAI-compatible `/v1/chat/completions` endpoint.

```
xbot -> xInfer (OpenAI-compatible)
```

## 1) Start xInfer (at port 8000)

```bash
./run --release --features cuda,nccl,flashinfer,cutlass \
  --m Qwen/Qwen3.6-27B-FP8
```

or

```shell
# Python
python3 -m xinfer.server --m Qwen/Qwen3.6-27B-FP8
```

## 2) Discover the served model name

```bash
curl http://localhost:8000/v1/models
```

Use the returned `id` in the xbot config.

## 3) Configure xbot

Install xbot:

```bash
npm install -g @trusted-ai/xbot
```
Or:

```bash
cargo install xbot
```

Run xbot provider config:

```bash
xbot onboard
xbot config --provider
```

Refer to the following selections:

```shell
> Select provider to configure: custom
> Enter a unique name for this custom provider: xinfer
> Enter API Base URL (e.g. https://api.yourprovider.com/v1): http://localhost:8000/v1/
> Enter API Key? (Optional for local/custom) No

Fetching available models...
> Select default model: Qwen3.6-27B-FP8
Using contextWindowTokens from model metadata: 262144

> Configure subagent provider/model now? No

Configuration saved successfully!
Config file: /root/.xbot/config.json

Final Provider Config:
{
  "api_key": "",
  "api_base": "http://localhost:8000/v1/",
  "extra_headers": {},
  "reasoning_effort": null
}
```

## 4) Run xbot

One-shot task

```bash
cd YOUR PROJECT
# scan project and init agent file
xbot chat /init
# real task
xbot chat "find bugs in this project."
```
**Interactive terminal** (rich TUI)

```bash
cd YOUR PROJECT
# you may add .xbot workspace folder to .gitignore
xbot repl
```

## Notes

- Tool calls follow the normal OpenAI request/response loop.
- For Qwen coder models, `--enforce-parser qwen_coder` is usually the most reliable parser setting.
- If xbot reports a model mismatch, compare your configured model against `GET /v1/models`.

## Troubleshooting

Chat logger:

```bash
# Log into files (in folder ./log)
export XINFER_CHAT_LOGGER=1
```

