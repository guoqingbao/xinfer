# Goose + vLLM.rs (OpenAI-compatible endpoint)

This guide connects Goose (Rust `AI Agent`) directly to vLLM.rs using the built-in OpenAI-compatible `/v1/chat/completions` API. No proxy required.

```
Goose -> vLLM.rs (OpenAI-compatible)
```

## 1) Start vLLM.rs on port 8000

```bash
# Rust
./run.sh --features cuda,nccl,flashinfer,cutlass --release --m Qwen/Qwen3-Coder-Next-FP8 --server --d 0,1
# Python
python3 -m vllm_rs.server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --server
```

## 2) Configure Goose

```shell
# For non-UI system,
export GOOSE_DISABLE_KEYRING=1
```
Export empty API KEY

```shell
export VLLM_API_KEY="empty"
```

### Download and install Goose: https://block.github.io/goose/docs/getting-started/installation/

### Configure goose with `Custom Providers` and API key `empty`

```shell
goose configure

┌   goose-configure
│
◇  What would you like to configure?
│  Custom Providers
│
◇  What would you like to do?
│  Add A Custom Provider
│
◇  What type of API is this?
│  OpenAI Compatible
│
◇  What should we call this provider?
│  vllm-rs
│
◇  Provider API URL:
│  http://127.0.0.1:8000/v1/
│
◇  API key:
│  ▪▪▪▪▪
│
◇  Available models (separate with commas):
│  default
│
◇  Does this provider support streaming responses?
│  Yes
│
◇  Does this provider require custom headers?
│  No
│
└  Custom provider added: vllm-rs
└  Configuration saved successfully to /root/.config/goose/config.yaml
```

### Run `goose` at any folder to work with your local server

```shell
goose
```

### Trouble shooting

1. Use the chat logger to monitor detailed interactions between Goose and vLLM.rs.

```shell
# Log into files (in folder ./log)
export VLLM_RS_CHAT_LOGGER=1
```