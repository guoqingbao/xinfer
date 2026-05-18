# Embedding Usage

This repository now exposes OpenAI-style embeddings for text-only models (Qwen3, Qwen3-MoE, LLaMa, GLM4, Gemma3). Use the standard server run path and hit `/v1/embeddings`.

## Start the server (embeddings enabled)
- CUDA example (Qwen3 text):  
  ```bash
  target/release/vllm-rs --server --m Qwen/Qwen2.5-7B-Instruct
  ```
- Metal example (LLaMa3 text):  
  ```bash
  target/release/vllm-rs --server --m meta-llama/Llama-3-8b --max-model-len 32768
  ```

## Request examples
- Float embeddings (default) with mean pooling:  
  ```bash
  curl -X POST http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"input":"hello world","model":"default","embedding_type":"mean"}'
  ```
- Base64-encoded embeddings with last-token pooling:  
  ```bash
  curl -X POST http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"input":["hello","hola"],"embedding_type":"last","encoding_format":"base64"}'
  ```

## Notes
- `model` defaults to the loaded model id; multiple models per request are not supported.
- Uses existing tokenizer; long prompts must fit `max_model_len` (same as chat).
- `embedding_type`: `mean` (default) averages tokens; `last` returns the final token hidden state.
- Responses mirror OpenAI schema: `data[].embedding`, `usage.prompt_tokens`.
