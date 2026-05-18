# Multimodal Model Usage

This project supports vision-language models (Qwen3-VL dense/MoE, Gemma3, Mistral3-VL). The server exposes `/v1/chat/completions` with mixed text+image content and optional web UI.

## Starting servers
- Qwen3-VL (CUDA):  
  ```bash
  target/release/vllm-rs --m Qwen/Qwen3-VL-8B-Instruct --ui-server
  ```
- Gemma3 (vision, No Flash attention support):  
  ```bash
  ./run.sh --release --features cuda -- \
    --m google/gemma-3-4b-it --ui-server
  ```
- Mistral3-VL (vision):  
  ```bash
  target/release/vllm-rs --m mistralai/Ministral-3-8B-Reasoning --ui-server
  ```

## Request payloads (OpenAI-compatible)
- Text + image URL:  
  ```json
  {
    "model": "default",
    "messages": [
      {"role":"user","content":[
        {"type":"text","text":"Describe this image"},
        {"type":"image_url","image_url":"https://example.com/cat.png"}
      ]}
    ]
  }
  ```
- Text + base64 image: `{"type":"image_base64","image_base64":"data:image/png;base64,..."}`

## Tips
- Use smaller `--max-model-len` on Metal if VRAM is tight; consider `--kv-fraction` on CUDA to reserve cache.
- For batch image inputs, keep concurrent images modest; too many images will increase prefill time.
- `--ui-server` opens the built-in chat UI for uploads; without it, send HTTP requests directly.***
