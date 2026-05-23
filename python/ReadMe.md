#### How to reproduce?
**xInfer**
```shell
pip install xinfer --index-url https://guoqingbao.github.io/xinfer/sm80/
python -m xinfer.completion --w /home/Qwen3-0.6B/ --batch 256 --max-tokens 1024 --max-model-len 1024

# Log
Allocating 8192 KV blocks (28672 MB) for [256 seqs x 1024 tokens]
Maximum batched tokens 262144 (8192 blocks x Block_Size 32 for KV cache).
Start inference with 256 prompts
--- Performance Metrics ---
Prompt tokens: 4096 in 0.28s (14894.55 tokens/s)
Decoded tokens: 258048 in 23.60s (10944.62 tokens/s)
```

---

### Python API

```python
from xinfer import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(weight_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
message = Message("user", "How are you?")

# Synchronous batch generation
outputs = engine.generate_sync([params, params], [[message], [message]])
print(outputs)

params.session_id = xxx  # Optional: track sessions in your own client

# Single-request streaming generation
(seq_id, prompt_length, stream) = engine.generate_stream(params, [message])
for item in stream:
   # item.datatype == "TOKEN"
   print(item.data)
```

### Client Usage of Prefix Cache

**Key changes for the client:**

```python
import openai

# xInfer service url
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

response = openai.chat.completions.create(
   model="",
   messages=messages + [user_msg],
   stream=True,
   max_tokens = max_tokens,
   temperature = temperature,
   top_p = top_p,
)
```

### Interactive Chat and Batch Processing

> Interactive Chat

```bash
# Prefix cache automatically enabled under chat mode
python3 -m xinfer.chat --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

```bash
python3 -m xinfer.chat --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1
```

```bash
# Enable maximum context (262144 tokens), two ranks (--d 0,1)
python3 -m xinfer.chat --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 262144
```

> Batch Processing

```bash
python3 -m xinfer.completion --f /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

```bash
python3 -m xinfer.completion --w /home/GLM-4-9B-0414 --d 0,1 --batch 8 --max-model-len 1024 --max-tokens 1024
```

### MCP Multi-Server Demo (Python Client)

Start xInfer with an MCP config file:

```shell
xinfer --m <model_id> --server --mcp-config ./mcp.json
```

Then call a prefixed MCP tool from Python:

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

response = openai.chat.completions.create(
   model="",
   messages=[{"role": "user", "content": "Use filesystem_read_file to read README.md"}],
)
print(response.choices[0].message)
```
