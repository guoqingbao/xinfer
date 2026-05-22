# Performance Benchmarks

This document contains detailed performance benchmarks for xInfer across different hardware platforms.

## 🚀 CUDA Performance (A100 40GB)

### Single Request Decoding Speed

| Model | Format | Size | Decoding Speed |
|-------|--------|------|----------------|
| Llama-3.1-8B | ISQ (BF16→Q4K) | 8B | **90.19** tokens/s |
| DeepSeek-R1-Distill-Llama-8B | Q2_K | 8B | **94.47** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **95** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **35.69** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)** | **75.91** tokens/s |

## 🍎 Metal Performance (Apple Silicon M4)

> Test Configuration:
> - Models: Qwen3-0.6B (BF16), Qwen3-4B (Q4_K_M), Qwen3-8B (Q2_K)
> - Concurrent Requests: 1 - 128
> - Max Model Length: 512 - 2048
> - Max Output Tokens/Request: 512 - 2048

| Model | Batch Size | Output Tokens | Time (s) | Throughput (tokens/s) |
|-------|------------|---------------|----------|----------------------|
| Qwen3-0.6B (BF16) | 128 | 63,488 | 83.13s | **763.73** |
| Qwen3-0.6B (BF16) | 32 | 15,872 | 23.53s | **674.43** |
| Qwen3-0.6B (BF16) | 1 | 456 | 9.23s | 49.42 |
| Qwen3-4B (Q4_K_M) | 1 | 1,683 | 52.62s | 31.98 |
| Qwen3-8B (Q2_K) | 1 | 1,300 | 80.88s | 16.07 |

## 📊 Performance Comparison

> Test Configuration:
> - Model: Qwen3-0.6B (BF16)
> - Concurrent Requests: 256
> - Max Model Length: 1024
> - Max Output Tokens/Request: 1024

| Inference Engine | Hardware | Tokens | Time (s) | Throughput (tokens/s) |
|------------------|----------|--------|----------|----------------------|
| vLLM (Reference) | RTX 4070 | 133,966 | 98.37 | 1,361.84 |
| Nano-vLLM (Reference) | RTX 4070 | 133,966 | 93.41 | 1,434.13 |
| **xInfer** | **A100** | 262,144 | 23.88 | **10,977.55** |
| Nano-vLLM | A100 | 262,144 | 34.22 | 7,660.26 |

### Key Insights

- **40%+ faster** than Nano-vLLM on A100
- **7x faster** than reference implementations on consumer hardware
- Efficient memory management with quantized models

## 🔧 Reproduce Benchmarks

See [python/ReadMe.md](../python/ReadMe.md) for reproducible benchmark steps.

## Optimization Tips

1. **Use KV Cache Quantization** (`--kvcache-dtype fp8|turbo4|turbo3`) for memory efficiency — turbo4 gives 3.7x compression with good quality
2. **Enable Flashinfer / Flash Attention** (`flashinfer` or `flashattn` feature) for maximum CUDA performance
3. **Prefix Cache** is enabled by default for multi-turn conversations
4. **Tune `--kv-fraction`** to balance memory usage and batch size
5. **Use PD Disaggregation** for long-context workloads to prevent decoding stalls
