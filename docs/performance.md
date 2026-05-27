# Performance Benchmarks

This document contains detailed performance benchmarks for xInfer across different hardware platforms.

## 🚀 CUDA Performance

> Tested on **V100-32G**, **A100-40G**, **Hopper-80G** and **RTX 5090**

### Single Request Decoding Speed

| Model | Format | Size | Hardware | Decoding Speed |
|-------|--------|------|----------|----------------|
| Qwen3-30B-A3B | NVFP4 | 30B MoE | RTX 5090 (SM120) | **175.30** tokens/s |
| Gemma4-26B-A4B | NVFP4 | 26B MoE | RTX 5090 (SM120) | **131** tokens/s |
| Ministral-3-3B (Multimodal) | ISQ (BF16→Q4K) | 3B | A100 (SM80) | **171.92** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | A100 (SM80) | **124.87** tokens/s |
| Llama-3.1-8B | ISQ (BF16→Q4K) | 8B | A100 (SM80) | **120.74** tokens/s |
| Qwen3-VL-8B-Instruct (Multimodal) | Q8_0 | 8B | A100 (SM80) | **105.31** tokens/s |
| Qwen3.6-35B-A3B (Multimodal) | FP8 | 35B MoE | H800 (SM90) | **102** tokens/s |
| GLM4.7 Flash | NVFP4 | 30B MoE | H800 (SM90) | **79** tokens/s |
| Qwen3-30B-A3B | NVFP4 | 30B MoE | V100 (SM70) | **67.10** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | A100 (SM80) | **70.38** tokens/s |
| MiniMax-M2.5 | NVFP4 | 229B MoE | H800 ×2 (SM90) | **62** tokens/s |
| Qwen3.5-27B (Multimodal) | Q4_K_M | 27B Dense | A100 (SM80) | **45.20** tokens/s |
| Qwen3.5-27B/Qwen3.6-27B | FP8 | 27B Dense | H800 (SM90) | **42** tokens/s |
| QwQ-32B | Q4_K_M | 32B | A100 (SM80) | **41.36** tokens/s |
| Gemma4-31B | ISQ (BF16→Q4K) | 31B Dense | H800 (SM90) | **41** tokens/s |

### V100 + NVFP4 + TurboQuant (First-Ever)

NVFP4 models running under low-bit KV cache on V100 (SM70) — no hardware FP4 support needed.

```bash
xinfer --m AxionML/Qwen3.5-4B-NVFP4 --ui-server
xinfer --m AxionML/Qwen3.5-4B-NVFP4 --kvcache-dtype turbo4 --ui-server
```

## 🍎 Metal Performance (Apple Silicon M4)

| Model | Batch Size | Output Tokens | Time (s) | Throughput (tokens/s) |
|-------|------------|---------------|----------|----------------------|
| Qwen3-0.6B (BF16) | 128 | 63,488 | 83.13s | **763.73** |
| Qwen3-0.6B (BF16) | 32 | 15,872 | 23.53s | **674.43** |
| Qwen3-0.6B (BF16) | 1 | 456 | 9.23s | 49.42 |
| Qwen3-4B (Q4_K_M) | 1 | 1,683 | 52.62s | 31.98 |
| Qwen3-8B (Q2_K) | 1 | 1,300 | 80.88s | 16.07 |
| Qwen3.5-4B (Q3_K_M) | 1 | 1,592 | 69.04s | 23.06 |
| Qwen3.5-2B (NVFP4) | 1 | 1,883 | 60.76s | 30.99 |
| Qwen3.5-2B (NVFP4) | 2 | 3,942 | 81.96s | 48.10 |

## 📊 Performance Notes

- **HW NVFP4**: Hardware-accelerated FP4 on Blackwell (SM120, RTX 5090/B200)
- **HW FP8**: Hardware-accelerated FP8 on Hopper (SM90, H800/H200)
- **SW FP4/NVFP4**: Software-emulated FP4 on Hopper (SM90) and V100 (SM70)
- V100 + NVFP4 + TurboQuant is a first-ever combination — no other engine has achieved this

## 🔧 Optimization Tips

1. **Use KV Cache Quantization** (`--kvcache-dtype fp8|turbo8|turbo4|turbo3`) for memory efficiency — turbo4 gives 3.7× compression with good quality
2. **Enable FlashInfer** (`flashinfer` feature, SM80+) for maximum CUDA performance
3. **Prefix Cache** is enabled by default for multi-turn conversations
4. **Tune `--kv-fraction`** to balance memory usage and batch size
5. **Use PD Disaggregation** for long-context workloads to prevent decoding stalls
6. **NVFP4 on V100** — works with TurboQuant KV cache for extended context on legacy hardware

## 🔧 Reproduce Benchmarks

See [python/ReadMe.md](../python/ReadMe.md) for reproducible benchmark steps.
