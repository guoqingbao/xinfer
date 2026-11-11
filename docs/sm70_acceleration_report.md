# SM70 acceleration report

Modern quantized checkpoints rely on hardware paths that only exist from
Hopper (SM90) and Blackwell (SM100) onwards. On Volta (SM70 / V100) the same
checkpoints fall back to software dequantization or reference recurrences.
That fallback is usually **slower** than a plain F16 model while saving only
memory, and it happens silently: nothing in the startup log says which path
was taken.

This document describes the acceleration report added to xinfer, which makes
that state observable.

## What it reports

At model load, xinfer evaluates each acceleration path and records whether it
is active:

| Path | Meaning when enabled |
|---|---|
| `nvfp4_hardware_gemm` | NVFP4 / MXFP4 weights use hardware FP4 tensor cores (SM100+) |
| `flashinfer_attention` | FlashInfer prefill/decode kernels are in use |
| `cutlass_kernels` | The loaded model dispatches to CUTLASS kernels |
| `cuda_graph` | CUDA graph capture succeeded |
| `gdn_fast_prefill` | Linear-attention (GDN) prefill uses the chunked kernel |

A disabled path always carries a reason, for example:

```
Acceleration report:
Device: SM70 | architecture: Qwen3_5ForConditionalGeneration | nvfp4 weights: true
  [ok]    cuda_graph
  [slow]  nvfp4_hardware_gemm — SM70 has no FP4 tensor cores; 4-bit weights are expanded to F16 at runtime (software dequantization), which is slower than a plain F16 GEMM and only saves memory
  [slow]  flashinfer_attention — the GQA group size is not supported by the FlashInfer kernels
  [slow]  gdn_fast_prefill — the chunked GDN prefill kernel requires SM90
```

## Why the NVFP4 case is called out explicitly

On SM70 the NVFP4 weight format is a net loss:

| | Weight storage | GEMM work |
|---|---|---|
| F16 weights, F16 GEMM | 100% | 1x |
| NVFP4 weights, SM70 software dequant | 25% | **> 1x** (dequantization is added on top) |

The format only pays off in memory. If a deployment is throughput-bound, an
AWQ or GPTQ checkpoint of the same model will usually be faster, because those
formats have SM70-optimized kernels (Marlin) while NVFP4 does not.

The report therefore emits a warning when an NVFP4 checkpoint runs on a device
without hardware FP4 support.

## HTTP endpoint

```
GET /v1/acceleration
```

Returns the same information as JSON:

```json
{
  "compute_capability": 70,
  "architecture": "Qwen3_5ForConditionalGeneration",
  "uses_nvfp4_weights": true,
  "entries": [
    { "name": "cuda_graph", "enabled": true, "reason": null },
    {
      "name": "nvfp4_hardware_gemm",
      "enabled": false,
      "reason": "SM70 has no FP4 tensor cores; ..."
    }
  ]
}
```

The endpoint uses the same authentication as the rest of the API.

## Failing fast on a degraded configuration

Set `XINFER_REQUIRE_ACCELERATION=1` to refuse to start when the NVFP4
hardware path is unavailable. This turns a silently slow deployment into an
immediate, explicit failure at startup:

```
Error: XINFER_REQUIRE_ACCELERATION is set but the NVFP4 hardware GEMM path
is unavailable on SM70; refusing to start a degraded configuration
```

This is intended for deployments where throughput is a hard requirement and a
fallback to software dequantization would violate the service contract.

## Design notes

### Single source of truth

The runner process owns the authoritative runtime decisions (which backend was
actually initialized, whether the GQA group size is supported, and so on),
but it lives in a separate process and cannot hand a report back to the
engine over the existing IPC.

Rather than duplicating the decision logic, `ReportInputs::from_engine` derives
the facts from the engine configuration and model config in one place, and the
runner overrides only the FlashInfer decision with its own authoritative
result. Both paths therefore produce the same report.

### Conservative CUDA graph reporting

`cuda_graph` is reported as active when the graph feature is compiled in and
the user did not disable it. The engine does not wait for the capture to
finish before serving requests, so it cannot observe the capture result
directly. A capture failure is logged separately by the runner.

## Related work

The acceleration surface follows the approach of
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a V100-focused vLLM
fork that exposes an equivalent `/v1/sm70/acceleration` endpoint and can
refuse to start when an expected path is disabled
(`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION=1`).
