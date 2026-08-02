# CUDA Precision Probe Suite

The precision suite is a kernel-level differential test harness. Use it to isolate a CUDA precision problem before attributing the symptom to a full model or sampler.

## Run the complete suite

From the xInfer repository root:

```bash
./tests/run_precision_suite.sh
```

The runner:

1. Detects every visible GPU's compute capability with `nvidia-smi` and uses the lowest capability for a safe multi-GPU build.
2. Builds all Rust probes once with the release CUDA configuration.
3. Runs each Rust probe followed by its paired Python comparator.

The probes cover Candle CUDA/PTX operations, ISQ/GGUF Q4_K and Q6_K, attention.rs kernels, FP8, GDN, paged FlashInfer attention, and the NVFP4 paths supported by the detected GPU. Python comparisons use independent PyTorch references and official Python FlashInfer when available. vLLM and sglang comparisons are attempted when those packages are installed.

The suite prints a retained result directory such as:

```text
/tmp/xinfer_precision_suite_sm120_<pid>
```

Read the first `FAIL` line and the corresponding stage's `[FAIL]` marker. The failing stage identifies the subsystem; the operation name identifies the kernel family or operation.

## Rerun one comparator

Probe binaries and their output files are retained, so a Python comparator can be rerun without rebuilding or executing the GPU probe again:

```bash
python3 tests/probes/compare_candle_probe.py \
  /tmp/xinfer_precision_suite_sm120_<pid>/candle.bin

python3 tests/probes/compare_flashinfer_probe.py \
  /tmp/xinfer_precision_suite_sm120_<pid>/flashinfer
```

Use the corresponding comparator in `tests/probes/` for the failed stage:

| Stage | Comparator |
|---|---|
| Candle and Q4_K/Q6_K | `compare_candle_probe.py` |
| attention.rs common kernels | `compare_attention_misc_probe.py` |
| FP8 | `compare_fp8_probe.py` |
| GDN | `compare_gdn_probe.py` |
| FlashInfer attention | `compare_flashinfer_probe.py` |
| NVFP4 | `compare_nvfp4_probe.py` |

## Optional GDN audit

The SM90 persistent FlashInfer GQA path is opt-in and excluded by default. Audit it explicitly with:

```bash
./tests/run_precision_suite.sh --include-optin-gdn
```

This flag is only for auditing that optional path; it does not enable or change the production routing.

## Interpreting SM capability results

For SM70 and SM75, the runner automatically builds the legacy CUDA path with only `cuda,nccl`. FlashInfer and CUTLASS-dependent attention and hardware-NVFP4 cases are skipped, but the software FP8 and software NVFP4 GEMM/MoE/helper cases still run and are compared with the independent PyTorch golden. Since attention.rs disables BF16 CUDA kernels below SM80, the legacy software probes use F16 there. The CUDA toolkit must also provide an `nvcc` that supports `compute_70`/`compute_75`; CUDA 12.6 is the recommended legacy-toolkit choice. CUDA 13 `nvcc` rejects these targets before Rust compilation begins.

On SM90, Blackwell-only NVFP4 hardware helpers are reported as `SKIP` because they cannot execute on that GPU. Run the suite on SM100/SM120 hardware to exercise hardware NVFP4 dense, GEMM, and MoE paths. Missing optional Python packages are reported explicitly; install versions matching the CUDA and PyTorch environment when those golden comparisons are required.
