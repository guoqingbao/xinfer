# xInfer Docker Image

This repository provides a Docker image for **xInfer**, a high-performance inference engine for large language models (LLMs), built using Rust and optimized for NVIDIA GPUs.

---

## Build Options

The image is built with the following features enabled by default:

```text
cuda,nccl,python,flashinfer,cutlass
```

CUDA graph capture is auto-enabled with `cuda`; use `--disable-cuda-graph` at runtime to skip it.
`flashinfer` or `flashattn` increases build time but improves long-context prefill/decoding performance.

For V100, remove `flashattn` or `flashinfer`, and `cutlass`.

For single-GPU machines, you may remove `nccl`.

For SM90+ GPUs, add feature `cutlass` will enable hardware FP8 acceleration.

---

## Docker Build

## Build From Dockerfile

To build this Docker image locally, choose the feature list, compute capability and CUDA version:

Build from script:

```bash
./build_docker.sh "cuda,nccl,flashinfer,python" sm_80 12.9.0
```

Build from command line:

```bash
docker build --network=host -f "Dockerfile" -t "xinfer:latest" \
  --build-arg CUDA_VERSION="12.9.0" \
  --build-arg UBUNTU_VERSION="22.04" \
  --build-arg CUDA_FLAVOR="cudnn-devel" \
  --build-arg WITH_FEATURES="cuda,nccl,flashinfer,cutlass,python" \
  --build-arg CUDA_COMPUTE_CAP="sm_80" \
  # --build-arg CHINA_MIRROR="0" \ Use Rust crate mirror in Chinese mainland
  .
```

## Run xInfer docker service

### xInfer Help:
```bash
docker run --rm -it --gpus all --network host xinfer:latest xinfer --help
```
### Run API server (make sure `--network host`):

```bash
docker run --rm -it --gpus all --network host xinfer:latest xinfer --m Qwen/Qwen3-0.6B --server
```

### Run UI + API Server:
Run interactively:
```bash
docker run --rm -it --gpus all --network host -v /home:/home -v /data:/data xinfer:latest bash
```
Start the UI + API server
```bash
xinfer --w /home/path/Qwen3-Coder-30B-A3B-Instruct-FP8 --ui-server
```

**Note:** if `Ctrl+C` not working in docker, you need `Ctrl+P` then `Ctrl+Q` to stop the server.