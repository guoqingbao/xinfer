# syntax = devthefuture/dockerfile-x

ARG CUDA_VERSION=12.9.0
ARG UBUNTU_VERSION=22.04
ARG CUDA_FLAVOR=cudnn-devel

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} AS base

ARG DEBIAN_FRONTEND=noninteractive

ARG CHINA_MIRROR=0

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends --allow-change-held-packages \
    libnccl-dev=$(apt-cache madison libnccl-dev | awk -v cuda="$(echo "$CUDA_VERSION" | cut -d'.' -f1,2)" '$0 ~ cuda {print $3; exit}') \
    libnccl2=$(apt-cache madison libnccl2 | awk -v cuda="$(echo "$CUDA_VERSION" | cut -d'.' -f1,2)" '$0 ~ cuda {print $3; exit}') \
    curl git ca-certificates \
    libssl-dev pkg-config \
    clang libclang-dev \
    python3-pip && \
  rm -rf /var/lib/apt/lists/*

RUN set -eux; \
  if [ "${CHINA_MIRROR}" = "1" ]; then \
    export RUSTUP_UPDATE_ROOT="https://mirrors.ustc.edu.cn/rust-static/rustup"; \
    export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"; \
  fi; \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
  if [ "${CHINA_MIRROR}" = "1" ]; then \
    mkdir -p /root/.cargo; \
    echo "RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static" >> /root/.cargo/env; \
    printf '%s\n' \
'[source.crates-io]' \
'replace-with = "ustc"' \
'' \
'[source.ustc]' \
'registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' \
'' \
'[registries.ustc]' \
'index = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' \
> /root/.cargo/config.toml; \
  fi


ENV PATH="/root/.cargo/bin:${PATH}"

ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=32
ENV CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS}"

ARG BUILD_FEATURES
ARG WITH_FEATURES="cuda,nccl,python,flashinfer,cutlass"

WORKDIR /xinfer
COPY . .

RUN set -eux; \
  FEATURES="${BUILD_FEATURES:-$WITH_FEATURES}"; \
  if echo "${FEATURES}" | grep -q '\bpython\b'; then \
    echo "Python feature detected: adding deps and executing build.sh" && \
    pip3 install --no-cache-dir maturin patchelf cffi; \
  else \
    echo "Python feature absent: executing build.sh for rust artifacts only"; \
  fi; \
  ./build.sh --release --features "${FEATURES}"

RUN set -eux; \
  FEATURES="${BUILD_FEATURES:-$WITH_FEATURES}"; \
  install -Dm755 target/release/xinfer /usr/local/bin/xinfer; \
  if echo "${FEATURES}" | grep -q '\bpython\b'; then \
    install -Dm755 target/release/libxinfer.so /usr/lib64/libxinfer.so; \
    pip3 install --no-cache-dir target/wheels/*; \
    printf '%s\n' '#!/bin/sh' 'exec python3 -m xinfer.server "$@"' > /usr/local/bin/xinfer-server; \
    chmod +x /usr/local/bin/xinfer-server; \
    cp -r target/wheels/ /opt/wheels; \
  else \
    mkdir /opt/wheels; \
    if [ ! -d /usr/lib64/ ]; then mkdir /usr/lib64 ; fi ; \
    touch /usr/lib64/libxinfer.so; \
  fi; \
  cargo clean

RUN set -eux; \
  arch="$(uname -m)"; \
  libdir="/usr/lib/${arch}-linux-gnu"; \
  if [ ! -e "${libdir}/libnccl.so" ] && [ -e "${libdir}/libnccl.so.2" ]; then \
    ln -s libnccl.so.2 "${libdir}/libnccl.so"; \
  fi

ENV HUGGINGFACE_HUB_CACHE=/data PORT=80
EXPOSE 80
CMD ["bash"]
