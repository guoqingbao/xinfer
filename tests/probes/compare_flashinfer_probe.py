"""Compare attention.rs probe files with independent PyTorch and FlashInfer goldens."""

from __future__ import annotations

import argparse
import pathlib
import struct

import torch
import flashinfer
from precision_metrics import report


def read_tensor(f):
    n = struct.unpack("<Q", f.read(8))[0]
    rank = struct.unpack("<Q", f.read(8))[0]
    shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(rank))
    data = torch.frombuffer(bytearray(f.read(4 * n)), dtype=torch.float32).clone()
    return data.reshape(shape)


def read_probe(path: pathlib.Path):
    with path.open("rb") as f:
        if f.read(10) != b"XINFPROBE1":
            raise ValueError(f"bad probe magic: {path}")
        prefix, append, hq, hkv, dim, page, pages = struct.unpack("<7Q", f.read(56))
        index_count = struct.unpack("<Q", f.read(8))[0]
        indices = list(struct.unpack(f"<{index_count}I", f.read(4 * index_count)))
        q, k, v, rust = (read_tensor(f) for _ in range(4))
    return {
        "prefix": prefix,
        "append": append,
        "hq": hq,
        "hkv": hkv,
        "dim": dim,
        "page": page,
        "pages": pages,
        "indices": indices,
        "q": q,
        "k": k,
        "v": v,
        "rust": rust,
        "fp8": path.name.startswith("fp8_"),
    }


def dense_fp32(case, device):
    q = case["q"].to(device=device, dtype=torch.bfloat16).float()
    k = case["k"].to(device=device, dtype=torch.bfloat16)
    v = case["v"].to(device=device, dtype=torch.bfloat16)
    if case["fp8"]:
        k = k.to(torch.float8_e4m3fn).float()
        v = v.to(torch.float8_e4m3fn).float()
    else:
        k = k.float()
        v = v.float()
    prefix = case["prefix"]
    hq, hkv = case["hq"], case["hkv"]
    group = hq // hkv
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1).unsqueeze(1).repeat(1, group, 1, 1)
    kh = kh.reshape(hq, k.shape[0], k.shape[2])
    vh = v.transpose(0, 1).unsqueeze(1).repeat(1, group, 1, 1)
    vh = vh.reshape(hq, v.shape[0], v.shape[2])
    scores = torch.matmul(qh, kh.transpose(-1, -2)) / case["dim"] ** 0.5
    qpos = torch.arange(q.shape[0], device=device) + prefix
    kpos = torch.arange(k.shape[0], device=device)
    scores = scores.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(0, 1)


def flashinfer_gold(case, device, backend="auto"):
    dtype = torch.bfloat16
    cache_dtype = torch.float8_e4m3fn if case["fp8"] else dtype
    q = case["q"].to(device=device, dtype=dtype)
    k = case["k"].to(device=device, dtype=dtype)
    v = case["v"].to(device=device, dtype=dtype)
    prefix, append, page = case["prefix"], case["append"], case["page"]
    hq, hkv, dim = case["hq"], case["hkv"], case["dim"]
    n_pages = case["pages"]
    indices = torch.tensor(case["indices"], device=device, dtype=torch.int32)
    indptr = torch.tensor([0, n_pages], device=device, dtype=torch.int32)
    last = torch.tensor([(prefix + append - 1) % page + 1], device=device, dtype=torch.int32)
    cache_k = torch.zeros((n_pages + 3, page, hkv, dim), device=device, dtype=cache_dtype)
    cache_v = torch.zeros_like(cache_k)
    for p in range((prefix + page - 1) // page):
        start = p * page
        length = min(prefix - start, page)
        physical = case["indices"][p]
        cache_k[physical, :length] = k[start : start + length].to(cache_dtype)
        cache_v[physical, :length] = v[start : start + length].to(cache_dtype)
    batch_indices = torch.zeros((append,), device=device, dtype=torch.int32)
    positions = torch.arange(prefix, prefix + append, device=device, dtype=torch.int32)
    flashinfer.append_paged_kv_cache(
        k[prefix:].to(cache_dtype), v[prefix:].to(cache_dtype), batch_indices, positions,
        (cache_k, cache_v), indices, indptr, last, kv_layout="NHD"
    )
    logical_k = torch.cat(
        [cache_k[physical, : page if p + 1 < n_pages else int(last[0])] for p, physical in enumerate(case["indices"])],
        dim=0,
    )
    logical_v = torch.cat(
        [cache_v[physical, : page if p + 1 < n_pages else int(last[0])] for p, physical in enumerate(case["indices"])],
        dim=0,
    )
    cache_error = max(
        (logical_k.float() - k.float()).abs().max().item(),
        (logical_v.float() - v.float()).abs().max().item(),
    )
    workspace = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend=backend
    )
    qo = torch.tensor([0, append], device=device, dtype=torch.int32)
    wrapper.plan(
        qo, indptr, indices, last, hq, hkv, dim, page,
        causal=True, q_data_type=dtype, kv_data_type=cache_dtype,
        o_data_type=dtype, sm_scale=dim ** -0.5,
    )
    if case["fp8"]:
        output = wrapper.run(
            q, (cache_k, cache_v), q_scale=1.0, k_scale=1.0, v_scale=1.0
        ).float()
    else:
        output = wrapper.run(q, (cache_k, cache_v)).float()
    return output, cache_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    print("torch", torch.__version__, "flashinfer", flashinfer.__version__)
    for package in ("vllm", "sglang"):
        try:
            module = __import__(package)
            print(package, "loaded", getattr(module, "__file__", ""), getattr(module, "__version__", ""))
        except Exception as exc:
            print(package, "unavailable:", repr(exc))
    failures = 0
    for path in sorted(args.directory.glob("*.bin")):
        case = read_probe(path)
        ref = dense_fp32(case, device)
        rust = case["rust"].to(device=device)
        fi, cache_error = flashinfer_gold(case, device, backend="auto")
        prefix = path.name
        failures += not report(
            prefix + " rust_vs_torch", rust, ref,
            dtype=torch.bfloat16, allowed_ulp=None, abs_tol=0.015625,
        )
        failures += not report(
            prefix + " rust_vs_flashinfer", rust, fi,
            dtype=torch.bfloat16, allowed_ulp=None, abs_tol=0.001,
        )
        failures += not report(
            prefix + " flashinfer_vs_torch", fi, ref,
            dtype=torch.bfloat16, allowed_ulp=None, abs_tol=0.015625,
        )
        print(f"{prefix} cache_roundtrip_max={cache_error:.8g}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
