"""Independent PyTorch goldens for attention.rs common CUDA kernels."""

from __future__ import annotations

import pathlib
import struct
import sys

import torch
import torch.nn.functional as F
from precision_metrics import report


def read_probe(path):
    out = {}
    with pathlib.Path(path).open("rb") as f:
        if f.read(9) != b"XINFMISC1":
            raise ValueError("bad misc probe magic")
        while True:
            raw = f.read(8)
            if not raw:
                break
            n = struct.unpack("<Q", raw)[0]
            name = f.read(n).decode()
            rank = struct.unpack("<Q", f.read(8))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(rank))
            count = struct.unpack("<Q", f.read(8))[0]
            data = torch.frombuffer(bytearray(f.read(4 * count)), dtype=torch.float32).clone()
            out[name] = data.reshape(shape)
    return out


def rope(q, k, cos, sin, positions, interleaved):
    q = q.float().clone()
    k = k.float().clone()
    c = cos[positions.long()].float()
    s = sin[positions.long()].float()
    if interleaved:
        qx, qy = q[..., 0::2], q[..., 1::2]
        kx, ky = k[..., 0::2], k[..., 1::2]
        q[..., 0::2], q[..., 1::2] = qx * c[:, None, :] - qy * s[:, None, :], qx * s[:, None, :] + qy * c[:, None, :]
        k[..., 0::2], k[..., 1::2] = kx * c[:, None, :] - ky * s[:, None, :], kx * s[:, None, :] + ky * c[:, None, :]
    else:
        h = q.shape[-1] // 2
        qx, qy = q[..., :h], q[..., h:]
        kx, ky = k[..., :h], k[..., h:]
        q[..., :h], q[..., h:] = qx * c[:, None, :] - qy * s[:, None, :], qx * s[:, None, :] + qy * c[:, None, :]
        k[..., :h], k[..., h:] = kx * c[:, None, :] - ky * s[:, None, :], kx * s[:, None, :] + ky * c[:, None, :]
    return q, k


def compare(name, got, gold, tol, failures):
    if "indices" in name:
        ok = torch.equal(got, gold)
        print(f"{'PASS_EXACT' if ok else 'FAIL':9} {name:34} exact={ok}")
        failures[0] += not ok
        return
    if name.endswith("_f16"):
        ok = report(name, got, gold, dtype=torch.float16, allowed_ulp=1, max_rel=1e-6)
    elif name.startswith(("rope_", "silu_")):
        ok = report(name, got, gold, dtype=torch.bfloat16, allowed_ulp=1, max_rel=1e-6)
    else:
        ok = report(name, got, gold, dtype=torch.float32, allowed_ulp=4, max_rel=1e-6)
    if not ok:
        failures[0] += 1


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    r = read_probe(sys.argv[1] if len(sys.argv) > 1 else "/tmp/xinfer_attention_misc_probe.bin")
    d = torch.device("cuda:0")
    failures = [0]
    for tag, interleaved in (("rope_noninterleaved", False), ("rope_interleaved", True)):
        q, k = r[f"{tag}_q"].to(d, dtype=torch.bfloat16), r[f"{tag}_k"].to(d, dtype=torch.bfloat16)
        c, s = r[f"{tag}_cos"].to(d, dtype=torch.bfloat16), r[f"{tag}_sin"].to(d, dtype=torch.bfloat16)
        pos = r[f"{tag}_positions"].to(d)
        eq, ek = rope(q, k, c, s, pos, interleaved)
        compare(f"{tag}_q", r[f"{tag}_q_out"].to(d), eq.to(torch.bfloat16), 0, failures)
        compare(f"{tag}_k", r[f"{tag}_k_out"].to(d), ek.to(torch.bfloat16), 0, failures)

    gate_up = r["silu_gate_up"].to(d, dtype=torch.bfloat16)
    n = gate_up.shape[-1] // 2
    expected = (F.silu(gate_up[..., :n].float()) * gate_up[..., n:].float()).to(torch.bfloat16)
    compare("silu_and_mul", r["silu_out"].to(d), expected, 0, failures)
    gate_up = r["silu_gate_up_f16"].to(d, dtype=torch.float16)
    expected = (F.silu(gate_up[..., :n].float()) * gate_up[..., n:].float()).to(torch.float16)
    compare("silu_and_mul_f16", r["silu_out_f16"].to(d), expected, 0, failures)

    logits = r["logits"].to(d)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, 5, dim=-1)
    compare("topk_softmax_weights", r["softmax_weights"].to(d), weights, 0, failures)
    compare("topk_softmax_indices", r["softmax_indices"].to(d), indices.float(), 0, failures)

    scores = r["scores"].to(d)
    weights, indices = torch.topk(scores, 5, dim=-1)
    compare("topk_select_weights", r["select_weights"].to(d), weights, 0, failures)
    compare("topk_select_indices", r["select_indices"].to(d), indices.float(), 0, failures)

    logits = r["sigmoid_logits"].to(d)
    bias = r["sigmoid_bias"].to(d)
    raw = torch.sigmoid(logits)
    weights, indices = torch.topk(raw + bias, 5, dim=-1)
    expected_weights = raw.gather(-1, indices)
    compare("fused_sigmoid_topk_weights", r["sigmoid_weights"].to(d), expected_weights, 0, failures)
    compare("fused_sigmoid_topk_indices", r["sigmoid_indices"].to(d), indices.float(), 0, failures)
    raise SystemExit(1 if failures[0] else 0)


if __name__ == "__main__":
    main()
