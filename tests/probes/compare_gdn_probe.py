"""Compare attention.rs GDN CUDA kernels with independent PyTorch formulas."""

from __future__ import annotations

import pathlib
import struct
import sys

import torch
import torch.nn.functional as F
from precision_metrics import report


def read_probe(path: pathlib.Path):
    records = {}
    with path.open("rb") as f:
        if f.read(8) != b"XINFGDN1":
            raise ValueError("bad GDN probe magic")
        while True:
            raw = f.read(8)
            if not raw:
                break
            name_len = struct.unpack("<Q", raw)[0]
            name = f.read(name_len).decode()
            rank = struct.unpack("<Q", f.read(8))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(rank))
            n = struct.unpack("<Q", f.read(8))[0]
            data = torch.frombuffer(bytearray(f.read(4 * n)), dtype=torch.float32).clone()
            records[name] = data.reshape(shape)
    return records


def bf16(x):
    return x.to(dtype=torch.bfloat16)


def compare(name, got, gold, tol, failures):
    if "flashinfer" in name:
        # This optional persistent SM90 path is being audited separately; it
        # must match the regular recurrence, not merely stay below 0.02.
        ok = report(name, got, gold, dtype=torch.bfloat16 if "out" in name else torch.float32,
                    allowed_ulp=0, max_rel=1e-6, abs_tol=1e-5)
    elif name.startswith(("gdn_fused",)):
        ok = report(name, got, gold, dtype=torch.float32, allowed_ulp=None,
                    max_rel=2e-6, abs_tol=2e-7)
    elif name.endswith(("_state", "_snapshots")):
        # Recurrent state is FP32, while q/k/v are BF16. The CUDA kernels use
        # fixed serial/warp reductions; PyTorch's reference uses GEMV
        # reductions with a different accumulation tree. Compare the actual
        # FP32 numerical error, not the meaningless ULP distance near zero.
        # This remains strict enough to catch state corruption or skipped
        # updates while allowing the expected reduction-order error.
        ok = report(name, got, gold, dtype=torch.float32, allowed_ulp=None,
                    max_rel=3e-5, abs_tol=5e-4, rel_floor=0.1)
    elif name.endswith("_out") or "rmsnorm" in name or name == "gdn_l2_norm":
        ok = report(name, got, gold, dtype=torch.bfloat16, allowed_ulp=1)
    else:
        # Recurrent states are FP32. Require both a small absolute and
        # relative error; a broad BF16-style absolute tolerance is invalid.
        ok = report(name, got, gold, dtype=torch.float32, allowed_ulp=None,
                    max_rel=2e-5, abs_tol=3e-4, rel_floor=0.1)
    if not ok:
        failures[0] += 1


def recurrence(q, k, v, g, beta, state, q_scale=1.0, head_map=None, slots=None, cu=None):
    """Reference recurrence; all accumulators intentionally remain FP32."""
    state = state.clone().float()
    out = torch.zeros_like(v, dtype=torch.float32)
    if cu is None:
        # Flat API layout is [BH, S, K] and [BH, S, V].
        for bh in range(q.shape[0]):
            s = state[bh]
            for t in range(q.shape[1]):
                s.mul_(torch.exp(g[bh, t].float()))
                kk = k[bh, t].float()
                delta = (v[bh, t].float() - torch.mv(s.t(), kk)) * beta[bh, t].float()
                s.add_(kk[:, None] * delta[None, :])
                out[bh, t] = torch.mv(s.t(), q[bh, t].float() * q_scale)
            state[bh] = s
        return out, state
    else:
        sequences = [(int(cu[i]), int(cu[i + 1])) for i in range(len(cu) - 1)]
        head_count = v.shape[1]
    for seq_idx, (start, end) in enumerate(sequences):
        slot = int(slots[seq_idx])
        for vh in range(head_count):
            kh = vh if head_map is None else head_map[vh]
            s = state[slot, vh] if state.ndim == 4 else state[vh]
            for t in range(start, end):
                qq = q[t, kh].float() if q.ndim == 3 else q[0, t, kh].float()
                kk = k[t, kh].float() if k.ndim == 3 else k[0, t, kh].float()
                vv = v[t, vh].float() if v.ndim == 3 else v[0, t, vh].float()
                gg = g[t, vh].float() if g.ndim == 2 else g[0, t, vh].float()
                bb = beta[t, vh].float() if beta.ndim == 2 else beta[0, t, vh].float()
                s.mul_(torch.exp(gg))
                delta = (vv - torch.mv(s.t(), kk)) * bb
                s.add_(kk[:, None] * delta[None, :])
                out[t, vh] = torch.mv(s.t(), qq * q_scale)
            if state.ndim == 4:
                state[slot, vh] = s
            else:
                state[vh] = s
    return out, state


def conv_ref(x, weight, bias, state, cu, silu=True):
    x = bf16(x).float()
    weight = bf16(weight).float()[:, 0]
    bias = bf16(bias).float()
    state = state.float().clone()
    out = torch.empty_like(x, dtype=torch.bfloat16)
    cu = cu.long()
    for b in range(len(cu) - 1):
        history = state[b].clone()
        for t in range(int(cu[b]), int(cu[b + 1])):
            value = x[t] * weight[:, -1]
            value = value + (history * weight[:, :-1]).sum(dim=1) + bias
            if silu:
                value = value / (1.0 + torch.exp(-value))
            out[t] = bf16(value)
            if history.shape[1] > 1:
                history[:, :-1] = history[:, 1:]
            history[:, -1] = x[t]
        state[b] = history
    return out, state


def conv_slots_ref(x, weight, bias, state, slots, silu=True):
    x = bf16(x).float()
    weight = bf16(weight).float()[:, 0]
    bias = bf16(bias).float()
    state = state.float().clone()
    out = torch.empty_like(x, dtype=torch.bfloat16)
    for b, slot in enumerate(slots.long().tolist()):
        history = state[slot].clone()
        value = x[b] * weight[:, -1]
        value = value + (history * weight[:, :-1]).sum(dim=1) + bias
        if silu:
            value = value / (1.0 + torch.exp(-value))
        out[b] = bf16(value)
        if history.shape[1] > 1:
            history[:, :-1] = history[:, 1:]
        history[:, -1] = x[b]
        state[slot] = history
    return out, state


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = read_probe(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/xinfer_gdn_probe.bin"))
    dev = torch.device("cuda:0")
    failures = [0]

    # Fused gating: inputs a and b are BF16, parameters and outputs are FP32.
    a_log = rec["gating_a_log"].to(dev)
    dt = rec["gating_dt_bias"].to(dev)
    a = rec["gating_a"].to(dev, dtype=torch.bfloat16)
    b = rec["gating_b"].to(dev, dtype=torch.bfloat16)
    x = a.float() + dt.view(1, 1, -1)
    expected_g = -torch.exp(a_log).view(1, 1, -1) * torch.where(
        x <= 20, torch.log1p(torch.exp(x)), x
    )
    expected_beta = torch.sigmoid(b.float())
    compare("gdn_fused_gating_g", rec["gating_g"].to(dev), expected_g, 3e-6, failures)
    compare("gdn_fused_gating_beta", rec["gating_beta"].to(dev), expected_beta, 3e-6, failures)

    x = rec["norm_x"].to(dev, dtype=torch.bfloat16)
    z = rec["norm_z"].to(dev, dtype=torch.bfloat16)
    w = rec["norm_w_group"].to(dev, dtype=torch.bfloat16)
    bias = rec["norm_b_group"].to(dev, dtype=torch.bfloat16)
    rms = torch.rsqrt(x.float().reshape(9, 4, 8).square().mean(dim=-1, keepdim=True) + 1e-5).expand(-1, -1, 8).reshape(9, 32)
    expected = bf16((x.float() * rms * w.repeat(4).view(1, 1, -1) + bias.repeat(4).view(1, 1, -1)) * F.silu(z.float()))
    compare("gdn_gated_rmsnorm_group", rec["norm_group"].to(dev), expected, 2e-2, failures)

    w = rec["norm_w_full"].to(dev)
    bias = rec["norm_b_full"].to(dev)
    expected = bf16((x.float() * rms * w.view(1, -1) + bias.view(1, -1)) * F.silu(z.float()))
    compare("gdn_gated_rmsnorm_full_wf32", rec["norm_full"].to(dev), expected, 2e-2, failures)

    expected = bf16(x.float() * torch.rsqrt(x.float().square().sum(dim=-1, keepdim=True) + 1e-6))
    compare("gdn_l2_norm", rec["l2_out"].to(dev), expected, 2e-2, failures)

    for tag in ("conv", "conv_k2", "conv_k4"):
        expected_out, expected_state = conv_ref(
            rec[f"{tag}_x"].to(dev),
            rec[f"{tag}_weight"].to(dev),
            rec[f"{tag}_bias"].to(dev),
            rec[f"{tag}_state_initial"].to(dev),
            rec[f"{tag}_cu"].to(dev),
        )
        compare(f"{tag}_prefill_out", rec[f"{tag}_out"].to(dev), expected_out, 2e-2, failures)
        compare(f"{tag}_prefill_state", rec[f"{tag}_state_final"].to(dev), expected_state, 2e-5, failures)

    expected_out, expected_state = conv_slots_ref(
        rec["conv_slots_x"].to(dev),
        rec["conv_slots_weight"].to(dev),
        rec["conv_slots_bias"].to(dev),
        rec["conv_slots_state_initial"].to(dev),
        rec["conv_slots_slots"].to(dev),
    )
    compare("conv_slots_out", rec["conv_slots_out"].to(dev), expected_out, 2e-2, failures)
    compare("conv_slots_state", rec["conv_slots_state_final"].to(dev), expected_state, 2e-5, failures)

    # Flat recurrence exercises both the tiled K=16 fallback and K=80 fallback.
    for tag in ("rec_k16", "rec_k64", "rec_k80"):
        q = rec[f"{tag}_q"].to(dev, dtype=torch.bfloat16)
        k = rec[f"{tag}_k"].to(dev, dtype=torch.bfloat16)
        v = rec[f"{tag}_v"].to(dev, dtype=torch.bfloat16)
        expected_out, expected_state = recurrence(
            q, k, v, rec[f"{tag}_g"].to(dev), rec[f"{tag}_beta"].to(dev),
            rec[f"{tag}_state_initial"].to(dev),
        )
        compare(f"{tag}_out", rec[f"{tag}_out"].to(dev), bf16(expected_out), 2e-2, failures)
        compare(f"{tag}_state", rec[f"{tag}_state_final"].to(dev), expected_state, 5e-4, failures)

    # Variable-length recurrence; compare token outputs and every state snapshot.
    q = rec["varlen_q"].to(dev, dtype=torch.bfloat16)
    k = rec["varlen_k"].to(dev, dtype=torch.bfloat16)
    v = rec["varlen_v"].to(dev, dtype=torch.bfloat16)
    expected_out, expected_state = recurrence(
        q, k, v, rec["varlen_g"].to(dev), rec["varlen_beta"].to(dev),
        rec["varlen_state_initial"].to(dev), slots=rec["varlen_slots"].to(dev),
        cu=rec["varlen_cu"].to(dev),
    )
    compare("varlen_out", rec["varlen_out"].to(dev), bf16(expected_out), 2e-2, failures)
    compare("varlen_state", rec["varlen_state_final"].to(dev), expected_state, 2e-4, failures)
    # Snapshots are FP32 and represent the state immediately after each token.
    snap = torch.zeros_like(rec["varlen_snapshots"].to(dev))
    state = rec["varlen_state_initial"].to(dev).float().clone()
    for seq_idx, (start, end) in enumerate([(0, 5), (5, 13)]):
        slot = int(rec["varlen_slots"][seq_idx].item())
        for t in range(start, end):
            for h in range(2):
                s = state[slot, h]
                s.mul_(torch.exp(rec["varlen_g"][t, h].to(dev)))
                kk = k[t, h].float()
                delta = (v[t, h].float() - torch.mv(s.t(), kk)) * rec["varlen_beta"][t, h].to(dev)
                s.add_(kk[:, None] * delta[None, :])
                snap[t, h] = s
    compare("varlen_snapshots", rec["varlen_snapshots"].to(dev), snap, 2e-4, failures)

    # GQA varlen: v heads map to k heads by integer group, q is multiplied by q_scale.
    q = rec["gqa_q"].to(dev, dtype=torch.bfloat16)
    k = rec["gqa_k"].to(dev, dtype=torch.bfloat16)
    v = rec["gqa_v"].to(dev, dtype=torch.bfloat16)
    expected_out, expected_state = recurrence(
        q, k, v, rec["gqa_g"].to(dev), rec["gqa_beta"].to(dev),
        rec["gqa_state_initial"].to(dev), q_scale=0.7,
        head_map=[0, 0, 1, 1], slots=rec["gqa_slots"].to(dev),
        cu=rec["gqa_cu"].to(dev),
    )
    compare("gqa_varlen_out", rec["gqa_out"].to(dev), bf16(expected_out), 2e-2, failures)
    compare("gqa_varlen_state", rec["gqa_state_final"].to(dev), expected_state, 2e-4, failures)
    if "gqa_flashinfer_out" in rec:
        compare("gqa_flashinfer_out", rec["gqa_flashinfer_out"].to(dev), bf16(expected_out), 2e-2, failures)
        compare("gqa_flashinfer_state", rec["gqa_flashinfer_state_final"].to(dev), expected_state, 2e-4, failures)
    else:
        print("SKIP gqa_flashinfer: SM90 FlashInfer kernel declined this probe")

    q = rec["decode_q"].to(dev, dtype=torch.bfloat16)
    k = rec["decode_k"].to(dev, dtype=torch.bfloat16)
    v = rec["decode_v"].to(dev, dtype=torch.bfloat16)
    expected_out, expected_state = recurrence(
        q, k, v, rec["decode_g"].to(dev), rec["decode_beta"].to(dev),
        rec["decode_state_initial"].to(dev), q_scale=0.7,
        head_map=[0, 0, 1, 1], slots=rec["decode_slots"].to(dev),
        cu=torch.tensor([0, 1, 2], device=dev),
    )
    compare("gqa_decode_out", rec["decode_out"].to(dev), bf16(expected_out), 2e-2, failures)
    compare("gqa_decode_state", rec["decode_state_final"].to(dev), expected_state, 2e-4, failures)

    q = rec["flat_decode_q"].to(dev, dtype=torch.bfloat16)
    k = rec["flat_decode_k"].to(dev, dtype=torch.bfloat16)
    v = rec["flat_decode_v"].to(dev, dtype=torch.bfloat16)
    expected_out, expected_state = recurrence(
        q, k, v, rec["flat_decode_g"].to(dev), rec["flat_decode_beta"].to(dev),
        rec["flat_decode_state_initial"].to(dev),
        slots=rec["flat_decode_slots"].to(dev),
        cu=torch.tensor([0, 1, 2], device=dev),
    )
    compare("flat_decode_out", rec["flat_decode_out"].to(dev), bf16(expected_out), 2e-2, failures)
    compare("flat_decode_state", rec["flat_decode_state_final"].to(dev), expected_state, 2e-4, failures)

    raise SystemExit(1 if failures[0] else 0)


if __name__ == "__main__":
    main()
