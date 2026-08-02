"""Strict comparison metrics shared by the CUDA differential probes."""

from __future__ import annotations

import torch


def ulp_distance(got: torch.Tensor, gold: torch.Tensor, dtype: torch.dtype) -> int:
    """Maximum ULP distance after both values are represented in output dtype."""
    a = got.to(dtype).contiguous()
    b = gold.to(dtype).contiguous()
    if dtype == torch.float16 or dtype == torch.bfloat16:
        bits = 16
        signed = torch.int16
    elif dtype == torch.float32:
        bits = 32
        signed = torch.int32
    else:
        return 0 if torch.equal(a, b) else 2**63 - 1
    ia = a.view(signed).to(torch.int64)
    ib = b.view(signed).to(torch.int64)
    min_int = -(1 << (bits - 1))
    oa = torch.where(ia < 0, min_int - ia, ia)
    ob = torch.where(ib < 0, min_int - ib, ib)
    return int((oa - ob).abs().max().item()) if oa.numel() else 0


def stats(got: torch.Tensor, gold: torch.Tensor, *, rel_floor: float = 1e-3):
    diff = (got.float() - gold.float()).abs()
    finite = bool(torch.isfinite(diff).all())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    # Relative error is meaningless when the reference is close to zero: a
    # one-bit error around zero can otherwise become an arbitrarily large
    # number.  Keep the absolute metric for that region and compute relative
    # error only where the reference has a useful scale.
    scale = gold.float().abs()
    useful = scale >= rel_floor
    max_rel = float((diff[useful] / scale[useful]).max().item()) if bool(useful.any()) else 0.0
    return finite, max_abs, mean_abs, max_rel


def report(
    name: str,
    got: torch.Tensor,
    gold: torch.Tensor,
    *,
    dtype: torch.dtype,
    allowed_ulp: int | None = 0,
    max_rel: float = 0.0,
    abs_tol: float = 0.0,
    rel_floor: float = 1e-3,
):
    got_d = got.to(dtype)
    gold_d = gold.to(dtype)
    finite, max_abs, mean_abs, rel = stats(got_d, gold_d, rel_floor=rel_floor)
    ulp = ulp_distance(got_d, gold_d, dtype) if finite else 2**63 - 1
    ok = (
        finite
        and (allowed_ulp is None or ulp <= allowed_ulp)
        and (max_rel <= 0.0 or rel <= max_rel)
        and (abs_tol <= 0.0 or max_abs <= abs_tol)
    )
    if ok and ulp == 0:
        status = "PASS_EXACT"
    elif ok:
        status = f"PASS_ULP{ulp}" if allowed_ulp is not None else "PASS_TOL"
    else:
        status = "FAIL"
    allowed = "any" if allowed_ulp is None else str(allowed_ulp)
    print(
        f"{status:9} {name:34} max={max_abs:.8g} mean={mean_abs:.8g} "
        f"rel={rel:.8g} ulp={ulp} allowed={allowed}"
    )
    return ok
