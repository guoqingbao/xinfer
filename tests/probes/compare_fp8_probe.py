"""Compare attention.rs FP8 GEMM outputs with an independent PyTorch reference."""

from __future__ import annotations

import pathlib
import struct
import sys

import torch
from precision_metrics import report


def read_tensor(f, name_len):
    name = f.read(name_len).decode()
    rank = struct.unpack("<Q", f.read(8))[0]
    shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(rank))
    n = struct.unpack("<Q", f.read(8))[0]
    raw = bytearray(f.read(4 * n))
    return name, torch.frombuffer(raw, dtype=torch.float32).clone().reshape(shape)


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/xinfer_fp8_probe.bin")
    records = {}
    with path.open("rb") as f:
        if f.read(8) != b"XINFFP81":
            raise ValueError("bad FP8 probe magic")
        while True:
            raw = f.read(8)
            if not raw:
                break
            name_len = struct.unpack("<Q", raw)[0]
            name, tensor = read_tensor(f, name_len)
            records[name] = tensor

    device = torch.device("cuda:0")
    x = records["input"].to(device=device, dtype=torch.bfloat16).float()
    # The Rust probe records the raw FP8 bytes as float values.  Recover the
    # bytes and let PyTorch perform the reference E4M3FN conversion.
    w_bytes = records["weight"].to(device=device).to(torch.uint8)
    w = w_bytes.view(torch.float8_e4m3fn).float()
    # CUTLASS is W8A8 here: attention.rs dynamically quantizes each 128-value
    # activation group and carries the inverse scale into the GEMM.  The
    # fallback is W8A16 and consumes BF16 activations directly.
    activation_scale = x.abs().amax(dim=1, keepdim=True) / 448.0
    x_cutlass = (x / activation_scale).to(torch.float8_e4m3fn).float() * activation_scale
    expected_cutlass = x_cutlass @ w.t()
    expected_fallback = x @ w.t()
    failures = 0
    for name in ("cutlass", "fallback"):
        got = records[name].to(device=device).float()
        expected = expected_cutlass if name == "cutlass" else expected_fallback
        # Rust returns BF16. Compare in the actual output dtype; a one-ULP
        # difference is reported explicitly rather than hidden by an absolute
        # 0.25 tolerance.
        ok = report(name, got, expected, dtype=torch.bfloat16, allowed_ulp=1, max_rel=1e-5)
        failures += not ok
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
