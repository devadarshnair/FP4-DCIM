"""Torch implementation of L1(Exact) for large shapes (CPU or GPU).

Block sums are computed with batched fp32 matmuls over integer-valued operands (|S| ≤ 4608 < 2^24, so
fp32 is exact; TF32 is disabled). Inter-block accumulation is done in float64 on the absolute block
terms P_b·2^E_b, which is exact whenever int_bits + ceil(log2 n_b) + Δ ≤ 53 (Δ = exponent spread).
Outputs violating a conservative spread bound are recomputed with the numpy reference.

For MX the result is bit-identical to golden.l1_dot(..., Exact()). For NVFP4 the per-tensor fp32
scales are applied in float64 after the exact sum, so results agree to ≤ 1 fp32 ulp.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..formats.codecs import e2m1_to_int_x2, e4m3_significand
from ..formats.types import FP4Block
from ..golden.accum import Exact
from ..golden.l1_dot import INT_BITS, l1_dot
from .fakequant import no_tf32


def _scales(q: FP4Block) -> tuple[np.ndarray, np.ndarray]:
    """-> (integer significand u, exponent e) per block; MX: u = 1."""
    if q.fmt == "mxfp4":
        return np.ones(q.scale_exp.shape, np.int64), q.scale_exp.astype(np.int64)
    return e4m3_significand(q.scale_e4m3)


def l1_exact_torch(qw: FP4Block, qx: FP4Block, device="cpu", mem_bytes: int = 256 << 20) -> np.ndarray:
    if qw.fmt != qx.fmt or qw.codes.shape[-1] != qx.codes.shape[-1]:
        raise ValueError("operands must share format and K")
    n, m, nb = qw.codes.shape[0], qx.codes.shape[0], qw.num_blocks
    uw, ew = _scales(qw)
    ux, ex = _scales(qx)

    def ints(q, u):
        return torch.as_tensor(
            e2m1_to_int_x2(q.blocked_codes()) * u[..., None], dtype=torch.float32, device=device
        )

    iw, ix = ints(qw, uw), ints(qx, ux)
    ew_t = torch.as_tensor(ew, dtype=torch.float64, device=device)
    ex_t = torch.as_tensor(ex, dtype=torch.float64, device=device)
    acc = torch.zeros((n, m), dtype=torch.float64, device=device)
    chunk = max(1, mem_bytes // (8 * n * m))
    with no_tf32():
        for b0 in range(0, nb, chunk):
            b1 = min(nb, b0 + chunk)
            # (c, n, bs) @ (c, bs, m) -> (c, n, m); integer-valued, |P| < 2^20 -> exact in fp32
            P = torch.bmm(iw[:, b0:b1].transpose(0, 1), ix[:, b0:b1].permute(1, 2, 0)).to(torch.float64)
            E = ew_t[:, b0:b1].T[:, :, None] + ex_t[:, b0:b1].T[:, None, :] - 2
            acc += torch.ldexp(P, E).sum(0)
    if qw.fmt == "nvfp4":
        acc = acc * (float(qw.tensor_scale) * float(qx.tensor_scale))
    out = acc.to(torch.float32).cpu().numpy()

    # conservative exactness bound; recompute offenders with the numpy reference
    spread = (ew.max(1) - ew.min(1))[:, None] + (ex.max(1) - ex.min(1))[None, :]
    bad = INT_BITS[qw.fmt] + math.ceil(math.log2(max(nb, 2))) + spread > 53
    if bad.any():
        rows, cols = np.nonzero(bad)
        for r, c in zip(rows, cols, strict=True):
            out[r, c] = l1_dot(_take(qw, r), _take(qx, c), Exact())[0, 0]
    return out


def _take(q: FP4Block, i: int) -> FP4Block:
    import dataclasses

    return dataclasses.replace(
        q,
        codes=q.codes[i : i + 1],
        scale_exp=q.scale_exp[i : i + 1],
        scale_nan=q.scale_nan[i : i + 1],
        scale_e4m3=None if q.scale_e4m3 is None else q.scale_e4m3[i : i + 1],
    )
