"""L0 golden model: dequantize to float64, then matmul.

Exact whenever every partial sum fits in float64's 53-bit significand. With block terms of at most
``int_bits`` bits, n_b blocks and block-exponent spread Δ, a sufficient condition is
    int_bits + ceil(log2 n_b) + Δ ≤ 53.
``l0_dot`` checks it and falls back to exact integer arithmetic (via golden.accum.Exact) otherwise,
so for MX, L0 is the exact rational result rounded once to float64. For NVFP4 the per-tensor fp32
scales are multiplied in afterwards (one more float64 rounding, ≤ 2^-52 relative).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..formats.mx_quant import dequantize
from ..formats.types import FP4Block
from .accum import exact_sum
from .l1_dot import INT_BITS, block_terms


def spread_ok(qw: FP4Block, qx: FP4Block) -> np.ndarray:
    """Per-output bool: the float64 fast path is exact."""
    _, E, _ = block_terms(qw, qx)
    delta = E.max(axis=-1) - E.min(axis=-1)
    nb = E.shape[-1]
    return INT_BITS[qw.fmt] + int(np.ceil(np.log2(max(nb, 2)))) + delta <= 53


def l0_dot(qw: FP4Block, qx: FP4Block) -> np.ndarray:
    """Exact W·Xᵀ as float64 (N, M). NVFP4 per-tensor scales included."""
    ok = spread_ok(qw, qx)
    post = 1.0
    if qw.fmt == "nvfp4":
        # keep the matmul exact: per-tensor fp32 scales (24-bit significands) are applied afterwards
        post = float(np.float64(qw.tensor_scale) * np.float64(qx.tensor_scale))
        qw, qx = (dataclasses.replace(q, tensor_scale=1.0) for q in (qw, qx))
    y = dequantize(qw) @ dequantize(qx).T  # float64, exact where `ok`
    if not ok.all():
        P, E, _ = block_terms(qw, qx)
        for n, m in zip(*np.nonzero(~ok), strict=True):
            N, e = exact_sum(P[n, m], E[n, m])
            y[n, m] = float(np.ldexp(float(N), e))  # one float64 rounding
    return y * post
