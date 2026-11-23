"""Fake-quant fast path: dequantize FP4Blocks to fp32 and run a plain fp32 GEMM (TF32 disabled).

For MX this is bit-identical to L1(Exact) whenever every partial sum of the GEMM is exactly
representable in fp32 - true at realistic scale spreads, and within
~1e-6 relative otherwise. Used for whole-model LLM evaluation (WP-11+); L1 remains the reference.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from ..formats.codecs import E2M1_VALUES, e4m3_decode
from ..formats.types import FP4Block


@contextlib.contextmanager
def no_tf32():
    """Force true fp32 matmuls (TF32 would silently drop mantissa bits)."""
    prev = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[:2]
        torch.set_float32_matmul_precision(prev[2])


def dequantize_torch(q: FP4Block, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """FP4Block -> dense tensor (exact for MX in fp32; NVFP4 rounds code·scale·tensor_scale once)."""
    vals = torch.as_tensor(E2M1_VALUES, dtype=torch.float64, device=device)
    codes = torch.as_tensor(q.blocked_codes().astype(np.int64), device=device)
    v = vals[codes]  # (..., nb, bs) float64
    if q.fmt == "mxfp4":
        e = torch.as_tensor(q.scale_exp.astype(np.int64), device=device)
        s = torch.ldexp(torch.ones(e.shape, dtype=torch.float64, device=device), e)
        if q.scale_nan.any():
            s = torch.where(torch.as_tensor(q.scale_nan, device=device), torch.nan, s)
    else:
        s = torch.as_tensor(e4m3_decode(q.scale_e4m3), device=device) * float(q.tensor_scale)
    return (v * s[..., None]).reshape(q.codes.shape).to(dtype)


def fakequant_matmul(qw: FP4Block, qx: FP4Block, device="cpu") -> torch.Tensor:
    """W·Xᵀ via dequantize -> fp32 GEMM (TF32 off). Returns float32 (N, M) on ``device``."""
    with no_tf32():
        w = dequantize_torch(qw, device)
        x = dequantize_torch(qx, device)
        return w @ x.T
