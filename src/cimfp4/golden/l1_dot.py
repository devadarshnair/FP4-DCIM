"""L1 golden model: bit-exact MXFP4 / NVFP4 dot products.

Intra-block arithmetic is *always* exact integer - this is the alignment-free property
the macro implements. Inter-block accumulation is selected by an AccumMode (golden.accum).

    y[n, m] = Σ_b  P[n, m, b] · 2^E[n, m, b]   (× NVFP4 per-tensor scales)

with W of shape (N, K) and X of shape (M, K), both FP4Block of the same format.
"""

from __future__ import annotations

import numpy as np

from ..formats.codecs import e2m1_to_int_x2, e4m3_scale_fields, e4m3_scale_nan
from ..formats.types import FP4Block
from .accum import AccumMode, Exact, accumulate

#: signed width of a block term P_b
INT_BITS = {"mxfp4": 14, "nvfp4": 21}


def block_terms(qw: FP4Block, qx: FP4Block) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (P, E, anchor) with P, E of shape (N, M, n_b) and anchor (N, M).

    anchor = max_b(scale exponent of w) + max_b(scale exponent of x) - 2 - the quantity the hardware
    knows *before* compute (static weight scales, broadcast activation scales), used by Window.
    """
    if qw.fmt != qx.fmt or qw.codes.shape[-1] != qx.codes.shape[-1]:
        raise ValueError("operands must share format and K")
    if qw.codes.ndim != 2 or qx.codes.ndim != 2:
        raise ValueError("expected 2-D operands (N, K) and (M, K)")
    if qw.scale_nan.any() or qx.scale_nan.any():
        raise ValueError("NaN block scales are not supported by L1 (OCP: result is NaN)")
    iw = e2m1_to_int_x2(qw.blocked_codes())  # (N, nb, bs) ints in [-12, 12]
    ix = e2m1_to_int_x2(qx.blocked_codes())  # (M, nb, bs)
    S = np.einsum("nbk,mbk->nmb", iw, ix)  # exact block sums ×4, |S| ≤ 4608 (MX) / 2304 (NV)
    if qw.fmt == "mxfp4":
        ew, ex = qw.scale_exp.astype(np.int64), qx.scale_exp.astype(np.int64)
        P = S
    else:
        uw, ew, _ = e4m3_scale_fields(qw.scale_e4m3)  # NaN scales handled by l1_dot (nv_nan_mask)
        ux, ex, _ = e4m3_scale_fields(qx.scale_e4m3)
        P = S * uw[:, None, :] * ux[None, :, :]
    E = ew[:, None, :] + ex[None, :, :] - 2
    anchor = ew.max(axis=-1)[:, None] + ex.max(axis=-1)[None, :] - 2
    return P.astype(np.int64), E.astype(np.int64), anchor.astype(np.int64)


def l1_dot(qw: FP4Block, qx: FP4Block, mode: AccumMode | None = None) -> np.ndarray:
    """Bit-exact dot products W·Xᵀ -> float32 array (N, M)."""
    mode = Exact() if mode is None else mode
    P, E, anchor = block_terms(qw, qx)
    nanmask = nv_nan_mask(qw, qx)
    post = (qw.tensor_scale, qx.tensor_scale) if qw.fmt == "nvfp4" else None
    ew, ex = block_exponents(qw, qx)
    y = accumulate(P, E, mode, anchor_exp=anchor, int_bits=INT_BITS[qw.fmt], post_scale=post, ew=ew, ex=ex)
    return y if nanmask is None else np.where(nanmask, np.float32(np.nan), y)


def nv_nan_mask(qw: FP4Block, qx: FP4Block) -> np.ndarray | None:
    """NVFP4 NaN rule (numerics.md, "NVFP4 scale NaN, zero, subnormal"): a dot product (n, m) is NaN iff
    any E4M3 scale code of weight row n or activation row m is 0x7F/0xFF. None for MXFP4 or no NaN."""
    if qw.fmt != "nvfp4":
        return None
    nw = e4m3_scale_nan(qw.scale_e4m3).any(axis=-1)
    nx = e4m3_scale_nan(qx.scale_e4m3).any(axis=-1)
    return (nw[:, None] | nx[None, :]) if (nw.any() or nx.any()) else None


def block_exponents(qw: FP4Block, qx: FP4Block) -> tuple[np.ndarray, np.ndarray]:
    """Per-block scale exponents broadcast to (N, M, n_b): MX UE8M0 exponents; NVFP4 E4M3 k (u·2^k)."""
    if qw.fmt == "mxfp4":
        ew, ex = qw.scale_exp.astype(np.int64), qx.scale_exp.astype(np.int64)
    else:
        _, ew, _ = e4m3_scale_fields(qw.scale_e4m3)
        _, ex, _ = e4m3_scale_fields(qx.scale_e4m3)
    n, m = ew.shape[0], ex.shape[0]
    return (
        np.broadcast_to(ew[:, None, :], (n, m, ew.shape[1])),
        np.broadcast_to(ex[None, :, :], (n, m, ex.shape[1])),
    )
