"""NVFP4 quantizer: E2M1 elements, block 16, E4M3 block scale, FP32 per-tensor scale.

Follows NVIDIA's recipe (arXiv 2509.25149 App. B):
    s_dec      = amax_tensor / (6·448)                (fp32; static/calibrated or dynamic)
    s_enc      = fp32(1 / s_dec)
    s_dec_b    = amax_b / 6                           (fp32)
    scale_b    = e4m3_RNE_satfinite(s_dec_b · s_enc)  (stored)
    x̂_i        = E2M1_RNE_sat( x_i / (scale_b · s_dec) )

Hardware convention "hw" (the online NV quantizer RTL, WP-68): c = fp32(s_enc / 6) is a configuration
register; scale_b = e4m3_RNE_satfinite(amax_b · c) computed *exactly* (one rounding instead of two);
elements use the exact threshold comparison. It differs from the reference only where the reference's
fp32 intermediate rounding moves amax_b/6·s_enc across an E4M3 rounding boundary (measured in tests).

Two element conventions (they differ only in rare near-tie cases):
    "recip_mul"  NVIDIA reference: s_enc_b = fp32(1 / fp32(scale_b · s_dec)); x̂ = q(fp32(x · s_enc_b))
                 - double rounding through fp32.
    "threshold"  exact: rounds x / (scale_b · s_dec) computed in float64 (what a 7-threshold
                 compare quantizer in hardware implements).
Dequantization: x ≈ code · scale_b · tensor_scale, with tensor_scale = s_dec.
"""

from __future__ import annotations

import numpy as np

from .codecs import E2M1_MAX, E4M3_MAX, e4m3_decode, e4m3_encode
from .mx_quant import rne_e2m1_magnitude
from .types import BLOCK_SIZE, FP4Block

CONVENTIONS = ("recip_mul", "threshold", "hw")


def nv_tensor_scale(amax_tensor: float) -> np.float32:
    """Per-tensor decode scale s_dec = amax / (6·448) in fp32 (1.0 for an all-zero tensor)."""
    a = np.float32(amax_tensor)
    if not np.isfinite(a):
        raise ValueError("tensor amax must be finite")
    return np.float32(1.0) if a == 0 else np.float32(a / np.float32(E2M1_MAX * E4M3_MAX))


def quantize_nvfp4(x, convention: str = "recip_mul", tensor_amax: float | None = None) -> FP4Block:
    """Quantize a real tensor (last axis multiple of 16) to NVFP4.

    tensor_amax: static (calibrated) per-tensor amax; None -> dynamic (amax of ``x``). Values beyond a
    static amax saturate (block scale clamps at 448, elements at ±6).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; choose from {CONVENTIONS}")
    v = np.asarray(x, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("NVFP4 quantizer input must be finite")
    k, bs = v.shape[-1], BLOCK_SIZE["nvfp4"]
    if k % bs:
        raise ValueError(f"last axis {k} not a multiple of {bs}")
    s_dec = nv_tensor_scale(np.abs(v).max() if tensor_amax is None else tensor_amax)
    s_enc = np.float32(np.float32(1.0) / s_dec)

    vb = v.reshape(*v.shape[:-1], k // bs, bs)
    amax_b = np.abs(vb).max(axis=-1)
    if convention == "hw":
        c = np.float32(s_enc / np.float32(E2M1_MAX))  # configuration register (fp32)
        scale_code = e4m3_encode(amax_b.astype(np.float64) * np.float64(c), saturate=True)  # exact product
    else:
        s_dec_b = (amax_b / np.float32(E2M1_MAX)).astype(np.float32)
        scale_code = e4m3_encode((s_dec_b * s_enc).astype(np.float32), saturate=True)
    scale_val = e4m3_decode(scale_code).astype(np.float32)  # ≥ 0 (amax ≥ 0)
    denom = (scale_val * s_dec).astype(np.float32)
    zero_scale = denom == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        if convention == "recip_mul":
            s_enc_b = np.where(zero_scale, 0, np.float32(1.0) / denom).astype(np.float32)
            scaled = np.abs(vb * s_enc_b[..., None]).astype(np.float32).astype(np.float64)
        else:  # threshold and hw: exact x / s
            d64 = scale_val.astype(np.float64) * np.float64(s_dec)
            scaled = np.where(zero_scale[..., None], 0.0, np.abs(vb.astype(np.float64)) / d64[..., None])
    mag = rne_e2m1_magnitude(scaled)
    sign = (np.signbit(vb) & (mag > 0)).astype(np.uint8)
    codes = (sign << 3 | mag).astype(np.uint8).reshape(v.shape)
    return FP4Block(
        codes=codes,
        scale_exp=np.zeros(scale_code.shape, dtype=np.int16),
        fmt="nvfp4",
        block=bs,
        scale_rule=f"nv_{convention}",
        scale_e4m3=scale_code,
        tensor_scale=float(s_dec),
    )
