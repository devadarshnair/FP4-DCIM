"""OCP-exact MXFP4 quantizer.

Per block of 32 along the last axis:
    E   = floor(log2(amax))                     (exact, via frexp - handles subnormal inputs)
    X   = E - emax_elem + adj(rule)             emax_elem = 2 for E2M1, clamped to [-127, 127]
    P_i = RNE_E2M1(V_i / 2^X), saturating to ±6 (OCP: normal values clamp to the max normal)

Scale rules:
    ocp_floor  OCP v1.0:  adj = 0                         amax/2^X ∈ [4, 8)  - clips (6, 8)
    ocp_v11    OCP v1.1:  adj = expAdj = [1.m(amax) ≥ 1.75] amax/2^X ∈ [3.5, 7) (≡ torchao EVEN)
    rceil      NVIDIA:    X = ceil(log2(amax/6))           amax/2^X ∈ (3, 6] - never clips
    ceil       torchao:   X = ceil(log2(amax)) - 2         amax/2^X ∈ (2, 4]

Zero blocks get X = -127 (all codes 0). A block containing NaN/Inf gets a NaN scale (0xFF) per
OCP; its codes are meaningless and set to 0.
"""

from __future__ import annotations

import numpy as np

from .codecs import E2M1_EMAX, E2M1_MAG_X2, UE8M0_EMAX, UE8M0_EMIN
from .types import BLOCK_SIZE, FP4Block

SCALE_RULES = ("ocp_floor", "ocp_v11", "rceil", "ceil")

_GRID = E2M1_MAG_X2.astype(np.float64) / 2.0  # [0, .5, 1, 1.5, 2, 3, 4, 6]


def rne_e2m1_magnitude(a: np.ndarray) -> np.ndarray:
    """Non-negative real(s) -> 3-bit E2M1 magnitude index, round-to-nearest-even, saturate at 6.

    Ties go to the grid point whose mantissa bit (= index LSB) is 0:
    0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4.
    """
    a = np.minimum(np.asarray(a, dtype=np.float64), _GRID[-1])
    hi = np.clip(np.searchsorted(_GRID, a, side="left"), 1, 7)
    lo = hi - 1
    d_lo, d_hi = a - _GRID[lo], _GRID[hi] - a
    up = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return np.where(up, hi, lo).astype(np.uint8)


def floor_log2(a: np.ndarray) -> np.ndarray:
    """Exact floor(log2(a)) for finite a > 0 (float64), including subnormals. Zeros -> 0 (masked by caller)."""
    mant, exp = np.frexp(np.where(a > 0, a, 1.0))  # a = mant · 2^exp, mant ∈ [0.5, 1)
    return (exp - 1).astype(np.int64)


def mx_scale_exponent(amax: np.ndarray, rule: str = "ocp_v11") -> np.ndarray:
    """Block amax (float64, ≥ 0, finite) -> unbiased scale exponent X (int64), clamped to UE8M0 range."""
    if rule not in SCALE_RULES:
        raise ValueError(f"unknown scale rule {rule!r}; choose from {SCALE_RULES}")
    amax = np.asarray(amax, dtype=np.float64)
    e = floor_log2(amax)
    frac = amax / np.ldexp(1.0, e.astype(np.int32))  # 1.m ∈ [1, 2), exact (power-of-two division)
    if rule == "ocp_floor":
        adj = 0
    elif rule == "ocp_v11":
        adj = (frac >= 1.75).astype(np.int64)
    elif rule == "rceil":
        adj = (frac > 1.5).astype(np.int64)  # ceil(log2(amax/6)) = E - 2 + [1.m > 1.5]
    else:  # ceil
        adj = (frac > 1.0).astype(np.int64)  # ceil(log2(amax)) - 2 = E - 2 + [amax not a power of 2]
    x = e - E2M1_EMAX + adj
    x = np.where(amax > 0, x, UE8M0_EMIN)
    return np.clip(x, UE8M0_EMIN, UE8M0_EMAX).astype(np.int64)


def quantize_mxfp4(x, rule: str = "ocp_v11") -> FP4Block:
    """Quantize a real tensor (last axis multiple of 32) to MXFP4."""
    v = np.asarray(x, dtype=np.float64)
    k = v.shape[-1]
    bs = BLOCK_SIZE["mxfp4"]
    if k % bs:
        raise ValueError(f"last axis {k} not a multiple of {bs}")
    vb = v.reshape(*v.shape[:-1], k // bs, bs)
    nonfinite = ~np.isfinite(vb).all(axis=-1)
    vb = np.where(np.isfinite(vb), vb, 0.0)
    amax = np.abs(vb).max(axis=-1)
    xexp = mx_scale_exponent(amax, rule)
    scaled = np.abs(vb) / np.ldexp(1.0, xexp[..., None].astype(np.int32))
    mag = rne_e2m1_magnitude(scaled)
    sign = (np.signbit(vb) & (mag > 0)).astype(np.uint8)  # canonical: no -0 codes from the quantizer
    codes = (sign << 3 | mag).astype(np.uint8)
    codes = np.where(nonfinite[..., None], 0, codes).astype(np.uint8)
    return FP4Block(
        codes=codes.reshape(v.shape),
        scale_exp=xexp.astype(np.int16),
        scale_nan=nonfinite,
        fmt="mxfp4",
        block=bs,
        scale_rule=rule,
    )


def dequantize(q: FP4Block) -> np.ndarray:
    """FP4Block -> float64 tensor (exact). NaN-scale blocks dequantize to NaN."""
    from .codecs import e2m1_decode, e4m3_decode

    vals = e2m1_decode(q.blocked_codes())
    if q.fmt == "mxfp4":
        s = np.ldexp(1.0, q.scale_exp.astype(np.int32))
        s = np.where(q.scale_nan, np.nan, s)
    else:
        s = e4m3_decode(q.scale_e4m3) * np.float64(q.tensor_scale)
    return (vals * s[..., None]).reshape(q.codes.shape)


# --------------------------------------------------------------------------------- MXFP8 (E4M3)
E4M3_EMAX = 8  # 448 = 1.75 * 2^8


def quantize_mxfp8_values(x, rule: str = "ocp_v11") -> np.ndarray:
    """MXFP8 (E4M3 elements, block 32, UE8M0 scale) fake-quant -> dequantized float64 values.

    Used for the attention BMM operands (attn_policy="mxfp8"). X = floor(log2 amax) -
    8 + adj, elements RNE to E4M3 with saturation at ±448 (OCP SAT mode). For E4M3 the v1.1 expAdj
    triggers when amax rounds up to the next binade at 3 mantissa bits (1.m ≥ 1.9375); RCEIL when
    1.m > 1.75 (amax/448 crosses a power of two).
    """
    from .codecs import e4m3_decode, e4m3_encode

    v = np.asarray(x, dtype=np.float64)
    k = v.shape[-1]
    if k % 32:
        raise ValueError(f"last axis {k} not a multiple of 32")
    vb = v.reshape(*v.shape[:-1], k // 32, 32)
    amax = np.abs(vb).max(axis=-1)
    e = floor_log2(amax)
    frac = amax / np.ldexp(1.0, e.astype(np.int32))
    if rule == "ocp_v11":
        adj = (frac >= 1.9375).astype(np.int64)
    elif rule == "rceil":
        adj = (frac > 1.75).astype(np.int64)
    elif rule == "ocp_floor":
        adj = 0
    else:
        raise ValueError(rule)
    xexp = np.clip(np.where(amax > 0, e - E4M3_EMAX + adj, -127), -127, 127)
    s = np.ldexp(1.0, xexp.astype(np.int32))[..., None]
    q = e4m3_decode(e4m3_encode(vb / s, saturate=True))
    return (q * s).reshape(v.shape)


# MXFP6 (E3M2): bias 3, all codes finite, max 1.75·2^4 = 28, min normal 2^-2, subnormal step 2^-4
E3M2_EMAX = 4
E3M2_MAX = 28.0
E3M2_EMIN = -2


def rne_e3m2(a: np.ndarray) -> np.ndarray:
    """Real values (float64) -> nearest E3M2 value (RNE, saturate at ±28), as values."""
    a = np.asarray(a, dtype=np.float64)
    mag = np.abs(a)
    e = np.maximum(floor_log2(mag), E3M2_EMIN)  # subnormals share the min-normal quantum
    quantum = np.ldexp(1.0, (e - 2).astype(np.int32))  # 2 mantissa bits
    q = np.minimum(np.rint(mag / quantum) * quantum, E3M2_MAX)  # np.rint = round half to even; exact
    return np.copysign(np.where(mag > 0, q, 0.0), a)


def quantize_mxfp6_values(x, rule: str = "ocp_v11") -> np.ndarray:
    """MXFP6 (E3M2 elements, block 32, UE8M0 scale) fake-quant -> dequantized float64 values.

    Used for the MIDAS-protocol rows (MIDAS Table II: MXFP6 = E3M2). X = floor(log2 amax) - 4 + adj;
    elements RNE to E3M2 with saturation at ±28. Same derivation as E2M1/E4M3: the v1.1 expAdj fires when
    amax rounds up to the next binade at 2 mantissa bits (1.m ≥ 1.875, tie-to-even goes up); RCEIL when
    amax/28 crosses a power of two (1.m > 1.75).
    """
    v = np.asarray(x, dtype=np.float64)
    k = v.shape[-1]
    if k % 32:
        raise ValueError(f"last axis {k} not a multiple of 32")
    vb = v.reshape(*v.shape[:-1], k // 32, 32)
    amax = np.abs(vb).max(axis=-1)
    e = floor_log2(amax)
    frac = amax / np.ldexp(1.0, e.astype(np.int32))
    if rule == "ocp_v11":
        adj = (frac >= 1.875).astype(np.int64)
    elif rule == "rceil":
        adj = (frac > 1.75).astype(np.int64)
    elif rule == "ocp_floor":
        adj = 0
    else:
        raise ValueError(rule)
    xexp = np.clip(np.where(amax > 0, e - E3M2_EMAX + adj, -127), -127, 127)
    s = np.ldexp(1.0, xexp.astype(np.int32))[..., None]
    return (rne_e3m2(vb / s) * s).reshape(v.shape)
