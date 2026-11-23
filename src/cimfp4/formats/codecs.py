"""Bit-level codecs for the element and scale formats used by MXFP4 and NVFP4.

Single source of truth for encodings. All functions are vectorized over
numpy arrays of codes (uint8) or values (float64).

Formats
-------
E2M1  (FP4)  : [s e1 e0 m], bias 1, no inf/NaN. Values ±{0, .5, 1, 1.5, 2, 3, 4, 6}.
UE8M0 (MX scale, OCP MX v1.1 name for E8M0): unsigned exponent, value 2^(code-127), 0xFF = NaN.
E4M3  (NVFP4 scale, "fn" variant): bias 7, no inf, NaN = S.1111.111, max finite 448.

The *integer view* used by the hardware: an E2M1 magnitude ×2 is an integer in {0,1,2,3,4,6,8,12}
= m·2^e with m ∈ {1, 3}; so a product ×4 is an integer in [0, 144].
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# E2M1
# ---------------------------------------------------------------------------------------------

E2M1_BIAS = 1
E2M1_MAX = 6.0
E2M1_EMAX = 2  # exponent of the largest normal (6 = 1.5 * 2^2); emax_elem in the OCP spec

#: magnitude ×2 for each 3-bit magnitude code (e1 e0 m)
E2M1_MAG_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)

#: all 16 code values as float64, index = code
E2M1_VALUES = np.concatenate([E2M1_MAG_X2, -E2M1_MAG_X2]).astype(np.float64) / 2.0


def _as_codes(codes, nbits: int) -> np.ndarray:
    c = np.asarray(codes)
    if not np.issubdtype(c.dtype, np.integer):
        raise TypeError(f"codes must be integers, got {c.dtype}")
    if c.size and (c.min() < 0 or c.max() >= (1 << nbits)):
        raise ValueError(f"codes out of range for {nbits}-bit format")
    return c.astype(np.int64)


def e2m1_fields(codes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split E2M1 codes into (sign, exponent, mantissa) bit fields."""
    c = _as_codes(codes, 4)
    return (c >> 3) & 1, (c >> 1) & 3, c & 1


def e2m1_decode(codes) -> np.ndarray:
    """E2M1 code(s) -> float64 value(s). -0 decodes to -0.0."""
    c = _as_codes(codes, 4)
    mag = E2M1_MAG_X2[c & 7].astype(np.float64) / 2.0
    return np.where(c >> 3, -mag, mag)


def e2m1_to_int_x2(codes) -> np.ndarray:
    """E2M1 code(s) -> signed integer value ×2 (∈ [-12, 12]); ±0 -> 0."""
    c = _as_codes(codes, 4)
    mag = E2M1_MAG_X2[c & 7]
    return np.where(c >> 3, -mag, mag)


def e2m1_mag_decomp(codes) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude ×2 = m · 2^e with m ∈ {0, 1, 3}. Returns (m, e); zero -> (0, 0).

    This is the decomposition the row driver / leaf multiplier exploit: a product is
    (m_w·m_x ∈ {1,3,9}) · 2^(e_w+e_x).
    """
    c = _as_codes(codes, 4) & 7
    m = np.select([c == 0, (c == 3) | (c == 5) | (c == 7)], [0, 3], default=1)
    #            mag×2: 0  1  2  3  4  6  8  12
    e_tab = np.array([0, 0, 1, 0, 2, 1, 3, 2], dtype=np.int64)
    return m.astype(np.int64), e_tab[c]


def e2m1_encode_exact(values) -> np.ndarray:
    """Exactly representable float value(s) -> E2M1 code(s). Raises if any value isn't on the grid.

    (Rounding of arbitrary reals is the quantizer's job - see formats.mx_quant.)
    """
    v = np.asarray(values, dtype=np.float64)
    mag2 = np.abs(v) * 2.0
    idx = np.searchsorted(E2M1_MAG_X2, mag2)
    idx = np.clip(idx, 0, 7)
    if not np.array_equal(E2M1_MAG_X2[idx].astype(np.float64), mag2):
        raise ValueError("value(s) not exactly representable in E2M1")
    return (np.signbit(v).astype(np.int64) << 3 | idx).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# UE8M0 (MX shared scale)
# ---------------------------------------------------------------------------------------------

UE8M0_BIAS = 127
UE8M0_NAN = 0xFF
UE8M0_EMIN, UE8M0_EMAX = -127, 127


def ue8m0_to_exp(codes) -> np.ndarray:
    """UE8M0 code(s) -> unbiased exponent int64. NaN code (0xFF) raises - callers must check first."""
    c = _as_codes(codes, 8)
    if np.any(c == UE8M0_NAN):
        raise ValueError("UE8M0 NaN code (0xFF) has no exponent")
    return c - UE8M0_BIAS


def ue8m0_decode(codes) -> np.ndarray:
    """UE8M0 code(s) -> float64 scale 2^(code-127); 0xFF -> NaN."""
    c = _as_codes(codes, 8)
    out = np.ldexp(1.0, (c - UE8M0_BIAS).astype(np.int32))
    return np.where(c == UE8M0_NAN, np.nan, out)


def ue8m0_encode(exps) -> np.ndarray:
    """Unbiased exponent(s) in [-127, 127] -> UE8M0 code(s)."""
    e = np.asarray(exps, dtype=np.int64)
    if e.size and (e.min() < UE8M0_EMIN or e.max() > UE8M0_EMAX):
        raise ValueError("UE8M0 exponent out of range [-127, 127]")
    return (e + UE8M0_BIAS).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# E4M3 (NVFP4 block scale; OCP/NVIDIA "fn" variant)
# ---------------------------------------------------------------------------------------------

E4M3_BIAS = 7
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9


def _e4m3_table() -> np.ndarray:
    vals = np.empty(256, dtype=np.float64)
    for c in range(256):
        s, e, m = (c >> 7) & 1, (c >> 3) & 0xF, c & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** (1 - E4M3_BIAS)
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - E4M3_BIAS)
        vals[c] = -v if s else v
    return vals


E4M3_VALUES = _e4m3_table()
_E4M3_POS = E4M3_VALUES[:127]  # codes 0x00..0x7E: all non-negative finite values, ascending


def e4m3_decode(codes) -> np.ndarray:
    """E4M3 code(s) -> float64 (NaN for 0x7F / 0xFF)."""
    return E4M3_VALUES[_as_codes(codes, 8)]


def e4m3_significand(codes) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 code(s) -> (u, k) with |value| = u · 2^k, u ∈ [0, 15] integer (1.mmm·8 or 0.mmm·8).

    The NVFP4 scale multiply in hardware works on these 4-bit integer significands.
    """
    c = _as_codes(codes, 8)
    e, m = (c >> 3) & 0xF, c & 7
    u = np.where(e == 0, m, m + 8)
    k = np.where(e == 0, 1, e) - E4M3_BIAS - 3
    return u.astype(np.int64), k.astype(np.int64)


def e4m3_scale_nan(codes) -> np.ndarray:
    """NVFP4 block-scale NaN flag: codes 0x7F and 0xFF (S.1111.111, the only NaN of the "fn" variant)."""
    return (_as_codes(codes, 8) & 0x7F) == 0x7F


def e4m3_scale_reserved(codes) -> np.ndarray:
    """Reserved NVFP4 scale codes: sign bit set and not NaN (0x80..0xFE). A block scale is amax-derived
    and non-negative, so the quantizer never emits them; the contract (numerics.md) leaves them undefined
    and golden/stimulus code must reject them rather than silently drop the sign."""
    c = _as_codes(codes, 8)
    return ((c & 0x80) != 0) & ((c & 0x7F) != 0x7F)


def e4m3_scale_fields(codes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E4M3 scale code(s) -> (u, k, nan) as stored in the macro scale RF. NaN codes give u = k = 0 (the
    values are don't-care; the NaN flag alone decides the output). Reserved codes raise ValueError.
    Zero (0x00) gives u = 0 (the block contributes exactly 0); subnormals give u = m in 1..7, k = -9."""
    c = _as_codes(codes, 8)
    if e4m3_scale_reserved(c).any():
        raise ValueError("reserved E4M3 scale code (sign bit set, not NaN)")
    nan = e4m3_scale_nan(c)
    u, k = e4m3_significand(np.where(nan, 0, c))
    return u, np.where(nan, 0, k), nan


def e4m3_encode(values, saturate: bool = True) -> np.ndarray:
    """float(s) -> E4M3 code(s), round-to-nearest-even. ``saturate`` clamps |x| > 448 to ±448
    (NVIDIA "satfinite"); otherwise overflow -> NaN. NaN input -> NaN code (0x7F)."""
    v = np.asarray(values, dtype=np.float64)
    a = np.abs(v)
    # nearest neighbours on the positive grid
    hi = np.clip(np.searchsorted(_E4M3_POS, a, side="left"), 0, 126)
    lo = np.clip(hi - 1, 0, 126)
    d_lo, d_hi = a - _E4M3_POS[lo], _E4M3_POS[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))  # ties -> even mantissa LSB
    code = np.where(pick_hi, hi, lo)
    code = np.where(a <= _E4M3_POS[0], 0, code)
    over = a > E4M3_MAX
    if saturate:
        code = np.where(over, 126, code)
    else:
        # values beyond 448 round to 448 only below the midpoint to the next (virtual) step 480
        code = np.where(over & (a >= 464.0), 127, np.where(over, 126, code))
    code = np.where(np.isnan(a), 127, code)
    return (np.signbit(v).astype(np.int64) << 7 | code).astype(np.uint8)
