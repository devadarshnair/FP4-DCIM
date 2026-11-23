"""Precomputed E2M1×E2M1 product tables - the exact behaviour the leaf multiplier must implement.

PRODUCT_X4[w_code, x_code] is the signed integer product ×4 (∈ [-144, 144]). Shared by the golden
models (cimfp4.golden), the RTL generator's LUT spec (cimfp4.rtlgen), and the formal miter.
"""

from __future__ import annotations

import numpy as np

from .codecs import e2m1_mag_decomp, e2m1_to_int_x2

CODES = np.arange(16, dtype=np.uint8)

#: signed product ×4, shape (16, 16), int64, index [w_code, x_code]
PRODUCT_X4 = np.outer(e2m1_to_int_x2(CODES), e2m1_to_int_x2(CODES)).astype(np.int64)

#: product sign bit (s_w XOR s_x) - note: the hardware sign of a zero product is irrelevant
PRODUCT_SIGN = ((CODES[:, None] >> 3) ^ (CODES[None, :] >> 3)).astype(np.int64)

#: unsigned product magnitude ×4, shape (16, 16), values in [0, 144]
PRODUCT_MAG_X4 = np.abs(PRODUCT_X4)

#: sorted distinct magnitudes ×4
DISTINCT_MAGS_X4 = np.unique(PRODUCT_MAG_X4)

#: width of the unsigned product-magnitude bus (144 -> 8 bits)
PRODUCT_MAG_BITS = int(PRODUCT_MAG_X4.max()).bit_length()


def product_mantissa_exponent() -> tuple[np.ndarray, np.ndarray]:
    """(m_w·m_x ∈ {0,1,3,9}, e_w+e_x ∈ [0,6]) tables, shape (16,16); |product×4| = m · 2^e."""
    m, e = e2m1_mag_decomp(CODES)
    return np.outer(m, m), e[:, None] + e[None, :]


def block_sum_bits(block: int) -> tuple[int, int]:
    """(unsigned per-polarity width, signed width) for an exact sum of ``block`` products."""
    max_mag = block * int(PRODUCT_MAG_X4.max())
    unsigned = max_mag.bit_length()
    return unsigned, unsigned + 1
