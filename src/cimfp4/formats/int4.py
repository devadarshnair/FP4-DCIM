"""D2 INT4 surrogate formats (docs/spec/d2_int4_contract.md v1).

MXINT4-SM (contract v1.1): 32-element blocks, shared UE8M0 power-of-two scale 2^X, elements sign + 3-bit
magnitude m ∈ [0, 7]. Scale rules:
  floor_sat (primary, v1.1): X = ⌊log₂ amax⌋ - 2, m = min(RNE(|v| / 2^X), 7) - the OCP MXINT8 pattern
                             (finest step; rare values in (7, 8)·2^X saturate); +1.1 dB weight SQNR.
  ceil      (v1, ablation) : X = ⌈log₂(amax / 7)⌉, never clips (coarser step for blocks with 1.m > 1.75).
Project-specific, not an OCP format.

Also the two lossy recodes of existing E2M1 codes used as ablations (`clip`, `half`), as exact tables.
Exact E2M1 integerization needs INT5 (2v ∈ [-12, 12]); see formats.codecs.e2m1_to_int_x2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codecs import e2m1_to_int_x2
from .mx_quant import floor_log2

BLOCK = 32
MAG_MAX = 7
UE8M0_EMIN, UE8M0_EMAX = -127, 127


@dataclass(frozen=True)
class Int4Block:
    """MXINT4-SM tensor: signed magnitudes (..., K) int8 in [-7, 7] and block scale exponents (..., K/32)."""

    m: np.ndarray
    scale_exp: np.ndarray

    def dequantize(self) -> np.ndarray:
        s = np.ldexp(1.0, self.scale_exp.astype(np.int32))
        mb = self.m.reshape(*self.m.shape[:-1], -1, BLOCK).astype(np.float64)
        return (mb * s[..., None]).reshape(self.m.shape)


RULES = ("floor_sat", "ceil")


def mxint4_scale_exponent(amax: np.ndarray, rule: str = "floor_sat") -> np.ndarray:
    """floor_sat: X = ⌊log₂ amax⌋ - 2; ceil: smallest X with amax / 2^X ≤ 7. Exact integer arithmetic."""
    if rule not in RULES:
        raise ValueError(rule)
    amax = np.asarray(amax, dtype=np.float64)
    pos = amax > 0
    e = floor_log2(np.where(pos, amax, 1.0))  # amax = f·2^e, f ∈ [1, 2)
    # candidate X = e - 2 gives amax / 2^X = 4f ∈ [4, 8): fits iff 4f ≤ 7
    frac = np.where(pos, amax, 1.0) / np.ldexp(1.0, e.astype(np.int32))
    x = e - 2 + ((4 * frac > MAG_MAX) & (rule == "ceil")).astype(np.int64)
    return np.clip(np.where(pos, x, UE8M0_EMIN), UE8M0_EMIN, UE8M0_EMAX).astype(np.int64)


def quantize_mxint4(x, rule: str = "floor_sat") -> Int4Block:
    v = np.asarray(x, dtype=np.float64)
    if np.isnan(v).any():
        raise ValueError("NaN input")
    k = v.shape[-1]
    if k % BLOCK:
        raise ValueError(f"last axis {k} not a multiple of {BLOCK}")
    vb = v.reshape(*v.shape[:-1], k // BLOCK, BLOCK)
    xexp = mxint4_scale_exponent(np.abs(vb).max(axis=-1), rule)
    # RNE (np.rint); dividing by a power of two is exact
    q = np.minimum(np.rint(np.abs(vb) / np.ldexp(1.0, xexp.astype(np.int32))[..., None]), MAG_MAX)
    m = (np.sign(vb) * q).astype(np.int8).reshape(v.shape)
    return Int4Block(m, xexp)


# ----------------------------------------------------------------------------- lossy recodes of E2M1 codes
def recode_e2m1_to_int4(codes, method: str) -> tuple[np.ndarray, int]:
    """E2M1 codes -> (signed INT4-SM magnitudes, extra scale exponent). Value ≈ m · 2^(block_X - 1 + extra).

    clip: m = clip(2v, ±7), extra 0 (2v = ±8, ±12 -> ±7)
    half: m = RNE(2v / 2), extra 1 (|2v| ∈ {1, 3} -> ties to even: 1 -> 0, 3 -> 2)
    """
    i2 = e2m1_to_int_x2(codes).astype(np.int64)
    if method == "clip":
        return np.clip(i2, -MAG_MAX, MAG_MAX).astype(np.int8), 0
    if method == "half":
        return (np.sign(i2) * np.rint(np.abs(i2) / 2.0)).astype(np.int8), 1
    raise ValueError(method)


# ----------------------------------------------------------------------------- torch fake-quant (GPU path)
def fq_mxint4(x, rule: str = "floor_sat"):
    """Torch MXINT4-SM fake-quant along the last axis; bitwise equal to quantize_mxint4(...).dequantize()."""
    import torch

    from ..kernels.torch_quant import _floor_log2

    shape, dtype = x.shape, x.dtype
    # Every step is exact power-of-two scaling, RNE and min/abs, so float32 is bitwise equal to float64 for
    # bf16/fp16/fp32 inputs; float64 is kept only for float64 inputs (tests). float64 on consumer GPUs is
    # slow and its temporaries overflowed the 8 GB RTX 4060 in lm-eval (2026-11-07).
    work = torch.float64 if dtype == torch.float64 else torch.float32
    v = x.to(work).reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    amax = v.abs().amax(-1)
    pos = amax > 0
    safe = torch.where(pos, amax, torch.ones_like(amax))
    e = _floor_log2(safe)
    frac = safe / torch.ldexp(torch.ones_like(safe), e)
    xexp = e - 2 + ((4 * frac > MAG_MAX) & (rule == "ceil")).to(torch.int64)
    xexp = torch.clamp(torch.where(pos, xexp, torch.full_like(xexp, UE8M0_EMIN)), UE8M0_EMIN, UE8M0_EMAX)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    q = torch.clamp(torch.round(v.abs() / s), max=MAG_MAX) * torch.sign(v)
    return (q * s).reshape(shape).to(dtype)
