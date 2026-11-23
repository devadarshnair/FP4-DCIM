"""Inter-block accumulation modes for the L1 golden model.

Every dot product reduces to a sequence of *block terms* ``P_b · 2^E_b`` with integer ``P_b``:
    MX     P_b = Σ_k (2w_k)(2x_k)                ∈ [-4608, 4608]   E_b = e_w,b + e_x,b - 2
    NVFP4  P_b = Σ_k (2w_k)(2x_k) · u_w,b·u_x,b   |P_b| ≤ 2304·225  E_b = k_w,b + k_x,b - 2
(u·2^k is the E4M3 scale as an integer significand). NVFP4 per-tensor scales are applied after
accumulation (``post_scale``).

Modes (all deterministic, applied in block order b = 0..n_b-1):
    Exact()                    exact sum, rounded once to fp32 (RNE)
    FP32Seq()                  acc <- fp32(acc + fp32(P_b·2^E_b))
    Fixed(W, frac, ref_exp)    W-bit two's-complement register, LSB = 2^(ref_exp-frac); right shifts
                               truncate (floor); overflow saturates or wraps - the blueprint's 24-bit
                               accumulator is Fixed(24, ...)
    Window(W, guard)           Fixed register whose LSB is anchored *before compute* at
                               2^(Etop - W + 1), Etop = max_b e_w,b + max_b e_x,b - 2 + int_bits + guard;
                               cannot overflow by construction (default W = 40)
    AlignTrunc(F)              floating accumulator: before each add, both operands are truncated (RZ)
                               to F bits below the larger operand's MSB (Blackwell-like; to be pinned
                               against MMA-Sim in WP-20)
    Approx(fn)                 user hook fn(P, E) -> float64 array, for approximation studies
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------- modes
@dataclass(frozen=True)
class Exact:
    pass


@dataclass(frozen=True)
class FP32Seq:
    pass


@dataclass(frozen=True)
class Fixed:
    width: int
    frac: int
    ref_exp: int = 0
    saturate: bool = True


@dataclass(frozen=True)
class Window:
    width: int = 40
    guard: int | None = None  # None -> ceil(log2(n_blocks))


@dataclass(frozen=True)
class AlignTrunc:
    frac: int = 35


@dataclass(frozen=True)
class Hier:
    """Hierarchical accumulation: K is split into tiles of `tile_blocks` consecutive
    blocks (one macro pass each). Each tile accumulates in a `window`-bit register anchored on its
    *tile-local* exponent maxima (max e_w + max e_x - 2 over the tile, + int_bits + guard); tile results
    then go to a reducer: "exact" (lossless), "fp32" (sequential fp32 adds in tile order), or
    ("window", Wr) (aligned to the max tile anchor, Wr-bit truncating register)."""

    tile_blocks: int
    window: int = 40
    guard: int | None = None  # None -> ceil(log2(tile_blocks))
    reducer: str | tuple = "exact"


@dataclass(frozen=True)
class Approx:
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


AccumMode = Exact | FP32Seq | Fixed | Window | AlignTrunc | Hier | Approx


# --------------------------------------------------------------------------------------- exact helpers
def f32_parts(x) -> tuple[np.ndarray, np.ndarray]:
    """Finite fp32 value(s) -> (integer significand, exponent) with x = s · 2^e exactly."""
    m, e = np.frexp(np.asarray(x, dtype=np.float64))
    return (m * (1 << 53)).astype(np.int64), (e - 53).astype(np.int64)


def round_int_to_f32(n: int, e: int) -> np.float32:
    """Exactly round the rational n·2^e to the nearest fp32 (RNE), incl. subnormals and overflow -> ±inf."""
    if n == 0:
        return np.float32(0.0)
    sign = -1 if n < 0 else 1
    n = abs(n)
    msb = n.bit_length() - 1 + e  # exponent of the leading bit
    lsb_exp = max(msb - 23, -149)  # fp32 ulp exponent at this magnitude (subnormal floor 2^-149)
    shift = lsb_exp - e
    if shift > 0:
        q, r = n >> shift, n & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
    else:
        q = n << (-shift)
    if q.bit_length() - 1 + lsb_exp > 127:
        return np.float32(sign * np.inf)
    return np.float32(sign * float(np.ldexp(float(q), lsb_exp)))  # q ≤ 2^24 -> exact in float64


def exact_sum(P: np.ndarray, E: np.ndarray) -> tuple[int, int]:
    """Exact Σ P_b·2^E_b for 1-D int arrays -> (N, m) with value N·2^m (python ints)."""
    m = int(E.min())
    span = int(E.max()) - m
    if span + int(np.abs(P).max(initial=0)).bit_length() + int(len(P)).bit_length() < 62:
        return int(np.sum(P.astype(np.int64) << (E - m).astype(np.int64))), m
    return sum(int(p) << int(e - m) for p, e in zip(P.tolist(), E.tolist(), strict=True)), m


# --------------------------------------------------------------------------------------- accumulate
def accumulate(
    P: np.ndarray,
    E: np.ndarray,
    mode: AccumMode,
    *,
    anchor_exp: np.ndarray | None = None,
    int_bits: int = 14,
    post_scale: tuple[float, float] | None = None,
    ew: np.ndarray | None = None,
    ex: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate block terms along the last axis.

    P, E        int64 arrays (..., n_b)
    anchor_exp  (...,) max_b e_w,b + max_b e_x,b - 2 (needed by Window)
    int_bits    signed width of P_b (14 for MX K=32; 21 for NVFP4 K=16 incl. scale significands)
    ew, ex      (..., n_b) per-block scale exponents (needed by Hier for tile-local anchors)
    post_scale  NVFP4 per-tensor fp32 scales (s_w, s_x) applied after accumulation
    Returns float32 array (...,).
    """
    P = np.asarray(P, dtype=np.int64)
    E = np.asarray(E, dtype=np.int64)
    lead = P.shape[:-1]
    Pf, Ef = P.reshape(-1, P.shape[-1]), E.reshape(-1, E.shape[-1])
    nb = P.shape[-1]

    if isinstance(mode, Exact):
        out = np.empty(Pf.shape[0], np.float32)
        if post_scale is not None:
            sn, se = f32_parts(np.asarray(post_scale, np.float32))
            ps_n, ps_e = int(sn[0]) * int(sn[1]), int(se[0]) + int(se[1])
        for i in range(Pf.shape[0]):
            n, m = exact_sum(Pf[i], Ef[i])
            if post_scale is not None:
                n, m = n * ps_n, m + ps_e
            out[i] = round_int_to_f32(n, m)
        return out.reshape(lead)

    if isinstance(mode, FP32Seq):
        acc = np.zeros(Pf.shape[0], np.float32)
        for b in range(nb):
            term = np.ldexp(Pf[:, b].astype(np.float64), Ef[:, b].astype(np.int32)).astype(np.float32)
            acc = (acc + term).astype(np.float32)
        return _post(acc, post_scale).reshape(lead)

    if isinstance(mode, Fixed | Window):
        if isinstance(mode, Fixed):
            lsb = np.full(Pf.shape[0], mode.ref_exp - mode.frac, np.int64)
            width, saturate = mode.width, mode.saturate
        else:
            if anchor_exp is None:
                raise ValueError("Window accumulation needs anchor_exp")
            guard = mode.guard if mode.guard is not None else max(1, int(np.ceil(np.log2(nb))))
            etop = np.asarray(anchor_exp, np.int64).reshape(-1) + int_bits + guard
            lsb = etop - mode.width + 1
            width, saturate = mode.width, True
        if width > 62:
            raise ValueError("register width > 62 not supported in the vectorized path")
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        acc = np.zeros(Pf.shape[0], np.int64)
        overflow = np.zeros(Pf.shape[0], bool)
        for b in range(nb):
            sh = Ef[:, b] - lsb
            p = Pf[:, b]
            big = sh > (62 - int_bits)  # left shift would leave int64 -> certainly out of range
            left = np.where(big, 0, p << np.clip(sh, 0, 62))
            right = p >> np.clip(-sh, 0, 63)
            term = np.where(sh >= 0, left, right)
            term = np.where(big & (p != 0), np.sign(p) * (hi + 1), term)
            acc = acc + term
            ovf = (acc > hi) | (acc < lo)
            overflow |= ovf
            if saturate:
                acc = np.clip(acc, lo, hi)
            else:
                acc = ((acc - lo) & ((1 << width) - 1)) + lo
        out = np.array([round_int_to_f32(int(a), int(s)) for a, s in zip(acc, lsb, strict=True)], np.float32)
        if isinstance(mode, Window) and overflow.any():
            raise AssertionError("Window accumulator overflowed — anchor/guard analysis violated")
        return _post(out, post_scale).reshape(lead)

    if isinstance(mode, AlignTrunc):
        out = np.empty(Pf.shape[0], np.float32)
        for i in range(Pf.shape[0]):
            n, m = 0, 0
            for p, e in zip(Pf[i].tolist(), Ef[i].tolist(), strict=True):
                n, m = _align_trunc_add(n, m, int(p), int(e), mode.frac)
            out[i] = round_int_to_f32(n, m)
        return _post(out, post_scale).reshape(lead)

    if isinstance(mode, Hier):
        if ew is None or ex is None:
            raise ValueError("Hier accumulation needs per-block ew and ex")
        ewf = np.asarray(ew, np.int64).reshape(Pf.shape[0], -1)
        exf = np.asarray(ex, np.int64).reshape(Pf.shape[0], -1)
        tb = mode.tile_blocks
        guard = mode.guard if mode.guard is not None else max(1, int(np.ceil(np.log2(max(tb, 2)))))
        out = np.empty(Pf.shape[0], np.float32)
        for i in range(Pf.shape[0]):
            tiles = []  # (integer count, lsb exponent, tile anchor msb)
            for s in range(0, nb, tb):
                p_t, e_t = Pf[i, s : s + tb], Ef[i, s : s + tb]
                anchor = int(ewf[i, s : s + tb].max() + exf[i, s : s + tb].max() - 2)
                lsb = anchor + int_bits + guard - mode.window + 1
                acc = 0
                for pb, eb in zip(p_t.tolist(), e_t.tolist(), strict=True):
                    sh = eb - lsb
                    acc += pb << sh if sh >= 0 else pb >> -sh  # floor on right shift, as in the RTL
                if not -(1 << (mode.window - 1)) <= acc < (1 << (mode.window - 1)):
                    raise AssertionError("tile window overflow")
                tiles.append((acc, lsb))
            red = mode.reducer
            if red == "exact":
                m = min(lsb for _, lsb in tiles)
                n = sum(a << (lsb - m) for a, lsb in tiles)
                out[i] = round_int_to_f32(n, m)
            elif red == "fp32":
                a32 = np.float32(0.0)
                for a, lsb in tiles:
                    a32 = np.float32(a32 + round_int_to_f32(a, lsb))
                out[i] = a32
            elif isinstance(red, tuple) and red[0] == "window":
                wr = int(red[1])
                top = max(lsb + mode.window for _, lsb in tiles) + int(np.ceil(np.log2(max(len(tiles), 2))))
                rl = top - wr
                n = sum((a << (lsb - rl)) if lsb >= rl else (a >> (rl - lsb)) for a, lsb in tiles)
                out[i] = round_int_to_f32(n, rl)
            else:
                raise ValueError(f"unknown reducer {red!r}")
        return _post(out, post_scale).reshape(lead)

    if isinstance(mode, Approx):
        return _post(np.asarray(mode.fn(Pf, Ef), np.float64).astype(np.float32), post_scale).reshape(lead)

    raise TypeError(f"unknown accumulation mode {mode!r}")


def _trunc_to(n: int, m: int, lsb: int) -> int:
    """Truncate n·2^m toward zero to a multiple of 2^lsb; returns the integer count of 2^lsb units."""
    if m >= lsb:
        return n << (m - lsb)
    q = abs(n) >> (lsb - m)
    return q if n >= 0 else -q


def _align_trunc_add(n1: int, m1: int, n2: int, m2: int, frac: int) -> tuple[int, int]:
    msbs = [abs(n).bit_length() - 1 + m for n, m in ((n1, m1), (n2, m2)) if n != 0]
    if not msbs:
        return 0, 0
    lsb = max(msbs) - frac
    return _trunc_to(n1, m1, lsb) + _trunc_to(n2, m2, lsb), lsb


def _post(acc: np.ndarray, post_scale: tuple[float, float] | None) -> np.ndarray:
    if post_scale is None:
        return acc.astype(np.float32)
    s = np.float32(np.float32(post_scale[0]) * np.float32(post_scale[1]))
    return (acc.astype(np.float32) * s).astype(np.float32)
