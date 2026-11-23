"""Operand statistics that drive hardware power.

All functions take FP4Blocks: weights (N, K) and activations (T, K), K blocked.

- code histograms                         -> which E2M1 codes the leaf multipliers actually see
- P(W=0), P(A=0), P(W·A=0)                -> zero-gating / AND-OR multiplier idle fraction
- per-block product sign counts            -> dual-polarity tree balance
- block partial-sum magnitudes |S_b|       -> effective tree width used (≤ 4608 for MX)
- scale-exponent histograms, per-dot spans -> accumulator window sizing
- consecutive-row toggle rate              -> α of the activation broadcast bus (WP-60, WP-28)
- bit-plane density of |int| magnitudes    -> work of a bit-serial baseline (C3)
"""

from __future__ import annotations

import numpy as np

from ..formats.codecs import e2m1_to_int_x2
from ..formats.types import FP4Block

_POP4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.int64)


def code_hist(q: FP4Block) -> np.ndarray:
    """Fraction of each of the 16 E2M1 codes."""
    return np.bincount(q.codes.ravel(), minlength=16) / q.codes.size


def zero_fractions(qw: FP4Block, qa: FP4Block) -> dict:
    """Exact P(W=0), P(A=0) and P(W·A=0) over all (n, t, k) MAC pairs of W·Aᵀ."""
    zw = (qw.codes & 7) == 0  # (N, K)
    za = (qa.codes & 7) == 0  # (T, K)
    nzw, nza = (~zw).mean(0), (~za).mean(0)  # per-k nonzero fractions
    return {
        "p_w_zero": float(zw.mean()),
        "p_a_zero": float(za.mean()),
        "p_prod_zero": float(1.0 - np.mean(nzw * nza)),
    }


def sample_pairs(n: int, t: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return rng.integers(0, n, n_pairs), rng.integers(0, t, n_pairs)


def block_products(qw: FP4Block, qa: FP4Block, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Signed products ×4 for sampled (w-row, a-row) pairs -> (n_pairs, n_blocks, block) int64."""
    iw = e2m1_to_int_x2(qw.blocked_codes()[rows])
    ia = e2m1_to_int_x2(qa.blocked_codes()[cols])
    return iw * ia


def sign_balance(prod: np.ndarray) -> dict:
    """Per-block counts of positive / negative / zero products."""
    pos, neg = (prod > 0).sum(-1), (prod < 0).sum(-1)
    zero = prod.shape[-1] - pos - neg
    nonzero = np.maximum(pos + neg, 1)
    return {
        "mean_pos": float(pos.mean()),
        "mean_neg": float(neg.mean()),
        "mean_zero": float(zero.mean()),
        "neg_share_of_nonzero": float((neg / nonzero).mean()),
        "hist_neg": np.bincount(neg.ravel(), minlength=prod.shape[-1] + 1) / neg.size,
    }


def block_partials(prod: np.ndarray) -> dict:
    """|S_b| distribution and the per-polarity partial sums the dual-polarity trees would see."""
    s = prod.sum(-1)
    sp, sn = np.where(prod > 0, prod, 0).sum(-1), np.where(prod < 0, -prod, 0).sum(-1)
    q = [50, 90, 99, 99.9, 100]
    return {
        "abs_sum_pct": dict(zip(q, np.percentile(np.abs(s), q).tolist(), strict=True)),
        "pos_sum_pct": dict(zip(q, np.percentile(sp, q).tolist(), strict=True)),
        "neg_sum_pct": dict(zip(q, np.percentile(sn, q).tolist(), strict=True)),
        "bits_needed_p999": int(np.ceil(np.log2(np.percentile(np.abs(s), 99.9) + 1))) + 1,
    }


def scale_exponents(q: FP4Block) -> np.ndarray:
    """Per-block scale exponent: MX UE8M0 exponent; NVFP4 E4M3 exponent (floor log2 of the scale)."""
    if q.fmt == "mxfp4":
        return q.scale_exp.astype(np.int64)
    from ..formats.codecs import e4m3_significand

    u, k = e4m3_significand(q.scale_e4m3)
    return np.where(u > 0, k + np.floor(np.log2(np.maximum(u, 1))).astype(np.int64), np.iinfo(np.int32).min)


def exponent_spans(qw: FP4Block, qa: FP4Block, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Per sampled dot product: max_b(e_w+e_a) - min_b(e_w+e_a) over its blocks (skips zero scales)."""
    ew, ea = scale_exponents(qw)[rows], scale_exponents(qa)[cols]
    e = ew + ea
    valid = (ew > np.iinfo(np.int32).min) & (ea > np.iinfo(np.int32).min)
    hi = np.where(valid, e, np.iinfo(np.int64).min).max(-1)
    lo = np.where(valid, e, np.iinfo(np.int64).max).min(-1)
    return np.where(valid.any(-1), hi - lo, 0)


def toggle_rate(qa: FP4Block) -> dict:
    """Mean fraction of toggling bits between consecutive activation rows (broadcast-bus α).

    'code' = 4-bit E2M1 code bus; 'mag_sm' = sign + 4-bit integer magnitude (|x|·2 ∈ [0,12]),
    i.e. the sign-magnitude row bus of the v2 row driver.
    """
    c = qa.codes.astype(np.int64)
    if c.shape[0] < 2:
        return {"code": float("nan"), "mag_sm": float("nan")}
    code_t = _POP4[c[1:] ^ c[:-1]].mean() / 4
    mag = np.abs(e2m1_to_int_x2(qa.codes)) | ((c >> 3) << 4)
    sm_t = np.array([bin(v).count("1") for v in range(32)])[mag[1:] ^ mag[:-1]].mean() / 5
    return {"code": float(code_t), "mag_sm": float(sm_t)}


def bitplane_density(q: FP4Block) -> np.ndarray:
    """Fraction of ones in each bit-plane (LSB first) of the 4-bit integer magnitude |x|·2 ∈ [0, 12]."""
    mag = np.abs(e2m1_to_int_x2(q.codes))
    return np.array([((mag >> b) & 1).mean() for b in range(4)])
