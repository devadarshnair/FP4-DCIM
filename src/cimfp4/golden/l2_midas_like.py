"""L2 model of baseline (b): MIDAS-like bit-serial cyclic-alignment macro (WP-37; choi2026midas §III-IV).

Mechanism modelled (approximation off, i.e. SADB/SSCR disabled):
  * vertical mapping: one LCA per (input row r, output column k) element pair; weights stored as sign,
    exponent and (mbit+1)-bit mantissa incl. hidden bit (STCA: offline spatial cyclic alignment);
  * a down-counter G-CNT walks bit significance c from the top (2·E_max + mbit, fixed: MX => max exponent
    sum ≈ constant, eq. 4) down to the macro's minimum exponent sum (MEPD early stop, §IV-D step 4);
  * LCA (r, k) is valid when 0 ≤ TAO = c - (E_x,r + E_w,rk) ≤ mbit (eq. 5-6, TAOR eq. 7) and then
    contributes input mantissa bit j = TAO times the weight mantissa (3 BLs -> MFA, sign-magnitude);
  * per cycle, the column sum Σ_r ±x_j·M_w (LPAT + HPAT) enters a shift accumulator acc = 2·acc + colsum.
The block integer equals Σ M_x·M_w·2^(E_x+E_w-2) (·2^-2·off scale), i.e. exactly the L1 block term
(×4 integer for E2M1), so functional == L1 with approximation off (WP-37 done criterion).

Element decode: value = M · 2^(E_eff - off), M = (hidden << mbit) | m, E_eff = max(E, 1) (subnormals share
the min-normal exponent; required for exactness). off = 2 (E2M1), 5 (E3M2).

Activity counters (feed power/midas_model.py): cycles per op, valid LCA-cycles (pre-charge gating),
active WL row-cycles (WGSG row gating), BL discharges (1-products), MFA/tree valid inputs, counter steps.
Assumptions (stated in the validation record): zero elements do not delay the early stop
(`zero_holds_early_stop=False`); inter-block accumulation (IAU) is outside the macro and is FP32 by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMATS = {
    "mxfp4": {"ebits": 2, "mbits": 1, "off": 2, "code_bits": 4},  # E2M1
    "mxfp6": {"ebits": 3, "mbits": 2, "off": 5, "code_bits": 6},  # E3M2
}


def decode(codes: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element codes -> (sign ∈ {±1}, mantissa M incl. hidden bit, effective exponent E_eff ≥ 1)."""
    f = FORMATS[fmt]
    c = np.asarray(codes, dtype=np.int64)
    mb, eb = f["mbits"], f["ebits"]
    m = c & ((1 << mb) - 1)
    e = (c >> mb) & ((1 << eb) - 1)
    s = np.where((c >> (mb + eb)) & 1, -1, 1)
    hidden = (e > 0).astype(np.int64)
    return s, (hidden << mb) | m, np.maximum(e, 1)


def element_values(codes: np.ndarray, fmt: str) -> np.ndarray:
    s, mant, e = decode(codes, fmt)
    return s * mant * np.ldexp(1.0, (e - FORMATS[fmt]["off"]).astype(np.int32))


@dataclass
class MidasLikeMacro:
    """One CA-CIM macro: 32 input rows (one MX block) × up to 16 output columns."""

    fmt: str = "mxfp4"
    rows: int = 32
    cols: int = 16
    zero_holds_early_stop: bool = False
    early_stop: bool = True  # MEPD; False = run the full significance range every op (ablation baseline)

    @property
    def top(self) -> int:
        f = FORMATS[self.fmt]
        return 2 * ((1 << f["ebits"]) - 1) + f["mbits"]

    @property
    def bottom(self) -> int:
        return 2  # min E_eff sum

    def run(self, w_codes: np.ndarray, x_codes: np.ndarray) -> tuple[np.ndarray, dict]:
        """w_codes (C, rows) weights of the stationary set (column k = output channel); x_codes (T, rows).

        Returns (block integers P (T, C), activity dict). P · 2^(2 - 2·off) = Σ_r x_r·w_rk exactly.
        """
        mb = FORMATS[self.fmt]["mbits"]
        w_codes = np.asarray(w_codes)
        x_codes = np.asarray(x_codes)
        ncol, t_ops = w_codes.shape[0], x_codes.shape[0]
        assert w_codes.shape[1] == x_codes.shape[1] == self.rows and ncol <= self.cols
        sw, mw, ew = decode(w_codes, self.fmt)  # (C, R)
        sx, mx, ex = decode(x_codes, self.fmt)  # (T, R)
        esum = ex[:, None, :] + ew[None, :, :]  # (T, C, R)
        nonzero = (mx[:, None, :] * mw[None, :, :]) != 0
        if self.zero_holds_early_stop:
            c_end = esum.reshape(t_ops, -1).min(1)
        else:
            c_end = np.where(nonzero, esum, self.top).reshape(t_ops, -1).min(1)
        if not self.early_stop:
            c_end = np.full(t_ops, self.bottom)
        c_end = np.minimum(c_end, self.top)
        sign = sx[:, None, :] * sw[None, :, :]
        acc = np.zeros((t_ops, ncol), np.int64)
        act = {k: np.zeros(t_ops, np.int64) for k in
               ("cycles", "valid_lca_cycles", "active_row_cycles", "bl_ones", "tree_valid_inputs")}
        for c in range(self.top, self.bottom - 1, -1):
            live = c >= c_end  # (T,) op still running this cycle
            tao = c - esum  # (T, C, R)
            valid = (tao >= 0) & (tao <= mb) & live[:, None, None]
            xbit = (mx[:, None, :] >> np.clip(tao, 0, mb)) & 1
            pp = np.where(valid, xbit * mw[None, :, :], 0)  # MFA output magnitude per LCA
            colsum = (sign * pp).sum(-1)  # LPAT + HPAT, (T, C)
            acc = np.where(live[:, None], 2 * acc + colsum, acc)
            act["cycles"] += live
            act["valid_lca_cycles"] += valid.sum((1, 2))
            act["active_row_cycles"] += valid.any(1).sum(-1)
            wbits = (mw[None, :, :, None] >> np.arange(mb + 1)) & 1  # (1, C, R, mb+1)
            act["bl_ones"] += (valid[..., None] & (xbit[..., None] == 1) & (wbits == 1)).sum((1, 2, 3))
            act["tree_valid_inputs"] += valid.sum((1, 2))
        # rescale from the early-stop point to the common origin (sum of E_eff = bottom)
        p = acc << (c_end - self.bottom)[:, None]
        act["c_end"] = c_end
        act["lca_cycles_total"] = act["cycles"] * ncol * self.rows
        return p, act


def block_integer_reference(w_codes: np.ndarray, x_codes: np.ndarray, fmt: str) -> np.ndarray:
    """Direct Σ_r x_r·w_rk in units of 2^(2·off - 2) (exact via float64 for these small integers)."""
    scale = 2.0 ** (2 * FORMATS[fmt]["off"] - 2)
    v = element_values(x_codes, fmt) @ element_values(w_codes, fmt).T * scale
    out = np.rint(v).astype(np.int64)
    assert np.array_equal(out, v), "non-integer reference"
    return out
