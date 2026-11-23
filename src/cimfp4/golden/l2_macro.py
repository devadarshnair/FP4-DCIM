"""L2 golden model: cycle/word-accurate v2 macro (MX mode) with toggle counters and storage events.

Structure mirrors the RTL (docs/spec/numerics.md): row drivers broadcast X' = 2|x|, 3X' and s_x;
each compute unit reads its 4-bit weight latch (reloaded from the SRAM set on a set switch) and forms
a 9-bit two's-complement product; each column sums 32 products (14 b), forms e_w + e_x - 2, computes the
alignment shift d = (max_e_w,col - e_w) + (max_e_x - e_x) and adds the aligned term into a W-bit window
accumulator whose LSB is anchored before compute. Bit-identical to golden.l1_dot with Window(W, guard).

Toggle counters (bit flips between consecutive ops) feed the switching-activity power proxy (WP-28/60).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..formats.codecs import e2m1_to_int_x2
from ..formats.types import FP4Block
from ..rtlgen.pkg import CimParams
from .accum import round_int_to_f32
from .transactions import ColumnResult, MacroOp, WeightWrite


def popcount_tc(a: np.ndarray, width: int) -> np.ndarray:
    """Popcount of the two's-complement ``width``-bit encoding of each int."""
    u = np.asarray(a, dtype=np.int64) & ((1 << width) - 1)
    cnt = np.zeros(u.shape, np.int64)
    while u.any():
        cnt += u & 1
        u = u >> 1
    return cnt


@dataclass
class ToggleCounters:
    row_bcast: int = 0  # 11 wires × ROWS (X' 4 b, 3X' 6 b, sign 1 b)
    weight_latch: int = 0  # 4 b × ROWS × COLS (changes only on set switch)
    leaf_out: int = 0  # 9 b × ROWS × COLS
    tree_out: int = 0  # 14 b × COLS (CPA output; CSA internals are counted by the bit-level model)
    acc_reg: int = 0  # W b × COLS
    set_switches: int = 0
    weight_bits_written: int = 0
    ops: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class L2Macro:
    params: CimParams = field(default_factory=CimParams)
    guard: int = 7  # ceil(log2(n_blocks)) of the dot products being computed (must match L1 Window)
    trace: list = field(default_factory=list)  # optional per-op signal values for brute-force checks
    keep_trace: bool = False

    def __post_init__(self) -> None:
        p = self.params
        self.R, self.C, self.M, self.W = p.rows, p.cols, p.weight_sets, p.window
        d = p.derived
        self.int_bits = d["MX_SUM_S_BITS"]
        self.sets = np.zeros((self.M, self.R, self.C), np.uint8)
        self.set_exp = np.zeros((self.M, self.C), np.int64)
        self.set_nan = np.zeros((self.M, self.C), bool)
        self.latch = np.zeros((self.R, self.C), np.uint8)
        self.latch_set = -1
        self.acc = np.zeros(self.C, np.int64)
        self.nan = np.zeros(self.C, bool)
        self.cnt = ToggleCounters()
        self._prev = {
            "bcast": np.zeros((self.R, 3), np.int64),
            "leaf": np.zeros((self.R, self.C), np.int64),
            "tree": np.zeros(self.C, np.int64),
            "acc": np.zeros(self.C, np.int64),
        }

    # ------------------------------------------------------------------ storage
    def write(self, w: WeightWrite) -> None:
        if not 0 <= w.set_idx < self.M:
            raise ValueError("set_idx out of range")
        self.sets[w.set_idx] = w.w_codes
        self.set_exp[w.set_idx] = w.w_scale_exp
        self.set_nan[w.set_idx] = False if w.w_scale_nan is None else w.w_scale_nan
        self.cnt.weight_bits_written += 4 * self.R * self.C + 8 * self.C

    # ------------------------------------------------------------------ compute
    def step(self, op: MacroOp) -> ColumnResult:
        C, W = self.C, self.W
        if op.set_idx != self.latch_set:  # SRAM read -> leaf latches
            new = self.sets[op.set_idx]
            self.cnt.weight_latch += int(popcount_tc(new ^ self.latch, 4).sum())
            self.latch, self.latch_set = new.copy(), op.set_idx
            self.cnt.set_switches += 1
        # row drivers
        xi = e2m1_to_int_x2(op.x_codes)  # signed 2x, [-12, 12]
        xmag, xs = np.abs(xi), (op.x_codes >> 3).astype(np.int64) & ((op.x_codes & 7) != 0)
        bcast = np.stack([xmag, 3 * xmag, xs], axis=1)
        self.cnt.row_bcast += int(
            (
                popcount_tc(bcast[:, 0] ^ self._prev["bcast"][:, 0], 4)
                + popcount_tc(bcast[:, 1] ^ self._prev["bcast"][:, 1], 6)
                + (bcast[:, 2] ^ self._prev["bcast"][:, 2])
            ).sum()
        )
        # leaves: 9 b two's complement products ×4
        leaf = e2m1_to_int_x2(self.latch) * xi[:, None]
        self.cnt.leaf_out += int(popcount_tc(leaf ^ self._prev["leaf"], 9).sum())
        # column tree + exponent path
        tree = leaf.sum(axis=0)
        e_w = self.set_exp[self.latch_set]
        exp_sum = e_w + op.x_scale_exp - 2
        d = (np.asarray(op.max_e_w) - e_w) + (op.max_e_x - op.x_scale_exp)
        if (d < 0).any():
            raise ValueError("anchor below block exponent: max_e_w/max_e_x must be maxima")
        base = W - self.int_bits - 1 - self.guard  # left shift of a d=0 term into the window
        sh = base - d
        term = np.where(sh >= 0, tree << np.clip(sh, 0, 62), tree >> np.clip(-sh, 0, 63))
        if op.acc_clear:
            self.acc = np.zeros(C, np.int64)
            self.nan = np.zeros(C, bool)
        self.acc = self.acc + term
        if ((self.acc >= (1 << (W - 1))) | (self.acc < -(1 << (W - 1)))).any():
            raise AssertionError("window accumulator overflow — anchor/guard analysis violated")
        self.nan |= self.set_nan[self.latch_set] | op.x_scale_nan
        self.cnt.tree_out += int(popcount_tc(tree ^ self._prev["tree"], self.int_bits).sum())
        self.cnt.acc_reg += int(popcount_tc(self.acc ^ self._prev["acc"], W).sum())
        self._prev.update(bcast=bcast, leaf=leaf, tree=tree, acc=self.acc.copy())
        self.cnt.ops += 1
        res = ColumnResult(
            leaf_tc=leaf, tree_sum=tree, exp_sum=exp_sum, shift=d, acc=self.acc.copy(), nan=self.nan.copy()
        )
        if self.keep_trace:
            self.trace.append(
                {
                    "bcast": bcast,
                    "leaf": leaf,
                    "tree": tree,
                    "acc": self.acc.copy(),
                    "latch": self.latch.copy(),
                }
            )
        return res

    def read(self, max_e_w: np.ndarray, max_e_x: int) -> np.ndarray:
        """Normalize the window accumulators to fp32 (NaN where flagged)."""
        etop = np.asarray(max_e_w) + max_e_x - 2 + self.int_bits + self.guard
        lsb = etop - self.W + 1
        out = np.array(
            [round_int_to_f32(int(a), int(s)) for a, s in zip(self.acc, lsb, strict=True)], np.float32
        )
        return np.where(self.nan, np.float32(np.nan), out)


def run_gemv(
    qw: FP4Block, qx: FP4Block, params: CimParams | None = None, macro: L2Macro | None = None
) -> tuple[np.ndarray, L2Macro]:
    """W (N ≤ COLS rows, K) · x (1, K) on one macro; weight blocks streamed through the M sets.

    Returns (y (N,) float32, macro). Matches l1_dot(qw, qx, Window(params.window)) bitwise.
    """
    params = params or CimParams()
    if qw.fmt != "mxfp4" or qx.fmt != "mxfp4":
        raise NotImplementedError("L2 currently models MX mode (NVFP4 half-trees: WP-33)")
    n, k = qw.codes.shape
    nb = k // params.rows
    if n > params.cols or qx.codes.shape != (1, k):
        raise ValueError("expected W (N<=COLS, K) and x (1, K)")
    guard = max(1, int(np.ceil(np.log2(nb))))
    mac = macro or L2Macro(params=params, guard=guard)
    wc = np.zeros((params.cols, k), np.uint8)
    wc[:n] = qw.codes
    we = np.zeros((params.cols, nb), np.int64)
    we[:n] = qw.scale_exp
    max_e_w = we[:, :].max(axis=1)
    max_e_w[n:] = 0
    max_e_x = int(qx.scale_exp.max())
    for b in range(nb):
        s = b % params.weight_sets
        mac.write(
            WeightWrite(
                set_idx=s,
                w_codes=wc[:, b * params.rows : (b + 1) * params.rows].T.copy(),
                w_scale_exp=we[:, b],
            )
        )
        mac.step(
            MacroOp(
                x_codes=qx.codes[0, b * params.rows : (b + 1) * params.rows],
                x_scale_exp=int(qx.scale_exp[0, b]),
                set_idx=s,
                max_e_x=max_e_x,
                max_e_w=max_e_w,
                acc_clear=(b == 0),
            )
        )
    return mac.read(max_e_w, max_e_x)[:n], mac


# ------------------------------------------------------------------------------------- NVFP4 path
def nv_column_reference(
    w_codes, x_codes, u_w, k_w, u_x, k_x, max_k_w, max_k_x, clear, window: int = 40, guard: int = 7
) -> np.ndarray:
    """Word-level NVFP4 column model (WP-31), mirroring rtl/src/column.sv in NV mode.

    Per op (32 rows = two 16-element NVFP4 blocks): S_h = Σ products×4 over rows 16h..16h+15 (13 b),
    P_h = S_h·u_w,h·u_x,h (21 b signed), d_h = (max_k_w - k_w,h) + (max_k_x - k_x,h) saturated at W-1,
    term_h = P_h << (BASE_NV - d_h) (or floor >>), BASE_NV = W - 21 - 1 - guard; acc += term_0 + term_1.
    Arrays: w_codes, x_codes (T, 32); u_*, k_* (T, 2); max_k_* (T,); clear (T,). Returns acc (T,).
    """
    prod = e2m1_to_int_x2(np.asarray(w_codes)) * e2m1_to_int_x2(np.asarray(x_codes))
    s = np.stack([prod[:, :16].sum(1), prod[:, 16:].sum(1)], axis=1)
    p = s * np.asarray(u_w, np.int64) * np.asarray(u_x, np.int64)
    d = (np.asarray(max_k_w)[:, None] - np.asarray(k_w)) + (np.asarray(max_k_x)[:, None] - np.asarray(k_x))
    if (d < 0).any():
        raise ValueError("anchor below block exponent")
    d = np.minimum(d, window - 1)
    sh = (window - 21 - 1 - guard) - d
    term = np.where(sh >= 0, p << np.clip(sh, 0, 62), p >> np.clip(-sh, 0, 63)).sum(axis=1)
    acc, a = np.empty(len(term), np.int64), 0
    for t in range(len(term)):
        a = (0 if clear[t] else a) + int(term[t])
        acc[t] = a
    lim = 1 << (window - 1)
    assert ((acc >= -lim) & (acc < lim)).all(), "NV window overflow"
    return acc


def nv_column_nan(w_nan, x_nan, clear) -> np.ndarray:
    """Sticky NaN flag per op for an NVFP4 column (same semantics as column.sv `r_nan`/`nan_out` and the
    MX path): flag = (clear ? 0 : previous) | w_nan | x_nan, where w_nan / x_nan are the per-op RF nan bit
    and op_x_nan. Arrays (T,) bool; returns (T,) bool."""
    out, f = np.empty(len(clear), bool), False
    for t in range(len(clear)):
        f = (False if clear[t] else f) or bool(w_nan[t]) or bool(x_nan[t])
        out[t] = f
    return out


def nv_ops_from_blocks(qw: FP4Block, qx: FP4Block, row: int = 0, col: int = 0) -> dict:
    """One NVFP4 dot product (weight row `row`, activation row `col`) as column ops of 32 rows."""
    from ..formats.codecs import e4m3_scale_fields

    uw, kw, nw = e4m3_scale_fields(qw.scale_e4m3[row])
    ux, kx, nx = e4m3_scale_fields(qx.scale_e4m3[col])
    nb = qw.num_blocks
    if nb % 2:
        raise ValueError("NV column ops need an even number of 16-element blocks")
    t = nb // 2
    return {
        "w": qw.codes[row].reshape(t, 32),
        "x": qx.codes[col].reshape(t, 32),
        "u_w": uw.reshape(t, 2),
        "k_w": kw.reshape(t, 2),
        "u_x": ux.reshape(t, 2),
        "k_x": kx.reshape(t, 2),
        "max_k_w": np.full(t, kw.max()),
        "max_k_x": np.full(t, kx.max()),
        "clear": np.arange(t) == 0,
        # per-op NaN-scale flags (macro RF nan bit = either half; op_x_nan = either activation half)
        "w_nan": nw.reshape(t, 2).any(axis=1),
        "x_nan": nx.reshape(t, 2).any(axis=1),
    }
