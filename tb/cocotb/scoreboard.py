"""Column scoreboard: stimulus generation + vectorized expected values (mirrors golden.l2_macro).

`column_reference` is a vectorized single-column version of L2Macro (checked against L2Macro in
tests/golden/test_scoreboard.py); the cocotb tests compare RTL accumulators to it op by op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cimfp4.formats.codecs import e2m1_to_int_x2
from cimfp4.formats.mx_quant import quantize_mxfp4

ROWS = 32


@dataclass
class ColumnStim:
    w: np.ndarray  # (T, 32) uint8 latched weights
    x: np.ndarray  # (T, 32) uint8 activation codes
    e_w: np.ndarray  # (T,) int
    e_x: np.ndarray  # (T,) int
    max_e_w: np.ndarray  # (T,) int (constant within a dot product)
    max_e_x: np.ndarray  # (T,) int
    clear: np.ndarray  # (T,) bool - first op of each dot product
    nan: np.ndarray  # (T,) bool

    def __len__(self) -> int:
        return len(self.e_w)


def column_reference(st: ColumnStim, window: int = 40, guard: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """-> (acc (T,) int64 after each op, nan flag (T,) bool)."""
    blk = (e2m1_to_int_x2(st.w) * e2m1_to_int_x2(st.x)).sum(axis=1)
    d = (st.max_e_w - st.e_w) + (st.max_e_x - st.e_x)
    if (d < 0).any():
        raise ValueError("anchor below block exponent")
    d = np.minimum(d, window - 1)
    base = window - 14 - 1 - guard
    sh = base - d
    term = np.where(sh >= 0, blk << np.clip(sh, 0, 62), blk >> np.clip(-sh, 0, 63))
    acc = np.empty(len(st), np.int64)
    nan = np.empty(len(st), bool)
    a, n = 0, False
    for t in range(len(st)):  # sequential by nature (clear resets)
        a = (0 if st.clear[t] else a) + int(term[t])
        n = (False if st.clear[t] else n) or bool(st.nan[t])
        acc[t], nan[t] = a, n
    lim = 1 << (window - 1)
    assert ((acc >= -lim) & (acc < lim)).all(), "window overflow in reference"
    return acc, nan


def int4sm_column_reference(st, window=40, guard=7):
    """D2 INT4-SM column (lossy surrogate) on the macro stimulus: block sums from the golden L2 model
    (BitSerialInt4Column), same alignment / W-bit window as column_reference (14-bit block-sum slot)."""
    from cimfp4.golden.l2_bitserial_int4 import BitSerialInt4Column

    def sm(c):
        c = np.asarray(c, np.int64)
        return np.where((c >> 3) & 1, -(c & 7), c & 7)

    blk = BitSerialInt4Column().block_sums(sm(st.w), sm(st.x))[0]
    assert (blk == (sm(st.w) * sm(st.x)).sum(axis=1)).all()  # golden == integer contract
    d = np.minimum((st.max_e_w - st.e_w) + (st.max_e_x - st.e_x), window - 1)
    assert (d >= 0).all()
    sh = window - 14 - 1 - guard - d
    term = np.where(sh >= 0, blk << np.clip(sh, 0, 62), blk >> np.clip(-sh, 0, 63))
    acc, a = np.empty(len(st), np.int64), 0
    for t in range(len(st)):
        a = (0 if st.clear[t] else a) + int(term[t])
        acc[t] = a
    lim = 1 << (window - 1)
    assert ((acc >= -lim) & (acc < lim)).all(), "window overflow in reference"
    return acc


def random_stim(
    rng: np.random.Generator, n_dots: int, max_blocks: int = 128, nan_rate: float = 0.0
) -> ColumnStim:
    """LLM-like random dot products: quantized Gaussian/heavy-tailed operands, varied block exponents."""
    parts = {k: [] for k in ("w", "x", "e_w", "e_x", "max_e_w", "max_e_x", "clear", "nan")}
    for _ in range(n_dots):
        nb = int(rng.integers(1, max_blocks + 1))
        k = nb * ROWS
        wv = rng.standard_normal(k) * (rng.random(k) > 0.1)
        xv = rng.standard_t(3, k) * np.exp(rng.normal(0, 1.0))
        qw, qx = quantize_mxfp4(wv[None]), quantize_mxfp4(xv[None])
        ew = np.clip(qw.scale_exp[0].astype(np.int64) + rng.integers(-4, 5), -127, 127)
        ex = np.clip(qx.scale_exp[0].astype(np.int64) + rng.integers(-4, 5), -127, 127)
        parts["w"].append(qw.codes[0].reshape(nb, ROWS))
        parts["x"].append(qx.codes[0].reshape(nb, ROWS))
        parts["e_w"].append(ew)
        parts["e_x"].append(ex)
        parts["max_e_w"].append(np.full(nb, ew.max()))
        parts["max_e_x"].append(np.full(nb, ex.max()))
        parts["clear"].append(np.arange(nb) == 0)
        parts["nan"].append(rng.random(nb) < nan_rate)
    return ColumnStim(**{k: np.concatenate(v) for k, v in parts.items()})


def corner_stim() -> ColumnStim:
    """R07 §3.2 corner suite, one short dot product each."""
    P6, N6, Z, NZ = 0b0111, 0b1111, 0b0000, 0b1000  # +6, -6, +0, -0
    sub = 0b0001  # +0.5 (subnormal)
    rows = []

    def dot(w_rows, x_rows, ew, ex, nan=None):
        nb = len(w_rows)
        rows.append(
            dict(
                w=np.array(w_rows, np.uint8),
                x=np.array(x_rows, np.uint8),
                e_w=np.array(ew),
                e_x=np.array(ex),
                max_e_w=np.full(nb, max(ew)),
                max_e_x=np.full(nb, max(ex)),
                clear=np.arange(nb) == 0,
                nan=np.array(nan if nan is not None else [False] * nb),
            )
        )

    full = lambda v: [v] * ROWS  # noqa: E731
    dot([full(P6)], [full(P6)], [0], [0])  # +4608
    dot([full(P6)], [full(N6)], [0], [0])  # -4608
    dot([full(P6)] * 4, [full(P6), full(N6)] * 2, [0] * 4, [0] * 4)  # alternating max
    dot([[P6, N6] * 16], [full(P6)], [0], [0])  # alternating within block
    dot([full(Z)], [full(Z)], [0], [0])  # all zero
    dot([[P6, NZ] * 16], [[NZ, N6] * 16], [0], [0])  # +-0 mixes
    dot([[P6] + [Z] * 31], [full(P6)], [0], [0])  # one-hot
    dot([full(sub)], [full(sub)], [3], [-2])  # subnormal only
    dot([full(P6), full(sub)], [full(P6), full(P6)], [-127, 127], [127, -127])  # UE8M0 0x00 / 0xFE extremes
    dot([full(P6)] * 2, [full(N6)] * 2, [5, 5], [5, 5], nan=[False, True])  # UE8M0 0xFF -> NaN flag
    dot([full(N6)] * 128, [full(P6)] * 128, [0] * 128, [0] * 128)  # 128 x -4608: window lower bound
    return ColumnStim(**{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})


def pack(rows: np.ndarray, width: int) -> list[int]:
    """(T, 32) small ints -> list of T python ints with row r at bits [width*r +: width]."""
    rows = np.asarray(rows, dtype=object)
    shifts = [1 << (width * r) for r in range(rows.shape[1])]
    return [int(sum(int(v) * s for v, s in zip(row, shifts, strict=True))) for row in rows]


def row_signals(x: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Activation codes -> packed (xp, xp3, xs) buses, as the row drivers would produce them."""
    xi = e2m1_to_int_x2(x)
    mag = np.abs(xi)
    xs = ((x >> 3) & 1) & (mag != 0)
    return pack(mag, 4), pack(3 * mag, 6), pack(xs, 1)
