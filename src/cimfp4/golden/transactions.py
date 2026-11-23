"""L2 <-> L3 transaction contract (also the RTL stimulus/response file schema).

One macro = ROWS × COLS compute units, each holding WEIGHT_SETS 4-bit weights in SRAM.
Field names mirror RTL port names so scoreboards compare 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightWrite:
    """Write one weight set: codes for all ROWS × COLS units plus one UE8M0 exponent per column."""

    set_idx: int
    w_codes: np.ndarray  # (ROWS, COLS) uint8 E2M1
    w_scale_exp: np.ndarray  # (COLS,) int, unbiased UE8M0 exponent of this column block
    w_scale_nan: np.ndarray | None = None  # (COLS,) bool


@dataclass
class MacroOp:
    """One compute cycle: broadcast an activation block, select a weight set, accumulate."""

    x_codes: np.ndarray  # (ROWS,) uint8 E2M1
    x_scale_exp: int  # UE8M0 exponent of this activation block
    set_idx: int  # weight set to use
    max_e_x: int  # anchor: max activation block exponent of the whole dot product
    max_e_w: np.ndarray  # (COLS,) anchor: max weight block exponent per column (static)
    acc_clear: bool = False  # clear accumulator before adding this op's term
    x_scale_nan: bool = False


@dataclass
class ColumnResult:
    """Per-op probes (RTL signal names)."""

    leaf_tc: np.ndarray  # (ROWS, COLS) int - 9 b two's-complement products ×4
    tree_sum: np.ndarray  # (COLS,) int - 14 b signed block sum
    exp_sum: np.ndarray  # (COLS,) int - e_w + e_x - 2 (9 b)
    shift: np.ndarray  # (COLS,) int - alignment amount d
    acc: np.ndarray  # (COLS,) int - W-bit accumulator after this op
    nan: np.ndarray  # (COLS,) bool
