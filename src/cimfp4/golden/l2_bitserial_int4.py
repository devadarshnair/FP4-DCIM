"""L2 model of the D2 surrogate: MIDAS-inspired bit-serial MXINT4-SM column (d2_int4_contract.md §4).

Operands are sign + 3-bit magnitude (MXINT4-SM). Weights are stationary bit-columns of the magnitude (3 per
element) plus a sign latch; activation magnitudes are fed LSB-first over 3 cycles. Following MIDAS's
sign-magnitude SACA dataflow, each one-bit product x_i & w_j is routed by the XOR of the two signs to a
positive or a negative 32-input popcount tree (rtlgen tree, w_in = 1 - the same generated netlist as the
INT5 baseline), and the block sum is Σ_i Σ_j (pc⁺_ij - pc⁻_ij) << (i + j). No per-element exponent
alignment is needed inside a block (shared power-of-two scale).

Compared with the exact INT5 control (`l2_bitserial.py`: 5 two's-complement cycles × 5 weight bit-columns =
25 tree evaluations per block) this needs 3 cycles × 3 bit-columns × 2 polarities = 18 tree evaluations per
block, in 3 instead of 5 cycles. Block sums are integers in [-1568, 1568] (12-bit signed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rtlgen.tree import build_column_tree, evaluate, toggle_counts

MAG_BITS = 3
CYCLES_PER_BLOCK = MAG_BITS


def mag_bits(m: np.ndarray) -> np.ndarray:
    """Signed magnitudes in [-7, 7] -> (..., 3) bool magnitude bits, LSB first."""
    a = np.abs(np.asarray(m, dtype=np.int64))
    if a.max(initial=0) > 7:
        raise ValueError("INT4-SM magnitude out of range")
    return ((a[..., None] >> np.arange(MAG_BITS)) & 1).astype(bool)


@dataclass
class BitSerialInt4Column:
    tree: object = field(default_factory=lambda: build_column_tree(1, "unsigned"))

    def block_sums(self, w_m: np.ndarray, x_m: np.ndarray, count_toggles: bool = False):
        """w_m, x_m: (T, 32) signed INT4-SM magnitudes per block op -> (block sums (T,), toggles or None)."""
        w3, x3 = mag_bits(w_m), mag_bits(x_m)  # (T, 32, 3)
        neg = (np.asarray(w_m) < 0) ^ (np.asarray(x_m) < 0)  # product sign per row
        t_ops = w3.shape[0]
        sums = np.zeros(t_ops, np.int64)
        tog = {"and_nodes": 0, "tree_cells": 0, "tree_inputs": 0, "tree_evals": 0}
        for j in range(MAG_BITS):  # weight magnitude bit-column
            pp = x3.transpose(0, 2, 1) & w3[:, None, :, j]  # (T, 3 cycles, 32 rows)
            for polarity, sel in ((1, ~neg), (-1, neg)):
                stream = (pp & sel[:, None, :]).reshape(t_ops * MAG_BITS, 32)
                nets, out = evaluate(self.tree, stream.astype(np.int64))
                pc = out["m"].reshape(t_ops, MAG_BITS)
                if count_toggles:
                    tog["and_nodes"] += int(np.count_nonzero(stream[1:] != stream[:-1]))
                    tc = toggle_counts(self.tree, nets)
                    tog["tree_cells"] += tc["cells_total"]
                    tog["tree_inputs"] += tc["inputs"]
                    tog["tree_evals"] += stream.shape[0]
                for i in range(MAG_BITS):
                    sums += polarity * (pc[:, i] << (i + j))
        return sums, (tog if count_toggles else None)
