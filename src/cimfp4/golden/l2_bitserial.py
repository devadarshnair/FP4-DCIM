"""L2 model of baseline (a): TSMC-style bit-serial INT5 DCIM column (WP-36).

E2M1 -> INT5 is lossless (integerize: |2v| ≤ 12): weights W'' = 2w ∈ [-12, 12] stored as five static
bit-columns; activations X'' = 2x fed LSB-first over 5 cycles (bit 4 weighted -16, two's complement).
Per cycle i and weight bit j, 32 one-bit products (AND/NOR in the bitcell) are popcounted by a 32-input
unsigned compressor tree (rtlgen.tree, w_in = 1); a combiner forms S_i = Σ_j ±(pc_ij << j) (j = 4
negative) and a shift-add accumulator forms the block sum Σ_i ±(S_i << i) (i = 4 negative) = Σ W''X''.
The block sum then goes through the *same* exponent alignment and W-bit window as v2, so results are
bit-identical to L1 Window(W) and to the v2 column.

Toggle counters use the generated netlist (bit-level evaluation) so that the v2/bit-serial switching
comparison (R03: ~2× fewer toggles per MAC for bit-parallel) is reproduced structurally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..formats.codecs import e2m1_to_int_x2
from ..rtlgen.tree import build_column_tree, evaluate, toggle_counts

BITS = 5
CYCLES_PER_BLOCK = BITS  # activation bit-serial cycles per 32-element block


def int5_bits(v: np.ndarray) -> np.ndarray:
    """Signed ints in [-16, 15] -> (..., 5) bool, two's complement, LSB first."""
    u = np.asarray(v, dtype=np.int64) & 0x1F
    return ((u[..., None] >> np.arange(BITS)) & 1).astype(bool)


@dataclass
class BitSerialColumn:
    window: int = 40
    guard: int = 7
    tree: object = field(default_factory=lambda: build_column_tree(1, "unsigned"))

    def block_sums(self, w_codes: np.ndarray, x_codes: np.ndarray, count_toggles: bool = False):
        """w_codes, x_codes: (T, 32) per block op -> (block sums (T,), toggle dict or None).

        Simulates the 5 cycles × 5 weight-bit trees explicitly (bit-level) to prove the dataflow.
        """
        w5 = int5_bits(e2m1_to_int_x2(w_codes))  # (T, 32, 5)
        x5 = int5_bits(e2m1_to_int_x2(x_codes))  # (T, 32, 5)
        t_ops = w_codes.shape[0]
        sums = np.zeros(t_ops, np.int64)
        tog = {"and_nodes": 0, "tree_cells": 0, "tree_inputs": 0}
        for j in range(BITS):  # weight bit-column (one tree each)
            # cycle stream for this tree: (T*5, 32) one-bit products, activation bit i per cycle
            pp = (x5.transpose(0, 2, 1) & w5[:, None, :, j]).reshape(t_ops * BITS, 32)
            nets, out = evaluate(self.tree, pp.astype(np.int64))
            pc = out["m"].reshape(t_ops, BITS)  # popcounts (T, 5 cycles)
            if count_toggles:
                tog["and_nodes"] += int(np.count_nonzero(pp[1:] != pp[:-1]))
                tc = toggle_counts(self.tree, nets)
                tog["tree_cells"] += tc["cells_total"]
                tog["tree_inputs"] += tc["inputs"]
            sign_j = -1 if j == BITS - 1 else 1
            for i in range(BITS):  # activation bit / cycle
                sign_i = -1 if i == BITS - 1 else 1
                sums += sign_i * sign_j * (pc[:, i] << (i + j))
        return sums, (tog if count_toggles else None)

    def accumulate(self, sums, e_w, e_x, max_e_w, max_e_x, clear):
        """Same alignment + window as v2 (docs/spec/numerics.md). Returns acc after each op."""
        d = np.minimum((max_e_w - e_w) + (max_e_x - e_x), self.window - 1)
        base = self.window - 14 - 1 - self.guard
        sh = base - d
        term = np.where(sh >= 0, sums << np.clip(sh, 0, 62), sums >> np.clip(-sh, 0, 63))
        acc, a = np.empty(len(sums), np.int64), 0
        for t in range(len(sums)):
            a = (0 if clear[t] else a) + int(term[t])
            acc[t] = a
        return acc
