"""Macro-boundary wire energy (WP-28) added to the gate-level power of each design.

Synthesis-only netlists carry no wire capacitance, but the compared designs differ in what crosses
the column group: v2 broadcasts X' (4 b), 3X' (6 b) and the sign per row once per op; the bit-serial
baseline broadcasts 1 bit per row per cycle (5 per op); the digital-MAC baseline broadcasts X' and the
sign (5 wires/row) and additionally moves each weight set from a *central* buffer over a long bus.
Energy per toggle = ½·C·V² with C = (cap/µm from the sky130 tech LEF) × length.

Lengths are estimates until P&R (WP-41): row wire ≈ side of the synthesized group area (√A), local
SRAM->latch path ≈ √A/2, central-buffer bus ≈ 2·√A. Coupling is absent from the LEF numbers, so the
`high` band doubles C. All toggles are counted from the same stimulus program as the GLS.
"""

from __future__ import annotations

import math

import numpy as np
import yaml

from ..formats.codecs import e2m1_to_int_x2
from ..utils.provenance import REPO_ROOT

VDD = 1.8
LAYER = "met2"
BANDS = {"low": 1.0, "nom": 1.0, "high": 2.0}  # capacitance multiplier (coupling)
LEN_BANDS = {"low": 0.7, "nom": 1.0, "high": 1.5}  # wire-length multiplier


def _popcount(a: np.ndarray, width: int) -> int:
    u = np.asarray(a, dtype=np.int64) & ((1 << width) - 1)
    return int(sum(((u >> b) & 1).sum() for b in range(width)))


def _toggles(seq: np.ndarray, width: int) -> int:
    """Bit toggles along axis 0 of an int array (T, ...) for a `width`-bit bus."""
    if len(seq) < 2:
        return 0
    return _popcount(seq[1:] ^ seq[:-1], width)


def broadcast_toggles(prog: dict, design: str) -> int:
    """Row-broadcast wire toggles over the whole program (all 32 rows)."""
    x = prog["x"]  # (T, 32) E2M1 codes
    xi = e2m1_to_int_x2(x)
    mag, sgn = np.abs(xi), ((x >> 3) & 1) & (xi != 0)
    if design in ("v2_sram_m16", "v2_latch_m16", "v2_latch_m1"):
        return _toggles(mag, 4) + _toggles(3 * mag, 6) + _toggles(sgn, 1)
    if design in ("dpu", "v2_generic"):
        return _toggles(mag, 4) + _toggles(sgn, 1)
    if design == "bs_int5":
        bits = ((xi[:, :, None] & 0x1F) >> np.arange(5)) & 1  # (T, 32, 5) LSB-first per op
        stream = bits.transpose(0, 2, 1).reshape(-1, x.shape[1])  # (T*5, 32) one bit per row per cycle
        return _toggles(stream, 1)
    if design == "bs4_int4sm":  # T5.4 D2 INT4-SM (codes are {sign, mag[2:0]}): 3 magnitude bits LSB-first + sign once per op
        m3, s1 = (x & 7).astype(np.int64), ((x >> 3) & 1).astype(np.int64)
        stream = ((m3[:, :, None] >> np.arange(3)) & 1).transpose(0, 2, 1).reshape(-1, x.shape[1])
        return _toggles(stream, 1) + _toggles(s1, 1)
    raise KeyError(design)


def weight_move_toggles(prog: dict) -> int:
    """Bit toggles of the 32-bit word stream moved on set switches (16 words per switch)."""
    sets = prog["sets"]  # (M, 4, 32) codes
    order = [prog["set"][0]] + [s for i, s in enumerate(prog["set"][1:], 1) if s != prog["set"][i - 1]]
    words = []
    for s in order:
        for c in range(sets.shape[1]):
            for k in range(4):
                words.append(sum(int(sets[s, c, 8 * k + j]) << (4 * j) for j in range(8)))
    return _toggles(np.array(words, dtype=np.int64), 32)


def wire_energy_fj(prog: dict, design: str, area_mm2: float, band: str = "nom") -> dict:
    model = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())
    c_per_um = model["layers"][LAYER]["cap_ff_per_um"] * BANDS[band]
    side_um = math.sqrt(area_mm2) * 1e3 * LEN_BANDS[band]
    e_tog = lambda length_um: 0.5 * c_per_um * length_um * VDD**2  # fJ per toggle  # noqa: E731
    bcast = broadcast_toggles(prog, design) * e_tog(side_um)
    wlen = 2.0 * side_um if design == "dpu" else 0.5 * side_um
    wmove = weight_move_toggles(prog) * e_tog(wlen)
    return {"e_bcast_fj": bcast, "e_wmove_fj": wmove, "side_um": side_um, "c_ff_per_um": c_per_um}
