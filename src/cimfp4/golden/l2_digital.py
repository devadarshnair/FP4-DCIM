"""L2 model of baseline (c): weight-stationary digital MAC lanes with a central weight SRAM (WP-39).

Arithmetic is identical to v2 (exact block sums, same alignment and window), so results come from
golden.l2_macro / the column scoreboard. What differs is *where weights travel*: on a set switch the
512-bit set of a 4-column group moves over a long bus (32 b/beat) from the central buffer to the PE
weight registers. This module counts bus beats and bit toggles for the power flow (WP-28), with the
wire capacitance per µm from flows/pdk/wire_model.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

from ..utils.provenance import REPO_ROOT

WORDS_PER_SET = 16  # 4 columns x 32 rows x 4 b / 32 b


@dataclass
class WeightBus:
    width: int = 32
    prev: int = 0
    beats: int = 0
    toggles: int = 0
    words_seen: list = field(default_factory=list)

    def transfer_set(self, words: np.ndarray) -> None:
        """Move one weight set (16 x 32-bit words) over the bus; count beats and bit toggles."""
        for w in np.asarray(words, dtype=np.uint64).tolist():
            self.toggles += bin(int(w) ^ self.prev).count("1")
            self.prev = int(w)
            self.beats += 1

    def energy_fj(self, length_um: float, layer: str = "met3", vdd: float = 1.8) -> float:
        """Dynamic energy of the counted toggles: ½·C·V² per toggle with C = cap/µm × length."""
        model = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())
        c_ff = model["layers"][layer]["cap_ff_per_um"] * length_um
        return 0.5 * c_ff * vdd**2 * self.toggles
