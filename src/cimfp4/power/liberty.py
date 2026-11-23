"""Minimal Liberty reader for per-action energy estimates (WP-37 power model; reusable for DSE).

Extracts, for selected cells: area (µm²), leakage (Liberty power unit), input-pin capacitances (pF) and
internal-power tables (energy per output transition, Liberty energy unit = pJ for sky130 since power is in
mW and time in ns). `toggle_energy_fj` = mean(rise, fall) internal energy at (slew, load), averaged over
related pins, + ½·C_load·V² switching energy - the energy of one output toggle.

Not a general Liberty parser: it brace-matches `cell ("name") { ... }` groups and reads the fields above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SKY130_TT = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"


def _group(text: str, start: int) -> tuple[int, int]:
    """Given index of an opening '{', return (start, end) of the balanced group contents."""
    depth, i = 0, start
    while True:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start + 1, i
        i += 1


def _floats(s: str) -> np.ndarray:
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)])


@dataclass
class Table:
    idx1: np.ndarray  # input transition (ns)
    idx2: np.ndarray  # output load (pF)
    values: np.ndarray  # (len(idx1), len(idx2))

    def at(self, slew: float, load: float) -> float:
        def axis(ix, v):
            v = float(np.clip(v, ix[0], ix[-1]))
            j = int(np.clip(np.searchsorted(ix, v) - 1, 0, len(ix) - 2))
            return j, (v - ix[j]) / (ix[j + 1] - ix[j])

        i, a = axis(self.idx1, slew)
        j, b = axis(self.idx2, load)
        v = self.values
        return float((1 - a) * (1 - b) * v[i, j] + a * (1 - b) * v[i + 1, j] + (1 - a) * b * v[i, j + 1]
                     + a * b * v[i + 1, j + 1])


@dataclass
class Cell:
    name: str
    area: float
    leakage: float
    input_cap: dict = field(default_factory=dict)  # pin -> pF
    # output pin -> list[(related_pin, rise Table | None, fall Table | None)]
    internal: dict = field(default_factory=dict)
    # input pin -> internal energy (fJ) per rise+fall pair (e.g. CLK of a flop: clock energy per cycle)
    pin_internal_fj: dict = field(default_factory=dict)

    def toggle_energy_fj(self, out: str | None = None, slew: float = 0.05, load: float = 0.005,
                         vdd: float = 1.8) -> float:
        """Energy (fJ) of one output toggle: mean internal (rise/fall, related pins) + ½·C·V² on the load."""
        pins = [out] if out else list(self.internal)
        vals = []
        for p in pins:
            for _rel, rise, fall in self.internal.get(p, []):
                e = [t.at(slew, load) for t in (rise, fall) if t is not None]
                if e:
                    vals.append(np.mean(e))
        internal_pj = float(np.mean(vals)) if vals else 0.0
        return internal_pj * 1e3 + 0.5 * load * 1e-12 * vdd**2 * 1e15

    def clock_energy_fj(self, clk: str = "CLK") -> float:
        """Internal energy (fJ) per clock cycle (rise + fall) on the clock pin of a sequential cell."""
        return self.pin_internal_fj.get(clk, 0.0)


def _table(block: str, kind: str) -> Table | None:
    m = re.search(kind + r'\s*\("[^"]*"\)\s*\{', block)
    if not m:
        return None
    s, e = _group(block, m.end() - 1)
    body = block[s:e]
    i1 = re.search(r'index_1\s*\("([^"]*)"\)', body)
    i2 = re.search(r'index_2\s*\("([^"]*)"\)', body)
    vals = _floats(re.search(r"values\s*\((.*?)\);", body, re.S).group(1))
    if i1 is None:
        return None
    x1 = _floats(i1.group(1))
    x2 = _floats(i2.group(1)) if i2 else np.array([0.0, 1.0])
    v = vals.reshape(len(x1), -1)
    if v.shape[1] == 1:
        v = np.repeat(v, 2, axis=1)
    return Table(x1, x2, v)


def _parse_cell(name: str, body: str) -> Cell:
    area = float(re.search(r"\barea\s*:\s*([\d.eE+-]+)", body).group(1))
    lk = re.search(r"cell_leakage_power\s*:\s*([\d.eE+-]+)", body)
    cell = Cell(name, area, float(lk.group(1)) if lk else 0.0)
    pin_internal = {}
    for m in re.finditer(r'\bpin\s*\("([^"]+)"\)\s*\{', body):
        s, e = _group(body, m.end() - 1)
        pbody, pin = body[s:e], m.group(1)
        if re.search(r'direction\s*:\s*"?input', pbody):
            c = re.search(r"\bcapacitance\s*:\s*([\d.eE+-]+)", pbody)
            if c:
                cell.input_cap[pin] = float(c.group(1))
            # input-pin internal power (e.g. CLK of flops): 1-D tables over transition
            for im in re.finditer(r"internal_power\s*\(\s*\)\s*\{", pbody):
                s2, e2 = _group(pbody, im.end() - 1)
                ib = pbody[s2:e2]
                vals = [t.at(0.05, 0.0) for t in (_table(ib, "rise_power"), _table(ib, "fall_power")) if t]
                if vals:
                    pin_internal.setdefault(pin, []).append(np.mean(vals) * 1e3)
        elif re.search(r'direction\s*:\s*"?output', pbody):
            lst = []
            for im in re.finditer(r"internal_power\s*\(\s*\)\s*\{", pbody):
                s2, e2 = _group(pbody, im.end() - 1)
                ib = pbody[s2:e2]
                rel = re.search(r'related_pin\s*:\s*"([^"]+)"', ib)
                lst.append((rel.group(1) if rel else "", _table(ib, "rise_power"), _table(ib, "fall_power")))
            cell.internal[pin] = lst
    cell.pin_internal_fj = {p: float(np.sum(v)) for p, v in pin_internal.items()}
    return cell


@lru_cache(maxsize=4)
def _text(path: str) -> str:
    return Path(path).read_text()


def load_cells(names: list[str], path: Path = SKY130_TT) -> dict[str, Cell]:
    text = _text(str(path))
    out = {}
    for n in names:
        m = re.search(r'cell\s*\("' + re.escape(n) + r'"\)\s*\{', text)
        if not m:
            raise KeyError(n)
        s, e = _group(text, m.end() - 1)
        out[n] = _parse_cell(n, text[s:e])
    return out
