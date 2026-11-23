"""Analytic, pre-registered gain budget (WP-60).

Each baseline's energy per MAC is 1.0, split into components (shares). v2 is reached by applying
*levers* in a fixed order; each lever rescales or removes specific components (or adds a v2
overhead). The lever value is E_before / E_after, so the levers multiply exactly (telescoping) to the
end-to-end ratio EE(v2)/EE(baseline) = E_baseline / E_v2.

Bands: every parameter has low/nom/high. The *optimistic* bound uses the end of each band that
favours v2, the *pessimistic* bound the other end. This is a planning instrument, not a result: its
job is to say, before RTL, which levers can possibly produce the G1 ratio and how big each can be.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Comp = dict[str, float]

# for each parameter, which band end favours v2 ("high" = larger value is better for v2)
FAVOURS_V2_HIGH = {
    "align_frac_left_hole": True,
    "align_frac_etc": True,
    "align_frac_lca": True,
    "align_frac_left": True,
    "bitparallel_factor": True,
    "glitch_penalty": False,
    "sram_read_rel": False,
    "reuse_R": True,
    "bits_per_weight_serial": True,
    "broadcast_overhead": False,
    "window_acc_overhead": False,
    "shift_add_removed": True,
    "cim_locality": True,
    "fp4_leaf_vs_mac": True,
}


def pick(params: dict, scenario: str) -> dict[str, float]:
    """scenario ∈ {low, nom, high, optimistic, pessimistic} -> {name: value}."""
    out = {}
    for name, band in params.items():
        if scenario in ("low", "nom", "high"):
            out[name] = float(band[scenario])
        else:
            hi_good = FAVOURS_V2_HIGH[name]
            end = (
                ("high" if hi_good else "low") if scenario == "optimistic" else ("low" if hi_good else "high")
            )
            out[name] = float(band[end])
    return out


@dataclass
class Lever:
    name: str
    apply: Callable[[Comp, dict], Comp]


def _total(c: Comp) -> float:
    return sum(c.values())


# ----------------------------------------------------------------------------------------- levers
def _align(c, p):
    c = dict(c)
    for comp, key in (
        ("left_hole", "align_frac_left_hole"),
        ("etc", "align_frac_etc"),
        ("lca_periph", "align_frac_lca"),
        ("left_periph", "align_frac_left"),
    ):
        if comp in c:
            c[comp] *= 1 - p[key]
    return c


def _storage(c, p):
    # array accessed every bit-serial cycle -> one SRAM read per weight-bit per R activations
    c = dict(c)
    if "cell_array" in c:
        c["cell_array"] *= p["sram_read_rel"] / p["reuse_R"]
    return c


def _bitparallel(c, p):
    c = dict(c)
    for comp in ("mfa_tree", "lca_periph", "left_periph"):  # per-cycle datapath + input feeding
        if comp in c:
            c[comp] *= p["glitch_penalty"] / p["bitparallel_factor"]
    return c


def _shift_add(c, p):
    c = dict(c)
    if "shift_add" in c:
        c["shift_add"] *= 1 - p["shift_add_removed"]
    return c


def _overheads(base_total: float):
    def f(c, p):
        c = dict(c)
        c["v2_broadcast"] = c.get("v2_broadcast", 0.0) + p["broadcast_overhead"] * base_total
        c["v2_window_acc"] = c.get("v2_window_acc", 0.0) + p["window_acc_overhead"] * base_total
        return c

    return f


def _weight_fetch(c, p):
    # central-SRAM fetch amortized over R in the WS array; v2 keeps the set in the macro -> remove it
    c = dict(c)
    c.pop("sram_fetch", None)
    return c


def _locality(c, p):
    c = dict(c)
    if "operand_regs" in c:
        c["operand_regs"] /= p["cim_locality"]
    if "mac" in c:
        c["mac"] /= p["fp4_leaf_vs_mac"]
    return c


def levers_for(baseline: str, base_total: float) -> list[Lever]:
    if baseline == "midas_like":
        return [
            Lever("alignment_removal", _align),
            Lever("storage_read", _storage),
            Lever("bit_parallel", _bitparallel),
            Lever("v2_overheads", _overheads(base_total)),
        ]
    if baseline == "bitserial_int5":
        return [
            Lever("storage_read", _storage),
            Lever("bit_parallel", _bitparallel),
            Lever("shift_add_removal", _shift_add),
            Lever("v2_overheads", _overheads(base_total)),
        ]
    if baseline == "digital_mac":
        return [
            Lever("weight_fetch", _weight_fetch),
            Lever("locality_and_leaf", _locality),
            Lever("v2_overheads", _overheads(base_total)),
        ]
    raise KeyError(baseline)


def budget(components: Comp, baseline: str, p: dict) -> tuple[float, list[tuple[str, float]], Comp]:
    """-> (end-to-end EE ratio v2/baseline, [(lever, ratio)], final v2 components)."""
    base = {k: v / 100.0 for k, v in components.items()}
    c, out = base, []
    for lv in levers_for(baseline, _total(base)):
        nxt = lv.apply(c, p)
        out.append((lv.name, _total(c) / _total(nxt)))
        c = nxt
    return _total(base) / _total(c), out, c


def alignment_upper_bound(midas_components: Comp, p_band: dict) -> float:
    """Largest possible EE gain from alignment removal alone: 1 / (1 - f_align,high)."""
    hi = {k: float(v["high"]) for k, v in p_band.items()}
    after = _align({k: v / 100 for k, v in midas_components.items()}, hi)
    return 1.0 / sum(after.values())
