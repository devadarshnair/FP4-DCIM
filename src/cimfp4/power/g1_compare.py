"""G1: relative same-PDK comparison and verdict (WP-30; thresholds pre-registered in G1_prereg.md).

Inputs are results only (no re-derivation): e12 (density, f_G1), e13 + e13_boundary (energy per MAC
incl. boundary wires, per window), e13_sdf (glitch factor per design). The verdict is computed by
code from the pre-registered thresholds in experiments/configs/gain_budget.yaml (g1_prereg block,
frozen with the G1 pre-registration note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    go_ratio: float = 1.5
    go_lower_ci: float = 1.3
    conditional_low: float = 1.2
    area_floor: float = 0.8
    fallback_area: float = 0.4
    cim_advantage_vs_c: float = 1.15

    @classmethod
    def from_prereg(cls, g: dict) -> Thresholds:
        return cls(
            go_ratio=g["go_ratio_vs_int5"],
            go_lower_ci=g["go_lower_ci"],
            conditional_low=g["conditional_low"],
            area_floor=g["area_floor_tflops_mm2"],
            cim_advantage_vs_c=g["cim_advantage_vs_c"],
        )


def paired_ratio(e_base: np.ndarray, e_v2: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> dict:
    """EE(v2)/EE(base) = E(base)/E(v2), paired by window; mean + 95 % bootstrap CI."""
    r = np.asarray(e_base, float) / np.asarray(e_v2, float)
    boots = [np.mean(rng.choice(r, len(r))) for _ in range(n_boot)] if len(r) > 1 else [r.mean()]
    return {
        "mean": float(r.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n": int(len(r)),
    }


def verdict(ee_vs_int5: dict, area_ratio_vs_int5: float, mismatches: int, th: Thresholds) -> dict:
    """G1 verdict per the pre-registered thresholds (blocking mismatch check first)."""
    if mismatches:
        return {"verdict": "BLOCKED", "reason": f"{mismatches} mismatches — debug before using any number"}
    if area_ratio_vs_int5 < th.fallback_area:
        return {"verdict": "FALLBACK", "reason": f"area ratio {area_ratio_vs_int5:.2f} < {th.fallback_area}"}
    m, lo = ee_vs_int5["mean"], ee_vs_int5["ci_lo"]
    if m >= th.go_ratio and lo >= th.go_lower_ci and area_ratio_vs_int5 >= th.area_floor:
        return {
            "verdict": "GO",
            "reason": f"EE ratio {m:.2f} (CI lo {lo:.2f}) ≥ {th.go_ratio}/{th.go_lower_ci}; "
            f"area ratio {area_ratio_vs_int5:.2f} ≥ {th.area_floor}",
        }
    if m >= th.conditional_low:
        return {
            "verdict": "CONDITIONAL",
            "reason": f"EE ratio {m:.2f} in [{th.conditional_low}, {th.go_ratio})"
            f" or CI lo {lo:.2f} < {th.go_lower_ci}",
        }
    return {"verdict": "FALLBACK", "reason": f"EE ratio {m:.2f} < {th.conditional_low}"}


def in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]
