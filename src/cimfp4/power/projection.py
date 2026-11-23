"""Informational technology projection (WP-30; label every projected number as informational).

E_target = E_source · energy_factor(source->target) · (V_target/V_source)^2 (dynamic, first order).
Scaling factors must come from a primary source (Stillmaker & Baas 2017 tables, key
`stillmaker2017scaling`) recorded in experiments/configs/projection.yaml with table/page; until WP-62
has opened the source, the YAML holds `factor: null` and projection is disabled (no invented numbers).
"""

from __future__ import annotations


def project_energy(
    e_source: float,
    energy_factor: float | None,
    v_source: float,
    v_target: float,
    voltage_included: bool = False,
) -> float | None:
    """Stillmaker/DeepScaleTool factors are voltage-dependent (``voltage_included=True``): do not apply
    the (V_t/V_s)^2 term again (WP-62 finding: doing so is ~4x too optimistic)."""
    if energy_factor is None:
        return None
    v_term = 1.0 if voltage_included else (v_target / v_source) ** 2
    return e_source * energy_factor * v_term


def project_tflops_per_w(
    tflops_per_w: float,
    energy_factor: float | None,
    v_source: float,
    v_target: float,
    voltage_included: bool = False,
):
    e = project_energy(1.0, energy_factor, v_source, v_target, voltage_included)
    return None if e is None else tflops_per_w / e
