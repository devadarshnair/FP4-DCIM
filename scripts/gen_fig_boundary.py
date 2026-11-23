#!/usr/bin/env python3
"""T8.2: costed-boundary figure -- which energy each headline number includes.

Two panels, every bar and tick read from pinned runs (experiments/configs/paper_runs.yaml), no typed numbers:
  (a) macro (e22_macro_compare, T2.6 td netlists, nominal SRAM, as-built clock, zero delay): stacked components of the
      macro headline fJ/MAC for v2 (MX, NV), bit-serial INT5 and DPU (hier.); ticks mark the cumulative boundaries
      logic -> +SRAM -> +wires [EST] (= headline).
  (b) tile G1 (e13_boundary_td, 4-column MX-only, R=256 M=16 SRAM): gate-level energy (logic + SRAM per-access, no SRAM
      idle clock) and the boundary-wire energy [EST] added on top; mean over the real windows.
The tile numbers are the gate/lever result only; the macro numbers are the headline.
Outputs: paper/figs/fig_costed_boundary.pdf, paper/figs/fig_costed_boundary.csv
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import json
import statistics as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "figs"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_pins
import style

OUT = paper_pins.ROOT / "paper" / "figs"
m = json.load(open(paper_pins.run_dir("e22_macro_compare") / "metrics.json"))

MACRO = [("v2_macro_mx", "v2 MX"), ("v2_macro_nv", "v2 NV"), ("bs_macro_mx", "bit-serial\nINT5"),
         ("dpu_macro_mx", "DPU MX\n(hier.)")]
# (metrics key, label, color); order = boundary growth order
COMP = [("e_logic_dyn_fj_per_mac", "logic dynamic", "#0072B2"), ("e_logic_leak_fj_per_mac", "logic leakage", "#56B4E9"),
        ("e_sram_access_fj_per_mac", "SRAM access", "#009E73"), ("e_sram_idle_clock_fj_per_mac", "SRAM idle clock", "#F0E442"),
        ("e_sram_leak_fj_per_mac", "SRAM leakage", "#E69F00"), ("e_wire_fj_per_mac", "boundary wires [EST]", "#CC79A7")]

style.use()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.TEXT_W_IN, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
rows = []
for i, (d, lab) in enumerate(MACRO):
    p = m[f"{d}_primary"]
    left = 0.0
    for k, cl, col in COMP:
        v = p[k]
        ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=0.4, label=cl if i == 0 else None)
        left += v
        rows.append(dict(panel="macro", design=d, component=k, fj_per_mac=v))
    logic = p["e_logic_dyn_fj_per_mac"] + p["e_logic_leak_fj_per_mac"]
    with_sram = logic + p["e_sram_access_fj_per_mac"] + p["e_sram_idle_clock_fj_per_mac"] + p["e_sram_leak_fj_per_mac"]
    for xv, mk in ((logic, "|"), (with_sram, "|")):
        ax.plot([xv, xv], [i - 0.42, i + 0.42], color="k", lw=0.6, ls=":")
    rows.append(dict(panel="macro", design=d, component="total_headline", fj_per_mac=p["e_total_fj_per_mac"]))
    ax.text(left + 25, i, f"{p['e_total_fj_per_mac']:.0f}", va="center", fontsize=7)
ax.set_yticks(range(len(MACRO)), [x[1] for x in MACRO])
ax.invert_yaxis()
ax.set_xlabel("fJ/MAC, 32x32 macro, sky130 40 MHz, zero-delay")
ax.set_title("(a) Macro headline (dotted: logic | +SRAM)")
ax.set_xlim(0, max(m[f"{d}_primary"]["e_total_fj_per_mac"] for d, _ in MACRO) * 1.12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

tile = {}
for r in csv.DictReader(open(paper_pins.run_dir("e13_boundary_td") / "power_boundary.csv")):
    if r["kind"] == "real":
        tile.setdefault(r["design"], []).append((float(r["e_gate_fj_per_mac"]), float(r["e_wire_fj_per_mac_nom"])))
TILE = [("v2_sram_m16", "v2"), ("bs_int5", "bit-serial\nINT5"), ("dpu", "DPU")]
for i, (d, lab) in enumerate(TILE):
    g = st.mean(x[0] for x in tile[d]); w = st.mean(x[1] for x in tile[d])
    bx.barh(i, g, color="#0072B2", edgecolor="white", linewidth=0.4, label="gate-level (logic + SRAM access)" if i == 0 else None)
    bx.barh(i, w, left=g, color="#CC79A7", edgecolor="white", linewidth=0.4, label="boundary wires [EST]" if i == 0 else None)
    bx.text(g + w + 25, i, f"{g + w:.0f}", va="center", fontsize=7)
    rows += [dict(panel="tile", design=d, component="gate", fj_per_mac=g), dict(panel="tile", design=d, component="wires_nom_est", fj_per_mac=w)]
bx.set_yticks(range(len(TILE)), [x[1] for x in TILE])
bx.invert_yaxis()
bx.set_xlabel("fJ/MAC, 4-column MX-only tile, R=256")
bx.set_title("(b) Tile G1 (no SRAM idle clock)")
bx.set_xlim(0, max(st.mean(x[0] + x[1] for x in tile[d]) for d, _ in TILE) * 1.15)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_costed_boundary.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "matplotlib"})
with open(OUT / "fig_costed_boundary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["panel", "design", "component", "fj_per_mac"], lineterminator="\n")
    w.writeheader()
    for r in rows:
        r["fj_per_mac"] = repr(float(r["fj_per_mac"]))
        w.writerow(r)
print(f"wrote {OUT / 'fig_costed_boundary.pdf'}")
