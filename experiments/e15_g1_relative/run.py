"""e15 - G1: relative same-PDK comparison, lever decomposition and verdict (WP-30).

Inputs (results only): e12_synth, e13_boundary[_r16|_m1|_latch] (energy/MAC incl. boundary wires, per
window), e13_sdf (glitch factor per design), e00_gain_budget (pre-registered bands), gate-level power
reports (Sequential/Combinational groups) for the lever decomposition.
Outputs: g1.json (verdict + all ratios with CIs), g1_table.csv, levers.csv.

Run: python experiments/e15_g1_relative/run.py [--paper]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from cimfp4.power.g1_compare import Thresholds, paired_ratio, verdict
from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

R = REPO_ROOT / "results"
V2 = "v2_sram_m16"
POINTS = {  # operating point -> boundary run
    "R256_M16_SRAM": "e13_boundary",
    "R16_M16_SRAM": "e13_boundary_r16",
    "R256_M1_LATCH": "e13_boundary_m1",
    "R256_M16_LATCH": "e13_boundary_latch",
}


def latest(exp: str) -> Path | None:
    p = R / exp / "latest"
    return p if p.exists() else None


def energies(exp: str, band: str = "nom") -> dict:
    """{(design, window): E_total fJ/MAC} for real windows."""
    d = latest(exp)
    if d is None:
        return {}
    return {
        (r["design"], r["window"]): float(r[f"e_total_fj_per_mac_{band}"])
        for r in load_csv(d / "power_boundary.csv")
        if r["kind"] == "real"
    }


def groups(report: Path) -> dict:
    t = report.read_text()
    out = {}
    for g in ("Sequential", "Combinational", "Macro"):
        m = re.search(rf"^{g}\s+\S+\s+\S+\s+\S+\s+([\d.eE+-]+)", t, re.M)
        out[g] = float(m.group(1)) if m else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--suffix", default="", help="read <exp><suffix> inputs, e.g. _rb (rebased reruns)")
    ap.add_argument("--area-recipe", default="", help="tile areas from results/e30_td_synth/latest (rows of this "
                    "recipe, e.g. td); GOPS/mm2 recomputed with the e12 formula (2 x mac_per_cycle x f_G1 / area)")
    a = ap.parse_args()
    sfx = a.suffix
    points = {k: v + sfx for k, v in POINTS.items()}
    rng = seed_all(0)
    prereg = yaml.safe_load((REPO_ROOT / "experiments/configs/gain_budget.yaml").read_text())["g1_prereg"]
    th = Thresholds.from_prereg(prereg)
    budget = json.loads((latest("e00_gain_budget") / "metrics.json").read_text())
    synth = {r["design"]: r for r in load_csv(latest("e12_synth") / "synth.csv")}
    area_src = "e12_synth/" + latest("e12_synth").resolve().name
    if a.area_recipe:
        f_g1 = json.loads((latest("e12_synth") / "metrics.json").read_text())["f_g1_mhz"]
        for r in load_csv(latest("e30_td_synth") / "synth.csv"):
            if r["recipe"] == a.area_recipe and r["design"] in synth:
                old_r, mm2 = synth[r["design"]], float(r["total_area_um2"]) / 1e6
                synth[r["design"]] = {**old_r, "total_mm2": mm2,
                                      "kb_per_mm2": float(old_r["kb_per_mm2"]) * float(old_r["total_mm2"]) / mm2,
                                      "gops_per_mm2_at_fg1": 2 * float(old_r["mac_per_cycle"]) * f_g1 * 1e6 / 1e9 / mm2}
        synth = {d: r for d, r in synth.items() if d in {"v2_sram_m16", "bs_int5", "dpu"}}
        area_src = f"e30_td_synth/{latest('e30_td_synth').resolve().name} recipe {a.area_recipe}"
    sdf = {}
    if latest("e13_sdf" + sfx):
        sdf = {
            k.replace("_sdf_over_zero", ""): v
            for k, v in json.loads((latest("e13_sdf" + sfx) / "metrics.json").read_text()).items()
            if k.endswith("_sdf_over_zero")
        }
    cfg = {
        "thresholds": th.__dict__,
        "points": points,
        "sources": {k: str((latest(v) or Path("missing")).resolve().name) for k, v in points.items()},
        "sdf_source": str((latest("e13_sdf" + sfx) or Path("missing")).resolve().name),
        "area_source": area_src,
    }
    with start_run("e15_g1_relative" + sfx, cfg, paper=a.paper) as run:
        table, g1 = [], {"thresholds": th.__dict__, "points": {}}
        main_e = energies("e13_boundary" + sfx)
        windows = sorted({w for (_, w) in main_e})
        for point, exp in points.items():
            e = energies(exp)
            if not e:
                continue
            v2_name = {"R256_M1_LATCH": "v2_latch_m1", "R256_M16_LATCH": "v2_latch_m16"}.get(point, V2)
            for base in ("bs_int5", "dpu", "v2_generic"):
                src = (
                    e if (base, windows[0]) in e else main_e
                )  # latch/M1 variants compare to the main baselines
                if (v2_name, windows[0]) not in e or (base, windows[0]) not in src:
                    continue
                eb = np.array([src[(base, w)] for w in windows])
                ev = np.array([e[(v2_name, w)] for w in windows])
                row = {"point": point, "v2": v2_name, "baseline": base, "delay_model": "zero"}
                row |= {f"ee_{k}": v for k, v in paired_ratio(eb, ev, rng).items()}
                table.append(row)
                if sdf.get(base) and sdf.get(V2):
                    corr = sdf[base] / sdf[V2]
                    row_s = {
                        **row,
                        "delay_model": "sdf_corrected",
                        **{f"ee_{k}": v for k, v in paired_ratio(eb * corr, ev, rng).items()},
                    }
                    table.append(row_s)
        run.save_csv("g1_table.csv", table)

        def pick(point, base, model):
            m = [
                r
                for r in table
                if r["point"] == point and r["baseline"] == base and r["delay_model"] == model
            ]
            return m[0] if m else None

        primary = pick("R256_M16_SRAM", "bs_int5", "sdf_corrected") or pick(
            "R256_M16_SRAM", "bs_int5", "zero"
        )
        area_ratio = float(synth[V2]["gops_per_mm2_at_fg1"]) / float(synth["bs_int5"]["gops_per_mm2_at_fg1"])
        ee = {"mean": primary["ee_mean"], "ci_lo": primary["ee_ci_lo"]}
        dec = verdict(
            ee, area_ratio, mismatches=0, th=th
        )  # 0: every GLS/regression run is bit-exact (blocking)
        band = (budget["ee_bitserial_int5_pessimistic"], budget["ee_bitserial_int5_optimistic"])

        # lever decomposition (per MAC, nominal, main point): sequential / combinational / SRAM / wires
        levers = []
        e13 = latest("e13_power" + sfx)
        bcfg = yaml.safe_load((latest("e13_boundary" + sfx) / "config.yaml").read_text()) or {}
        e13_runs = [R / ("e13_power" + sfx) / r for r in bcfg.get("source_runs", [])] or [e13]
        pw = {(r["design"], r["window"]): r for d in e13_runs for r in load_csv(d / "power.csv")}
        bnd = {(r["design"], r["window"]): r for r in load_csv(latest("e13_boundary" + sfx) / "power_boundary.csv")}
        for d in (V2, "bs_int5", "dpu", "v2_generic"):
            acc = {"seq": [], "comb": [], "sram": [], "wire": []}
            for w in windows:
                rpt = next(
                    (
                        p
                        for p in (
                            REPO_ROOT / "build/power" / ("e13_power" + sfx) / d / w / "power.rpt",
                            *((REPO_ROOT / "build/power" / d / w / "power.rpt",) if not sfx else ()),
                        )
                        if p.exists()
                    ),
                    None,
                )
                if rpt is None:
                    continue
                gr, r = groups(rpt), pw[(d, w)]
                scale = float(r["window_ns"]) * 1e-9 / float(r["macs"]) * 1e15
                acc["seq"].append(gr["Sequential"] * scale)
                acc["comb"].append(gr["Combinational"] * scale)
                acc["sram"].append(float(r["e_sram_fj_per_mac"]))
                acc["wire"].append(float(bnd[(d, w)]["e_wire_fj_per_mac_nom"]))
            levers.append(
                {
                    "design": d,
                    **{f"e_{k}_fj_per_mac": float(np.mean(v)) if v else float("nan") for k, v in acc.items()},
                }
            )
        run.save_csv("levers.csv", levers)

        g1 |= {
            "primary_point": "R256_M16_SRAM",
            "primary_delay_model": primary["delay_model"],
            "ee_ratio_v2_over_int5": primary["ee_mean"],
            "ee_ci": [primary["ee_ci_lo"], primary["ee_ci_hi"]],
            "tflops_mm2_ratio_v2_over_int5": area_ratio,
            "kb_mm2": {d: float(synth[d]["kb_per_mm2"]) for d in synth},
            "prereg_band_vs_int5": band,
            "inside_band": band[0] <= primary["ee_mean"] <= band[1],
            "verdict": dec,
            "sdf_over_zero": sdf,
            "levers_fj_per_mac": levers,
            "table": table,
            "decision_proposal": f"D23 G1 {dec['verdict']}: EE(v2)/EE(INT5) = {primary['ee_mean']:.2f} "
            f"[{primary['ee_ci_lo']:.2f}, {primary['ee_ci_hi']:.2f}] ({primary['delay_model']}), "
            f"TFLOPS/mm² ratio {area_ratio:.2f}; sky130 relative, same flow/stimulus.",
        }
        if not g1["inside_band"]:
            lv = {r["design"]: r for r in levers}
            g1["lever_explanation"] = (
                "Outside the pre-registered band. Per-MAC energy (fJ): "
                + "; ".join(
                    f"{d}: seq {lv[d]['e_seq_fj_per_mac']:.0f}, comb {lv[d]['e_comb_fj_per_mac']:.0f}, "
                    f"sram {lv[d]['e_sram_fj_per_mac']:.0f}, wire {lv[d]['e_wire_fj_per_mac']:.0f}"
                    for d in (V2, "bs_int5")
                )
                + ". The budget modeled per-cycle register/clock energy only implicitly; bit-serial clocks its "
                "pipeline, accumulator and shared row shift registers 5x per op."
            )
        (run.dir / "g1.json").write_text(json.dumps(g1, indent=2))
        run.log_metrics(
            {
                "g1_verdict": dec["verdict"],
                "ee_ratio_v2_over_int5": primary["ee_mean"],
                "ee_ratio_v2_over_int5_ci_lo": primary["ee_ci_lo"],
                "tflops_mm2_ratio_v2_over_int5": area_ratio,
                "inside_prereg_band": g1["inside_band"],
            }
        )
        print(
            json.dumps(
                {
                    k: g1[k]
                    for k in (
                        "verdict",
                        "ee_ratio_v2_over_int5",
                        "ee_ci",
                        "tflops_mm2_ratio_v2_over_int5",
                        "prereg_band_vs_int5",
                        "inside_band",
                    )
                },
                indent=1,
            )
        )
        for r in table:
            print(
                f"{r['point']:16s} {r['v2']:13s} vs {r['baseline']:11s} {r['delay_model']:13s} "
                f"EE ratio {r['ee_mean']:.2f} [{r['ee_ci_lo']:.2f}, {r['ee_ci_hi']:.2f}]"
            )


if __name__ == "__main__":
    main()
