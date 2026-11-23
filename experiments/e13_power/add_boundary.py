"""e13b - add macro-boundary wire energy (row broadcast + weight movement) to e13 gate-level energies.

Reads results/e13_power/latest/power.csv and the stimulus programs (build/power/programs), adds
cimfp4.power.boundary energies for each band (low/nom/high), writes power_boundary.csv into a new
e13_boundary run and the boundary-inclusive EE ratios v2/baseline with paired bootstrap CIs.

Run: python experiments/e13_power/add_boundary.py [--paper]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cimfp4.power.boundary import BANDS, wire_energy_fj
from cimfp4.stimulus.writers import load_program
from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--src-exp", default="e13_power")
    ap.add_argument("--src-runs", nargs="*", default=[], help="run ids of --src-exp to concatenate (default: latest)")
    ap.add_argument("--area-recipe", default="", help="take tile areas from results/e30_td_synth/latest synth.csv "
                    "rows of this recipe (e.g. td) instead of e12_synth (wire lengths scale with sqrt(area))")
    args = ap.parse_args()
    rng = seed_all(0)
    src = REPO_ROOT / "results" / args.src_exp / "latest"
    runs = [REPO_ROOT / "results" / args.src_exp / r for r in args.src_runs] if args.src_runs else [src]
    rows = [r for d in runs for r in load_csv(d / "power.csv")]  # per-design e13 runs are concatenated
    synth = {r["design"]: r for r in load_csv(REPO_ROOT / "results/e12_synth/latest/synth.csv")}
    area_src = "e12_synth/" + (REPO_ROOT / "results/e12_synth/latest").resolve().name
    if args.area_recipe:
        e30 = REPO_ROOT / "results/e30_td_synth/latest"
        for r in load_csv(e30 / "synth.csv"):
            if r["recipe"] == args.area_recipe and r["design"] in synth:
                synth[r["design"]] = {**synth[r["design"]], "total_mm2": float(r["total_area_um2"]) / 1e6}
        area_src = f"e30_td_synth/{e30.resolve().name} recipe {args.area_recipe}"
    cfg = {"source_run": (src / "metrics.json").resolve().parent.name if not args.src_runs else args.src_runs[0],
           "source_runs": [d.resolve().name for d in runs], "bands": list(BANDS), "area_source": area_src}
    exp = args.src_exp.replace("e13_power", "e13_boundary")
    with start_run(exp, cfg, paper=args.paper) as run:
        out = []
        for r in rows:
            pdir = REPO_ROOT / "build/power" / args.src_exp / "programs"
            if not pdir.exists():  # runs before the --exp option stored programs here
                pdir = REPO_ROOT / "build/power/programs"
            prog = load_program(pdir / f"{r['window']}.npz")
            macs = float(r["macs"])
            rec = {k: r[k] for k in ("design", "window", "kind", "macs")}
            rec["e_gate_fj_per_mac"] = float(r["e_total_fj_per_mac"])
            for band in BANDS:
                w = wire_energy_fj(prog, r["design"], float(synth[r["design"]]["total_mm2"]), band)
                e_wire = (w["e_bcast_fj"] + w["e_wmove_fj"]) / macs
                rec[f"e_wire_fj_per_mac_{band}"] = e_wire
                rec[f"e_total_fj_per_mac_{band}"] = rec["e_gate_fj_per_mac"] + e_wire
            out.append(rec)
        run.save_csv("power_boundary.csv", out)
        designs = sorted({r["design"] for r in out})
        windows = sorted({r["window"] for r in out if r["kind"] == "real"})
        get = {(r["design"], r["window"]): r for r in out}
        metrics = {}
        for band in BANDS:
            for d in designs:
                vals = [get[(d, w)][f"e_total_fj_per_mac_{band}"] for w in windows]
                metrics[f"{d}_e_fj_per_mac_{band}"] = float(np.mean(vals))
                metrics[f"{d}_tflops_per_w_{band}"] = float(2e3 / np.mean(vals))
            for base in (b for b in designs if b != "v2_sram_m16" and "v2_sram_m16" in designs):
                ratios = np.array(
                    [
                        get[(base, w)][f"e_total_fj_per_mac_{band}"]
                        / get[("v2_sram_m16", w)][f"e_total_fj_per_mac_{band}"]
                        for w in windows
                    ]
                )
                boots = [np.mean(rng.choice(ratios, len(ratios))) for _ in range(1000)]
                metrics[f"ee_ratio_v2_over_{base}_{band}"] = float(ratios.mean())
                metrics[f"ee_ratio_v2_over_{base}_{band}_ci_lo"] = float(np.percentile(boots, 2.5))
                metrics[f"ee_ratio_v2_over_{base}_{band}_ci_hi"] = float(np.percentile(boots, 97.5))
        run.log_metrics(metrics)
        print(
            json.dumps({k: round(v, 3) for k, v in metrics.items() if "ratio" in k and "_nom" in k}, indent=1)
        )


if __name__ == "__main__":
    main()
