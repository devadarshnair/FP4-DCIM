#!/usr/bin/env python3
"""T4.2: macro synthesis area delta of NV_GATE=1 against the as-built (NV_GATE=0) td macro netlists.

Post-processing only. NV_GATE=1 runs: the four remote t32 N_t42 e30_td_synth runs (v2 flat/hier, DPU hier/flat).
As-built references: the e30 runs pinned by the paper e22 run (macro_syn in its config.yaml) for v2 flat / DPU hier
(dpu_macro_mx) / DPU flat (dpuflat_macro_mx), and the latest e30_td_macro_v2_macro_hier run for v2 hier.
Macro power of the NV_GATE=1 netlists was NOT run (pending).
Usage: python experiments/t42_nvgate_area/run.py [--paper]
"""
from __future__ import annotations

import argparse
import json

import yaml

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

R = REPO_ROOT / "results"
NVG = {"v2_flat": "e30_td_macro_v2_macro_nvg1/20261116T104617485780Z_74f9502",
       "v2_hier": "e30_td_macro_v2_macro_hier_nvg1/20261116T105315658010Z_74f9502",
       "dpu_hier": "e30_td_macro_dpu_macro_hier_nvg1/20261116T105546220303Z_74f9502",
       "dpu_flat": "e30_td_macro_dpu_macro_nvg1/20261116T105837147057Z_74f9502"}
KEYS = ("total_mm2", "logic_area_um2", "seq_area_um2", "cells", "seq_cells", "buf_cells")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--paper", action="store_true"); a = ap.parse_args()
    pins = yaml.safe_load((REPO_ROOT / "experiments/configs/paper_runs.yaml").read_text())
    e22 = pins["figures_tables"]["e22_macro_compare"]
    ms = yaml.safe_load((R / "e22_macro_compare" / e22 / "config.yaml").read_text())["macro_syn"]
    hier_v2 = (R / "e30_td_macro_v2_macro_hier/latest").resolve().name
    base = {"v2_flat": f"e30_td_macro_v2_macro/{ms['v2_macro_mx']}",
            "v2_hier": f"e30_td_macro_v2_macro_hier/{hier_v2}",
            "dpu_hier": f"e30_td_macro_dpu_macro_hier/{ms['dpu_macro_mx']}",
            "dpu_flat": f"e30_td_macro_dpu_macro/{ms['dpuflat_macro_mx']}"}
    rows, m = [], {}
    for k in NVG:
        g = json.loads((R / NVG[k] / "metrics.json").read_text())
        b = json.loads((R / base[k] / "metrics.json").read_text())
        row = {"variant": k, "nvg1_run": NVG[k], "as_built_run": base[k]}
        for q in KEYS:
            row[f"{q}_as_built"], row[f"{q}_nvg1"] = b[q], g[q]
            row[f"{q}_delta"] = g[q] - b[q]
            row[f"{q}_delta_pct"] = 100 * (g[q] / b[q] - 1)
        rows.append(row)
        m[f"{k}_area_delta_pct"] = row["total_mm2_delta_pct"]
        m[f"{k}_area_delta_mm2"] = row["total_mm2_delta"]
        m[f"{k}_seq_cells_delta"] = row["seq_cells_delta"]
    m["macro_power_nvg1"] = "pending (not run)"
    with start_run("t42_nvgate_area", {"nvg1_runs": NVG, "as_built_runs": base, "e22_run": e22,
                                       "label": "Yosys td macro synthesis area (no P&R); NV_GATE=1 vs as-built"},
                   paper=a.paper) as run:
        run.save_csv("area_delta.csv", rows)
        run.log_metrics(m)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
