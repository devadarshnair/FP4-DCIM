"""e26 compare (T2.1): 3-way NVFP4 macro row from e21 v2 NV (flat) + e26 DPU NV (hierarchical) windows.

Same energy composition as e22 (experiments/e21_macro_power/compare.py op_energy/wire_fj): measured logic (GLS+OpenSTA) +
e20 SRAM corners (low/nominal/high[EST]) + [EST] boundary wires; zero-delay, 40 MHz, as-built SRAM clocks.
Flat-equivalent DPU [EST]: logic energy / 1.19112 (e23 tile hier/keepcol penalty, measured in MX mode, assumed equal in NV).
INT5 bit-serial macro is an MX-only control (no NV); its MX number is the pre-T2.7 stale one.

  python experiments/e26_dpu_nv_power/compare.py --e26 <run_id> [--paper]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import numpy as np
import yaml

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

spec = importlib.util.spec_from_file_location("e22cmp", REPO_ROOT / "experiments/e21_macro_power/compare.py")
c = importlib.util.module_from_spec(spec)
sys.modules["e22cmp"] = c
spec.loader.exec_module(c)
R = REPO_ROOT / "results"
V2_RUN = "20261110T050824693997Z_f25a5c5"
HIER = 1.19112  # e23 run 20261115T023614915387Z_20bfba0; was 1.186 before the e23 rerun; results/e23_tile_synth: DPU tile hierarchical/keepcol logic-energy ratio (MX-measured)
NUM = ("n_ops", "cycles", "window_ns", "p_total_leak_w", "p_sram_sta_leak_w", "p_logic_w", "sram_dual_read_cycles",
       "set_switches", "cycles_per_op", "e_logic_fj_per_mac", "bcast_toggles", "wmove_toggles")


def load(path, design):
    rows = [r for r in load_csv(path) if r["design"] == design]
    for r in rows:
        for k in NUM:
            r[k] = float(r[k])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e26", required=True)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    v2 = load(R / "e21_macro_power" / V2_RUN / "power.csv", "v2_macro_nv")
    dp = load(R / "e26_dpu_nv_power" / a.e26 / "power.csv", "dpu_macro_nv")
    corners = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())["corners"]
    cff = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())["layers"]["met2"]["cap_ff_per_um"]
    area = {k: json.loads((R / d / "latest/metrics.json").read_text())["total_mm2"] for k, d in
            (("v2", "e12_macro_synth_v2_macro"), ("dpu", "e12_macro_synth_dpu_macro_hier"))}
    cfg = {"v2_nv_e21_run": V2_RUN, "dpu_nv_e26_run": a.e26, "hier_factor_e23": HIER, "f_hz": 40e6,
           "policy": "as_built", "delay": "zero", "wire": "nom [EST]", "areas_mm2": area}
    with start_run("e26_dpu_nv_power", {**cfg, "kind": "compare"}, paper=a.paper) as run:
        out = []
        for kind in ("real", "rand"):
            for cn, corner in corners.items():
                for pol in ("as_built", "gated"):
                    def tot(rows, ar, central, logic_div=1.0):
                        ps = []
                        for r in rows:
                            if r["kind"] != kind:
                                continue
                            r = dict(r)
                            if logic_div != 1.0:  # logic energy and leakage scale together (hier netlist bloat)
                                r["p_logic_w"] /= logic_div
                                r["p_total_leak_w"] = (r["p_total_leak_w"] - r["p_sram_sta_leak_w"]) / logic_div + r["p_sram_sta_leak_w"]
                            ps.append(c.op_energy(r, corner, 40e6, 1.0, pol, c.wire_fj(r, ar, cff, "nom", central)))
                        return {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
                    v, dh, df = tot(v2, area["v2"], False), tot(dp, area["dpu"], True), tot(dp, area["dpu"], True, HIER)
                    out.append({"kind": kind, "sram": cn, "sram_clock": pol, "v2_nv": v["total"], "dpu_nv_hier": dh["total"],
                                "dpu_nv_flat_eq_EST": df["total"], "v2_logic": v["logic_dyn"] + v["logic_leak"],
                                "dpu_hier_logic": dh["logic_dyn"] + dh["logic_leak"],
                                "ratio_v2_over_dpu_hier": dh["total"] / v["total"],
                                "ratio_v2_over_dpu_flat_eq_EST": df["total"] / v["total"]})
        run.save_csv("nv_row.csv", out)
        for r in out:
            if r["kind"] == "real" and r["sram_clock"] == "as_built":
                print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
        rr = [r for r in out if r["kind"] == "real"]
        run.log_metrics({"rows": out, "ratio_hier_range_real": [min(r["ratio_v2_over_dpu_hier"] for r in rr), max(r["ratio_v2_over_dpu_hier"] for r in rr)],
                         "ratio_flat_eq_range_real": [min(r["ratio_v2_over_dpu_flat_eq_EST"] for r in rr), max(r["ratio_v2_over_dpu_flat_eq_EST"] for r in rr)]})


if __name__ == "__main__":
    main()
