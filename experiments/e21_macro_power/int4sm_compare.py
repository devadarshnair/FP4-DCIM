"""T5.4 summary (e34-style, local, light): D2 INT4-SM bit-serial macro (bs4_macro_mx, LOSSY surrogate) vs the v2 / INT5 /
DPU macros on the SAME paired windows real0-7 (T2.7 run, int4sm_programs.py re-quantises the identical rows / slices /
tokens). Results only; no simulation.

Energy per MAC per window = compare.op_energy (e22 model, no boundary wires): logic dynamic (x glitch) + logic leakage
+ OpenRAM access / idle clock / leakage at f = 40 MHz, SRAM corner nominal, policies as_built and gated.
Glitch (SDF/zero, tile-level ratio transferred to the macro, [EST]): v2 / INT5 / DPU from e13_sdf_td (as e22 T2.7),
bs4 from e34_d2_tile_sdf (the bs4_int4sm tile, td recipe).  Ratios E(bs4)/E(x) of the window means with paired
bootstrap 95 % CIs (resampling windows; compare.boot_ratio), at zero delay and glitch-corrected.

Run: python experiments/e21_macro_power/int4sm_compare.py --bs4 <e21 run id with bs4_macro_mx rows> [--paper]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import boot_ratio, op_energy  # noqa: E402

from cimfp4.utils.io import load_csv  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402
from cimfp4.utils.runlog import RESULTS, start_run  # noqa: E402

T27_RUN = "20261114T023643822998Z_cada632"
REF = {"v2": "v2_macro_mx", "int5": "bs_macro_mx", "dpu": "dpu_macro_mx"}
TILE = {"v2_macro_mx": "v2_sram_m16", "bs_macro_mx": "bs_int5", "dpu_macro_mx": "dpu"}
WINDOWS = [f"real{i}" for i in range(8)]
NUM = ("n_ops", "cycles", "window_ns", "p_total_leak_w", "p_sram_sta_leak_w", "p_logic_w", "sram_dual_read_cycles",
       "cycles_per_op", "e_logic_fj_per_mac")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs4", required=True, help="e21 run id (results/e21_macro_power/<id>) holding the bs4_macro_mx rows")
    ap.add_argument("--t27", default=T27_RUN, help="e21 run with v2/INT5/DPU MX rows on the same windows")
    ap.add_argument("--sdf13", default="e13_sdf_td")
    ap.add_argument("--sdf34", default="e34_d2_tile_sdf")
    ap.add_argument("--f-mhz", type=float, default=40.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--root", default=str(RESULTS), help="results root (inputs and output; smoke tests only)")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    rows = [r for run in (a.t27, a.bs4) for r in load_csv(root / "e21_macro_power" / run / "power.csv")]
    rows = [r for r in rows if r["design"] in (*REF.values(), "bs4_macro_mx") and r["window"] in WINDOWS]
    for r in rows:
        for k in NUM:
            r[k] = float(r[k])
    by = {(r["design"], r["window"]): r for r in rows}
    missing = [(d, w) for d in (*REF.values(), "bs4_macro_mx") for w in WINDOWS if (d, w) not in by]
    if missing:
        raise SystemExit(f"missing paired rows: {missing}")
    s13 = json.loads((root / a.sdf13 / "latest" / "metrics.json").read_text())
    s34 = json.loads((root / a.sdf34 / "latest" / "metrics.json").read_text())
    glitch = {d: s13[f"{TILE[d]}_sdf_over_zero"] for d in REF.values()}
    glitch["bs4_macro_mx"] = s34["sdf_over_zero"]
    corner = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())["corners"]["nominal"]
    f = a.f_mhz * 1e6
    cfg = {"t27_run": a.t27, "bs4_run": a.bs4, "windows": WINDOWS, "f_hz": f, "sram_corner": "nominal",
           "glitch": glitch, "sdf13_run": (root / a.sdf13 / "latest").resolve().name,
           "sdf34_run": (root / a.sdf34 / "latest").resolve().name, "n_boot": a.n_boot,
           "note": "bs4 = D2 MIDAS-inspired LOSSY INT4-SM surrogate; no boundary wires; glitch [EST]"}
    rng = np.random.default_rng(0)
    with start_run("e22_int4sm_compare", cfg, paper=a.paper, root=root) as run:
        tab, m = [], {}
        for delay, policy, part in itertools.product(("zero", "sdf_est"), ("as_built", "gated"), ("logic", "total")):
            def e(d):
                out = {}
                for w in WINDOWS:
                    p = op_energy(by[(d, w)], corner, f, glitch[d] if delay == "sdf_est" else 1.0, policy)
                    out[w] = p["logic_dyn"] + p["logic_leak"] if part == "logic" else p["total"]
                return out
            if part == "logic" and policy == "gated":
                continue  # logic energy does not depend on the SRAM clock policy
            eb = e("bs4_macro_mx")
            for name, d in REF.items():
                r, lo, hi, _ = boot_ratio(e(d), eb, rng, a.n_boot)
                key = f"bs4_over_{name}_{part}_{delay}" + ("" if part == "logic" else f"_{policy}")
                m[key], m[key + "_ci"] = r, [lo, hi]
                tab.append({"ratio": f"bs4/{name}", "part": part, "delay": delay, "policy": policy if part == "total" else "-",
                            "mean": r, "ci_lo": lo, "ci_hi": hi, "n_windows": len(WINDOWS)})
        run.save_csv("ratios_ci.csv", tab)
        run.save_csv("windows.csv", [{"design": d, "window": w, "e_logic_fj_per_mac": by[(d, w)]["e_logic_fj_per_mac"],
                                      "cycles_per_op": by[(d, w)]["cycles_per_op"]}
                                     for d in (*REF.values(), "bs4_macro_mx") for w in WINDOWS])
        run.log_metrics(m)
        for t in tab:
            print(f"{t['ratio']:9s} {t['part']:5s} {t['delay']:7s} {t['policy']:8s} {t['mean']:.4f} [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
