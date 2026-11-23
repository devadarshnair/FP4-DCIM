"""e12 - sky130 synthesis + STA of all G1 designs (WP-27).

Runs flows/synth/sweep.py (Yosys, sky130_fd_sc_hd tt_025C_1v80, OpenRAM macro as black box) and
OpenSTA (openlane2 docker image) per netlist; records area, cells, NAND2-eq, critical path, fmax,
throughput, compute density and memory density, and the common clock f_G1 = min fmax over the G1
designs (v2_sram_m16, bs_int5, dpu) rounded down to 5 MHz.

Run: python experiments/e12_synth/run.py [--paper] [--skip-synth]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402

IMG = "ghcr.io/efabless/openlane2:2.3.10"
G1_DESIGNS = ("v2_sram_m16", "bs_int5", "dpu")
MAC_PER_CYCLE = {"bs_int5": 4 * 32 / 5}  # 4 columns, 5 cycles per block; others 4 x 32
TIMING_EXCLUDED = {"v2_latch_m16": "unoptimized 256-word latch read mux (ablation storage only)"}


def sta(design: str, top: str, period_ns: float = 10.0) -> dict:
    h = str(Path.home())
    out = REPO_ROOT / "build/synth" / f"{design}_sta.txt"
    env = {
        "NETLIST": REPO_ROOT / "build/synth" / f"{design}_netlist.v",
        "TOP": top,
        "LIB": sweep.LIB,
        "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
        "CLK_PERIOD_NS": period_ns,
        "SDC": REPO_ROOT / "flows/synth/constraints/column.sdc",
        "OUT": out,
    }
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
        "-v",
        f"{h}/.ciel:{h}/.ciel",
        "-w",
        str(REPO_ROOT),
    ]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    subprocess.run(
        cmd + [IMG, "sta", "-no_splash", "-exit", "flows/synth/sta.tcl"], check=True, capture_output=True
    )
    txt = out.read_text()
    arrival = float(re.search(r"([\d.]+)\s+data arrival time", txt).group(1))
    return {"crit_path_ns": arrival, "fmax_mhz": 1e3 / arrival}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--skip-synth", action="store_true", help="reuse build/synth netlists")
    args = ap.parse_args()
    cfg = {
        "lib": str(sweep.LIB.name),
        "sram": sweep.SRAM,
        "designs": sweep.DESIGNS,
        "sta_image": IMG,
        "flow": "yosys synth -flatten; dfflibmap; abc -liberty (no delay target); OpenSTA tt",
    }
    with start_run("e12_synth", cfg, paper=args.paper) as run:
        rows = (
            json.loads((REPO_ROOT / "build/synth/synth.json").read_text())
            if args.skip_synth
            else sweep.run(list(sweep.DESIGNS), REPO_ROOT / "build/synth")
        )
        for r in rows:
            r.update(sta(r["design"], r["top"]))
            r["timing_excluded"] = TIMING_EXCLUDED.get(r["design"], "")
            r["mac_per_cycle"] = MAC_PER_CYCLE.get(r["design"], 4 * 32)
            r["total_mm2"] = r["total_area_um2"] / 1e6
            r["kb_per_mm2"] = (4 * 16 * 32 * 4 / 1024 if r.get("M", 16) == 16 else 4 * 32 * 4 / 1024) / r[
                "total_mm2"
            ]
        f_g1 = 5 * math.floor(min(r["fmax_mhz"] for r in rows if r["design"] in G1_DESIGNS) / 5)
        for r in rows:
            r["gops_per_mm2_at_fg1"] = 2 * r["mac_per_cycle"] * f_g1 * 1e6 / 1e9 / r["total_mm2"]
        run.save_csv("synth.csv", rows)
        by = {r["design"]: r for r in rows}
        run.log_metrics(
            {
                "f_g1_mhz": f_g1,
                "density_ratio_v2_over_int5": by["v2_sram_m16"]["gops_per_mm2_at_fg1"]
                / by["bs_int5"]["gops_per_mm2_at_fg1"],
                "density_ratio_v2_over_dpu": by["v2_sram_m16"]["gops_per_mm2_at_fg1"]
                / by["dpu"]["gops_per_mm2_at_fg1"],
                "leaf_logic_saving_vs_generic": 1
                - by["v2_sram_m16"]["logic_area_um2"] / by["v2_generic"]["logic_area_um2"],
                **{f"{d}_total_mm2": by[d]["total_mm2"] for d in by},
                **{f"{d}_fmax_mhz": by[d]["fmax_mhz"] for d in by if not by[d]["timing_excluded"]},
            }
        )
        for r in rows:
            print(
                f"{r['design']:14s} {r['total_mm2']:.4f} mm2  fmax {r['fmax_mhz']:6.1f} MHz  "
                f"{r['gops_per_mm2_at_fg1']:7.1f} GOPS/mm2 @f_G1  {r['kb_per_mm2']:5.1f} Kb/mm2 {r['timing_excluded']}"
            )
        print("f_G1 =", f_g1, "MHz")


if __name__ == "__main__":
    main()
