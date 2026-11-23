"""e12 macro - Sky130 synthesis + OpenSTA of the full 32-column macros (v2_macro, bs_macro, dpu_macro) with provenance.

Same authoritative flow as experiments/e12_synth (flows/synth/sweep.py: Yosys synth -flatten, dfflibmap,
abc -liberty with no delay target, sky130_fd_sc_hd tt_025C_1v80, OpenRAM 1 KB macros as black boxes;
OpenSTA from the openlane2 docker image, container memory-capped). Output: results/e12_macro_synth_<design>/<run>/.
No power here (power needs macro-level GLS + VCD; see e21).

Run (one design at a time):
  python scripts/safe_run.py --max-rss-mb 5500 -- python experiments/e12_macro_synth/run.py --design v2_macro
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402

IMG = "ghcr.io/efabless/openlane2:2.3.10"
OUTDIR = {"v2_macro": "build/synth_macro_v2", "bs_macro": "build/synth_macro_bs",
          "dpu_macro": "build/synth_macro_dpu"}
PERIOD_NS = 10.0
DOCKER_MEM = "4g"


def sta(design: str, top: str, out_dir: Path, period_ns: float) -> dict:
    h = str(Path.home())
    rpt = out_dir / f"{design}_sta.txt"
    env = {
        "NETLIST": out_dir / f"{design}_netlist.v",
        "TOP": top,
        "LIB": sweep.LIB,
        "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
        "CLK_PERIOD_NS": period_ns,
        "SDC": REPO_ROOT / "flows/synth/constraints/column.sdc",
        "OUT": rpt,
        "OUT_R2R": out_dir / f"{design}_sta_r2r.txt",
    }
    cmd = ["docker", "run", "--rm", f"--memory={DOCKER_MEM}", f"--memory-swap={DOCKER_MEM}",
           "-v", f"{REPO_ROOT}:{REPO_ROOT}", "-v", f"{h}/.ciel:{h}/.ciel", "-w", str(REPO_ROOT)]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    ver = subprocess.run(["docker", "run", "--rm", IMG, "sta", "-version"], capture_output=True, text=True)
    subprocess.run(cmd + [IMG, "sta", "-no_splash", "-exit", "flows/synth/sta_macro.tcl"], check=True, capture_output=True)
    txt = rpt.read_text()
    arrival = float(re.search(r"([\d.]+)\s+data arrival time", txt).group(1))
    setup = re.search(r"(-?[\d.]+)\s+[\d.]+\s+library setup time", txt)
    wns = float(re.search(r"worst slack\s+(-?[\d.]+)", txt).group(1))
    sp = re.search(r"Startpoint: (\S+) \(([^)]*)\)", txt).group(0)
    ep = re.search(r"Endpoint: (\S+) \(([^)]*)\)", txt).group(0)
    min_period = period_ns - wns  # ideal clock, zero skew/uncertainty
    r2r = (out_dir / f"{design}_sta_r2r.txt").read_text()
    r_arr = float(re.search(r"([\d.]+)\s+data arrival time", r2r).group(1))
    r_wns = float(re.search(r"(-?[\d.]+)\s+slack \(", r2r).group(1))  # path slack (worst slack line is global)
    r_sp = re.search(r"Startpoint: (\S+)", r2r).group(1)
    r_ep = re.search(r"Endpoint: (\S+)", r2r).group(1)
    return {
        "r2r_crit_arrival_ns": r_arr,
        "r2r_wns_ns": r_wns,
        "r2r_min_period_ns": period_ns - r_wns,
        "r2r_fmax_mhz": 1e3 / (period_ns - r_wns),
        "r2r_startpoint": r_sp,
        "r2r_endpoint": r_ep,
        "sta_tool_version": ver.stdout.strip(),
        "sta_period_ns": period_ns,
        "wns_ns": wns,
        "crit_arrival_ns": arrival,
        "setup_ns": float(setup.group(1)) if setup else None,
        "min_period_ns": min_period,
        "fmax_mhz": 1e3 / min_period,
        "startpoint": sp,
        "endpoint": ep,
        "path_through_sram": bool(re.search(r"sky130_sram", txt)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=list(OUTDIR))
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--hier", action="store_true",
                    help="hierarchical Yosys (no -flatten; abc per module): fallback when the flat run exceeds RAM")
    ap.add_argument("--period-ns", type=float, default=PERIOD_NS)
    a = ap.parse_args()
    top, params = sweep.DESIGNS[a.design]
    sweep.HIER = a.hier
    out = REPO_ROOT / (OUTDIR[a.design] + ("_hier" if a.hier else ""))
    cfg = {
        "design": a.design, "top": top, "params": params, "lib": sweep.LIB.name, "sram": sweep.SRAM,
        "sram_lib": f"{sweep.SRAM}_TT_1p8V_25C.lib", "sta_image": IMG, "sdc": "flows/synth/constraints/column.sdc", "sta_tcl": "flows/synth/sta_macro.tcl",
        "sta_period_ns": a.period_ns, "sources": sweep.SOURCES,
        "hier": a.hier, "flow": ("yosys synth (hierarchical, abc per module); " if a.hier else "yosys synth -flatten; ") + " dfflibmap; abc -liberty (no delay target); OpenSTA ideal clock, tt_025C_1v80",
    }
    with start_run(f"e12_macro_synth_{a.design}" + ("_hier" if a.hier else ""), cfg, paper=a.paper) as run:
        row = sweep.run([a.design], out)[0]
        t = sta(a.design, top, out, a.period_ns)
        row.update(t)
        row["total_mm2"] = row["total_area_um2"] / 1e6
        run.save_csv("synth.csv", [row])
        for suffix in ("stat.txt", "yosys.log", "yosys_full.log", "sta.txt", "sta_r2r.txt"):
            shutil.copy(out / f"{a.design}_{suffix}", run.path(f"{a.design}_{suffix}"))
        shutil.copy(out / f"{a.design}.ys", run.path(f"{a.design}.ys"))
        run.log_metrics({"corner": "sky130_fd_sc_hd tt_025C_1v80 + OpenRAM TT_1p8V_25C; ideal clock; no power",
                         "timing_caveat": "synthesis-only, unbuffered (abc no delay target, no fanout buffering): fmax values are pessimistic lower bounds, NOT achievable frequency; do not quote",
                         **row})
        print({k: row[k] for k in ("design", "cells", "total_mm2", "logic_nand2_eq", "min_period_ns", "fmax_mhz", "r2r_min_period_ns", "r2r_fmax_mhz")})


if __name__ == "__main__":
    main()
