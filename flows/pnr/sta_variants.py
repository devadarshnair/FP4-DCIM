#!/usr/bin/env python3
"""Decompose post-route power: same rebased VCD window, STA on (a) OL pre-placement synth netlist, (b) post-route
netlist without SPEF, (c) post-route netlist + SPEF (= post_route_power.py). Usage: sta_variants.py <rundir> <postroute dir>"""
import json, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e13_power")); sys.path.insert(0, str(ROOT / "flows/synth"))
import run as e13, sweep  # noqa
rd, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
TOP = {"v2_sram_m16": "column_top", "bs_int5": "bs_column_top", "dpu": "dpu_column"}[sys.argv[3] if len(sys.argv) > 3 else "v2_sram_m16"]
w = json.loads((out / "window.json").read_text()); dur = (w["t_end_ns"] - w["t_start_ns"]) * 1e-9; macs = w["n_ops"] * e13.MACS_PER_OP
tcl = out / "sta_var.tcl"
tcl.write_text("""read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
if {[info exists ::env(SPEF)] && $::env(SPEF) ne ""} { read_spef $::env(SPEF) }
read_vcd -scope $::env(TOP) $::env(VCD)
report_power -digits 6 > $::env(OUT)
report_power -instances [get_cells -hierarchical -filter "ref_name =~ sky130_sram*"] -digits 6 >> $::env(OUT)
exit
""")
h = str(Path.home())
V = {"a_synth_nospef": (rd / f"02-yosys-synthesis/{TOP}.nl.v", ""),
     "b_postroute_nospef": (rd / f"48-openroad-fillinsertion/{TOP}.nl.v", ""),
     "c_postroute_spef": (rd / f"48-openroad-fillinsertion/{TOP}.nl.v", rd / f"50-openroad-rcx/nom/{TOP}.nom.spef")}
res = {}
for k, (net, spef) in V.items():
    env = {"TOP": TOP, "NETLIST": net, "LIB": sweep.LIB, "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
           "SDC": ROOT / "flows/synth/constraints/column.sdc", "CLK_PERIOD_NS": 25.0, "SPEF": spef,
           "VCD": out / "window.vcd", "OUT": out / f"power_{k}.rpt"}
    cmd = ["docker", "run", "--rm", "--memory=4000m", "-v", f"{ROOT}:{ROOT}", "-v", f"{h}/.ciel:{h}/.ciel", "-w", str(ROOT)]
    for a, b in env.items(): cmd += ["-e", f"{a}={b}"]
    r = subprocess.run(cmd + [e13.IMG, "sta", "-no_splash", "-exit", str(tcl)], capture_output=True, text=True)
    pw = e13.parse_power((out / f"power_{k}.rpt").read_text())
    ann = [l for l in (r.stdout + r.stderr).splitlines() if "Annotated" in l]
    pl = pw["p_total_w"] - pw["p_sram_sta_w"]
    res[k] = {**pw, "annot": ann, "e_logic_fj_per_mac": pl * dur / macs * 1e15}
    print(k, res[k]["e_logic_fj_per_mac"], ann, flush=True)
(out / "sta_variants.json").write_text(json.dumps(res, indent=2, default=str))
