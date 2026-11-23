#!/usr/bin/env python3
"""Post-route power of the v2 column: OpenSTA + SPEF + one rebased VCD window.

The VCD comes from zero-delay Icarus GLS of the run's *pre-placement synthesis* netlist (post-CTS GLS fails
with clock-buffer delta races) and is annotated by net name onto the post-route netlist + SPEF;
CTS/repair buffers have no VCD entry; OpenSTA fills the Clock group from SDC clock activity (assumed toggle rate). Driven by the same e13 program/window (real0, M=16, 25 ns clock) with the e13 harness and flows/power/vcd_dump.v,
sliced with slice_vcd(rebase=True). Energy bookkeeping is identical to experiments/e13_power/run.py.

Usage: python flows/pnr/post_route_power.py <openlane run dir> <out dir> [--window real0] [--prog-exp e13_power_rb]
"""
import argparse, json, subprocess, sys, shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e13_power"))
sys.path.insert(0, str(ROOT / "flows/synth"))
import run as e13  # noqa: E402  (reuses parse_power, sram_accesses, constants)
import sweep  # noqa: E402
from cimfp4.power.vcd import slice_vcd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rundir"); ap.add_argument("out")
ap.add_argument("--window", default="real0"); ap.add_argument("--prog-exp", default="e13_power_rb")
ap.add_argument("--netlist-only", action="store_true")
ap.add_argument("--design", default="v2_sram_m16", help="e13 design key; top name from build/synth/synth.json")
a = ap.parse_args()
TOP = {"v2_sram_m16": "column_top", "bs_int5": "bs_column_top", "dpu": "dpu_column"}[a.design]
rd, out = Path(a.rundir).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
# (the flow stopped on deferred LVS/ss-corner checks before final/ was written: take the step outputs)
# step numbers differ per run (DPU s3 is shifted by one: 49/51, B14c), so resolve the step dirs by name


def step(name: str) -> Path:
    m = sorted(rd.glob(f"[0-9]*-{name}"), key=lambda p: int(p.name.split("-")[0]))
    if not m:
        raise SystemExit(f"no *-{name} step dir in {rd}")
    return m[0]  # first occurrence (later re-runs of a step carry a -1/-2 suffix and do not match)


net = step("openroad-fillinsertion") / f"{TOP}.nl.v"
spef = step("openroad-rcx") / f"nom/{TOP}.nom.spef"
prog_path = ROOT / "build/power" / a.prog_exp / "programs" / f"{a.window}.npz"
clk_ps = 25000
CELLS = e13.CELLS

# Zero-delay GLS of the post-CTS netlist races (clock-tree buffer delta delays vs data: 5773 output mismatches), so the
# VCD is taken from zero-delay GLS of the same flow's pre-placement synthesis netlist (identical logic cells/names) and
# annotated onto the post-route netlist + SPEF by net name. Clock-tree/repair buffers have no VCD activity entry.
simnet = step("yosys-synthesis") / f"{TOP}.nl.v"
# --- GLS
from cocotb_tools.runner import get_results, get_runner  # noqa: E402
vcd, win = out / "dump.vcd", out / "window.json"
runner = get_runner("icarus")
orig = runner._test_command
runner._test_command = lambda: [["-vcd" if x == "-none" else x for x in c] for c in orig()]
runner.build(sources=[CELLS / "primitives.v", CELLS / "sky130_fd_sc_hd.v", simnet, e13.SRAM_GLS, ROOT / "flows/power/vcd_dump.v"],
             hdl_toplevel=TOP, build_dir=out / "b", always=True,
             defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0", "VCD_FILE": f'"{vcd}"', "VCD_TOP": TOP},
             build_args=["-s", "vcd_dump"], timescale=("1ns", "1ps"))
xml = runner.test(hdl_toplevel=TOP, test_module="test_column_top", test_dir=ROOT / "tb/cocotb", build_dir=out / "b",
                  results_xml=str(out / "r.xml"),
                  extra_env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tb/cocotb'}", "CT_PROGRAM": str(prog_path),
                             "CT_WINDOW_OUT": str(win), "CT_M": "16", "CT_NOPS": "0", "CT_CLK_PS": str(clk_ps)})
if get_results(xml) != (1, 0):
    raise SystemExit("GLS mismatch on post-route netlist")
w = json.loads(win.read_text())
sliced = out / "window.vcd"
st = slice_vcd(vcd, sliced, w["t_start_ns"], w["t_end_ns"], rebase=True)
vcd.unlink()

# --- OpenSTA + SPEF
(out / "sta_spef.tcl").write_text("""read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
read_spef $::env(SPEF)
read_vcd -scope $::env(SCOPE) $::env(VCD)
report_power -digits 6 > $::env(OUT)
report_power -instances [get_cells -hierarchical -filter "ref_name =~ sky130_sram*"] -digits 6 >> $::env(OUT)
exit
""")
h = str(Path.home())
env = {"NETLIST": net, "TOP": TOP, "LIB": sweep.LIB, "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
       "SDC": ROOT / "flows/synth/constraints/column.sdc", "CLK_PERIOD_NS": clk_ps / 1e3, "SPEF": spef, "VCD": sliced,
       "SCOPE": TOP, "OUT": out / "power.rpt"}
cmd = ["docker", "run", "--rm", "--memory=4000m", "-v", f"{ROOT}:{ROOT}", "-v", f"{h}/.ciel:{h}/.ciel", "-w", str(ROOT)]
for k, v in env.items():
    cmd += ["-e", f"{k}={v}"]
r = subprocess.run(cmd + [e13.IMG, "sta", "-no_splash", "-exit", str(out / "sta_spef.tcl")], capture_output=True, text=True)
(out / "sta.log").write_text(r.stdout + r.stderr)
pw = e13.parse_power((out / "power.rpt").read_text())
dur_s = (w["t_end_ns"] - w["t_start_ns"]) * 1e-9
macs = w["n_ops"] * e13.MACS_PER_OP
prog = dict(__import__("numpy").load(prog_path, allow_pickle=True))
e_access_pj = yaml.safe_load((ROOT / "flows/pdk/sram_macros.yaml").read_text())["macros"][sweep.SRAM]["read_energy_pj_per_word_tt"]
p_logic = pw["p_total_w"] - pw["p_sram_sta_w"]
res = {**pw, "window": a.window, "n_ops": w["n_ops"], "macs": macs, "window_ns": dur_s * 1e9, "spef": str(spef),
       "p_logic_w": p_logic, "e_logic_fj_per_mac": p_logic * dur_s / macs * 1e15,
       "e_sram_fj_per_mac": e13.sram_accesses(prog, a.design) * e_access_pj * 1e3 / macs,
       "e_leak_fj_per_mac": pw["p_leakage_w"] * dur_s / macs * 1e15, "vcd_slice": st}
res["e_total_fj_per_mac"] = res["e_logic_fj_per_mac"] + res["e_sram_fj_per_mac"]
(out / "postroute_power.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
