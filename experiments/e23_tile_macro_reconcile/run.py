"""e23 - tile-vs-macro reconciliation runs (gap between the e15 tile ratios and the e22 macro ratios).

Question: why do the tile (e13/e15, 4-column group) and macro (e21/e22, 32x32) EE ratios disagree
(v2/INT5 4.02 tile vs 3.10 macro; v2/DPU 1.52 tile vs 1.92 macro)? This script produces the few
*measured* tile-level points the decomposition needs; experiments/e23_tile_macro_reconcile/analyze.py
assembles the reconciliation tables. No macro re-synthesis, no macro re-simulation.

Tile netlist variants (4 columns; Yosys + abc, sky130_fd_sc_hd tt, same flow as e12_synth):
  flat     build/synth/<d>_netlist.v (the e13 netlists, `synth -flatten`): the MX-only column_top has
           nv_mode = 0 and the NV scale inputs constant at the column ports, so Yosys constant-folds the
           whole NVFP4 path (half-tree outputs, u*u products, second shifter, NV pipeline registers).
  keepcol  `synth -flatten` with keep_hierarchy on the `column` (bs_column) module: leaves, tree and
           stage 2 are still optimised together inside the column, but the column's NV ports are no
           longer constants, so the NV datapath is kept exactly as in the flat macro (where op_nv and the
           scale RF fields are primary inputs). This is the flat-equivalent "macro column" at tile scale.
  hier     `synth` without -flatten (abc per module): the flow of the DPU macro netlist (flat DPU macro
           exceeded RAM, e12). Leaves, tree, column, store, row drivers are separate modules.

Stimulus:
  tile     the e13_power_rb programs real0..3 (the same 4 capture layers as the macro windows).
  mslice   the e21 macro MX programs real0..3, column group g (4 of 32 columns: weights, block
           exponents, anchors of those columns; identical activations, sets, reuse, clear): the tile then
           processes exactly the data one OpenRAM group of the macro sees.

Per run: Icarus GLS (bit-exact vs the scoreboard, tb/cocotb/test_column_top.py), VCD sliced to the
compute window and rebased to t = 0, native OpenSTA 3.1.0 (as e21) report_power plus a per-leaf-
instance report aggregated by hierarchy (hier / keepcol netlists). Logic energy excludes the OpenRAM
Liberty power (as e13/e21).

Run (one job; light: 4-column netlists):
  python scripts/safe_run.py --max-rss-mb 3000 -- python experiments/e23_tile_macro_reconcile/run.py --synth
  python scripts/safe_run.py --max-rss-mb 3000 -- python experiments/e23_tile_macro_reconcile/run.py

--recipe td (T2.7): the T2.6 timing-driven recipe (docs/spec/synth_recipe.md). flat = the e30 td tile netlists
(build/e30_td_synth/td), keepcol / hier = the same e30 build_one (Yosys td + OpenROAD repair) with the column kept
or no flattening; mslice programs from the T2.7 e21 macro run. Experiments e23_tile_synth_td /
e23_tile_macro_reconcile_td; build tree build/e23_reconcile/td.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import load_program, save_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402

HERE = Path(__file__).resolve().parent
STA_BIN = os.environ.get("CIM_STA_BIN", "sta")
CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
SRAM_GLS = REPO_ROOT / "build/models/sky130_sram_1kbyte_1rw1r_32x256_8_gls.v"
BUILD = REPO_ROOT / "build/e23_reconcile"
E13_PROGS = REPO_ROOT / "build/power/e13_power_rb/programs"
E21_RUN = "20261110T050824693997Z_f25a5c5"
E21_PROGS = REPO_ROOT / "build/e21_macro_power" / E21_RUN / "programs"
MACS_PER_OP = 4 * 32
CLK_PS = 25000
DESIGNS = ("v2_sram_m16", "bs_int5", "dpu")
TAIL = {"v2_sram_m16": 3, "dpu": 3, "bs_int5": 9}  # e21: cycles - n_ops - stall_cycles (537-512-22, 3098-512-2577)
RECIPE = "base"
E21_TD_RUN = "20261114T023643822998Z_cada632"  # T2.7 e21 (td macro netlists, fixed INT5)


def configure(recipe: str) -> None:
    global RECIPE, BUILD, E21_RUN, E21_PROGS
    RECIPE = recipe
    if recipe == "td":
        BUILD = REPO_ROOT / "build/e23_reconcile/td"
        E21_RUN = E21_TD_RUN
        E21_PROGS = REPO_ROOT / "build/e21_macro_power" / E21_RUN / "programs"


KEEP = {"v2_sram_m16": r"$paramod*\column", "dpu": r"$paramod*\column", "bs_int5": r"$paramod\bs_column*"}


def netlist(variant: str, design: str) -> Path:
    if variant == "flat":
        if RECIPE == "td":
            return REPO_ROOT / "build/e30_td_synth/td" / f"{design}_netlist.v"
        return REPO_ROOT / "build/synth" / f"{design}_netlist.v"
    return BUILD / f"synth_{variant}" / f"{design}_netlist.v"


def synth(variant: str, design: str) -> dict:
    """Yosys run of one tile variant (keepcol | hier) via flows/synth/sweep.py's script generator."""
    out = BUILD / f"synth_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    if RECIPE == "td":  # e30 build_one: Yosys td recipe + OpenROAD repair_design / repair_timing, final stat + STA
        import importlib.util
        spec = importlib.util.spec_from_file_location("e30_run", REPO_ROOT / "experiments/e30_td_synth/run.py")
        e30 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e30)
        top, params = sweep.DESIGNS[design]
        e30.KEEP = KEEP[design]
        row, _ = e30.build_one(design, top, params, out, "td", variant == "keepcol", variant == "hier", "3g")
        sweep.RECIPE, sweep.HIER = "base", False
        return {"variant": variant, "design": design, **row}
    sweep.HIER = variant == "hier"
    top, params = sweep.DESIGNS[design]
    ys = sweep.yosys_script(design, top, params, out)
    if variant == "keepcol":  # keep the column module boundary inside the otherwise flat run
        ys = ys.replace("\nsynth -flatten", f"\nsetattr -mod -set keep_hierarchy 1 {KEEP[design]}\nsynth -flatten")
        ys = ys.replace(" stat -liberty", f" stat -top {top} -liberty")
    (out / f"{design}.ys").write_text(ys)
    with (out / f"{design}_yosys.log").open("w") as f:
        subprocess.run([shutil.which("yosys"), "-q", "-l", str(out / f"{design}_yosys_full.log"), "-s",
                        str(out / f"{design}.ys")], stdout=f, stderr=subprocess.STDOUT, check=True)
    net = out / f"{design}_netlist.v"
    net.write_text(re.sub(r"\b(input|output|wire|reg)\s+signed\b", r"\1", net.read_text()))
    st = sweep.parse_stat((out / f"{design}_stat.txt").read_text())
    st["logic_area_um2"] = st["total_area_um2"] - st["sram_instances"] * 190712.55  # flows/pdk/sram_macros.yaml
    return {"variant": variant, "design": design, "top": top, **st}


def mslice_program(window: int, group: int) -> dict:
    """Column group `group` of e21 macro MX program real<window> in the column_top program format."""
    p = load_program(E21_PROGS / f"mxfp4_real{window}.npz")
    cols = slice(4 * group, 4 * group + 4)
    return {"sets": p["sets"][:, cols].copy(), "ew": p["e0"][:, cols].copy(), "maxw": p["maxw"][cols].copy(),
            "x": p["x"], "e_x": p["e_x0"], "max_e_x": p["max_e_x"], "set": p["set"], "clear": p["clear"]}


def gls(net: Path, top: str, program: Path, wd: Path) -> dict:
    from cocotb_tools.runner import get_results, get_runner

    vcd, win = wd / "dump.vcd", wd / "window.json"
    srcs = [CELLS / "primitives.v", CELLS / "sky130_fd_sc_hd.v", net, SRAM_GLS, REPO_ROOT / "flows/power/vcd_dump.v"]
    runner = get_runner("icarus")
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if a == "-none" else a for a in c] for c in orig()]
    runner.build(sources=srcs, hdl_toplevel=top, build_dir=wd / "b", always=True,
                 defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0", "VCD_FILE": f'"{vcd}"', "VCD_TOP": top},
                 build_args=["-s", "vcd_dump"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel=top, test_module="test_column_top", test_dir=REPO_ROOT / "tb/cocotb",
                      build_dir=wd / "b", results_xml=str(wd / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tb/cocotb'}",
                                 "CT_PROGRAM": str(program), "CT_WINDOW_OUT": str(win), "CT_M": "16",
                                 "CT_NOPS": "0", "CT_CLK_PS": str(CLK_PS)})
    if get_results(xml) != (1, 0):
        raise RuntimeError(f"GLS mismatch for {net} on {program.name}")
    return {**json.loads(win.read_text()), "vcd": vcd}


def sta(net: Path, top: str, vcd: Path, wd: Path) -> tuple[dict, dict]:
    env = {**os.environ, "NETLIST": str(net), "TOP": top, "LIB": str(sweep.LIB),
           "SRAM_LIB": str(sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib")),
           "SDC": str(REPO_ROOT / "flows/synth/constraints/column.sdc"), "CLK_PERIOD_NS": str(CLK_PS / 1e3),
           "VCD": str(vcd), "SCOPE": top, "OUT": str(wd / "power.rpt"), "OUT_INST": str(wd / "inst.rpt")}
    r = subprocess.run([STA_BIN, "-no_splash", "-exit", str(HERE / "sta_power_inst.tcl")], capture_output=True,
                       text=True, env=env, cwd=REPO_ROOT)
    (wd / "sta.log").write_text(r.stdout + r.stderr)
    if r.returncode or not (wd / "power.rpt").exists():
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return groups((wd / "power.rpt").read_text()), modules(wd / "inst.rpt")


def groups(txt: str) -> dict:
    out = {}
    for g in ("Sequential", "Combinational", "Clock", "Macro", "Total"):
        m = re.search(rf"^{g}\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt, re.M)
        out[f"p_{g.lower()}_w"] = float(m.group(4)) if m else 0.0
        if g == "Total" and m:
            out["p_total_leak_w"] = float(m.group(3))
    sm = re.findall(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+\S*sram\S*\s*$", txt, re.M)
    out["p_sram_sta_w"] = sum(float(s[3]) for s in sm[:1]) if sm else 0.0  # one OpenRAM macro per tile
    return out


MODULE_RULES = [  # (regex on the hierarchical leaf-instance path, module class); first match wins
    (r"u_sram$", "sram_liberty"),
    (r"(^|[/.])g_leaf\[\d+\]\.[^/]*u_leaf/", "leaf"),
    (r"(^|[/.])u_tree/", "tree"),
    (r"(^|[/.])g_col\[\d+\]\.u_col/", "column_local"),
    (r"(^|[/.])(g_sram\.)?u_store/", "store"),
    (r"(^|[/.])g_row\[\d+\]\.u_rd/", "row_driver"),
    (r"(^|[/.])u_icg", "icg"),
]


def modules(inst_rpt: Path) -> dict:
    """Sum leaf-instance total power (W) by module class; hierarchical instances (prefixes) excluded."""
    rows = re.findall(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\S+)\s*$",
                      inst_rpt.read_text(), re.M)
    names = {r[4] for r in rows}
    hier = {n.rsplit("/", 1)[0] for n in names if "/" in n}
    agg = defaultdict(float)
    for *_, tot, name in rows:
        if name in hier:
            continue
        clean = name.replace("\\", "")  # OpenSTA escapes [ ] of Yosys instance names
        cls = next((c for pat, c in MODULE_RULES if re.search(pat, clean)), "top_local")
        agg[cls] += float(tot)
    agg["n_leaf"] = sum(1 for n in names if n not in hier)
    return dict(agg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true", help="synthesise the keepcol / hier tile variants and exit")
    ap.add_argument("--plan", default="default", help="default | smoke")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--recipe", default="base", choices=("base", "td"))
    a = ap.parse_args()
    configure(a.recipe)
    sfx = "_td" if a.recipe == "td" else ""
    BUILD.mkdir(parents=True, exist_ok=True)
    if a.synth:
        with start_run("e23_tile_synth" + sfx, {"variants": ["keepcol", "hier"], "designs": DESIGNS, "recipe": a.recipe,
                                          "flow": "flows/synth/sweep.py script; keepcol = keep_hierarchy on column"
                                                  + ("; td: e30 build_one (docs/spec/synth_recipe.md)" if sfx else "")},
                       paper=a.paper) as run:
            rows = []
            for v in ("keepcol", "hier"):
                for d in DESIGNS:
                    t0 = time.time()
                    r = synth(v, d)
                    r["synth_s"] = time.time() - t0
                    rows.append(r)
                    shutil.copy(BUILD / f"synth_{v}" / f"{d}_stat.txt", run.path(f"{v}_{d}_stat.txt"))
                    for extra in ("final_stat.txt", "sta.txt", "sta_r2r.txt"):
                        if sfx and (BUILD / f"synth_{v}" / f"{d}_{extra}").exists():
                            shutil.copy(BUILD / f"synth_{v}" / f"{d}_{extra}", run.path(f"{v}_{d}_{extra}"))
                    shutil.copy(BUILD / f"synth_{v}" / f"{d}.ys", run.path(f"{v}_{d}.ys"))
                    print(r, flush=True)
            run.save_csv("synth.csv", rows)
        return

    tops = {r["design"]: r["top"] for r in json.loads((REPO_ROOT / "build/synth/synth.json").read_text())}
    # (variant, stimulus, window, group)
    if a.plan == "smoke":
        plan = [("hier", "mslice", 0, 0)]
    else:
        plan = [("flat", "tile", 0, None)]                                # tool check vs e13 (docker STA 2.6.0)
        plan += [("flat", "mslice", w, g) for w in range(4) for g in (0, 5)]   # workload / slicing
        plan += [("keepcol", "mslice", w, g) for w in range(4) for g in (0, 5)]  # NV datapath kept
        plan += [("hier", "mslice", w, 0) for w in range(4)]               # hierarchical flow
    cfg = {"plan": [list(p) for p in plan], "designs": DESIGNS, "e21_programs": str(E21_PROGS.relative_to(REPO_ROOT)),
           "e13_programs": str(E13_PROGS.relative_to(REPO_ROOT)), "clk_ps": CLK_PS, "recipe": a.recipe,
           "sta": "OpenSTA native (as e21), ideal clock, tt_025C_1v80; VCD rebased (B9)",
           "netlists": {v: {d: str(netlist(v, d).relative_to(REPO_ROOT)) for d in DESIGNS} for v in ("flat", "keepcol", "hier")}}
    with start_run("e23_tile_macro_reconcile" + sfx, cfg, paper=a.paper) as run:
        rows = []
        for variant, stim, w, g in plan:
            if stim == "tile":
                ppath, name = E13_PROGS / f"real{w}.npz", f"tile_real{w}"
            else:
                name = f"mslice_real{w}_g{g}"
                ppath = save_program(BUILD / "programs" / f"{name}.npz", mslice_program(w, g))
            for d in DESIGNS:
                wd = BUILD / run.run_id / variant / d / name
                wd.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                gw = gls(netlist(variant, d), tops[d], ppath, wd)
                # window conventions: e13 = through the 16 drain cycles of the tile harness; macro = e21's
                # [t_start, t_last_valid + 1 clk] = n_ops + stalls + TAIL[d] cycles (TAIL from the e21 rows)
                convs = {"macro": gw["t_start_ns"] + (gw["n_ops"] + gw["stall_cycles"] + TAIL[d]) * CLK_PS / 1e3}
                if stim == "tile":
                    convs["e13"] = gw["t_end_ns"]
                for conv, t1 in convs.items():
                    sliced = wd / "window.vcd"
                    slice_vcd(gw["vcd"], sliced, gw["t_start_ns"], t1, rebase=True)
                    pw, mods = sta(netlist(variant, d), tops[d], sliced, wd)
                    sliced.unlink()
                    dur = (t1 - gw["t_start_ns"]) * 1e-9
                    macs = gw["n_ops"] * MACS_PER_OP
                    k = dur / macs * 1e15
                    row = {"variant": variant, "design": d, "stim": stim, "window": w,
                           "group": "" if g is None else g, "conv": conv, "n_ops": gw["n_ops"],
                           "stall_cycles": gw["stall_cycles"], "cycles": round(dur * 1e9 * 1e3 / CLK_PS),
                           "window_ns": dur * 1e9,
                           "e_logic_fj_per_mac": (pw["p_total_w"] - pw["p_sram_sta_w"]) * k,
                           "e_seq_fj_per_mac": pw["p_sequential_w"] * k,
                           "e_comb_fj_per_mac": pw["p_combinational_w"] * k,
                           "e_clock_fj_per_mac": pw["p_clock_w"] * k,
                           **{f"m_{c}_fj_per_mac": v * k for c, v in mods.items() if c != "n_leaf"},
                           "n_leaf": mods.get("n_leaf", 0), "wall_s": time.time() - t0}
                    rows.append(row)
                    run.save_csv("power.csv", rows)
                    shutil.copy(wd / "power.rpt", run.path(f"power_{variant}_{d}_{name}_{conv}.rpt"))
                    print(f"{variant:8s} {d:12s} {name:18s} {conv:5s} logic {row['e_logic_fj_per_mac']:8.1f} fJ/MAC "
                          f"(seq {row['e_seq_fj_per_mac']:.0f}, comb {row['e_comb_fj_per_mac']:.0f}, "
                          f"clk {row['e_clock_fj_per_mac']:.0f}) {row['wall_s']:.0f}s", flush=True)
                    if (wd / "inst.rpt").exists():  # kept (gzip) for re-aggregation
                        subprocess.run(["gzip", "-f", "-S", f".{conv}.gz", str(wd / "inst.rpt")], check=True)
                gw["vcd"].unlink()
        run.log_metrics({"n_rows": len(rows)})


if __name__ == "__main__":
    main()
