"""e30 - T2.6 timing-driven resynthesis of the energy netlists (docs/spec/synth_recipe.md; T2.6).

One recipe for every design (v2 / bit-serial INT5 (fixed RTL) / DPU, NV_GATE = 0/1):
  stage 1  Yosys `synth -flatten`, dfflibmap, `abc -D 20000 -constr` (buffer; upsize; dnsize) with a dont-use set
           (flows/synth/sweep.py, SYNTH_RECIPE=td)
  stage 2  OpenROAD buffer_ports + repair_design (max fanout 10, max transition 0.75 ns, Liberty max cap) +
           repair_timing -setup at 25 ns, ideal clock, no placement (flows/synth/td_repair.tcl)
Recipe-sensitivity axis: base (the e12/e13 recipe, re-run on the current RTL) | td | td_ol (td + the OpenLane2
no_synth.cells dont-use list, i.e. no x1 drive strengths: the cell-set choice of the e24 OL2 synthesis).

Phases (each one safe_run job):
  --synth    tile netlists (the e13/e24 4-column tiles v2_sram_m16 / bs_int5 / dpu, and the keepcol v2 tile with
             NV_GATE 0/1) for every recipe: area, cells, buffers, OpenSTA (docker, as e12) WNS any-path and
             register-to-register at 25 ns. -> results/e30_td_synth
  --power    zero-delay GLS + rebased VCD + OpenSTA report_power on the e13_power_rb programs (8 real + 2 random
             windows; same flow as e13: Icarus, docker OpenSTA 2.6.0, SRAM per-access charge). -> results/e30_td_power
  --macro D  macro netlist (v2_macro | bs_macro | bs4_macro (T5.4 D2 INT4-SM) | dpu_macro, dpu kept hierarchical) with the td recipe: area +
             STA only. -> results/e30_td_macro_<D>

Run:
  python scripts/safe_run.py --max-rss-mb 4500 -- python experiments/e30_td_synth/run.py --synth
  python scripts/safe_run.py --max-rss-mb 4500 -- python experiments/e30_td_synth/run.py --power
  python scripts/safe_run.py --max-rss-mb 5500 -- python experiments/e30_td_synth/run.py --macro v2_macro
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import load_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e13 = _load("e13_run", REPO_ROOT / "experiments/e13_power/run.py")  # parse_power, sram_accesses, bootstrap_ci
e23 = _load("e23_run", REPO_ROOT / "experiments/e23_tile_macro_reconcile/run.py")  # gls(net, top, program, wd)

IMG = "ghcr.io/efabless/openlane2:2.3.10"
PDK = Path.home() / ".ciel/sky130A"
TECH_LEF = PDK / "libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef"
CELL_LEF = PDK / "libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef"
SRAM_LEF = sweep.SRAM_DIR / "lef" / f"{sweep.SRAM}.lef"
SRAM_LIB = sweep.SRAM_DIR / "lib" / f"{sweep.SRAM}_TT_1p8V_25C.lib"
SDC = REPO_ROOT / "flows/synth/constraints/column.sdc"
BUILD = REPO_ROOT / "build/e30_td_synth"
PROGS = REPO_ROOT / "build/power/e13_power_rb/programs"
WINDOWS = [f"real{i}" for i in range(8)] + ["rand0", "rand1"]
PERIOD_NS = sweep.TD_PERIOD_NS
MAX_FANOUT = 10
MAX_TRAN_NS = 0.75
CLK_PS = 25000
MACS_PER_OP = 4 * 32
SRAM_AREA = 190712.55  # flows/pdk/sram_macros.yaml, TT

V2 = ("column_top", {"STORE": 0, "M": 16, "LEAF": 0})
TILES = {  # name -> (sweep design, top, params, keepcol)
    "v2_sram_m16": ("v2_sram_m16", *V2, False),
    "bs_int5": ("bs_int5", *sweep.DESIGNS["bs_int5"], False),
    "dpu": ("dpu", *sweep.DESIGNS["dpu"], False),
    "v2_keepcol_nvg0": ("v2_sram_m16", V2[0], {**V2[1], "NV_GATE": 0}, True),
    "v2_keepcol_nvg1": ("v2_sram_m16", V2[0], {**V2[1], "NV_GATE": 1}, True),
}
RECIPES = ("base", "td", "td_ol")
PLAN_SYNTH = [(r, d) for r in RECIPES for d in ("v2_sram_m16", "bs_int5", "dpu")] + [
    ("base", "v2_keepcol_nvg0"), ("base", "v2_keepcol_nvg1"), ("td", "v2_keepcol_nvg0"), ("td", "v2_keepcol_nvg1")]
KEEP = r"$paramod*\column"  # e23 keepcol (v2 only here)
MACRO_HIER = {"dpu_macro": True}  # DPU macro stays hierarchical (flat Yosys OOMs on 7.5 GB)


def docker(args: list[str], env: dict, mem: str = "3g") -> subprocess.CompletedProcess:
    h = str(Path.home())
    ids = f"{os.getuid()}:{os.getgid()}"  # files written by the container stay user-owned
    cmd = ["docker", "run", "--rm", "-u", ids, f"--memory={mem}", f"--memory-swap={mem}", "-v", f"{REPO_ROOT}:{REPO_ROOT}",
           "-v", f"{h}/.ciel:{h}/.ciel", "-w", str(REPO_ROOT)]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    return subprocess.run(cmd + [IMG, *args], capture_output=True, text=True)


def unsign(net: Path) -> None:  # OpenSTA's reader rejects `signed` (no meaning in a mapped netlist)
    net.write_text(re.sub(r"\b(input|output|wire|reg)\s+signed\b", r"\1", net.read_text()))


def yosys(name: str, top: str, params: dict, out: Path, recipe: str, keepcol: bool, hier: bool = False) -> Path:
    sweep.RECIPE, sweep.HIER = recipe, hier
    ys = sweep.yosys_script(name, top, params, out)
    if keepcol:
        ys = ys.replace("\nsynth -flatten", f"\nsetattr -mod -set keep_hierarchy 1 {KEEP}\nsynth -flatten")
        ys = ys.replace(" stat -liberty", f" stat -top {top} -liberty")
    (out / f"{name}.ys").write_text(ys)
    with (out / f"{name}_yosys.log").open("w") as f:
        subprocess.run([shutil.which("yosys"), "-q", "-l", str(out / f"{name}_yosys_full.log"), "-s",
                        str(out / f"{name}.ys")], stdout=f, stderr=subprocess.STDOUT, check=True)
    net = out / f"{name}_netlist.v"
    unsign(net)
    return net


def repair(net: Path, top: str, out_net: Path, mem: str) -> dict:
    env = {"TECH_LEF": TECH_LEF, "CELL_LEF": CELL_LEF, "SRAM_LEF": SRAM_LEF, "LIB": sweep.LIB, "SRAM_LIB": SRAM_LIB,
           "NETLIST": net, "TOP": top, "SDC": SDC, "CLK_PERIOD_NS": PERIOD_NS, "MAX_FANOUT": MAX_FANOUT,
           "MAX_TRAN_NS": MAX_TRAN_NS, "OUT_NET": out_net, "OUT_RPT": f"{out_net}.rpt"}
    r = docker(["openroad", "-no_splash", "-exit", str(REPO_ROOT / "flows/synth/td_repair.tcl")], env, mem)
    Path(f"{out_net}.log").write_text(r.stdout + r.stderr)
    if r.returncode or not out_net.exists():
        raise RuntimeError(f"OpenROAD repair failed for {net}: {r.stdout[-1500:]}{r.stderr[-1500:]}")
    unsign(out_net)
    txt = r.stdout
    ins = [int(n) for n in re.findall(r"Inserted (\d+) (?:input|output)? ?buffers", txt)]
    viol = {k: int(m.group(1)) if (m := re.search(rf"Found (\d+) {k} violations", txt)) else 0
            for k in ("slew", "fanout", "cap")}
    return {"repair_inserted_buffers": sum(ins), **{f"repair_left_{k}": v for k, v in viol.items()}}


def stat(net: Path, top: str, out: Path, name: str) -> dict:
    """Yosys `stat -liberty` of the final (possibly OpenROAD-written) netlist: one area parser for all recipes."""
    ys = (f"read_liberty -lib {sweep.LIB}\nread_liberty -lib {SRAM_LIB}\nread_verilog {net}\nhierarchy -top {top}\n"
          f"tee -o {out / (name + '_final_stat.txt')} stat -top {top} -liberty {sweep.LIB} -liberty {SRAM_LIB}\n")
    (out / f"{name}_final_stat.ys").write_text(ys)
    subprocess.run([shutil.which("yosys"), "-q", "-s", str(out / f"{name}_final_stat.ys")], check=True,
                   capture_output=True)
    txt = (out / f"{name}_final_stat.txt").read_text()
    st = sweep.parse_stat(txt)
    st["logic_area_um2"] = st["total_area_um2"] - st["sram_instances"] * SRAM_AREA
    st["buf_cells"] = sum(int(n) for n in re.findall(r"^\s*(\d+)\s+\S+\s+sky130_fd_sc_hd__(?:buf|clkbuf)_\d+\s*$",
                                                     txt, re.M))
    st["x1_cells"] = sum(int(n) for n in re.findall(r"^\s*(\d+)\s+\S+\s+sky130_fd_sc_hd__\w+_[01]\s*$", txt, re.M))
    st["lpflow_cells"] = sum(int(n) for n in re.findall(r"^\s*(\d+)\s+\S+\s+sky130_fd_sc_hd__lpflow\w*\s*$", txt, re.M))
    return st


def sta(net: Path, top: str, out: Path, name: str, mem: str = "3g") -> dict:
    env = {"NETLIST": net, "TOP": top, "LIB": sweep.LIB, "SRAM_LIB": SRAM_LIB, "CLK_PERIOD_NS": PERIOD_NS,
           "SDC": SDC, "OUT": out / f"{name}_sta.txt", "OUT_R2R": out / f"{name}_sta_r2r.txt"}
    r = docker(["sta", "-no_splash", "-exit", "flows/synth/sta_macro.tcl"], env, mem)
    if r.returncode:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    txt, r2r = (out / f"{name}_sta.txt").read_text(), (out / f"{name}_sta_r2r.txt").read_text()
    wns = float(re.search(r"worst slack\s+(-?[\d.]+)", txt).group(1))
    r_wns = float(re.search(r"(-?[\d.]+)\s+slack \(", r2r).group(1))
    return {"sta_period_ns": PERIOD_NS, "wns_ns": wns, "fmax_any_mhz": 1e3 / (PERIOD_NS - wns),
            "r2r_wns_ns": r_wns, "r2r_min_period_ns": PERIOD_NS - r_wns, "r2r_fmax_mhz": 1e3 / (PERIOD_NS - r_wns),
            "r2r_arrival_ns": float(re.search(r"([\d.]+)\s+data arrival time", r2r).group(1)),
            "r2r_startpoint": re.search(r"Startpoint: (\S+)", r2r).group(1),
            "r2r_endpoint": re.search(r"Endpoint: (\S+)", r2r).group(1)}


def build_one(name: str, top: str, params: dict, out: Path, recipe: str, keepcol: bool, hier: bool,
              mem: str) -> tuple[dict, Path]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    net = yosys(name, top, params, out, recipe, keepcol, hier)
    row = {"recipe": recipe, "top": top, **params, "keepcol": keepcol, "hier": hier, "yosys_s": time.time() - t0}
    if recipe != "base":
        t1 = time.time()
        abc_net = out / f"{name}_abc_netlist.v"
        net.replace(abc_net)
        row.update(repair(abc_net, top, net, mem))
        row["repair_s"] = time.time() - t1
    row.update(stat(net, top, out, name))
    row.update(sta(net, top, out, name, mem))
    return row, net


def tile_netlist(recipe: str, design: str) -> Path:
    return BUILD / recipe / f"{design}_netlist.v"


def phase_synth(paper: bool) -> None:
    cfg = {"plan": [list(p) for p in PLAN_SYNTH], "period_ns": PERIOD_NS, "abc_D_ps": sweep.TD_ABC_D_PS,
           "abc_driving_cell": sweep.TD_DRIVING_CELL, "abc_load_ff": sweep.TD_LOAD_FF,
           "dont_use_td": sweep.TD_DONT_USE, "ol_no_synth": str(sweep.OL_NO_SYNTH.relative_to(REPO_ROOT)),
           "max_fanout": MAX_FANOUT, "max_tran_ns": MAX_TRAN_NS, "image": IMG, "sdc": str(SDC.relative_to(REPO_ROOT)),
           "repair_tcl": "flows/synth/td_repair.tcl", "sta_tcl": "flows/synth/sta_macro.tcl",
           "recipe_spec": "docs/spec/synth_recipe.md"}
    with start_run("e30_td_synth", cfg, paper=paper) as run:
        rows = []
        for recipe, d in PLAN_SYNTH:
            _, top, params, keepcol = TILES[d]
            out = BUILD / recipe
            row, net = build_one(d, top, params, out, recipe, keepcol, False, "3g")
            rows.append({"design": d, **row})
            for suf in ("stat.txt", "final_stat.txt", "sta.txt", "sta_r2r.txt", ".ys"):
                src = out / (f"{d}{suf}" if suf == ".ys" else f"{d}_{suf}")
                shutil.copy(src, run.path(f"{recipe}_{src.name}"))
            for suf in (".rpt", ".rpt.pre_drv", ".rpt.post_drv", ".log"):
                p = Path(f"{net}{suf}")
                if p.exists():
                    shutil.copy(p, run.path(f"{recipe}_{d}_repair{suf}"))
            run.save_csv("synth.csv", rows)
            print(f"{recipe:6s} {d:16s} logic {row['logic_area_um2']:9.0f} um2 cells {row['cells']:6d} "
                  f"buf {row['buf_cells']:5d} x1 {row['x1_cells']:6d}  r2r WNS@25 {row['r2r_wns_ns']:7.3f} "
                  f"({row['r2r_fmax_mhz']:.1f} MHz)  any WNS {row['wns_ns']:.3f}", flush=True)
        by = {(r["recipe"], r["design"]): r for r in rows}
        m = {}
        for (rc, d), r in by.items():
            m[f"{rc}_{d}_logic_area_um2"] = r["logic_area_um2"]
            m[f"{rc}_{d}_r2r_fmax_mhz"] = r["r2r_fmax_mhz"]
            m[f"{rc}_{d}_r2r_wns_ns"] = r["r2r_wns_ns"]
        run.log_metrics(m)


def phase_power(paper: bool, recipes: list[str], designs: list[str], windows: list[str],
                keepcol_recipes: tuple[str, ...] = ("td",)) -> None:
    import yaml

    inv = yaml.safe_load((REPO_ROOT / "flows/pdk/sram_macros.yaml").read_text())
    e_access_pj = inv["macros"][sweep.SRAM]["read_energy_pj_per_word_tt"]
    synth_rows = {(r["recipe"], r["design"]): r for r in
                  csv.DictReader((REPO_ROOT / "results/e30_td_synth/latest/synth.csv").open())}
    cfg = {"recipes": recipes, "designs": designs, "windows": windows, "keepcol_recipes": list(keepcol_recipes), "programs": str(PROGS.relative_to(REPO_ROOT)),
           "clk_ps": CLK_PS, "sram_access_pj": e_access_pj, "synth_run": str((REPO_ROOT / "results/e30_td_synth/latest")
                                                                              .resolve().name),
           "flow": "e13: Icarus zero-delay GLS (test_column_top), VCD sliced to the compute window and rebased (B9), "
                   "docker OpenSTA 2.6.0 report_power, OpenRAM Liberty power excluded, SRAM per-access charge"}
    with start_run("e30_td_power", cfg, paper=paper) as run:
        rows = []
        for w in windows:
            ppath = PROGS / f"{w}.npz"
            prog = load_program(ppath)
            for recipe in recipes:
                for d in designs:
                    if (recipe, d) not in synth_rows or (TILES[d][3] and recipe not in keepcol_recipes):
                        continue
                    top = TILES[d][1]
                    net = tile_netlist(recipe, d)
                    wd = BUILD / "power" / recipe / d / w
                    wd.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    g = e23.gls(net, top, ppath, wd)
                    sliced = wd / "window.vcd"
                    slice_vcd(g["vcd"], sliced, g["t_start_ns"], g["t_end_ns"], rebase=True)
                    g["vcd"].unlink()
                    env = {"NETLIST": net, "TOP": top, "LIB": sweep.LIB, "SRAM_LIB": SRAM_LIB, "SDC": SDC,
                           "CLK_PERIOD_NS": CLK_PS / 1e3, "VCD": sliced, "SCOPE": top, "OUT": wd / "power.rpt"}
                    for attempt in range(2):  # one retry: a transient docker failure (rc != 0, no output) seen once
                        (wd / "power.rpt").unlink(missing_ok=True)
                        r = docker(["sta", "-no_splash", "-exit", "flows/power/sta_power.tcl"], env, "4g")
                        if r.returncode == 0 and (wd / "power.rpt").exists():
                            break
                        print(f"STA attempt {attempt} rc={r.returncode}: {r.stdout[-500:]}{r.stderr[-500:]}", flush=True)
                    else:
                        raise RuntimeError(f"STA power failed rc={r.returncode} {r.stdout[-2000:]}{r.stderr[-2000:]}")
                    sliced.unlink()
                    pw = e13.parse_power((wd / "power.rpt").read_text())
                    grp = e23.groups((wd / "power.rpt").read_text())
                    dur = (g["t_end_ns"] - g["t_start_ns"]) * 1e-9
                    macs = g["n_ops"] * MACS_PER_OP
                    k = dur / macs * 1e15
                    e_logic = (pw["p_total_w"] - pw["p_sram_sta_w"]) * k
                    e_sram = e13.sram_accesses(prog, d) * e_access_pj * 1e3 / macs
                    row = {"recipe": recipe, "design": d, "window": w, "kind": w[:4], "n_ops": g["n_ops"],
                           "stall_cycles": g.get("stall_cycles", ""), "window_ns": dur * 1e9,
                           "cycles_per_op": dur * 1e12 / CLK_PS / g["n_ops"], **pw,
                           "e_logic_fj_per_mac": e_logic, "e_sram_fj_per_mac": e_sram,
                           "e_total_fj_per_mac": e_logic + e_sram,
                           "e_seq_fj_per_mac": grp["p_sequential_w"] * k, "e_comb_fj_per_mac": grp["p_combinational_w"] * k,
                           "e_clock_fj_per_mac": grp["p_clock_w"] * k, "e_leak_fj_per_mac": pw["p_leakage_w"] * k,
                           "wall_s": time.time() - t0}
                    rows.append(row)
                    run.save_csv("power.csv", rows)
                    shutil.copy(wd / "power.rpt", run.path(f"power_{recipe}_{d}_{w}.rpt"))
                    print(f"{w:6s} {recipe:6s} {d:16s} E/MAC {row['e_total_fj_per_mac']:8.1f} fJ (logic {e_logic:7.1f}, "
                          f"seq {row['e_seq_fj_per_mac']:.0f} comb {row['e_comb_fj_per_mac']:.0f}) "
                          f"cyc/op {row['cycles_per_op']:.3f} {row['wall_s']:.0f}s", flush=True)
        run.log_metrics(summarise(rows))
        print(json.dumps(run.metrics, indent=1))


def summarise(rows: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    m: dict = {}
    keys = sorted({(r["recipe"], r["design"]) for r in rows})
    for rc, d in keys:
        for kind in ("real", "rand"):
            for q in ("e_total_fj_per_mac", "e_logic_fj_per_mac"):
                v = [r[q] for r in rows if (r["recipe"], r["design"], r["kind"]) == (rc, d, kind)]
                if v:
                    m[f"{rc}_{d}_{q.replace('_fj_per_mac', '')}_{kind}"] = float(np.mean(v))
    real = sorted({r["window"] for r in rows if r["kind"] == "real"})

    def e(rc, d, w):
        return next((r["e_total_fj_per_mac"] for r in rows if (r["recipe"], r["design"], r["window"]) == (rc, d, w)),
                    None)

    for rc in sorted({k[0] for k in keys}):
        for base in ("bs_int5", "dpu"):
            pairs = [e(rc, base, w) / e(rc, "v2_sram_m16", w) for w in real
                     if e(rc, base, w) is not None and e(rc, "v2_sram_m16", w) is not None]
            if pairs:
                lo, hi = e13.bootstrap_ci(np.array(pairs), rng)
                m[f"{rc}_ee_v2_over_{base}"] = float(np.mean(pairs))
                m[f"{rc}_ee_v2_over_{base}_ci_lo"], m[f"{rc}_ee_v2_over_{base}_ci_hi"] = lo, hi
                p0 = [x for x, w in zip(pairs, real) if w == "real0"]
                if p0:
                    m[f"{rc}_ee_v2_over_{base}_real0"] = p0[0]
        for d in ("v2_sram_m16", "bs_int5", "dpu", "v2_keepcol_nvg0", "v2_keepcol_nvg1"):
            ratios = [e(rc, d, w) / e("base", d, w) for w in real if e(rc, d, w) and e("base", d, w)]
            if ratios and rc != "base":
                m[f"{rc}_over_base_{d}_energy"] = float(np.mean(ratios))
        g0 = [e(rc, "v2_keepcol_nvg0", w) / e(rc, "v2_sram_m16", w) for w in real
              if e(rc, "v2_keepcol_nvg0", w) and e(rc, "v2_sram_m16", w)]
        g1 = [e(rc, "v2_keepcol_nvg1", w) / e(rc, "v2_sram_m16", w) for w in real
              if e(rc, "v2_keepcol_nvg1", w) and e(rc, "v2_sram_m16", w)]
        if g0:
            m[f"{rc}_idle_nv_tax_nvg0"] = float(np.mean(g0)) - 1
        if g1:
            m[f"{rc}_idle_nv_tax_nvg1"] = float(np.mean(g1)) - 1
    return m


def phase_macro(design: str, paper: bool, recipe: str, hier: bool = False, flat: bool = False,
                nv_gate: int | None = None) -> None:
    top, params = sweep.DESIGNS[design]
    if nv_gate is not None:  # T4.2: NV_GATE macro netlist (T4.1 RTL, docs/spec/nv_gate.md); None = as-built (unchanged)
        if design in ("bs_macro", "bs4_macro"):
            raise SystemExit("--nv-gate: the bit-serial INT5 / INT4-SM macros have no NV path")
        params = {**params, "NV_GATE": nv_gate}
    hier = False if flat else (hier or MACRO_HIER.get(design, False))  # --flat: T2.11, larger VM
    nvs = "" if nv_gate is None else f"_nvg{nv_gate}"
    out = BUILD / (f"macro_{recipe}" + ("_hier" if hier else "") + nvs)
    cfg = {"design": design, "top": top, "params": params, "recipe": recipe, "hier_yosys": hier,
           "period_ns": PERIOD_NS, "abc_D_ps": sweep.TD_ABC_D_PS, "max_fanout": MAX_FANOUT, "max_tran_ns": MAX_TRAN_NS,
           "image": IMG, "recipe_spec": "docs/spec/synth_recipe.md",
           "note": "hierarchical Yosys keeps abc per module (each module gets the same -D target); OpenROAD links "
                   "(flattens) the netlist for repair and STA" if hier else "flat Yosys"}
    with start_run(f"e30_td_macro_{design}" + ("_hier" if hier else "") + nvs, cfg, paper=paper) as run:
        row, net = build_one(design, top, params, out, recipe, False, hier, "5g")
        row["total_mm2"] = row["total_area_um2"] / 1e6
        run.save_csv("synth.csv", [{"design": design, **row}])
        for p in out.glob(f"{design}_*"):
            if p.suffix in (".txt", ".ys", ".rpt", ".log", ".pre_drv", ".post_drv") and "full" not in p.name:
                shutil.copy(p, run.path(p.name))
        run.log_metrics({"design": design, **row})
        print({k: row[k] for k in ("logic_area_um2", "cells", "buf_cells", "r2r_wns_ns", "r2r_fmax_mhz", "wns_ns")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--macro", choices=["v2_macro", "bs_macro", "bs4_macro", "dpu_macro"])
    ap.add_argument("--recipe", default="td", help="macro recipe")
    ap.add_argument("--hier", action="store_true", help="hierarchical Yosys for the macro (always for dpu_macro)")
    ap.add_argument("--flat", action="store_true", help="T2.11: force flat Yosys, overriding MACRO_HIER")
    ap.add_argument("--nv-gate", type=int, choices=[0, 1], default=None,
                    help="T4.2: set the NV_GATE RTL param of the macro (v2_macro/dpu_macro); omitted = as-built")
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--designs", default=",".join(TILES))
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--keepcol-recipes", default="td", help="recipes whose keepcol NV_GATE tiles get power runs")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    if a.synth:
        phase_synth(a.paper)
    if a.power:
        phase_power(a.paper, a.recipes.split(","), a.designs.split(","), a.windows.split(","),
                    tuple(a.keepcol_recipes.split(",")))
    if a.macro:
        phase_macro(a.macro, a.paper, a.recipe, a.hier, a.flat, a.nv_gate)


if __name__ == "__main__":
    main()
