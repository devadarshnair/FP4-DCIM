#!/usr/bin/env python3
"""sky130 synthesis sweep (WP-27): Yosys -> mapped netlist + area/cells/NAND2-eq; OpenSTA -> critical path.

Designs (all at the 4-column group level, same storage interface; docs/spec/numerics.md):
    v2_sram_m16    column_top  STORE=SRAM  M=16 LEAF=FP4     (main design)
    v2_latch_m16   column_top  STORE=LATCH M=16 LEAF=FP4     (storage ablation)
    v2_latch_m1    column_top  STORE=LATCH M=1  LEAF=FP4
    v2_generic     column_top  STORE=SRAM  M=16 LEAF=GENERIC (leaf ablation)
    bs_int5        bs_column_top STORE=SRAM M=16             (baseline a)
    dpu            dpu_column  M=16                          (baseline c)
The OpenRAM macro is a black box: its area comes from its Liberty (`stat -liberty`), timing excluded
from the logic critical path.

Usage: python flows/synth/sweep.py [--designs v2_sram_m16,...] [--out build/synth] [--sta]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PDK = Path.home() / ".ciel" / "sky130A"
LIB = PDK / "libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
SRAM_DIR = PDK / "libs.ref/sky130_sram_macros"
SRAM = "sky130_sram_1kbyte_1rw1r_32x256_8"
# SYNTH_HIER=1: hierarchical synthesis (no -flatten; abc per module) for designs whose flat Yosys run exceeds the
# 7.5 GB machine (dpu_macro). Default (unset) = the authoritative flat flow.
HIER = bool(os.environ.get("SYNTH_HIER"))
NAND2_AREA = 3.7536  # sky130_fd_sc_hd__nand2_1, tt_025C_1v80 (Liberty)
# Synthesis recipe (docs/spec/synth_recipe.md, T2.6). "base" = the e12/e13 recipe (abc with no delay target,
# no buffering; the netlists behind all pre-T2.6 energy numbers) and stays the default so old drivers reproduce.
# "td" = the T2.6 timing-driven recipe (stage 1 here: dont-use set + abc -D + buffer/upsize/dnsize via -constr;
# stage 2 = OpenROAD repair, flows/synth/td_repair.tcl, run by the caller). "td_ol" = td with OpenLane2's
# sky130_fd_sc_hd no_synth.cells added to the dont-use set (no x1 drive strengths): recipe-sensitivity axis that
# reproduces the cell-set choice of the e24 OpenLane2 synthesis.
RECIPE = os.environ.get("SYNTH_RECIPE", "base")
TD_PERIOD_NS = 25.0  # 40 MHz
TD_ABC_D_PS = 20000  # 0.8 x period: 20 % margin for wires, clock skew/uncertainty and setup
TD_DRIVING_CELL = "sky130_fd_sc_hd__inv_2"  # abc -constr: drive of every combinational input (flop Q / port)
TD_LOAD_FF = 5.0  # abc -constr: load on every combinational output (flop D ~1.7-2 fF; column.sdc port load 5 fF)
TD_DONT_USE = ["sky130_fd_sc_hd__lpflow_*", "sky130_fd_sc_hd__clk*", "sky130_fd_sc_hd__dly*", "sky130_fd_sc_hd__probe*"]
OL_NO_SYNTH = ROOT / "flows/synth/constraints/ol_no_synth.cells"


def dont_use(recipe: str = None) -> list[str]:
    recipe = recipe or RECIPE
    if recipe == "base":
        return []
    cells = list(TD_DONT_USE)
    if recipe == "td_ol":
        cells += [c.strip() for c in OL_NO_SYNTH.read_text().split() if c.strip()]
    return cells

SOURCES = [
    "rtl/gen/col_tree_oc_w9_r32.sv",
    "rtl/gen/col_tree_unsigned_w1_r32.sv",
    "rtl/src/cim_icg.sv",
    "rtl/src/row_driver.sv",
    "rtl/src/leaf_mult.sv",
    "rtl/baselines/digital_mac/generic_leaf.sv",
    "rtl/src/column.sv",
    "rtl/src/wset_latch.sv",
    "rtl/src/wset_sram.sv",
    "rtl/src/column_top.sv",
    "rtl/baselines/bitserial_int5/bs_column.sv",
    "rtl/baselines/bitserial_int5/bs_column_top.sv",
    "rtl/baselines/bitserial_int4sm/bs4_column.sv",
    "rtl/baselines/bitserial_int4sm/bs4_column_top.sv",
    "rtl/baselines/digital_mac/dpu_column.sv",
    "rtl/src/macro_top.sv",
    "rtl/baselines/bitserial_int5/bs_macro_top.sv",
    "rtl/baselines/bitserial_int4sm/bs4_macro_top.sv",
    "rtl/baselines/digital_mac/dpu_macro.sv",
]

DESIGNS = {
    "v2_sram_m16": ("column_top", {"STORE": 0, "M": 16, "LEAF": 0}),
    "v2_latch_m16": ("column_top", {"STORE": 1, "M": 16, "LEAF": 0}),
    "v2_latch_m1": ("column_top", {"STORE": 1, "M": 1, "LEAF": 0}),
    "v2_generic": ("column_top", {"STORE": 0, "M": 16, "LEAF": 1}),
    "bs_int5": ("bs_column_top", {"STORE": 0, "M": 16}),
    "bs4_int4sm": ("bs4_column_top", {"STORE": 0, "M": 16}),
    "dpu": ("dpu_column", {"M": 16}),
    "v2_macro": ("macro_top", {"M": 16, "STORE": 0}),
    "bs_macro": ("bs_macro_top", {"M": 16, "STORE": 0}),
    "bs4_macro": ("bs4_macro_top", {"M": 16, "STORE": 0}),  # T5.4: D2 INT4-SM macro (lossy)
    "dpu_macro": ("dpu_macro", {"M": 16, "STORE": 0}),
}


def yosys_script(name: str, top: str, params: dict, out: Path) -> str:
    chparams = " ".join(f"-chparam {k} {v}" for k, v in params.items())
    srcs = " ".join(str(ROOT / s) for s in SOURCES)
    sram_lib = SRAM_DIR / "lib" / (SRAM + "_TT_1p8V_25C.lib")
    du = "".join(f" -dont_use {c}" for c in dont_use())
    if RECIPE == "base":
        abc_cmd = f"abc -liberty {LIB}"
    else:
        constr = out / f"{name}_abc.constr"
        constr.write_text(f"set_driving_cell {TD_DRIVING_CELL}\nset_load {TD_LOAD_FF}\n")
        abc_cmd = f"abc -D {TD_ABC_D_PS} -constr {constr}{du} -liberty {LIB}"
    return f"""
read_liberty -lib {LIB}
read_liberty -lib {SRAM_DIR / "lib" / (SRAM + "_TT_1p8V_25C.lib")}
read_verilog -sv -I{ROOT / "rtl/include"} {srcs}
hierarchy -top {top} {chparams}
{"synth -top " + top if HIER else "synth -flatten -top " + top}
dfflibmap{du} -liberty {LIB}
{abc_cmd}
opt_clean -purge
hilomap -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO
tee -o {out / (name + "_stat.txt")} stat {"-top " + top if HIER else ""} -liberty {LIB} -liberty {sram_lib}
write_verilog -noattr -noexpr {out / (name + "_netlist.v")}
"""


def parse_stat(text: str) -> dict:
    if "=== design hierarchy ===" in text:  # hierarchical run: totals are in the hierarchy section
        text = text.split("=== design hierarchy ===", 1)[1]
    cells = int(re.search(r"^\s*(\d+)\s+\S+\s+cells\s*$", text, re.M).group(1))
    total = float(re.findall(r"Chip area for (?:top )?module.*?:\s+([\d.]+)", text)[-1])
    seq_m = re.search(r"used for sequential elements:\s+([\d.]+)", text)
    sram_cnt = sum(int(n) for n in re.findall(rf"^\s*(\d+)\s+\S+\s+{SRAM}\s*$", text, re.M))
    seq = sum(
        int(n) for n in re.findall(r"^\s*(\d+)\s+\S+\s+sky130_fd_sc_hd__(?:df|dl|sdf|edf)\w*\s*$", text, re.M)
    )
    return {
        "cells": cells,
        "total_area_um2": total,
        "seq_area_um2": float(seq_m.group(1)) if seq_m else 0.0,
        "sram_instances": sram_cnt,
        "seq_cells": seq,
    }


def run(names: list[str], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    yosys = shutil.which("yosys")
    rows = []
    for name in names:
        top, params = DESIGNS[name]
        script = out / f"{name}.ys"
        script.write_text(yosys_script(name, top, params, out))
        log = out / f"{name}_yosys.log"
        with log.open("w") as f:
            subprocess.run([yosys, "-q", "-l", str(out / f"{name}_yosys_full.log"), "-s", str(script)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        net = out / f"{name}_netlist.v"  # OpenSTA's reader rejects `signed` (no meaning in a mapped netlist)
        net.write_text(re.sub(r"\b(input|output|wire|reg)\s+signed\b", r"\1", net.read_text()))
        stat = (out / f"{name}_stat.txt").read_text()
        r = {"design": name, "top": top, **params, **parse_stat(stat)}
        inv = yaml.safe_load((ROOT / "flows/pdk/sram_macros.yaml").read_text())
        r["sram_area_um2"] = r["sram_instances"] * inv["macros"][SRAM]["corners"]["TT_1p8V_25C"]["area_um2"]
        r["logic_area_um2"] = r["total_area_um2"] - r["sram_area_um2"]
        r["logic_nand2_eq"] = r["logic_area_um2"] / NAND2_AREA
        rows.append(r)
        print(
            f"{name:14s} cells {r['cells']:7d}  logic {r['logic_area_um2'] / 1e6:7.4f} mm2 "
            f"({r['logic_nand2_eq']:9.0f} NAND2-eq)  sram {r['sram_area_um2'] / 1e6:6.3f} mm2  "
            f"seq {r['seq_cells']}"
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "synth")
    a = ap.parse_args(argv)
    rows = run(a.designs.split(","), a.out)
    (a.out / "synth.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
