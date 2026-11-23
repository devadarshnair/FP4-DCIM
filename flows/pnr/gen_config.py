#!/usr/bin/env python3
"""Write an OpenLane2 config (sky130A / sky130_fd_sc_hd, tt) for a 4-column group design with the OpenRAM macro.

Usage: python flows/pnr/gen_config.py <design> <out.json> [--clk-ns 25] [--die W H]
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDK = Path.home() / ".ciel/sky130A"
SRAM = "sky130_sram_1kbyte_1rw1r_32x256_8"
SM = PDK / "libs.ref/sky130_sram_macros"
# (top, source files) -- default module parameters equal the e12/e13 configuration (M=16, STORE=SRAM, LEAF=FP4)
COMMON = ["rtl/gen/col_tree_oc_w9_r32.sv", "rtl/gen/col_tree_unsigned_w1_r32.sv", "rtl/src/cim_icg.sv",
          "rtl/src/row_driver.sv", "rtl/src/leaf_mult.sv", "rtl/baselines/digital_mac/generic_leaf.sv",
          "rtl/src/column.sv", "rtl/src/wset_latch.sv", "rtl/src/wset_sram.sv"]
DESIGNS = {
    "v2_sram_m16": ("column_top", COMMON + ["rtl/src/column_top.sv"]),
    "bs_int5": ("bs_column_top", COMMON + ["rtl/baselines/bitserial_int5/bs_column.sv",
                                           "rtl/baselines/bitserial_int5/bs_column_top.sv"]),
    "dpu": ("dpu_column", COMMON + ["rtl/src/column_top.sv", "rtl/baselines/digital_mac/dpu_column.sv"]),
}

ap = argparse.ArgumentParser()
ap.add_argument("design"); ap.add_argument("out")
ap.add_argument("--clk-ns", type=float, default=25.0)
ap.add_argument("--drc", action="store_true"); ap.add_argument("--lvs", action="store_true")
ap.add_argument("--die", type=float, nargs=2, default=[900.0, 700.0])
# T3.1 signoff options (defaults reproduce the e24 recipe)
ap.add_argument("--macro-at", type=float, nargs=2, default=[30.0, 30.0],
                help="SRAM macro origin (um). e24 used 30,30, which leaves <15 um row slivers left of the macro that get "
                     "no tap cell -> floating N-wells -> Netgen LVS mismatch (T3.1 diagnosis)")
ap.add_argument("--hold-margin", type=float, default=None, help="GRT (post-global-route) resizer hold slack margin (ns)")
ap.add_argument("--pl-hold-margin", type=float, default=None, help="PL (post-CTS) resizer hold slack margin (ns); default = --hold-margin")
ap.add_argument("--density", type=float, default=55.0, help="PL_TARGET_DENSITY_PCT")
ap.add_argument("--signoff", action="store_true", help="KLayout DRC via flows/pnr/stubs/sky130A_mr_wrap.drc, 2 threads")
a = ap.parse_args()
top, srcs = DESIGNS[a.design]
W, H = a.die
inst = {"v2_sram_m16": "g_sram.u_store.u_sram", "bs_int5": "g_sram.u_store.u_sram", "dpu": "u_lanes.g_sram.u_store.u_sram"}[a.design]
cfg = {
    "DESIGN_NAME": top,
    "VERILOG_FILES": [str(ROOT / s) for s in srcs] + [str(ROOT / "flows/pnr/stubs/sky130_icg_blackbox.v")],
    "VERILOG_INCLUDE_DIRS": [str(ROOT / "rtl/include")],
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": a.clk_ns,
    "PNR_SDC_FILE": str(ROOT / "flows/pnr/constraints/column_pnr.sdc"),
    "SIGNOFF_SDC_FILE": str(ROOT / "flows/pnr/constraints/column_pnr.sdc"),
    "FP_SIZING": "absolute",
    "DIE_AREA": [0, 0, W, H],
    "PL_TARGET_DENSITY_PCT": a.density,
    "TECH_LEFS": {"nom_*": str(PDK / "libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef"),
                 "min_*": str(PDK / "libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__min.tlef"),
                 "max_*": str(PDK / "libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__max.tlef")},
    "RUN_LINTER": False,
    "RUN_KLAYOUT_DRC": a.drc, "RUN_MAGIC_DRC": False, "RUN_LVS": a.lvs, "RUN_KLAYOUT_XOR": False,
    "RUN_ANTENNA_REPAIR": True,
    "RUN_CTS": True, "RUN_POST_GPL_DESIGN_REPAIR": True, "RUN_POST_CTS_RESIZER_TIMING": True,
    "RUN_POST_GRT_RESIZER_TIMING": True,
    "RT_MAX_LAYER": "met4",
    "MACROS": {
        SRAM: {
            "gds": [str(SM / f"gds/{SRAM}.gds")], "lef": [str(SM / f"lef/{SRAM}.lef")],
            "lib": {"*": str(SM / f"lib/{SRAM}_TT_1p8V_25C.lib")}, "spice": [str(SM / f"spice/{SRAM}.spice")],
            "instances": {inst: {"location": list(a.macro_at), "orientation": "N"}} if inst else {},
        }
    },
    "PDN_MACRO_CONNECTIONS": [f"{inst} VPWR VGND vccd1 vssd1"] if inst else [],
    "FP_PDN_VOFFSET": 5, "FP_PDN_HOFFSET": 5,
}
if a.hold_margin is not None:
    cfg["GRT_RESIZER_HOLD_SLACK_MARGIN"] = a.hold_margin
    cfg["PL_RESIZER_HOLD_SLACK_MARGIN"] = a.hold_margin if a.pl_hold_margin is None else a.pl_hold_margin
if a.signoff:
    cfg["KLAYOUT_DRC_RUNSET"] = str(ROOT / "flows/pnr/stubs/sky130A_mr_wrap.drc"); cfg["KLAYOUT_DRC_THREADS"] = 2
    cfg["HOLD_VIOLATION_CORNERS"] = ["nom_tt_025C_1v80"]  # T3.1 acceptance corner; other corners reported, not gated
Path(a.out).write_text(json.dumps(cfg, indent=2))
