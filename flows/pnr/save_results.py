#!/usr/bin/env python3
"""Collect the v2 column P&R run (OpenLane2 t14) into results/e24_pnr_v2/<run>/ (metrics + key reports; DEF/GDS stay in build/)."""
import json, re, shutil, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cimfp4.utils.runlog import start_run  # noqa

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "v2_sram_m16"; TAG = sys.argv[2] if len(sys.argv) > 2 else "t14"
RUN = {"v2_sram_m16": "e24_pnr_v2"}.get(DESIGN, f"e24_pnr_{DESIGN}")
TOP = {"v2_sram_m16": "column_top", "bs_int5": "bs_column_top", "dpu": "dpu_column"}[DESIGN]
B = ROOT / f"build/pnr/{DESIGN}"; T = B / f"runs/{TAG}"; PR = B / "postroute"
def step(name):  # step numbers shift by one when the (failing) KLayout.DRC step is not run
    return next(d for d in sorted(T.iterdir()) if d.name.split("-", 1)[-1] == name)
cfg = json.loads((B / "config.json").read_text())
m = json.loads((T / "51-openroad-stapostpnr/state_out.json").read_text())["metrics"]
pre = json.loads((T / "08-openroad-staprepnr/state_out.json").read_text())["metrics"]
lvs = json.loads((step("netgen-lvs") / "state_out.json").read_text())["metrics"]
pp = json.loads((PR / "postroute_power.json").read_text()); var = json.loads((PR / "sta_variants.json").read_text())
drc_xml = (T / "drc_manual/drc.xml").read_text()
drc_n = len(re.findall(r"<item>", drc_xml))
P = cfg["CLOCK_PERIOD"]; ws = m["timing__setup__ws__corner:nom_tt_025C_1v80"]
e13 = {r["window"]: r for r in __import__("csv").DictReader(open(ROOT / "results/e13_power_rb/latest/power.csv")) if r["design"] == DESIGN}["real0"]
sdf = json.loads((ROOT / "results/e13_sdf_rb/latest/metrics.json").read_text())
SDF = sdf[f"{DESIGN}_sdf_over_zero"]
metrics = {
    "clock_period_ns_constraint": P,
    "die_area_um2": m["design__die__area"], "core_area_um2": m["design__core__area"],
    "instance_area_um2": m["design__instance__area"], "stdcell_area_um2": m["design__instance__area__stdcell"],
    "macro_area_um2": m["design__instance__area__macros"],
    "utilization_total": m["design__instance__utilization"], "utilization_stdcell_of_core": m["design__instance__utilization__stdcell"],
    "stdcell_count": m["design__instance__count__stdcell"],
    "setup_ws_ns_nom_tt": ws, "hold_ws_ns_nom_tt": m["timing__hold__ws__corner:nom_tt_025C_1v80"],
    "hold_reg2reg_ws_ns_nom_tt": m["timing__hold_r2r__ws__corner:nom_tt_025C_1v80"],
    "hold_violations_nom_tt": m["timing__hold_vio__count__corner:nom_tt_025C_1v80"],
    "hold_reg2reg_violations_nom_tt": m["timing__hold_r2r_vio__count__corner:nom_tt_025C_1v80"],
    "setup_violations_nom_tt": m["timing__setup_vio__count__corner:nom_tt_025C_1v80"],
    "fmax_mhz_postroute_nom_tt": 1e3 / (P - ws),
    "fmax_mhz_prepnr_ideal_clk_ol_netlist": 1e3 / (P - pre["timing__setup__ws__corner:nom_tt_025C_1v80"]),
    "fmax_mhz_e12_zero_wire_yosys": json.loads((ROOT / "results/e12_synth/latest/metrics.json").read_text())[f"{DESIGN}_fmax_mhz"],
    "setup_ws_ns_all_corners": {k.split("__corner:")[1]: v for k, v in m.items() if k.startswith("timing__setup__ws__corner")},
    "hold_ws_ns_all_corners": {k.split("__corner:")[1]: v for k, v in m.items() if k.startswith("timing__hold__ws__corner")},
    "drc_klayout_violations": drc_n, "route_drc_errors_openroad": m["route__drc_errors"],
    "lvs_error_count": lvs["design__lvs_error__count"], "lvs_device_difference": lvs["design__lvs_device_difference__count"],
    "lvs_net_difference": lvs["design__lvs_net_difference__count"], "lvs_unmatched_pins": lvs["design__lvs_unmatched_pin__count"],
    "antenna_violations_route": m["route__antenna_violation__count"],
    "postroute_e_logic_fj_per_mac": pp["e_logic_fj_per_mac"], "postroute_e_total_fj_per_mac": pp["e_total_fj_per_mac"],
    "postroute_tflops_per_w": 2 / (pp["e_total_fj_per_mac"] * 1e-15) / 1e12,
    "sta_variants_e_logic_fj_per_mac": {k: v["e_logic_fj_per_mac"] for k, v in var.items()},
    "e13_zero_delay_real0_e_logic_fj_per_mac": float(e13["e_logic_fj_per_mac"]), "e13_zero_delay_real0_e_total_fj_per_mac": float(e13["e_total_fj_per_mac"]),
    "e13_sdf_over_zero": SDF,
    "e13_sdf_est_real0_e_total_fj_per_mac": float(e13["e_logic_fj_per_mac"]) * SDF + float(e13["e_sram_fj_per_mac"]),
    "postroute_over_e13_zero_total": pp["e_total_fj_per_mac"] / float(e13["e_total_fj_per_mac"]),
    "vcd_annotated_pins": var["c_postroute_spef"]["annot"],
}
with start_run(RUN, {"design": f"{DESIGN} ({TOP}, 4 cols, M=16, SRAM)", "flow": "OpenLane2 2.3.10 Classic, sky130_fd_sc_hd, tt/nom RCX",
        "window": "e13_power_rb real0, rebased VCD, 40 MHz", "clk_ns": P, "openlane_config": cfg}) as run:
    run.log_metrics(metrics)
    d = run.path("reports"); d.mkdir(exist_ok=True)
    for src, nm in [(T / "51-openroad-stapostpnr/summary.rpt", "sta_summary.rpt"), (T / "08-openroad-staprepnr/summary.rpt", "sta_prepnr_summary.rpt"),
                    (step("netgen-lvs") / "reports/lvs.netgen.json", "lvs.netgen.json"), (T / "drc_manual/drc.xml", "drc_klayout.xml"),
                    (PR / "power.rpt", "power_postroute_spef.rpt"), (PR / "power_a_synth_nospef.rpt", "power_a_synth_nospef.rpt"),
                    (PR / "power_b_postroute_nospef.rpt", "power_b_postroute_nospef.rpt"), (PR / "sta_variants.json", "sta_variants.json"),
                    (PR / "postroute_power.json", "postroute_power.json"), (PR / "window.json", "window.json"),
                    (T / "resolved.json", "openlane_resolved.json"), (B / "config.json", "openlane_config.json"),
                    (T / "50-openroad-rcx/rcx.log" if False else T / "50-openroad-rcx/nom/rcx.log", "rcx.log"),
                    (T / "51-openroad-stapostpnr/nom_tt_025C_1v80/clock.rpt", "clock_nom_tt.rpt"),
                    (ROOT / "build/pnr/pdkroot/sky130A/libs.tech/openlane/config.tcl", "pdk_shim_config.tcl")]:
        shutil.copy(src, d / nm)
    mx = (T / "51-openroad-stapostpnr/nom_tt_025C_1v80/max.rpt").read_text().splitlines()[:260]
    (d / "worst_setup_path_nom_tt.rpt").write_text("\n".join(mx))
    (d / "lvs.netgen.rpt.head").write_text("\n".join((step("netgen-lvs") / "reports/lvs.netgen.rpt").read_text().splitlines()[:200]))
print(json.dumps(metrics, indent=1, default=str))
