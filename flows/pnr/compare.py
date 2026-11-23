#!/usr/bin/env python3
"""e24_compare: post-route like-for-like ratios v2 vs INT5 bit-serial vs DPU at the 4-column tile boundary (MX-only folded).
NOT macro level. Not signoff (LVS not clean; CTS/repair buffers unannotated in the VCD). Numbers only; no G1 re-verdict."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from cimfp4.utils.runlog import start_run  # noqa
L = lambda n: json.loads((ROOT / f"results/e24_pnr_{n}/latest/metrics.json").read_text())
cfg = lambda n: __import__("yaml").safe_load((ROOT / f"results/e24_pnr_{n}/latest/config.yaml").read_text())["openlane_config"]["DIE_AREA"]
D = {"v2": L("v2"), "int5": L("bs_int5"), "dpu": L("dpu")}
E = {k: v["postroute_e_total_fj_per_mac"] for k, v in D.items()}
G = lambda k: D[k]["sta_variants_e_logic_fj_per_mac"]
sram = {k: D[k]["postroute_e_total_fj_per_mac"] - D[k]["postroute_e_logic_fj_per_mac"] for k in D}
e15 = json.loads((ROOT / "results/e15_g1_relative_rb/latest/g1.json").read_text())
tab = {r["baseline"] + "_" + r["delay_model"]: r["ee_mean"] for r in e15["table"] if r["point"] == "R256_M16_SRAM" and r["v2"] == "v2_sram_m16"}
m = {"scope": "tile boundary (4 columns, MX-only folded, M=16, OpenRAM macro), post-route OL2 sky130 tt; NOT macro",
     "per_design": {k: {"die_um2": v["die_area_um2"], "die_wxh": cfg({"v2": "v2", "int5": "bs_int5", "dpu": "dpu"}[k]),
                        "core_um2": v["core_area_um2"], "stdcell_um2": v["stdcell_area_um2"], "macro_um2": v["macro_area_um2"],
                        "instance_um2": v["instance_area_um2"], "util_total": v["utilization_total"],
                        "util_stdcell_of_core": v["utilization_stdcell_of_core"], "fj_per_mac_total": E[k],
                        "fj_per_mac_logic": v["postroute_e_logic_fj_per_mac"], "setup_ws_tt_ns": v["setup_ws_ns_nom_tt"],
                        "fmax_lb_mhz": v["fmax_mhz_postroute_nom_tt"], "setup_ws_ss_ns": v["setup_ws_ns_all_corners"]["nom_ss_100C_1v60"],
                        "hold_ws_tt_ns": v["hold_ws_ns_nom_tt"], "hold_r2r_ws_tt_ns": v["hold_reg2reg_ws_ns_nom_tt"],
                        "lvs_errors": v["lvs_error_count"], "drc": v["drc_klayout_violations"], "antenna": v["antenna_violations_route"]}
                    for k, v in D.items()}}
r = {}
for b in ("int5", "dpu"):
    r[f"EE_v2_over_{b}_postroute_spef_total"] = E[b] / E["v2"]
    for var in ("a_synth_nospef", "b_postroute_nospef", "c_postroute_spef"):
        r[f"EE_v2_over_{b}_logic_{var}"] = G(b)[var] / G("v2")[var]
    r[f"EE_v2_over_{b}_e13_zero_real0_total"] = D[b]["e13_zero_delay_real0_e_total_fj_per_mac"] / D["v2"]["e13_zero_delay_real0_e_total_fj_per_mac"]
    r[f"EE_v2_over_{b}_e13_sdf_est_real0_total"] = D[b]["e13_sdf_est_real0_e_total_fj_per_mac"] / D["v2"]["e13_sdf_est_real0_e_total_fj_per_mac"]
    r[f"postroute_over_zero_total_{b}"] = D[b]["postroute_over_e13_zero_total"]
    r[f"area_{b}_over_v2_instance"] = D[b]["instance_area_um2"] / D["v2"]["instance_area_um2"]
    r[f"area_{b}_over_v2_stdcell"] = D[b]["stdcell_area_um2"] / D["v2"]["stdcell_area_um2"]
    r[f"area_{b}_over_v2_die"] = D[b]["die_area_um2"] / D["v2"]["die_area_um2"]
    r[f"fmax_v2_over_{b}"] = D["v2"]["fmax_mhz_postroute_nom_tt"] / D[b]["fmax_mhz_postroute_nom_tt"]
    # TFLOPS/mm2 at each design's own fmax lower bound is not computed (cycle counts differ: INT5 is bit-serial); use e15 for that.
    r[f"e15_g1_zero_window_mean_v2_over_{b}"] = tab[("bs_int5" if b == "int5" else "dpu") + "_zero"]
    r[f"e15_g1_sdf_window_mean_v2_over_{b}"] = tab[("bs_int5" if b == "int5" else "dpu") + "_sdf_corrected"]
# sensitivity: OpenSTA "Clock" power group (clock-tree nets/buffers; activity comes from the SDC clock, not the VCD) of the SPEF run
import re
def pw(n, f="power_postroute_spef.rpt"):
    t = (ROOT / f"results/e24_pnr_{n}/latest/reports/{f}").read_text()
    g = lambda k: float(re.search(rf"^{k}\s+\S+\s+\S+\s+\S+\s+(\S+)", t, re.M).group(1))
    return g("Total") - g("Macro"), g("Clock")
nc = {}
for k, n in (("v2", "v2"), ("int5", "bs_int5"), ("dpu", "dpu")):
    lg, ck = pw(n); nc[k] = {"logic_w": lg, "clock_group_w": ck, "clock_frac_of_logic": ck / lg,
                              "e_logic_excl_clock_fj_per_mac": D[k]["postroute_e_logic_fj_per_mac"] * (1 - ck / lg)}
m["clock_group_sensitivity"] = nc
for b in ("int5", "dpu"):
    r[f"EE_v2_over_{b}_postroute_logic_excl_clock_group"] = nc[b]["e_logic_excl_clock_fj_per_mac"] / nc["v2"]["e_logic_excl_clock_fj_per_mac"]
m["ratios"] = r
with start_run("e24_compare", {"inputs": ["e24_pnr_v2", "e24_pnr_bs_int5", "e24_pnr_dpu", "e15_g1_relative_rb"],
                               "caveats": "tile boundary not macro; not signoff; CTS buffers unannotated; DPU die 1000x900 vs 900x700; no G1 re-evaluation"}) as run:
    run.log_metrics(m)
print(json.dumps(m, indent=1))
