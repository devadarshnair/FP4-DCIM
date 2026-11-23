#!/usr/bin/env python3
"""Collect the T3.1 signoff P&R runs (build/pnr/t31/<design>/runs/<tag>/final) into results/t31_signoff_pnr/<run>/.

Usage: python flows/pnr/save_t31.py v2_sram_m16=s2 bs_int5=s2 dpu=s2   (or a single tag for all)   (DEF/GDS/SPEF stay in build/; paths recorded)
       [--exp t32_pushed_clock] [--paper]   (T3.2: same collector on the pushed-clock runs, fmax = 1/(T - WS) at nom_tt)
"""
import json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cimfp4.utils.runlog import start_run  # noqa

DES = {"v2_sram_m16": "e24_pnr_v2", "bs_int5": "e24_pnr_bs_int5", "dpu": "e24_pnr_dpu"}
C = "nom_tt_025C_1v80"
_args = sys.argv[1:]
PAPER = "--paper" in _args
EXP = _args[_args.index("--exp") + 1] if "--exp" in _args else "t31_signoff_pnr"
_a = [x for i, x in enumerate(_args) if x not in ("--paper", "--exp") and (i == 0 or _args[i - 1] != "--exp")] or ["s2"]
TAGS = dict(x.split("=") for x in _a) if "=" in _a[0] else {d: _a[0] for d in DES}
K = {"die_area_um2": "design__die__area", "core_area_um2": "design__core__area", "instance_area_um2": "design__instance__area",
     "stdcell_area_um2": "design__instance__area__stdcell", "macro_area_um2": "design__instance__area__macros",
     "stdcell_count": "design__instance__count__stdcell", "setup_ws_ns_nom_tt": f"timing__setup__ws__corner:{C}",
     "hold_ws_ns_nom_tt": f"timing__hold__ws__corner:{C}", "hold_violations_nom_tt": f"timing__hold_vio__count__corner:{C}",
     "hold_violations_all_corners": "timing__hold_vio__count", "setup_violations_nom_tt": f"timing__setup_vio__count__corner:{C}",
     "klayout_drc_errors": "klayout__drc_error__count", "route_drc_errors": "route__drc_errors",
     "lvs_errors": "design__lvs_error__count", "lvs_device_difference": "design__lvs_device_difference__count",
     "lvs_net_difference": "design__lvs_net_difference__count", "lvs_unmatched_pins": "design__lvs_unmatched_pin__count",
     "antenna_violating_nets": "antenna__violating__nets", "antenna_violating_pins": "antenna__violating__pins",
     "max_cap_violations_nom_tt": f"design__max_cap_violation__count__corner:{C}",
     "max_slew_violations_nom_tt": f"design__max_slew_violation__count__corner:{C}",
     "disconnected_pins": "design__disconnected_pin__count", "power_total_w_nom_tt_sdc_activity": "power__total"}
out, cfgs = {}, {}
for d, old in DES.items():
    T = ROOT / f"build/pnr/t31/{d}/runs/{TAGS[d]}"; F = T / "final"
    m = json.loads((F / "metrics.json").read_text())
    r = {k: m.get(v) for k, v in K.items()}
    rc = json.loads((T / "resolved.json").read_text())  # the run's own resolved config (build/.../config.json may be newer/older)
    r["clock_period_ns"] = float(rc["CLOCK_PERIOD"])
    r["die_area_cfg_um"] = rc["DIE_AREA"]
    r["fmax_mhz_nom_tt_at_constraint"] = 1e3 / (r["clock_period_ns"] - r["setup_ws_ns_nom_tt"])
    r["setup_ws_ns_all_corners"] = {k.split("corner:")[1]: v for k, v in m.items() if k.startswith("timing__setup__ws__corner")}
    r["hold_ws_ns_all_corners"] = {k.split("corner:")[1]: v for k, v in m.items() if k.startswith("timing__hold__ws__corner")}
    r["final_dir"] = str(F)
    k4 = ROOT / f"results/{old}/latest/metrics.json"
    if k4.exists():
        o = json.loads(k4.read_text())
        r["vs_e24_k416"] = {"stdcell_area_ratio": r["stdcell_area_um2"] / o["stdcell_area_um2"],
                            "stdcell_count_ratio": r["stdcell_count"] / o["stdcell_count"],
                            "setup_ws_ns_e24": o["setup_ws_ns_nom_tt"], "die_area_um2_e24": o["die_area_um2"]}
    out[d] = r
    cfgs[d] = {k: rc.get(k) for k in ("DIE_AREA", "CLOCK_PERIOD", "PL_TARGET_DENSITY_PCT", "PL_RESIZER_HOLD_SLACK_MARGIN",
                                       "GRT_RESIZER_HOLD_SLACK_MARGIN", "MACRO_PLACEMENT_CFG", "MACROS")}
with start_run(EXP, {"tags": TAGS, "flow": "OpenLane2 2.3.10 Classic via flows/pnr/run_t31.sh, sky130_fd_sc_hd",
                                   "die_um": "per design: openlane_configs[*].DIE_AREA (run resolved.json)", "macro_at_um": [60, 60],
                                   "clk_ns": {d: c["CLOCK_PERIOD"] for d, c in cfgs.items()}, "hold_margin_ns": 0.3,
                                   "fmax_recipe": "1e3 / (CLOCK_PERIOD - timing__setup__ws__corner:nom_tt_025C_1v80), every design",
                                   "openlane_configs": cfgs}, paper=PAPER) as run:
    run.log_metrics(out)
    rep = run.path("reports"); rep.mkdir(exist_ok=True)
    for d in DES:
        T = ROOT / f"build/pnr/t31/{d}/runs/{TAGS[d]}"
        st = lambda n: next(p for p in sorted(T.iterdir()) if p.name.split("-", 1)[-1] == n)
        for src, nm in [(st("netgen-lvs") / "reports/lvs.netgen.rpt", "lvs.netgen.rpt"),
                        (st("klayout-drc") / "reports/drc_violations.klayout.json", "drc_klayout.json"),
                        (st("openroad-stapostpnr") / "summary.rpt", "sta_summary.rpt"),
                        (st("misc-reportmanufacturability") / "manufacturability.rpt", "manufacturability.rpt"),
                        (T / "resolved.json", "openlane_resolved.json")]:
            if src.exists():
                (rep / d).mkdir(exist_ok=True); shutil.copy(src, rep / d / nm)
        a = sorted(T.glob("*-openroad-checkantennas-1/reports/antenna_summary.rpt"))
        if a: shutil.copy(a[-1], rep / d / "antenna_summary.rpt")
print(json.dumps(out, indent=1, default=str))
