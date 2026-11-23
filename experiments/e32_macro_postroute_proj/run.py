#!/usr/bin/env python3
"""T3.4: macro-level post-route projection [EST] -- tile-measured factors transferred to the macro.

Post-processing only. Re-evaluates the e22 energy model (experiments/e21_macro_power/compare.py `op_energy`) on the e21 /
td-synthesis / e13_sdf_td runs recorded in the pinned e22 run (same inputs as t210_sram_gating), and scales ONLY the
logic dynamic energy by a tile-measured post-route factor. SRAM, boundary wires [EST] and the SRAM-clock policy are
left as in e22/T2.10. Everything here is [EST] "tile-measured factor transferred to macro"; NOT signoff, NOT macro P&R.

Factors (from results/e24_pnr_{v2,dpu,bs_int5}/latest metrics.json; same rebased real0 window, OL2 sky130 tt, 14 ns):
  a = OL2 synthesis netlist, no wires (sta_variants a_synth_nospef)       -- the macro e22 logic is also a no-wire synthesis
  c = post-route netlist + SPEF, incl. the OpenSTA CTS Clock power group    (c_postroute_spef)
  c_excl = c minus the Clock group (e24_compare clock_group_sensitivity)
  e13 = zero-delay e13 real0 logic energy (different synthesis recipe: e12, unbuffered)
Scenarios applied to macro logic dynamic energy:
  L  = c_excl / a   wire cap + CTS/repair buffers, excluding the assumed-full-rate Clock group
  M  = c / a        as measured, including the Clock group
  H  = c / e13      upper bound only: also contains the OL2-vs-e12 synthesis-recipe gap (double counts with the td netlists)
Projected range = [L, M]; outer bound = H.  INT5: the e24 INT5 tile predates the INT5 back-to-back fix, so its own factors are EXCLUDED from the
range (INT5 uses the v2 factor as a proxy; its own factors appear only in rows labelled CONTAMINATED).
--int5-factor s3 (T3.2/T3.3 s3 refresh): INT5 gets its own post-fix factor from the s3 signoff tiles instead of the v2
proxy: r = H_s3(INT5) / H_s3(v2), H_s3 = s3 post-route + SPEF zero-delay logic energy (flows/pnr/post_route_power.py,
build/pnr/t31/<d>/power_s3, real0-3 mean) over the zero-delay td-netlist logic energy of the same windows (e13_power_td runs
listed in e13_boundary_td/latest), and INT5 L/M = v2 L/M x r, H = H_s3(INT5) [EST: assumes the a/e13 recipe ratio of v2].
Usage: python experiments/e32_macro_postroute_proj/run.py [--paper] [--int5-factor v2proxy|s3]
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import yaml

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

spec_ = importlib.util.spec_from_file_location("e22cmp", REPO_ROOT / "experiments/e21_macro_power/compare.py")
C = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(C)
R = C.R
POLICIES = ("gated", "as_built")  # gated = primary
SCEN = ("none", "L", "M", "H")


def tile_factors() -> tuple[dict, dict]:
    cmp_ = C.rd(R / "e24_compare/latest/metrics.json")
    pins = {}
    out = {}
    for key, exp in (("v2", "e24_pnr_v2"), ("dpu", "e24_pnr_dpu"), ("int5", "e24_pnr_bs_int5")):
        m = C.rd(R / exp / "latest/metrics.json")
        pins[key] = (R / exp / "latest").resolve().name
        v = m["sta_variants_e_logic_fj_per_mac"]
        a, b, c = v["a_synth_nospef"], v["b_postroute_nospef"], v["c_postroute_spef"]
        c_ex = cmp_["clock_group_sensitivity"][key]["e_logic_excl_clock_fj_per_mac"]
        out[key] = {"a": a, "b": b, "c": c, "c_excl": c_ex, "e13_logic": m["e13_zero_delay_real0_e_logic_fj_per_mac"],
                    "L": c_ex / a, "M": c / a, "H": c / m["e13_zero_delay_real0_e_logic_fj_per_mac"],
                    "cts_buffers_a_to_b": b / a, "spef_b_to_c": c / b,
                    "fmax_lb_mhz": m["fmax_mhz_postroute_nom_tt"], "util_total": m["utilization_total"],
                    "setup_ws_tt_ns": m["setup_ws_ns_nom_tt"]}
    pins["e24_compare"] = (R / "e24_compare/latest").resolve().name
    return out, pins


def s3_factors() -> dict:
    """H_s3 per design: s3 post-route zero-delay logic / e13 td zero-delay logic, real0-3 window mean."""
    src = yaml.safe_load((R / "e13_boundary_td/latest/config.yaml").read_text())["source_runs"]
    td = {}
    for run in src:  # later runs override
        for r in load_csv(R / "e13_power_td" / run / "power.csv"):
            td[r["design"], r["window"]] = float(r["e_logic_fj_per_mac"])
    out, per = {}, {}
    for d in ("v2_sram_m16", "bs_int5", "dpu"):
        f = []
        for w in ("real0", "real1", "real2", "real3"):
            z = C.rd(REPO_ROOT / "build/pnr/t31" / d / "power_s3" / w / "postroute_power.json")
            f.append(z["e_logic_fj_per_mac"] / td[d, w]); per[f"{d}_{w}"] = f[-1]
        out[d] = float(np.mean(f))
    return {"H_s3": out, "per_window": per, "pins": {"e13_power_td": src, "power_s3": "build/pnr/t31/<d>/power_s3/real0-3"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--int5-factor", choices=("v2proxy", "s3"), default="v2proxy")
    args = ap.parse_args()
    pins = yaml.safe_load((REPO_ROOT / "experiments/configs/paper_runs.yaml").read_text())
    e22_id = pins["figures_tables"]["e22_macro_compare"]
    cfg22 = yaml.safe_load((R / "e22_macro_compare" / e22_id / "config.yaml").read_text())
    rows = [r for d in cfg22["e21_runs"] for r in load_csv(R / "e21_macro_power" / d / "power.csv")]
    for r in rows:
        for k in ("n_ops", "cycles", "window_ns", "p_total_leak_w", "p_sram_sta_leak_w", "p_logic_w", "sram_dual_read_cycles",
                  "set_switches", "cycles_per_op", "e_logic_fj_per_mac", "bcast_toggles", "wmove_toggles"):
            r[k] = float(r[k])
    corners = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())["corners"]
    rows = [r for r in rows if "hier" not in r["design"]]
    hier = cfg22["flat_eq_hier_penalty"]
    rows += C.flat_eq_rows(rows, hier)
    C.TILE.update({"dpufeq_macro_mx": "dpu", "dpufeq_macro_nv": "dpu"})
    macro_syn = {**C.MACRO_SYN_SETS["td"], "dpufeq_macro_mx": C.MACRO_SYN_SETS["td"]["dpu_macro_mx"],
                 "dpufeq_macro_nv": C.MACRO_SYN_SETS["td"]["dpu_macro_nv"]}
    designs = sorted({r["design"] for r in rows})
    c_ff = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())["layers"]["met2"]["cap_ff_per_um"]
    area = {d: C.rd(R / macro_syn[d] / cfg22["macro_syn"][d] / "metrics.json")["total_mm2"] for d in designs}
    glitch = cfg22["glitch_factors_tile_sdf_over_zero"]
    fac, fac_pins = tile_factors()
    # design -> factor key. INT5 -> v2 proxy (own e24 factors predate the INT5 fix)
    s3 = None
    if args.int5_factor == "s3":
        s3 = s3_factors()
        r = s3["H_s3"]["bs_int5"] / s3["H_s3"]["v2_sram_m16"]
        fac["int5s3"] = {**{k: fac["v2"][k] * r for k in ("L", "M")}, "H": s3["H_s3"]["bs_int5"], "r_vs_v2": r}
        fac_pins["s3"] = s3["pins"]
    fkey = lambda d: "dpu" if d.startswith("dpu") else ("int5s3" if s3 and d.startswith("bs") else "v2")  # noqa: E731
    f40 = 40e6
    cfg = {"e22_run": e22_id, "e21_runs": cfg22["e21_runs"], "macro_syn": cfg22["macro_syn"], "flat_eq_hier_penalty": hier,
           "e24_inputs": fac_pins, "n_boot": args.n_boot, "policies": list(POLICIES), "scenarios": list(SCEN),
           "label": "[EST] tile-measured post-route factor transferred to macro; not signoff; not macro P&R",
           "int5_policy": ("e24 INT5 tile is pre-B15: own factors excluded from range, v2 factor used as proxy"
                           if not s3 else "INT5 own post-B15 factor from the s3 tiles: v2 L/M x H_s3(INT5)/H_s3(v2)"),
           "int5_factor": args.int5_factor, "s3": s3}
    metrics, tab, ftab = {}, [], []
    with start_run("e32_macro_postroute_proj", cfg, paper=args.paper) as run:
        def wt(d, scen, pol, g=1.0, corner="nominal"):
            f = 1.0 if scen == "none" else fac[fkey(d)][scen]
            out = {}
            for r in (x for x in rows if x["design"] == d and x["kind"] == "real"):
                b = C.op_energy(r, corners[corner], f40, g, pol, C.wire_fj(r, area[d], c_ff, "nom", d.startswith("dpu")))
                out[r["window"]] = b["total"] + (f - 1.0) * b["logic_dyn"]
            return out

        for key, v in fac.items():
            if key == "int5s3":
                ftab.append({"design": key, "L": v["L"], "M": v["M"], "H": v["H"], "r_vs_v2": v["r_vs_v2"],
                             "use": "INT5 own post-B15 factor (s3 tiles) [EST]"})
                for k in ("L", "M", "H", "r_vs_v2"):
                    metrics[f"factor_{key}_{k}"] = v[k]
                continue
            ftab.append({"design": key, **{k: v[k] for k in ("a", "b", "c", "c_excl", "e13_logic", "cts_buffers_a_to_b",
                                                            "spef_b_to_c", "L", "M", "H", "fmax_lb_mhz", "util_total")},
                         "use": "headline range" if key != "int5" else "CONTAMINATED (pre-B15, unbuffered tile): excluded; v2 proxy used"})
            for k in ("L", "M", "H", "cts_buffers_a_to_b", "spef_b_to_c", "fmax_lb_mhz"):
                metrics[f"factor_{key}_{k}"] = v[k]
        for a, b in C.RATIO_PAIRS:
            if a not in designs or b not in designs:
                continue
            key = f"{a.split('_')[0]}_{a[-2:]}_over_{b.split('_')[0]}_{b[-2:]}"
            for pol in POLICIES:
                for scen in SCEN:
                    for gname, gon in (("zero_delay", False), ("glitch_est", True)):
                        ea = wt(a, scen, pol, glitch[a] if gon else 1.0)
                        eb = wt(b, scen, pol, glitch[b] if gon else 1.0)
                        r, lo, hi, w = C.boot_ratio(ea, eb, np.random.default_rng(0), args.n_boot)
                        lab = "[EST]" if scen != "none" or gon else "[MEASURED]"
                        if "bs" in b and scen != "none" and s3:
                            lab += " INT5 side uses its own s3 factor (v2 L/M x H_s3 ratio)"
                        elif "bs" in b and scen != "none":
                            lab += " INT5 side uses v2 factor as proxy (own e24 INT5 factor contaminated)"
                        tab.append({"pair": key, "sram_clock": pol, "scenario": scen, "activity": gname, "ee_ratio": r,
                                    "ci_lo": lo, "ci_hi": hi, "e_num_fj_per_mac": float(np.mean(list(ea.values()))),
                                    "e_den_fj_per_mac": float(np.mean(list(eb.values()))), "n_windows": len(w), "label": lab})
                        metrics[f"ee_{key}_{pol}_{scen}_{gname}"] = r
                        metrics[f"ee_{key}_{pol}_{scen}_{gname}_ci_lo"] = lo
                        metrics[f"ee_{key}_{pol}_{scen}_{gname}_ci_hi"] = hi
            # contaminated sensitivity (INT5 own e24 factors), as-built/gated, zero-delay
            if "bs" in b:
                for pol in POLICIES:
                    for scen in ("L", "M", "H"):
                        ea = wt(a, scen, pol)
                        eb = {}
                        for rw in (x for x in rows if x["design"] == b and x["kind"] == "real"):
                            bb = C.op_energy(rw, corners["nominal"], f40, 1.0, pol,
                                             C.wire_fj(rw, area[b], c_ff, "nom", False))
                            eb[rw["window"]] = bb["total"] + (fac["int5"][scen] - 1.0) * bb["logic_dyn"]
                        r, lo, hi, w = C.boot_ratio(ea, eb, np.random.default_rng(0), args.n_boot)
                        tab.append({"pair": key, "sram_clock": pol, "scenario": scen + "_int5own", "activity": "zero_delay",
                                    "ee_ratio": r, "ci_lo": lo, "ci_hi": hi,
                                    "e_num_fj_per_mac": float(np.mean(list(ea.values()))),
                                    "e_den_fj_per_mac": float(np.mean(list(eb.values()))), "n_windows": len(w),
                                    "label": "[EST] CONTAMINATED: pre-B15 INT5 factor; not for paper numbers"})
        # headline ranges (zero-delay activity, [L, M]) and G3(1)-style informational test
        for a, b in C.RATIO_PAIRS:
            key = f"{a.split('_')[0]}_{a[-2:]}_over_{b.split('_')[0]}_{b[-2:]}"
            for pol in POLICIES:
                g = lambda s: metrics.get(f"ee_{key}_{pol}_{s}_zero_delay")  # noqa: E731
                if g("none") is None:
                    continue
                lm = [g("L"), g("M")]
                metrics[f"range_{key}_{pol}_lo"] = min(lm)
                metrics[f"range_{key}_{pol}_hi"] = max(lm)
                metrics[f"outer_{key}_{pol}_H"] = g("H")
                # historical G3(1): (ratio_post - 1) >= 0.7 (ratio_pre - 1), pre = e22 same policy, evaluated at M and H
                metrics[f"g3_1_{key}_{pol}_M_pass"] = bool((g("M") - 1) >= 0.7 * (g("none") - 1))
                metrics[f"g3_1_{key}_{pol}_H_pass"] = bool((g("H") - 1) >= 0.7 * (g("none") - 1))
        # absolute energies, v2/DPU/INT5, primary (gated, zero delay)
        for d in designs:
            for pol in POLICIES:
                for scen in SCEN:
                    metrics[f"e_{d}_{pol}_{scen}_fj_per_mac"] = float(np.mean(list(wt(d, scen, pol).values())))
        # timing / area transfer [EST]
        for k in ("v2", "dpu"):
            metrics[f"fmax_tile_postroute_lb_{k}_mhz"] = fac[k]["fmax_lb_mhz"]
            metrics[f"headroom_at_40MHz_{k}"] = fac[k]["fmax_lb_mhz"] / 40.0
        metrics["fmax_v2_over_dpu_tile_lb"] = fac["v2"]["fmax_lb_mhz"] / fac["dpu"]["fmax_lb_mhz"]
        utils = [fac["v2"]["util_total"], fac["dpu"]["util_total"]]
        for d in ("v2_macro_mx", "bs_macro_mx", "dpu_macro_mx", "dpuflat_macro_mx"):
            metrics[f"synth_area_mm2_{d}"] = area[d]
            metrics[f"die_est_mm2_{d}_lo"] = area[d] / max(utils)
            metrics[f"die_est_mm2_{d}_hi"] = area[d] / min(utils)
        metrics["die_util_range"] = utils
        run.save_csv("projection.csv", tab)
        run.save_csv("factors.csv", ftab)
        run.log_metrics(metrics)
        for t in tab:
            if t["activity"] == "zero_delay" and "int5own" not in t["scenario"] and t["sram_clock"] == "gated":
                print(f"{t['pair']:26s} {t['scenario']:5s} {t['ee_ratio']:.3f} [{t['ci_lo']:.3f},{t['ci_hi']:.3f}] "
                      f"{t['e_num_fj_per_mac']:.0f}/{t['e_den_fj_per_mac']:.0f}")
        for f in ftab:
            print(f["design"], {k: round(v, 3) for k, v in f.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
