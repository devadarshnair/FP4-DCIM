#!/usr/bin/env python3
"""T2.10: gated vs free-running SRAM-clock normalisation rows on the T2.7 inputs.

Post-processing only (no simulation): re-evaluates the e22 energy model (experiments/e21_macro_power/compare.py,
`op_energy`, SRAM idle-clock term) on exactly the e21 runs, td macro-synthesis runs and e13_sdf_td run recorded in the
pinned e22 run's config.yaml (experiments/configs/paper_runs.yaml), for both SRAM clock policies:
  as_built = every OpenRAM store is clocked every cycle (free-running, as in the RTL),
  gated    = the store is clocked only on its access cycles ([EST] design change: an ICG on each SRAM clock).
Primary point as in e22: real windows, nominal e20 SRAM corner, zero delay, 40 MHz, nominal wire band [EST].
CIs: paired bootstrap over windows (ratio of means, 2000 resamples, same resample indices for both policies).
Also: iso-throughput / iso-area normalisation [DERIVED] from e22 throughput and area (energy/MAC is unchanged by
replicating columns; only area changes).
Usage: python experiments/t210_sram_gating/run.py [--paper]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools

import numpy as np
import yaml

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

spec_ = importlib.util.spec_from_file_location("e22cmp", REPO_ROOT / "experiments/e21_macro_power/compare.py")
C = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(C)
R = C.R
POLICIES = ("as_built", "gated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    pins = yaml.safe_load((REPO_ROOT / "experiments/configs/paper_runs.yaml").read_text())
    e22_id = pins["figures_tables"]["e22_macro_compare"]
    e22 = R / "e22_macro_compare" / e22_id
    cfg22 = yaml.safe_load((e22 / "config.yaml").read_text())
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
    area = {d: C.rd(R / macro_syn[d] / cfg22["macro_syn"][d] / "metrics.json") for d in designs}
    glitch = {d: cfg22["glitch_factors_tile_sdf_over_zero"][d] for d in designs}
    f40 = 40e6
    cfg = {"e22_run": e22_id, "e21_runs": cfg22["e21_runs"], "macro_syn": cfg22["macro_syn"],
           "sdf_run": cfg22["sdf_run"], "flat_eq_hier_penalty": hier, "n_boot": args.n_boot,
           "policies": list(POLICIES), "primary": "real, nominal SRAM, zero delay, 40 MHz, nominal wires"}
    metrics, tab = {}, []
    with start_run("t210_sram_gating", cfg, paper=args.paper) as run:
        def wt(d, corner, g, pol, wb="nom"):
            return C.window_totals(rows, d, "real", corners[corner], f40, g, pol, area[d]["total_mm2"], c_ff, wb)

        for a, b in C.RATIO_PAIRS:
            if a not in designs or b not in designs:
                continue
            key = f"{a.split('_')[0]}_{a[-2:]}_over_{b.split('_')[0]}_{b[-2:]}"
            for pol in POLICIES:
                for label, corner, gon in (("primary", "nominal", False), ("glitch_est", "nominal", True),
                                           ("sram_low", "low", False), ("sram_high", "high", False)):
                    ea = wt(a, corner, glitch[a] if gon else 1.0, pol)
                    eb = wt(b, corner, glitch[b] if gon else 1.0, pol)
                    r, lo, hi, w = C.boot_ratio(ea, eb, np.random.default_rng(0), args.n_boot)
                    tag = "[EST]" if gon or corner == "high" else "[MEASURED]"
                    tab.append({"pair": key, "sram_clock": pol, "variant": label, "ee_ratio": r, "ci_lo": lo, "ci_hi": hi,
                                "e_num_fj_per_mac": float(np.mean(list(ea.values()))),
                                "e_den_fj_per_mac": float(np.mean(list(eb.values()))), "n_windows": len(w),
                                "label": tag + (" [DERIVED]" if "dpufeq" in key else "") + (" [EST: gated clock is a design change]" if pol == "gated" else "")})
                    metrics[f"ee_{key}_{pol}_{label}"] = r
                    metrics[f"ee_{key}_{pol}_{label}_ci_lo"] = lo
                    metrics[f"ee_{key}_{pol}_{label}_ci_hi"] = hi
        # per-design energies
        for d in designs:
            for pol in POLICIES:
                metrics[f"e_{d}_{pol}_fj_per_mac"] = float(np.mean(list(wt(d, "nominal", 1.0, pol).values())))
        # iso-throughput / iso-area [DERIVED] (v2 vs INT5 and vs DPU, MX; per-op energy independent of column count)
        thr = {d: C.MACS * f40 / float(np.mean([r["cycles_per_op"] for r in rows if r["design"] == d and r["kind"] == "real"]))
               for d in designs}
        for b in ("bs_macro_mx", "dpu_macro_mx", "dpufeq_macro_mx", "dpuflat_macro_mx"):
            if b not in designs:
                continue
            n = b.split("_")[0]
            tr = thr["v2_macro_mx"] / thr[b]
            metrics[f"throughput_v2_over_{n}"] = tr
            metrics[f"area_{n}_over_v2"] = area[b]["total_mm2"] / area["v2_macro_mx"]["total_mm2"]
            # iso-throughput: replicate the baseline `tr` times (energy/MAC unchanged) -> its area is tr*A_b, i.e.
            # area_ratio_iso_throughput = tr*A_b/A_v2; at iso-area the throughput ratio is the same number (= density ratio)
            metrics[f"iso_throughput_area_{n}_over_v2"] = tr * area[b]["total_mm2"] / area["v2_macro_mx"]["total_mm2"]
            metrics[f"iso_area_throughput_v2_over_{n}"] = tr * area[b]["total_mm2"] / area["v2_macro_mx"]["total_mm2"]
        run.save_csv("rows.csv", tab)
        run.log_metrics(metrics)
        for t in tab:
            if t["variant"] == "primary":
                print(f"{t['pair']:28s} {t['sram_clock']:9s} {t['ee_ratio']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]"
                      f"  {t['e_num_fj_per_mac']:.1f} / {t['e_den_fj_per_mac']:.1f}")


if __name__ == "__main__":
    main()
