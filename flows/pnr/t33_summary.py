#!/usr/bin/env python3
"""T3.3 summary (local, light): routed-SDF / zero-delay post-route energy factor per window and design, with a
window-level bootstrap CI, plus same-window v2/INT5 and v2/DPU energy-efficiency ratios on the routed numbers.

Inputs (from the remote t32 run):
  build/pnr/t31/<design>/t33/<window>/routed_gls.json        flows/pnr/routed_sdf_gls.py
  build/pnr/t31/<design>/power_s3/<window>/postroute_power.json  flows/pnr/post_route_power.py (zero-delay reference)
Also: v2/INT5 and v2/DPU ratios on routed logic energy alone, and on logic + SRAM with the e20 SRAM model
(docs/spec/sram_energy.yaml, nominal corner; tile = 1 OpenRAM macro; read words = e13 sram_accesses of the window program,
2 words per dual-read cycle) for both SRAM-clock policies: as_built = macro clocked every window cycle, gated = clocked
only on its dual-read cycles [EST design change]. Zero-delay (power_s3) ratios on the same bookkeeping for reference.
Acceptance (T3.3): every window GLS-exact (a json exists only for a passing GLS), >= 99 % pins annotated.

  python flows/pnr/t33_summary.py [--paper]   -> results/t33_routed_sdf_gls/<run>/
"""
import argparse, json
from pathlib import Path

import numpy as np
import yaml

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

DESIGNS = ("v2_sram_m16", "bs_int5", "dpu")
WINDOWS = ("real0", "real1", "real2", "real3", "rand0")
B = REPO_ROOT / "build/pnr/t31"
PROGS = REPO_ROOT / "build/power/e13_power_rb/programs"
POLICIES = ("as_built", "gated")


def sram_fj_per_mac(window: str, window_ns: float, clk_ps: float, macs: int, c: dict) -> dict:
    """e20 SRAM energy of the tile (1 macro) per policy, fJ/MAC."""
    s = np.load(PROGS / f"{window}.npz", allow_pickle=True)["set"]
    words = 16 * (1 + int(np.count_nonzero(np.diff(s) != 0)))  # = experiments/e13_power/run.py sram_accesses
    dual = int(np.ceil(words / 2))
    cycles = window_ns * 1e3 / clk_ps
    acc = words * c["de_read_per_word_dual_pj"]
    leak = c["leakage_uw"] * 1e-6 * window_ns * 1e-9 * 1e12  # pJ
    return {pol: (acc + (cycles if pol == "as_built" else dual) * c["e_idle_clock_cycle_pj"] + leak) * 1e3 / macs
            for pol in POLICIES}


def ci(x, rng, n=2000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--paper", action="store_true")
    ap.add_argument("--t43-plain", action="store_true", help="t43 reruns with plain VCD scopes: t43/<window>_plain")
    a = ap.parse_args()
    rows, e = [], {}
    corner = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())["corners"]["nominal"]
    ek = {}  # (kind, design, window) -> fJ/MAC, kind in logic_routed, logic_zero, <pol>_routed, <pol>_zero
    for d in DESIGNS:
        for w in WINDOWS:
            fr = B / d / ("t43" if a.t43_plain else "t33") / (f"{w}_plain" if a.t43_plain else w) / "routed_gls.json"
            fz = B / d / "power_s3" / w / "postroute_power.json"
            if not (fr.exists() and fz.exists()):
                rows.append({"design": d, "window": w, "missing": True}); continue
            r, z = json.loads(fr.read_text()), json.loads(fz.read_text())
            assert r["n_ops"] == z["n_ops"], (d, w)
            e[d, w] = r["e_total_fj_per_mac"]
            sr = sram_fj_per_mac(w, r["window_ns"], r["clk_ps"], r["macs"], corner)
            sz = sram_fj_per_mac(w, z["window_ns"], r["clk_ps"], z["macs"], corner)
            ek["logic_routed", d, w], ek["logic_zero", d, w] = r["e_logic_fj_per_mac"], z["e_logic_fj_per_mac"]
            for pol in POLICIES:
                ek[f"{pol}_routed", d, w] = r["e_logic_fj_per_mac"] + sr[pol]
                ek[f"{pol}_zero", d, w] = z["e_logic_fj_per_mac"] + sz[pol]
            rows.append({"design": d, "window": w, "missing": False, "gls_pass": r["gls_pass"],
                         "annotated_frac": r["annotation"].get("annotated_frac_excl_internal", r["annotation"]["annotated_frac"]),
                         "annotated_frac_raw": r["annotation"]["annotated_frac"],
                         "e_routed_fj_per_mac": r["e_total_fj_per_mac"], "e_zero_fj_per_mac": z["e_total_fj_per_mac"],
                         "e_routed_logic": r["e_logic_fj_per_mac"], "e_zero_logic": z["e_logic_fj_per_mac"],
                         "routed_over_zero": r["e_total_fj_per_mac"] / z["e_total_fj_per_mac"],
                         "routed_over_zero_logic": r["e_logic_fj_per_mac"] / z["e_logic_fj_per_mac"],
                         "sdf_coverage": r["sdf_coverage"]["coverage"], "mismatches": r["mismatches"],
                         "window_ns_routed": r["window_ns"], "window_ns_zero": z["window_ns"],
                         **{f"e_sram_d33_{p}_routed": sr[p] for p in POLICIES},
                         **{f"e_{p}_routed_fj_per_mac": ek[f"{p}_routed", d, w] for p in POLICIES}})
    rng = np.random.default_rng(0)
    m = {}
    for d in DESIGNS:
        f = [r["routed_over_zero"] for r in rows if r["design"] == d and not r["missing"] and r["window"].startswith("real")]
        m[f"{d}_routed_over_zero_real_mean"] = float(np.mean(f)) if f else float("nan")
        m[f"{d}_routed_over_zero_real_ci"] = ci(f, rng)
        m[f"{d}_windows_done"] = sum(1 for r in rows if r["design"] == d and not r["missing"])
        m[f"{d}_min_sdf_coverage"] = min((r["sdf_coverage"] for r in rows if r["design"] == d and not r["missing"]),
                                         default=float("nan"))
        m[f"{d}_total_mismatches"] = sum(r["mismatches"] for r in rows if r["design"] == d and not r["missing"])
        fl = [r["routed_over_zero_logic"] for r in rows if r["design"] == d and not r["missing"] and r["window"].startswith("real")]
        m[f"{d}_routed_over_zero_logic_real_mean"] = float(np.mean(fl)) if fl else float("nan")
        m[f"{d}_routed_over_zero_logic_real_ci"] = ci(fl, rng)
        m[f"{d}_min_annotated_frac"] = min((r["annotated_frac"] for r in rows if r["design"] == d and not r["missing"]),
                                           default=float("nan"))
    for base, k in (("bs_int5", "v2_over_int5"), ("dpu", "v2_over_dpu")):
        q = [e[base, w] / e["v2_sram_m16", w] for w in WINDOWS if w.startswith("real") and (base, w) in e
             and ("v2_sram_m16", w) in e]  # EE ratio = E(base)/E(v2), same windows only
        m[f"ee_{k}_routed_real_mean"] = float(np.mean(q)) if q else float("nan")
        m[f"ee_{k}_routed_real_ci"] = ci(q, rng)
        for kind in ("logic_routed", "logic_zero") + tuple(f"{p}_{t}" for p in POLICIES for t in ("routed", "zero")):
            q = [ek[kind, base, w] / ek[kind, "v2_sram_m16", w] for w in WINDOWS if w.startswith("real")
                 and (kind, base, w) in ek and (kind, "v2_sram_m16", w) in ek]
            m[f"ee_{k}_{kind}_real_mean"] = float(np.mean(q)) if q else float("nan")
            m[f"ee_{k}_{kind}_real_ci"] = ci(q, rng)
    m["accept_annotation"] = all(not r["missing"] and r["annotated_frac"] >= 0.99 for r in rows)
    m["accept_gls_and_sdf"] = all(not r["missing"] and r["gls_pass"] and r["sdf_coverage"] >= 0.99 for r in rows)
    m["accept"] = all(not r["missing"] and r["annotated_frac"] >= 0.99 for r in rows)
    with start_run("t33_routed_sdf_gls", {"t43_plain": a.t43_plain, "designs": DESIGNS, "windows": WINDOWS, "ci": "bootstrap over real windows",
                                           "sram_model": "docs/spec/sram_energy.yaml nominal, 1 macro/tile, policies " + ",".join(POLICIES)},
                   paper=a.paper) as run:
        run.save_csv("windows.csv", rows)
        run.log_metrics(m)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
