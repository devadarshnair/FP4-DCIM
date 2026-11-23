"""e22 - equal-boundary macro comparison + sensitivity analysis. Results only; no simulation.

Boundary (identical for both designs, rtl/src/macro_top.sv vs rtl/baselines/bitserial_int5/bs_macro_top.sv):
32 rows x 32 columns, 8 x sky130 OpenRAM 1 KB weight-set stores (M = 16 sets), same write port, same
per-(set, column) scale register files, same set-switch control, same 40-bit window outputs.

Inputs:
  results/e21_macro_power/latest      macro-level zero-delay GLS + OpenSTA logic power (measured)
  docs/spec/sram_energy.yaml          OpenRAM 1 KB energy range (low / nominal / high, from e20)
  results/e12_macro_synth_*/latest    macro area (measured, synthesis) + unbuffered macro timing
  results/e12_synth/latest            column-group fmax (synthesis, buffered-equivalent tile timing)
  results/e13_sdf/latest              tile-level SDF/zero-delay dynamic-energy ratio per design [EST at macro]

Energy per op (one op = 32x32 = 1024 MACs):
  E_logic   = e21 STA logic energy (dynamic + leakage), optionally x SDF/zero glitch factor on the
              dynamic part (tile-level ratio transferred to the macro: [EST]).
  E_sram    = dual-read cycles x dE_dual_read + 8 macros x cycles x E_idle_clock (as-built: the OpenRAM
              clocks are free-running in the RTL) + 8 x cycles x P_leak x T.  "gated" policy (an ICG on
              each OpenRAM clock, not in the RTL: [EST]) drops E_idle_clock on non-access cycles.
  leakage of logic and SRAM scales with cycles_per_op / f; dynamic energy is frequency-independent.
Throughput = 1024 MAC x f / cycles_per_op. Clock options: common 40 MHz (as simulated), tile fmax
(e12_synth column-group designs), macro unbuffered r2r fmax (e12_macro_synth; pessimistic lower bound).

T2.7 options (defaults reproduce the pre-T2.7 e22 run): --macro-syn td (area from the e30 td macro runs), --sdf-exp
e13_sdf_td (glitch factors), --clocks common_40MHz,common_34MHz (equal clock only, T0.7; 34 MHz = OpenRAM silicon-verified
clock), --flat-eq 1.19112 (adds the derived flat-equivalent DPU rows dpufeq_*: DPU logic / e23 tile hier penalty
[DERIVED]), paired window bootstrap CIs of every ratio at the primary and glitch-corrected points, measured-axis range
(SRAM low/nominal, zero delay, as-built, nominal wires, 40 MHz) kept separate from the full [EST] grid, an NV row
(nv_row.csv, the e26 table) and the v2 hier/flat calibration when v2hier rows are present.

Run: python experiments/e21_macro_power/compare.py [--paper]   (defaults: e21 v2+INT5 run and e21 DPU run, in that order;
pass --e21 <run> ... to override. e12 DPU area is the hierarchical-Yosys run: see its NOTE.md)
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import yaml

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

R = REPO_ROOT / "results"
MACS = 1024
NG = 8
PAIRS = {"v2": "v2_macro_mx", "bs": "bs_macro_mx", "dpu": "dpu_macro_mx"}
TILE = {"v2_macro_mx": "v2_sram_m16", "bs_macro_mx": "bs_int5", "v2_macro_nv": "v2_sram_m16",
        "dpu_macro_mx": "dpu", "dpu_macro_nv": "dpu",
        "dpuflat_macro_mx": "dpu", "dpuflat_macro_nv": "dpu"}
MACRO_SYN = {"v2_macro_mx": "e12_macro_synth_v2_macro", "bs_macro_mx": "e12_macro_synth_bs_macro",
             "v2_macro_nv": "e12_macro_synth_v2_macro", "dpu_macro_mx": "e12_macro_synth_dpu_macro_hier",
             "dpu_macro_nv": "e12_macro_synth_dpu_macro_hier"}


MACRO_SYN_SETS = {
    "e12": MACRO_SYN,
    "td": {"v2_macro_mx": "e30_td_macro_v2_macro", "bs_macro_mx": "e30_td_macro_bs_macro",
           "v2_macro_nv": "e30_td_macro_v2_macro", "dpu_macro_mx": "e30_td_macro_dpu_macro_hier",
           "dpu_macro_nv": "e30_td_macro_dpu_macro_hier",
           # T2.11: measured flat-Yosys DPU macro (e21 designs dpuflat_macro_*)
           "dpuflat_macro_mx": "e30_td_macro_dpu_macro", "dpuflat_macro_nv": "e30_td_macro_dpu_macro"},
}
# ratio pairs (numerator design = the v2 macro): EE(a)/EE(b) = E(b)/E(a)
RATIO_PAIRS = [("v2_macro_mx", "bs_macro_mx"), ("v2_macro_mx", "dpu_macro_mx"), ("v2_macro_mx", "dpufeq_macro_mx"),
               ("v2_macro_nv", "dpu_macro_nv"), ("v2_macro_nv", "dpufeq_macro_nv"),
               ("v2_macro_mx", "dpuflat_macro_mx"), ("v2_macro_nv", "dpuflat_macro_nv")]  # dpuflat: T2.11 measured flat


def flat_eq_rows(rows: list[dict], hier: float) -> list[dict]:
    """[DERIVED] flat-equivalent DPU: logic power (dynamic and leakage) / the e23 tile hier penalty; SRAM terms unchanged."""
    out = []
    for r in rows:
        if r["design"].startswith("dpu_"):
            q = dict(r)
            q["design"] = r["design"].replace("dpu_", "dpufeq_")
            q["p_logic_w"] = r["p_logic_w"] / hier
            q["p_total_leak_w"] = (r["p_total_leak_w"] - r["p_sram_sta_leak_w"]) / hier + r["p_sram_sta_leak_w"]
            q["e_logic_fj_per_mac"] = r["e_logic_fj_per_mac"] / hier
            out.append(q)
    return out


def rd(path):
    return json.loads(path.read_text())


WIRE_BANDS = {"low": (1.0, 0.7), "nom": (1.0, 1.0), "high": (2.0, 1.5)}  # (C multiplier, length multiplier)


def wire_fj(row: dict, area_mm2: float, c_ff_um: float, band: str, central: bool = False) -> float:
    """[EST] macro-boundary wires absent from the synthesis netlist: row broadcast over the macro width
    (sqrt(A)) and the SRAM->weight-latch move inside each group (sqrt(A/8)/2); 1/2 C V^2 per toggle."""
    cm, lm = WIRE_BANDS[band]
    side = np.sqrt(area_mm2) * 1e3 * lm
    e_tog = lambda length: 0.5 * c_ff_um * cm * length * 1.8**2  # fJ  # noqa: E731
    # central=True (DPU baseline): weights come from a central buffer over a bus of 2*sqrt(A) (boundary.py), [EST]
    wlen = 2.0 * side if central else np.sqrt(area_mm2 / NG) * 1e3 * lm / 2
    return row["bcast_toggles"] * e_tog(side) + row["wmove_toggles"] * e_tog(wlen)


def op_energy(row: dict, sram: dict, f_hz: float, glitch: float, policy: str, e_wire_fj: float = 0.0) -> dict:
    """fJ/MAC breakdown of one e21 row at clock f, SRAM corner `sram`, glitch factor, clock policy."""
    n_ops, cycles = row["n_ops"], row["cycles"]
    dur_sim = row["window_ns"] * 1e-9
    p_leak = row["p_total_leak_w"] - row["p_sram_sta_leak_w"]
    p_dyn = row["p_logic_w"] - p_leak
    e_dyn = p_dyn * dur_sim * glitch  # J, frequency-independent switching energy
    t = cycles / f_hz
    e_leak = p_leak * t
    n_acc = row["sram_dual_read_cycles"]  # summed over the 8 OpenRAM macros
    e_sram_acc = n_acc * sram["de_dual_read_cycle_pj"] * 1e-12  # energy above a clocked idle cycle
    # as_built: every macro is clocked every cycle; gated: only on its access cycles
    clocked = NG * cycles if policy == "as_built" else n_acc
    e_sram_idle = clocked * sram["e_idle_clock_cycle_pj"] * 1e-12
    e_sram_leak = NG * sram["leakage_uw"] * 1e-6 * t
    k = 1e15 / (n_ops * MACS)
    out = {"logic_dyn": e_dyn * k, "logic_leak": e_leak * k, "wire": e_wire_fj * 1e-15 * k,
           "sram_access": e_sram_acc * k, "sram_idle_clock": e_sram_idle * k, "sram_leak": e_sram_leak * k}
    out["total"] = sum(out.values())
    return out


def window_totals(rows, d, kind, corner, f, g, policy, area_mm2, c_ff_um, wb):
    """{window: total fJ/MAC} of design d (one e21 row per window)."""
    return {r["window"]: op_energy(r, corner, f, g, policy, wire_fj(r, area_mm2, c_ff_um, wb, d.startswith("dpu")))["total"]
            for r in rows if r["design"] == d and r["kind"] == kind}


def boot_ratio(ea: dict, eb: dict, rng, n: int) -> tuple[float, float, float, list]:
    """Ratio of window means E(b)/E(a) on the common windows, paired bootstrap 95 % CI (resampling windows)."""
    w = sorted(set(ea) & set(eb))
    a, b = np.array([ea[k] for k in w]), np.array([eb[k] for k in w])
    idx = rng.integers(0, len(w), (n, len(w)))
    bs = b[idx].mean(1) / a[idx].mean(1)
    return float(b.mean() / a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), w


def extended(rows, designs, corners, glitch, area, c_ff_um, f_opts, n_boot, run) -> dict:
    """T2.7/T2.12 additions: paired CIs, measured vs [EST] ranges, per-window ratios, throughput/density, NV row."""
    rng = np.random.default_rng(0)
    out, tab = {}, []
    f40 = 40e6
    kinds = sorted({r["kind"] for r in rows})
    for a, b in RATIO_PAIRS:
        if a not in designs or b not in designs:
            continue
        key = f"ee_{a.split('_')[0]}_{a[-2:]}_over_{b.split('_')[0]}_{b[-2:]}"
        for kind in kinds:
            res = {}
            for label, g_on, corner, pol, wb in (("primary", False, "nominal", "as_built", "nom"),
                                                 ("glitch_est", True, "nominal", "as_built", "nom")):
                ea = window_totals(rows, a, kind, corners[corner], f40, glitch[a] if g_on else 1.0, pol, area[a]["total_mm2"], c_ff_um, wb)
                eb = window_totals(rows, b, kind, corners[corner], f40, glitch[b] if g_on else 1.0, pol, area[b]["total_mm2"], c_ff_um, wb)
                if not ea or not eb:
                    continue
                r, lo, hi, w = boot_ratio(ea, eb, rng, n_boot)
                res[label] = (r, lo, hi)
                out[f"{key}_{kind}_{label}"] = r
                out[f"{key}_{kind}_{label}_ci"] = [lo, hi]
                if label == "primary":
                    out[f"{key}_{kind}_by_window"] = {k: eb[k] / ea[k] for k in w}
                    out[f"{key}_{kind}_n_windows"] = len(w)
            if not res:
                continue
            # measured-axis range: SRAM corners measured by SPICE (low, nominal); zero delay; as-built; nominal wires; 40 MHz
            meas, est = [], []
            for cn, delay, pol, wb, fk in itertools.product(corners, ("zero", "sdf_est"), ("as_built", "gated"),
                                                           WIRE_BANDS, f_opts):
                ga = glitch[a] if delay == "sdf_est" else 1.0
                gb = glitch[b] if delay == "sdf_est" else 1.0
                ea = window_totals(rows, a, kind, corners[cn], f_opts[fk][a], ga, pol, area[a]["total_mm2"], c_ff_um, wb)
                eb = window_totals(rows, b, kind, corners[cn], f_opts[fk][b], gb, pol, area[b]["total_mm2"], c_ff_um, wb)
                rr = float(np.mean(list(eb.values())) / np.mean(list(ea.values())))
                est.append(rr)
                if cn in ("low", "nominal") and delay == "zero" and pol == "as_built" and wb == "nom" and fk == "common_40MHz":
                    meas.append(rr)
            out[f"{key}_{kind}_range_measured_axes"] = [min(meas), max(meas)] if meas else None
            out[f"{key}_{kind}_range_est_grid"] = [min(est), max(est)]
            tab.append({"pair": key, "kind": kind, "primary": res["primary"][0], "ci_lo": res["primary"][1],
                        "ci_hi": res["primary"][2], "glitch_est": res["glitch_est"][0], "glitch_ci_lo": res["glitch_est"][1],
                        "glitch_ci_hi": res["glitch_est"][2], "meas_lo": min(meas) if meas else "",
                        "meas_hi": max(meas) if meas else "", "est_lo": min(est), "est_hi": max(est)})
    run.save_csv("ratios_ci.csv", tab)
    # throughput and density at equal clock (T0.7): every clock option
    for fk, fd in f_opts.items():
        for d in designs:
            cpo = float(np.mean([r["cycles_per_op"] for r in rows if r["design"] == d and r["kind"] == "real"]))
            thr = MACS * fd[d] / cpo
            out[f"{d}_{fk}_gmac_per_s"] = thr / 1e9
            out[f"{d}_{fk}_gflops_per_mm2"] = 2 * thr / 1e9 / area[d]["total_mm2"]
            out[f"{d}_cycles_per_op_real"] = cpo
        for a, b in (("v2_macro_mx", "bs_macro_mx"), ("v2_macro_mx", "dpu_macro_mx")):
            if a in designs and b in designs:
                out[f"throughput_{a}_over_{b}_{fk}"] = out[f"{a}_{fk}_gmac_per_s"] / out[f"{b}_{fk}_gmac_per_s"]
                out[f"density_{a}_over_{b}_{fk}"] = out[f"{a}_{fk}_gflops_per_mm2"] / out[f"{b}_{fk}_gflops_per_mm2"]
    # NV row (the e26 table), 40 MHz, zero delay, nominal wires
    if "v2_macro_nv" in designs and "dpu_macro_nv" in designs:
        nv = []
        for kind in kinds:
            for cn, corner in corners.items():
                for pol in ("as_built", "gated"):
                    def tot(d, kind=kind, corner=corner, pol=pol):
                        ps = [op_energy(r, corner, f40, 1.0, pol, wire_fj(r, area[d]["total_mm2"], c_ff_um, "nom", d.startswith("dpu")))
                              for r in rows if r["design"] == d and r["kind"] == kind]
                        return {k: float(np.mean([p[k] for p in ps])) for k in ps[0]} if ps else None
                    v, dh = tot("v2_macro_nv"), tot("dpu_macro_nv")
                    df = tot("dpufeq_macro_nv") if "dpufeq_macro_nv" in designs else None
                    dm = tot("dpuflat_macro_nv") if "dpuflat_macro_nv" in designs else None
                    if v is None or dh is None:
                        continue
                    nv.append({"kind": kind, "sram": cn, "sram_clock": pol, "v2_nv": v["total"], "dpu_nv_hier": dh["total"],
                               "dpu_nv_flat_eq_DERIVED": df["total"] if df else "",
                               "v2_logic": v["logic_dyn"] + v["logic_leak"], "dpu_hier_logic": dh["logic_dyn"] + dh["logic_leak"],
                               "ratio_v2_over_dpu_hier": dh["total"] / v["total"],
                               "ratio_v2_over_dpu_flat_eq_DERIVED": df["total"] / v["total"] if df else "",
                               "dpu_nv_flat_measured": dm["total"] if dm else "",
                               "ratio_v2_over_dpu_flat_measured": dm["total"] / v["total"] if dm else ""})
        run.save_csv("nv_row.csv", nv)
    # v2 NV/MX cost on the same windows (paired)
    if "v2_macro_nv" in designs and "v2_macro_mx" in designs:
        for d0, d1 in (("v2_macro_mx", "v2_macro_nv"), ("dpu_macro_mx", "dpu_macro_nv")):
            if d0 in designs and d1 in designs:
                e0 = window_totals(rows, d0, "real", corners["nominal"], f40, 1.0, "as_built", area[d0]["total_mm2"], c_ff_um, "nom")
                e1 = window_totals(rows, d1, "real", corners["nominal"], f40, 1.0, "as_built", area[d1]["total_mm2"], c_ff_um, "nom")
                r, lo, hi, _ = boot_ratio(e0, e1, rng, n_boot)
                out[f"{d1}_over_{d0}_energy"] = r
                out[f"{d1}_over_{d0}_energy_ci"] = [lo, hi]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--e21", nargs="+", default=["20261110T050824693997Z_f25a5c5", "20261110T151121726136Z_58882a0"],
                    help="e21 run dirs (names under results/e21_macro_power) whose power.csv rows are concatenated")
    ap.add_argument("--macro-syn", choices=sorted(MACRO_SYN_SETS), default="e12", help="macro area/timing source runs")
    ap.add_argument("--sdf-exp", default="e13_sdf", help="tile SDF/zero glitch factors (e13_sdf_rb, e13_sdf_td)")
    ap.add_argument("--clocks", default="common_40MHz,tile_fmax,macro_unbuffered_r2r",
                    help="clock options: common_40MHz, common_34MHz, tile_fmax, macro_unbuffered_r2r")
    ap.add_argument("--flat-eq", type=float, default=0.0, help="e23 DPU tile hier penalty (1.19112) -> dpufeq_* rows")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    macro_syn = MACRO_SYN_SETS[args.macro_syn]
    e21 = R / "e21_macro_power" / args.e21[0]
    rows = [r for d in args.e21 for r in load_csv(R / "e21_macro_power" / d / "power.csv")]
    num = ("n_ops", "cycles", "window_ns", "p_total_leak_w", "p_sram_sta_leak_w", "p_logic_w",
           "sram_dual_read_cycles", "set_switches", "cycles_per_op", "e_logic_fj_per_mac",
           "bcast_toggles", "wmove_toggles")
    for r in rows:
        for k in num:
            r[k] = float(r[k])
    spec = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())
    corners = spec["corners"]
    sdf = rd(R / args.sdf_exp / "latest" / "metrics.json")
    e12 = rd(R / "e12_synth" / "latest" / "metrics.json")
    calib = {}  # v2 hierarchical-vs-flat Yosys logic-energy ratio on the same windows (flow calibration)
    hier_rows = {r["window"]: r for r in rows if r["design"] == "v2hier_macro_mx"}
    for r in rows:
        if r["design"] == "v2_macro_mx" and r["window"] in hier_rows:
            calib[r["window"]] = hier_rows[r["window"]]["e_logic_fj_per_mac"] / r["e_logic_fj_per_mac"]
    rows = [r for r in rows if "hier" not in r["design"]]  # hier-vs-flat calibration rows: see NOTE, not compared
    if args.flat_eq:
        rows += flat_eq_rows(rows, args.flat_eq)
        TILE.update({"dpufeq_macro_mx": "dpu", "dpufeq_macro_nv": "dpu"})
        macro_syn = {**macro_syn, "dpufeq_macro_mx": macro_syn["dpu_macro_mx"], "dpufeq_macro_nv": macro_syn["dpu_macro_nv"]}
    designs = sorted({r["design"] for r in rows})
    c_ff_um = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())["layers"]["met2"]["cap_ff_per_um"]
    area = {d: rd(R / macro_syn[d] / "latest" / "metrics.json") for d in designs}
    glitch = {d: sdf[f"{TILE[d]}_sdf_over_zero"] for d in designs}
    f_all = {"common_40MHz": {d: 40e6 for d in designs}, "common_34MHz": {d: 34e6 for d in designs},
             "tile_fmax": {d: e12[f"{TILE[d]}_fmax_mhz"] * 1e6 for d in designs},
             "macro_unbuffered_r2r": {d: area[d]["r2r_fmax_mhz"] * 1e6 for d in designs}}
    f_opts = {k: f_all[k] for k in args.clocks.split(",")}
    cfg = {"e21_runs": [(R / "e21_macro_power" / d).resolve().name for d in args.e21], "sram_spec": spec.get("source_run"),
           "glitch_factors_tile_sdf_over_zero": glitch, "f_options_hz": f_opts,
           "macs_per_op": MACS, "policies": ["as_built", "gated"], "macro_syn": {d: (R / macro_syn[d] / "latest").resolve().name
                                                                                  for d in designs},
           "macro_syn_set": args.macro_syn, "sdf_exp": args.sdf_exp,
           "sdf_run": (R / args.sdf_exp / "latest").resolve().name, "flat_eq_hier_penalty": args.flat_eq}
    with start_run("e22_macro_compare", cfg, paper=args.paper) as run:
        grid = []
        for d, kind, corner, delay, policy, fopt, wb in itertools.product(
            designs, ("real", "rand"), corners, ("zero", "sdf_est"), ("as_built", "gated"), f_opts, WIRE_BANDS
        ):
            sel = [r for r in rows if r["design"] == d and r["kind"] == kind]
            if not sel:
                continue
            g = glitch[d] if delay == "sdf_est" else 1.0
            f = f_opts[fopt][d]
            parts = [op_energy(r, corners[corner], f, g, policy, wire_fj(r, area[d]["total_mm2"], c_ff_um, wb, d.startswith("dpu")))
                     for r in sel]
            mean = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
            cpo = float(np.mean([r["cycles_per_op"] for r in sel]))
            thr = MACS * f / cpo  # MAC/s
            grid.append({"design": d, "kind": kind, "sram": corner, "delay": delay, "sram_clock": policy,
                         "clock": fopt, "wire": wb, "f_mhz": f / 1e6, "cycles_per_op": cpo,
                         **{f"e_{k}_fj_per_mac": v for k, v in mean.items()},
                         "tflops_per_w": 2 / (mean["total"] * 1e-15) / 1e12,
                         "gmac_per_s": thr / 1e9, "area_mm2": area[d]["total_mm2"],
                         "gflops_per_mm2": 2 * thr / 1e9 / area[d]["total_mm2"]})
        run.save_csv("grid.csv", grid)

        def pick(d, **kw):
            return next(g for g in grid if g["design"] == d and all(g[k] == v for k, v in kw.items()))

        ratios = []
        for kind, corner, delay, policy, fopt, wb in itertools.product(
            ("real", "rand"), corners, ("zero", "sdf_est"), ("as_built", "gated"), f_opts, WIRE_BANDS
        ):
            kw = dict(kind=kind, sram=corner, delay=delay, sram_clock=policy, clock=fopt, wire=wb)
            a, b = pick(PAIRS["v2"], **kw), pick(PAIRS["bs"], **kw)
            c = pick(PAIRS["dpu"], **kw) if PAIRS["dpu"] in designs else None
            ratios.append({**kw, **({
                "ee_ratio_v2_over_dpu": c["e_total_fj_per_mac"] / a["e_total_fj_per_mac"],
                "throughput_ratio_v2_over_dpu": a["gmac_per_s"] / c["gmac_per_s"],
                "density_ratio_v2_over_dpu": a["gflops_per_mm2"] / c["gflops_per_mm2"],
                "dpu_fj_per_mac": c["e_total_fj_per_mac"],
                "dpu_fj_over_bs_fj": c["e_total_fj_per_mac"] / b["e_total_fj_per_mac"]} if c else {}),
                           "ee_ratio_v2_over_bs": b["e_total_fj_per_mac"] / a["e_total_fj_per_mac"],
                           "throughput_ratio_v2_over_bs": a["gmac_per_s"] / b["gmac_per_s"],
                           "density_ratio_v2_over_bs": a["gflops_per_mm2"] / b["gflops_per_mm2"],
                           "v2_fj_per_mac": a["e_total_fj_per_mac"], "bs_fj_per_mac": b["e_total_fj_per_mac"],
                           "v2_sram_share": (a["e_sram_access_fj_per_mac"] + a["e_sram_idle_clock_fj_per_mac"]
                                             + a["e_sram_leak_fj_per_mac"]) / a["e_total_fj_per_mac"],
                           "bs_sram_share": (b["e_sram_access_fj_per_mac"] + b["e_sram_idle_clock_fj_per_mac"]
                                             + b["e_sram_leak_fj_per_mac"]) / b["e_total_fj_per_mac"]})
        run.save_csv("ratios.csv", ratios)
        prim = dict(kind="real", sram="nominal", delay="zero", sram_clock="as_built", clock="common_40MHz", wire="nom")
        p = next(r for r in ratios if all(r[k] == v for k, v in prim.items()))
        ee = [r["ee_ratio_v2_over_bs"] for r in ratios if r["kind"] == "real"]
        if PAIRS["dpu"] in designs:
            eed = [r["ee_ratio_v2_over_dpu"] for r in ratios if r["kind"] == "real"]
            m_dpu = {"ee_ratio_v2_over_dpu_min_real": float(min(eed)), "ee_ratio_v2_over_dpu_max_real": float(max(eed)),
                     "area_ratio_v2_over_dpu": area[PAIRS["v2"]]["total_mm2"] / area[PAIRS["dpu"]]["total_mm2"]}
        else:
            m_dpu = {}
        m = {**m_dpu, "primary_point": prim, **{f"primary_{k}": v for k, v in p.items() if k not in prim},
             "ee_ratio_min_real": float(min(ee)), "ee_ratio_max_real": float(max(ee)),
             "area_mm2": {d: area[d]["total_mm2"] for d in designs},
             "area_ratio_v2_over_bs": area[PAIRS["v2"]]["total_mm2"] / area[PAIRS["bs"]]["total_mm2"]}
        for d in designs:
            g0 = pick(d, **prim)
            m[f"{d}_primary"] = {k: g0[k] for k in g0 if k.startswith("e_") or k in
                                 ("tflops_per_w", "gmac_per_s", "gflops_per_mm2", "cycles_per_op")}
        # WP-72 "reload energy per set switch extractable": SRAM read (access above idle) + weight-move bus wire
        # [EST wire], real windows, nominal SRAM corner, nominal wire band
        rl = {}
        for d in designs:
            sel = [r for r in rows if r["design"] == d and r["kind"] == "real" and r["set_switches"] > 0]
            if sel:
                rl[d] = float(np.mean([
                    (r["sram_dual_read_cycles"] * corners["nominal"]["de_dual_read_cycle_pj"] * 1e3
                     + r["wmove_toggles"] * 0.5 * c_ff_um * 1.8**2 * np.sqrt(area[d]["total_mm2"]) * 1e3
                     * (2.0 if d.startswith("dpu") else np.sqrt(1 / NG) / 2)) / r["set_switches"] for r in sel]))
        m["reload_fj_per_set_switch_nominal"] = rl
        m["v2_hier_over_flat_logic_by_window"] = calib
        if calib:
            m["v2_hier_over_flat_logic_mean"] = float(np.mean(list(calib.values())))
        # T2.11: measured DPU hier/flat logic-energy penalty on the same windows (replaces the e23 tile transfer)
        for fmt in ("mx", "nv"):
            hq = {r["window"]: r["e_logic_fj_per_mac"] for r in rows if r["design"] == f"dpu_macro_{fmt}"}
            fq = {r["window"]: r["e_logic_fj_per_mac"] for r in rows if r["design"] == f"dpuflat_macro_{fmt}"}
            w = sorted(set(hq) & set(fq))
            if w:
                m[f"dpu_hier_over_flat_logic_{fmt}_by_window"] = {k: hq[k] / fq[k] for k in w}
                m[f"dpu_hier_over_flat_logic_{fmt}_mean_ratio"] = float(np.mean([hq[k] for k in w]) / np.mean([fq[k] for k in w]))
                m[f"dpu_hier_over_flat_logic_{fmt}_n_windows"] = len(w)
        m.update(extended(rows, designs, corners, glitch, area, c_ff_um, f_opts, args.n_boot, run))
        run.log_metrics(m)
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
