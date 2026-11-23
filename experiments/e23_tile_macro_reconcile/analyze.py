"""e23 analysis - reconcile the tile-level G1 ratios (e13/e15) with the macro-level ratios
(e21/e22). Results only (no simulation): reads e13_power_rb / e13_boundary_rb, e15_g1_relative_rb,
e21_macro_power, e22 (SRAM/wire model in experiments/e21_macro_power/compare.py) and the e23 tile runs.

Chain (per design, fJ/MAC, ratio of means; each step replaces one quantity, all else unchanged):
  A0  tile G1 as reported (mean of 8 per-window ratios)                              [e15]
  A1  ratio of means over the 8 windows (the macro quotes ratio of means)            [e13]
  A2  windows real0..3 only (the 4 capture layers the macro windows use)             [e13]
  L1  STA tool: native OpenSTA 3.1.0 (e21) instead of docker 2.6.0 (e13), real0     [e23 measured]
  L2  window convention: macro [t0, last valid + 1 clk] instead of e13's +16 drain   [e23 measured, real0]
  L3  workload: tile flat netlist on the macro's own data (column-group slices)      [e23 measured]
  L4  NV datapath + 25-bit scale fields kept (keepcol netlist; MX run, NV idle)      [e23 measured]
  L5  amortisation 4 -> 32 columns of the per-op shared logic (row drivers / bit-serial
      row shift registers, set-switch control, static RF, ICGs): x 1/8               [DERIVED from e23 hier modules]
  L6  hierarchical synthesis (DPU only; the macro DPU netlist is hierarchical)       [e23 measured hier - keepcol]
  L7  residual to the measured macro logic (macro-scale synthesis, groups not sampled, interactions)  [residual]
  S1  SRAM access energy: Liberty per-access placeholder -> e20 nominal access        [e20]
  S2  SRAM idle-clock + leakage (as-built free-running OpenRAM clocks)              [e20 x cycles]
  W1  boundary wires: tile wire model -> macro wire model                          [EST both]
  end = e22 primary point (real, nominal SRAM, as-built clocks, nominal wires, zero-delay, 40 MHz)

Run: python experiments/e23_tile_macro_reconcile/analyze.py [--e23 <run>] [--paper]
     python experiments/e23_tile_macro_reconcile/analyze.py --recipe td [--paper]   (T2.7: td netlists, fixed INT5;
       inputs e13_power_td / e13_boundary_td / e15_g1_relative_td / T2.7 e21 / e22 / e13_sdf_td, e23_tile_macro_reconcile_td;
       the chain end uses windows real0-3, e22 quotes 9 windows, so the end check differs by the window set)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

sys.path.insert(0, str(REPO_ROOT / "experiments/e21_macro_power"))
import compare as e22  # noqa: E402

R = REPO_ROOT / "results"
E13 = R / "e13_power_rb/20261110T070421270260Z_4ce41ca"
E13B = R / "e13_boundary_rb/20261110T083829218191Z_4ce41ca"
E15 = R / "e15_g1_relative_rb/20261110T143517398932Z_14375a0"
E21 = ["20261110T050824693997Z_f25a5c5", "20261110T151121726136Z_58882a0", "20261110T163649712098Z_58882a0"]
E22 = R / "e22_macro_compare/20261110T163352430523Z_58882a0"
SDF_EXPS = (("e13_sdf_as_e22", "e13_sdf"), ("e13_sdf_rb", "e13_sdf_rb"))
E13_RUNS = None  # td: several per-design e13 runs


def configure(recipe: str) -> None:
    global E13, E13B, E15, E21, E22, AREA, SDF_EXPS, E13_RUNS
    if recipe != "td":
        return
    E13_RUNS = [R / "e13_power_td" / x for x in ("20261114T155933438101Z_5adef9e", "20261114T164825201867Z_be5d1aa",
                                                  "20261114T171607862607Z_be5d1aa")]  # v2 / bs_int5 / dpu (e13_boundary_td sources)
    E13 = E13_RUNS[0]
    E13B = R / "e13_boundary_td/20261114T185712483825Z_be5d1aa"
    E15 = R / "e15_g1_relative_td/20261114T185718218285Z_be5d1aa"
    E22 = R / "e22_macro_compare/20261114T185702027770Z_be5d1aa"
    E21 = yaml.safe_load((E22 / "config.yaml").read_text())["e21_runs"]
    AREA = {"v2_macro_mx": "e30_td_macro_v2_macro", "bs_macro_mx": "e30_td_macro_bs_macro",
            "dpu_macro_mx": "e30_td_macro_dpu_macro_hier", "v2hier_macro_mx": "e30_td_macro_v2_macro_hier"}
    SDF_EXPS = (("e13_sdf_td", "e13_sdf_td"),)


TILE = {"v2": "v2_sram_m16", "bs": "bs_int5", "dpu": "dpu"}
MACRO = {"v2": "v2_macro_mx", "bs": "bs_macro_mx", "dpu": "dpu_macro_mx"}
AREA = {"v2_macro_mx": "e12_macro_synth_v2_macro", "bs_macro_mx": "e12_macro_synth_bs_macro",
        "dpu_macro_mx": "e12_macro_synth_dpu_macro_hier", "v2hier_macro_mx": "e12_macro_synth_v2_macro_hier"}
WIN4 = [f"real{i}" for i in range(4)]
SHARED = ("row_driver", "top_local", "icg")  # per-op shared logic at tile scale (one copy per 4 columns)
RULES = [  # (regex on the de-escaped hierarchical leaf path, class); first match wins
    (r"u_sram$", "sram_liberty"),
    (r"(^|[/.])g_leaf\[\d+\]\.[^/]*u_leaf/", "leaf"),
    (r"(^|[/.])u_tree/", "tree"),
    (r"(^|[/.])g_col\[\d+\]\.u_col/", "column_local"),
    (r"(^|[/.])(g_sram\.)?u_store/", "store"),
    (r"(^|[/.])g_row\[\d+\]\.u_rd/", "row_driver"),
    (r"(^|[/.])u_icg", "icg"),
]


def module_split(gz: Path) -> dict:
    """Leaf-instance power (W) by class and component (internal / switching / leakage) from an e23 per-instance
    OpenSTA report; hierarchical instances (OpenSTA reports their sums too) are excluded."""
    txt = gzip.open(gz, "rt").read()
    rows = re.findall(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\S+)\s*$", txt, re.M)
    names = {r[4] for r in rows}
    hier = {n.rsplit("/", 1)[0] for n in names if "/" in n}
    agg = defaultdict(float)
    for i, sw, lk, _tot, name in rows:
        if name in hier:
            continue
        clean = name.replace("\\", "")
        cls = next((c for pat, c in RULES if re.search(pat, clean)), "top_local")
        agg[f"{cls}_int"] += float(i)
        agg[f"{cls}_sw"] += float(sw)
        agg[f"{cls}_leak"] += float(lk)
    return dict(agg)


def tile_e13() -> dict:
    """{(design, window): {logic, sram, wire, total}} from the e13 rebased runs (zero-delay, docker STA)."""
    pw = {(r["design"], r["window"]): r for d in (E13_RUNS or [E13]) for r in load_csv(d / "power.csv")}
    bd = {(r["design"], r["window"]): r for r in load_csv(E13B / "power_boundary.csv")}
    out = {}
    for (d, w), r in pw.items():
        if not w.startswith("real"):
            continue
        out[(d, w)] = {"logic": float(r["e_logic_fj_per_mac"]), "sram": float(r["e_sram_fj_per_mac"]),
                       "wire": float(bd[(d, w)]["e_wire_fj_per_mac_nom"]),
                       "total": float(bd[(d, w)]["e_total_fj_per_mac_nom"])}
    return out


def macro_e21() -> dict:
    """{(design, window): parts} at the e22 primary point (fJ/MAC)."""
    rows = [r for d in E21 for r in load_csv(R / "e21_macro_power" / d / "power.csv") if r["fmt"] == "mxfp4"
            and r["design"] in AREA]
    spec = yaml.safe_load((REPO_ROOT / "docs/spec/sram_energy.yaml").read_text())["corners"]["nominal"]
    c_ff = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())["layers"]["met2"]["cap_ff_per_um"]
    out = {}
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
        area = json.loads((R / AREA[r["design"]] / "latest/metrics.json").read_text())["total_mm2"]
        wire = e22.wire_fj(r, area, c_ff, "nom", r["design"].startswith("dpu"))
        p = e22.op_energy(r, spec, 40e6, 1.0, "as_built", wire)
        p["logic"] = p["logic_dyn"] + p["logic_leak"]
        out[(r["design"], r["window"])] = p
    return out


def mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e23", default="latest", help="e23_tile_macro_reconcile run dir")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--recipe", default="base", choices=("base", "td"))
    a = ap.parse_args()
    configure(a.recipe)
    sfx = "_td" if a.recipe == "td" else ""
    e23dir = (R / ("e23_tile_macro_reconcile" + sfx) / a.e23).resolve()
    e23 = load_csv(e23dir / "power.csv")
    for r in e23:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    t13 = tile_e13()
    mac = macro_e21()
    g1 = {(r["baseline"], r["delay_model"]): float(r["ee_mean"]) for r in load_csv(E15 / "g1_table.csv")
          if r["point"] == "R256_M16_SRAM"}
    e22m = json.loads((E22 / "metrics.json").read_text())

    def sel(**kw):
        return [r for r in e23 if all((r[k] == v) for k, v in kw.items())]

    E = {}  # E[step][design] = dict(logic, sram, wire) fJ/MAC
    steps = []

    def push(step, label, evidence, parts):
        E[step] = parts
        steps.append((step, label, evidence))

    # A1/A2: tile accounting, e13 numbers
    for step, wins, label in (("A1", [f"real{i}" for i in range(8)], "tile, ratio of means over 8 windows"),
                              ("A2", WIN4, "tile, windows real0-3 (same 4 layers as the macro)")):
        push(step, label, "measured (e13_power_rb, e13_boundary_rb)",
             {k: {q: mean(t13[(TILE[k], w)][q] for w in wins) for q in ("logic", "sram", "wire")} for k in TILE})

    def logic_step(step, label, evidence, fn):
        prev = E[steps[-1][0]]
        push(step, label, evidence, {k: {**prev[k], "logic": fn(k, prev[k]["logic"])} for k in TILE})

    # L1 tool and L2 window convention: factors measured on tile real0 (flat netlists)
    f_tool, f_conv = {}, {}
    for k, d in TILE.items():
        e13_real0 = t13[(d, "real0")]["logic"]
        nat_e13 = sel(variant="flat", design=d, stim="tile", conv="e13")[0]["e_logic_fj_per_mac"]
        nat_mac = sel(variant="flat", design=d, stim="tile", conv="macro")[0]["e_logic_fj_per_mac"]
        f_tool[k], f_conv[k] = nat_e13 / e13_real0, nat_mac / nat_e13
    logic_step("L1", "STA tool: native OpenSTA 3.1.0 (as e21) vs docker 2.6.0 (e13)", "measured, real0 factor",
               lambda k, x: x * f_tool[k])
    logic_step("L2", "window convention: macro [t0, last valid+1] vs e13 +16 drain cycles", "measured, real0 factor",
               lambda k, x: x * f_conv[k])

    def ms(variant, k, field="e_logic_fj_per_mac", group=None):
        rows = sel(variant=variant, design=TILE[k], stim="mslice", conv="macro")
        if group is not None:
            rows = [r for r in rows if r["group"] == group]
        return mean(r[field] for r in rows), rows

    flat_ms = {k: ms("flat", k)[0] for k in TILE}
    keep_ms = {k: ms("keepcol", k)[0] for k in TILE}
    logic_step("L3", "workload: tile flat netlist on the macro's data (8 column-group slices)",
               "measured (e23 flat x mslice)", lambda k, x: flat_ms[k])
    logic_step("L4", "NV datapath kept (keepcol: column ports not constant; MX run)",
               "measured (e23 keepcol x mslice)", lambda k, x: keep_ms[k])
    # L5: shared per-op logic from the hierarchical per-module split (group-0 slices). Internal + leakage energy of
    # the shared cells and the ICG clock nets (which drive shared flops only) are paid once per op -> x 1/8 at
    # 32 columns. Switching energy of the shared cells' output nets is NOT amortised in the primary estimate
    # (those nets fan out to every column, so their load grows with the column count); the bound amortises it too.
    shared, shared_hi, mod, modrows = {}, {}, {}, []
    build = REPO_ROOT / "build/e23_reconcile" / ("td" if sfx else "") / e23dir.name
    for k in TILE:
        acc_m = defaultdict(list)
        for variant in ("hier", "keepcol"):
            for r in sel(variant=variant, design=TILE[k], stim="mslice", conv="macro"):
                if r["group"] != 0:
                    continue
                gz = build / variant / TILE[k] / f"mslice_real{int(r['window'])}_g0" / "inst.rpt.macro.gz"
                kf = r["window_ns"] * 1e-9 / (r["n_ops"] * 128) * 1e15
                ms_ = {c: v * kf for c, v in module_split(gz).items()}
                modrows.append({"variant": variant, "design": k, "window": int(r["window"]), **ms_})
                if variant == "hier":
                    for c, v in ms_.items():
                        acc_m[c].append(v)
        mod[k] = {c: mean(v) for c, v in acc_m.items()}
        g = lambda c: mod[k].get(c, 0.0)  # noqa: E731
        shared[k] = sum(g(f"{c}_int") + g(f"{c}_leak") for c in ("row_driver", "top_local")) + sum(
            g(f"icg_{q}") for q in ("int", "sw", "leak"))
        shared_hi[k] = shared[k] + g("row_driver_sw") + g("top_local_sw")
    logic_step("L5", "amortise per-op shared logic over 32 instead of 4 columns (x 1/8)",
               "derived (e23 hier per-module internal+leak of row drivers / top-level logic + ICG nets)",
               lambda k, x: x - shared[k] * 7 / 8)
    hier_ms0 = {k: mean(r["e_logic_fj_per_mac"] for r in sel(variant="hier", design=TILE[k], stim="mslice", conv="macro"))
                for k in TILE}
    keep_ms0 = {k: mean(r["e_logic_fj_per_mac"] for r in sel(variant="keepcol", design=TILE[k], stim="mslice", conv="macro")
                        if r["group"] == 0) for k in TILE}
    d_hier = {k: hier_ms0[k] - keep_ms0[k] for k in TILE}
    logic_step("L6", "hierarchical synthesis (DPU macro netlist only)", "measured (e23 hier - keepcol, group-0 slices)",
               lambda k, x: x + (d_hier[k] if k == "dpu" else 0.0))
    macro_logic = {k: mean(mac[(MACRO[k], w)]["logic"] for w in WIN4) for k in TILE}
    logic_step("L7", "residual to the measured macro logic", "residual (unexplained)", lambda k, x: macro_logic[k])
    resid = {k: E["L7"][k]["logic"] - E["L6"][k]["logic"] for k in TILE}
    # SRAM and wires
    prev = E["L7"]
    acc = {k: mean(mac[(MACRO[k], w)]["sram_access"] for w in WIN4) for k in TILE}
    idle = {k: mean(mac[(MACRO[k], w)]["sram_idle_clock"] + mac[(MACRO[k], w)]["sram_leak"] for w in WIN4) for k in TILE}
    push("S1", "SRAM access: Liberty per-access placeholder -> D33 nominal access energy", "derived (e20)",
         {k: {**prev[k], "sram": acc[k]} for k in TILE})
    prev = E["S1"]
    push("S2", "SRAM idle clock + leakage, as-built free-running OpenRAM clocks (8 macros x cycles)",
         "derived (e20/D33 x simulated cycles)", {k: {**prev[k], "sram": acc[k] + idle[k]} for k in TILE})
    prev = E["S2"]
    mwire = {k: mean(mac[(MACRO[k], w)]["wire"] for w in WIN4) for k in TILE}
    push("W1", "boundary wires: tile model -> macro model", "[EST] both", {k: {**prev[k], "wire": mwire[k]} for k in TILE})

    # trajectory table
    traj = [{"step": "A0", "label": "tile G1 as reported (mean of per-window ratios)", "evidence": "e15",
             "v2_fj": "", "bs_fj": "", "dpu_fj": "", "ratio_v2_over_int5": g1[("bs_int5", "zero")],
             "ratio_v2_over_dpu": g1[("dpu", "zero")]}]
    for step, label, ev in steps:
        tot = {k: sum(E[step][k].values()) for k in TILE}
        traj.append({"step": step, "label": label, "evidence": ev, "v2_fj": tot["v2"], "bs_fj": tot["bs"],
                     "dpu_fj": tot["dpu"], "ratio_v2_over_int5": tot["bs"] / tot["v2"],
                     "ratio_v2_over_dpu": tot["dpu"] / tot["v2"]})
    for i in range(1, len(traj)):
        for kk in ("int5", "dpu"):
            key = f"ratio_v2_over_{kk}"
            traj[i][f"dlog_{kk}"] = round(math.log(traj[i][key] / traj[i - 1][key]), 6)
    end = {"int5": e22m["primary_ee_ratio_v2_over_bs"], "dpu": e22m["primary_ee_ratio_v2_over_dpu"]}
    # one-at-a-time impact of each step on the A2 endpoint (order-independent view)
    base = {k: sum(E["A2"][k].values()) for k in TILE}
    ids = [s for s, *_ in steps]
    alone = []
    for i, s in enumerate(ids):
        if i <= ids.index("A2"):
            continue
        p = ids[i - 1]
        dlt = {k: sum(E[s][k].values()) - sum(E[p][k].values()) for k in TILE}
        alone.append({"step": s, **{f"d_{k}_fj": dlt[k] for k in TILE},
                      "ratio_int5_if_alone": (base["bs"] + dlt["bs"]) / (base["v2"] + dlt["v2"]),
                      "ratio_dpu_if_alone": (base["dpu"] + dlt["dpu"]) / (base["v2"] + dlt["v2"])})
    # flat-equivalent DPU macro (measured tile hier bias instead of the 5 % assumption)
    dpu_flat_eq = {w: mac[("dpu_macro_mx", w)]["total"] - d_hier["dpu"] for w in WIN4}
    v2_tot = {w: mac[("v2_macro_mx", w)]["total"] for w in WIN4}
    ratio_dpu_flat_eq = mean(dpu_flat_eq.values()) / mean(v2_tot.values())
    # glitch-corrected macro ratios [EST transfer of tile SDF/zero factors to the dynamic macro logic], with the
    # pre-rebase factors e22 used (results/e13_sdf) and the rebased factors (results/e13_sdf_rb)
    glitch = {}
    for tag, exp in SDF_EXPS:
        f = json.loads((R / exp / "latest/metrics.json").read_text())
        gf = {k: f[f"{TILE[k]}_sdf_over_zero"] for k in TILE}

        def tot(k, w, hier_corr=0.0):
            p = mac[(MACRO[k], w)]
            return (p["logic_dyn"] - hier_corr) * gf[k] + p["logic_leak"] + p["wire"] + p["sram_access"] \
                + p["sram_idle_clock"] + p["sram_leak"]
        v2g = mean(tot("v2", w) for w in WIN4)
        glitch[tag] = {"factors": gf, "v2_over_int5": mean(tot("bs", w) for w in WIN4) / v2g,
                       "v2_over_dpu_hier": mean(tot("dpu", w) for w in WIN4) / v2g,
                       "v2_over_dpu_flat_equivalent": mean(tot("dpu", w, d_hier["dpu"]) for w in WIN4) / v2g}
    v2_hier_macro = mac[("v2hier_macro_mx", "real0")]["logic"] - mac[("v2_macro_mx", "real0")]["logic"]
    cfg = {"e13": str(E13.relative_to(REPO_ROOT)), "e13_boundary": str(E13B.relative_to(REPO_ROOT)),
           "e15": str(E15.relative_to(REPO_ROOT)), "e21": E21, "e22": str(E22.relative_to(REPO_ROOT)),
           "e23_runs": str(e23dir.relative_to(REPO_ROOT)), "recipe": a.recipe, "e13_runs": [str(d.relative_to(REPO_ROOT)) for d in (E13_RUNS or [E13])],
           "area": AREA, "sdf_exps": [e for _, e in SDF_EXPS], "shared_classes": SHARED, "windows": WIN4}
    with start_run("e23_reconcile_analysis" + sfx, cfg, paper=a.paper) as run:
        run.save_csv("trajectory.csv", traj)
        run.save_csv("step_deltas.csv", alone)
        comp = []
        for step, *_ in steps:
            for k in TILE:
                comp.append({"step": step, "design": k, **E[step][k], "total": sum(E[step][k].values())})
        run.save_csv("components.csv", comp)
        run.save_csv("tile_modules_group0.csv", modrows)
        m = {"end_check_ratio_v2_over_int5": {"chain": traj[-1]["ratio_v2_over_int5"], "e22": end["int5"]},
             "end_check_ratio_v2_over_dpu": {"chain": traj[-1]["ratio_v2_over_dpu"], "e22": end["dpu"]},
             "f_tool_native_over_docker_real0": f_tool, "f_conv_macro_over_e13_real0": f_conv,
             "logic_flat_mslice": flat_ms, "logic_keepcol_mslice": keep_ms, "nv_datapath_fj": {k: keep_ms[k] - flat_ms[k] for k in TILE},
             "shared_logic_tile_fj": shared, "amortisation_fj": {k: -shared[k] * 7 / 8 for k in TILE},
             "amortisation_bound_fj_incl_switching": {k: -shared_hi[k] * 7 / 8 for k in TILE},
             "hier_minus_keepcol_tile_fj": d_hier, "hier_over_keepcol_tile": {k: hier_ms0[k] / keep_ms0[k] for k in TILE},
             "v2_macro_hier_minus_flat_real0_fj": v2_hier_macro,
             "residual_logic_fj": resid, "residual_logic_frac_of_macro_logic": {k: resid[k] / macro_logic[k] for k in TILE},
             "macro_logic_fj": macro_logic,
             "ratio_v2_over_dpu_flat_equivalent_measured_bias": ratio_dpu_flat_eq,
             "dpu_macro_flat_equivalent_fj": mean(dpu_flat_eq.values()), "glitch_corrected_macro_EST": glitch}
        run.log_metrics(m)
        for t in traj:
            print(f"{t['step']:3s} {t['ratio_v2_over_int5']:.3f} {t['ratio_v2_over_dpu']:.3f}  "
                  + (f"v2 {t['v2_fj']:.0f} bs {t['bs_fj']:.0f} dpu {t['dpu_fj']:.0f}  " if t["v2_fj"] != "" else "")
                  + t["label"])
        print(json.dumps(m, indent=1, default=float))


if __name__ == "__main__":
    main()
