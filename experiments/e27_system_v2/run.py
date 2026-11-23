"""e27 / T2.2 - system model rerun (e25_system v2): flat-equivalent DPU, MX and NV modes, e28 quantizer bundle,
idle-cycle logic in the base case, gated SRAM clock primary with free-running side by side.

Analytic post-processing of measured per-op costs (cimfp4.sysmodel.system_model). Every system-level number is a
[DERIVED] model output; reload/DRAM/write/idle-logic/gated-clock/wire terms are [EST]. No decode efficiency claim:
decode rows are reported as a limitation table only.
Run: python experiments/e27_system_v2/run.py [--paper]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from dataclasses import asdict

import yaml

from cimfp4.sysmodel import system_model as sm
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

_s = importlib.util.spec_from_file_location("e25run", REPO_ROOT / "experiments/e25_system/run.py")
E25 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E25)

import yaml  # noqa: E402

_y = yaml.safe_load((REPO_ROOT / "experiments/configs/paper_runs.yaml").read_text())
_PINS = {**_y["numbers"], **_y["figures_tables"]}
E22_RUN = _PINS["e22_macro_compare"]  # pinned e22 (T2.11: includes the measured flat DPU dpuflat_*)
T210_RUN = _PINS["t210_sram_gating"]
E28_RUN = "20261114T234120941760Z_3c20927"
PREFILL = (512, 1024, 2048)
BATCHED = (8, 32)  # batched decode / verify rows (compute-bound batched operation)
DECODE_LIMIT = (1,)  # batch 8 appears under "batched" (reload-bound, see R*)
# T2.11: the measured flat-Yosys DPU replaces the [DERIVED] flat-eq. (hier / 1.191) DPU in the paper pairs
MX = ("v2", "bs", "dpu", "dpuflat")
NV = ("v2nv", "dpunv", "dpuflatnv")
PAIRS = [("MX", "v2/INT5", "v2", "bs"), ("MX", "v2/DPU flat [MEASURED]", "v2", "dpuflat"), ("MX", "v2/DPU hier", "v2", "dpu"),
         ("NV", "v2/DPU flat [MEASURED]", "v2nv", "dpuflatnv"), ("NV", "v2/DPU hier", "v2nv", "dpunv")]
# macro-level references (e22 as-built, T2.10 gated) used for the 1 % reproduction check
MACRO_KEY = {"v2/INT5": "v2_mx_over_bs_mx", "v2/DPU flat [MEASURED]": "v2_{f}_over_dpuflat_{f}", "v2/DPU hier": "v2_{f}_over_dpu_{f}"}


def tok_rows(mode, n):
    return n  # prefill: n tokens share each weight set; batched/decode: batch n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    R = REPO_ROOT / "results"
    for exp, rid in (("e22_macro_compare", E22_RUN), ("t210_sram_gating", T210_RUN)):
        got = (R / exp / "latest").resolve().name
        assert got == rid, f"{exp}/latest is {got}, expected pinned input {rid}"
    inp = sm.load_inputs(REPO_ROOT, nv=True, bundle_run=E28_RUN, with_bundle=True)
    assert set(MX + NV) <= set(inp.designs), inp.designs.keys()
    t210 = {(r["pair"], r["sram_clock"], r["variant"]): float(r["ee_ratio"])
            for r in __import__("csv").DictReader((R / "t210_sram_gating" / T210_RUN / "rows.csv").open())}
    cfgs, meta = {}, {}
    for m in E25.HF:
        c, snap, sha = E25.hf_cfg(m)
        cfgs[m] = c
        meta[m] = {"snapshot": snap, "config_sha256_12": sha}
    base = sm.Assumptions(idle_logic=True, sram_clock="gated", bundle="per_token")
    scen = [("prefill", n) for n in PREFILL] + [("batched", n) for n in BATCHED] + [("decode_limitation", n) for n in DECODE_LIMIT]
    cfg_run = {"inputs": {"e22_macro_compare": E22_RUN, "t210_sram_gating": T210_RUN, "e28_quant_summary": E28_RUN,
                          **inp.provenance}, "bundle_fj": inp.bundle, "designs": {k: asdict(v) for k, v in inp.designs.items()},
               "base": asdict(base), "scenarios": scen, "models": meta,
               "grid": "bundle{none,per_token,per_op} x sram_clock{gated,as_built} x idle_logic{1,0} at 64 macros/40 MHz/nominal; "
                       "plus n_macros{16,256}, sram{low,high}, post_route at the base point",
               "basis": "primary bundle = e28 per-token reuse basis (activation block quantised once per token row, reused "
                        "over N/32 column groups; HIER per useful MAC); per_op = e28 per-MAC table (conservative)"}
    rows = []
    with start_run("e27_system_v2", cfg_run, paper=args.paper) as run:
        def ev(model, mode, n, dn, a):
            r = sm.evaluate(cfgs[model], n, inp.designs[dn], inp.sram, a, inp.bundle)
            rw = E25.row(model, mode, n, dn, a, r)
            rw.update({"fmt": inp.designs[dn].fmt, "bundle": a.bundle,
                       "nJ_per_token": r["energy_j"] / n * 1e9})
            rows.append(rw)
            return rw
        for model, (mode, n), dn in itertools.product(cfgs, scen, MX + NV):
            for bnd, clk, il in itertools.product(("none", "per_token", "per_op"), ("gated", "as_built"), (True, False)):
                ev(model, mode, n, dn, sm.with_(base, bundle=bnd, sram_clock=clk, idle_logic=il))
            for kw in ({"n_macros": 16}, {"n_macros": 256}, {"sram_corner": "low"}, {"sram_corner": "high"}, {"post_route": True}):
                ev(model, mode, n, dn, sm.with_(base, **kw))
        run.save_csv("grid.csv", rows)

        def pick(model, mode, n, dn, **kw):
            want = {**{"n_macros": 64, "sram": "nominal", "post_route": False, "bundle": "per_token", "sram_clock": "gated",
                       "idle_logic": True}, **kw}
            hit = [r for r in rows if r["model"] == model and r["mode"] == mode and r["seq_or_batch"] == n and r["design"] == dn
                   and all(r[k] == v for k, v in want.items())]
            assert len(hit) == 1, (model, mode, n, dn, kw, len(hit))
            return hit[0]

        ratios, checks = [], []
        for model, (mode, n), (fmt, pname, a, b) in itertools.product(cfgs, scen, PAIRS):
            for bnd, clk in itertools.product(("none", "per_token", "per_op"), ("gated", "as_built")):
                for il in (True, False):
                    ra, rb = pick(model, mode, n, a, bundle=bnd, sram_clock=clk, idle_logic=il), \
                        pick(model, mode, n, b, bundle=bnd, sram_clock=clk, idle_logic=il)
                    ratios.append({"model": model, "mode": mode, "n": n, "fmt": fmt, "pair": pname, "bundle": bnd,
                                   "sram_clock": clk, "idle_logic": il, "ee_ratio": rb["uJ_per_token"] / ra["uJ_per_token"]})
        run.save_csv("ratios.csv", ratios)

        # Acceptance: prefill 2048, no bundle, idle-logic off -> macro ratios (e22 as-built / T2.10 gated) within 1 %
        for model, (fmt, pname, a, b), clk in itertools.product(cfgs, PAIRS, ("as_built", "gated")):
            key = MACRO_KEY[pname].format(f=fmt.lower())
            ref = t210[(key, clk, "primary")]
            sysr = [r for r in ratios if r["model"] == model and r["mode"] == "prefill" and r["n"] == 2048 and r["fmt"] == fmt
                    and r["pair"] == pname and r["bundle"] == "none" and r["sram_clock"] == clk and not r["idle_logic"]][0]
            checks.append({"model": model, "fmt": fmt, "pair": pname, "sram_clock": clk, "macro_ref": ref,
                           "system_prefill2048": sysr["ee_ratio"], "rel_dev": sysr["ee_ratio"] / ref - 1})
        run.save_csv("macro_reproduction_check.csv", checks)
        worst = max(abs(c["rel_dev"]) for c in checks)

        def rat(model, mode, n, fmt, pname, bnd, clk="gated", il=True):
            return [r["ee_ratio"] for r in ratios if r["model"] == model and r["mode"] == mode and r["n"] == n and r["fmt"] == fmt
                    and r["pair"] == pname and r["bundle"] == bnd and r["sram_clock"] == clk and r["idle_logic"] == il][0]

        L = ["# e27 / T2.2 system model v2 - base-point tables [DERIVED analytic model; [EST] terms listed below]",
             "Base: 64 macros, 40 MHz, D33 nominal SRAM, **gated SRAM clock primary [EST design change]** (free-running as-built "
             "side by side), zero-delay measured logic + [EST] wires, **idle-cycle logic ON** [EST], lm_head excluded, "
             "**e28 bundle on the per-token reuse basis** (quantizer+anchor+gated tile buffer once per activation block per token "
             "row, HIER reducer per MAC; [EST] for tile buffer and HIER). INT5 is an MX-only control. "
             "Decode rows are a limitation, not an efficiency claim.", "",
             f"Macro reproduction check (prefill 2048, no bundle, idle logic off, vs e22/T2.10 macro ratios): worst |dev| = {worst*100:.2f} %", ""]
        for model in cfgs:
            L += [f"## {model}: energy per token (uJ/token) [DERIVED]",
                  "| mode | n | design | fmt | gated, +bundle (primary) | free-running, +bundle | gated, no bundle | free-running, no bundle | gated, per-op bundle | bundle share | us/token | TOPS/W (primary) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for (mode, n), dn in itertools.product(scen, MX + NV):
                p = pick(model, mode, n, dn)
                fr = pick(model, mode, n, dn, sram_clock="as_built")
                g0, f0 = pick(model, mode, n, dn, bundle="none"), pick(model, mode, n, dn, bundle="none", sram_clock="as_built")
                po = pick(model, mode, n, dn, bundle="per_op")
                sh = p["share_quant_anchor_buf"] + p["share_hier"]
                L.append(f"| {mode} | {n} | {inp.designs[dn].name} | {p['fmt'].upper()} | {p['uJ_per_token']:.1f} | {fr['uJ_per_token']:.1f} | "
                         f"{g0['uJ_per_token']:.1f} | {f0['uJ_per_token']:.1f} | {po['uJ_per_token']:.1f} | {sh*100:.1f} % | "
                         f"{p['latency_us_per_token']:.2f} | {p['tops_per_w']:.3f} |")
            L += ["", f"## {model}: energy-efficiency ratios (other / v2 energy per token) [DERIVED]",
                  "| mode | n | fmt | pair | gated +bundle (primary) | gated no bundle | free-running +bundle | free-running no bundle | gated per-op bundle |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for (mode, n), (fmt, pname, a, b) in itertools.product(scen, PAIRS):
                L.append(f"| {mode} | {n} | {fmt} | {pname} | {rat(model, mode, n, fmt, pname, 'per_token'):.3f} | "
                         f"{rat(model, mode, n, fmt, pname, 'none'):.3f} | {rat(model, mode, n, fmt, pname, 'per_token', 'as_built'):.3f} | "
                         f"{rat(model, mode, n, fmt, pname, 'none', 'as_built'):.3f} | {rat(model, mode, n, fmt, pname, 'per_op'):.3f} |")
            L.append("")
        (run.dir / "tables.md").write_text("\n".join(L))
        met = {"n_rows": len(rows), "macro_reproduction_worst_rel_dev": worst, "macro_reproduction_within_1pct": worst <= 0.01,
               "basis": "per_token",
               "base": {k: v for k, v in asdict(base).items()}}
        for model in cfgs:
            mk = model.split("-")[0].lower().replace(".", "")
            for (mode, n), dn in itertools.product([s for s in scen if s[0] != "decode_limitation"], MX + NV):
                met[f"{mk}_{mode}{n}_{dn}_uJ_per_token"] = pick(model, mode, n, dn)["uJ_per_token"]
            for (mode, n), (fmt, pname, a, b) in itertools.product([s for s in scen if s[0] != "decode_limitation"], PAIRS):
                tag = f"{mk}_{mode}{n}_{fmt.lower()}_{a}_over_{b}"
                met[f"{tag}_ee_primary"] = rat(model, mode, n, fmt, pname, "per_token")
                met[f"{tag}_ee_nobundle"] = rat(model, mode, n, fmt, pname, "none")
                met[f"{tag}_ee_freerun_bundle"] = rat(model, mode, n, fmt, pname, "per_token", "as_built")
        run.log_metrics(met)
        print("\n".join(L))
        print(run.dir)


if __name__ == "__main__":
    main()
