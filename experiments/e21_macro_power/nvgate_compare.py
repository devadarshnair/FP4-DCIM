"""T4.2: macro-level NV_GATE=1 vs as-built (NV_GATE=0) logic energy, the idle-NV tax before/after, and one
mixed-format stream. Results only (no simulation): reads e21_macro_power power.csv rows.

Pairs (as-built -> NV_GATE=1, same netlist flow and the same T2.7 program windows):
  v2      v2_macro_*       -> v2nvg1_macro_*       (flat Yosys td)
  v2hier  v2hier_macro_*   -> v2hiernvg1_macro_*   (hier Yosys td; as-built has 2 windows -> paired on those only)
  dpu     dpu_macro_*      -> dpunvg1_macro_*      (hier Yosys td, the T2.7 DPU)
  dpuflat dpuflat_macro_*  -> dpuflatnvg1_macro_*  (flat Yosys td, T2.11)

Quantities (logic energy e_logic_fj_per_mac; SRAM is identical for both settings and composed elsewhere):
  gate_ratio_<f>     mean over windows of E_nvg1(w) / E_asbuilt(w), f = mx | nv | mixed  (bootstrap 95 % CI)
  tax_removed_mx     mean of E_asbuilt,mx(w) - E_nvg1,mx(w) (fJ/MAC): the MX-mode energy the gate removes [MEASURED]
  idle_nv_tax_<g>    idle-NV tax in MX mode before (g = nvg0) and after (nvg1) gating, relative to an MX-only (NV-folded)
                     macro. No MX-only macro netlist exists, so E_mxonly = E_asbuilt,mx / (1 + T_tile,nvg0) with the
                     tile-level measured tax T_tile,nvg0 (e30_td_power): [DERIVED]; tax_after = E_nvg1,mx/E_mxonly - 1.
  nv_over_mx         mean E_nv(w)/E_mx(w) per setting (the active NV cost)
  mixed_overhead     E_mixed(w) / ((n_mx E_mx(w) + n_nv E_nv(w)) / n) - 1 per setting: the cost of interleaving MX and NV
                     ops (op_nv switching, both formats' sets resident, extra set switches) over running them separately.

  python experiments/e21_macro_power/nvgate_compare.py [--runs <run id> ...] [--paper]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import RESULTS, start_run

PAIRS = {"v2": ("v2", "v2nvg1"), "v2hier": ("v2hier", "v2hiernvg1"), "dpu": ("dpu", "dpunvg1"),
         "dpuflat": ("dpuflat", "dpuflatnvg1")}
FMTS = {"mx": "mx", "nv": "nv", "mixed": "mixed"}
TILE_TAX = "results/e30_td_power/20261113T205342783613Z_65f5076"


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(rows: list[dict]) -> dict:
    """{(design, window): row}; a later row (later run id, rows sorted by run) replaces an earlier one."""
    return {(r["design"], r["window"]): r for r in rows}


def analyse(t: dict, tile_tax_nvg0: float, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    E = lambda d, w: float(t[(d, w)]["e_logic_fj_per_mac"]) if (d, w) in t else None
    wins = lambda d: {w for dd, w in t if dd == d}
    out = {}
    for name, (a, g) in PAIRS.items():
        res = {}
        for f in FMTS:
            da, dg = f"{a}_macro_{f}", f"{g}_macro_{f}"
            ws = sorted(wins(da) & wins(dg))
            if not ws:
                continue
            r = [E(dg, w) / E(da, w) for w in ws]
            res[f"gate_ratio_{f}"] = float(np.mean(r))
            res[f"gate_ratio_{f}_ci"] = boot(r, rng)
            res[f"e_asbuilt_{f}"] = float(np.mean([E(da, w) for w in ws]))
            res[f"e_nvg1_{f}"] = float(np.mean([E(dg, w) for w in ws]))
            res[f"windows_{f}"] = ws
            if f == "mx":
                d = [E(da, w) - E(dg, w) for w in ws]
                res["tax_removed_mx_fj_per_mac"] = float(np.mean(d))
                res["tax_removed_mx_ci"] = boot(d, rng)
                mxonly = [E(da, w) / (1 + tile_tax_nvg0) for w in ws]
                res["idle_nv_tax_nvg0_derived"] = tile_tax_nvg0
                res["idle_nv_tax_nvg1_derived"] = float(np.mean([E(dg, w) / m - 1 for w, m in zip(ws, mxonly)]))
        for lab, p in (("asbuilt", a), ("nvg1", g)):
            ws = sorted(wins(f"{p}_macro_mx") & wins(f"{p}_macro_nv"))
            if ws:
                res[f"nv_over_mx_{lab}"] = float(np.mean([E(f"{p}_macro_nv", w) / E(f"{p}_macro_mx", w) for w in ws]))
            ws = sorted(wins(f"{p}_macro_mixed") & wins(f"{p}_macro_mx") & wins(f"{p}_macro_nv"))
            if ws:
                o = []
                for w in ws:
                    rm = t[(f"{p}_macro_mixed", w)]
                    n, nnv = int(rm["n_ops"]), int(rm["nv_ops"])
                    ref = ((n - nnv) * E(f"{p}_macro_mx", w) + nnv * E(f"{p}_macro_nv", w)) / n
                    o.append(E(f"{p}_macro_mixed", w) / ref - 1)
                res[f"mixed_overhead_{lab}"] = float(np.mean(o))
                res[f"mixed_overhead_{lab}_ci"] = boot(o, rng)
                res[f"mixed_windows_{lab}"] = ws
        if res:
            out[name] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="e21 run ids (default: every run with a power.csv)")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    base = RESULTS / "e21_macro_power"
    runs = a.runs or sorted(p.parent.name for p in base.glob("*/power.csv") if p.parent.name != "latest")
    rows = []
    for r in sorted(runs):
        for row in load_csv(base / r / "power.csv"):
            rows.append({**row, "run": r})
    tile = json.loads((REPO_ROOT / TILE_TAX / "metrics.json").read_text())["td_idle_nv_tax_nvg0"]
    t = table(rows)
    res = analyse(t, tile)
    cfg = {"runs": runs, "tile_tax_source": TILE_TAX, "tile_idle_nv_tax_nvg0": tile,
           "rows_used": sorted({(d, w, t[(d, w)]["run"]) for d, w in t if any(d.startswith(p) for p in
                                ("v2", "dpu"))})}
    with start_run("e21_nvgate_compare", cfg, paper=a.paper) as run:
        (run.dir / "nvgate.json").write_text(json.dumps(res, indent=1))
        flat = {f"{k}_{q}": v for k, d in res.items() for q, v in d.items() if isinstance(v, float)}
        run.log_metrics(flat)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
