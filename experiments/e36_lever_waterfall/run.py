"""e36 - T5.5: lever waterfall (C3).  Separates the leaf (number-format / decoder) lever from the bit-serial -> bit-parallel lever.

Pure analysis (CPU, seconds): no new simulation.  Inputs are the pinned same-window tile energies of T5.3 (e34, recipe td):
v2_sram_m16 (D1), bs4_int4sm (D2 INT4-SM, LOSSY), bs_int5 (exact INT5 control), dpu (generic-leaf digital MAC, same local
stores), on the 8 real e13_power_rb windows, gate + boundary wires (nominal band).

Waterfall (energy per MAC, each step = E(previous)/E(next) = EE(next)/EE(previous), paired per window):
  S1  exact INT5 bit-serial  -> INT4-SM bit-serial   operand-width/element-grid lever (5 -> 3 cycles/op); LOSSY/rescaled, not iso-accuracy
  S2  INT4-SM bit-serial     -> DPU bit-parallel      bit-serial -> bit-parallel lever, generic integer-style leaf
  S3  DPU generic leaf       -> v2 FP4 AND-OR leaf     the leaf (number-format / decoder) lever
  end-to-end = E(INT5)/E(v2) = EE(v2)/EE(INT5).  Acceptance: |prod(S)/end-to-end - 1| <= 2 %.
Not in the product: MIDAS-inspired behavioural step (no same-boundary RTL; behavioural only), the NV-gated step, and any
CIM / organisation lever: DPU has the same local stores as v2, so NO organisation effect is measured (needs a separate experiment).

Two delay models: zero-delay (measured GLS) and SDF-corrected (per-design scalar glitch factor from 2 SDF windows, [EST]).
The windows' energies are reconstructed exactly as e34 --summary does; the script asserts the reconstruction matches e34.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from cimfp4.stimulus.writers import load_program
from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

R = REPO_ROOT / "results"
RUNS = {  # pinned inputs (T5.3 summary 20261115T011019844029Z_76b624d, recipe td)
    "e34_summary": "e34_d2_tile/20261115T011019844029Z_76b624d",
    "e34_power": "e34_d2_tile_power/20261115T001509951613Z_76b624d",
    "e34_sdf": "e34_d2_tile_sdf/20261115T004133086514Z_76b624d",
    "e34_synth": "e34_d2_tile_synth/20261115T000548088218Z_76b624d",
    "e13_boundary": "e13_boundary_td/20261114T185712483825Z_be5d1aa",
    "e13_sdf": "e13_sdf_td/20261114T073508435587Z_3a17c1d",
}
REAL = [f"real{i}" for i in range(8)]
V2, I4, I5, DPU = "v2_sram_m16", "bs4_int4sm", "bs_int5", "dpu"
STEPS = [
    ("S1", "INT5 bit-serial -> INT4-SM bit-serial (element grid, 5->3 cycles; LOSSY)", I5, I4, "measured"),
    ("S2", "INT4-SM bit-serial -> DPU bit-parallel (generic leaf)", I4, DPU, "measured"),
    ("S3", "DPU generic leaf -> v2 FP4 AND-OR leaf (leaf lever)", DPU, V2, "measured"),
]
TOL = 0.02


def _wire_fn():
    p = REPO_ROOT / "experiments/e34_d2_tile/run.py"
    src = p.read_text()
    start = src.index("def wire_energy_nom")
    ns: dict = {"np": np, "REPO_ROOT": REPO_ROOT}
    exec(src[start:src.index("\n\n\n", start)], ns)  # the e34 boundary-wire model verbatim (no e34 module side effects)
    return ns["wire_energy_nom"], hashlib.sha256(src.encode()).hexdigest()


def energies(programs: Path) -> tuple[dict, dict, dict]:
    wire, e34_sha = _wire_fn()
    synth = next(r for r in load_csv(R / RUNS["e34_synth"] / "synth.csv") if r["recipe"] == "td")
    area_mm2 = float(synth["total_area_um2"]) / 1e6
    mine = {r["window"]: r for r in load_csv(R / RUNS["e34_power"] / "power.csv") if r["recipe"] == "td"}
    assert all(int(mine[w]["mismatches"]) == 0 for w in REAL)
    ref = {(r["design"], r["window"]): r for r in load_csv(R / RUNS["e13_boundary"] / "power_boundary.csv")}
    e = {d: np.array([float(ref[(d, w)]["e_total_fj_per_mac_nom"]) for w in REAL]) for d in (V2, I5, DPU)}
    hashes = {}
    vals = []
    for w in REAL:
        f = programs / f"{w}.npz"
        hashes[w] = hashlib.sha256(f.read_bytes()).hexdigest()
        vals.append(float(mine[w]["e_total_fj_per_mac"]) + wire(load_program(f), area_mm2) / float(mine[w]["macs"]))
    e[I4] = np.array(vals)
    sdf = {k.replace("_sdf_over_zero", ""): float(v) for k, v in json.loads((R / RUNS["e13_sdf"] / "metrics.json").read_text()).items()
           if k.endswith("_sdf_over_zero")}
    sdf[I4] = float(json.loads((R / RUNS["e34_sdf"] / "metrics.json").read_text())["sdf_over_zero"])
    # reconstruction check against the pinned e34 summary
    des = {r["design"]: r for r in load_csv(R / RUNS["e34_summary"] / "designs.csv")}
    for d in (V2, I4, I5, DPU):
        m = float(des[d]["e_nom_fj_per_mac_real_mean"])
        assert abs(e[d].mean() / m - 1) < 1e-9, f"reconstruction mismatch {d}: {e[d].mean()} vs {m}"
        assert abs(sdf[d] / float(des[d]["sdf_over_zero"]) - 1) < 1e-12
    return e, sdf, {"programs_sha256": hashes, "e34_run_py_sha256": e34_sha}


def boot(r: np.ndarray, rng, n=2000) -> tuple[float, float]:
    idx = rng.integers(0, len(r), (n, len(r)))
    b = r[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=Path, default=REPO_ROOT / "build/e34/programs",
                    help="e34 INT4-SM program dir (gitignored build product; sha256 recorded)")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    e, sdf, prov = energies(a.programs.resolve())
    rng = seed_all(0)
    cfg = {"inputs": RUNS, "windows": REAL, "steps": [s[:2] for s in STEPS], "tolerance": TOL, "bootstrap": "paired by window, 2000, seed 0",
           "ratio_statistic": "mean over windows of per-window energy ratio (as e34/e15)", "programs_dir": str(a.programs), **prov}
    rows, m = [], {}
    with start_run("e36_lever_waterfall", cfg, paper=a.paper) as run:
        for model in ("zero", "sdf_corrected"):
            f = {d: (sdf[d] if model == "sdf_corrected" else 1.0) for d in e}
            E = {d: e[d] * f[d] for d in e}
            prod_w = np.ones(len(REAL))
            prod_mean = 1.0
            cum = 1.0
            for sid, label, a_, b_, lvl in STEPS:
                r = E[a_] / E[b_]
                lo, hi = boot(r, rng)
                prod_w *= r
                prod_mean *= r.mean()
                cum *= r.mean()
                rows.append({"delay_model": model, "step": sid, "lever": label, "from": a_, "to": b_, "ratio_mean": r.mean(),
                             "ci_lo": lo, "ci_hi": hi, "cumulative": cum, "log_share": None,
                             "evidence": lvl if model == "zero" else "measured x [EST] glitch factor"})
                m[f"{sid}_{model}"], m[f"{sid}_{model}_ci_lo"], m[f"{sid}_{model}_ci_hi"] = r.mean(), lo, hi
            e2e = E[I5] / E[V2]
            lo, hi = boot(e2e, rng)
            ln = math.log(e2e.mean())
            for row in rows[-3:]:
                row["log_share"] = math.log(row["ratio_mean"]) / ln
            # bit-serial->bit-parallel lever measured directly on the exact control (S1*S2 path cross-check)
            bp = E[I5] / E[DPU]
            rows.append({"delay_model": model, "step": "E2E", "lever": "INT5 -> v2 end-to-end", "from": I5, "to": V2,
                         "ratio_mean": e2e.mean(), "ci_lo": lo, "ci_hi": hi, "cumulative": e2e.mean(), "log_share": 1.0, "evidence": "measured"})
            dev = prod_mean / e2e.mean() - 1
            m |= {f"e2e_{model}": e2e.mean(), f"e2e_{model}_ci_lo": lo, f"e2e_{model}_ci_hi": hi, f"product_{model}": prod_mean,
                  f"product_over_e2e_minus1_{model}": dev, f"pass_{model}": bool(abs(dev) <= TOL),
                  f"int5_over_dpu_{model}": bp.mean(), f"leaf_log_share_{model}": rows[-2]["log_share"],
                  f"exact_identity_per_window_{model}": float(np.max(np.abs(prod_w / e2e - 1)))}
        run.save_csv("waterfall.csv", rows)
        m["sdf_factors_est"] = sdf
        m["pass"] = bool(m["pass_zero"] and m["pass_sdf_corrected"])
        run.log_metrics(m)
        for r in rows:
            print(f"{r['delay_model']:14s} {r['step']:4s} {r['ratio_mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
                  f"cum {r['cumulative']:.3f} share {r['log_share']:+.2f}  {r['lever']}")
        print(json.dumps({k: v for k, v in m.items() if k.startswith(("product", "e2e_", "pass"))}, indent=1))
        assert m["pass"], "waterfall product does not reconcile within 2 %"


if __name__ == "__main__":
    main()
