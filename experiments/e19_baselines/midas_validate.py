"""e19 - MIDAS-like baseline fidelity (WP-37).

Activity from the L2 MIDAS-like macro on real MXFP4 tiles (Qwen2.5-0.5B captures: 3 layers × 7 projections,
random 16-column weight sets × 256 tokens), energies/areas from the sky130 Liberty (power/midas_model.py, no
fitting). Compared with MIDAS Fig. 21 shares (docs/spec/midas_targets.yaml): label "MIDAS-like" iff every
component is within ±5 pp in power and area, else "MIDAS-inspired" (metrics.json). Fig. 13/22 ablation steps:
todo (needs the pre-aligned 2C FP-CIM baseline model). MXFP6 MEPD latency check: todo (needs raw captures).

Run: python experiments/e19_baselines/midas_validate.py [--tiles 4] [--tokens 256]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from cimfp4.golden.l2_midas_like import MidasLikeMacro
from cimfp4.llm.capture import load_capture
from cimfp4.power.midas_model import MidasAssumptions, compare_shares, macro_record, shares_pass
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

CAPTURE = REPO_ROOT / "data/captures/qwen2.5-0.5b@060db6499f32/prefill"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=4, help="random 16-col weight sets per captured module")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    targets = yaml.safe_load((REPO_ROOT / "docs/spec/midas_targets.yaml").read_text())
    files = sorted(Path(CAPTURE).glob("*.npz"))
    cfg = {**vars(a), "capture": str(CAPTURE.relative_to(REPO_ROOT)), "n_modules": len(files),
           "assumptions": MidasAssumptions().__dict__}
    with start_run("e19_midas_validate", cfg, paper=a.paper) as run:
        rng = np.random.default_rng(0)
        acts = []
        for f in files:
            qw, qa = load_capture(f, "mxfp4")
            wb, xb = qw.blocked_codes(), qa.blocked_codes()
            for _ in range(a.tiles):
                b = int(rng.integers(0, wb.shape[1]))
                n0 = int(rng.integers(0, wb.shape[0] - 16))
                tok = rng.choice(xb.shape[0], a.tokens, replace=False)
                _, act = MidasLikeMacro("mxfp4").run(wb[n0 : n0 + 16, b], xb[tok, b])
                acts.append(act)
        act = {k: np.concatenate([x[k] for x in acts]) for k in acts[0] if k != "c_end"}
        rec = macro_record(act)
        cmp = compare_shares(rec, targets["fig21_macro_shares_pct"])
        rows = [{"component": k, **v} for k, v in cmp.items()]
        run.save_csv("midas_validation.csv", rows)
        label = "MIDAS-like" if shares_pass(cmp) else "MIDAS-inspired"
        run.log_metrics(
            {
                "label": label,
                "fig21_all_within_5pp": shares_pass(cmp),
                "max_abs_dpp_power": max(abs(v["power_dpp"]) for v in cmp.values()),
                "max_abs_dpp_area": max(abs(v["area_dpp"]) for v in cmp.values()),
                "cycles_per_op_mxfp4": rec["cycles_per_op"],
                "fj_per_mac_sky130_model": rec["fj_per_mac"],
                "n_ops": rec["n_ops"],
                "ablation_fig13_fig22": "todo",
                "mepd_latency_mxfp6": "todo (needs raw activation capture)",
            }
        )
        for r in rows:
            print(f"{r['component']:16s} P {r['power_model']:5.1f}/{r['power_midas']:5.1f} "
                  f"A {r['area_model']:5.1f}/{r['area_midas']:5.1f} {'ok' if r['pass'] else 'FAIL'}")
        print("label:", label)


if __name__ == "__main__":
    main()
