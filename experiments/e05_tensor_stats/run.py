"""e05 - real-tensor operand statistics per model / phase / module / format (WP-15).

Reads every capture directory under data/captures/ (from e04), verifies its manifest, and writes
code_hist / zeros / sign_balance / blk_partial / exp_span / toggle / bitplane CSVs.
Recipe: RTN (the same script is re-run later on GPTQ/Hadamard captures).

Run: python experiments/e05_tensor_stats/run.py [--pairs 4000] [--paper]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cimfp4.llm.capture import load_capture
from cimfp4.stimulus import stats as S
from cimfp4.stimulus.manifest import verify_manifest
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, default=REPO_ROOT / "data" / "captures")
    ap.add_argument("--pairs", type=int, default=4000, help="sampled (weight row, activation row) pairs")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    rng = seed_all(0)
    cap_dirs = sorted(p for p in args.captures.iterdir() if (p / "manifest.json").exists())
    cfg = {"pairs": args.pairs, "recipe": "rtn", "captures": [p.name for p in cap_dirs]}
    tables = {
        k: [] for k in ("code_hist", "zeros", "sign_balance", "blk_partial", "exp_span", "toggle", "bitplane")
    }

    with start_run("e05_tensor_stats", cfg, paper=args.paper) as run:
        for cap in cap_dirs:
            assert not verify_manifest(cap), f"manifest check failed for {cap}"
            model = cap.name.split("@")[0]
            for npz in sorted(cap.glob("*/*.npz")):
                phase, module = npz.parent.name, npz.stem
                proj = module.rsplit(".", 1)[-1]
                layer = int(module.split("layers.")[1].split(".")[0])
                for fmt in ("mxfp4", "nvfp4"):
                    qw, qa = load_capture(npz, fmt)
                    key = {
                        "model": model,
                        "phase": phase,
                        "layer": layer,
                        "proj": proj,
                        "fmt": fmt,
                        "K": qw.codes.shape[1],
                        "N": qw.codes.shape[0],
                        "T": qa.codes.shape[0],
                    }
                    for who, q in (("w", qw), ("a", qa)):
                        tables["code_hist"].append(
                            {**key, "operand": who, **{f"c{i:02d}": v for i, v in enumerate(S.code_hist(q))}}
                        )
                        tables["bitplane"].append(
                            {
                                **key,
                                "operand": who,
                                **{f"b{i}": v for i, v in enumerate(S.bitplane_density(q))},
                            }
                        )
                    tables["zeros"].append({**key, **S.zero_fractions(qw, qa)})
                    rows, cols = S.sample_pairs(qw.codes.shape[0], qa.codes.shape[0], args.pairs, rng)
                    prod = S.block_products(qw, qa, rows, cols)
                    sb = S.sign_balance(prod)
                    sb.pop("hist_neg")
                    tables["sign_balance"].append({**key, **sb})
                    bp = S.block_partials(prod)
                    tables["blk_partial"].append(
                        {
                            **key,
                            "bits_needed_p999": bp["bits_needed_p999"],
                            **{f"abs_p{k}": v for k, v in bp["abs_sum_pct"].items()},
                            **{f"pos_p{k}": v for k, v in bp["pos_sum_pct"].items()},
                        }
                    )
                    spans = S.exponent_spans(qw, qa, rows, cols)
                    tables["exp_span"].append(
                        {
                            **key,
                            "span_p50": float(np.percentile(spans, 50)),
                            "span_p99": float(np.percentile(spans, 99)),
                            "span_max": int(spans.max()),
                        }
                    )
                    tables["toggle"].append({**key, **S.toggle_rate(qa)})
        for name, rows in tables.items():
            run.save_csv(f"{name}.csv", rows)

        # headline metrics (prefill, MXFP4 / NVFP4, averaged over modules, per model)
        metrics = {}
        for fmt in ("mxfp4", "nvfp4"):
            for model in sorted({r["model"] for r in tables["zeros"]}):
                sel = [
                    r
                    for r in tables["zeros"]
                    if r["model"] == model and r["fmt"] == fmt and r["phase"] == "prefill"
                ]
                tg = [
                    r
                    for r in tables["toggle"]
                    if r["model"] == model and r["fmt"] == fmt and r["phase"] == "prefill"
                ]
                sp = [r for r in tables["exp_span"] if r["model"] == model and r["fmt"] == fmt]
                m = model.replace(".", "p").replace("-", "_")
                metrics[f"{m}_{fmt}_p_w_zero"] = float(np.mean([r["p_w_zero"] for r in sel]))
                metrics[f"{m}_{fmt}_p_a_zero"] = float(np.mean([r["p_a_zero"] for r in sel]))
                metrics[f"{m}_{fmt}_p_prod_zero"] = float(np.mean([r["p_prod_zero"] for r in sel]))
                metrics[f"{m}_{fmt}_toggle_code"] = float(np.mean([r["code"] for r in tg]))
                metrics[f"{m}_{fmt}_toggle_mag_sm"] = float(np.mean([r["mag_sm"] for r in tg]))
                metrics[f"{m}_{fmt}_span_max"] = int(max(r["span_max"] for r in sp))
        run.log_metrics(metrics)
        print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
