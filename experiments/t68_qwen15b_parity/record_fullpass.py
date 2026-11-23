"""T6.8 supplement: record the full-pass (146-window) fake-quant rows of Qwen2-1.5B next to the 16-window parity run.

The parity run (results/t68_qwen15b_parity/20261115T211447255973Z_7eaafa3) evaluates fakequant / down_l1_exact /
down_l1_hier40_kt512 on the first 16 WikiText-2 windows. The L1 modes on the full pass need > 8 GB VRAM (K=8960 tiles),
so only the `fakequant` mode was run over all 146 windows (local RTX 4060, run.py --fmts F --modes fakequant). This
script copies those two checkpoint rows (build/checkpoints/t68_qwen15b_parity.jsonl, no evaluation) into a results run
with provenance. It does NOT change the parity assert of run.py --assemble.
Run: python experiments/t68_qwen15b_parity/record_fullpass.py --paper
"""
from __future__ import annotations

import argparse
import csv
import json

import yaml

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

CKPT = REPO_ROOT / "build/checkpoints/t68_qwen15b_parity.jsonl"
PARITY = "20261115T211447255973Z_7eaafa3"
FULL_WINDOWS = 146


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    full = {}  # fmt -> list of (tag, row) with the full pass fakequant
    for line in CKPT.read_text().splitlines():
        d = json.loads(line)
        k = json.loads(d["key"])
        if k[0] == "row" and k[2] == "fakequant" and d["row"]["n_windows"] == FULL_WINDOWS:
            full.setdefault(k[1], []).append((d["tag"], d["row"]))
    assert set(full) == {"nvfp4", "mxfp4"}, full.keys()
    par = {r["fmt"]: r for r in csv.DictReader((REPO_ROOT / "results/t68_qwen15b_parity" / PARITY / "parity.csv").open())}
    with start_run("t68_fullpass_fakequant", {"checkpoint": str(CKPT.relative_to(REPO_ROOT)), "full_windows": FULL_WINDOWS,
                                             "parity_run": f"t68_qwen15b_parity/{PARITY}"}, paper=a.paper) as run:
        out, metrics = [], {}
        for fmt, lst in sorted(full.items()):
            vals = {r["token_ppl"] for _, r in lst}
            assert len(vals) == 1, (fmt, vals)  # duplicate rows under different code tags must agree bit-exactly
            tag, r = lst[-1]
            out.append({"fmt": fmt, "mode": "fakequant", "n_windows": r["n_windows"], "n_tokens": r["n_tokens"],
                        "token_ppl": r["token_ppl"], "word_ppl": r["word_ppl"], "nv_act_scale": r["nv_act_scale"],
                        "attn_policy": r["attn_policy"], "revision": r["revision"],
                        "ppl_16win_fakequant_parity_run": par[fmt]["ppl_fakequant"], "ckpt_tag": tag,
                        "n_duplicate_rows_identical": len(lst), "seconds": r["seconds"], "peak_vram_gb": r["peak_vram_gb"]})
            metrics[f"{fmt}_fakequant_full146_token_ppl"] = r["token_ppl"]
            metrics[f"{fmt}_fakequant_16win_token_ppl"] = float(par[fmt]["ppl_fakequant"])
        run.save_csv("fullpass_fakequant.csv", out)
        run.path("rows_full.yaml").write_text(yaml.safe_dump(out, sort_keys=True))
        run.log_metrics(metrics)
        run.path("NOTE.md").write_text(
            "# Qwen2-1.5B full-pass fake-quant (supplement to the T6.8 parity run)\n\n"
            f"- Cross-reference: results/t68_qwen15b_parity/{PARITY} (16 windows; Hier vs Exact vs fake-quant parity, "
            "unchanged and passing).\n"
            "- Here: fakequant over all 146 WikiText-2 windows (seqlen 1024, 298,862 tokens), BF16 attention BMMs; "
            "NVFP4 uses the static calibrated tensor scale, MXFP4 dynamic. Values are NOT comparable to the 16-window "
            "rows (different text subset).\n"
            "- The L1 modes (down_l1_exact / down_l1_hier40_kt512) were NOT run on the full pass: K=8960 L1 layers need "
            "> 8 GB VRAM. The full-pass parity claim therefore rests only on the 16-window run.\n"
            "- No evaluation here: the rows are copied from the checkpoint written by experiments/t68_qwen15b_parity/"
            "run.py on the local RTX 4060 (2026-11-16, ~/tmp/gpu_local/t68_{nv,mx}_fq.log). The NVFP4 full-pass row "
            "exists under two checkpoint tags (code-path hash changed between runs) with bit-identical token PPL.\n")


if __name__ == "__main__":
    main()
