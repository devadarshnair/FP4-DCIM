"""e02 - WikiText-2 perplexity baselines (BF16), fixed protocol (WP-12).

Run: python experiments/e02_ppl_baseline/run.py [--models a,b] [--seqlen 2048] [--paper]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

DEFAULT = "smollm2-360m,qwen2.5-0.5b,gpt2-large,qwen2-1.5b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    cfg = {
        **vars(a),
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "dataset_revision": ds_rev,
        "dtype": "bfloat16",
        "protocol": "WikiText-2 test, non-overlapping windows",
    }
    with start_run("e02_ppl_baseline", cfg, paper=a.paper) as run:
        rows = []
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            model = (
                AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16)
                .to("cuda")
                .eval()
            )
            seqlen = min(a.seqlen, getattr(model.config, "max_position_embeddings", a.seqlen) or a.seqlen)
            r = perplexity(model, tok, text, seqlen=seqlen)
            row = {"model": name, "hf_id": spec["hf_id"], "revision": rev, "seqlen": seqlen, **r.as_dict()}
            if name == a.models.split(",")[0]:  # determinism check on the first model
                r2 = perplexity(model, tok, text, seqlen=seqlen)
                row["repeat_identical_4dp"] = round(r.token_ppl, 4) == round(r2.token_ppl, 4)
            rows.append(row)
            print(
                f"{name:14s} seqlen {seqlen}  token-PPL {r.token_ppl:8.4f}  word-PPL {r.word_ppl:8.4f}  "
                f"windows {r.n_windows}  bos_added {r.bos_added}"
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        run.save_csv("bf16_ppl.csv", rows)
        run.log_metrics(
            {f"{r['model'].replace('.', 'p').replace('-', '_')}_bf16_token_ppl": r["token_ppl"] for r in rows}
            | {f"{r['model'].replace('.', 'p').replace('-', '_')}_bf16_word_ppl": r["word_ppl"] for r in rows}
            | {"repeat_identical": bool(rows[0].get("repeat_identical_4dp"))}
        )


if __name__ == "__main__":
    main()
