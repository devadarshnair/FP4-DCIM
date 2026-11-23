"""e02b - BF16 WikiText-2 PPL vs sequence length, to pin down MIDAS's protocol (docs/spec/midas_protocol.md).

MIDAS Table II ground truth: GPT2-L 19.44 (we match at seqlen 1024), Llama3.2-1B 10.99, Qwen2-1.5B 10.70 (we get
9.54 at 2048). Token-PPL with non-overlapping windows at each seqlen; results in results/e02b_seqlen_sweep.

Run: python experiments/e02_ppl_baseline/seqlen_sweep.py [--models qwen2-1.5b,gpt2-large] [--seqlens 256,...]
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

MIDAS_GT = {"gpt2-large": 19.44, "llama-3.2-1b": 10.99, "qwen2-1.5b": 10.70}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2-1.5b,gpt2-large")
    ap.add_argument("--seqlens", default="256,512,1024,2048")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    with start_run("e02b_seqlen_sweep", {**vars(a), "dataset_revision": ds_rev, "midas_gt": MIDAS_GT},
                   paper=a.paper) as run:
        rows = []
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16)
            model = model.to("cuda").eval()
            max_pos = getattr(model.config, "max_position_embeddings", None) or 1 << 30
            for s in (int(x) for x in a.seqlens.split(",")):
                if s > max_pos:
                    continue
                r = perplexity(model, tok, text, seqlen=s)
                gt = MIDAS_GT.get(name)
                rows.append({"model": name, "revision": rev, "seqlen": s, "midas_gt": gt, **r.as_dict()})
                print(f"{name:12s} seqlen {s:5d} token-PPL {r.token_ppl:8.4f} word-PPL {r.word_ppl:8.4f} "
                      f"(MIDAS GT {gt})", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        run.save_csv("ppl_vs_seqlen.csv", rows)
        best = {}
        for r in rows:
            if r["midas_gt"] is not None:
                d = abs(r["token_ppl"] - r["midas_gt"])
                if r["model"] not in best or d < best[r["model"]][1]:
                    best[r["model"]] = (r["seqlen"], d)
        run.log_metrics({f"{m.replace('.', 'p').replace('-', '_')}_closest_seqlen": v[0] for m, v in best.items()}
                        | {f"{m.replace('.', 'p').replace('-', '_')}_closest_abs_diff": v[1] for m, v in best.items()})


if __name__ == "__main__":
    main()
