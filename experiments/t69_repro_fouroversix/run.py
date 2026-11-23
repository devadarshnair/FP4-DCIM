"""T6.9 - reproduce one published Four-over-Six row on the local RTX 4060.

Target: Cook et al., "Four Over Six" (arXiv 2512.02010v5), Table 3 / Table 5, Llama-3.2-1B, WikiText-2 *word*
perplexity, W4A4 PTQ: BF16 11.98, NVFP4 RTN (M=6) 14.27, MXFP4 17.67.
Paper protocol as stated (§2.2, §2.1): embeddings, LM head and attention BMMs kept in high precision; every
other Linear W4A4; NVFP4 = E2M1 + E4M3 scale per 16 + FP32 per-tensor scale amax/(6*448) (== our "dynamic"
tensor scale); MXFP4 = E8M0 per 32. The paper does not state the eval harness/seqlen; the word-PPL metric name
matches lm-eval's `wikitext` task (document-level wikitext-2-raw-v1 test, rolling loglikelihood), so we use it
(lm-eval 0.4.13) and report the max_length we used.

Run: python experiments/t69_repro_fouroversix/run.py --configs bf16 --max-length 2048 [--paper]
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from cimfp4.llm.config import CIMConfig
from cimfp4.llm.lmeval import run_lmeval
from cimfp4.llm.patch import patch_model
from cimfp4.utils.provenance import REPO_ROOT, env_info
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

EXP = "t69_repro_fouroversix"
HF_ID, REV = "meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"  # = t66 Llama rows
PUBLISHED = {"bf16": 11.98, "nv_rtn": 14.27, "mx_rtn": 17.67}  # Cook et al. v5 Tab.3/5, WikiText-2 word PPL
CONFIGS = {
    "bf16": None,
    "nv_rtn": {"fmt": "nvfp4", "nv_convention": "hw", "nv_act_scale": "dynamic", "attn_policy": "bf16"},
    "mx_rtn": {"fmt": "mxfp4", "scale_rule": "ocp_v11", "attn_policy": "bf16"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="bf16")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_all(0)
    tok = AutoTokenizer.from_pretrained(HF_ID, revision=REV)
    names = a.configs.split(",")
    cfg = {"model": HF_ID, "revision": REV, "task": "lm-eval wikitext (word_perplexity)", "max_length": a.max_length,
           "batch_size": a.batch_size, "limit": a.limit, "published": PUBLISHED,
           "configs": {n: CONFIGS[n] for n in names}}
    with start_run(EXP, cfg, paper=a.paper) as run:
        rows = []
        for n in names:
            m = AutoModelForCausalLM.from_pretrained(HF_ID, revision=REV, dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
            npat = len(patch_model(m, CIMConfig(**CONFIGS[n]))["patched"]) if CONFIGS[n] else 0
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            le = run_lmeval(m, tok, tasks={"wikitext": "word_perplexity"}, batch_size=a.batch_size,
                            max_length=a.max_length, limit=a.limit)
            w = le["wikitext"]["value"]
            row = {"config": n, "n_patched": npat, "word_ppl": w, "published": PUBLISHED[n],
                   "gap": w - PUBLISHED[n], "rel_gap": w / PUBLISHED[n] - 1, "max_length": a.max_length,
                   "seconds": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "gpu": env_info().get("gpu"), "lmeval_meta": le["_meta"]}
            print(f"[row] {n}: word_ppl={w:.4f} published={PUBLISHED[n]} gap={row['gap']:+.4f}", flush=True)
            rows.append(row)
            del m
            torch.cuda.empty_cache()
        (run.dir / "rows.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))
        run.log_metrics({r["config"]: {k: r[k] for k in ("word_ppl", "published", "gap")} for r in rows})


if __name__ == "__main__":
    main()
