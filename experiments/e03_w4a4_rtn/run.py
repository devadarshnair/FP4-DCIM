"""e03 - W4A4 round-to-nearest accuracy sweep on WikiText-2 (WP-13), protocol §11.2.

All Linear layers of the decoder (not lm_head/embeddings) run through CIMLinear (weights + activations
quantized along K); attention BMM operands follow `attn_policy`. One row per (model, config).

Run: python experiments/e03_w4a4_rtn/run.py [--models ...] [--seqlen 2048] [--max-windows N] [--paper]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m,qwen2.5-0.5b,gpt2-large")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    qcfg = yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text())
    text, ds_rev = wikitext2_text()
    cfg = {
        **vars(a),
        "dataset_revision": ds_rev,
        "quant": qcfg,
        "model_dtype": "bfloat16",
        "cimlinear_compute": "fp32 fake-quant GEMM (TF32 off)",
    }
    models = a.models.split(",")
    with start_run("e03_w4a4_rtn", cfg, paper=a.paper) as run:
        rows = []
        ck = RowCheckpoint("e03_w4a4_rtn", {"seqlen": a.seqlen, "max_windows": a.max_windows, "quant": qcfg},
                           ["src/cimfp4", "experiments/configs", "experiments/e03_w4a4_rtn"])
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        for mi, name in enumerate(models):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            todo = dict(qcfg["configs"]) | (dict(qcfg["ablations"]) if mi == 0 else {})
            for cname, c in todo.items():
                if (done := ck.get(name, cname)) is not None:
                    rows.append(done)
                    print(f"{name:14s} {cname:24s} token-PPL {done['token_ppl']:9.4f}  (checkpoint)", flush=True)
                    continue
                model = AutoModelForCausalLM.from_pretrained(
                    spec["hf_id"], revision=rev, dtype=torch.bfloat16
                )
                model = model.to("cuda").eval()
                cc = CIMConfig(**c)
                rec = patch_model(model, cc)
                seqlen = min(a.seqlen, getattr(model.config, "max_position_embeddings", a.seqlen) or a.seqlen)
                r = perplexity(model, tok, text, seqlen=seqlen, max_windows=a.max_windows)
                rows.append(
                    {
                        "model": name,
                        "revision": rev,
                        "config": cname,
                        **cc.as_dict(),
                        "seqlen": seqlen,
                        "n_patched": len(rec["patched"]),
                        **r.as_dict(),
                    }
                )
                ck.put(rows[-1], name, cname)
                print(f"{name:14s} {cname:24s} token-PPL {r.token_ppl:9.4f}  word-PPL {r.word_ppl:9.4f}", flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        run.save_csv("ppl.csv", rows)
        run.log_metrics(
            {
                f"{r['model'].replace('.', 'p').replace('-', '_')}_{r['config']}_token_ppl": r["token_ppl"]
                for r in rows
            }
        )


if __name__ == "__main__":
    main()
