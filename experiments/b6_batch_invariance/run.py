"""B6 - batch-invariance of the e09 lm-eval rows: same patched model, batch 2 vs batch 8, per-sample loglikelihoods.

e09_mxint4 was run at batch 2 (local 8 GB GPU) while the other T2 rows ran at batch 8 (cloud GPU). Padding changes
the shapes of the bf16 GEMMs, so tiny numeric differences are possible; this measures them directly.
Run: python experiments/b6_batch_invariance/run.py [--model smollm2-360m] [--config mxint4_rtn|mx_rtn|bf16] (bf16 = unpatched control) [--limit 300]
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch
import yaml
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

TASKS = {"arc_easy": "acc_norm", "lambada_openai": "acc", "boolq": "acc"}


def evaluate(spec, rev, tok, qcfg, batch, limit, seqlen):
    seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=torch.bfloat16, revision=rev).to("cuda").eval()
    if qcfg is not None:
        patch_model(model, qcfg)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch, max_length=seqlen)
    res = simple_evaluate(model=lm, tasks=list(TASKS), limit=limit, log_samples=True, random_seed=0,
                          numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
    ll = {t: np.array([[float(r[0][0]) for r in s["resps"]] for s in res["samples"][t]], dtype=object)
          for t in TASKS}
    ll = {t: np.concatenate([np.array(x, dtype=np.float64) for x in v]) for t, v in ll.items()}
    acc = {t: res["results"][t][f"{m},none"] for t, m in TASKS.items()}
    del model, lm
    gc.collect()
    torch.cuda.empty_cache()
    return ll, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-360m")
    ap.add_argument("--config", default="mxint4_rtn")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batches", default="2,8")
    ap.add_argument("--seqlen", type=int, default=2048)
    a = ap.parse_args()
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    q = yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text())
    spec = specs[a.model]
    rev = spec["revision"]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
    mx = q["configs"]["mxfp4_v11"]
    qcfg = {"bf16": None, "mxint4_rtn": CIMConfig(**{**mx, "fmt": "mxint4"}), "mx_rtn": CIMConfig(**mx)}[a.config]
    with start_run("b6_batch_invariance", vars(a)) as run:
        out = {b: evaluate(spec, rev, tok, qcfg, int(b), a.limit, a.seqlen) for b in a.batches.split(",")}
        b0, *rest = out
        metrics = {}
        for b in rest:
            for t in TASKS:
                d = np.abs(out[b0][0][t] - out[b][0][t])
                metrics[f"{t}_b{b0}_vs_b{b}_n_loglik"] = int(d.size)
                metrics[f"{t}_b{b0}_vs_b{b}_max_abs_dloglik"] = float(d.max())
                metrics[f"{t}_b{b0}_vs_b{b}_n_bitwise_diff"] = int((d != 0).sum())
                metrics[f"{t}_b{b0}_vs_b{b}_acc_equal"] = bool(out[b0][1][t] == out[b][1][t])
        for b in out:
            for t in TASKS:
                metrics[f"{t}_acc_b{b}"] = float(out[b][1][t])
        run.log_metrics(metrics)
        print(yaml.safe_dump(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
