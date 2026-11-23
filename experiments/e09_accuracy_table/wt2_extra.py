"""e09_wt2_extra - WikiText-2 token-PPL for the T2 configs that no earlier experiment covers.

Fills two gaps:
  * MIDAS Table II protocol rows (token-PPL, non-overlapping windows, seqlen 1024) for GPT2-L and
    Qwen2-1.5B: bf16, mx_rtn, nv_rtn, midas_mix, mxfp6 and the D2 surrogate mxint4_rtn.
  * T2 protocol rows (seqlen 2048, capped at the model's context) for midas_mix / mxfp6 / mxint4_rtn on the
    T2 models, merged into t2.csv by run.py's wikitext_lookup.
Configs come from run.py:configs() so the quantization is identical to the T2 rows. Rows are checkpointed.
A config name with the suffix `_attn_bf16` is the same config with attn_policy="bf16" (attention BMMs
unquantized; attention-K ablation: Qwen2-1.5B's near-constant K channels do not survive any 8-bit FP K format).

Run: python experiments/e09_accuracy_table/wt2_extra.py
     python experiments/e09_accuracy_table/wt2_extra.py --rows qwen2-1.5b:mxfp6_attn_bf16:1024,... \
         --run-name e09_wt2_attn_bf16
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace

import torch
import yaml
from huggingface_hub import model_info
from run import configs, load  # same directory
from transformers import AutoTokenizer

from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

MIDAS_MODELS = "gpt2-large,qwen2-1.5b"
MIDAS_CONFIGS = "bf16,mx_rtn,nv_rtn,midas_mix,mxfp6,mxint4_rtn"
T2_MODELS = "smollm2-360m,qwen2.5-0.5b,gpt2-large"
T2_CONFIGS = "midas_mix,mxfp6,mxint4_rtn"
ATTN_BF16 = "_attn_bf16"


def resolve(cfgs: dict, cname: str):
    """(source, CIMConfig | None) for a config name, including the `<config>_attn_bf16` ablations."""
    if cname.endswith(ATTN_BF16):
        source, cc = cfgs[cname[: -len(ATTN_BF16)]]
        return source, replace(cc, attn_policy="bf16")
    return (None, None) if cfgs[cname] is None else cfgs[cname]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--rows", default=None, help="model:config:seqlen,... (default: the built-in plan)")
    ap.add_argument("--run-name", default="e09_wt2_extra", help="results/<run-name> and its checkpoint file")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    cfgs = configs(yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text()), hw=False)
    text, ds_rev = wikitext2_text()
    plan = [(m, c, 1024) for m in MIDAS_MODELS.split(",") for c in MIDAS_CONFIGS.split(",")]
    plan += [(m, c, 2048) for m in T2_MODELS.split(",") for c in T2_CONFIGS.split(",")]
    if a.rows:
        plan = [(m, c, int(s)) for m, c, s in (r.split(":") for r in a.rows.split(","))]
    cfg = {**vars(a), "dataset_revision": ds_rev, "plan": plan}
    with start_run(a.run_name, cfg, paper=a.paper) as run:
        ck = RowCheckpoint(
            a.run_name,
            {"dataset_revision": ds_rev},
            ["src/cimfp4/llm", "src/cimfp4/formats", "src/cimfp4/utils/checkpoint.py", "experiments/configs",
             "experiments/e09_accuracy_table"],
        )
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        rows, toks = [], {}
        for name, cname, s_req in plan:
            row = ck.get(name, cname, s_req)
            if row is None:
                spec = specs[name]
                rev = spec["revision"] or model_info(spec["hf_id"]).sha
                if name not in toks:
                    toks[name] = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
                source, cc = resolve(cfgs, cname)
                model = load(spec, rev, source, name)
                if cc is not None:
                    patch_model(model, cc)
                s = min(s_req, getattr(model.config, "max_position_embeddings", s_req))
                r = perplexity(model, toks[name], text, seqlen=s)
                row = {"model": name, "revision": rev, "config": cname, "seqlen_requested": s_req, "seqlen": s,
                       "attn_policy": "bf16 (unpatched)" if cc is None else cc.attn_policy, **r.as_dict()}
                ck.put(row, name, cname, s_req)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            print(f"{name:14s} {cname:12s} seqlen {row['seqlen']:5d} token-PPL {row['token_ppl']:9.4f}", flush=True)
        run.save_csv("ppl.csv", rows)
        run.log_metrics({f"{r['model'].replace('.', 'p').replace('-', '_')}_{r['config']}_s{r['seqlen']}_token_ppl":
                         r["token_ppl"] for r in rows})


if __name__ == "__main__":
    main()
