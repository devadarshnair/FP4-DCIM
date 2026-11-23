"""e03 reproducibility check (WP-13 "rerun reproduces"): recompute selected rows from scratch (no checkpoint)
and compare token-PPL with results/e03_w4a4_rtn/latest/ppl.csv under the reproduction contract
(cimfp4.utils.repro): bit-identical on the reference platform (GPU, torch build, CUDA) and for
order-independent formats (MXFP4) everywhere; NVFP4 fake-quant rows on a different platform within 0.25 %.
The 4-decimal match of the original check is still reported (match_4dp) but no longer decides.

Run: python experiments/e03_w4a4_rtn/repro.py [--rows smollm2-360m:nvfp4_hw,qwen2.5-0.5b:mxfp4_v11]
Exit code 0 iff every row passes its criterion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.provenance import REPO_ROOT, env_info
from cimfp4.utils.repro import check_reproduction, platform_of
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="smollm2-360m:nvfp4_hw,qwen2.5-0.5b:mxfp4_v11")
    a = ap.parse_args()
    # Match the deterministic CUDA/TF32 setup used by the reference sweep.
    seed_all(0)
    ref_dir = (REPO_ROOT / "results/e03_w4a4_rtn/latest").resolve()
    ref = {(r["model"], r["config"]): r for r in csv.DictReader((ref_dir / "ppl.csv").open())}
    q = yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text())
    cfgs = dict(q["configs"]) | dict(q["ablations"])
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    ref_platform = platform_of(json.loads((ref_dir / "env.json").read_text()))
    platform = platform_of(env_info())
    rows, ok = [], True
    cfg = {**vars(a), "reference_run": ref_dir.name, "dataset_revision": ds_rev,
           "reference_platform": list(ref_platform), "platform": list(platform)}
    with start_run("e03_repro", cfg) as run:
        for item in a.rows.split(","):
            name, cname = item.split(":")
            r0 = ref[(name, cname)]
            spec = specs[name]
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=r0["revision"])
            model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=r0["revision"], dtype=torch.bfloat16)
            model = model.to("cuda").eval()
            patch_model(model, CIMConfig(**cfgs[cname]))
            r = perplexity(model, tok, text, seqlen=int(r0["seqlen"]))
            ref_ppl = float(r0["token_ppl"])
            match = round(r.token_ppl, 4) == round(ref_ppl, 4)
            v = check_reproduction(cfgs[cname]["fmt"], r.token_ppl, ref_ppl, platform, ref_platform)
            ok &= v.passed
            rows.append({"model": name, "config": cname, "fmt": cfgs[cname]["fmt"], "token_ppl": r.token_ppl,
                         "reference_token_ppl": ref_ppl, "match_4dp": match, **v.as_dict()})
            print(f"{name:14s} {cname:20s} {r.token_ppl:.9f} vs {ref_ppl:.9f} rel {v.rel_diff:.3e} "
                  f"[{v.criterion}, same_platform={v.same_platform}] {'PASS' if v.passed else 'FAIL'}", flush=True)
            del model
            torch.cuda.empty_cache()
        run.save_csv("repro.csv", rows)
        run.log_metrics({"all_pass": ok, "all_match_4dp": all(r["match_4dp"] for r in rows)})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
