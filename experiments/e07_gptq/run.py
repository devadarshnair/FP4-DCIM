"""e07 - GPTQ (+ Four-over-Six for NVFP4) weight quantization, W4A4 PPL (WP-17).

True-sequential GPTQ: for each decoder layer in order, the Hessians of its Linear inputs are collected
by running the calibration set through the model (earlier layers already quantized), its weights are
replaced by GPTQ-quantized values (on the FP4 grid), then evaluation patches the model with CIMLinear
(activations RTN-quantized online, attention MXFP8). Calibration: WikiText-2 *train*, `--nsamples` random
windows of `--seqlen` tokens per seed (recorded). Seeds give the spread reported in T2.

Run: python experiments/e07_gptq/run.py [--models a,b] [--seeds 0,1,2] [--nsamples 32] [--paper]
"""

from __future__ import annotations

import argparse
import gc
import json

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.llm.recipes.gptq import gptq_quantize
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

RECIPES = {
    "mxfp4_gptq": dict(fmt="mxfp4", rule="ocp_v11", fos=False),
    "nvfp4_gptq": dict(fmt="nvfp4", rule=None, fos=False),
    "nvfp4_gptq_4over6": dict(fmt="nvfp4", rule=None, fos=True),
}


def calib_batches(tok, nsamples, seqlen, seed):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seqlen - 1, (nsamples,), generator=g)
    return [ids[s : s + seqlen][None] for s in starts.tolist()]


@torch.no_grad()
def gptq_model(model, batches, recipe):
    layers = model.model.layers
    for li, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        hs = {
            n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device="cuda")
            for n, m in lins.items()
        }
        cnt = {n: 0 for n in lins}

        def hook(name, hs=hs, cnt=cnt):
            def fn(_m, args, _o):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                hs[name].mul_(cnt[name] / (cnt[name] + x.shape[0])).add_(
                    2 * x.T @ x / (cnt[name] + x.shape[0])
                )
                cnt[name] += x.shape[0]

            return fn

        handles = [m.register_forward_hook(hook(n)) for n, m in lins.items()]
        for (
            b
        ) in batches:  # forward only up to this layer would be cheaper; full forward keeps the code simple
            model(b.to("cuda"))
        for h in handles:
            h.remove()
        for n, m in lins.items():
            wq = gptq_quantize(
                m.weight.float(),
                hs[n],
                recipe["fmt"],
                recipe["rule"] or "ocp_v11",
                four_over_six=recipe["fos"],
            )
            m.weight.copy_(wq.to(m.weight.dtype))
        torch.cuda.empty_cache()
        print(f"  layer {li} quantized", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m,qwen2.5-0.5b")
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    cfg = {
        **vars(a),
        "calibration": "wikitext-2-raw-v1 train, random windows",
        "eval_dataset_revision": ds_rev,
        "gptq": "true-sequential, percdamp 0.01, blocksize 128, group = MX 32 / NV 16",
    }
    with start_run("e07_gptq", cfg, paper=a.paper) as run:
        rows = []
        ck = RowCheckpoint("e07_gptq", {"nsamples": a.nsamples, "seqlen": a.seqlen},
                           [
                "src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "src/cimfp4/golden/accum.py", "src/cimfp4/golden/l1_dot.py", "src/cimfp4/utils/checkpoint.py", "experiments/configs",
                "experiments/e07_gptq",
            ])
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            for rname in a.recipes.split(","):
                rec = RECIPES[rname]
                for seed in (int(s) for s in a.seeds.split(",")):
                    if (done := ck.get(name, rname, seed)) is not None:
                        rows.append(done)
                        print(f"{name:14s} {rname:20s} seed {seed} token-PPL {done['token_ppl']:.4f} (checkpoint)", flush=True)
                        continue
                    model = (
                        AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.float32)
                        .to("cuda")
                        .eval()
                    )
                    gptq_model(model, calib_batches(tok, a.nsamples, a.seqlen, seed), rec)
                    model = model.to(torch.bfloat16)
                    if seed == 0:  # reused by e09 (T2 GPTQ rows, lm-eval) without re-running GPTQ
                        wdir = REPO_ROOT / "build" / "gptq" / f"{name}__{rname}"
                        model.save_pretrained(wdir)
                        (wdir / "gptq_meta.json").write_text(json.dumps(
                            {"model": name, "revision": rev, "recipe": rname, "seed": seed,
                             "nsamples": a.nsamples, "seqlen": a.seqlen, "commit": run.git["sha"], "run_id": run.run_id}, indent=2))
                    cc = CIMConfig(fmt=rec["fmt"], attn_policy="mxfp8")
                    patch_model(model, cc)
                    seqlen = min(a.seqlen, getattr(model.config, "max_position_embeddings", a.seqlen))
                    r = perplexity(model, tok, text, seqlen=seqlen)
                    rows.append({"model": name, "recipe": rname, "seed": seed, **cc.as_dict(), **r.as_dict()})
                    ck.put(rows[-1], name, rname, seed)
                    print(f"{name:14s} {rname:20s} seed {seed} token-PPL {r.token_ppl:.4f}", flush=True)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
        run.save_csv("ppl.csv", rows)
        import numpy as np

        m = {}
        for name in a.models.split(","):
            for rname in a.recipes.split(","):
                v = [r["token_ppl"] for r in rows if r["model"] == name and r["recipe"] == rname]
                key = f"{name.replace('.', 'p').replace('-', '_')}_{rname}"
                m[f"{key}_token_ppl_mean"], m[f"{key}_token_ppl_std"] = float(np.mean(v)), float(np.std(v))
        run.log_metrics(m)


if __name__ == "__main__":
    main()
