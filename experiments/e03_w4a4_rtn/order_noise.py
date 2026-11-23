"""e03_order_noise - how far does a fake-quant token-PPL move when only the fp32 GEMM summation order changes?

Evidence for the E03 cross-platform reproduction contract. On ONE GPU, each CIMLinear GEMM is
recomputed as a split-K sum: y = sum_i x[:, K_i] @ W[:, K_i]^T in fp32 (TF32 off), with the chunks summed
forward or reversed. Every variant is a legitimate fp32 GEMM that a different GPU / cuBLAS build could
choose, so the spread of token-PPL over variants measures the platform-induced noise floor. Expected:
MXFP4 rows are bit-identical for every order (exact products and sums), NVFP4 rows are not.

Run: python experiments/e03_w4a4_rtn/order_noise.py [--rows smollm2-360m:nvfp4_hw,...] [--max-windows N]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.kernels.fakequant import no_tf32
from cimfp4.llm import cim_linear
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

# (name, number of K chunks, reversed chunk order); "default" = the unmodified CIMLinear forward
VARIANTS = [("default", 0, False), ("splitk2", 2, False), ("splitk4", 4, False), ("splitk4_rev", 4, True),
            ("splitk8_rev", 8, True)]
_ORIG_FORWARD = cim_linear.CIMLinear.forward
_MODE = {"chunks": 0, "rev": False}


def _split_forward(self, x):
    n, rev = _MODE["chunks"], _MODE["rev"]
    if n == 0 or self.cfg.backend != "fakequant":
        return _ORIG_FORWARD(self, x)
    xq = cim_linear.fake_quant(x.float(), self.cfg) if (self.cfg.quantize_acts and self.cfg.fmt != "bf16") \
        else x.float()
    w = self.weight.float()
    k = w.shape[1]
    step = -(-k // n)
    bounds = [(i, min(i + step, k)) for i in range(0, k, step)]
    if rev:
        bounds = bounds[::-1]
    with no_tf32():
        y = None
        for lo, hi in bounds:
            part = torch.nn.functional.linear(xq[..., lo:hi], w[:, lo:hi])
            y = part if y is None else y + part
        if self.bias is not None:
            y = y + self.bias.float()
    return y.to(x.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="smollm2-360m:nvfp4_hw,smollm2-360m:mxfp4_v11")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    q = yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text())
    cfgs = dict(q["configs"])
    text, ds_rev = wikitext2_text()
    cim_linear.CIMLinear.forward = _split_forward
    cfg = {**vars(a), "dataset_revision": ds_rev, "variants": VARIANTS}
    with start_run("e03_order_noise", cfg) as run:
        rows, metrics = [], {}
        for item in a.rows.split(","):
            name, cname = item.split(":")
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16)
            model = model.to("cuda").eval()
            patch_model(model, CIMConfig(**cfgs[cname]))
            seqlen = min(a.seqlen, getattr(model.config, "max_position_embeddings", a.seqlen) or a.seqlen)
            ppls = {}
            for vname, n, rv in VARIANTS:
                _MODE.update(chunks=n, rev=rv)
                r = perplexity(model, tok, text, seqlen=seqlen, max_windows=a.max_windows)
                ppls[vname] = r.token_ppl
                rel = r.token_ppl / ppls["default"] - 1.0
                rows.append({"model": name, "config": cname, "fmt": cfgs[cname]["fmt"], "variant": vname,
                             "chunks": n, "reversed": rv, "seqlen": seqlen, "n_windows": r.n_windows,
                             "token_ppl": r.token_ppl, "rel_vs_default": rel})
                print(f"{name:14s} {cname:10s} {vname:12s} {r.token_ppl:.9f}  rel {rel:+.3e}", flush=True)
            vals = list(ppls.values())
            key = f"{name.replace('.', 'p').replace('-', '_')}_{cname}"
            metrics[f"{key}_max_rel_spread"] = (max(vals) - min(vals)) / ppls["default"]
            metrics[f"{key}_bit_identical"] = len(set(vals)) == 1
            del model
            gc.collect()
            torch.cuda.empty_cache()
        run.save_csv("order_noise.csv", rows)
        run.log_metrics(metrics)


if __name__ == "__main__":
    main()
