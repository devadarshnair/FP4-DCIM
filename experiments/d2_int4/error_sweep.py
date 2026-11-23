"""d2_int4_error - quantization error of the D2 INT4 surrogate vs MXFP4/NVFP4 on real tensors (CPU only).

Acceptance step 5 of docs/spec/d2_int4_contract.md: before any accuracy run or RTL, measure SQNR (dB) and
relative RMS error of each 4-bit format on the *same source tensors*:
  mxfp4 (OCP v1.1) · nvfp4 (hw) · mxint4_sm (contract v1.1 primary, floor_sat) · mxint4_sm_ceil (v1 rule) ·
  int4_clip / int4_half (lossy recodes of the MXFP4 codes - ablations)
Weights: every decoder Linear of Qwen2.5-0.5B (bf16 checkpoint -> fp32). Activations: inputs of the Linears in
layers {0, 12, 23} from one CPU forward pass over the first `--tokens` WikiText-2 test tokens.

Run: python experiments/d2_int4/error_sweep.py [--tokens 256]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.formats import mx_quant, nv_quant
from cimfp4.formats.int4 import quantize_mxint4, recode_e2m1_to_int4
from cimfp4.llm.ppl import wikitext2_text
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

FORMATS = ("mxfp4", "nvfp4", "mxint4_sm", "mxint4_sm_ceil", "int4_clip", "int4_half")
ACT_LAYERS = (0, 12, 23)


def quantize(v: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "mxfp4":
        return mx_quant.dequantize(mx_quant.quantize_mxfp4(v, "ocp_v11"))
    if fmt == "nvfp4":
        return mx_quant.dequantize(nv_quant.quantize_nvfp4(v, "hw"))
    if fmt == "mxint4_sm":
        return quantize_mxint4(v, "floor_sat").dequantize()
    if fmt == "mxint4_sm_ceil":
        return quantize_mxint4(v, "ceil").dequantize()
    q = mx_quant.quantize_mxfp4(v, "ocp_v11")
    m, extra = recode_e2m1_to_int4(q.blocked_codes(), fmt.split("_")[1])
    s = np.ldexp(1.0, (q.scale_exp.astype(np.int64) - 1 + extra).astype(np.int32))
    return (m.astype(np.float64) * s[..., None]).reshape(v.shape)


def err(v: np.ndarray, q: np.ndarray) -> dict:
    sig, noise = float(np.sum(v * v)), float(np.sum((v - q) ** 2))
    return {"sqnr_db": 10 * np.log10(sig / noise) if noise > 0 else float("inf"),
            "rel_rms": float(np.sqrt(noise / sig)) if sig > 0 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    spec = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())[a.model]
    from huggingface_hub import model_info

    rev = spec["revision"] or model_info(spec["hf_id"]).sha
    text, ds_rev = wikitext2_text()
    cfg = {**vars(a), "revision": rev, "dataset_revision": ds_rev, "formats": FORMATS, "act_layers": ACT_LAYERS}
    with start_run("d2_int4_error", cfg, paper=a.paper) as run:
        tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.float32).eval()
        rows = []
        lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "layers." in n}
        for name, mod in lins.items():
            w = mod.weight.detach().double().numpy()
            for fmt in FORMATS:
                rows.append({"tensor": "weight", "module": name, "fmt": fmt, **err(w, quantize(w, fmt))})
        acts = {}
        hooks = [m.register_forward_hook(lambda mod, i, o, n=n: acts.__setitem__(n, i[0].detach()))
                 for n, m in lins.items() if int(n.split("layers.")[1].split(".")[0]) in ACT_LAYERS]
        ids = tok(text, return_tensors="pt").input_ids[:, : a.tokens]
        with torch.no_grad():
            model(ids)
        for h in hooks:
            h.remove()
        for name, x in acts.items():
            v = x.reshape(-1, x.shape[-1]).double().numpy()
            for fmt in FORMATS:
                rows.append({"tensor": "activation", "module": name, "fmt": fmt, **err(v, quantize(v, fmt))})
        run.save_csv("error.csv", rows)
        summary = {}
        for kind in ("weight", "activation"):
            for fmt in FORMATS:
                s = [r["sqnr_db"] for r in rows if r["tensor"] == kind and r["fmt"] == fmt]
                summary[f"{kind}_{fmt}_sqnr_db_mean"] = float(np.mean(s))
                summary[f"{kind}_{fmt}_sqnr_db_min"] = float(np.min(s))
                print(f"{kind:10s} {fmt:10s} SQNR mean {np.mean(s):6.2f} dB  min {np.min(s):6.2f} dB  (n={len(s)})")
        run.log_metrics(summary)


if __name__ == "__main__":
    main()
