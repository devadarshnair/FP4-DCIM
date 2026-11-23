"""e09_attn_k_diag - attention-K diagnosis: which attention BMM operand breaks Qwen2-1.5B under MXFP8, and why.

Linear layers stay BF16 (unpatched weights); only the attention BMM operands are fake-quantized, one operand
(or layer subset) at a time, with the same fq_mxfp8 blocks as cimfp4.llm.attn_quant (Q, K along head_dim;
P along keys; V along tokens). Two K-quantizer controls separate the mechanism from the MX block structure:
  k_e4m3_tensor   plain E4M3 cast of K, scale 1 (per-tensor FP8, no blocks)
  k_mx_smooth     MXFP8 of K - mean_t(K) (+ mean added back): subtracting a per-(head, channel) constant
                  from every key leaves softmax unchanged, so this isolates the near-constant channels.
Also records the K-channel statistics of the layer with the largest |K| (bias, mean, std, quant error).

Run: python experiments/e09_accuracy_table/attn_k_diag.py [--models qwen2-1.5b,qwen2.5-0.5b,smollm2-360m]
     [--seqlen 1024] [--max-windows 8]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from cimfp4.kernels.torch_quant import fq_mxfp8
from cimfp4.llm.attn_quant import _apply_mask, install
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

S = {"ops": "", "k_layers": None, "k_mode": "mx", "stats": None}


def _fq(x: torch.Tensor, dim: int) -> torch.Tensor:
    xt = x.transpose(dim, -1)
    pad = (-xt.shape[-1]) % 32
    if pad:
        xt = torch.nn.functional.pad(xt, (0, pad))
    q = fq_mxfp8(xt.float())
    if pad:
        q = q[..., : xt.shape[-1] - pad]
    return q.transpose(dim, -1).to(x.dtype)


def _fq_k(k: torch.Tensor) -> torch.Tensor:
    if S["k_mode"] == "mx":
        return _fq(k, -1)
    if S["k_mode"] == "e4m3_tensor":
        return torch.clamp(k.float(), -448.0, 448.0).to(torch.float8_e4m3fn).to(k.dtype)
    if S["k_mode"] == "mx_smooth":
        mu = k.float().mean(dim=-2, keepdim=True)
        return (fq_mxfp8(k.float() - mu) + mu).to(k.dtype)
    raise ValueError(S["k_mode"])


def diag_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    layer, ops = module.layer_idx, S["ops"]
    if S["stats"] is not None:  # before the GQA repeat: K keeps its n_kv heads (matches the k_proj bias)
        S["stats"][layer] = (query.detach().float().cpu(), key.detach().float().cpu())
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    q = _fq(query, -1) if "q" in ops else query
    k = _fq_k(key) if ("k" in ops and (S["k_layers"] is None or layer in S["k_layers"])) else key
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scaling
    scores = _apply_mask(scores, attention_mask, key.shape[-2])
    probs = torch.softmax(scores, dim=-1)
    p = _fq(probs, -1) if "p" in ops else probs
    v = _fq(value, -2) if "v" in ops else value
    out = torch.matmul(p.float(), v.float()).to(query.dtype)
    return out.transpose(1, 2).contiguous(), probs


def channel_stats(model, name: str, layer: int, q: torch.Tensor, k: torch.Tensor) -> list[dict]:
    """Top-4 |K| channels of the KV head holding the largest |K| in `layer` (q: (1, H, T, D), k: (1, Hkv, T, D),
    post-RoPE; query head h * n_rep is the first one reading KV head h)."""
    kf, qf = k[0], q[0]
    n_rep = qf.shape[0] // kf.shape[0]
    chmax = kf.abs().amax(1)  # (Hkv, D)
    h, _ = divmod(int(chmax.argmax()), kf.shape[-1])
    bias = getattr(model.model.layers[layer].self_attn.k_proj, "bias", None)
    kq = fq_mxfp8(kf[h])
    rows = []
    for d in chmax[h].topk(4).indices.tolist():
        col, qc = kf[h, :, d], qf[h * n_rep, :, d]
        rows.append({"model": name, "layer": layer, "kv_head": h, "channel": d,
                     "k_proj_bias": None if bias is None else float(bias.view(kf.shape[0], -1)[h, d]),
                     "k_mean": float(col.mean()), "k_std": float(col.std()), "k_min": float(col.min()),
                     "k_max": float(col.max()), "q_std": float(qc.std()),
                     "mxfp8_err_rms": float((kq[:, d] - col).pow(2).mean().sqrt()),
                     "mxfp8_distinct_values": int(kq[:, d].unique().numel())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2-1.5b,qwen2.5-0.5b,smollm2-360m")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=8)
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    with start_run("e09_attn_k_diag", {**vars(a), "dataset_revision": ds_rev}) as run:
        rows, chans, metrics = [], [], {}
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16)
            model = model.to("cuda").eval()
            install(model, "bf16")
            AttentionInterface.register("cim_eager", diag_attention)
            # K statistics pass (1 window): rank layers by max |K|
            S.update(ops="", stats={})
            perplexity(model, tok, text, seqlen=a.seqlen, max_windows=1)
            kmax = {layer: float(k.abs().max()) for layer, (_, k) in S["stats"].items()}
            ranked = sorted(kmax, key=kmax.get, reverse=True)
            top = ranked[0]
            chans += channel_stats(model, name, top, *S["stats"][top])
            S["stats"] = None
            n_layers = len(kmax)
            variants = [("none", "", None, "mx"), ("q", "q", None, "mx"), ("k", "k", None, "mx"),
                        ("p", "p", None, "mx"), ("v", "v", None, "mx"), ("qkpv", "qkpv", None, "mx"),
                        (f"k_layer{top}_only", "k", {top}, "mx"),
                        (f"k_except_layers{ranked[0]}_{ranked[1]}", "k", set(range(n_layers)) - set(ranked[:2]), "mx"),
                        ("k_e4m3_tensor", "k", None, "e4m3_tensor"), ("k_mx_smooth", "k", None, "mx_smooth")]
            for vname, ops, layers, mode in variants:
                S.update(ops=ops, k_layers=layers, k_mode=mode)
                r = perplexity(model, tok, text, seqlen=a.seqlen, max_windows=a.max_windows)
                rows.append({"model": name, "revision": rev, "variant": vname, "seqlen": a.seqlen,
                             "n_windows": r.n_windows, "token_ppl": r.token_ppl,
                             "kmax_layer_ranked": ranked[:3], "kmax_top": kmax[top]})
                metrics[f"{name.replace('.', 'p').replace('-', '_')}_{vname}_token_ppl"] = r.token_ppl
                print(f"{name:14s} {vname:22s} token-PPL {r.token_ppl:9.4f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        run.save_csv("attn_k_diag.csv", rows)
        run.save_csv("k_channels.csv", chans)
        run.log_metrics(metrics)


if __name__ == "__main__":
    main()
