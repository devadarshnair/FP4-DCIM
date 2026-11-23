"""e25_midas_accuracy - validate our MX behaviour against every MIDAS published accuracy point (D2).

MIDAS [choi2026midas] publishes WikiText-2 token-PPL at 1024 tokens (Fig. 1(b) "Token=1024"; Table II) and states
the OCP v1.0 shared-exponent rule e_shared = floor(log2 max|x|) - emax_elem (eq. (1)) = our `ocp_floor`. Our
default (`ocp_v11`) differs, and our attention policy (cimfp4.llm.attn_quant) always uses ocp_v11 MXFP8. This
script re-runs the MIDAS operating points with an experiment-local attention implementation whose four BMM
operands (Q, K along head_dim; P along keys; V along tokens, as attn_quant) each take an explicit quantizer:

    bf16                   no quantization
    mxfp8:<rule>           MX E4M3 (block 32), rule in ocp_floor | ocp_v11 | rceil   (cimfp4 fq_mxfp8)
    e5m2:floor             MX E5M2 (block 32, X = floor(log2 amax) - 15, RNE, saturate +-57344)   [local]
    mxint8                 MX INT8 (block 32, X = floor(log2 amax), elements int8 * 2^-6, RNE, clip) [local]
    mxfp8tok:<rule>        K only: MXFP8 blocks along the token axis instead of head_dim
    off+mxfp8:<rule>       K only: subtract a static per-(layer, kv-head, channel) offset c calibrated on the
                           WikiText-2 *train* split (8 windows, same linear quantization, BF16 attention), then
                           MXFP8. Q.(k - c) = Q.k - Q.c is constant per query, so softmax is unchanged exactly;
                           c is static (causal, deployable as a bias-folded constant). Not MIDAS protocol.
    mean+mxfp8:<rule>      K only: subtract the per-window token mean (non-causal; = k_mx_smooth)

Control row `ctrl_gpt2l_mx_v11` (MXFP4 ocp_v11 linears, all four operands mxfp8:ocp_v11) must reproduce
e09_wt2_extra gpt2-large mx_rtn @1024 = 21.939188290538986 bit-for-bit; this validates the local attention.

Phases: `diag` (Qwen2-1.5B, 8 windows, MXFP6-floor linears, Q/P/V MXFP8-floor, one K variant per row) and
`full` (all 280/292 windows). Rows are checkpointed (build/checkpoints/e25_midas_accuracy.jsonl).

Run: python experiments/e25_midas_accuracy/run.py [--phases diag,full] [--rows name,...]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from cimfp4.kernels.torch_quant import _e4m3_rne_satfinite, _floor_log2, fq_mxfp8
from cimfp4.llm import cim_linear
from cimfp4.llm.attn_quant import _apply_mask, install
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

SEQLEN = 1024
DIAG_WINDOWS = 8
CALIB_WINDOWS = 8
F = "mxfp8:ocp_floor"  # MIDAS eq. (1) rule for the MXFP8 attention operands
ALL_F = {"q": F, "k": F, "p": F, "v": F}
ALL_BF16 = {"q": "bf16", "k": "bf16", "p": "bf16", "v": "bf16"}
MIX = (("attn.", "mxfp6"),)  # MIDAS Table II footnote (2): MXFP6 attention projections (GPT-2 names)

# name -> (model, linear spec, attention spec, phase, MIDAS point it maps to)
#   linear spec: (fmt, scale_rule, fmt_overrides, quantize lm_head)
ROWS: dict[str, tuple] = {}


def _row(name, model, lin, attn, phase, maps):
    ROWS[name] = (model, lin, attn, phase, maps)


# --- diag: which K quantizer survives Qwen2-1.5B layer-0 K (MXFP6-floor linears, Q/P/V MXFP8-floor) -----------
for kname, kmode in [("bf16", "bf16"), ("mxfp8_floor", F), ("mxfp8_v11", "mxfp8:ocp_v11"),
                     ("mxfp8_rceil", "mxfp8:rceil"), ("e5m2_floor", "e5m2:floor"), ("mxint8", "mxint8"),
                     ("mxfp8tok_floor", "mxfp8tok:ocp_floor"), ("offset_mxfp8_floor", "off+" + F),
                     ("mean_mxfp8_floor", "mean+" + F)]:
    _row(f"diag_qwen_k_{kname}", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), False),
         {**ALL_F, "k": kmode}, "diag", "Q14 mechanism")

# --- full: GPT2-Large (MIDAS Fig. 1(b) and Table II) -----------------------------------------------------------
_row("ctrl_gpt2l_mx_v11", "gpt2-large", ("mxfp4", "ocp_v11", (), False),
     {k: "mxfp8:ocp_v11" for k in "qkpv"}, "full", "control = e09_wt2_extra mx_rtn")
_row("gpt2l_mxfp8_floor", "gpt2-large", ("mxfp8", "ocp_floor", (), False), ALL_F, "full", "Fig1b MX 8-bit 19.65")
_row("gpt2l_mxfp6_floor", "gpt2-large", ("mxfp6", "ocp_floor", (), False), ALL_F, "full", "Fig1b MX 6-bit 19.66")
_row("gpt2l_mxfp4_floor", "gpt2-large", ("mxfp4", "ocp_floor", (), False), ALL_F, "full", "Fig1b MX 4-bit 22.85")
_row("gpt2l_mix_floor", "gpt2-large", ("mxfp4", "ocp_floor", MIX, False), ALL_F, "full", "TabII GPT2-L 20.11")
_row("gpt2l_mix_floor_attn_bf16", "gpt2-large", ("mxfp4", "ocp_floor", MIX, False), ALL_BF16, "full",
     "TabII GPT2-L 20.11 (attention sensitivity)")
_row("gpt2l_mix_floor_lmhead", "gpt2-large", ("mxfp4", "ocp_floor", MIX, True), ALL_F, "full",
     "TabII GPT2-L 20.11 (lm_head MXFP4 sensitivity)")
_row("gpt2l_mxfp4_floor_attn_bf16", "gpt2-large", ("mxfp4", "ocp_floor", (), False), ALL_BF16, "full",
     "Fig1b MX 4-bit 22.85 (attention sensitivity)")
# --- full: Qwen2-1.5B (MIDAS Table II MXFP6 11.12) ---------------------------------------------------------------
_row("qwen_mxfp6_floor", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), False), ALL_F, "full", "TabII Qwen 11.12 literal")
_row("qwen_mxfp6_floor_attn_bf16", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), False), ALL_BF16, "full",
     "TabII Qwen 11.12 (BF16 BMMs)")
_row("qwen_mxfp6_floor_kbf16", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), False), {**ALL_F, "k": "bf16"}, "full",
     "TabII Qwen 11.12 (K BF16, Q/P/V MXFP8)")
_row("qwen_mxfp6_floor_koffset", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), False), {**ALL_F, "k": "off+" + F},
     "full", "TabII Qwen 11.12 (static K offset, all MXFP8; non-MIDAS)")
_row("qwen_mxfp6_floor_kbf16_lmhead", "qwen2-1.5b", ("mxfp6", "ocp_floor", (), True), {**ALL_F, "k": "bf16"},
     "full", "TabII Qwen 11.12 (K BF16 + lm_head MXFP6 sensitivity)")

# ---------------------------------------------------------------------------------------------------------------
A: dict = {"spec": ALL_BF16, "offsets": None, "collect": None}


def _mx_e5m2(v: torch.Tensor) -> torch.Tensor:
    amax = v.abs().amax(-1)
    xexp = torch.clamp(torch.where(amax > 0, _floor_log2(amax) - 15, torch.full_like(amax, -127, dtype=torch.int64)),
                       -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    q = torch.clamp(v / s, -57344.0, 57344.0).to(torch.float8_e5m2).to(torch.float64)
    return q * s


def _mxint8(v: torch.Tensor) -> torch.Tensor:
    amax = v.abs().amax(-1)
    xexp = torch.clamp(torch.where(amax > 0, _floor_log2(amax), torch.full_like(amax, -127, dtype=torch.int64)),
                       -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp - 6)[..., None]  # element LSB = X * 2^-6
    return torch.clamp(torch.round(v / s), -128, 127) * s  # torch.round = half to even


def _block_q(x: torch.Tensor, dim: int, mode: str) -> torch.Tensor:
    """Quantize x in blocks of 32 along `dim` with `mode` (mxfp8:<rule> | e5m2:floor | mxint8)."""
    xt = x.transpose(dim, -1).to(torch.float64)
    pad = (-xt.shape[-1]) % 32
    if pad:
        xt = torch.nn.functional.pad(xt, (0, pad))
    if mode.startswith("mxfp8:"):
        q = fq_mxfp8(xt, mode.split(":", 1)[1])
    else:
        v = xt.reshape(*xt.shape[:-1], xt.shape[-1] // 32, 32)
        q = (_mx_e5m2(v) if mode == "e5m2:floor" else _mxint8(v) if mode == "mxint8" else None)
        if q is None:
            raise ValueError(mode)
        q = q.reshape(xt.shape)
    if pad:
        q = q[..., : xt.shape[-1] - pad]
    return q.transpose(dim, -1).to(x.dtype)


def _quant_k(key: torch.Tensor, mode: str, layer: int) -> torch.Tensor:
    """K (B, Hkv, T, D), before the GQA repeat."""
    if mode == "bf16":
        return key
    if mode.startswith("mxfp8tok:"):
        return _block_q(key, -2, "mxfp8:" + mode.split(":", 1)[1])
    if mode.startswith("off+"):
        c = A["offsets"][layer].to(key.device, torch.float32)[None, :, None, :]
        return _block_q((key.float() - c), -1, mode[4:])  # softmax-invariant shift; no add-back needed
    if mode.startswith("mean+"):
        mu = key.float().mean(dim=-2, keepdim=True)
        return _block_q(key.float() - mu, -1, mode[5:])
    return _block_q(key, -1, mode)


def e25_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    layer, spec = getattr(module, "layer_idx", 0), A["spec"]
    if A["collect"] is not None:
        s, n = A["collect"].get(layer, (0.0, 0))
        A["collect"][layer] = (s + key.detach().float().sum(dim=(0, 2)).cpu(), n + key.shape[0] * key.shape[2])
    k = _quant_k(key, spec["k"], layer)
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    q = query if spec["q"] == "bf16" else _block_q(query, -1, spec["q"])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scaling
    scores = _apply_mask(scores, attention_mask, key.shape[-2])
    probs = torch.softmax(scores, dim=-1)
    p = probs if spec["p"] == "bf16" else _block_q(probs, -1, spec["p"])
    v = value if spec["v"] == "bf16" else _block_q(value, -2, spec["v"])
    out = torch.matmul(p.float(), v.float()).to(query.dtype)
    return out.transpose(1, 2).contiguous(), probs


_FQ = cim_linear.fake_quant
CHUNK_ROWS = 8192


def _fake_quant_chunked(x, cfg, tensor_amax=None):
    """Row-chunked weight fake-quant for MX formats (T6.1): MX blocks lie along the last axis, so each row is
    quantized independently and chunking over rows is bit-exact. Avoids the float64 OOM on Qwen2-1.5B's
    151936x1536 lm_head (8 GB GPU). NVFP4 (tensor-level scale) and small tensors take the original path."""
    if cfg.fmt.startswith("mx") and x.dim() == 2 and x.shape[0] > CHUNK_ROWS:
        return torch.cat([_FQ(c, cfg, tensor_amax) for c in torch.split(x, CHUNK_ROWS, dim=0)], dim=0)
    return _FQ(x, cfg, tensor_amax)


cim_linear.fake_quant = _fake_quant_chunked


def build(spec: dict, rev: str, lin: tuple):
    fmt, rule, overrides, lm_head = lin
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16).to("cuda").eval()
    excl = ("embed", "wte", "wpe") if lm_head else ("lm_head", "embed", "wte", "wpe")
    cc = CIMConfig(fmt=fmt, scale_rule=rule, attn_policy="bf16", fmt_overrides=overrides, exclude=excl)
    rec = patch_model(model, cc)
    install(model, "bf16")  # registers cim_eager + its eager mask builder; then swap in the local function
    AttentionInterface.register("cim_eager", e25_attention)
    return model, cc, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="diag,full")
    ap.add_argument("--rows", default=None, help="comma list of row names (default: all rows of the phases)")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    train, _ = wikitext2_text("train", ds_rev)
    names = a.rows.split(",") if a.rows else [n for n, r in ROWS.items() if r[3] in a.phases.split(",")]
    cfg = {**vars(a), "dataset_revision": ds_rev, "seqlen": SEQLEN, "diag_windows": DIAG_WINDOWS,
           "calib_windows": CALIB_WINDOWS, "calib_split": "train",
           "rows": {n: {"model": ROWS[n][0], "linear": ROWS[n][1], "attn": ROWS[n][2], "phase": ROWS[n][3],
                        "maps_to": ROWS[n][4]} for n in names}}
    with start_run("e25_midas_accuracy", cfg) as run:
        ck = RowCheckpoint("e25_midas_accuracy", {"dataset_revision": ds_rev, "seqlen": SEQLEN},
                           ["src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "experiments/configs",
                            "experiments/e25_midas_accuracy"])
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        rows, toks = [], {}
        for name in names:
            mname, lin, attn, phase, maps = ROWS[name]
            row = ck.get(name)
            if row is None:
                spec = specs[mname]
                rev = spec["revision"] or model_info(spec["hf_id"]).sha
                if mname not in toks:
                    toks[mname] = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
                model, cc, rec = build(spec, rev, lin)
                A["offsets"] = None
                if attn["k"].startswith("off+"):  # calibrate the static K offset on the train split
                    A.update(spec=ALL_BF16, collect={})
                    perplexity(model, toks[mname], train, seqlen=SEQLEN, max_windows=CALIB_WINDOWS)
                    A["offsets"] = {lay: s / n for lay, (s, n) in A["collect"].items()}
                    A["collect"] = None
                A["spec"] = attn
                r = perplexity(model, toks[mname], text, seqlen=SEQLEN,
                               max_windows=DIAG_WINDOWS if phase == "diag" else None)
                row = {"row": name, "model": mname, "revision": rev, "phase": phase, "maps_to": maps,
                       "fmt": lin[0], "scale_rule": lin[1], "fmt_overrides": str(lin[2]), "lm_head_quant": lin[3],
                       "q": attn["q"], "k": attn["k"], "p": attn["p"], "v": attn["v"], "seqlen": SEQLEN,
                       "n_patched": len(rec["patched"]), **r.as_dict()}
                ck.put(row, name)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            print(f"{name:34s} token-PPL {row['token_ppl']:10.4f}  ({row['n_windows']} windows)", flush=True)
            run.save_csv("ppl.csv", rows)
        run.log_metrics({f"{r['row']}_token_ppl": r["token_ppl"] for r in rows})


if __name__ == "__main__":
    main()
