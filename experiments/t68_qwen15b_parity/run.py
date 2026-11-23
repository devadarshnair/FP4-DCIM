"""T6.8 - hardware-mode parity (e06 protocol) on Qwen2-1.5B, largest-K layers.

Qwen2-1.5B: hidden 1536, intermediate 8960 -> the largest reduction is K = 8960 in every `mlp.down_proj`
(28 layers; every other projection has K = 1536). Those 28 layers run on the L1 backend; all other layers use
the fp32 fake-quant path (== L1 Exact at realistic spreads). Per format (MXFP4, NVFP4) three modes on the
full WikiText-2 pass: `fakequant` (all layers), `down_l1_exact`, `down_l1_hier40_kt512` (W=40, Kt=512;
K=8960 = 17 full tiles + one half tile). Acceptance: |PPL(Hier) - PPL(Exact)| <= 0.01 (acceptance criterion).
NVFP4 uses the static calibrated activation tensor scale (T6.6, `nv_act_scale="static"`, 32 WikiText-2 train
windows, seed 0); attention BMMs BF16 (MXFP8 K destroys this model's layer-0 K channels; identical in all
modes). Each (fmt, mode) is a checkpoint row; run with --paper from a clean tree.

Run: python experiments/t68_qwen15b_parity/run.py [--fmts mxfp4,nvfp4] [--modes ...] [--max-windows N] [--paper]

Full pass in window shards (remote GPU run): `--shard i/n`
evaluates windows [i*N/n, (i+1)*N/n) of the N full-pass windows (same non-overlapping windows as
llm.ppl.perplexity) and stores a ("shard", fmt, mode, i, n) row with its nll_sum / n_tokens / n_words. With
--assemble (no --max-windows), a (fmt, mode) without a full-pass row is combined from a complete shard set:
token-PPL = exp(sum nll / sum tokens), identical to one pass up to float summation order. Run the NV fakequant
row first: it writes the calibration that the shards reuse.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.golden.accum import Exact, Hier
from cimfp4.llm.act_calib import apply_nv_act_amax, calib_batches, calibrate_nv_act_amax, saturation_stats
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, token_ppl, wikitext2_text, word_ppl
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

EXP = "t68_qwen15b_parity"
MODEL = "qwen2-1.5b"
CALIB = {"nsamples": 32, "seed": 0}
LARGE_K = "down_proj"


def block(fmt):
    return 32 if fmt == "mxfp4" else 16


def modes_for(fmt):
    return {
        "fakequant": None,
        "down_l1_exact": Exact(),
        "down_l1_hier40_kt512": Hier(512 // block(fmt), 40),
    }


def build(spec, rev, fmt, accum, scale):
    m = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    base = dict(fmt=fmt, attn_policy="bf16", nv_act_scale=scale)
    if accum is None:
        patch_model(m, CIMConfig(**base))
    else:  # large-K layers on L1, everything else fake-quant
        patch_model(m, CIMConfig(**base, exclude=("lm_head", "embed", "wte", "wpe", LARGE_K)))
        rec = patch_model(m, CIMConfig(**base, backend="l1", accum=accum, exclude=("lm_head", "embed")))
        assert len(rec["patched"]) == 28 and all(LARGE_K in n for n in rec["patched"]), rec["patched"]
        assert all(m.get_submodule(n).in_features == 8960 for n in rec["patched"])
    return m


@torch.no_grad()  # as llm.ppl.perplexity (without it the autograd graph is kept: 7.9 GB and >10 min for 1 window)
def ppl_shard(model, tok, text, seqlen, i, n, device="cuda"):
    """llm.ppl.perplexity restricted to shard i of n of the full-pass windows (same windows, same loss)."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_all = len(ids) // seqlen
    w0, w1 = i * n_all // n, (i + 1) * n_all // n
    nll = 0.0
    for w in range(w0, w1):
        chunk = ids[w * seqlen:(w + 1) * seqlen][None].to(device)
        logits = model(chunk).logits.float()
        nll += float(torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum"))
    n_tok = (w1 - w0) * (seqlen - 1)
    n_words = len(text.split()) * ((w1 - w0) * seqlen) / len(ids)
    return {"nll_sum": nll, "n_tokens": n_tok, "n_windows": w1 - w0, "n_words": n_words, "w0": w0, "w1": w1,
            "n_windows_full": n_all, "token_ppl": token_ppl(nll, n_tok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmts", default="mxfp4,nvfp4")
    ap.add_argument("--modes", default="fakequant,down_l1_exact,down_l1_hier40_kt512")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None, help="None = full pass")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: evaluate shard i of n of the full pass (no --max-windows)")
    a = ap.parse_args()
    sh = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    assert sh is None or (a.max_windows is None and 0 <= sh[0] < sh[1]), "--shard needs a full pass"
    seed_all(0)
    spec = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())[MODEL]
    rev = spec["revision"] or model_info(spec["hf_id"]).sha
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
    text, ds_rev = wikitext2_text()
    ck = RowCheckpoint(
        EXP, {"seqlen": a.seqlen, "max_windows": a.max_windows, "calib": CALIB, "rev": rev},
        ["src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "src/cimfp4/golden/accum.py",
         "src/cimfp4/golden/l1_dot.py", "src/cimfp4/utils/checkpoint.py", "experiments/configs", f"experiments/{EXP}"],
    )
    print(f"[checkpoint] {ck.path} rows: {len(ck)}", flush=True)
    if not a.assemble:
        for fmt in a.fmts.split(","):
            scale = "static" if fmt == "nvfp4" else "dynamic"  # dynamic is meaningless for MX (no tensor scale)
            for mode in a.modes.split(","):
                key = ("row", fmt, mode) if sh is None else ("shard", fmt, mode, *sh)
                if ck.get(*key) is not None:
                    print(f"[skip] {fmt} {mode}", flush=True)
                    continue
                t0 = time.time()
                m = build(spec, rev, fmt, modes_for(fmt)[mode], scale)
                row = {"model": MODEL, "revision": rev, "fmt": fmt, "mode": mode, "nv_act_scale": scale,
                       "attn_policy": "bf16"}
                if scale == "static":
                    if (cal := ck.get("calib", fmt)) is None:
                        cal = {"amax": calibrate_nv_act_amax(m, calib_batches(tok, CALIB["nsamples"], a.seqlen,
                                                                              CALIB["seed"], None))}
                        ck.put(cal, "calib", fmt)
                    apply_nv_act_amax(m, cal["amax"])
                torch.cuda.reset_peak_memory_stats()
                if sh is None:
                    row |= perplexity(m, tok, text, seqlen=a.seqlen, max_windows=a.max_windows).as_dict()
                else:
                    row |= ppl_shard(m, tok, text, a.seqlen, *sh) | {"shard": f"{sh[0]}/{sh[1]}"}
                if scale == "static":
                    st = saturation_stats(m)
                    st.pop("per_layer")
                    row |= {"sat_" + k: v for k, v in st.items()}
                row |= {"seconds": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
                ck.put(row, *key)
                print(f"[{key[0]}] {fmt} {mode} {a.shard or ''} token-PPL {row['token_ppl']:.4f} ({row['seconds']:.0f}s)", flush=True)
                del m
                gc.collect()
                torch.cuda.empty_cache()
        return
    # Rows are matched by key and window count, not by checkpoint tag: the tag embeds the git rev of
    # experiments/configs, which other sessions advanced (paper_runs.yaml pins only; `git diff 930868c HEAD` over
    # src/cimfp4/{llm,kernels,formats}, golden/{accum,l1_dot}.py and this directory is empty) while the jobs ran.
    import json

    want = a.max_windows
    rows, shards = {}, {}
    for line in ck.path.read_text().splitlines():
        rec = json.loads(line)
        k = json.loads(rec["key"])
        if k[0] == "row" and (want is None or rec["row"].get("n_windows") == want):
            rows[(k[1], k[2])] = rec["row"]
        elif k[0] == "shard" and want is None:
            shards.setdefault((k[1], k[2], k[4]), {})[k[3]] = rec["row"]
    for (f, md, n), got in shards.items():  # combine a complete shard set into a full-pass row
        if (f, md) in rows or len(got) != n:
            continue
        parts = [got[i] for i in range(n)]
        assert [p["w0"] for p in parts[1:]] == [p["w1"] for p in parts[:-1]] and parts[-1]["w1"] == parts[0]["n_windows_full"]
        nll, nt, nw = (sum(p[x] for p in parts) for x in ("nll_sum", "n_tokens", "n_words"))
        r = {k: v for k, v in parts[0].items() if k not in ("w0", "w1", "shard")}
        r |= {"nll_sum": nll, "n_tokens": nt, "n_words": nw, "n_windows": sum(p["n_windows"] for p in parts),
              "token_ppl": token_ppl(nll, nt), "word_ppl": word_ppl(nll, nw), "n_shards": n,
              "seconds": sum(p["seconds"] for p in parts), "peak_vram_gb": max(p["peak_vram_gb"] for p in parts),
              **{"sat_" + x: max(p.get("sat_" + x) or 0 for p in parts) for x in ("max_ratio",) if "sat_max_ratio" in parts[0]}}
        rows[(f, md)] = r
    ck.get = lambda tag, f, m: rows.get((f, m))
    out = []
    with start_run(EXP, {**vars(a), "dataset_revision": ds_rev, "calib": CALIB, "bound": 0.01}, paper=a.paper) as run:
        for fmt in a.fmts.split(","):
            r = {md: ck.get("row", fmt, md) for md in modes_for(fmt)}
            assert all(r.values()), f"missing rows for {fmt}"
            p = {k: v["token_ppl"] for k, v in r.items()}
            d_h_e, d_h_f, d_e_f = (p["down_l1_hier40_kt512"] - p["down_l1_exact"],
                                   p["down_l1_hier40_kt512"] - p["fakequant"], p["down_l1_exact"] - p["fakequant"])
            out.append({"fmt": fmt, "nv_act_scale": r["fakequant"]["nv_act_scale"], **{f"ppl_{k}": v for k, v in p.items()},
                        "dppl_hier_vs_exact": d_h_e, "dppl_hier_vs_fakequant": d_h_f, "dppl_exact_vs_fakequant": d_e_f,
                        "bound": 0.01, "pass_hier_vs_exact": abs(d_h_e) <= 0.01, "pass_hier_vs_fakequant": abs(d_h_f) <= 0.01})
            print(out[-1], flush=True)
        run.save_csv("parity.csv", out)
        run.save_csv("rows.csv", [{k: v for k, v in ck.get("row", f, m).items()} for f in a.fmts.split(",") for m in modes_for(f)])
        run.log_metrics({f"{o['fmt']}_{k}": o[k] for o in out for k in o if k.startswith("dppl")})


if __name__ == "__main__":
    main()
