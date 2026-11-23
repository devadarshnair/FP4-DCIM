"""e09 - accuracy table T2 (WP-19): lm-eval zero-shot + C4 PPL per (model, config), WikiText-2 merged in.

Configs (all W4A4 along K; attention BMMs in MXFP8 as MIDAS keeps attention at MXFP8/6):
    bf16                 unpatched reference
    mx_rtn / nv_rtn      RTN, fake-quant (== L1 Exact at realistic spreads)
    mx_hw / nv_hw        RTN on the hardware accumulation path: L1 Hier(Kt=512, W=40) (--hw; slow)
    mx_gptq / nv_gptq / nv_gptq_4over6   GPTQ weights saved by e07 (seed 0), then patched
    mxint4_rtn           D2 surrogate MXINT4-SM (d2_int4_contract.md v1.1), RTN
    midas_mix / mxfp6    MIDAS Table II precision (MXFP4 + MXFP6 attention projections; MXFP6 W/A), RTN
WikiText-2 token-PPL is merged from the latest e02 (BF16), e03 (RTN), e06 (HW parity), e07 (GPTQ, 3-seed mean)
runs (and e09_wt2_extra for the remaining configs) rather than recomputed. Each (model, config) row is checkpointed (build/checkpoints/e09_accuracy_table.jsonl).

Run: python experiments/e09_accuracy_table/run.py [--models ...] [--limit N] [--hw] [--hw-limit 500] [--paper]
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.golden.accum import Hier
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.lmeval import C4_WINDOWS, TASKS, c4_text, run_lmeval
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

RES = REPO_ROOT / "results"
GPTQ_DIR = REPO_ROOT / "build" / "gptq"


def configs(q: dict, hw: bool) -> dict:
    mx, nv = q["configs"]["mxfp4_v11"], q["configs"]["nvfp4_hw"]
    c = {
        "bf16": None,
        "mx_rtn": ("hf", CIMConfig(**mx)),
        "nv_rtn": ("hf", CIMConfig(**nv)),
        "mx_gptq": ("mxfp4_gptq", CIMConfig(**mx)),
        "nv_gptq": ("nvfp4_gptq", CIMConfig(**nv)),
        "nv_gptq_4over6": ("nvfp4_gptq_4over6", CIMConfig(**nv)),
        # MIDAS Table II protocol (docs/spec/midas_protocol.md): GPT2-L = MXFP4 with MXFP6 attention
        # projections; Llama/Qwen = MXFP6 W/A. Attention BMMs MXFP8 in both (mx config's attn_policy).
        "midas_mix": ("hf", CIMConfig(**mx, fmt_overrides=(("attn.", "mxfp6"),))),
        "mxfp6": ("hf", CIMConfig(**{**mx, "fmt": "mxfp6"})),
        # D2 surrogate (docs/spec/d2_int4_contract.md v1.1): MXINT4-SM W/A, floor_sat scale, RTN
        "mxint4_rtn": ("hf", CIMConfig(**{**mx, "fmt": "mxint4"})),
    }
    if hw:
        c["mx_hw"] = ("hf", CIMConfig(**mx, backend="l1", accum=Hier(512 // 32, 40)))
        c["nv_hw"] = ("hf", CIMConfig(**nv, backend="l1", accum=Hier(512 // 16, 40)))
    return c


def _latest_csv(exp: str, name: str) -> list[dict]:
    p = RES / exp / "latest" / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def wikitext_lookup() -> dict:
    """(model, config) -> WikiText-2 token-PPL from earlier experiments (None where not yet run)."""
    w = {}
    for r in _latest_csv("e02_ppl_baseline", "bf16_ppl.csv"):
        w[(r["model"], "bf16")] = float(r["token_ppl"])
    for r in _latest_csv("e03_w4a4_rtn", "ppl.csv"):
        key = {"mxfp4_v11": "mx_rtn", "nvfp4_hw": "nv_rtn"}.get(r["config"])
        if key:
            w[(r["model"], key)] = float(r["token_ppl"])
    for r in _latest_csv("e06_hw_parity_accum", "parity.csv"):
        if r["mode"] == "l1_hier40_kt512":
            w[(r["model"], {"mxfp4": "mx_hw", "nvfp4": "nv_hw"}[r["fmt"]])] = float(r["token_ppl"])
    acc: dict = {}
    for r in _latest_csv("e07_gptq", "ppl.csv"):
        key = {"mxfp4_gptq": "mx_gptq", "nvfp4_gptq": "nv_gptq", "nvfp4_gptq_4over6": "nv_gptq_4over6"}[r["recipe"]]
        acc.setdefault((r["model"], key), []).append(float(r["token_ppl"]))
    w |= {k: sum(v) / len(v) for k, v in acc.items()}
    # Configs not covered above (midas_mix, mxfp6, mxint4_rtn): T2-protocol rows of e09_wt2_extra.
    for r in _latest_csv("e09_wt2_extra", "ppl.csv"):
        if r["seqlen_requested"] == "2048":
            w.setdefault((r["model"], r["config"]), float(r["token_ppl"]))
    return w


def load(spec, rev, source, name):
    if source in (None, "hf"):
        path, kw = spec["hf_id"], {"revision": rev}
    else:
        path, kw = GPTQ_DIR / f"{name}__{source}", {}
        if not Path(path).exists():
            return None
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kw).to("cuda").eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m,qwen2.5-0.5b,gpt2-large")
    ap.add_argument("--configs", default=None, help="comma list; default all available")
    ap.add_argument("--limit", type=int, default=None, help="lm-eval docs per task (None = full)")
    ap.add_argument("--hw", action="store_true", help="include L1 Hier hardware-path rows (slow)")
    ap.add_argument("--hw-limit", type=int, default=500, help="lm-eval docs per task for the hw rows")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--run-name", default="e09_accuracy_table", help="results/<run-name>; use another name for "
                    "partial config subsets so results/e09_accuracy_table/latest stays the full T2 table")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    q = yaml.safe_load((REPO_ROOT / "experiments/configs/quant.yaml").read_text())
    cfgs = configs(q, a.hw)
    if a.configs:
        cfgs = {k: v for k, v in cfgs.items() if k in a.configs.split(",")}
    c4, c4_rev = c4_text()
    cfg = {**vars(a), "tasks": TASKS, "c4_revision": c4_rev, "c4_windows": C4_WINDOWS, "configs": list(cfgs)}
    with start_run(a.run_name, cfg, paper=a.paper) as run:
        ck = RowCheckpoint(
            "e09_accuracy_table",
            {k: cfg[k] for k in ("limit", "hw_limit", "seqlen", "tasks", "c4_revision")},
            [
                "src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "src/cimfp4/golden/accum.py", "src/cimfp4/golden/l1_dot.py", "src/cimfp4/utils/checkpoint.py", "experiments/configs",
                "experiments/e09_accuracy_table",
            ],
        )
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        rows, wt = [], wikitext_lookup()
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            for cname, c in cfgs.items():
                row = ck.get(name, cname)
                if row is None:
                    source, cc = (None, None) if c is None else c
                    model = load(spec, rev, source, name)
                    if model is None:
                        print(f"{name:14s} {cname:16s} skipped (no GPTQ weights in {GPTQ_DIR})", flush=True)
                        continue
                    if cc is not None:
                        patch_model(model, cc)
                    seqlen = min(a.seqlen, getattr(model.config, "max_position_embeddings", a.seqlen))
                    lim = a.hw_limit if cname.endswith("_hw") else a.limit
                    le = run_lmeval(model, tok, batch_size=a.batch_size, limit=lim, max_length=seqlen)
                    c4r = perplexity(model, tok, c4, seqlen=seqlen, max_windows=C4_WINDOWS)
                    row = {
                        "model": name,
                        "revision": rev,
                        "config": cname,
                        "quant": None if cc is None else cc.as_dict(),
                        "lmeval_limit": lim,
                        "c4_token_ppl": c4r.token_ppl,
                        "c4_windows": c4r.n_windows,
                        **{f"{t}": v["value"] for t, v in le.items() if t != "_meta"},
                        **{f"{t}_stderr": v["stderr"] for t, v in le.items() if t != "_meta"},
                        "lambada_ppl": le["lambada_openai"].get("perplexity"),
                        "lmeval_meta": le["_meta"],
                    }
                    ck.put(row, name, cname)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                row = {**row, "wikitext2_token_ppl": wt.get((name, cname))}
                avg = [row[t] for t in TASKS if row.get(t) is not None]
                row["avg_acc"] = sum(avg) / len(avg) if avg else None
                rows.append(row)
                print(
                    f"{name:14s} {cname:16s} avg {row['avg_acc']:.4f}  C4 {row['c4_token_ppl']:.3f}  "
                    f"WT2 {row['wikitext2_token_ppl']}",
                    flush=True,
                )
        flat = [{k: v for k, v in r.items() if k not in ("lmeval_meta", "quant")} for r in rows]
        run.save_csv("t2.csv", flat)
        run.path("lmeval_meta.yaml").write_text(
            yaml.safe_dump({f"{r['model']}/{r['config']}": r["lmeval_meta"] for r in rows}, sort_keys=True)
        )
        run.log_metrics(
            {f"{r['model'].replace('.', 'p').replace('-', '_')}_{r['config']}_avg_acc": r["avg_acc"] for r in rows}
        )


if __name__ == "__main__":
    main()
