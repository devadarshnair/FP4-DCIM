"""e06 - hardware-mode parity and accumulator window/tile sweep (WP-16).

(a) parity on the full WikiText-2 pass: fakequant vs L1 Exact vs L1 Hier(W=40, Kt=512) for MX and NV.
(b) sweep W ∈ {24,28,32,36,40,48} × Kt ∈ {128, 512, 2048, K} (+ FP32Seq, AlignTrunc(35)) on the first
    `--sweep-windows` windows (documented subset), ΔPPL vs L1 Exact on the same subset; W* = minimum W
    with |ΔPPL| ≤ 0.01 for every Kt is the proposed window/tile setting.

Run: python experiments/e06_hw_parity_accum/run.py [--models a,b] [--sweep-windows 24] [--paper]
"""

from __future__ import annotations

import argparse
import gc

import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.golden.accum import AlignTrunc, Exact, FP32Seq, Hier, Window
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.checkpoint import RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

WS = (24, 28, 32, 36, 40, 48)
KTS = (128, 512, 2048, None)  # None = whole K (single window)


def tile_blocks(kt, fmt):
    return None if kt is None else kt // (32 if fmt == "mxfp4" else 16)


def run_one(spec, rev, tok, text, cc, seqlen, max_windows):
    model = (
        AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    patch_model(model, cc)
    r = perplexity(model, tok, text, seqlen=seqlen, max_windows=max_windows)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return r


def cached(ck, key, fn, *args):
    """Run fn() once per key (resumable): {"ok": True, **ppl} or {"ok": False, "error": ...} on window overflow."""
    if (row := ck.get(*key)) is not None:
        print(f"  (checkpoint) {key}", flush=True)
        return row
    try:
        row = {"ok": True, **fn(*args).as_dict()}
    except AssertionError as e:  # window overflow: the anchor/guard analysis rejects this W
        row = {"ok": False, "error": str(e)}
    ck.put(row, *key)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m,qwen2.5-0.5b")
    ap.add_argument("--fmts", default="mxfp4,nvfp4")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--sweep-windows", type=int, default=24)
    ap.add_argument("--parity-windows", type=int, default=None, help="None = full pass")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    seed_all(0)
    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    text, ds_rev = wikitext2_text()
    with start_run(
        "e06_hw_parity_accum", {**vars(a), "dataset_revision": ds_rev, "Ws": WS, "Kts": KTS}, paper=a.paper
    ) as run:
        parity, sweep, wstar = [], [], {}
        ck = RowCheckpoint(
            "e06_hw_parity_accum",
            {"seqlen": a.seqlen, "sweep_windows": a.sweep_windows, "parity_windows": a.parity_windows},
            [
                "src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "src/cimfp4/golden/accum.py", "src/cimfp4/golden/l1_dot.py", "src/cimfp4/utils/checkpoint.py", "experiments/configs",
                "experiments/e06_hw_parity_accum",
            ],
        )
        print(f"[checkpoint] {ck.path} reusable rows: {len(ck)}", flush=True)
        for name in a.models.split(","):
            spec = specs[name]
            rev = spec["revision"] or model_info(spec["hf_id"]).sha
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
            for fmt in a.fmts.split(","):
                base = dict(fmt=fmt, attn_policy="mxfp8")
                # (a) parity, full pass
                for label, cc in (
                    ("fakequant", CIMConfig(**base)),
                    ("l1_exact", CIMConfig(**base, backend="l1", accum=Exact())),
                    (
                        "l1_hier40_kt512",
                        CIMConfig(**base, backend="l1", accum=Hier(tile_blocks(512, fmt), 40)),
                    ),
                ):
                    r = cached(ck, ("parity", name, fmt, label),
                               run_one, spec, rev, tok, text, cc, a.seqlen, a.parity_windows)
                    if not r["ok"]:
                        raise AssertionError(f"parity run {name} {fmt} {label}: {r['error']}")
                    r = {k: v for k, v in r.items() if k != "ok"}
                    parity.append({"model": name, "fmt": fmt, "mode": label, **r})
                    print(f"[parity] {name} {fmt} {label:16s} token-PPL {r['token_ppl']:.4f}", flush=True)
                # (b) sweep on a subset
                ref = cached(ck, ("sweep", name, fmt, "ref_exact"), run_one, spec, rev, tok, text,
                             CIMConfig(**base, backend="l1", accum=Exact()), a.seqlen, a.sweep_windows)
                assert ref["ok"], ref
                modes = {"fp32seq": FP32Seq(), "aligntrunc35": AlignTrunc(35)}
                for w in WS:
                    for kt in KTS:
                        tb = tile_blocks(kt, fmt)
                        modes[f"W{w}_Kt{kt or 'K'}"] = (
                            Window(w) if tb is None else Hier(tb, w, reducer="fp32")
                        )
                ok_w = {w: True for w in WS}
                for label, mode in modes.items():
                    r = cached(ck, ("sweep", name, fmt, label), run_one, spec, rev, tok, text,
                               CIMConfig(**base, backend="l1", accum=mode), a.seqlen, a.sweep_windows)
                    if r["ok"]:
                        d, status = r["token_ppl"] - ref["token_ppl"], "ok"
                    else:
                        d, status = float("nan"), f"overflow: {r['error']}"
                    sweep.append(
                        {
                            "model": name,
                            "fmt": fmt,
                            "mode": label,
                            "delta_ppl": d,
                            "status": status,
                            "token_ppl": r.get("token_ppl"),
                            "ref_token_ppl": ref["token_ppl"],
                        }
                    )
                    print(f"[sweep] {name} {fmt} {label:16s} dPPL {d:+.5f} {status}", flush=True)
                    if label.startswith("W"):
                        w = int(label[1:].split("_")[0])
                        ok_w[w] &= status == "ok" and abs(d) <= 0.01
                wstar[f"{name}_{fmt}"] = min((w for w in WS if ok_w[w]), default=None)
        run.save_csv("parity.csv", parity)
        run.save_csv("ppl_vs_W.csv", sweep)
        run.log_metrics(
            {f"wstar_{k.replace('.', 'p').replace('-', '_')}": v for k, v in wstar.items()}
            | {"wstar_max": max((v for v in wstar.values() if v), default=None)}
        )
        print("W* per model/format:", wstar)


if __name__ == "__main__":
    main()
