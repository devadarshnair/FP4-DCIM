"""T6.6 - NVFP4 rows with a calibrated static per-layer activation tensor scale.

Reruns the NV accuracy rows of e03 / e06 / e07(seed 0) / e09 with `nv_act_scale="static"`
(cimfp4.llm.act_calib: max calibration on 32 random WikiText-2 *train* windows, seed 0, activations unquantized
during calibration), i.e. the fixed `act_quant_nv` register c = fp32(s_enc/6) instead of the per-call amax of
the whole activation tensor. Every other setting follows the original protocols:

    part   protocol                                                         old dynamic rows
    wt2    WikiText-2 test token-PPL, seqlen 2048 (GPT2-L 1024), e03 / e06 / e07 harness   e03, e06 parity, e07 seed 0
    c4     C4 PPL, 256 windows (e09)                                        e09 t2.csv
    lm:<t> lm-eval 0.4.13 zero-shot, full task, batch 8 (e09), ONE task per call   e09 t2.csv

Configs (all NVFP4 "hw" convention, W4A4 along K, attention BMM operands MXFP8 unless stated):
    nv_rtn            RTN fake-quant                   (e03 nvfp4_hw, e06 fakequant, e09 nv_rtn)
    nv_rtn_attn_bf16  RTN, BF16 attention BMMs          (e03 ablation, SmolLM2 only)
    nv_l1_exact       L1 Exact accumulation             (e06 l1_exact)
    nv_hw             L1 Hier(Kt=512, W=40)             (e06 l1_hier40_kt512, T2 nv_hw)
    nv_gptq / nv_gptq_4over6   e07 seed-0 GPTQ weights   (e07 seed 0, e09)

Calibration is per (model, weight source, attention policy) and shared by the fake-quant and L1 backends (the
calibration forward is backend-independent). Each (model, config, part) result is a checkpoint row
(build/checkpoints/t66_nv_static_scale.jsonl) with its own platform/commit record, so long sweeps run as
separate < 1 h jobs. `--assemble` writes one results run with all rows, the calibration amax and the delta table
against the dynamic rows. `--scale dynamic` reruns a config with the old scale on this platform (controls).

Caveats (read before quoting the delta table):
  - lm-eval rows: the old e09 rows ran on an L4 at batch 8 (MXINT4 locally at batch 2), these rows on the local
    RTX 4060 at batch 8 / 2 (see the `gpu`, `batch_size` columns). Task-accuracy deltas therefore confound the
    static scale with GPU and batch effects and are at sampling-noise level (SmolLM2/Qwen task n=~300-2400);
    only the PPL deltas, and the `dynamic` control rows rerun on this platform (`delta_vs_local`), isolate the scale.
  - GPTQ rows use e07 seed 0 only (no 3-seed mean/std); the static scale was not recalibrated per seed.
  - The e06 sweep was NOT rerun: only its l1_exact / l1_hier40_kt512 / fakequant NV rows (SmolLM2, Qwen) were.
  - Calibration = 32 random WikiText-2 train windows (seed 0); PPL/eval data are test / C4 / lm-eval, disjoint.

Run: python experiments/t66_nv_static_scale/run.py --models smollm2-360m --configs nv_rtn --parts calib,wt2,c4
     python experiments/t66_nv_static_scale/run.py --assemble
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
import yaml

from cimfp4.golden.accum import Exact, Hier
from cimfp4.llm.act_calib import apply_nv_act_amax, calib_batches, calibrate_nv_act_amax, saturation_stats
from cimfp4.llm.config import CIMConfig
from cimfp4.llm.lmeval import C4_WINDOWS, TASKS, c4_text, run_lmeval
from cimfp4.llm.patch import patch_model
from cimfp4.llm.ppl import perplexity, wikitext2_text
from cimfp4.utils.checkpoint import CKPT_DIR, RowCheckpoint
from cimfp4.utils.provenance import REPO_ROOT, env_info, git_info
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

EXP = "t66_nv_static_scale"
RES = REPO_ROOT / "results"
GPTQ_DIRS = [REPO_ROOT / "results/gptq_archive/gptq", REPO_ROOT / "build/gptq"]
# Pinned to the revisions of the dynamic rows being compared (results/e03_w4a4_rtn, e09_accuracy_table configs).
MODEL_REV = {
    "smollm2-360m": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "qwen2.5-0.5b": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    # T6.5 (remote GPU run) and the Qwen2-1.5B NV static rows
    "qwen2-1.5b": "8a16abf2848eda07cc5253dec660bf1ce007ad7a",  # = results/t68_qwen15b_parity
    "llama-3.2-1b": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "llama-3.2-3b": "13afe5124825b4f3751f836b40dafda64c1ed062",
}
# models not in experiments/configs/models.yaml (kept here so that file, and every checkpoint tag over
# experiments/configs, is unchanged)
EXTRA_SPECS = {"llama-3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B", "license": "Llama-3.2-Community (gated)",
                                "revision": None}}
WT2_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REV = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
CALIB = {"nsamples": 32, "seed": 0, "dataset": "wikitext-2-raw-v1 train", "revision": WT2_REV, "stat": "max"}
BASE = {"fmt": "nvfp4", "nv_convention": "hw", "attn_policy": "mxfp8"}
CONFIGS = {  # name -> (weight source, CIMConfig kwargs)
    "nv_rtn": ("hf", {}),
    "nv_rtn_attn_bf16": ("hf", {"attn_policy": "bf16"}),
    "nv_l1_exact": ("hf", {"backend": "l1", "accum": Exact()}),
    "nv_hw": ("hf", {"backend": "l1", "accum": Hier(512 // 16, 40)}),
    "nv_gptq": ("nvfp4_gptq", {}),
    "nv_gptq_4over6": ("nvfp4_gptq_4over6", {}),
    # T6.5 controls on the same harness (run with --scale dynamic; no tensor scale exists for them):
    # bf16 = unpatched, mx_rtn = quant.yaml mxfp4_v11 (e09 mx_rtn)
    "bf16": ("hf", None),
    "mx_rtn": ("hf", {"fmt": "mxfp4", "scale_rule": "ocp_v11", "attn_policy": "mxfp8"}),
}
NON_NV = ("bf16", "mx_rtn")
CODE_PATHS = [
    "src/cimfp4/llm", "src/cimfp4/kernels", "src/cimfp4/formats", "src/cimfp4/golden/accum.py",
    "src/cimfp4/utils/checkpoint.py", "experiments/configs/models.yaml",
]  # only files that determine the rows (run.py edits and unrelated config files such as paper_runs.yaml do not
# invalidate rows); every row records its own commit


def checkpoint() -> RowCheckpoint:
    return RowCheckpoint(EXP, {"calib": CALIB, "c4_windows": C4_WINDOWS, "wt2": WT2_REV, "c4": C4_REV}, CODE_PATHS)


def gptq_dir(model: str, source: str) -> Path:
    for d in GPTQ_DIRS:
        if (p := d / f"{model}__{source}").exists():
            return p
    raise FileNotFoundError(f"no GPTQ weights for {model}__{source} in {GPTQ_DIRS}")


def load_model(spec, model, source):
    from transformers import AutoModelForCausalLM

    path, kw = (spec["hf_id"], {"revision": MODEL_REV[model]}) if source == "hf" else (gptq_dir(model, source), {})
    # device_map="cuda" streams the weights to the GPU (GPT2-L's fp32 checkpoint peaked at 3.9 GB host RSS when
    # converted on the CPU); the bf16 weights are the same either way.
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", **kw).eval()


def platform() -> dict:
    e, g = env_info(), git_info()
    return {"gpu": e.get("gpu"), "torch": e.get("torch"), "cuda": e.get("cuda"), "commit": g["sha"],
            "dirty": g["dirty"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def seqlen_of(model, req: int) -> int:
    return min(req, getattr(model.config, "max_position_embeddings", req) or req)


def get_calib(ck, specs, tok, model, source, attn, seqlen_req):
    key = ("calib", model, source, attn)
    if (row := ck.get(*key)) is not None:
        return row
    m = load_model(specs[model], model, source)
    patch_model(m, CIMConfig(**{**BASE, "attn_policy": attn}, nv_act_scale="static"))
    sl = seqlen_of(m, seqlen_req)
    t0 = time.time()
    amax = calibrate_nv_act_amax(m, calib_batches(tok, CALIB["nsamples"], sl, CALIB["seed"], WT2_REV))
    row = {"model": model, "source": source, "attn_policy": attn, "calib_seqlen": sl, **CALIB, "amax": amax,
           "seconds": time.time() - t0, **platform()}
    ck.put(row, *key)
    print(f"[calib] {model} {source} attn={attn}: {len(amax)} layers, amax {min(amax.values()):.3g}"
          f"..{max(amax.values()):.3g}", flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_part(ck, specs, tok, model, cname, part, scale, a):
    key = ("row", model, cname, scale, part)
    if ck.get(*key) is not None:
        print(f"[skip] {key} (checkpoint)", flush=True)
        return
    source, kw = CONFIGS[cname]
    if cname in NON_NV and scale != "dynamic":
        raise SystemExit(f"{cname}: run with --scale dynamic (no NV tensor scale)")
    if kw is None:
        cc = None
    elif cname in NON_NV:
        cc = CIMConfig(**kw)
    else:
        cc = CIMConfig(**{**BASE, **kw}, nv_act_scale=scale)
    if scale == "static":
        cal = get_calib(ck, specs, tok, model, source, cc.attn_policy, a.seqlen)
    m = load_model(specs[model], model, source)
    rec = patch_model(m, cc) if cc is not None else {"patched": []}
    if scale == "static":
        apply_nv_act_amax(m, cal["amax"])
    sl = seqlen_of(m, a.seqlen)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    row = {"model": model, "revision": MODEL_REV[model], "config": cname, "nv_act_scale": scale, "part": part,
           "source": source, "quant": None if cc is None else cc.as_dict(), "n_patched": len(rec["patched"]), "seqlen": sl}
    if part == "wt2":
        text, _ = wikitext2_text(revision=WT2_REV)
        row |= perplexity(m, tok, text, seqlen=sl, max_windows=a.smoke_limit).as_dict()
    elif part == "c4":
        text, _ = c4_text(revision=C4_REV)
        row |= perplexity(m, tok, text, seqlen=sl, max_windows=a.smoke_limit or C4_WINDOWS).as_dict()
    elif part.startswith("lm:"):
        task = part[3:]
        le = run_lmeval(m, tok, tasks={task: TASKS[task]}, batch_size=a.batch_size, max_length=sl, limit=a.smoke_limit)
        row |= {"task": task, **le[task], "batch_size": a.batch_size, "lmeval_meta": le["_meta"]}
    else:
        raise ValueError(part)
    if scale == "static":
        st = saturation_stats(m)
        worst = sorted(st.pop("per_layer").items(), key=lambda kv: -kv[1]["max_ratio"])[:5]
        row |= {"sat_" + k: v for k, v in st.items()} | {"sat_worst": worst}
    if a.smoke_limit:
        row["smoke_limit"] = a.smoke_limit
    row |= {"seconds": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, **platform()}
    ck.put(row, *key)
    print(f"[row] {model} {cname} {scale} {part}: "
          + ", ".join(f"{k}={row[k]}" for k in ("token_ppl", "value", "sat_max_ratio", "seconds") if k in row),
          flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- assemble
def _csv(exp, name):
    p = RES / exp / "latest" / name
    return (list(csv.DictReader(p.open())), str((RES / exp / "latest").resolve().relative_to(RES))) if p.exists() else ([], None)


def dynamic_reference() -> dict:
    """(model, config, metric) -> (value, source run) for the pre-T6.6 dynamic-scale rows."""
    ref = {}
    rows, src = _csv("e03_w4a4_rtn", "ppl.csv")
    for r in rows:
        c = {"nvfp4_hw": "nv_rtn", "nvfp4_hw_attn_bf16": "nv_rtn_attn_bf16"}.get(r["config"])
        if c:
            ref[(r["model"], c, "wt2")] = (float(r["token_ppl"]), f"{src} (e03)")
    rows, src = _csv("e06_hw_parity_accum", "parity.csv")
    for r in rows:
        c = {"l1_exact": "nv_l1_exact", "l1_hier40_kt512": "nv_hw"}.get(r["mode"])
        if r["fmt"] == "nvfp4" and c:
            ref[(r["model"], c, "wt2")] = (float(r["token_ppl"]), f"{src} (e06)")
    rows, src = _csv("e07_gptq", "ppl.csv")
    for r in rows:
        c = {"nvfp4_gptq": "nv_gptq", "nvfp4_gptq_4over6": "nv_gptq_4over6"}.get(r["recipe"])
        if c and str(r["seed"]) == "0":
            ref[(r["model"], c, "wt2")] = (float(r["token_ppl"]), f"{src} (e07 seed 0)")
    rows, src = _csv("e09_accuracy_table", "t2.csv")
    for r in rows:
        if r["config"] in ("nv_rtn", "nv_gptq", "nv_gptq_4over6"):
            ref[(r["model"], r["config"], "c4")] = (float(r["c4_token_ppl"]), f"{src} (e09)")
            for t in TASKS:
                if r.get(t):
                    ref[(r["model"], r["config"], f"lm:{t}")] = (float(r[t]), f"{src} (e09)")
    return ref


def assemble(a) -> None:
    ck = checkpoint()
    rows = [v for k, v in ck._rows.items() if k.startswith('["row"')]
    if any(r.get("smoke_limit") for r in rows) and a.paper:
        raise SystemExit("checkpoint holds --smoke-limit rows: refusing a --paper assemble")
    calib = [v for k, v in ck._rows.items() if k.startswith('["calib"')]
    ref = dynamic_reference()
    with start_run(EXP, {"calibration": CALIB, "configs": {k: [s, repr(v)] for k, (s, v) in CONFIGS.items()},
                         "model_revisions": MODEL_REV, "wt2_revision": WT2_REV, "c4_revision": C4_REV,
                         "c4_windows": C4_WINDOWS, "tasks": TASKS, "checkpoint_tag": ck.tag,
                         "checkpoint": str(ck.path.relative_to(REPO_ROOT))}, paper=a.paper) as run:
        flat = []
        for r in rows:
            metric = r["token_ppl"] if r["part"] in ("wt2", "c4") else r.get("value")
            flat.append({k: v for k, v in r.items() if k not in ("quant", "lmeval_meta", "sat_worst")}
                        | {"metric_value": metric, "sat_worst_layer": (r.get("sat_worst") or [[None]])[0][0]})
        run.save_csv("rows.csv", flat)
        run.path("rows_full.yaml").write_text(yaml.safe_dump(rows, sort_keys=True))
        run.path("calib_amax.yaml").write_text(yaml.safe_dump(calib, sort_keys=True))
        delta, metrics = [], {}
        for r in flat:
            if r["nv_act_scale"] != "static":
                continue
            k = (r["model"], r["config"], r["part"])
            dyn_local = next((x["metric_value"] for x in flat if x["nv_act_scale"] == "dynamic"
                              and (x["model"], x["config"], x["part"]) == k), None)
            d, src = ref.get(k, (None, None))
            s = r["metric_value"]
            delta.append({
                "model": r["model"], "config": r["config"], "part": r["part"],
                "metric": "token_ppl" if r["part"] in ("wt2", "c4") else TASKS[r["part"][3:]],
                "dynamic": d, "static": s, "delta": None if d is None else s - d,
                "delta_rel_pct": None if d is None else 100 * (s - d) / d, "dynamic_source": src,
                "dynamic_local": dyn_local, "delta_vs_local": None if dyn_local is None else s - dyn_local,
                "static_gpu": r["gpu"], "sat_max_ratio": r.get("sat_max_ratio"),
                "sat_frac_elems_over": r.get("sat_frac_elems_over"),
            })
            tag = f"{r['model'].replace('.', 'p').replace('-', '_')}_{r['config']}_{r['part'].replace(':', '_')}"
            metrics[f"{tag}_static"] = s
            if d is not None:
                metrics[f"{tag}_delta"] = s - d
        run.save_csv("delta.csv", delta)
        # T2-shaped NV rows (same columns as e09 t2.csv) where every part of a (model, config) is present
        t2 = []
        for model in MODEL_REV:
            for cname in ("nv_rtn", "nv_gptq", "nv_gptq_4over6", "nv_hw", "nv_rtn_attn_bf16") + NON_NV:
                want = "dynamic" if cname in NON_NV else "static"
                got = {r["part"]: r for r in flat if r["model"] == model and r["config"] == cname
                       and r["nv_act_scale"] == want}
                if not got:
                    continue
                row = {"model": model, "revision": MODEL_REV[model],
                       "config": cname if cname in NON_NV else cname + "_static",
                       "c4_token_ppl": got.get("c4", {}).get("token_ppl"),
                       **{t: got.get(f"lm:{t}", {}).get("value") for t in TASKS},
                       **{f"{t}_stderr": got.get(f"lm:{t}", {}).get("stderr") for t in TASKS},
                       "wikitext2_token_ppl": got.get("wt2", {}).get("token_ppl")}
                acc = [row[t] for t in TASKS]
                row["avg_acc"] = sum(acc) / len(acc) if all(v is not None for v in acc) else None
                t2.append(row)
        run.save_csv("t2_nv_static.csv", t2)
        # per (model, config) summary: WT2 / C4 PPL and mean task accuracy, dynamic (old rows) vs static
        summ = []
        for row in t2:
            model, cname = row["model"], row["config"].removesuffix("_static")
            dyn_acc = [ref.get((model, cname, f"lm:{t}"), (None,))[0] for t in TASKS]
            d_avg = sum(dyn_acc) / len(dyn_acc) if all(v is not None for v in dyn_acc) else None
            summ.append({
                "model": model, "config": cname,
                "wt2_dynamic": ref.get((model, cname, "wt2"), (None,))[0], "wt2_static": row["wikitext2_token_ppl"],
                "c4_dynamic": ref.get((model, cname, "c4"), (None,))[0], "c4_static": row["c4_token_ppl"],
                "avg_acc_dynamic": d_avg, "avg_acc_static": row["avg_acc"],
                "avg_acc_delta_pts": None if d_avg is None or row["avg_acc"] is None else 100 * (row["avg_acc"] - d_avg),
            })
        run.save_csv("summary.csv", summ)
        run.log_metrics(metrics | {"n_rows": len(flat), "n_calib": len(calib)})
        print(f"assembled {len(flat)} rows, {len(calib)} calibrations -> {run.dir}")
        if a.done:
            (CKPT_DIR / f"{EXP}.done").write_text(time.strftime("%F %T UTC\n", time.gmtime()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m,qwen2.5-0.5b,gpt2-large")
    ap.add_argument("--configs", default="nv_rtn")
    ap.add_argument("--parts", default="calib,wt2", help="calib, wt2, c4, lm:<task>, lm (= every task)")
    ap.add_argument("--scale", default="static", choices=["static", "dynamic"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--done", action="store_true", help="with --assemble: write the done marker")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--smoke-limit", type=int, default=None, help="SMOKE TEST ONLY: PPL windows / "
                    "lm-eval docs cap; rows are tagged smoke_limit and must never be assembled for the paper")
    a = ap.parse_args()
    seed_all(0)
    if a.assemble:
        return assemble(a)
    from transformers import AutoTokenizer

    specs = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text()) | EXTRA_SPECS
    parts = [p for x in a.parts.split(",") for p in ([f"lm:{t}" for t in TASKS] if x == "lm" else [x])]
    ck = checkpoint()
    print(f"[checkpoint] {ck.path} tag {ck.tag} rows {len(ck)}", flush=True)
    for model in a.models.split(","):
        tok = AutoTokenizer.from_pretrained(specs[model]["hf_id"], revision=MODEL_REV[model])
        for cname in a.configs.split(","):
            source, kw = CONFIGS[cname]
            if source != "hf" and not any((d / f"{model}__{source}").exists() for d in GPTQ_DIRS):
                print(f"[skip] {model} {cname}: no GPTQ weights", flush=True)
                continue
            for part in parts:
                if part == "calib":
                    if a.scale == "static":
                        get_calib(ck, specs, tok, model, source, kw.get("attn_policy", BASE["attn_policy"]), a.seqlen)
                    continue
                run_part(ck, specs, tok, model, cname, part, a.scale, a)


if __name__ == "__main__":
    main()
