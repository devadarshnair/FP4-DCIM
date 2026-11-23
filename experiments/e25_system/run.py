"""e25 - analytical system-level model: reuse, utilization, reload, prefill, decode.

Uses only measured per-op costs (e21/e22 macro, e20 SRAM, e24 post-route factors) via cimfp4.sysmodel.system_model.
Run: python experiments/e25_system/run.py [--paper]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
from dataclasses import asdict

from cimfp4.sysmodel import system_model as sm
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

HF = {"Qwen2.5-0.5B": "models--Qwen--Qwen2.5-0.5B", "SmolLM2-360M": "models--HuggingFaceTB--SmolLM2-360M"}
SCEN = [("prefill", 512), ("prefill", 1024), ("prefill", 2048), ("decode", 1), ("decode", 8)]


def hf_cfg(name):
    p = glob.glob(str(__import__("pathlib").Path.home() / f".cache/huggingface/hub/{HF[name]}/snapshots/*/config.json"))[0]
    raw = open(p, "rb").read()
    return json.loads(raw), p.split("/snapshots/")[1], hashlib.sha256(raw).hexdigest()[:12]


def row(model, mode, n, dname, a, r):
    p = r["parts_j"]
    tok = n if mode == "prefill" else n
    return {"model": model, "mode": mode, "seq_or_batch": n, "design": dname, "n_macros": a.n_macros, "f_mhz": round(a.f_mhz, 2),
            "sram": a.sram_corner, "sram_clock": a.sram_clock, "post_route": a.post_route, "idle_logic": a.idle_logic,
            "lm_head": a.include_lm_head, "R": r["weight_reuse_R"], "tiles": r["tiles"], "set_visits": r["set_visits"],
            "lane_util": r["lane_util"], "time_util": r["time_util"], "overhead_share": r["overhead_share"],
            "uJ_per_token": r["energy_j"] / tok * 1e6, "latency_us_per_token": r["latency_s"] / tok * 1e6,
            "pass_latency_ms": r["latency_s"] * 1e3, "tops_per_w": r["tops_per_w"],
            "dram_MB_per_token": r["dram_bytes"] / tok / 1e6,
            **{f"share_{k}": v / r["energy_j"] for k, v in p.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    inp = sm.load_inputs(REPO_ROOT)
    cfgs, meta = {}, {}
    for m in HF:
        c, snap, sha = hf_cfg(m)
        cfgs[m] = c
        meta[m] = {"snapshot": snap, "config_sha256_12": sha,
                   **{k: c[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                                        "num_key_value_heads", "vocab_size")}}
    base = sm.Assumptions()
    cfg_run = {"scenarios": SCEN, "base": asdict(base), "designs": {k: asdict(v) for k, v in inp.designs.items()},
               "models": meta, "inputs": inp.provenance, "grid": "n_macros{16,64,256} x f{40,postroute fmax lb} x sram{low,nominal,high} x "
               "clock{as_built,gated} x post_route{0,1} x idle_logic{0,1}; lm_head excluded (separate rows)"}
    rows, bd = [], []
    with start_run("e25_system", cfg_run, paper=args.paper) as run:
        for model, cfg in cfgs.items():
            for (mode, n), dn in itertools.product(SCEN, inp.designs):
                d = inp.designs[dn]
                for nm, fk, corner, clk, pr, il in itertools.product(
                        (16, 64, 256), ("40", "fmax"), ("low", "nominal", "high"), ("as_built", "gated"),
                        (False, True), (False, True)):
                    a = sm.Assumptions(n_macros=nm, f_mhz=40.0 if fk == "40" else d.fmax_pr_mhz, sram_corner=corner,
                                       sram_clock=clk, post_route=pr, idle_logic=il)
                    r = sm.evaluate(cfg, n, d, inp.sram, a)
                    rows.append(row(model, mode, n, dn, a, r))
                a = sm.Assumptions(include_lm_head=True)
                rows.append(row(model, mode, n, dn, a, sm.evaluate(cfg, n, d, inp.sram, a)))
                if n in (512, 1):
                    a = sm.Assumptions()
                    for g in sm.evaluate(cfg, n, d, inp.sram, a)["breakdown"]:
                        bd.append({"model": model, "mode": mode, "n": n, "design": dn, **g})
        run.save_csv("grid.csv", rows)
        run.save_csv("layer_breakdown.csv", bd)
        be = {dn: {c: sm.breakeven_rows(d, inp.sram, sm.Assumptions(sram_corner=c)) for c in ("low", "nominal", "high")}
              for dn, d in inp.designs.items()}
        base_rows = [r for r in rows if r["n_macros"] == 64 and r["f_mhz"] in (40.0,) and r["sram"] == "nominal"
                     and r["sram_clock"] == "as_built" and not r["post_route"] and not r["idle_logic"] and not r["lm_head"]]
        lines = ["# e25 base-point tables [DERIVED analytic model; assumptions in README of run config]",
                 "Base: 64 macros, 40 MHz, D33 nominal SRAM, as-built free-running SRAM clock, zero-delay measured logic + [EST] wires, "
                 "idle-cycle logic energy off, lm_head excluded.", ""]
        for model in HF:
            lines += [f"## {model}", "| mode | n | design | R | set visits | lane util | time util | uJ/token | us/token | TOPS/W | DRAM MB/token |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in base_rows:
                if r["model"] == model:
                    lines.append(f"| {r['mode']} | {r['seq_or_batch']} | {r['design']} | {r['R']} | {r['set_visits']:.0f} | {r['lane_util']:.3f} | "
                                 f"{r['time_util']:.3f} | {r['uJ_per_token']:.1f} | {r['latency_us_per_token']:.2f} | {r['tops_per_w']:.3f} | {r['dram_MB_per_token']:.3f} |")
            lines.append("")
        lines += ["## Break-even reuse R* (per-visit switch+write+overhead-idle-clock energy = R ops of logic)", json.dumps(be, indent=1)]
        (run.dir / "tables.md").write_text("\n".join(lines))
        run.log_metrics({"n_rows": len(rows), "breakeven_R": be, "models": meta})
        print("\n".join(lines))
        print(run.dir)


if __name__ == "__main__":
    main()
