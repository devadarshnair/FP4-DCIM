"""T5.4: MXINT4-SM (D2, MIDAS-inspired, LOSSY surrogate) macro programs on the SAME windows as the T2.7 e21 run.

The T2.7 macro windows (`results/e21_macro_power/<run>/windows.json`: capture + per-window seed) are replayed with the rng
draws of `macro_program_from_capture` (output-row offset r0, 16 K-slices, 256 tokens); the replay is checked against the
stored MXFP4 program of that run.  The operands are then re-quantised to MXINT4-SM (docs/spec/d2_int4_contract.md, rule
floor_sat) from the ORIGINAL fp32 weight / activation tensors (CPU fp32 re-capture, as e34 does for the tile: the stored
captures hold only FP4 codes), so the INT4-SM macro sees the same rows / slices / tokens / set pattern as v2 / INT5 / DPU.
The random control (uniform 4-bit codes) is reused unchanged: a uniform nibble is a uniform {sign, mag}.  The GPTQ-weight
window (real8) is not built: its weights exist only as GPTQ-dequantised MXFP4/NVFP4 values, which have no INT4-SM analogue.

Phases (each one safe_run job):
  --capture   fp32 CPU forward of the models in the window list -> build/e21_int4sm/raw/<window>.npz   (heavy: torch + model)
  --programs  build int4sm_<window>.npz + int4sm_programs.json -> build/e21_int4sm/programs           (light, numpy only)
Then: e21 run.py --netlists td --designs bs4_macro_mx --int4sm-programs build/e21_int4sm/programs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from cimfp4.formats.int4 import quantize_mxint4
from cimfp4.stimulus.writers import MCOL, _pack_macro, load_program, save_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

T27_RUN = "20261114T023643822998Z_cada632"  # T2.7 MX macro run (paired windows)
BUILD = REPO_ROOT / "build/e21_int4sm"
M, SETS_USED, REUSE = 16, 2, 256  # the T2.7 run's --sets-used / --reuse (config.yaml)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(run_id: str = T27_RUN) -> list[dict]:
    """rtn windows of the T2.7 run -> r0 / slices / tokens, each verified against the stored MXFP4 program."""
    from cimfp4.llm.capture import load_capture

    meta = json.loads((REPO_ROOT / "results/e21_macro_power" / run_id / "windows.json").read_text())
    progdir = REPO_ROOT / "build/e21_macro_power" / run_id / "programs"
    out = []
    for w in meta:
        if w["weights"] != "rtn":
            continue
        cpath = REPO_ROOT / w["capture"]
        qw, qa = load_capture(cpath, "mxfp4")
        rng = np.random.default_rng(w["seed"])  # same draw order as macro_program_from_capture
        r0 = int(rng.integers(0, qw.codes.shape[0] - MCOL + 1))
        nslice = qw.codes.shape[1] // 32
        slices = rng.choice(nslice, size=min(M, nslice), replace=False)
        tokens = rng.choice(qa.codes.shape[0], size=REUSE, replace=qa.codes.shape[0] < REUSE)
        ref = progdir / f"mxfp4_{w['window']}.npz"
        verified = None
        if ref.exists():
            p = load_program(ref)
            sets = np.stack([qw.codes[r0:r0 + MCOL, 32 * b:32 * (b + 1)] for b in slices])
            x = np.array([qa.codes[t, 32 * slices[s]:32 * (slices[s] + 1)] for s in range(SETS_USED) for t in tokens])
            verified = bool(np.array_equal(sets, p["sets"][:len(slices)]) and np.array_equal(x, p["x"]))
            if not verified:
                raise RuntimeError(f"replay of {w['window']} does not reproduce {ref}")
        with np.load(cpath) as z:
            module = str(z["module"])
        out.append({"window": w["window"], "capture": w["capture"], "model": cpath.parts[-3], "module": module,
                    "seed": w["seed"], "r0": r0, "slices": slices.tolist(), "tokens": tokens.tolist(),
                    "replay_verified_against_mxfp4_program": verified})
    return out


def phase_capture(paper: bool) -> None:
    import torch
    import yaml
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cimfp4.formats.mx_quant import quantize_mxfp4
    from cimfp4.llm.capture import ActivationRecorder

    wins = replay()
    models = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    (BUILD / "raw").mkdir(parents=True, exist_ok=True)
    cfg = {"t27_run": T27_RUN, "windows": [{k: v for k, v in w.items() if k != "tokens"} for w in wins], "seqlen": 512,
           "prefill_windows": 4, "dtype": "float32 forward on CPU", "dataset": "wikitext-2-raw-v1 test",
           "note": "raw fp32 weights/activations of the selected rows/slices/tokens (as e34 --capture), for MXINT4-SM"}
    with start_run("e21_int4sm_capture", cfg, paper=paper) as run:
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        rows = []
        for mname in sorted({w["model"] for w in wins}):
            spec = models[mname.split("@")[0]]
            snap = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["hf_id"].replace("/", "--")) / "snapshots"
            rev = next(d for d in snap.iterdir() if d.name.startswith(mname.split("@")[1]))
            tok = AutoTokenizer.from_pretrained(str(rev))
            model = AutoModelForCausalLM.from_pretrained(str(rev), torch_dtype=torch.float32).eval()
            mine = [w for w in wins if w["model"] == mname]
            mods = {n: m for n, m in model.named_modules() if n in {w["module"] for w in mine}}
            ids = tok(text, return_tensors="pt").input_ids[0]
            starts = np.linspace(0, len(ids) - 512 - 1, 4).astype(int)
            with torch.no_grad(), ActivationRecorder(mods) as rec:
                for s in starts:
                    model(ids[s:s + 512][None])
            for w in mine:
                raw_a = rec.stacked(w["module"])  # (2048, K) fp32
                raw_w = mods[w["module"]].weight.detach().float().numpy()
                tk, sl = np.array(w["tokens"]), np.array(w["slices"])
                wsel = np.stack([raw_w[w["r0"]:w["r0"] + MCOL, b * 32:(b + 1) * 32] for b in sl])  # (16, 32, 32)
                asel = np.stack([raw_a[tk][:, b * 32:(b + 1) * 32] for b in sl[:SETS_USED]])  # (S, 256, 32)
                with np.load(REPO_ROOT / w["capture"]) as z:
                    ac = z["mxfp4_a_codes"].reshape(z["mxfp4_a_codes"].shape[0], -1, 32)[tk][:, sl[:SETS_USED]]
                    wc = z["mxfp4_w_codes"].reshape(z["mxfp4_w_codes"].shape[0], -1, 32)[w["r0"]:w["r0"] + MCOL][:, sl]
                qa = quantize_mxfp4(asel.transpose(1, 0, 2).reshape(-1, 32), "ocp_v11")
                qw = quantize_mxfp4(wsel.reshape(-1, 32), "ocp_v11")
                row = {"window": w["window"], "model": mname, "module": w["module"],
                       "act_codes_match_frac": float(np.mean(qa.codes.reshape(ac.shape) == ac)),
                       "w_codes_match_frac": float(np.mean(qw.codes.reshape(wsel.shape) == wc.transpose(1, 0, 2)))}
                rows.append(row)
                np.savez(BUILD / "raw" / f"{w['window']}.npz", w=wsel, a=asel)
                print(row, flush=True)
            del model, rec
        run.save_csv("capture_check.csv", rows)
        run.log_metrics({"min_act_codes_match_frac": min(r["act_codes_match_frac"] for r in rows),
                         "min_w_codes_match_frac": min(r["w_codes_match_frac"] for r in rows),
                         "raw_sha256": {w["window"]: sha256(BUILD / "raw" / f"{w['window']}.npz") for w in wins}})


def to_code(m: np.ndarray) -> np.ndarray:
    """signed magnitude (int8, [-7, 7]) -> {sign, mag[2:0]} nibble (+0 for zero)."""
    return ((m < 0).astype(np.uint8) << 3) | np.abs(m).astype(np.uint8)


def int4sm_macro_program(wsel: np.ndarray, asel: np.ndarray) -> dict:
    """wsel (M, 32 cols, 32 rows) fp32, asel (S, T, 32) fp32 -> macro program (fmt int4sm, MX field layout)."""
    m_sets = wsel.shape[0]
    qw = quantize_mxint4(wsel.reshape(-1, 32), "floor_sat")
    sets = to_code(qw.m).reshape(m_sets, MCOL, 32)
    e0 = qw.scale_exp.astype(np.int64).reshape(m_sets, MCOL)
    z = np.zeros_like(e0)
    qa = quantize_mxint4(asel.reshape(-1, 32), "floor_sat")
    xc = to_code(qa.m).reshape(asel.shape)
    ex = qa.scale_exp.astype(np.int64).reshape(asel.shape[:2])
    ops = [(xc[s, t], int(ex[s, t]), 0, 1, 1, int(ex[s, t]), s) for s in range(asel.shape[0]) for t in range(asel.shape[1])]
    pg = _pack_macro(sets, e0, z, z + 1, z + 1, ops, "mxfp4")  # MX field layout (maxw = max e0)
    pg["fmt"] = np.array("int4sm")
    return pg


def phase_programs(paper: bool) -> None:
    wins = replay()
    out = BUILD / "programs"
    out.mkdir(parents=True, exist_ok=True)
    prov, rows = {}, []
    ref_dir = REPO_ROOT / "build/e21_macro_power" / T27_RUN / "programs"
    with start_run("e21_int4sm_programs", {"rule": "floor_sat", "t27_run": T27_RUN,
                                           "windows": [w["window"] for w in wins] + ["rand0"]}, paper=paper) as run:
        for w in wins:
            with np.load(BUILD / "raw" / f"{w['window']}.npz") as z:
                pg = int4sm_macro_program(z["w"], z["a"])
            ref = load_program(ref_dir / f"mxfp4_{w['window']}.npz")
            p = save_program(out / f"int4sm_{w['window']}.npz", pg)
            prov[w["window"]] = {"source": w["capture"], "replay_verified": w["replay_verified_against_mxfp4_program"]}
            rows.append({"window": w["window"], "n_ops": len(pg["x"]), "set_pattern_identical": bool(np.array_equal(pg["set"], ref["set"])),
                         "x_zero_frac": float(np.mean((pg["x"] & 7) == 0)), "x_sat7_frac": float(np.mean((pg["x"] & 7) == 7)),
                         "w_zero_frac": float(np.mean((pg["sets"] & 7) == 0)), "sha256": sha256(p)})
        shutil.copy(ref_dir / "mxfp4_rand0.npz", out / "int4sm_rand0.npz")  # uniform nibbles = uniform {sign, mag}
        prov["rand0"] = {"source": "random", "copied_from": f"build/e21_macro_power/{T27_RUN}/programs/mxfp4_rand0.npz"}
        (out / "int4sm_programs.json").write_text(json.dumps(prov, indent=1))
        run.save_csv("programs.csv", rows)
        run.log_metrics({"program_sha256": {p.stem: sha256(p) for p in sorted(out.glob("*.npz"))},
                         "all_set_patterns_identical": all(r["set_pattern_identical"] for r in rows)})
        print(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--programs", action="store_true")
    ap.add_argument("--replay-check", action="store_true", help="only replay + verify the windows (light)")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    if a.replay_check:
        print(json.dumps([{k: v for k, v in w.items() if k not in ("tokens", "slices")} for w in replay()], indent=1))
    if a.capture:
        phase_capture(a.paper)
    if a.programs:
        phase_programs(a.paper)


if __name__ == "__main__":
    main()
