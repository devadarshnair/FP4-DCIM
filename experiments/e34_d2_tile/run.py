"""e34 - T5.3: same-flow tile measurement of the INT4-SM bit-serial baseline (D2, MIDAS-inspired, LOSSY).

DUT: rtl/baselines/bitserial_int4sm/bs4_column_top.sv (4-column tile, same storage/boundary as the e13/e15 tiles v2_sram_m16,
bs_int5, dpu; sweep design `bs4_int4sm`).  Flow = e30/e13/e15 verbatim (no new flow): Yosys (recipe base or td, flows/synth/sweep.py)
+ OpenROAD repair (td) + Yosys-liberty area + OpenSTA at 25 ns; zero-delay Icarus GLS on the e13_power_rb windows, bit-exact against
an independent integer reference (tb/cocotb/test_bs4_top.py), VCD sliced to the compute window and rebased to t=0, docker OpenSTA
report_power, SRAM per-access charge, boundary wires (add_boundary model), glitch factor from the 2-window SDF subset (sdf_subset.py).

Stimulus = the SAME windows as e13_power_rb (replayed from seed 0: same capture file, 4 weight rows, K-blocks and tokens), but the
operands are re-quantized to MXINT4-SM (docs/spec/d2_int4_contract.md, rule floor_sat) from the original fp32 weight/activation
tensors (CPU re-capture; the stored MXFP4 captures are reproduced to >= 98 % of codes, as in e28).  The two random control windows
(uniform 4-bit codes) are used unchanged: a uniform nibble is a uniform {sign, mag}.

Phases (each one safe_run job):
  --capture              CPU fp32 forward of the 8 real e13 windows + 2 SDF windows -> build/e34/raw   (results/e34_d2_tile_capture)
  --programs             build the INT4-SM programs -> build/e34/programs
  --synth RECIPE         tile synthesis + STA of bs4_int4sm                              -> results/e34_d2_tile_synth
  --power RECIPE         GLS + power for --windows                                       -> results/e34_d2_tile_power
  --sdf RECIPE           zero-delay vs SDF GLS energy on sdf0/sdf1                       -> results/e34_d2_tile_sdf
  --summary              e15-style table: v2 / INT4-SM / INT5 / DPU with paired bootstrap CIs -> results/e34_d2_tile
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.formats.int4 import quantize_mxint4
from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import _pack, load_program, save_program
from cimfp4.utils.io import load_csv
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e13 = _load("e13_run", REPO_ROOT / "experiments/e13_power/run.py")
e30 = _load("e30_run", REPO_ROOT / "experiments/e30_td_synth/run.py")
e23 = e30.e23
R = REPO_ROOT / "results"
BUILD = REPO_ROOT / "build/e34"
RB_PROGS = REPO_ROOT / "build/power/e13_power_rb/programs"
SDF_PROGS = REPO_ROOT / "build/sdf/e13_sdf_td/programs"
DESIGN = "bs4_int4sm"
TOP, PARAMS = sweep.DESIGNS[DESIGN]
CLK_PS = 25000
NCOL = 4
REAL = [f"real{i}" for i in range(8)]
WINDOWS = REAL + ["rand0", "rand1"]
SDFW = ["sdf0", "sdf1"]
MACS_PER_OP = 4 * 32
RECIPES = {"td": "td", "base": "base"}  # recipe -> reference result suffix (td: *_td, base: *_rb)
REF_SFX = {"td": "_td", "base": "_rb"}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest(exp: str) -> Path:
    return R / exp / "latest"


# ------------------------------------------------------------------------------------------------ replay + capture
def replay(seed: int, n: int, sets_used: int, reuse: int, prefix: str, progdir: Path) -> list[dict]:
    """Replay the rng stream of program_from_capture; assert it reproduces the stored e13/e13_sdf program."""
    from cimfp4.llm.capture import load_capture

    rng = seed_all(seed)
    caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
    out = []
    for w in range(n):
        cpath = caps[int(rng.integers(0, len(caps)))]
        qw, qa = load_capture(cpath, "mxfp4")
        r0 = int(rng.integers(0, qw.codes.shape[0] - NCOL))
        blocks = rng.choice(qw.num_blocks, size=min(16, qw.num_blocks), replace=False)
        tokens = rng.choice(qa.codes.shape[0], size=reuse, replace=qa.codes.shape[0] < reuse)
        prog = load_program(progdir / f"{prefix}{w}.npz")
        x = np.array([qa.blocked_codes()[t, blocks[s]] for s in range(sets_used) for t in tokens], np.uint8)
        sets = np.stack([qw.blocked_codes()[r0:r0 + NCOL, b] for b in blocks])
        if not (np.array_equal(x, prog["x"]) and np.array_equal(sets, prog["sets"][:len(blocks)])):
            raise RuntimeError(f"replay of {prefix}{w} does not reproduce the stored program")
        with np.load(cpath) as z:
            module = str(z["module"])
        out.append({"name": f"{prefix}{w}", "capture": str(cpath.relative_to(REPO_ROOT)), "model": cpath.parts[-3],
                    "module": module, "r0": r0, "blocks": blocks.tolist(), "tokens": tokens.tolist(),
                    "sets_used": sets_used})
    return out


def all_windows() -> list[dict]:
    return replay(0, 8, 8, 256, "real", RB_PROGS) + replay(1, 2, 4, 128, "sdf", SDF_PROGS)


def phase_capture(paper: bool) -> None:
    import torch
    import yaml
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cimfp4.llm.capture import ActivationRecorder

    wins = all_windows()
    models = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    (BUILD / "raw").mkdir(parents=True, exist_ok=True)
    cfg = {"windows": [{k: v for k, v in w.items() if k != "tokens"} for w in wins], "seqlen": 512, "prefill_windows": 4,
           "dtype": "float32 forward on CPU (e04 used cuda when available)", "dataset": "wikitext-2-raw-v1 test",
           "note": "raw fp32 weights/activations of the selected rows/blocks/tokens, to be re-quantized to MXINT4-SM"}
    with start_run("e34_d2_tile_capture", cfg, paper=paper) as run:
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        rows = []
        for mname in sorted({w["model"] for w in wins}):
            spec = models[mname.split("@")[0]]
            snap_root = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["hf_id"].replace("/", "--")) / "snapshots"
            rev = next(d for d in snap_root.iterdir() if d.name.startswith(mname.split("@")[1]))
            tok = AutoTokenizer.from_pretrained(str(rev))
            model = AutoModelForCausalLM.from_pretrained(str(rev), torch_dtype=torch.float32).eval()
            mine = [w for w in wins if w["model"] == mname]
            mods = {n: m for n, m in model.named_modules() if n in {w["module"] for w in mine}}
            ids = tok(text, return_tensors="pt").input_ids[0]
            starts = np.linspace(0, len(ids) - 512 - 1, 4).astype(int)
            with torch.no_grad(), ActivationRecorder(mods) as rec:
                for s in starts:
                    model(ids[s: s + 512][None])
            for w in mine:
                raw_a = rec.stacked(w["module"])  # (2048, K) fp32
                raw_w = mods[w["module"]].weight.detach().float().numpy()
                tok_idx = np.array(w["tokens"])
                blk = np.array(w["blocks"])
                wsel = np.stack([raw_w[w["r0"]:w["r0"] + NCOL, b * 32:(b + 1) * 32] for b in blk])  # (16, 4, 32)
                asel = np.stack([raw_a[tok_idx][:, b * 32:(b + 1) * 32] for b in blk[:w["sets_used"]]])  # (S, tok, 32)
                with np.load(REPO_ROOT / w["capture"]) as z:
                    sc = z["mxfp4_a_codes"].reshape(z["mxfp4_a_codes"].shape[0], -1, 32)[tok_idx][:, blk[:w["sets_used"]]]
                    wc = z["mxfp4_w_codes"].reshape(z["mxfp4_w_codes"].shape[0], -1, 32)[w["r0"]:w["r0"] + NCOL][:, blk]
                qa = quantize_mxfp4(asel.transpose(1, 0, 2).reshape(-1, 32), "ocp_v11")
                qw = quantize_mxfp4(wsel.reshape(-1, 32), "ocp_v11")
                row = {"window": w["name"], "model": w["model"], "module": w["module"],
                       "act_codes_match_frac": float(np.mean(qa.codes.reshape(sc.shape) == sc)),
                       "w_codes_match_frac": float(np.mean(qw.codes.reshape(wsel.shape) == wc.transpose(1, 0, 2)))}
                rows.append(row)
                np.savez(BUILD / "raw" / f"{w['name']}.npz", w=wsel, a=asel)
                print(row, flush=True)
            del model, rec
        run.save_csv("capture_check.csv", rows)
        run.log_metrics({"min_act_codes_match_frac": min(r["act_codes_match_frac"] for r in rows),
                         "min_w_codes_match_frac": min(r["w_codes_match_frac"] for r in rows),
                         "raw_sha256": {w["name"]: sha256(BUILD / "raw" / f"{w['name']}.npz") for w in wins}})


# ------------------------------------------------------------------------------------------------ INT4-SM programs
def to_code(m: np.ndarray) -> np.ndarray:
    """signed magnitude (int8, [-7, 7]) -> {sign, mag[2:0]} nibble (+0 for zero)."""
    return ((m < 0).astype(np.uint8) << 3) | np.abs(m).astype(np.uint8)


def int4_program(raw: dict, prog_fp4: dict | None = None) -> dict:
    wsel, asel = raw["w"], raw["a"]  # (16, 4, 32), (S, T, 32)
    m_sets = wsel.shape[0]
    qw = quantize_mxint4(wsel.reshape(-1, 32), "floor_sat")
    sets = to_code(qw.m).reshape(m_sets, NCOL, 32)
    ew = qw.scale_exp.reshape(m_sets, NCOL)
    qa = quantize_mxint4(asel.reshape(-1, 32), "floor_sat")
    xc = to_code(qa.m).reshape(asel.shape)
    ex = qa.scale_exp.reshape(asel.shape[:2])
    ops = [(xc[s, t], int(ex[s, t]), int(ex[s, t]), s) for s in range(asel.shape[0]) for t in range(asel.shape[1])]
    return _pack(sets, ew, ops)


def phase_programs(paper: bool) -> None:
    (BUILD / "programs").mkdir(parents=True, exist_ok=True)
    wins = all_windows()  # re-verifies the replay
    rows = []
    with start_run("e34_d2_tile_programs", {"rule": "floor_sat", "windows": [w["name"] for w in wins]}, paper=paper) as run:
        for w in wins:
            with np.load(BUILD / "raw" / f"{w['name']}.npz") as z:
                raw = {k: z[k] for k in z.files}
            pg = int4_program(raw)
            ref = load_program((RB_PROGS if w["name"].startswith("real") else SDF_PROGS) / f"{w['name']}.npz")
            save_program(BUILD / "programs" / f"{w['name']}.npz", pg)
            same_set = float(np.mean(pg["set"] == ref["set"]))
            rows.append({"window": w["name"], "n_ops": len(pg["x"]), "set_pattern_identical": same_set == 1.0,
                         "x_zero_frac": float(np.mean((pg["x"] & 7) == 0)), "x_sat7_frac": float(np.mean((pg["x"] & 7) == 7)),
                         "w_zero_frac": float(np.mean((pg["sets"] & 7) == 0)),
                         "x_zero_frac_fp4": float(np.mean((ref["x"] & 7) == 0))})
        for r in ("rand0", "rand1"):  # uniform 4-bit codes are uniform {sign, mag}
            shutil.copy(RB_PROGS / f"{r}.npz", BUILD / "programs" / f"{r}.npz")
        run.save_csv("programs.csv", rows)
        run.log_metrics({"program_sha256": {p.stem: sha256(p) for p in sorted((BUILD / "programs").glob("*.npz"))}})
        print(rows)


# ------------------------------------------------------------------------------------------------ synth
def phase_synth(recipe: str, paper: bool) -> None:
    out = BUILD / "synth" / recipe
    cfg = {"design": DESIGN, "top": TOP, "params": PARAMS, "recipe": recipe, "period_ns": e30.PERIOD_NS,
           "abc_D_ps": sweep.TD_ABC_D_PS, "max_fanout": e30.MAX_FANOUT, "max_tran_ns": e30.MAX_TRAN_NS,
           "image": e30.IMG, "recipe_spec": "docs/spec/synth_recipe.md",
           "rtl_sha256": {p: sha256(REPO_ROOT / p) for p in sweep.SOURCES if "int4sm" in p}}
    with start_run("e34_d2_tile_synth", cfg, paper=paper) as run:
        row, net = e30.build_one(DESIGN, TOP, PARAMS, out, recipe, False, False, "3g")
        row = {"design": DESIGN, **row}
        row["total_mm2"] = row["total_area_um2"] / 1e6
        run.save_csv("synth.csv", [row])
        for suf in ("stat.txt", "final_stat.txt", "sta.txt", "sta_r2r.txt"):
            shutil.copy(out / f"{DESIGN}_{suf}", run.path(f"{recipe}_{DESIGN}_{suf}"))
        for suf in (".rpt", ".log"):
            if Path(f"{net}{suf}").exists():
                shutil.copy(f"{net}{suf}", run.path(f"{recipe}_{DESIGN}_repair{suf}"))
        run.log_metrics({"recipe": recipe, **{k: v for k, v in row.items() if isinstance(v, (int, float, str, bool))}})
        print({k: row[k] for k in ("logic_area_um2", "total_area_um2", "cells", "r2r_wns_ns", "r2r_fmax_mhz")})


# ------------------------------------------------------------------------------------------------ GLS
def gls(net: Path, program: Path, wd: Path, sdf: Path | None = None, sdfm=None) -> dict:
    """Icarus GLS of the netlist with the INT4-SM harness (bit-exact check inside); zero delay or SDF."""
    from cocotb_tools.runner import get_results, get_runner

    vcd, win = wd / "dump.vcd", wd / "window.json"
    srcs = [e23.CELLS / "primitives.v", e23.CELLS / "sky130_fd_sc_hd.v", net, e23.SRAM_GLS, REPO_ROOT / "flows/power/vcd_dump.v"]
    defines = {"VCD_FILE": f'"{vcd}"', "VCD_TOP": TOP}
    build_args = ["-s", "vcd_dump"]
    if sdf is None:
        defines |= {"FUNCTIONAL": 1, "UNIT_DELAY": "#0"}
    else:
        srcs.append(REPO_ROOT / "flows/power/sdf_annotate.v")
        defines |= {"SDF_FILE": f'"{sdf}"', "SDF_TOP": TOP}
        build_args += ["-s", "sdf_annotate", "-gspecify"]
    runner = get_runner("icarus")
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if a == "-none" else a for a in c] for c in orig()]
    runner.build(sources=srcs, hdl_toplevel=TOP, build_dir=wd / "b", always=True, defines=defines, build_args=build_args,
                 timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel=TOP, test_module="test_bs4_top", test_dir=REPO_ROOT / "tb/cocotb", build_dir=wd / "b",
                      results_xml=str(wd / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tb/cocotb'}", "CT_PROGRAM": str(program),
                                 "CT_WINDOW_OUT": str(win), "CT_M": "16", "CT_NOPS": "0", "CT_CLK_PS": str(CLK_PS),
                                 "CT_MISMATCH_OUT": str(wd / "mismatch.txt")})
    if get_results(xml) != (1, 0):
        raise RuntimeError(f"GLS mismatch for {net} on {program.name}")
    return {**json.loads(win.read_text()), "vcd": vcd, "mismatches": int((wd / "mismatch.txt").read_text().split()[0])}


def sta_power(net: Path, vcd: Path, wd: Path) -> dict:
    env = {"NETLIST": net, "TOP": TOP, "LIB": sweep.LIB, "SRAM_LIB": e30.SRAM_LIB, "SDC": e30.SDC,
           "CLK_PERIOD_NS": CLK_PS / 1e3, "VCD": vcd, "SCOPE": TOP, "OUT": wd / "power.rpt"}
    for attempt in range(2):
        (wd / "power.rpt").unlink(missing_ok=True)
        r = e30.docker(["sta", "-no_splash", "-exit", "flows/power/sta_power.tcl"], env, "4g")
        if r.returncode == 0 and (wd / "power.rpt").exists():
            return {**e23.groups((wd / "power.rpt").read_text()), **e13.parse_power((wd / "power.rpt").read_text())}
        print(f"STA attempt {attempt} rc={r.returncode}: {r.stdout[-500:]}{r.stderr[-500:]}", flush=True)
    raise RuntimeError(f"STA power failed: {r.stdout[-1500:]}{r.stderr[-1500:]}")


def window_energy(net: Path, prog_path: Path, wd: Path, e_access_pj: float, sdf: Path | None = None) -> dict:
    wd.mkdir(parents=True, exist_ok=True)
    prog = load_program(prog_path)
    g = gls(net, prog_path, wd, sdf)
    sliced = wd / "window.vcd"
    slice_vcd(g["vcd"], sliced, g["t_start_ns"], g["t_end_ns"], rebase=True)
    g["vcd"].unlink()
    pw = sta_power(net, sliced, wd)
    sliced.unlink()
    dur = (g["t_end_ns"] - g["t_start_ns"]) * 1e-9
    macs = g["n_ops"] * MACS_PER_OP
    k = dur / macs * 1e15
    e_logic = (pw["p_total_w"] - pw["p_sram_sta_w"]) * k
    e_sram = e13.sram_accesses(prog, DESIGN) * e_access_pj * 1e3 / macs
    return {"n_ops": g["n_ops"], "stall_cycles": g.get("stall_cycles", ""), "mismatches": g["mismatches"],
            "window_ns": dur * 1e9, "cycles_per_op": dur * 1e12 / CLK_PS / g["n_ops"], "macs": macs, **pw,
            "e_logic_fj_per_mac": e_logic, "e_sram_fj_per_mac": e_sram, "e_total_fj_per_mac": e_logic + e_sram,
            "e_seq_fj_per_mac": pw["p_sequential_w"] * k, "e_comb_fj_per_mac": pw["p_combinational_w"] * k,
            "e_clock_fj_per_mac": pw["p_clock_w"] * k, "e_leak_fj_per_mac": pw["p_leakage_w"] * k}


def e_access() -> float:
    import yaml

    return yaml.safe_load((REPO_ROOT / "flows/pdk/sram_macros.yaml").read_text())["macros"][sweep.SRAM]["read_energy_pj_per_word_tt"]


def phase_power(recipe: str, windows: list[str], paper: bool) -> None:
    net = BUILD / "synth" / recipe / f"{DESIGN}_netlist.v"
    cfg = {"design": DESIGN, "recipe": recipe, "windows": windows, "netlist_sha256": sha256(net), "clk_ps": CLK_PS,
           "sram_access_pj": e_access(), "program_sha256": {w: sha256(BUILD / "programs" / f"{w}.npz") for w in windows},
           "synth_run": str(latest("e34_d2_tile_synth").resolve().name),
           "flow": "e30 phase_power: Icarus zero-delay GLS, VCD sliced + rebased (B9), docker OpenSTA 2.6.0 report_power, "
                   "OpenRAM Liberty power excluded, SRAM per-access charge; harness tb/cocotb/test_bs4_top.py"}
    with start_run("e34_d2_tile_power", cfg, paper=paper) as run:
        rows = []
        for w in windows:
            t0 = time.time()
            wd = BUILD / "power" / recipe / w
            r = window_energy(net, BUILD / "programs" / f"{w}.npz", wd, e_access())
            shutil.copy(wd / "power.rpt", run.path(f"power_{recipe}_{w}.rpt"))
            rows.append({"recipe": recipe, "design": DESIGN, "window": w, "kind": w[:4], **r, "wall_s": time.time() - t0})
            run.save_csv("power.csv", rows)
            print(f"{w:6s} {recipe} E/MAC {r['e_total_fj_per_mac']:8.1f} fJ (logic {r['e_logic_fj_per_mac']:.1f}, "
                  f"seq {r['e_seq_fj_per_mac']:.0f} comb {r['e_comb_fj_per_mac']:.0f}) cyc/op {r['cycles_per_op']:.3f} "
                  f"mismatches {r['mismatches']} {time.time() - t0:.0f}s", flush=True)
        run.log_metrics({"mismatches_total": int(sum(r["mismatches"] for r in rows)), "windows": windows})


def phase_sdf(recipe: str, paper: bool) -> None:
    sdfm = _load("sdf_subset_run", REPO_ROOT / "experiments/e13_power/sdf_subset.py")
    net = BUILD / "synth" / recipe / f"{DESIGN}_netlist.v"
    e13m = sdfm.e13
    e13m.NETDIR = net.parent
    sdfm.BUILD = BUILD / "sdf" / recipe
    sdfm.BUILD.mkdir(parents=True, exist_ok=True)
    cfg = {"design": DESIGN, "recipe": recipe, "windows": SDFW, "netlist_sha256": sha256(net),
           "sdf": "OpenSTA cell delays, no parasitics (sdf_subset.write_sdf), reuse 128 / 4 sets like e13_sdf",
           "program_sha256": {w: sha256(BUILD / "programs" / f"{w}.npz") for w in SDFW}}
    with start_run("e34_d2_tile_sdf", cfg, paper=paper) as run:
        sdf = sdfm.write_sdf(DESIGN, TOP, CLK_PS / 1e3)
        rows = []
        for w in SDFW:
            res = {}
            for mode, s in (("zero", None), ("sdf", sdf)):
                wd = BUILD / "sdf" / recipe / f"{w}_{mode}"
                res[mode] = window_energy(net, BUILD / "programs" / f"{w}.npz", wd, e_access(), s)
            rows.append({"recipe": recipe, "design": DESIGN, "window": w, "e_zero_fj_per_mac": res["zero"]["e_total_fj_per_mac"],
                         "e_sdf_fj_per_mac": res["sdf"]["e_total_fj_per_mac"],
                         "sdf_over_zero": res["sdf"]["e_total_fj_per_mac"] / res["zero"]["e_total_fj_per_mac"],
                         "mismatches": res["zero"]["mismatches"] + res["sdf"]["mismatches"]})
            run.save_csv("sdf_ratio.csv", rows)
            print(rows[-1], flush=True)
        run.log_metrics({"sdf_over_zero": float(np.mean([r["sdf_over_zero"] for r in rows])),
                         "mismatches_total": int(sum(r["mismatches"] for r in rows))})


# ------------------------------------------------------------------------------------------------ summary
def wire_energy_nom(prog: dict, area_mm2: float) -> float:
    """Boundary wire energy (fJ) of the INT4-SM tile on `prog`, nominal band; same model as cimfp4.power.boundary:
    the magnitude bits cross the row wires one bit per row per cycle (3 per op, LSB-first), the sign once per op."""
    import math

    import yaml

    from cimfp4.power.boundary import VDD, _toggles, weight_move_toggles

    model = yaml.safe_load((REPO_ROOT / "flows/pdk/wire_model.yaml").read_text())
    c = model["layers"]["met2"]["cap_ff_per_um"]
    side = math.sqrt(area_mm2) * 1e3
    tog = lambda length: 0.5 * c * length * VDD ** 2  # noqa: E731
    x = prog["x"]
    mag, sgn = (x & 7).astype(np.int64), ((x >> 3) & 1).astype(np.int64)
    bits = ((mag[:, :, None]) >> np.arange(3)) & 1  # (T, 32, 3)
    stream = bits.transpose(0, 2, 1).reshape(-1, x.shape[1])  # (T*3, 32)
    bcast = (_toggles(stream, 1) + _toggles(sgn, 1)) * tog(side)
    return bcast + weight_move_toggles(prog) * tog(0.5 * side)


def merge_rows(exp: str, run_ids: list[str] | None, recipe: str) -> list[dict]:
    dirs = [R / exp / r for r in run_ids] if run_ids else [latest(exp)]
    return [r for d in dirs for r in load_csv(d / "power.csv") if r["recipe"] == recipe]


def phase_summary(recipe: str, power_runs: list[str], sdf_run: str | None, synth_run: str | None, paper: bool) -> None:
    from cimfp4.power.g1_compare import paired_ratio

    sfx = REF_SFX[recipe]
    rng = seed_all(0)
    mine = {r["window"]: r for r in merge_rows("e34_d2_tile_power", power_runs, recipe)}
    synth = load_csv((R / "e34_d2_tile_synth" / synth_run if synth_run else latest("e34_d2_tile_synth")) / "synth.csv")
    synth = next(r for r in synth if r["recipe"] == recipe)
    area_mm2 = float(synth["total_area_um2"]) / 1e6
    # reference tiles: e13_boundary{sfx} (gate + wires, nominal band) for v2 / INT5 / DPU, e13_power{sfx} for cycles
    ref = {(r["design"], r["window"]): r for r in load_csv(latest("e13_boundary" + sfx) / "power_boundary.csv")}
    bcfg = json.loads(json.dumps(__import__("yaml").safe_load((latest("e13_boundary" + sfx) / "config.yaml").read_text())))
    cyc = {}
    for rid in bcfg.get("source_runs", []):
        for r in load_csv(R / ("e13_power" + sfx) / rid / "power.csv"):
            cyc[(r["design"], r["window"])] = float(r["window_ns"]) * 1e3 / CLK_PS / float(r["n_ops"])
    sdfref = {k.replace("_sdf_over_zero", ""): v for k, v in json.loads((latest("e13_sdf" + sfx) / "metrics.json").read_text()).items()
              if k.endswith("_sdf_over_zero")}
    sdf4 = None
    if sdf_run:
        sdf4 = float(json.loads((R / "e34_d2_tile_sdf" / sdf_run / "metrics.json").read_text())["sdf_over_zero"])
    V2 = "v2_sram_m16"
    cfg = {"recipe": recipe, "power_runs": power_runs, "sdf_run": sdf_run, "synth_run": synth_run or latest("e34_d2_tile_synth").resolve().name,
           "reference": {"boundary": latest("e13_boundary" + sfx).resolve().name, "sdf": latest("e13_sdf" + sfx).resolve().name,
                         "e13_power_runs": bcfg.get("source_runs")},
           "boundary_model": "e13 add_boundary (met2 cap, sqrt(area) row wires, nominal band); INT4-SM: 3 magnitude bits/cycle + sign",
           "area_um2_source": "e34_d2_tile_synth (INT4-SM), e30_td_synth td (others)" if recipe == "td" else "e34_d2_tile_synth, e12_synth"}
    with start_run("e34_d2_tile" + ("" if recipe == "td" else "_" + recipe), cfg, paper=paper) as run:
        assert all(int(r["mismatches"]) == 0 for r in mine.values()), "GLS mismatch in an e34 power run"
        windows = [w for w in REAL if w in mine]
        assert len(windows) == 8, f"need 8 real windows, got {len(windows)}"
        e4 = {}
        for w in WINDOWS:
            if w in mine:
                prog = load_program(BUILD / "programs" / f"{w}.npz")
                wire = wire_energy_nom(prog, area_mm2) / float(mine[w]["macs"])
                e4[w] = float(mine[w]["e_total_fj_per_mac"]) + wire
        energies = {d: {w: float(ref[(d, w)]["e_total_fj_per_mac_nom"]) for w in windows} for d in (V2, "bs_int5", "dpu")}
        energies[DESIGN] = {w: e4[w] for w in windows}
        sdfd = {V2: sdfref.get(V2), "bs_int5": sdfref.get("bs_int5"), "dpu": sdfref.get("dpu"), DESIGN: sdf4}
        table = []

        def pr(name, num, den, model):
            en = np.array([energies[num][w] for w in windows]) * (sdfd[num] if model == "sdf_corrected" else 1.0)
            ed = np.array([energies[den][w] for w in windows]) * (sdfd[den] if model == "sdf_corrected" else 1.0)
            table.append({"ratio": name, "delay_model": model, **{f"ee_{k}": v for k, v in paired_ratio(en, ed, rng).items()}})

        for model in ("zero", "sdf_corrected"):
            if model == "sdf_corrected" and sdf4 is None:
                continue
            pr("EE(v2)/EE(INT4-SM) = E(int4sm)/E(v2)", DESIGN, V2, model)
            pr("EE(v2)/EE(INT5)    = E(int5)/E(v2)", "bs_int5", V2, model)
            pr("EE(INT4-SM)/EE(INT5) = E(int5)/E(int4sm)", "bs_int5", DESIGN, model)
            pr("EE(v2)/EE(DPU)     = E(dpu)/E(v2)", "dpu", V2, model)
            pr("EE(INT4-SM)/EE(DPU) = E(dpu)/E(int4sm)", "dpu", DESIGN, model)
        run.save_csv("ratios.csv", table)
        # per-design absolute rows
        a_rows = []
        for d in (V2, DESIGN, "bs_int5", "dpu"):
            v = np.array([energies[d][w] for w in windows])
            rows_d = {"design": d, "e_nom_fj_per_mac_real_mean": float(v.mean()), "e_nom_fj_per_mac_real_min": float(v.min()),
                      "e_nom_fj_per_mac_real_max": float(v.max()), "sdf_over_zero": sdfd[d]}
            if d == DESIGN:
                rows_d |= {"cycles_per_op_mean": float(np.mean([float(mine[w]["cycles_per_op"]) for w in windows])),
                           "e_gate_logic_fj_per_mac": float(np.mean([float(mine[w]["e_logic_fj_per_mac"]) for w in windows])),
                           "e_sram_fj_per_mac": float(np.mean([float(mine[w]["e_sram_fj_per_mac"]) for w in windows])),
                           "e_seq_fj_per_mac": float(np.mean([float(mine[w]["e_seq_fj_per_mac"]) for w in windows])),
                           "e_comb_fj_per_mac": float(np.mean([float(mine[w]["e_comb_fj_per_mac"]) for w in windows])),
                           "e_nom_fj_per_mac_rand_mean": float(np.mean([e4[w] for w in ("rand0", "rand1") if w in e4]))}
            else:
                rows_d["cycles_per_op_mean"] = float(np.mean([cyc[(d, w)] for w in windows if (d, w) in cyc]))
            a_rows.append(rows_d)
        # area / density: td areas for v2/INT5/DPU from e30 (as e15 --area-recipe), INT4-SM from e34 synth
        f_g1 = json.loads((latest("e12_synth") / "metrics.json").read_text())["f_g1_mhz"]
        if recipe == "td":
            areas = {r["design"]: float(r["total_area_um2"]) for r in load_csv(latest("e30_td_synth") / "synth.csv") if r["recipe"] == "td"}
            logic = {r["design"]: float(r["logic_area_um2"]) for r in load_csv(latest("e30_td_synth") / "synth.csv") if r["recipe"] == "td"}
        else:
            e12 = {r["design"]: r for r in load_csv(latest("e12_synth") / "synth.csv")}
            areas = {d: float(e12[d]["total_area_um2"]) for d in (V2, "bs_int5", "dpu")}
            logic = {d: float(e12[d]["logic_area_um2"]) for d in (V2, "bs_int5", "dpu")}
        areas[DESIGN], logic[DESIGN] = float(synth["total_area_um2"]), float(synth["logic_area_um2"])
        mpc = {V2: 128, "bs_int5": 128 / 5, "dpu": 128, DESIGN: 128 / 3}
        for r in a_rows:
            d = r["design"]
            r |= {"total_area_um2": areas[d], "logic_area_um2": logic[d], "mac_per_cycle_nominal": mpc[d],
                  "gops_per_mm2_at_fg1": 2 * mpc[d] * f_g1 * 1e6 / 1e9 / (areas[d] / 1e6)}
        run.save_csv("designs.csv", a_rows)
        by = {r["design"]: r for r in a_rows}
        m = {"recipe": recipe, "gls_mismatches": 0, "n_real_windows": len(windows),
             "e_int4sm_fj_per_mac": by[DESIGN]["e_nom_fj_per_mac_real_mean"], "cycles_per_op_int4sm": by[DESIGN]["cycles_per_op_mean"],
             "area_total_um2_int4sm": areas[DESIGN], "area_logic_um2_int4sm": logic[DESIGN],
             "gops_mm2_int4sm_over_int5": by[DESIGN]["gops_per_mm2_at_fg1"] / by["bs_int5"]["gops_per_mm2_at_fg1"],
             "gops_mm2_v2_over_int4sm": by[V2]["gops_per_mm2_at_fg1"] / by[DESIGN]["gops_per_mm2_at_fg1"]}
        for t in table:
            key = re.sub(r"[^a-z0-9]+", "_", t["ratio"].split(" = ")[0].lower()).strip("_") + "_" + t["delay_model"]
            m[key], m[key + "_ci_lo"], m[key + "_ci_hi"] = t["ee_mean"], t["ee_ci_lo"], t["ee_ci_hi"]
        run.log_metrics(m)
        print(json.dumps(m, indent=1))
        for t in table:
            print(f"{t['ratio']:44s} {t['delay_model']:14s} {t['ee_mean']:.3f} [{t['ee_ci_lo']:.3f}, {t['ee_ci_hi']:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--programs", action="store_true")
    ap.add_argument("--synth", choices=list(RECIPES))
    ap.add_argument("--power", choices=list(RECIPES))
    ap.add_argument("--sdf", choices=list(RECIPES))
    ap.add_argument("--summary", choices=list(RECIPES))
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--power-runs", default="", help="summary: comma-separated e34_d2_tile_power run ids")
    ap.add_argument("--sdf-run", default="")
    ap.add_argument("--synth-run", default="")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    if a.capture:
        phase_capture(a.paper)
    if a.programs:
        phase_programs(a.paper)
    if a.synth:
        phase_synth(a.synth, a.paper)
    if a.power:
        phase_power(a.power, a.windows.split(","), a.paper)
    if a.sdf:
        phase_sdf(a.sdf, a.paper)
    if a.summary:
        phase_summary(a.summary, [r for r in a.power_runs.split(",") if r], a.sdf_run or None, a.synth_run or None, a.paper)


if __name__ == "__main__":
    main()
