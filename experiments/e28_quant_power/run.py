"""e28 - T2.3: cost of the online quantizer, tile anchor (+ buffer), HIER fp32 reducer and the E2M1 -> INT5 activation conversion.

Blocks (docs/spec/numerics.md "Online activation quantizers and tile anchor", "HIER interface"; T2.3):
  mx        act_quant_mx   (rtl/src, 32-element bf16 block per cycle)          -> E per 32-element block / per op
  nv        act_quant_nv   (rtl/src, 16-element block; 2 per 32-element op)    -> E per 16-element block, 2 per op
  anchor    anchor_detect  (rtl/src, running max of the block exponents)
  tilebuf   e28_tile_buf   [EST organisation] ping-pong flop buffer of one K-tile of codes+exponent (136 b / op)
  hier      e28_hier_reducer [EST model] int40 window -> fp32, fp32 RNE adder, one column's reducer (no RTL in the repo)
  int5      e28_int5_conv  E2M1 -> INT5 per-row conversion of the bit-serial INT5 baseline (verbatim copy of its g_xrow logic;
                           already inside the INT5 macro netlist, measured here only to isolate its share)
Flow = the T2.6 `td` recipe (flows/synth/sweep.py + OpenROAD repair, same as e30) then, per block, zero-delay Icarus GLS
bit-exact against a numpy golden, VCD sliced to the window and rebased to t=0, OpenSTA `report_power` (docker image as e13/e30).
Stimulus: the 8 real e13_power_rb windows (replayed from seed 0: same capture, K-blocks and tokens) with their raw bf16
activations re-captured from the models, plus the 2 random control windows.

Phases:
  --capture   (GPU) re-run the e04 prefill forward and save the raw bf16 activations of the 8 real windows -> build/e28/raw
  --synth     td synthesis + STA of the six blocks                                 -> results/e28_quant_synth
  --power D   GLS + power of block D (mx|nv|anchor|tilebuf|int5|hier)             -> results/e28_quant_power_<D>
  --summary   per-block / per-op / per-MAC table and shares of the macro energy    -> results/e28_quant_summary
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.formats.nv_quant import nv_tensor_scale, quantize_nvfp4
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import load_program
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

HERE = REPO_ROOT / "experiments/e28_quant_power"
BUILD = REPO_ROOT / "build/e28"
PROGS = REPO_ROOT / "build/power/e13_power_rb/programs"
CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
RTL = HERE / "rtl"
PERIOD_NS = 25.0
P_PS = 25000
T0 = 112.5  # first window edge (ns): clock edge 4
N_TOK = 128  # tokens per window (x TILE blocks = ops)
NV_TOK = 32  # tokens per NV window (the NV quantizer netlist is 49k cells: 128 tokens take > 20 min of Icarus per window)
TILE = 8  # blocks per K-tile = the 8 weight sets used by the e13 programs
NCOL = 4
WIN_W, BASE_GUARD = 40, 3  # window bits CIM_WINDOW_BITS, guard = ceil(log2 TILE)
MAC_PER_OP = 32 * 32  # one 32-element activation block feeds 32 rows x 32 columns of the macro (e21)
REAL = [f"real{i}" for i in range(8)]
RAND = ["rand0", "rand1"]
WINDOWS = REAL + RAND
DUTS = ["mx", "nv", "anchor", "tilebuf", "tilebuf_cg", "int5", "hier"]
NV_SRC_FIX = BUILD / "act_quant_nv_ysfix.sv"

# ---------------------------------------------------------------- DUT descriptions
SPEC = {
    "mx": dict(mod="act_quant_mx", clocked=False, src=["rtl/src/act_quant_mx.sv"], hier_top=None,
               ins=[("x_bf16", 512)], tie={"mode_rceil": "1'b0"}, outs=[("codes", 128), ("scale_code", 8), ("scale_exp", 8),
                                                                      ("nan", 1)], obs=["scale_code", "codes"]),
    "nv": dict(mod="act_quant_nv", clocked=False, src=[str(NV_SRC_FIX)], ins=[("x_bf16", 256)],
               static=[("c_fp32", 32), ("sdec_fp32", 32)], outs=[("codes", 64), ("scale_e4m3", 8)],
               obs=["scale_e4m3", "codes"]),
    "anchor": dict(mod="anchor_detect", clocked=True, src=["rtl/src/anchor_detect.sv"],
                   ins=[("valid", 1), ("last", 1), ("e_x", 8)], outs=[("anchor", 8), ("anchor_valid", 1)],
                   obs=["anchor_valid", "anchor"]),
    "tilebuf": dict(mod="e28_tile_buf", clocked=True, src=["experiments/e28_quant_power/rtl/e28_tile_buf.sv"],
                    params={"TILE": TILE}, ins=[("wr", 1), ("din", 136)], outs=[("dout", 136)], obs=["dout"]),
    "tilebuf_cg": dict(mod="e28_tile_buf_cg", clocked=True,
                       src=["rtl/src/cim_icg.sv", "experiments/e28_quant_power/rtl/e28_tile_buf_cg.sv"],
                       params={"TILE": TILE}, ins=[("wr", 1), ("din", 136)], outs=[("dout", 136)], obs=["dout"]),
    "int5": dict(mod="e28_int5_conv", clocked=False, src=["experiments/e28_quant_power/rtl/e28_int5_conv.sv"],
                 ins=[("op_x", 128)], outs=[("xd_all", 160)], obs=["xd_all"]),
    "hier": dict(mod="e28_hier_reducer", clocked=True, src=["experiments/e28_quant_power/rtl/e28_hier_reducer.sv"],
                 ins=[("valid", 1), ("clear", 1), ("win", 40), ("lsb", 10)], outs=[("acc", 32)], obs=["acc"]),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- stimulus (replay of the e13_power_rb windows)
def replay_windows() -> list[dict]:
    """Replay the rng stream of e13 (seed 0) to recover, per real window, the capture file, the weight rows, the 16
    K-blocks (8 used) and the 256 tokens; asserts that the result reproduces the stored e13_power_rb programs."""
    from cimfp4.llm.capture import load_capture

    rng = seed_all(0)
    caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
    out = []
    for w in range(8):
        cpath = caps[int(rng.integers(0, len(caps)))]
        qw, qa = load_capture(cpath, "mxfp4")
        r0 = int(rng.integers(0, qw.codes.shape[0] - NCOL))
        blocks = rng.choice(qw.num_blocks, size=min(16, qw.num_blocks), replace=False)
        tokens = rng.choice(qa.codes.shape[0], size=256, replace=qa.codes.shape[0] < 256)
        prog = load_program(PROGS / f"real{w}.npz")
        x = np.array([qa.blocked_codes()[t, blocks[s]] for s in range(8) for t in tokens], np.uint8)
        if not np.array_equal(x, prog["x"]):
            raise RuntimeError(f"replay of real{w} does not reproduce the stored e13_power_rb program")
        with np.load(cpath) as z:
            module = str(z["module"])
        out.append({"name": f"real{w}", "capture": str(cpath.relative_to(REPO_ROOT)), "model": cpath.parts[-3],
                    "module": module, "r0": r0, "blocks": blocks[:8].tolist(), "tokens": tokens.tolist(),
                    "k_blocks": int(qa.num_blocks)})
    return out


def bf16_rne(x32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x32, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f64(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def phase_capture(paper: bool) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import yaml

    from cimfp4.llm.capture import ActivationRecorder

    wins = replay_windows()
    models = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    (BUILD / "raw").mkdir(parents=True, exist_ok=True)
    cfg = {"windows": [{k: v for k, v in w.items() if k != "tokens"} for w in wins], "seqlen": 512, "prefill_windows": 4,
           "dtype": "float32 forward as e04, activations rounded to bf16 (RNE)", "dataset": "wikitext-2-raw-v1 test"}
    with start_run("e28_capture", cfg, paper=paper) as run:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        rows = []
        for mname in sorted({w["model"].split("@")[0] for w in wins}):
            spec = models[mname]
            snap_root = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["hf_id"].replace("/", "--")) / "snapshots"
            rev = next(d for d in snap_root.iterdir() if d.name.startswith(wins[[w["model"].split("@")[0] for w in wins]
                                                                          .index(mname)]["model"].split("@")[1]))
            tok = AutoTokenizer.from_pretrained(str(rev))
            model = AutoModelForCausalLM.from_pretrained(str(rev), torch_dtype=torch.float32).to(dev).eval()
            need = {w["module"] for w in wins if w["model"].startswith(mname)}
            mods = {n: m for n, m in model.named_modules() if n in need}
            ids = tok(text, return_tensors="pt").input_ids[0]
            starts = np.linspace(0, len(ids) - 512 - 1, 4).astype(int)
            with torch.no_grad(), ActivationRecorder(mods) as rec:
                for s in starts:
                    model(ids[s: s + 512][None].to(dev))
            for w in [w for w in wins if w["model"].startswith(mname)]:
                raw = rec.stacked(w["module"])  # (2048, K) fp32
                with np.load(REPO_ROOT / w["capture"]) as z:
                    stored_codes, stored_exp = z["mxfp4_a_codes"], z["mxfp4_a_scale_exp"]
                    t_amax_stored = float(z["nvfp4_a_tensor_scale"]) * 6 * 448
                tok_idx = np.array(w["tokens"][:N_TOK])
                blk = np.array(w["blocks"])
                sel = np.stack([raw[tok_idx][:, b * 32:(b + 1) * 32] for b in blk], axis=1)  # (tok, 8, 32)
                q = quantize_mxfp4(bf16_f64(bf16_rne(sel)).reshape(-1, 32), "ocp_v11")
                sc = stored_codes.reshape(stored_codes.shape[0], -1, 32)[tok_idx][:, blk]
                se = stored_exp[tok_idx][:, blk]
                row = {"window": w["name"], "model": w["model"], "module": w["module"],
                       "codes_match_frac": float(np.mean(q.codes.reshape(sc.shape) == sc)),
                       "scale_match_frac": float(np.mean(q.scale_exp.reshape(se.shape) == se)),
                       "t_amax_raw": float(np.abs(raw).max()), "t_amax_stored_from_e04": t_amax_stored}
                rows.append(row)
                np.savez(BUILD / "raw" / f"{w['name']}.npz", bf16=bf16_rne(sel), t_amax=np.float64(row["t_amax_raw"]),
                         tokens=tok_idx, blocks=blk)
                print(row, flush=True)
            del model, rec
            torch.cuda.empty_cache()
        run.save_csv("capture_check.csv", rows)
        for w in WINDOWS[:8]:
            shutil.copy(BUILD / "raw" / f"{w}.npz", run.path(f"raw_{w}.npz"))
        run.log_metrics({"min_codes_match_frac": min(r["codes_match_frac"] for r in rows),
                         "min_scale_match_frac": min(r["scale_match_frac"] for r in rows),
                         "raw_sha256": {w: sha256(BUILD / "raw" / f"{w}.npz") for w in WINDOWS[:8]}})


# ---------------------------------------------------------------- vectors + goldens
def hexw(v: int, bits: int) -> str:
    return f"{v & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def pack(vals: list[tuple[int, int]]) -> int:
    """[(value, width)] first = MSB."""
    word = 0
    for v, w in vals:
        word = (word << w) | (v & ((1 << w) - 1))
    return word


def rand_bf16_blocks(rng, n: int, per: int) -> np.ndarray:
    """50 %-random control: random sign + mantissa bits, element exponent field in a narrow band (block-level spread)."""
    e = 127 + rng.integers(-3, 1, (n, 1)) - rng.integers(0, 3, (n, per))
    s = rng.integers(0, 2, (n, per))
    m = rng.integers(0, 128, (n, per))
    return ((s << 15) | (e << 7) | m).astype(np.uint16)


def program_stream(name: str):
    """Token-major (token, set) stream of the window: codes (n,32), e_x, plus the program for the INT5 order."""
    prog = load_program(PROGS / f"{name}.npz")
    nt = 256
    xi = np.array([s * nt + t for t in range(N_TOK) for s in range(TILE)])
    return prog, prog["x"][xi], prog["e_x"][xi]


def window_vectors(dut: str, name: str, wins: dict) -> dict:
    """Per-cycle input words + expected (golden) output words + care masks for one DUT and one window."""
    spec = SPEC[dut]
    rng = np.random.default_rng(1000 + WINDOWS.index(name))
    real = name.startswith("real")
    ins_w = [w for _, w in spec["ins"]]
    prog, xs, exs = program_stream(name)
    n_ops = N_TOK * TILE
    out: dict = {"static": {}}
    if dut in ("mx", "nv"):
        if real:
            raw = np.load(BUILD / "raw" / f"{name}.npz")
            blocks = raw["bf16"].reshape(n_ops, 32)  # (token, set) order = token-major
            t_amax = float(raw["t_amax"])
        else:
            blocks = rand_bf16_blocks(rng, n_ops, 32)
            t_amax = float(np.abs(bf16_f64(blocks)).max())
        if dut == "mx":
            vals = bf16_f64(blocks)
            q = quantize_mxfp4(vals, "ocp_v11")
            ins = [pack([(sum(int(v) << (16 * i) for i, v in enumerate(b)), 512)]) for b in blocks]
            exp = [pack([(int(q.scale_exp[k, 0]) + 127, 8), (sum(int(c) << (4 * i) for i, c in enumerate(q.codes[k])), 128)])
                   for k in range(n_ops)]
            assert not q.scale_nan.any()
            out["scale_exp"] = q.scale_exp[:, 0].astype(np.int64)  # for the anchor stream
            out["codes128"] = [sum(int(c) << (4 * i) for i, c in enumerate(q.codes[k])) for k in range(n_ops)]
        else:
            half = blocks[: NV_TOK * TILE].reshape(NV_TOK * TILE * 2, 16)  # op = two consecutive 16-element NV blocks
            vals = bf16_f64(half)
            s_dec = nv_tensor_scale(t_amax)
            s_enc = np.float32(np.float32(1.0) / s_dec)
            c = np.float32(s_enc / np.float32(6.0))
            q = quantize_nvfp4(vals, "hw", tensor_amax=t_amax)
            out["static"] = {"c_fp32": int(c.view(np.uint32)), "sdec_fp32": int(np.float32(s_dec).view(np.uint32))}
            ins = [pack([(sum(int(v) << (16 * i) for i, v in enumerate(b)), 256)]) for b in half]
            exp = [pack([(int(q.scale_e4m3[k, 0]), 8), (sum(int(cd) << (4 * i) for i, cd in enumerate(q.codes[k])), 64)])
                   for k in range(half.shape[0])]
            out["n_items_per_op"] = 1
        care = [(1 << sum(dict(spec["outs"])[o] for o in spec["obs"])) - 1] * len(exp)
        out.update(ins=ins, exp=exp, care=care)
    elif dut == "int5":
        x = prog["x"]  # program order (set-major, as the INT5 macro consumes it)
        ins = [pack([(sum(int(c) << (4 * r) for r, c in enumerate(row)), 128)]) for row in x]
        exp = []
        for row in x:
            word = 0
            for r, c in enumerate(row):
                c = int(c)
                xe = (c >> 1) & 3
                xm = ((1 if xe else 0) << 1 | (c & 1)) << (0 if xe == 0 else xe - 1)
                xd = (-xm if c & 8 else xm) & 31
                word |= xd << (5 * r)
            exp.append(word)
        out.update(ins=ins, exp=exp, care=[(1 << 160) - 1] * len(exp))
    elif dut in ("anchor", "tilebuf", "tilebuf_cg"):
        if dut == "anchor":
            ins, exp, run_max, have, anchor, av = [], [], 0, False, 0, 0
            for i, ex in enumerate(exs):
                last = int((i % TILE) == TILE - 1)
                ex = int(ex)
                ins.append(pack([(1, 1), (last, 1), (ex, 8)]))
                av = 0
                if last:
                    anchor = ex if (not have or ex > run_max) else run_max
                    av, have = 1, False
                else:
                    run_max = ex if (not have or ex > run_max) else run_max
                    have = True
                exp.append(pack([(av, 1), (anchor, 8)]))
            out.update(ins=ins, exp=exp, care=[(1 << 9) - 1] * len(exp))
        else:
            depth = 2 * TILE
            mem, wp, ins, exp, care = [None] * depth, 0, [], [], []
            for i in range(len(xs)):
                din = pack([(int(exs[i]) & 255, 8), (sum(int(c) << (4 * r) for r, c in enumerate(xs[i])), 128)])
                ins.append(pack([(1, 1), (din, 136)]))
                mem[wp] = din
                wp = (wp + 1) % depth
                v = mem[(wp + TILE) % depth]
                exp.append(0 if v is None else v)
                care.append(0 if v is None else (1 << 136) - 1)
            out.update(ins=ins, exp=exp, care=care)
    elif dut == "hier":
        ins, exp, care = [], [], []
        acc = np.float32(0)
        if real:
            wsets, ew = prog["sets"], prog["ew"]
            tab = np.array([0, 1, 2, 3, 4, 6, 8, 12])
            base = WIN_W - 14 - 1 - BASE_GUARD

            def sval(code):
                code = np.asarray(code, np.int64)
                return np.where(code & 8, -1, 1) * tab[code & 7]

            for c in range(NCOL):
                for ti in range(N_TOK):
                    ex = np.array([prog["e_x"][s * 256 + ti] for s in range(TILE)])
                    maxw, maxx = int(ew[:TILE, c].max()), int(ex.max())
                    win = 0
                    for s in range(TILE):
                        sv = int((sval(wsets[s, c]) * sval(prog["x"][s * 256 + ti])).sum())
                        sh = base - ((maxw - int(ew[s, c])) + (maxx - int(ex[s])))
                        win += (sv << sh) if sh >= 0 else (sv >> (-sh))
                    lsb = maxw + maxx - 2 - base
                    assert -(1 << 39) <= win < (1 << 39)
                    clear = int(ti % 4 == 0)
                    b = np.float32(np.ldexp(float(win), lsb))
                    acc = b if clear else np.float32(acc + b)
                    assert acc == 0 or abs(acc) > 2.0**-120
                    ins.append(pack([(1, 1), (clear, 1), (win, 40), (lsb, 10)]))
                    exp.append(int(np.float32(acc).view(np.uint32)))
        else:
            for c in range(NCOL):
                for ti in range(N_TOK):
                    win = int(rng.integers(-(1 << 38), 1 << 38))
                    lsb = int(rng.integers(-30, -19))
                    clear = int(ti % 4 == 0)
                    b = np.float32(np.ldexp(float(win), lsb))
                    acc = b if clear else np.float32(acc + b)
                    ins.append(pack([(1, 1), (clear, 1), (win, 40), (lsb, 10)]))
                    exp.append(int(np.float32(acc).view(np.uint32)))
        out.update(ins=ins, exp=exp, care=[(1 << 32) - 1] * len(exp))
    out["n_ops_stream"] = len(out["ins"])
    out["in_bits"] = sum(ins_w)
    return out


# ---------------------------------------------------------------- testbench / simulation
def make_tb(dut: str, n: int, vec: Path, exp: Path, care: Path, static: dict) -> str:
    s = SPEC[dut]
    in_w = sum(w for _, w in s["ins"])
    obs_w = sum(dict(s["outs"])[o] for o in s["obs"])
    decl = "".join(f"  reg [{w - 1}:0] {n_};\n" for n_, w in s["ins"])
    decl += "".join(f"  reg [{w - 1}:0] {n_};\n" for n_, w in s.get("static", []))
    decl += "".join(f"  wire [{w - 1}:0] {n_};\n" for n_, w in s["outs"])
    conns = [f".{n_}({n_})" for n_, _ in s["ins"]] + [f".{n_}({n_})" for n_, _ in s.get("static", [])]
    conns += [f".{k}({v})" for k, v in s.get("tie", {}).items()] + [f".{n_}({n_})" for n_, _ in s["outs"]]
    if s["clocked"]:
        conns += [".clk(clk)", ".rst_n(rst_n)"]
    params = ""  # the synthesised netlist is already specialised (no parameters)
    unpack, off = "", in_w
    for n_, w in s["ins"]:
        off -= w
        unpack += f"      {n_} = vec[i][{off + w - 1}:{off}];\n"
    st = "".join(f"    {n_} = {w}'h{static[n_]:x};\n" for n_, w in s.get("static", []))
    zero = "".join(f"    {n_} = 0;\n" for n_, _ in s["ins"])
    obs = "{" + ", ".join(s["obs"]) + "}"
    chk_t = f"{T0 + 25 + 20}" if s["clocked"] else f"{T0 + 20}"
    return f"""`timescale 1ns/1ps
module tb;
  localparam N = {n};
  reg [{in_w - 1}:0] vec [0:N-1];
  reg [{obs_w - 1}:0] expv [0:N-1];
  reg [{obs_w - 1}:0] care [0:N-1];
{decl}  reg clk = 0;
  reg rst_n = 0;
  integer i, k, bad;
  always #12.5 clk = ~clk;
  {s['mod']} {params}dut ({", ".join(conns)});
  initial begin
    $readmemh("{vec}", vec); $readmemh("{exp}", expv); $readmemh("{care}", care);
{zero}{st}    #50 rst_n = 1;
    #({T0} - 50 + 1);
    for (i = 0; i < N; i = i + 1) begin
{unpack}      #25;
    end
  end
  initial begin
    bad = 0;
    #({chk_t});
    for (k = 0; k < N; k = k + 1) begin
      if (((({obs}) ^ expv[k]) & care[k]) !== 0) begin
        bad = bad + 1;
        if (bad <= 3) $display("E28_MISMATCH i=%0d got=%h want=%h", k, {obs}, expv[k]);
      end
      #25;
    end
    $display("E28_RESULT mismatches=%0d n=%0d", bad, N);
    $finish;
  end
endmodule
"""


def gls(dut: str, net: Path, name: str, wd: Path, vecs: dict) -> dict:
    s = SPEC[dut]
    wd.mkdir(parents=True, exist_ok=True)
    n = len(vecs["ins"])
    in_w, obs_w = vecs["in_bits"], sum(dict(s["outs"])[o] for o in s["obs"])
    (wd / "vec.hex").write_text("\n".join(hexw(v, in_w) for v in vecs["ins"]) + "\n")
    (wd / "exp.hex").write_text("\n".join(hexw(v, obs_w) for v in vecs["exp"]) + "\n")
    (wd / "care.hex").write_text("\n".join(hexw(v, obs_w) for v in vecs["care"]) + "\n")
    (wd / "tb.v").write_text(make_tb(dut, n, wd / "vec.hex", wd / "exp.hex", wd / "care.hex", vecs["static"]))
    vcd = wd / "dump.vcd"
    cmd = ["iverilog", "-g2012", "-DFUNCTIONAL", "-DUNIT_DELAY=#0", f'-DVCD_FILE="{vcd}"', "-DVCD_TOP=tb.dut", "-s", "tb",
           "-s", "vcd_dump", "-o", str(wd / "sim.vvp"), str(CELLS / "primitives.v"), str(CELLS / "sky130_fd_sc_hd.v"), str(net),
           str(wd / "tb.v"), str(REPO_ROOT / "flows/power/vcd_dump.v")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    r = subprocess.run(["vvp", "-n", str(wd / "sim.vvp")], capture_output=True, text=True)
    (wd / "sim.log").write_text(r.stdout + r.stderr)
    m = re.search(r"E28_RESULT mismatches=(\d+) n=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"GLS produced no result: {r.stdout[-500:]}{r.stderr[-500:]}")
    return {"mismatches": int(m.group(1)), "n": int(m.group(2)), "vcd": vcd, "t_start_ns": T0, "t_end_ns": T0 + 25 * n}


# ---------------------------------------------------------------- synthesis
def prep_sources() -> None:
    """Yosys 0.69 rejects the unpacked-array localparam of act_quant_nv.sv; synthesise a text-substituted copy (the GLS
    check against the golden is the equivalence evidence; rtl/src is not touched)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    s = (REPO_ROOT / "rtl/src/act_quant_nv.sv").read_text()
    old = "localparam int NM [7] = '{1, 3, 5, 7, 10, 14, 20};"
    assert old in s and "NM[j]" in s
    s = s.replace(old, "function automatic int nm_f(input int j);\n    case (j) 0: nm_f = 1; 1: nm_f = 3; 2: nm_f = 5; 3: nm_f = 7; "
                       "4: nm_f = 10; 5: nm_f = 14; default: nm_f = 20; endcase\n  endfunction")
    NV_SRC_FIX.write_text(s.replace("NM[j]", "nm_f(j)"))
    base = (REPO_ROOT / "rtl/baselines/bitserial_int5/bs_column_top.sv").read_text()
    conv = (RTL / "e28_int5_conv.sv").read_text()
    for line in ("assign xe = op_x[4*r+1 +: 2];",
                 "assign xm = {2'b00, (xe != 2'b00), op_x[4*r]} << ((xe == 2'b00) ? 2'd0 : xe - 2'd1);",
                 "assign xd = op_x[4*r+3] ? (5'd0 - {1'b0, xm}) : {1'b0, xm};"):
        assert line in base and line in conv, f"INT5 conversion expression drifted from the baseline: {line}"


def repair_design_only(net: Path, top: str, out_net: Path, mem: str) -> dict:
    """Stage 2 of the td recipe without repair_timing (e30.repair with experiments/e28_quant_power/repair_design_only.tcl):
    repair_timing on the 55 ns-deep quantizer paths (target 25 ns) ran > 20 min of OpenROAD without converging."""
    env = {"TECH_LEF": e30.TECH_LEF, "CELL_LEF": e30.CELL_LEF, "SRAM_LEF": e30.SRAM_LEF, "LIB": sweep.LIB, "SRAM_LIB": e30.SRAM_LIB,
           "NETLIST": net, "TOP": top, "SDC": e30.SDC, "CLK_PERIOD_NS": PERIOD_NS, "MAX_FANOUT": e30.MAX_FANOUT,
           "MAX_TRAN_NS": e30.MAX_TRAN_NS, "OUT_NET": out_net, "OUT_RPT": f"{out_net}.rpt"}
    r = e30.docker(["openroad", "-no_splash", "-exit", str(HERE / "repair_design_only.tcl")], env, mem)
    Path(f"{out_net}.log").write_text(r.stdout + r.stderr)
    if r.returncode or not out_net.exists():
        raise RuntimeError(f"OpenROAD repair failed for {net}: {r.stdout[-1500:]}{r.stderr[-1500:]}")
    e30.unsign(out_net)
    viol = {k: int(m.group(1)) if (m := re.search(rf"Found (\d+) {k} violations", r.stdout)) else 0 for k in ("slew", "fanout", "cap")}
    return {"repair": "design_only", **{f"repair_left_{k}": v for k, v in viol.items()}}


def build_dut(dut: str, out: Path, recipe: str = "td") -> tuple[dict, Path]:
    s = SPEC[dut]
    out.mkdir(parents=True, exist_ok=True)
    sweep.SOURCES = [str(x) for x in s["src"]]
    sweep.RECIPE, sweep.HIER = recipe, False
    params = {k: v for k, v in s.get("params", {}).items()}
    # yosys_script reads rtl/include (defines) and the SRAM liberty; both are harmless for these blocks
    t0 = time.time()
    ys = sweep.yosys_script(dut, s["mod"], params, out)
    (out / f"{dut}.ys").write_text(ys)
    with (out / f"{dut}_yosys.log").open("w") as f:
        subprocess.run([shutil.which("yosys"), "-q", "-l", str(out / f"{dut}_yosys_full.log"), "-s", str(out / f"{dut}.ys")],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    net = out / f"{dut}_netlist.v"
    e30.unsign(net)
    row = {"dut": dut, "top": s["mod"], "recipe": recipe, "yosys_s": time.time() - t0}
    sdc = HERE / "constraints" / ("seq.sdc" if s["clocked"] else "comb.sdc")
    e30.SDC = sdc
    abc_net = out / f"{dut}_abc_netlist.v"
    net.replace(abc_net)
    row.update(repair_design_only(abc_net, s["mod"], net, "3g"))
    row.update(e30.stat(net, s["mod"], out, dut))
    env = {"NETLIST": net, "TOP": s["mod"], "LIB": sweep.LIB, "CLK_PERIOD_NS": PERIOD_NS, "SDC": sdc,
           "OUT": out / f"{dut}_sta.txt"}
    if s["clocked"]:
        env["OUT_R2R"] = out / f"{dut}_sta_r2r.txt"
    r = e30.docker(["sta", "-no_splash", "-exit", str(HERE / "sta.tcl")], env, "3g")
    if r.returncode:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    txt = (out / f"{dut}_sta.txt").read_text()
    wns = float(re.search(r"worst slack\s+(-?[\d.]+)", txt).group(1))
    arr = re.search(r"([\d.]+)\s+data arrival time", txt)
    row.update({"sta_period_ns": PERIOD_NS, "wns_ns": wns, "min_period_ns": PERIOD_NS - wns,
                "crit_arrival_ns": float(arr.group(1)) if arr else None})
    if s["clocked"]:
        t2 = (out / f"{dut}_sta_r2r.txt").read_text()
        m = re.search(r"(-?[\d.]+)\s+slack \(", t2)
        row["r2r_wns_ns"] = float(m.group(1)) if m else None
    return row, net


def phase_synth(paper: bool, duts: list[str]) -> None:
    prep_sources()
    cfg = {"duts": duts, "recipe": "td", "period_ns": PERIOD_NS, "abc_D_ps": sweep.TD_ABC_D_PS, "recipe_spec": "docs/spec/synth_recipe.md",
           "rtl_sha256": {d: [sha256(Path(x) if Path(x).is_absolute() else REPO_ROOT / x) for x in SPEC[d]["src"]] for d in duts},
           "note": "act_quant_nv is synthesised from a text-substituted copy (unpacked localparam array -> function); "
                   "tile buffer and HIER reducer are [EST] measurement models; int5 conversion is a verbatim copy of bs_column_top g_xrow"}
    with start_run("e28_quant_synth", cfg, paper=paper) as run:
        rows = []
        for d in duts:
            row, net = build_dut(d, BUILD / "synth")
            rows.append(row)
            for suf in ("stat.txt", "final_stat.txt", "sta.txt", "sta_r2r.txt"):
                p = BUILD / "synth" / f"{d}_{suf}"
                if p.exists():
                    shutil.copy(p, run.path(p.name))
            shutil.copy(BUILD / "synth" / f"{d}.ys", run.path(f"{d}.ys"))
            run.save_csv("synth.csv", rows)
            print(f"{d:8s} area {row['logic_area_um2']:9.0f} um2 cells {row['cells']:6d} seq {row['seq_cells']:5d} "
                  f"crit {row.get('crit_arrival_ns')} ns  WNS@25 {row['wns_ns']}", flush=True)
        run.log_metrics({f"{r['dut']}_{k}": r[k] for r in rows for k in ("logic_area_um2", "cells", "seq_cells", "wns_ns")})


# ---------------------------------------------------------------- power
def phase_power(paper: bool, dut: str, windows: list[str]) -> None:
    s = SPEC[dut]
    syn = {r["dut"]: r for r in __import__("csv").DictReader((REPO_ROOT / "results/e28_quant_synth/latest/synth.csv").open())}
    net = BUILD / "synth" / f"{dut}_netlist.v"
    sdc = HERE / "constraints" / ("seq.sdc" if s["clocked"] else "comb.sdc")
    cfg = {"dut": dut, "windows": windows, "n_tok": N_TOK, "tile_blocks": TILE, "clk_ps": P_PS, "netlist": str(net.relative_to(REPO_ROOT)),
           "netlist_sha256": sha256(net), "synth_run": str((REPO_ROOT / "results/e28_quant_synth/latest").resolve().name),
           "flow": "Icarus zero-delay GLS (own testbench, bit-exact vs numpy golden), VCD sliced + rebased (B9), docker OpenSTA report_power; "
                   "power x window duration / ops (leakage included at 25 ns per op)"}
    with start_run(f"e28_quant_power_{dut}", cfg, paper=paper) as run:
        rows = []
        for w in windows:
            vecs = window_vectors(dut, w, {})
            wd = BUILD / "power" / dut / w
            t0 = time.time()
            g = gls(dut, net, w, wd, vecs)
            if g["mismatches"]:
                raise RuntimeError(f"{dut}/{w}: GLS mismatches {g['mismatches']} of {g['n']}")
            sliced = wd / "window.vcd"
            slice_vcd(g["vcd"], sliced, g["t_start_ns"], g["t_end_ns"], rebase=True)
            g["vcd"].unlink()
            env = {"NETLIST": net, "TOP": s["mod"], "LIB": sweep.LIB, "SDC": sdc, "CLK_PERIOD_NS": P_PS / 1e3, "VCD": sliced,
                   "SCOPE": "tb/dut", "OUT": wd / "power.rpt"}
            for attempt in range(2):
                (wd / "power.rpt").unlink(missing_ok=True)
                r = e30.docker(["sta", "-no_splash", "-exit", str(HERE / "power.tcl")], env, os.environ.get("E28_STA_MEM", "3g"))
                if r.returncode == 0 and (wd / "power.rpt").exists():
                    break
                print(f"STA attempt {attempt} rc={r.returncode}: {r.stdout[-400:]}{r.stderr[-400:]}", flush=True)
            else:
                raise RuntimeError("STA power failed")
            sliced.unlink()
            pw = e13.parse_power((wd / "power.rpt").read_text())
            grp = _groups((wd / "power.rpt").read_text())
            dur = (g["t_end_ns"] - g["t_start_ns"]) * 1e-9
            k = dur / g["n"] * 1e15  # fJ per cycle-op = P(W) * window / n
            row = {"dut": dut, "window": w, "kind": w[:4], "n": g["n"], "mismatches": 0, "window_ns": dur * 1e9, **pw,
                   "e_fj_per_item": pw["p_total_w"] * k, "e_leak_fj_per_item": pw["p_leakage_w"] * k,
                   "e_seq_fj_per_item": grp["p_sequential_w"] * k, "e_comb_fj_per_item": grp["p_combinational_w"] * k,
                   "e_clock_fj_per_item": grp["p_clock_w"] * k, "wall_s": time.time() - t0}
            rows.append(row)
            run.save_csv("power.csv", rows)
            shutil.copy(wd / "power.rpt", run.path(f"power_{w}.rpt"))
            shutil.copy(wd / "sim.log", run.path(f"gls_{w}.log"))
            print(f"{dut} {w:6s} n={g['n']} E/item {row['e_fj_per_item']:9.2f} fJ (leak {row['e_leak_fj_per_item']:.2f}) {row['wall_s']:.0f}s",
                  flush=True)
        real = [r["e_fj_per_item"] for r in rows if r["kind"] == "real"]
        rnd = [r["e_fj_per_item"] for r in rows if r["kind"] == "rand"]
        m = {"dut": dut, "item": "op (32-element block / 32 row conversions / reduction / cycle)"}
        if real:
            lo, hi = e13.bootstrap_ci(np.array(real), np.random.default_rng(0))
            m.update(e_real_mean_fj=float(np.mean(real)), e_real_ci_lo=lo, e_real_ci_hi=hi, e_real_min=min(real), e_real_max=max(real),
                     e_leak_real_mean_fj=float(np.mean([r["e_leak_fj_per_item"] for r in rows if r["kind"] == "real"])))
        if rnd:
            m["e_rand_mean_fj"] = float(np.mean(rnd))
        m["gls_mismatches_total"] = 0
        m["gls_items_total"] = int(sum(r["n"] for r in rows))
        run.log_metrics(m)
        print(json.dumps(m, indent=1))


def _groups(txt: str) -> dict:
    out = {}
    for g, key in (("Sequential", "p_sequential_w"), ("Combinational", "p_combinational_w"), ("Clock", "p_clock_w")):
        m = re.search(rf"^{g}\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt, re.M)
        out[key] = float(m.group(4)) if m else 0.0
    return out


# ---------------------------------------------------------------- summary
def _latest(exp: str) -> Path:
    return (REPO_ROOT / "results" / exp / "latest").resolve()


def phase_summary(paper: bool) -> None:
    import csv

    from transformers import AutoConfig
    import yaml

    def runs(d):  # all completed run dirs of the block (NV is split into two jobs of 5 windows)
        base = REPO_ROOT / "results" / f"e28_quant_power_{d}"
        # a run that crashed after some windows (STA docker OOM on NV real3) still has verified rows in power.csv
        return sorted(x for x in base.iterdir() if x.is_dir() and not x.is_symlink() and (x / "power.csv").exists())  # skip the `latest` symlink (double counts)

    pw_rows = {d: [r for x in runs(d) for r in csv.DictReader((x / "power.csv").open())] for d in DUTS}
    p = {}
    for d in DUTS:
        real = [float(r["e_fj_per_item"]) for r in pw_rows[d] if r["kind"] == "real"]
        rnd = [float(r["e_fj_per_item"]) for r in pw_rows[d] if r["kind"] == "rand"]
        lo, hi = e13.bootstrap_ci(np.array(real), np.random.default_rng(0))
        p[d] = {"e_real_mean_fj": float(np.mean(real)), "e_real_ci_lo": lo, "e_real_ci_hi": hi, "e_real_min": min(real),
                "e_real_max": max(real), "e_rand_mean_fj": float(np.mean(rnd)), "n_real_windows": len(real),
                "e_leak_real_mean_fj": float(np.mean([float(r["e_leak_fj_per_item"]) for r in pw_rows[d] if r["kind"] == "real"])),
                "gls_items_total": int(sum(int(r["n"]) for r in pw_rows[d])),
                "gls_mismatches_total": int(sum(int(r["mismatches"]) for r in pw_rows[d])),
                "runs": [x.name for x in runs(d)]}
    syn = {}
    for x in sorted((REPO_ROOT / "results/e28_quant_synth").iterdir()):  # synth runs in time order; later rows win
        if x.is_dir() and not x.is_symlink() and (x / "metrics.json").exists():
            syn.update({r["dut"]: r for r in csv.DictReader((x / "synth.csv").open())})
    e22_dir = REPO_ROOT / "results/e22_macro_compare/20261114T185702027770Z_be5d1aa"  # T2.7 macro numbers
    e22 = json.loads((e22_dir / "metrics.json").read_text())
    macro = {"v2_MX": e22["v2_macro_mx_primary"]["e_total_fj_per_mac"], "v2_NV": e22["v2_macro_nv_primary"]["e_total_fj_per_mac"],
             "INT5_MX": e22["bs_macro_mx_primary"]["e_total_fj_per_mac"], "DPU_MX": e22["dpu_macro_mx_primary"]["e_total_fj_per_mac"],
             "DPU_NV": e22["dpu_macro_nv_primary"]["e_total_fj_per_mac"]}
    e = {d: p[d]["e_real_mean_fj"] for d in DUTS}  # fJ per item (op / block / reduction), real-window mean
    per_op = {  # fJ per macro op (32-element activation block feeding 32 rows x 32 columns = 1024 MACs)
        "quant_MX": e["mx"], "quant_NV": 2 * e["nv"], "anchor": e["anchor"], "tilebuf": e["tilebuf_cg"], "tilebuf_ungated": e["tilebuf"], "int5_conv": e["int5"]}
    mac_per_red = TILE * 32  # MACs of one column over one tile (tile_blocks x 32 rows)
    per_mac = {k: v / MAC_PER_OP for k, v in per_op.items()}
    per_mac["hier"] = e["hier"] / mac_per_red
    per_mac["hier_tile16"] = e["hier"] / (16 * 32)  # [DERIVED] tile_blocks 16: fewer reductions per MAC
    rows = []
    for key, base in macro.items():
        mode = key.split("_")[1]
        q = per_mac[f"quant_{mode}"]
        add = q + per_mac["anchor"] + per_mac["tilebuf"] + per_mac["hier"]
        add_ungated = q + per_mac["anchor"] + per_mac["tilebuf_ungated"] + per_mac["hier"]
        rows.append({"design_mode": key, "macro_fj_per_mac": base, "quant": q, "anchor": per_mac["anchor"], "tilebuf": per_mac["tilebuf"],
                     "hier": per_mac["hier"], "bundle_fj_per_mac": add, "total_fj_per_mac": base + add,
                     "share_of_total_pct": 100 * add / (base + add), "increase_pct": 100 * add / base,
                     "bundle_ungated_buf_fj_per_mac": add_ungated, "bundle_glitch_x1p97_est_fj_per_mac": 1.97 * add})
    ratio = {"v2_over_int5_macro": macro["INT5_MX"] / macro["v2_MX"],
             "v2_over_int5_with_bundle": (macro["INT5_MX"] + rows[2]["bundle_fj_per_mac"]) / (macro["v2_MX"] + rows[0]["bundle_fj_per_mac"]),
             "v2_over_dpu_macro_hier": macro["DPU_MX"] / macro["v2_MX"],
             "v2_over_dpu_with_bundle": (macro["DPU_MX"] + rows[3]["bundle_fj_per_mac"]) / (macro["v2_MX"] + rows[0]["bundle_fj_per_mac"])}
    models = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())
    tokrows = []
    for mname in ("smollm2-360m", "qwen2.5-0.5b"):
        snap = next((Path.home() / ".cache/huggingface/hub" / ("models--" + models[mname]["hf_id"].replace("/", "--")) / "snapshots").iterdir())
        c = AutoConfig.from_pretrained(str(snap))
        H, I, L, nh, nkv = c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        hd = getattr(c, "head_dim", None) or H // nh
        macs = L * (H * nh * hd + 2 * H * nkv * hd + nh * hd * H + 2 * H * I + I * H)
        blocks = L * (3 * (H // 32) + I // 32)  # qkv-in, o-in, gate/up-in (width H) + down-in (width I)
        reds = macs / mac_per_red
        for mode in ("MX", "NV"):
            e_q = blocks * per_op[f"quant_{mode}"]
            e_anchor = blocks * (per_op["anchor"] + per_op["tilebuf"])
            e_hier = reds * e["hier"]
            e_macro = macs * macro[f"v2_{mode}"]
            tokrows.append({"model": mname, "mode": mode, "macs_per_token": macs, "act_blocks_per_token": blocks,
                            "quant_nj": e_q / 1e6, "anchor_buf_nj": e_anchor / 1e6, "hier_nj": e_hier / 1e6,
                            "bundle_nj": (e_q + e_anchor + e_hier) / 1e6, "v2_macro_uj": e_macro / 1e9,
                            "bundle_share_of_v2_pct": 100 * (e_q + e_anchor + e_hier) / (e_q + e_anchor + e_hier + e_macro)})
    cfg = {"inputs": {d: p[d]["runs"] for d in DUTS} | {"synth": [x.name for x in sorted((REPO_ROOT / "results/e28_quant_synth").iterdir()) if x.is_dir() and not x.is_symlink() and (x / "metrics.json").exists()],
                                                                                  "e22_macro": e22_dir.name},
           "macs_per_op": MAC_PER_OP, "tile_blocks": TILE, "n_tok": N_TOK}
    with start_run("e28_quant_summary", cfg, paper=paper) as run:
        run.save_csv("per_mac.csv", rows)
        run.save_csv("per_token.csv", tokrows)
        sr = [{"dut": d, **{k: syn[d][k] for k in ("logic_area_um2", "cells", "seq_cells", "wns_ns", "crit_arrival_ns")},
               **{k: v for k, v in p[d].items() if k.startswith("e_") or k.startswith("gls")}} for d in DUTS]
        run.save_csv("per_block.csv", sr)
        m = {"per_item_fj_real_mean": e, "per_op_fj": per_op, "per_mac_fj": per_mac, "macro_fj_per_mac": macro, "ratios": ratio,
             "area_um2": {d: float(syn[d]["logic_area_um2"]) for d in DUTS}}
        run.log_metrics(m)
        print(json.dumps(m, indent=1))
        for r in rows:
            print(r)
        for r in tokrows:
            print(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--duts", default=",".join(DUTS))
    ap.add_argument("--power", choices=DUTS)
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    if a.capture:
        phase_capture(a.paper)
    if a.synth:
        phase_synth(a.paper, a.duts.split(","))
    if a.summary:
        phase_summary(a.paper)
    if a.power:
        phase_power(a.paper, a.power, a.windows.split(","))


if __name__ == "__main__":
    main()
