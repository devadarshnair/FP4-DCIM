"""e21 - macro-level zero-delay gate-level power of the full 32-column macros (macro-level power).

Netlists: the e12_macro_synth outputs (build/synth_macro_{v2,bs}/ and build/synth_macro_dpu_hier/*_netlist.v; same Yosys/abc sky130 flow,
results/e12_macro_synth_*/ for provenance). Per design x program:
  1. Icarus GLS of the netlist (sky130_fd_sc_hd functional models, OpenRAM behavioural model) driven by
     tb/cocotb/test_macro_power.py, bit-exact checked against the per-column reference models; VCD of
     every net (flows/power/vcd_dump.v).
  2. VCD sliced to the compute window (weight loading excluded; set switches included), OpenSTA
     read_vcd + report_power (native OpenSTA 3.1.0; sky130_fd_sc_hd tt_025C_1v80, ideal clock: no clock
     tree). The window is rebased to t = 0 (OpenSTA averages toggles from time 0).
  3. Logic energy = (STA total - OpenRAM instance power) x window. The OpenRAM Liberty power is a
     placeholder and is excluded; SRAM energy is composed afterwards from e20 x access/cycle counts
     (experiments/e21_macro_power/compare.py), so the SRAM range can be swept without re-simulating.

Workload (same operating point as G1/e13): `sets_used` weight sets, each reused by `reuse` activation
vectors (real RTN captures, Qwen2.5-0.5B / SmolLM2-360M prefill) or a 50 %-random control. MXFP4 runs
on both macros with the identical program; NVFP4 only on v2 (the bit-serial INT5 baseline is MX-only).

Run (heavy; one design at a time is enforced by running designs sequentially in one process):
  python scripts/safe_run.py --max-rss-mb 5000 -- python experiments/e21_macro_power/run.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.power.boundary import broadcast_toggles, weight_move_toggles
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import load_program, macro_program_from_capture, macro_program_random, save_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402

STA_BIN = os.environ.get("CIM_STA_BIN", "sta")  # override with CIM_STA_BIN
CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
SRAM_GLS = REPO_ROOT / "build/models/sky130_sram_1kbyte_1rw1r_32x256_8_gls.v"
BUILD = REPO_ROOT / "build/e21_macro_power"
DESIGNS = {  # name -> (netlist, top, fmt)
    "v2_macro_mx": ("build/synth_macro_v2/v2_macro_netlist.v", "macro_top", "mxfp4"),
    "bs_macro_mx": ("build/synth_macro_bs/bs_macro_netlist.v", "bs_macro_top", "mxfp4"),
    "v2_macro_nv": ("build/synth_macro_v2/v2_macro_netlist.v", "macro_top", "nvfp4"),
    # hierarchical-Yosys netlists (flat dpu_macro exceeds the 7.5 GB machine; see e12 --hier). v2hier = calibration
    # of the hierarchical-vs-flat flow on v2 (same program windows).
    "v2hier_macro_mx": ("build/synth_macro_v2_hier/v2_macro_netlist.v", "macro_top", "mxfp4"),
    "dpu_macro_mx": ("build/synth_macro_dpu_hier/dpu_macro_netlist.v", "dpu_macro", "mxfp4"),
    "dpu_macro_nv": ("build/synth_macro_dpu_hier/dpu_macro_netlist.v", "dpu_macro", "nvfp4"),
}
# T2.7: the T2.6 timing-driven (`td`) netlists (experiments/e30_td_synth, docs/spec/synth_recipe.md): v2/INT5 flat Yosys,
# DPU hierarchical Yosys (OpenROAD writes it flat), v2hier = the hierarchical-vs-flat calibration of the same flow.
TD = "build/e30_td_synth"
NETLIST_SETS = {
    "e12": DESIGNS,
    "td": {
        "v2_macro_mx": (f"{TD}/macro_td/v2_macro_netlist.v", "macro_top", "mxfp4"),
        "bs_macro_mx": (f"{TD}/macro_td/bs_macro_netlist.v", "bs_macro_top", "mxfp4"),
        "v2_macro_nv": (f"{TD}/macro_td/v2_macro_netlist.v", "macro_top", "nvfp4"),
        "v2hier_macro_mx": (f"{TD}/macro_td_hier/v2_macro_netlist.v", "macro_top", "mxfp4"),
        "dpu_macro_mx": (f"{TD}/macro_td_hier/dpu_macro_netlist.v", "dpu_macro", "mxfp4"),
        "dpu_macro_nv": (f"{TD}/macro_td_hier/dpu_macro_netlist.v", "dpu_macro", "nvfp4"),
        # T2.11: flat-Yosys td DPU netlist (e30 --macro dpu_macro --flat), same programs/windows
        "dpuflat_macro_mx": (f"{TD}/macro_td/dpu_macro_netlist.v", "dpu_macro", "mxfp4"),
        "dpuflat_macro_nv": (f"{TD}/macro_td/dpu_macro_netlist.v", "dpu_macro", "nvfp4"),
        # T5.4: D2 INT4-SM bit-serial macro (LOSSY surrogate; e30 --macro bs4_macro). fmt int4sm = MX-style scaling with
        # {sign, mag[2:0]} codes; its programs come from --int4sm-programs (experiments/e21_macro_power/int4sm_programs.py)
        "bs4_macro_mx": (f"{TD}/macro_td/bs4_macro_netlist.v", "bs4_macro_top", "int4sm"),
        # T4.2: NV_GATE=1 netlists (e30 --macro ... --nv-gate 1, results/e30_td_macro_*_nvg1), same programs/windows as
        # the as-built rows above; fmt mixed = one MX+NV interleaved stream (writers.macro_program_mixed, --mixed-chunk)
        "v2nvg1_macro_mx": (f"{TD}/macro_td_nvg1/v2_macro_netlist.v", "macro_top", "mxfp4"),
        "v2nvg1_macro_nv": (f"{TD}/macro_td_nvg1/v2_macro_netlist.v", "macro_top", "nvfp4"),
        "v2hiernvg1_macro_mx": (f"{TD}/macro_td_hier_nvg1/v2_macro_netlist.v", "macro_top", "mxfp4"),
        "v2hiernvg1_macro_nv": (f"{TD}/macro_td_hier_nvg1/v2_macro_netlist.v", "macro_top", "nvfp4"),
        "dpunvg1_macro_mx": (f"{TD}/macro_td_hier_nvg1/dpu_macro_netlist.v", "dpu_macro", "mxfp4"),
        "dpunvg1_macro_nv": (f"{TD}/macro_td_hier_nvg1/dpu_macro_netlist.v", "dpu_macro", "nvfp4"),
        "dpuflatnvg1_macro_mx": (f"{TD}/macro_td_nvg1/dpu_macro_netlist.v", "dpu_macro", "mxfp4"),
        "dpuflatnvg1_macro_nv": (f"{TD}/macro_td_nvg1/dpu_macro_netlist.v", "dpu_macro", "nvfp4"),
        "v2_macro_mixed": (f"{TD}/macro_td/v2_macro_netlist.v", "macro_top", "mixed"),
        "dpu_macro_mixed": (f"{TD}/macro_td_hier/dpu_macro_netlist.v", "dpu_macro", "mixed"),
        "dpuflat_macro_mixed": (f"{TD}/macro_td/dpu_macro_netlist.v", "dpu_macro", "mixed"),
        "v2nvg1_macro_mixed": (f"{TD}/macro_td_nvg1/v2_macro_netlist.v", "macro_top", "mixed"),
        "dpunvg1_macro_mixed": (f"{TD}/macro_td_hier_nvg1/dpu_macro_netlist.v", "dpu_macro", "mixed"),
        "dpuflatnvg1_macro_mixed": (f"{TD}/macro_td_nvg1/dpu_macro_netlist.v", "dpu_macro", "mixed"),
    },
}
NETLIST_PROV = {"e12": "results/e12_macro_synth_{v2_macro,bs_macro,dpu_macro_hier}/latest (same build/ outputs)",
                "td": "results/e30_td_macro_{v2_macro,bs_macro,bs4_macro,v2_macro_hier,dpu_macro_hier,dpu_macro}/latest (build/e30_td_synth)"}
GPTQ_DIR = REPO_ROOT / "results/gptq_archive/gptq"
GPTQ_MODULE = ("qwen2.5-0.5b", "model.layers.0.self_attn.q_proj")  # layer-0 q_proj: its input (RMSNorm of the embeddings)
# does not depend on any quantized weight, so the RTN-model activation capture is also the GPTQ-model activation
MACS_PER_OP = 32 * 32  # 32 rows x 32 columns
NG = 8  # OpenRAM macros per design


def gls(netlist: Path, top: str, fmt: str, program: Path, clk_ps: int, wd: Path) -> dict:
    from cocotb_tools.runner import get_results, get_runner

    vcd, win = wd / "dump.vcd", wd / "window.json"
    srcs = [CELLS / "primitives.v", CELLS / "sky130_fd_sc_hd.v", netlist, SRAM_GLS,
            REPO_ROOT / "flows/power/vcd_dump.v"]
    runner = get_runner("icarus")
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if a == "-none" else a for a in c] for c in orig()]
    runner.build(sources=srcs, hdl_toplevel=top, build_dir=wd / "b", always=True,
                 defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0", "VCD_FILE": f'"{vcd}"', "VCD_TOP": top},
                 build_args=["-s", "vcd_dump"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel=top, test_module="test_macro_power", test_dir=REPO_ROOT / "tb/cocotb",
                      build_dir=wd / "b", results_xml=str(wd / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tb/cocotb'}",
                                 "MP_PROGRAM": str(program), "MP_FMT": fmt, "MP_CLK_PS": str(clk_ps),
                                 "MP_WINDOW_OUT": str(win)})
    if get_results(xml) != (1, 0):
        raise RuntimeError(f"GLS mismatch for {netlist.name} on {program.name}")
    return {**json.loads(win.read_text()), "vcd": vcd}


def sta_power(netlist: Path, top: str, vcd: Path, clk_ns: float, out: Path) -> dict:
    """Native OpenSTA (STA_BIN, 3.1.0): ~0.85 GB for the 263k-cell v2 netlist, against > 3 GB of container
    memory for the docker 2.6.0 image (outside safe_run's RSS accounting). Same sta_power.tcl as e13."""
    env = {**os.environ, "NETLIST": str(netlist), "TOP": top, "LIB": str(sweep.LIB),
           "SRAM_LIB": str(sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib")),
           "SDC": str(REPO_ROOT / "flows/synth/constraints/column.sdc"), "CLK_PERIOD_NS": str(clk_ns),
           "VCD": str(vcd), "SCOPE": top, "OUT": str(out)}
    r = subprocess.run([STA_BIN, "-no_splash", "-exit", str(REPO_ROOT / "flows/power/sta_power.tcl")],
                       capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    (out.parent / "sta.log").write_text(r.stdout + r.stderr)
    if r.returncode or not out.exists():
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return parse_power(out.read_text())


def parse_power(txt: str) -> dict:
    """OpenSTA report_power -> group rows (W) + totals + summed OpenRAM instance power."""
    res = {}
    for grp in ("Sequential", "Combinational", "Clock", "Macro", "Pad", "Total"):
        m = re.search(rf"^{grp}\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt, re.M)
        if m:
            i, s, lk, t = (float(v) for v in m.groups())
            res[f"p_{grp.lower()}_int_w"], res[f"p_{grp.lower()}_sw_w"] = i, s
            res[f"p_{grp.lower()}_leak_w"], res[f"p_{grp.lower()}_w"] = lk, t
    sm = re.findall(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+\S*sram\S*\s*$", txt, re.M)
    res["p_sram_sta_w"] = sum(float(r[3]) for r in sm)
    res["p_sram_sta_leak_w"] = sum(float(r[2]) for r in sm)
    res["n_sram_inst"] = len(sm)
    return res


def gptq_capture(fmt: str) -> tuple:
    """(qw, qa, meta): weight of GPTQ_MODULE from the format's GPTQ checkpoint (e07 archive), re-quantized with
    the hardware quantizer (exact iff it picks the GPTQ block scales; the match fraction is reported); activations
    from the RTN capture of the same module."""
    import torch
    from safetensors import safe_open

    from cimfp4.formats.mx_quant import dequantize
    from cimfp4.llm.capture import load_capture, quantize_operand

    model, mod = GPTQ_MODULE
    ck = GPTQ_DIR / f"{model}__{fmt}_gptq"
    with safe_open(str(ck / "model.safetensors"), framework="pt") as f:
        wt = f.get_tensor(f"{mod}.weight")
    w = wt.to(torch.float64).numpy()
    cap = next((REPO_ROOT / "data/captures").glob(f"{model}@*/prefill/{mod}.npz"))
    _, qa = load_capture(cap, fmt)
    qw = quantize_operand(w, fmt)
    # the checkpoint stores the GPTQ-dequantized weights in its own dtype (bf16): compare in that dtype
    match = float((torch.from_numpy(dequantize(qw)).to(wt.dtype) == wt).float().mean())
    meta = {"checkpoint": str(ck.relative_to(REPO_ROOT)), "module": mod, "activations": str(cap.relative_to(REPO_ROOT)),
            "requant_exact_fraction": match, "checkpoint_dtype": str(wt.dtype), "gptq_meta": json.loads((ck / "gptq_meta.json").read_text())}
    return qw, qa, meta


def paired_programs(rng, fmts: list[str], per_model: int, gptq: bool, n_random: int, sets_used: int, reuse: int,
                    build: Path) -> tuple[dict, list]:
    """T2.12 windows: `per_model` real RTN-capture windows from EACH model (stratified), optionally one GPTQ-weight
    window, and `n_random` random controls. MX and NV programs of a real window use the same capture and the same
    rows/slices/tokens (one child seed per window; the selection draws depend only on the tensor shapes, which are
    equal for both formats), so MX/NV ratios are paired by window."""
    from cimfp4.llm.capture import load_capture  # imports torch (~0.45 GB RSS): only when programs are generated

    caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
    models = sorted({c.parts[-3] for c in caps})
    wins = []
    for model in models:
        mc = [c for c in caps if c.parts[-3] == model]
        for j in rng.choice(len(mc), size=per_model, replace=False):
            wins.append(mc[int(j)])
    progs = {f: [] for f in fmts}
    meta = []
    for w, cap in enumerate(wins):
        seed = int(rng.integers(0, 2**31 - 1))
        for fmt in fmts:
            qw, qa = load_capture(cap, fmt)
            p = macro_program_from_capture(qw, qa, np.random.default_rng(seed), m=16, sets_used=sets_used, reuse=reuse)
            progs[fmt].append((f"real{w}", save_program(build / "programs" / f"{fmt}_real{w}.npz", p),
                               str(cap.relative_to(REPO_ROOT)), p))
        meta.append({"window": f"real{w}", "capture": str(cap.relative_to(REPO_ROOT)), "seed": seed, "weights": "rtn"})
    if gptq:
        seed = int(rng.integers(0, 2**31 - 1))
        for fmt in fmts:
            qw, qa, gm = gptq_capture(fmt)
            p = macro_program_from_capture(qw, qa, np.random.default_rng(seed), m=16, sets_used=sets_used, reuse=reuse)
            name = f"real{len(wins)}"
            progs[fmt].append((name, save_program(build / "programs" / f"{fmt}_{name}.npz", p), "gptq:" + gm["checkpoint"], p))
            meta.append({"window": name, "seed": seed, "weights": f"gptq_{fmt}", **gm})
    for fmt in fmts:
        for w in range(n_random):
            p = macro_program_random(rng, fmt, m=16, sets_used=sets_used, reuse=reuse)
            progs[fmt].append((f"rand{w}", save_program(build / "programs" / f"{fmt}_rand{w}.npz", p), "random", p))
    return progs, meta


def programs_from(run_id: str, build: Path) -> tuple[dict, list]:
    """Reuse the saved programs of an earlier paired run (no torch import: keeps the job RSS for the large GLS)."""
    import shutil

    src = BUILD / run_id / "programs"
    meta = json.loads((REPO_ROOT / "results/e21_macro_power" / run_id / "windows.json").read_text())
    names = sorted({p.stem.split("_", 1)[1] for p in src.glob("*.npz")}, key=lambda n: (n[:4] != "real", int(n[4:])))
    srcs = {m["window"] + ("" if m["weights"] == "rtn" else "_" + m["weights"].split("_")[-1]):
            m.get("capture") if m["weights"] == "rtn" else "gptq:" + m["checkpoint"] for m in meta}
    progs = {}
    for fmt in ("mxfp4", "nvfp4"):
        progs[fmt] = []
        for n in names:
            f = src / f"{fmt}_{n}.npz"
            if not f.exists():
                continue
            dst = build / "programs" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
            s = "random" if n.startswith("rand") else srcs.get(n) or srcs[f"{n}_{fmt}"]
            progs[fmt].append((n, dst, s, load_program(dst)))
    return progs, meta


def mixed_programs(progs: dict, chunk: int, build: Path) -> list:
    """T4.2: one mixed MX/NV stream per window that has both a mxfp4 and an nvfp4 program (paired windows)."""
    from cimfp4.stimulus.writers import macro_program_mixed

    nv = {n: (s, p) for n, _, s, p in progs.get("nvfp4", [])}
    out = []
    for n, _, s, p in progs.get("mxfp4", []):
        if n in nv:
            q = macro_program_mixed(p, nv[n][1], chunk)
            out.append((n, save_program(build / "programs" / f"mixed_{n}.npz", q), s, q))
    if not out:
        raise SystemExit("fmt mixed needs paired mxfp4 + nvfp4 programs (--programs-from or --paired)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=2, help="real-capture programs per format")
    ap.add_argument("--random", type=int, default=1)
    ap.add_argument("--reuse", type=int, default=256)
    ap.add_argument("--sets-used", type=int, default=2)
    ap.add_argument("--designs", default="v2_macro_mx,bs_macro_mx,v2_macro_nv")
    ap.add_argument("--clk-ps", type=int, default=25000, help="f_G1 = 40 MHz (e12), as e13")
    ap.add_argument("--keep-vcd", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--netlists", choices=sorted(NETLIST_SETS), default="e12",
                    help="netlist set: e12 (pre-T2.6 unbuffered, default) | td (T2.6 timing-driven, T2.7)")
    ap.add_argument("--paired", action="store_true",
                    help="T2.12 windows: --windows real windows PER MODEL, MX/NV paired on the same windows")
    ap.add_argument("--gptq", action="store_true", help="with --paired: add one GPTQ-weight window (GPTQ_MODULE)")
    ap.add_argument("--only-windows", default="", help="comma list: simulate only these windows (programs unchanged)")
    ap.add_argument("--exp", default="e21_macro_power")
    ap.add_argument("--programs-from", default="", help="paired run id whose saved programs are reused (no regeneration)")
    ap.add_argument("--mixed-chunk", type=int, default=64,
                    help="T4.2 fmt mixed: ops per MX/NV chunk of the interleaved stream (from the paired mxfp4/nvfp4 programs)")
    ap.add_argument("--int4sm-programs", default="",
                    help="T5.4: dir of int4sm_<window>.npz macro programs (int4sm_programs.py) for the bs4_macro_mx design")
    args = ap.parse_args()
    rng = seed_all(0)
    designs = args.designs.split(",")
    DES = NETLIST_SETS[args.netlists]
    sta_ver = subprocess.run([STA_BIN, "-version"], capture_output=True, text=True).stdout.strip()
    cfg = {**vars(args), "corner": "sky130_fd_sc_hd tt_025C_1v80 (STA); ideal clock (no CTS); zero-delay GLS",
           "sta": f"OpenSTA {sta_ver} (native)", "vcd_window": "rebased to t=0 (B9)",
           "netlists": {d: DES[d][0] for d in designs},
           "netlist_provenance": NETLIST_PROV[args.netlists],
           "sram_policy": "OpenRAM Liberty instance power excluded; SRAM composed in compare.py from e20",
           "macs_per_op": MACS_PER_OP,
           "stimulus": "Qwen2.5-0.5B + SmolLM2-360M prefill RTN captures + 50%-random control"}
    caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
    with start_run(args.exp, cfg, paper=args.paper) as run:
        build = BUILD / run.run_id
        progs = {"mxfp4": [], "nvfp4": [], "int4sm": [], "mixed": []}
        if args.programs_from:
            progs, meta = programs_from(args.programs_from, build)
            (run.dir / "windows.json").write_text(json.dumps(meta, indent=1))
        elif args.paired:  # both formats are always generated so that every design sees the same windows
            progs, meta = paired_programs(rng, ["mxfp4", "nvfp4"], args.windows, args.gptq, args.random,
                                          args.sets_used, args.reuse, build)
            (run.dir / "windows.json").write_text(json.dumps(meta, indent=1))
        for fmt in progs:
            if fmt in ("int4sm", "mixed") or args.paired or args.programs_from or not any(DES[d][2] == fmt for d in designs):
                continue
            for w in range(args.windows):
                from cimfp4.llm.capture import load_capture

                cap = caps[int(rng.integers(0, len(caps)))]
                qw, qa = load_capture(cap, fmt)
                p = macro_program_from_capture(qw, qa, rng, m=16, sets_used=args.sets_used, reuse=args.reuse)
                progs[fmt].append((f"real{w}", save_program(build / "programs" / f"{fmt}_real{w}.npz", p),
                                   str(cap.relative_to(REPO_ROOT)), p))
            for w in range(args.random):
                p = macro_program_random(rng, fmt, m=16, sets_used=args.sets_used, reuse=args.reuse)
                progs[fmt].append((f"rand{w}", save_program(build / "programs" / f"{fmt}_rand{w}.npz", p),
                                   "random", p))
        progs["mixed"] = mixed_programs(progs, args.mixed_chunk, build) if any(DES[d][2] == "mixed" for d in designs) else []
        if args.int4sm_programs:  # T5.4: INT4-SM re-quantisation of the same windows (rows/slices/tokens)
            import shutil

            progs["int4sm"] = []
            src4 = Path(args.int4sm_programs)
            if not src4.is_absolute():
                src4 = REPO_ROOT / src4
            prov4 = json.loads((src4 / "int4sm_programs.json").read_text()) if (src4 / "int4sm_programs.json").exists() else {}
            for f in sorted(src4.glob("int4sm_*.npz"), key=lambda p: (p.stem[7:11] != "real", int(p.stem[11:]))):
                dst = build / "programs" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
                name = f.stem.split("_", 1)[1]
                progs["int4sm"].append((name, dst, prov4.get(name, {}).get("source", "int4sm"), load_program(dst)))
            (run.dir / "int4sm_programs.json").write_text(json.dumps({"dir": str(src4), **prov4}, indent=1))
        elif any(DES[d][2] == "int4sm" for d in designs):
            raise SystemExit("bs4_macro_mx needs --int4sm-programs")
        rows = []
        only = set(filter(None, args.only_windows.split(",")))
        for d in designs:
            net, top, fmt = DES[d]
            net = REPO_ROOT / net
            for name, ppath, src, prog in progs[fmt]:
                if only and name not in only:
                    continue
                wd = build / d / name
                wd.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                g = gls(net, top, fmt, ppath, args.clk_ps, wd)
                t_gls = time.time() - t0
                t1 = g["t_last_valid_ns"] + args.clk_ps / 1e3  # through the last result cycle
                sliced = wd / "window.vcd"
                st = slice_vcd(g["vcd"], sliced, g["t_start_ns"], t1, rebase=True)  # OpenSTA averages from t=0
                if not args.keep_vcd:
                    g["vcd"].unlink()
                t0 = time.time()
                pw = sta_power(net, top, sliced, args.clk_ps / 1e3, wd / "power.rpt")
                t_sta = time.time() - t0
                if not args.keep_vcd:
                    sliced.unlink()
                (run.dir / f"power_{d}_{name}.rpt").write_text((wd / "power.rpt").read_text())
                dur_s = (t1 - g["t_start_ns"]) * 1e-9
                cycles = round((t1 - g["t_start_ns"]) * 1e3 / args.clk_ps)
                p_logic = pw["p_total_w"] - pw["p_sram_sta_w"]
                macs = g["n_ops"] * MACS_PER_OP
                row = {"design": d, "fmt": fmt, "window": name, "kind": name[:4], "source": src,
                       "weights": "gptq" if src.startswith("gptq:") else ("random" if src == "random" else "rtn"),
                       "n_ops": g["n_ops"], "nv_ops": int(prog["op_nv"].sum()) if fmt == "mixed" else
                       (g["n_ops"] if fmt == "nvfp4" else 0), "macs": macs, "cycles": cycles, "window_ns": dur_s * 1e9,
                       "set_switches": g["set_switches"], "stall_cycles": g["stall_cycles"],
                       "sram_read_words": g["set_switches"] * NG * 16,
                       "sram_dual_read_cycles": g["set_switches"] * NG * 8,
                       "vcd_changes": st["changes_kept"], "gls_s": t_gls, "sta_s": t_sta,
                       **pw, "p_logic_w": p_logic,
                       "e_logic_pj_per_op": p_logic * dur_s / g["n_ops"] * 1e12,
                       "e_logic_fj_per_mac": p_logic * dur_s / macs * 1e15,
                       "e_leak_fj_per_mac": (pw["p_total_leak_w"] - pw["p_sram_sta_leak_w"]) * dur_s / macs * 1e15,
                       "cycles_per_op": cycles / g["n_ops"],
                       # macro-boundary wires (no wire caps in the synthesis netlist): toggles for compare.py
                       "bcast_toggles": broadcast_toggles(prog, "bs4_int4sm" if d.startswith("bs4") else
                                                      {"bs": "bs_int5", "dp": "dpu"}.get(d[:2], "v2_sram_m16")),
                       "wmove_toggles": weight_move_toggles(prog)}
                rows.append(row)
                run.save_csv("power.csv", rows)
                print(f"{d:12s} {name:6s} {row['e_logic_fj_per_mac']:8.2f} fJ/MAC logic, "
                      f"{row['cycles_per_op']:.3f} cyc/op, gls {t_gls:.0f}s sta {t_sta:.0f}s", flush=True)
        m = {}
        for d in designs:
            for kind in ("real", "rand"):
                v = [r["e_logic_fj_per_mac"] for r in rows if r["design"] == d and r["kind"] == kind]
                if v:
                    m[f"{d}_e_logic_fj_per_mac_{kind}"] = float(np.mean(v))
            m[f"{d}_cycles_per_op"] = float(np.mean([r["cycles_per_op"] for r in rows if r["design"] == d]))
        run.log_metrics(m)
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
