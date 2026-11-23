"""e13 SDF subset (WP-65): glitch-aware power of v2 vs bit-serial vs DPU on a subset of windows.

Same harness/program/VCD/OpenSTA flow as e13, but the netlists are simulated with sky130 cell timing
(specify blocks) back-annotated from OpenSTA SDF (cell delays; no wire parasitics before P&R), so
combinational glitches are present in the activity. Reports SDF and zero-delay energy per MAC on the
*same* windows and their ratio per design (deep bit-parallel trees are expected to glitch more, R-06).

Run: python experiments/e13_power/sdf_subset.py [--windows 2] [--reuse 128] [--sets-used 4] [--paper]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from cimfp4.llm.capture import load_capture
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import program_from_capture, save_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

sys.path.insert(0, str(Path(__file__).parent))
import run as e13  # noqa: E402

BUILD = REPO_ROOT / "build/sdf"


def clean_sdf(text: str) -> str:
    """Pre-P&R SDF for an ideal clock: drop INTERCONNECT lines and the CELL blocks of clock gates (their
    CLK->GCLK delay reflects an unbuffered gated-clock net) and of the SRAM macro (behavioural model,
    no specify blocks). Remaining: standard-cell IOPATH delays and timing checks."""
    out, block, depth = [], [], 0
    for line in text.splitlines(True):
        if "(INTERCONNECT" in line:
            continue
        if depth == 0 and line.strip().startswith("(CELL") and not line.strip().startswith("(CELLTYPE"):
            block, depth = [line], line.count("(") - line.count(")")
            continue
        if depth > 0:
            block.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                body = "".join(block)
                if "sky130_fd_sc_hd__dlclkp" not in body and "sky130_sram" not in body:
                    out.append(body)
                depth = 0
            continue
        out.append(line)
    text = "".join(out)
    # Icarus picks the typ value of each triplet: fill empty typ fields (a::b -> a:a:b), incl. the header
    text = re.sub(r"(-?[\d.]+)::(-?[\d.]+)", r"\1:\1:\2", text)
    # drop the empty top-level CELL (no delays) that Icarus rejects
    return re.sub(
        r"\(CELL\s*\(CELLTYPE \"[^\"]+\"\)\s*\(INSTANCE\)\s*\(DELAY\s*\(ABSOLUTE\s*\)\s*\)\s*\)\s*", "", text
    )


def write_sdf(design: str, top: str, clk_ns: float) -> Path:
    h = str(Path.home())
    out = BUILD / f"{design}.sdf"
    env = {
        "NETLIST": e13.NETDIR / f"{design}_netlist.v",
        "TOP": top,
        "LIB": e13.sweep.LIB,
        "SRAM_LIB": e13.sweep.SRAM_DIR / "lib" / (e13.sweep.SRAM + "_TT_1p8V_25C.lib"),
        "SDC": REPO_ROOT / "flows/synth/constraints/column.sdc",
        "CLK_PERIOD_NS": clk_ns,
        "OUT": out,
    }
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
        "-v",
        f"{h}/.ciel:{h}/.ciel",
        "-w",
        str(REPO_ROOT),
    ]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    subprocess.run(
        cmd + [e13.IMG, "sta", "-no_splash", "-exit", "flows/power/write_sdf.tcl"],
        check=True,
        capture_output=True,
    )
    # pre-P&R: ideal clock and unknown wires -> keep cell delays only (drop INTERCONNECT, incl. the
    # unbuffered-clock-net delay that would inject artificial skew)
    clean = BUILD / f"{design}_cells.sdf"
    clean.write_text(clean_sdf(out.read_text()))
    return clean


def gls(design: str, top: str, program: Path, clk_ps: int, wd: Path, sdf: Path | None) -> dict:
    from cocotb_tools.runner import get_results, get_runner

    vcd, win = wd / "dump.vcd", wd / "window.json"
    srcs = [
        e13.CELLS / "primitives.v",
        e13.CELLS / "sky130_fd_sc_hd.v",
        e13.NETDIR / f"{design}_netlist.v",
        e13.SRAM_GLS,
        REPO_ROOT / "flows/power/vcd_dump.v",
    ]
    defines = {"VCD_FILE": f'"{vcd}"', "VCD_TOP": top}
    build_args = ["-s", "vcd_dump"]
    if sdf is None:
        defines |= {"FUNCTIONAL": 1, "UNIT_DELAY": "#0"}
    else:
        srcs.append(REPO_ROOT / "flows/power/sdf_annotate.v")
        defines |= {"SDF_FILE": f'"{sdf}"', "SDF_TOP": top}
        build_args += ["-s", "sdf_annotate", "-gspecify"]
    runner = get_runner("icarus")
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if a == "-none" else a for a in c] for c in orig()]
    runner.build(
        sources=srcs,
        hdl_toplevel=top,
        build_dir=wd / "b",
        always=True,
        defines=defines,
        build_args=build_args,
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_column_top",
        test_dir=REPO_ROOT / "tb/cocotb",
        build_dir=wd / "b",
        results_xml=str(wd / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tb/cocotb'}",
            "CT_PROGRAM": str(program),
            "CT_WINDOW_OUT": str(win),
            "CT_M": "16",
            "CT_NOPS": "0",
            "CT_CLK_PS": str(clk_ps),
        },
    )
    if get_results(xml) != (1, 0):
        raise RuntimeError(f"{'SDF ' if sdf else ''}GLS mismatch: {design} {program.name}")
    return {**json.loads(win.read_text()), "vcd": vcd}


def energy(design, top, g, clk_ps, wd, prog, e_access_pj, rebase=False) -> float:
    sliced = wd / "window.vcd"
    slice_vcd(g["vcd"], sliced, g["t_start_ns"], g["t_end_ns"], rebase=rebase)
    g["vcd"].unlink()
    pw = e13.sta_power(design, top, sliced, clk_ps / 1e3, wd / "power.rpt")
    sliced.unlink()
    dur = (g["t_end_ns"] - g["t_start_ns"]) * 1e-9
    macs = g["n_ops"] * e13.MACS_PER_OP
    return (pw["p_total_w"] - pw["p_sram_sta_w"]) * dur / macs * 1e15 + e13.sram_accesses(
        prog, design
    ) * e_access_pj * 1e3 / macs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--reuse", type=int, default=128)
    ap.add_argument("--sets-used", type=int, default=4)
    ap.add_argument("--designs", default="v2_sram_m16,bs_int5,dpu")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--rebase", action="store_true", help="rebase VCD windows to t=0")
    ap.add_argument("--exp", default="e13_sdf")
    ap.add_argument("--netlist-dir", default="build/synth", help="<design>_netlist.v directory (repo-relative)")
    a = ap.parse_args()
    global BUILD
    e13.NETDIR = REPO_ROOT / a.netlist_dir
    if a.netlist_dir != "build/synth":  # keep the SDF/programs/work dirs of different netlist sets apart
        BUILD = BUILD / a.exp
    rng = seed_all(1)
    import yaml

    f_g1 = json.loads((REPO_ROOT / "results/e12_synth/latest/metrics.json").read_text())["f_g1_mhz"]
    clk_ps = int(round(1e6 / f_g1))
    e_acc = yaml.safe_load((REPO_ROOT / "flows/pdk/sram_macros.yaml").read_text())["macros"][e13.sweep.SRAM][
        "read_energy_pj_per_word_tt"
    ]
    synth = {r["design"]: r for r in json.loads((REPO_ROOT / "build/synth/synth.json").read_text())}
    with start_run(
        a.exp, {**vars(a), "f_g1_mhz": f_g1, "sdf": "OpenSTA cell delays, no parasitics"}, paper=a.paper
    ) as run:
        caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
        rows = []
        for w in range(a.windows):
            qw, qa = load_capture(caps[int(rng.integers(0, len(caps)))], "mxfp4")
            prog = program_from_capture(qw, qa, rng, sets_used=a.sets_used, reuse=a.reuse)
            ppath = save_program(BUILD / "programs" / f"sdf{w}.npz", prog)
            for d in a.designs.split(","):
                top = synth[d]["top"]
                sdf = write_sdf(d, top, clk_ps / 1e3)
                res = {}
                for mode, s in (("zero", None), ("sdf", sdf)):
                    wd = BUILD / d / f"sdf{w}_{mode}"
                    wd.mkdir(parents=True, exist_ok=True)
                    res[mode] = energy(d, top, gls(d, top, ppath, clk_ps, wd, s), clk_ps, wd, prog, e_acc, a.rebase)
                rows.append(
                    {
                        "design": d,
                        "window": f"sdf{w}",
                        "e_zero_fj_per_mac": res["zero"],
                        "e_sdf_fj_per_mac": res["sdf"],
                        "sdf_over_zero": res["sdf"] / res["zero"],
                    }
                )
                print(
                    f"{d:12s} sdf{w}: zero {res['zero']:8.2f}  sdf {res['sdf']:8.2f} fJ/MAC  ratio {res['sdf'] / res['zero']:.3f}"
                )
        run.save_csv("sdf_ratio.csv", rows)
        m = {}
        for d in a.designs.split(","):
            m[f"{d}_sdf_over_zero"] = float(np.mean([r["sdf_over_zero"] for r in rows if r["design"] == d]))
        v = m["v2_sram_m16_sdf_over_zero"]
        for d in a.designs.split(","):
            if d != "v2_sram_m16":
                m[f"glitch_correction_ratio_{d}_vs_v2"] = m[f"{d}_sdf_over_zero"] / v
        run.log_metrics(m)
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
