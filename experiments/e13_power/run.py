"""e13 - zero-delay gate-level power of the G1 designs on identical real-LLM stimulus (WP-28, G1).

Per design × stimulus window:
  1. Icarus GLS of the sky130 netlist (build/synth) driven by the shared program (stimulus/writers),
     bit-exact checked by the harness, VCD of every net (flows/power/vcd_dump.v).
  2. VCD sliced to the compute phase (weight loading excluded) -> OpenSTA read_vcd + report_power.
  3. Energy per MAC = logic power (all std cells: internal + switching + leakage) × window / MACs
     + SRAM accesses × E_access / MACs.
     SRAM: the OpenRAM Liberty charges a constant energy on *every* clock edge (placeholder
     characterization, also when deselected), which would bill designs by cycle count rather than by
     accesses. We therefore exclude the macro from the STA total and charge E_access per actual read
     or write access (Liberty value, rise+fall); disclosed in the paper. Bands: ×0.5 / ×1 / ×2.
Clock: f_G1 (e12) for every design. CI: 95 % bootstrap over windows (1000 resamples).

Run: python experiments/e13_power/run.py [--windows 8] [--random 2] [--designs ...] [--paper]
     [--netlist-dir build/e30_td_synth/td --rebase --exp e13_power_td --designs v2_sram_m16,bs_int5,dpu]  (T2.7)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from cimfp4.llm.capture import load_capture
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import program_from_capture, program_random, save_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all

sys.path.insert(0, str(REPO_ROOT / "flows/synth"))
import sweep  # noqa: E402

IMG = "ghcr.io/efabless/openlane2:2.3.10"
CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
SRAM_GLS = REPO_ROOT / "build/models/sky130_sram_1kbyte_1rw1r_32x256_8_gls.v"
MACS_PER_OP = 4 * 32  # 4 columns x 32 rows per op, all designs
BUILD = REPO_ROOT / "build/power"
NETDIR = REPO_ROOT / "build/synth"  # --netlist-dir: e.g. build/e30_td_synth/td (T2.6 timing-driven netlists, T2.7)


def gls_with_vcd(design: str, top: str, program: Path, clk_ps: int, workdir: Path, m_sets: int = 16) -> dict:
    """Run the harness on the netlist with a VCD dump; return the compute window json + vcd path."""
    from cocotb_tools.runner import get_results, get_runner

    vcd = workdir / "dump.vcd"
    win = workdir / "window.json"
    srcs = [
        CELLS / "primitives.v",
        CELLS / "sky130_fd_sc_hd.v",
        NETDIR / f"{design}_netlist.v",
        SRAM_GLS,
        REPO_ROOT / "flows/power/vcd_dump.v",
    ]
    runner = get_runner("icarus")
    # cocotb's Icarus runner appends `-none` (no waveforms) to vvp; switch it to `-vcd` for our dump root
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if a == "-none" else a for a in c] for c in orig()]
    runner.build(
        sources=srcs,
        hdl_toplevel=top,
        build_dir=workdir / "b",
        always=True,
        defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0", "VCD_FILE": f'"{vcd}"', "VCD_TOP": top},
        build_args=["-s", "vcd_dump"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_column_top",
        test_dir=REPO_ROOT / "tb/cocotb",
        build_dir=workdir / "b",
        results_xml=str(workdir / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tb/cocotb'}",
            "CT_PROGRAM": str(program),
            "CT_WINDOW_OUT": str(win),
            "CT_M": str(m_sets),
            "CT_NOPS": "0",
            "CT_CLK_PS": str(clk_ps),
        },
    )
    if get_results(xml) != (1, 0):
        raise RuntimeError(f"GLS mismatch for {design} on {program.name}")
    return {**json.loads(win.read_text()), "vcd": vcd}


def sta_power(design: str, top: str, vcd: Path, clk_ns: float, out: Path) -> dict:
    h = str(Path.home())
    env = {
        "NETLIST": NETDIR / f"{design}_netlist.v",
        "TOP": top,
        "LIB": sweep.LIB,
        "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
        "SDC": REPO_ROOT / "flows/synth/constraints/column.sdc",
        "CLK_PERIOD_NS": clk_ns,
        "VCD": vcd,
        "SCOPE": top,
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
    r = subprocess.run(
        cmd + [IMG, "sta", "-no_splash", "-exit", "flows/power/sta_power.tcl"], capture_output=True, text=True
    )
    if r.returncode or not out.exists():
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return parse_power(out.read_text())


def parse_power(txt: str) -> dict:
    """OpenSTA report_power -> totals (W) and the SRAM instance total (W)."""
    m = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt, re.M)
    internal, switching, leakage, total = (float(v) for v in m.groups())
    sram = 0.0
    sm = re.findall(
        r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+\S*sram\S*\s*$", txt, re.M
    )
    if sm:
        sram = sum(float(row[3]) for row in sm)
    return {
        "p_internal_w": internal,
        "p_switching_w": switching,
        "p_leakage_w": leakage,
        "p_total_w": total,
        "p_sram_sta_w": sram,
    }


def sram_accesses(prog: dict, design: str) -> int:
    """Read words on set switches (16 per switch via both ports) - writes are excluded (load phase)."""
    switches = 1 + int(np.count_nonzero(np.diff(prog["set"]) != 0))
    return switches * 16


def bootstrap_ci(x: np.ndarray, rng, n: int = 1000) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    means = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--reuse", type=int, default=256)
    ap.add_argument("--sets-used", type=int, default=8)
    ap.add_argument("--designs", default="v2_sram_m16,bs_int5,dpu,v2_generic")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--m", type=int, default=16, help="weight sets of the designs (1 for *_m1)")
    ap.add_argument("--rebase", action="store_true", help="rebase VCD windows to t=0")
    ap.add_argument("--exp", default="e13_power", help="experiment name (variants: e13_power_r16, ...)")
    ap.add_argument("--netlist-dir", default="build/synth", help="<design>_netlist.v directory (repo-relative)")
    args = ap.parse_args()
    global NETDIR
    NETDIR = REPO_ROOT / args.netlist_dir
    rng = seed_all(0)
    build = BUILD / args.exp
    e12 = json.loads((REPO_ROOT / "results/e12_synth/latest/metrics.json").read_text())
    f_g1 = e12["f_g1_mhz"]
    clk_ps = int(round(1e6 / f_g1))
    inv = yaml.safe_load((REPO_ROOT / "flows/pdk/sram_macros.yaml").read_text())
    e_access_pj = inv["macros"][sweep.SRAM]["read_energy_pj_per_word_tt"]
    synth = {r["design"]: r for r in json.loads((REPO_ROOT / "build/synth/synth.json").read_text())}
    cfg = {
        **vars(args),
        "f_g1_mhz": f_g1,
        "clk_ps": clk_ps,
        "sram_access_pj": e_access_pj,
        "corner": "tt_025C_1v80",
        "sram_policy": "per-access (Liberty E/edge rise+fall), STA macro power excluded",
        "stimulus": "Qwen2.5-0.5B + SmolLM2-360M RTN captures (MXFP4 ocp_v11) + 50%-random",
    }
    if not SRAM_GLS.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/patch_sram_model.py")], check=True)

    with start_run(args.exp, cfg, paper=args.paper) as run:
        caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
        progs = []
        for w in range(args.windows):
            qw, qa = load_capture(caps[int(rng.integers(0, len(caps)))], "mxfp4")
            progs.append(
                (
                    f"real{w}",
                    program_from_capture(qw, qa, rng, m=args.m, sets_used=args.sets_used, reuse=args.reuse),
                )
            )
        for w in range(args.random):
            progs.append(
                (f"rand{w}", program_random(rng, m=args.m, sets_used=args.sets_used, reuse=args.reuse))
            )
        rows = []
        for name, prog in progs:
            ppath = save_program(build / "programs" / f"{name}.npz", prog)
            for design in args.designs.split(","):
                top = synth[design]["top"]
                wd = build / design / name
                wd.mkdir(parents=True, exist_ok=True)
                g = gls_with_vcd(design, top, ppath, clk_ps, wd, args.m)
                sliced = wd / "window.vcd"
                slice_vcd(g["vcd"], sliced, g["t_start_ns"], g["t_end_ns"], rebase=args.rebase)
                g["vcd"].unlink()
                pw = sta_power(design, top, sliced, clk_ps / 1e3, wd / "power.rpt")
                sliced.unlink()
                dur_s = (g["t_end_ns"] - g["t_start_ns"]) * 1e-9
                macs = g["n_ops"] * MACS_PER_OP
                p_logic = pw["p_total_w"] - pw["p_sram_sta_w"]
                e_logic_fj = p_logic * dur_s / macs * 1e15
                e_sram_fj = sram_accesses(prog, design) * e_access_pj * 1e3 / macs
                row = {
                    "design": design,
                    "window": name,
                    "kind": name[:4],
                    "n_ops": g["n_ops"],
                    "macs": macs,
                    "window_ns": dur_s * 1e9,
                    **pw,
                    "p_logic_w": p_logic,
                    "e_logic_fj_per_mac": e_logic_fj,
                    "e_sram_fj_per_mac": e_sram_fj,
                    "e_total_fj_per_mac": e_logic_fj + e_sram_fj,
                    "e_leak_fj_per_mac": pw["p_leakage_w"] * dur_s / macs * 1e15,
                }
                row["tflops_per_w"] = 2 / (row["e_total_fj_per_mac"] * 1e-15) / 1e12
                rows.append(row)
                print(
                    f"{design:12s} {name:6s} E/MAC {row['e_total_fj_per_mac']:8.2f} fJ "
                    f"(logic {e_logic_fj:7.2f}, sram {e_sram_fj:6.2f})  {row['tflops_per_w']:6.3f} TFLOPS/W"
                )
        run.save_csv("power.csv", rows)
        metrics = {}
        for d in args.designs.split(","):
            real = np.array(
                [r["e_total_fj_per_mac"] for r in rows if r["design"] == d and r["kind"] == "real"]
            )
            rand = np.array(
                [r["e_total_fj_per_mac"] for r in rows if r["design"] == d and r["kind"] == "rand"]
            )
            metrics[f"{d}_e_fj_per_mac_real"] = float(real.mean())
            metrics[f"{d}_e_fj_per_mac_rand"] = float(rand.mean()) if len(rand) else float("nan")
            metrics[f"{d}_tflops_per_w_real"] = float(2 / (real.mean() * 1e-15) / 1e12)
        # G1 ratio EE(v2)/EE(base) = E(base)/E(v2), paired by window, bootstrap CI
        designs = args.designs.split(",")
        for base in ("bs_int5", "dpu", "v2_generic"):
            if base not in designs or "v2_sram_m16" not in designs:  # per-design runs: ratios in add_boundary/e15
                continue
            pairs = np.array(
                [
                    next(r["e_total_fj_per_mac"] for r in rows if r["design"] == base and r["window"] == w)
                    / next(
                        r["e_total_fj_per_mac"]
                        for r in rows
                        if r["design"] == "v2_sram_m16" and r["window"] == w
                    )
                    for w, _ in progs
                    if w.startswith("real")
                ]
            )
            lo, hi = bootstrap_ci(pairs, rng)
            metrics[f"ee_ratio_v2_over_{base}"] = float(pairs.mean())
            metrics[f"ee_ratio_v2_over_{base}_ci_lo"] = lo
            metrics[f"ee_ratio_v2_over_{base}_ci_hi"] = hi
        run.log_metrics(metrics)
        print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
