"""e26 (T2.1) - DPU (digital-MAC baseline) NVFP4 macro power on the SAME NV windows as the v2 NV numbers.

Reuses the e21 flow (GLS bit-exact check + rebased VCD slice + native OpenSTA, tt 1.8 V, 40 MHz, ideal clock) on the
hierarchical dpu_macro netlist (build/synth_macro_dpu_hier, results/e12_macro_synth_dpu_macro_hier). The NV programs are
the saved .npz files of the v2 NV run (e21 20261110T050824693997Z_f25a5c5, programs/nvfp4_*.npz), NOT regenerated:
e21's RNG stream depends on the MX programs being drawn first, so a fresh NV-only run would pick different windows.

  python scripts/safe_run.py --max-rss-mb 5200 -- python experiments/e26_dpu_nv_power/run.py [--paper] [--windows real0,...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.power.boundary import broadcast_toggles, weight_move_toggles
from cimfp4.power.vcd import slice_vcd
from cimfp4.stimulus.writers import load_program
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

spec = importlib.util.spec_from_file_location("e21run", REPO_ROOT / "experiments/e21_macro_power/run.py")
e21 = importlib.util.module_from_spec(spec)
sys.modules["e21run"] = e21
spec.loader.exec_module(e21)

REF = "20261110T050824693997Z_f25a5c5"  # e21 run that produced the v2 NV numbers
NET, TOP, FMT = "build/synth_macro_dpu_hier/dpu_macro_netlist.v", "dpu_macro", "nvfp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="real0,real1,real2,real3,rand0")
    ap.add_argument("--clk-ps", type=int, default=25000)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    ref_csv = {r["window"]: r for r in __import__("csv").DictReader(
        open(REPO_ROOT / "results/e21_macro_power" / REF / "power.csv")) if r["design"] == "v2_macro_nv"}
    progdir = REPO_ROOT / "build/e21_macro_power" / REF / "programs"
    names = args.windows.split(",")
    cfg = {**vars(args), "design": "dpu_macro_nv", "netlist": NET, "ref_e21_run": REF,
           "program_sha256": {n: hashlib.sha256((progdir / f"nvfp4_{n}.npz").read_bytes()).hexdigest() for n in names},
           "corner": "sky130_fd_sc_hd tt_025C_1v80 (STA); ideal clock; zero-delay GLS; VCD rebased (B9)",
           "sram_policy": "OpenRAM Liberty power excluded; SRAM composed from e20/D33 in compare.py",
           "macs_per_op": e21.MACS_PER_OP}
    net = REPO_ROOT / NET
    with start_run("e26_dpu_nv_power", cfg, paper=args.paper) as run:
        build = e21.BUILD.parent / "e26_dpu_nv_power" / run.run_id
        rows = []
        for name in names:
            ppath = progdir / f"nvfp4_{name}.npz"
            prog = load_program(ppath)
            wd = build / name
            wd.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            g = e21.gls(net, TOP, FMT, ppath, args.clk_ps, wd)  # raises on any mismatch
            t_gls = time.time() - t0
            t1 = g["t_last_valid_ns"] + args.clk_ps / 1e3
            sliced = wd / "window.vcd"
            st = slice_vcd(g["vcd"], sliced, g["t_start_ns"], t1, rebase=True)
            g["vcd"].unlink()
            t0 = time.time()
            pw = e21.sta_power(net, TOP, sliced, args.clk_ps / 1e3, wd / "power.rpt")
            t_sta = time.time() - t0
            sliced.unlink()
            (run.dir / f"power_dpu_macro_nv_{name}.rpt").write_text((wd / "power.rpt").read_text())
            dur_s = (t1 - g["t_start_ns"]) * 1e-9
            cycles = round((t1 - g["t_start_ns"]) * 1e3 / args.clk_ps)
            p_logic = pw["p_total_w"] - pw["p_sram_sta_w"]
            macs = g["n_ops"] * e21.MACS_PER_OP
            row = {"design": "dpu_macro_nv", "fmt": FMT, "window": name, "kind": name[:4],
                   "source": ref_csv[name]["source"], "n_ops": g["n_ops"], "macs": macs, "cycles": cycles,
                   "window_ns": dur_s * 1e9, "set_switches": g["set_switches"], "stall_cycles": g["stall_cycles"],
                   "sram_read_words": g["set_switches"] * e21.NG * 16,
                   "sram_dual_read_cycles": g["set_switches"] * e21.NG * 8, "vcd_changes": st["changes_kept"],
                   "gls_mismatches": 0, "gls_s": t_gls, "sta_s": t_sta, **pw, "p_logic_w": p_logic,
                   "e_logic_pj_per_op": p_logic * dur_s / g["n_ops"] * 1e12,
                   "e_logic_fj_per_mac": p_logic * dur_s / macs * 1e15,
                   "e_leak_fj_per_mac": (pw["p_total_leak_w"] - pw["p_sram_sta_leak_w"]) * dur_s / macs * 1e15,
                   "cycles_per_op": cycles / g["n_ops"],
                   "bcast_toggles": broadcast_toggles(prog, "dpu"), "wmove_toggles": weight_move_toggles(prog)}
            rows.append(row)
            run.save_csv("power.csv", rows)
            print(f"{name:6s} {row['e_logic_fj_per_mac']:8.2f} fJ/MAC logic, {row['cycles_per_op']:.3f} cyc/op, "
                  f"gls {t_gls:.0f}s sta {t_sta:.0f}s (v2 NV ref {float(ref_csv[name]['e_logic_fj_per_mac']):.1f})", flush=True)
        run.log_metrics({"dpu_macro_nv_e_logic_fj_per_mac_real": float(np.mean([r["e_logic_fj_per_mac"] for r in rows if r["kind"] == "real"])),
                         "n_windows": len(rows), "gls_mismatches_total": 0})
        print(json.dumps(run.metrics if hasattr(run, "metrics") else {}, indent=1))


if __name__ == "__main__":
    main()
