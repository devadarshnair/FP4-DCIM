"""e29 (T2.9) -- read access time / minimum cycle of the sky130 OpenRAM 1 KB macro in ngspice, reusing the e20
harness (trimmed m-multiplied array, same PDK netlist, tt models, 1.8 V, 27 C, 5 fF output load).

One run = one clock period x one array model (schematic or magic-extracted bitcells, as e20 low/nominal).
Sequence: 4 back-to-back port-0 writes (rows 0,1,8,8), then 5 back-to-back dual-port reads that alternate rows
(the first read directly follows the last write = the e20 `warm2` pattern), an idle cycle, a single read, idle.
Per read cycle: (a) data sampled just before the next rising edge (e20 style); (b) settle time of every dout bit
after the clock rising edge (last threshold crossing, vdd/2) from a saved waveform; (c) port-0 wordline overlap
(more than one of the observed rows above 0.9 V at once).

  python scripts/safe_run.py --max-rss-mb 3200 -- python experiments/e29_sram_timing/run.py --period-ns 25 --model ext
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

spec = importlib.util.spec_from_file_location("e20run", REPO_ROOT / "experiments/e20_sram_energy/run.py")
e20 = importlib.util.module_from_spec(spec)
sys.modules["e20run"] = e20
spec.loader.exec_module(e20)

EXT = REPO_ROOT / "results/e20_sram_cell_caps/20261110T034325916292Z_8b6180b/openram_cells_ext.spice"
KEEP = [0, 1, 8]
WLS = {"wl0_r0": "xsram.xbank0.wl_0_0", "wl0_r1": "xsram.xbank0.wl_0_1", "wl0_r8": "xsram.xbank0.wl_0_8",
       "wl1_r0": "xsram.xbank0.wl_1_0", "wl1_r1": "xsram.xbank0.wl_1_1", "wl1_r8": "xsram.xbank0.wl_1_8",
       "wl_en0": "xsram.wl_en0", "s_en0": "xsram.s_en0", "p_en_bar0": "xsram.p_en_bar0", "w_en0": "xsram.w_en0",
       "wl_en1": "xsram.wl_en1", "s_en1": "xsram.s_en1"}


def sequence(variant: str):
    if variant == "e20warm":  # exact e20 warm-up prefix (warm0 idle, warm1 write, warm1b write, warm2 dual read), then re-reads
        return [("warm0", None, None), ("warm1", ("w", 0x00, 0xA5A5A5A5), None),
                ("warm1b", ("w", 0x02, 0x3C3C5AA5), None), ("warm2", ("r", 0x00), ("r", 0x01)),
                ("idle_a", None, None), ("reread", ("r", 0x00), ("r", 0x02)), ("idle_b", None, None),
                ("reread2", ("r", 0x00), ("r", 0x02)), ("idle_c", None, None), ("end", None, None)]
    A, B, C, D = 0xA5A5A5A5, 0x3C3C5AA5, 0x5A5A5A5A, 0xC3C3A55A
    s = [("c0_idle", None, None), ("c1_wA", ("w", 0x00, A), None), ("c2_wB", ("w", 0x02, B), None),
         ("c3_wC", ("w", 0x10, C), None), ("c4_wD", ("w", 0x11, D), None)]
    if variant == "gap":  # control: one idle cycle between the last write and the first read
        s.append(("c4b_idle", None, None))
    s += [("rd1", ("r", 0x00), ("r", 0x11)), ("rd2", ("r", 0x02), ("r", 0x10)),
          ("rd3", ("r", 0x10), ("r", 0x00)), ("rd4", ("r", 0x11), ("r", 0x02)),
          ("rd5", ("r", 0x00), ("r", 0x11)), ("idle_a", None, None), ("rd_single", ("r", 0x02), None),
          ("idle_b", None, None), ("end", None, None)]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-ns", type=float, required=True)
    ap.add_argument("--model", choices=["schem", "ext"], default="ext")
    ap.add_argument("--variant", choices=["std", "gap", "e20warm"], default="std")
    ap.add_argument("--vdd", type=float, default=1.8)
    ap.add_argument("--tr-ps", type=float, default=100.0)
    ap.add_argument("--c-load-ff", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-run", default="", help="re-analyse waveform.txt of an earlier run dir")
    args = ap.parse_args()
    seq = sequence(args.variant)
    period = args.period_ns * 1e-9
    rng = np.random.default_rng(args.seed)
    cfg = {**vars(args), "macro": e20.MACRO, "keep_rows": KEEP, "temperature_c": 27, "sequence": [s[0] for s in seq],
           "cell_ext": str(EXT) if args.model == "ext" else "", "note": "dout launched on clk falling edge (OpenRAM)"}
    with start_run("e29_sram_timing", cfg) as run:
        wd = e20.BUILD.parent / "e29_sram_timing" / run.run_id
        deck, _ = e20.build_deck(seq, period, args.tr_ps * 1e-12, args.c_load_ff * 1e-15, args.vdd, 1, rng, wd,
                                 KEEP, EXT if args.model == "ext" else None)
        txt = deck.read_text()
        # drop the quiet window measurement and add waveform dump instead of quit
        # bracketed bus names cannot be used by save/wrdata: mirror them on plain nodes with ideal B sources
        mon = "".join(f"Bm{p}_{i} mon{p}_{i} 0 V=V(dout{p}[{i}])\n" for p in (0, 1) for i in range(32))
        vecs = [f"mon{p}_{i}" for p in (0, 1) for i in range(32)] + list(WLS.values())
        txt = txt.replace(".options reltol", mon + ".options reltol", 1)
        txt = re.sub(r"find v\(dout(\d)\[(\d+)\]\)", r"find v(mon\1_\2)", txt)
        txt = re.sub(r"^meas tran q_quiet_leak.*\n", "", txt, flags=re.M)
        txt = txt.replace(".control\nrun\n", ".control\nsave i(Vdd) " + " ".join(vecs) + "\nrun\n")  # keep memory low
        txt = txt.replace("quit\n", "set wr_singlescale\nset wr_vecnames\nwrdata waveform.txt " + " ".join(vecs) + "\nquit\n")
        deck.write_text(txt)
        t0 = time.time()
        wf = wd / "waveform.txt"
        if args.from_run:
            wf = Path(args.from_run) / "waveform.txt"
            log = (Path(args.from_run) / "ngspice.log").read_text()
        else:
            r = subprocess.run(["ngspice", "-b", deck.name], cwd=wd, capture_output=True, text=True)
            log = r.stdout + r.stderr
        (run.dir / "ngspice.log").write_text(log[-100000:])
        if not args.from_run:
            if not wf.exists():
                raise RuntimeError("ngspice produced no waveform.txt; see ngspice.log")
            (run.dir / "waveform.txt").write_bytes(wf.read_bytes())
        wall = time.time() - t0
        meas = {m.group(1).lower(): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+=\s+([-+\d.eE]+)", log, re.M)}
        # waveform
        with open(wf) as f:
            hdr = f.readline().split()
        arr = np.loadtxt(wf, skiprows=1)
        t, cols = arr[:, 0], {h: arr[:, i] for i, h in enumerate(hdr)}
        vdd = args.vdd
        exp = e20.expected_reads(seq)
        rows, fails = [], []
        for k, (lab, p0, p1) in enumerate(seq[:-1]):
            rise = (k + 0.5) * period
            for p, op in ((0, p0), (1, p1)):
                if not (op and op[0] == "r"):
                    continue
                want = exp[(p, lab)]
                if want < 0:  # never written (random initial content): no expectation
                    continue
                got = 0
                for i in range(32):
                    got |= (1 if meas[f"dout{p}_{lab}_{i}".lower()] > vdd / 2 else 0) << i
                ok = got == want
                # settle time: last time in [rise, rise+period] any bit disagrees with `want`
                win = (t >= rise) & (t <= rise + period)
                t_settle, n_first = 0.0, 0
                for i in range(32):
                    wbit = (want >> i) & 1
                    v = cols[f"mon{p}_{i}"]
                    bad = ((v[win] > vdd / 2) != bool(wbit))
                    if bad.any():
                        t_settle = max(t_settle, t[win][bad][-1] - rise)
                    # data present at time of rising edge from previous read? (was bit already correct at rise)
                    n_first += 0
                fell = (k + 1.0) * period
                rows.append({"cycle": lab, "port": p, "want": hex(want), "got_before_next_edge": hex(got), "ok": ok,
                             "t_settle_ns_after_rise": t_settle * 1e9,
                             "t_settle_ns_after_fall": (t_settle - (fell - rise)) * 1e9})
                if not ok:
                    fails.append(f"{lab}/p{p}:{hex(got)}!={hex(want)}")
        run.save_csv("reads.csv", rows)
        # wordline overlap among the observed port-0 rows in each cycle
        wl = np.vstack([cols[WLS[n]] for n in ("wl0_r0", "wl0_r1", "wl0_r8")])
        n_on = (wl > 0.9 * vdd).sum(axis=0)
        multi = []
        for k in range(len(seq) - 1):
            w = (t >= (k + 0.5) * period) & (t < (k + 1.5) * period)
            tt, mm = t[w], n_on[w] > 1
            dur = float(np.sum(np.diff(tt, prepend=tt[0])[mm])) * 1e9
            multi.append({"cycle": seq[k][0], "wl0_multi_row_ns": dur})
        run.save_csv("wl_overlap.csv", multi)
        tset = [r["t_settle_ns_after_rise"] for r in rows]
        m = {"wall_s": wall, "period_ns": args.period_ns, "model": args.model, "variant": args.variant,
             "read_fail_count": len(fails), "read_fails": fails, "n_reads": len(rows),
             "t_settle_max_ns_after_rise": max(tset) if tset else None,
             "t_settle_steady_dual_ns_after_rise": max([r["t_settle_ns_after_rise"] for r in rows if r["cycle"] in ("rd2", "rd3", "rd4", "rd5")] or [None]),
             "reads": [f"{r['cycle']}/p{r['port']}:{'ok' if r['ok'] else r['got_before_next_edge']}" for r in rows],
             "wl0_multi_row_ns_max": max(x["wl0_multi_row_ns"] for x in multi)}
        run.log_metrics(m)
        print(m)


if __name__ == "__main__":
    main()
