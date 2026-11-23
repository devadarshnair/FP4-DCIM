"""e20 - transistor-level energy characterization of the sky130 OpenRAM 1 KB macro
(replaces the placeholder Liberty SRAM energy).

The shipped Liberty (`sky130_sram_1kbyte_1rw1r_32x256_8_TT_1p8V_25C.lib`) charges one constant
9.589 pJ per clock edge for every condition (read, write, deselected), i.e. it is an uncharacterized
placeholder. Here the macro's own transistor netlist shipped with the PDK
(`libs.ref/sky130_sram_macros/spice/sky130_sram_1kbyte_1rw1r_32x256_8.spice`, OpenRAM-generated,
schematic/LVS netlist: devices only, NO layout parasitics) is simulated in ngspice with the sky130 tt
models at 1.8 V / 27 C (ngspice default temperature; the Liberty corner is 25 C), and the supply energy
is integrated per clock cycle for a scripted sequence: idle (both ports deselected, clock running),
port-0 write (all / half / no bits flipping), port-0 read, port-1 read, dual-port read (the wset_sram
set-switch pattern), plus a quiet window with the clock stopped (leakage).

Bitcell state: every one of the 8192 storage nodes gets a seeded random `.ic` so the DC operating point
starts from valid stored data (no metastable cells). Outputs are loaded with C_LOAD each.

Because wiring parasitics are absent, the measured energies are a LOWER bound on the real macro;
the range construction (low / nominal / high) lives in `range.py` and docs/spec/sram_energy.md.

Run (heavy: ~150k devices):
  python scripts/safe_run.py --max-rss-mb 4000 -- python experiments/e20_sram_energy/run.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

PDK = Path.home() / ".ciel/sky130A"
MACRO = "sky130_sram_1kbyte_1rw1r_32x256_8"
NETLIST = PDK / "libs.ref/sky130_sram_macros/spice" / f"{MACRO}.spice"
MODELS = PDK / "libs.tech/ngspice/sky130.lib.spice"
BUILD = REPO_ROOT / "build/e20_sram_energy"
SPICEINIT = """set ngbehavior=hsa
set skywaterpdk
set ng_nomodcheck
set num_threads=4
option noinit
option klu
"""

# One entry per clock cycle: (label, port0 op, port1 op). port op: None | ("r", addr) | ("w", addr, data)
# words_per_row = 2: addr[0] selects the column-mux word, addr[7:1] the row (OpenRAM, 64 bl x 128 wl).


def sequence(rng: np.random.Generator) -> list[tuple[str, tuple | None, tuple | None]]:
    """OpenRAM restores (precharges) the bitlines a read/write discharged at the start of the NEXT cycle, so
    every access is followed by a measured tail cycle, and the set-switch pattern of wset_sram (dual-port
    reads of consecutive words, a0 = set*16 + 2*step, a1 = a0 + 1) is run back-to-back (steady state)."""
    d0 = 0xA5A5A5A5
    dual = [(f"dual{i}", ("r", 0x10 + 2 * i), ("r", 0x11 + 2 * i)) for i in range(4)]
    return [
        ("warm0", None, None),
        ("warm1", ("w", 0x00, d0), None),
        ("warm1b", ("w", 0x02, 0x3C3C5AA5), None),
        ("warm2", ("r", 0x00), ("r", 0x01)),
        ("idle_a", None, None),
        ("idle_b", None, None),
        ("idle_c", None, None),
        ("write_flip_all", ("w", 0x00, d0 ^ 0xFFFFFFFF), None),
        ("write_flip_none", ("w", 0x00, d0 ^ 0xFFFFFFFF), None),
        ("write_flip_half", ("w", 0x00, (d0 ^ 0xFFFFFFFF) ^ 0x0F0F0F0F), None),
        ("tail_w", None, None),
        ("idle_d", None, None),
        ("read_p0", ("r", 0x00), None),
        ("tail_r0", None, None),
        ("read_p1", None, ("r", 0x02)),
        ("tail_r1", None, None),
        *dual,
        ("tail_dual", None, None),
        ("idle_e", None, None),
        ("end", None, None),
    ]


def trimmed_netlist(keep_rows: list[int], out: Path, cell: str = "sky130_fd_bd_sram__openram_dp_cell") -> Path:
    """Copy of the PDK netlist whose bitcell_array keeps only `keep_rows` as individual cells. Each column's
    other 128 - k cells are represented by two multiplied instances of the same dp_cell (m = n/2 storing
    0, m = n - n/2 storing 1) with both wordlines grounded: their bitline junction/overlap capacitance and
    access-transistor leakage are kept by the device models themselves (the inactive rows' wordlines never
    rise in the sequence). Validated against the full netlist on the smoke sequence (see run notes)."""
    lines = NETLIST.read_text().splitlines()
    res, inside = [], False
    n_l = 128 - len(keep_rows)
    for ln in lines:
        if ln.startswith(".SUBCKT bitcell_array "):
            inside = True
        elif inside and ln.startswith(".ENDS"):
            for c in range(64):
                pins = f"bl_0_{c} br_0_{c} bl_1_{c} br_1_{c} gnd gnd vdd gnd"
                res.append(f"Xlump0_c{c} {pins} {cell} m={n_l // 2}")
                res.append(f"Xlump1_c{c} {pins} {cell} m={n_l - n_l // 2}")
            inside = False
        elif inside and ln.startswith("Xbit_r"):
            r = int(ln.split()[0][len("Xbit_r"):].split("_")[0])
            if r not in keep_rows:
                continue
            ln = ln.rsplit(" ", 1)[0] + f" {cell}"
        if cell.endswith("_ext") and ln.startswith("X"):
            for v in ("_replica", "_dummy"):  # replica/dummy cells follow the array (replica-bitline timing)
                if ln.endswith(f" sky130_fd_bd_sram__openram_dp_cell{v}"):
                    ln += "_ext"
        res.append(ln)
    out.write_text("\n".join(res) + "\n")
    return out


def bits(v: int, n: int) -> list[int]:
    return [(v >> i) & 1 for i in range(n)]


def pwl(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{t:.4e} {v:.4g}" for t, v in points)


def build_deck(seq, period: float, tr: float, c_load: float, vdd: float, quiet_cycles: int,
               rng: np.random.Generator, wd: Path, keep_rows: list[int] | None = None,
               cell_ext: Path | None = None) -> tuple[Path, list[tuple[str, float, float]]]:
    """Return deck path and [(label, t0, t1)] energy windows (rising edge to next rising edge)."""
    n = len(seq)
    t_end = (n + quiet_cycles + 0.5) * period
    wd.mkdir(parents=True, exist_ok=True)
    cell = "sky130_fd_bd_sram__openram_dp_cell" + ("_ext" if cell_ext else "")
    net = NETLIST if keep_rows is None else trimmed_netlist(keep_rows, wd / f"{MACRO}_trimmed.spice", cell)
    lines = [f"* e20 OpenRAM {MACRO} energy characterization", f'.lib "{MODELS}" tt', f'.include "{net}"']
    if cell_ext:
        lines.append(f'.include "{cell_ext}"')
    lines.append(f"Vdd vccd1 0 {vdd}")
    lines.append("Vss vssd1 0 0")
    # clocks: low in [kT, kT+T/2), rising at (k+0.5)T; stopped (low) during the quiet window
    clk_pts = [(0.0, 0.0)]
    for k in range(n):
        tr0 = (k + 0.5) * period
        tf0 = (k + 1.0) * period
        clk_pts += [(tr0, 0.0), (tr0 + tr, vdd), (tf0, vdd), (tf0 + tr, 0.0)]
    for c in ("clk0", "clk1"):
        lines.append(f"V{c} {c} 0 PWL({pwl(clk_pts)})")

    # control/address/data change at kT (falling edge of the previous cycle), held through cycle k
    def sig_pts(values: list[int]) -> list[tuple[float, float]]:
        pts = [(0.0, values[0] * vdd)]
        for k in range(1, n):
            if values[k] != values[k - 1]:
                pts += [(k * period, values[k - 1] * vdd), (k * period + tr, values[k] * vdd)]
        return pts

    csb0 = [0 if s[1] else 1 for s in seq]
    csb1 = [0 if s[2] else 1 for s in seq]
    web0 = [0 if (s[1] and s[1][0] == "w") else 1 for s in seq]
    # hold address/data at their previous value when the port is idle (no spurious input toggles)
    a0, a1, din = [], [], []
    pa0 = pa1 = pd = 0
    for _, p0, p1 in seq:
        if p0:
            pa0 = p0[1]
            if p0[0] == "w":
                pd = p0[2]
        if p1:
            pa1 = p1[1]
        a0.append(pa0)
        a1.append(pa1)
        din.append(pd)
    for name, vals in (("csb0", csb0), ("csb1", csb1), ("web0", web0)):
        lines.append(f"V{name} {name} 0 PWL({pwl(sig_pts(vals))})")
    for i in range(4):
        lines.append(f"Vwmask{i} wmask0[{i}] 0 {vdd}")
    for i in range(8):
        lines.append(f"Va0_{i} addr0[{i}] 0 PWL({pwl(sig_pts([bits(a, 8)[i] for a in a0]))})")
        lines.append(f"Va1_{i} addr1[{i}] 0 PWL({pwl(sig_pts([bits(a, 8)[i] for a in a1]))})")
    for i in range(32):
        lines.append(f"Vd{i} din0[{i}] 0 PWL({pwl(sig_pts([bits(d, 32)[i] for d in din]))})")
    pins = (
        [f"din0[{i}]" for i in range(31, -1, -1)]
        + [f"addr0[{i}]" for i in range(7, -1, -1)]
        + [f"addr1[{i}]" for i in range(7, -1, -1)]
        + ["csb0", "csb1", "web0", "clk0", "clk1"]
        + [f"wmask0[{i}]" for i in range(3, -1, -1)]
        + [f"dout0[{i}]" for i in range(31, -1, -1)]
        + [f"dout1[{i}]" for i in range(31, -1, -1)]
        + ["vccd1", "vssd1"]
    )
    lines.append("Xsram " + " ".join(pins) + f" {MACRO}")
    for p in (0, 1):
        for i in range(32):
            lines.append(f"Cl{p}_{i} dout{p}[{i}] 0 {c_load}")
    # random stored data: 128 rows x 64 columns of the main bitcell array
    ics = []
    q = rng.integers(0, 2, size=(128, 64))
    base = "xsram.xbank0.xbitcell_array.xbitcell_array"
    qn = ".xraw" if cell_ext else ""  # extracted cell = wrapper around the raw magic subckt
    for r in range(128):
        if keep_rows is not None and r not in keep_rows:
            continue
        for c in range(64):
            v = vdd * q[r, c]
            ics.append(f"v({base}.xbit_r{r}_c{c}{qn}.q)={v:g} v({base}.xbit_r{r}_c{c}{qn}.q_bar)={vdd - v:g}")
    if keep_rows is not None:
        for c in range(64):
            ics.append(f"v({base}.xlump0_c{c}{qn}.q)=0 v({base}.xlump0_c{c}{qn}.q_bar)={vdd:g}")
            ics.append(f"v({base}.xlump1_c{c}{qn}.q)={vdd:g} v({base}.xlump1_c{c}{qn}.q_bar)=0")
    for i in range(0, len(ics), 8):
        lines.append(".ic " + " ".join(ics[i : i + 8]))
    lines.append(".options reltol=1e-3 abstol=1e-12 vntol=1e-6 method=gear")
    lines.append(f".tran 100p {t_end:.4e} 0 250p")
    windows = [(seq[k][0], (k + 0.5) * period, (k + 1.5) * period) for k in range(n - 1)]
    # quiet: clock stopped, nothing switching -> static leakage of the whole macro
    q0 = (n + 0.5) * period
    windows.append(("quiet_leak", q0 + 0.25 * quiet_cycles * period, t_end))
    lines.append(".control")
    lines.append("run")
    for lab, t0, t1 in windows:
        lines.append(f"meas tran q_{lab} integ i(Vdd) from={t0:.6e} to={t1:.6e}")
    for p in (0, 1):
        for k, (lab, _, _) in enumerate(seq[:-1]):
            if seq[k][1 + p] and seq[k][1 + p][0] == "r":
                t_s = (k + 1.45) * period
                for i in range(32):
                    lines.append(f"meas tran dout{p}_{lab}_{i} find v(dout{p}[{i}]) at={t_s:.6e}")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")
    (wd / ".spiceinit").write_text(SPICEINIT)
    deck = wd / "e20.sp"
    deck.write_text("\n".join(lines) + "\n")
    return deck, windows


def expected_reads(seq) -> dict[tuple[int, str], int]:
    mem: dict[int, int] = {}
    exp = {}
    for lab, p0, p1 in seq:
        for p, op in ((0, p0), (1, p1)):
            if op and op[0] == "r":
                exp[(p, lab)] = mem.get(op[1], -1)
        if p0 and p0[0] == "w":
            mem[p0[1]] = p0[2]
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-ns", type=float, default=25.0, help="40 MHz = f_G1 (e12)")
    ap.add_argument("--tr-ps", type=float, default=100.0)
    ap.add_argument("--c-load-ff", type=float, default=5.0)
    ap.add_argument("--vdd", type=float, default=1.8)
    ap.add_argument("--quiet-cycles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-ext", default="", help="openram_cells_ext.spice (e20_sram_cell_caps); needs --keep-rows")
    ap.add_argument("--keep-rows", default="", help="e.g. 0-4: trimmed array (see trimmed_netlist); '' = full")
    ap.add_argument("--from-log", default="", help="re-analyse a previous run's ngspice.log (no simulation)")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    seq = sequence(rng)
    keep = None
    if args.keep_rows:
        a, _, b = args.keep_rows.partition("-")
        keep = list(range(int(a), int(b or a) + 1))
    period = args.period_ns * 1e-9
    cfg = {
        **vars(args),
        "macro": MACRO,
        "netlist": str(NETLIST),
        "models": f"{MODELS} tt",
        "temperature_c": 27,
        "parasitics": ("bitcells: magic-extracted layout (devices with AD/PD + in-cell wiring); periphery: none"
                       if args.cell_ext else "none (schematic/LVS netlist shipped with the PDK)"),
        "array": "full 128x64" if keep is None else f"trimmed: rows {args.keep_rows} individual, rest lumped (m-multiplied cells)",
        "sequence": [s[0] for s in seq],
        "energy_window": "rising clock edge k to rising edge k+1, supply Vdd current only",
    }
    ver = subprocess.run(["ngspice", "-v"], capture_output=True, text=True).stdout
    cfg["ngspice"] = next((ln.strip("* ").strip() for ln in ver.splitlines() if "ngspice-" in ln), "?")
    with start_run("e20_sram_energy", cfg, paper=args.paper) as run:
        wd = BUILD / run.run_id
        deck, windows = build_deck(seq, period, args.tr_ps * 1e-12, args.c_load_ff * 1e-15, args.vdd,
                                   args.quiet_cycles, rng, wd, keep,
                                   Path(args.cell_ext) if args.cell_ext else None)
        t0 = time.time()
        if args.from_log:  # re-analysis of a previous run's ngspice output (same deck, regenerated above)
            log, rc = Path(args.from_log).read_text(), 0
        else:
            r = subprocess.run(["ngspice", "-b", deck.name], cwd=wd, capture_output=True, text=True)
            log, rc = r.stdout + r.stderr, r.returncode
        wall = time.time() - t0
        (run.dir / "ngspice.log").write_text(log[-200000:])
        (run.dir / "e20.sp.head").write_text("\n".join(deck.read_text().splitlines()[:120]) + "\n")
        meas = {m.group(1).lower(): float(m.group(2))
                for m in re.finditer(r"^(\S+)\s+=\s+([-+\d.eE]+)", log, re.M)}
        rows = []
        for lab, ta, tb in windows:
            qv = meas.get(f"q_{lab}".lower())
            if qv is None:
                raise RuntimeError(f"missing measurement q_{lab}; see {run.dir / 'ngspice.log'}")
            e_pj = -qv * args.vdd * 1e12  # i(Vdd) is negative when the source delivers current
            rows.append({"window": lab, "t0_ns": ta * 1e9, "t1_ns": tb * 1e9, "energy_pj": e_pj,
                         "avg_power_uw": e_pj * 1e-12 / (tb - ta) * 1e6})
        run.save_csv("cycle_energy.csv", rows)
        # functional check of every read against the written data
        exp = expected_reads(seq)
        fails = []
        for (p, lab), want in exp.items():
            if want < 0:
                continue
            got = 0
            for i in range(32):
                v = meas[f"dout{p}_{lab}_{i}".lower()]
                got |= (1 if v > args.vdd / 2 else 0) << i
            if got != want:
                fails.append({"port": p, "cycle": lab, "want": hex(want), "got": hex(got)})
        # warm-up cycles only prime the macro after the DC operating point; their checks are reported, and
        # only failures in measured cycles invalidate the run
        warm_fails = [f for f in fails if f["cycle"].startswith("warm")]
        fails = [f for f in fails if not f["cycle"].startswith("warm")]
        e = {r["window"]: r["energy_pj"] for r in rows}
        idle = float(np.mean([e["idle_b"], e["idle_c"], e["idle_e"]]))  # clocked, deselected, nothing to restore
        leak_w = e["quiet_leak"] * 1e-12 / (windows[-1][2] - windows[-1][1])
        dual_ss = float(np.mean([e["dual1"], e["dual2"], e["dual3"]]))  # each also restores the previous read
        m = {
            "wall_s": wall,
            "ngspice_rc": rc,
            "read_check_failures": fails,
            "warmup_read_check_failures": warm_fails,
            "read_checks": [f"port{p}:{lab}" for (p, lab), w in exp.items() if w >= 0],
            "e_idle_cycle_pj": idle,
            "e_idle_spread_pj": [e["idle_b"], e["idle_c"], e["idle_e"]],
            "leakage_uw": leak_w * 1e6,
            "e_leak_per_cycle_pj": leak_w * period * 1e12,
            "e_dual_read_steady_cycle_pj": dual_ss,
            # energy above a clocked idle cycle; steady state = access + restore of the previous access
            "de_read_dual_pj": dual_ss - idle,
            "de_read_dual_first_pj": e["dual0"] - idle,
            "de_dual_tail_restore_pj": e["tail_dual"] - idle,
            "de_read_p0_pj": e["read_p0"] + e["tail_r0"] - 2 * idle,
            "de_read_p1_pj": e["read_p1"] + e["tail_r1"] - 2 * idle,
            "de_write_flip_all_pj": e["write_flip_all"] - idle,
            "de_write_flip_none_pj": e["write_flip_none"] - idle,
            "de_write_flip_half_pj": e["write_flip_half"] - idle,
            "de_write_tail_restore_pj": e["tail_w"] - idle,
        }
        m["de_read_per_word_dual_pj"] = m["de_read_dual_pj"] / 2
        # one wset_sram set switch per OpenRAM macro = 8 back-to-back dual reads (first + 7 steady + tail)
        m["de_set_switch_per_macro_pj"] = (m["de_read_dual_first_pj"] + 7 * m["de_read_dual_pj"]
                                           + m["de_dual_tail_restore_pj"])
        m["liberty_placeholder_pj_per_edge"] = 9.589466
        run.log_metrics(m)
        if fails:
            raise RuntimeError(f"read check failed: {fails[:3]}")
        print({k: v for k, v in m.items() if k != "read_check_failures"})


if __name__ == "__main__":
    main()
