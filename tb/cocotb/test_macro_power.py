"""Macro-level power harness (macro-level power): drives macro_top / bs_macro_top (RTL or sky130 netlist) with a
saved macro program (src/cimfp4/stimulus/writers.py::macro_program_*), checks every op bit-exactly
against the per-column reference models, and writes the compute-window times for VCD slicing.

Program convention (same as the G1 column-group programs, writers.py): a weight set stays active
while `reuse` activation vectors stream through it; every op starts a new partial sum (clear = 1).
Env: MP_PROGRAM (.npz), MP_FMT (mxfp4|nvfp4|int4sm|mixed: per-op op_nv, T4.2), MP_CLK_PS, MP_WINDOW_OUT (json, optional).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scoreboard import ColumnStim, column_reference, int4sm_column_reference  # noqa: E402

from cimfp4.golden.l2_macro import nv_column_reference  # noqa: E402
from cimfp4.stimulus.writers import load_program  # noqa: E402

GUARD, W = 7, 40


def expected(prog: dict, fmt: str) -> np.ndarray:
    """(n_ops, 32) expected window value of every column after every op."""
    if fmt == "mixed":  # T4.2 mixed stream (writers.macro_program_mixed): every op clears, so split by op_nv
        nv = prog["op_nv"].astype(bool)
        out = np.zeros((len(nv), 32), np.int64)
        per_op = ("x", "e_x0", "e_x1", "u_x0", "u_x1", "max_e_x", "set", "clear")
        for f, sel in (("mxfp4", ~nv), ("nvfp4", nv)):
            if sel.any():
                out[sel] = expected({**prog, **{k: prog[k][sel] for k in per_op}}, f)
        return out
    s = prog["set"]
    n = len(s)
    out = np.zeros((n, 32), np.int64)
    for c in range(32):
        w = prog["sets"][s, c]  # (n, 32)
        if fmt in ("mxfp4", "int4sm"):  # int4sm: T5.4 D2 INT4-SM macro, MX-style scaling, {sign, mag} codes
            st = ColumnStim(
                w=w,
                x=prog["x"],
                e_w=prog["e0"][s, c],
                e_x=prog["e_x0"],
                max_e_w=np.full(n, prog["maxw"][c]),
                max_e_x=prog["max_e_x"],
                clear=prog["clear"].astype(bool),
                nan=np.zeros(n, bool),
            )
            out[:, c] = int4sm_column_reference(st, W, GUARD) if fmt == "int4sm" else column_reference(st, W, GUARD)[0]
        else:
            out[:, c] = nv_column_reference(
                w,
                prog["x"],
                np.stack([prog["u0"][s, c], prog["u1"][s, c]], 1),
                np.stack([prog["e0"][s, c], prog["e1"][s, c]], 1),
                np.stack([prog["u_x0"], prog["u_x1"]], 1),
                np.stack([prog["e_x0"], prog["e_x1"]], 1),
                np.full(n, prog["maxw"][c]),
                prog["max_e_x"],
                prog["clear"].astype(bool),
                W,
                GUARD,
            )
    return out


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer
    from cocotb.utils import get_sim_time

    @cocotb.test()
    async def macro_power_program(dut):
        fmt = os.environ["MP_FMT"]
        prog = load_program(Path(os.environ["MP_PROGRAM"]))
        exp = expected(prog, fmt)
        clk_ps = int(os.environ.get("MP_CLK_PS", "25000"))
        cocotb.start_soon(Clock(dut.clk, clk_ps, unit="ps").start())
        for sig in ("rst_n", "wr_en", "sc_wr_en", "maxw_wr_en", "op_valid", "op_x_nan", "op_clear"):
            getattr(dut, sig).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        used = sorted(set(int(v) for v in prog["set"]))
        for b in used:  # only the sets the program uses are loaded (the rest stay zero)
            for c in range(32):
                for k in range(4):
                    word = sum(int(prog["sets"][b, c, 8 * k + j]) << (4 * j) for j in range(8))
                    await FallingEdge(dut.clk)
                    dut.wr_en.value, dut.wr_group.value = 1, c // 4
                    dut.wr_addr.value, dut.wr_data.value = b * 16 + (c % 4) * 4 + k, word
                await FallingEdge(dut.clk)
                dut.wr_en.value = 0
                dut.sc_wr_en.value, dut.sc_wr_set.value, dut.sc_wr_col.value = 1, b, c
                dut.sc_wr_e0.value = int(prog["e0"][b, c]) & 0xFF
                dut.sc_wr_e1.value = int(prog["e1"][b, c]) & 0xFF
                dut.sc_wr_u0.value, dut.sc_wr_u1.value = int(prog["u0"][b, c]), int(prog["u1"][b, c])
                dut.sc_wr_nan.value = 0
        for c in range(32):
            await FallingEdge(dut.clk)
            dut.sc_wr_en.value = 0
            dut.maxw_wr_en.value, dut.maxw_wr_col.value = 1, c
            dut.maxw_wr_val.value = int(prog["maxw"][c]) & 0xFF
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0
        await FallingEdge(dut.clk)
        t_start = get_sim_time("ns")
        t_last = t_start
        got = [[] for _ in range(32)]
        i, drain, stalls, mask = 0, 0, 0, (1 << W) - 1
        n_ops = len(prog["set"])
        while i < n_ops or drain < 16:
            await FallingEdge(dut.clk)
            av = int(dut.acc_valid.value)
            if av:
                t_last = get_sim_time("ns")
                a = int(dut.acc.value)
                for c in range(32):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
            if i < n_ops:
                dut.op_valid.value, dut.op_nv.value = 1, (int(prog["op_nv"][i]) if fmt == "mixed" else int(fmt == "nvfp4"))
                dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(prog["x"][i]))
                dut.op_e_x0.value, dut.op_e_x1.value = int(prog["e_x0"][i]) & 0xFF, int(prog["e_x1"][i]) & 0xFF
                dut.op_u_x0.value, dut.op_u_x1.value = int(prog["u_x0"][i]), int(prog["u_x1"][i])
                dut.op_max_e_x.value = int(prog["max_e_x"][i]) & 0xFF
                dut.op_set.value, dut.op_clear.value = int(prog["set"][i]), int(prog["clear"][i])
                await Timer(1, unit="ns")
                if int(dut.op_ready.value):
                    i += 1
                else:
                    stalls += 1
            else:
                dut.op_valid.value = 0
                drain += 1
        t_end = get_sim_time("ns")
        g = np.array(got, np.int64).T
        assert g.shape == exp.shape, (g.shape, exp.shape)
        mism = int((g != exp).sum())
        assert mism == 0, f"{mism} mismatches"
        n_sw = int(dut.set_switches.value)
        if os.environ.get("MP_WINDOW_OUT"):
            Path(os.environ["MP_WINDOW_OUT"]).write_text(
                json.dumps(
                    {
                        "t_start_ns": t_start,
                        "t_end_ns": t_end,
                        "t_last_valid_ns": t_last,
                        "n_ops": n_ops,
                        "set_switches": n_sw,
                        "stall_cycles": stalls,
                        "cycles": round((t_end - t_start) * 1000 / clk_ps),
                        "drain_cycles": drain,
                    }
                )
            )
        dut._log.info(f"macro power {fmt}: {n_ops} ops x 32 col, {n_sw} switches, {stalls} stalls, 0 mismatches")

except ImportError:
    pass
