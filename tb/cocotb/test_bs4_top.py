"""T5.3: INT4-SM bit-serial column group (bs4_column_top) vs an independent integer reference.

Same op interface, weight/exponent write sequence, window measurement and drain as test_column_top.py, but the
operand codes are MXINT4-SM {sign, mag[2:0]} nibbles (programs saved by experiments/e34_d2_tile) and the reference is
the contract arithmetic of tb/cocotb/test_bs4_verify.py (block sum sum_r (-1)^(sw^sx)|w||x|, alignment d, W-bit window,
sticky ovf), computed per column with the program's own exponents.  Used for RTL (Verilator) and for GLS (Icarus on a
Yosys/OpenROAD netlist; CT_PROGRAM + CT_WINDOW_OUT), exactly as e13 uses test_column_top.
Env: CT_PROGRAM (npz) | CT_M, CT_NOPS, CT_SEED (random program), CT_CLK_PS, CT_WINDOW_OUT, CT_EXPECT_MIN_GAP.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_bs4_verify import ref_step  # noqa: E402

from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
NCOL, GUARD, W = 4, 7, 40


def random_program(rng, m, n_ops):
    sets = rng.integers(0, 16, (m, NCOL, 32)).astype(np.uint8)
    ew = rng.integers(-6, 3, (m, NCOL))
    maxw = ew.max(axis=0)
    ops = []
    for j in range(n_ops):
        e_x = int(rng.integers(-8, 9))
        ops.append(dict(x=rng.integers(0, 16, 32).astype(np.uint8), e_x=e_x, max_e_x=e_x + int(rng.integers(0, 4)),
                        set=int(rng.integers(0, m)) if j % 5 == 0 else (ops[-1]["set"] if ops else 0),
                        clear=bool(rng.random() < 0.3), nan=False))
    return sets, ew, maxw, ops


def expected(sets, ew, maxw, ops):
    """(n_ops, NCOL) window accumulators from the integer contract."""
    out = np.zeros((len(ops), NCOL), np.int64)
    for c in range(NCOL):
        acc, ovf, nanf = 0, 0, 0
        for i, o in enumerate(ops):
            w = sets[o["set"], c].astype(np.int64)
            x = np.asarray(o["x"], np.int64)
            sw, mw, sx, mx = (w >> 3) & 1, w & 7, (x >> 3) & 1, x & 7
            blk = int((np.where(sw ^ sx, -1, 1) * mw * mx).sum())
            d_raw = int(maxw[c]) - int(ew[o["set"], c]) + o["max_e_x"] - o["e_x"]
            acc, ovf, nanf = ref_step(acc, ovf, nanf, blk, d_raw, int(o["clear"]), int(o["nan"]))
            out[i, c] = acc
    return out


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer

    @cocotb.test()
    async def bs4_top_matches(dut):
        m, n_ops = int(os.environ["CT_M"]), int(os.environ["CT_NOPS"])
        rng = np.random.default_rng(int(os.environ.get("CT_SEED", "0")))
        if os.environ.get("CT_PROGRAM"):
            from cimfp4.stimulus.writers import load_program

            pg = load_program(os.environ["CT_PROGRAM"])
            sets, ew, maxw = pg["sets"], pg["ew"], pg["maxw"]
            ops = [dict(x=pg["x"][i], e_x=int(pg["e_x"][i]), max_e_x=int(pg["max_e_x"][i]), set=int(pg["set"][i]),
                        clear=bool(pg["clear"][i]), nan=False) for i in range(len(pg["x"]))]
            m = sets.shape[0]
        else:
            sets, ew, maxw, ops = random_program(rng, m, n_ops)
        exp = expected(sets, ew, maxw, ops)
        cocotb.start_soon(Clock(dut.clk, int(os.environ.get("CT_CLK_PS", "10000")), unit="ps").start())
        for s in ("rst_n", "wr_en", "ew_wr_en", "maxw_wr_en", "op_valid"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        for s in range(m):
            for c in range(NCOL):
                for k in range(4):
                    word = sum(int(sets[s, c, 8 * k + j]) << (4 * j) for j in range(8))
                    await FallingEdge(dut.clk)
                    dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, s * 16 + c * 4 + k, word
                await FallingEdge(dut.clk)
                dut.wr_en.value = 0
                dut.ew_wr_en.value, dut.ew_wr_set.value, dut.ew_wr_col.value = 1, s, c
                dut.ew_wr_val.value, dut.ew_wr_nan.value = int(ew[s, c]) & 0xFF, 0
        for c in range(NCOL):
            await FallingEdge(dut.clk)
            dut.ew_wr_en.value = 0
            dut.maxw_wr_en.value, dut.maxw_wr_col.value, dut.maxw_wr_val.value = 1, c, int(maxw[c]) & 0xFF
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0
        from cocotb.utils import get_sim_time

        t_start = get_sim_time("ns")
        got = [[] for _ in range(NCOL)]
        i, stalls, guard = 0, 0, 0
        mask = (1 << W) - 1
        cyc, acc_cyc = 0, []
        while i < len(ops) or guard < 16:  # drain: 3 + 3 cycles of latency
            await FallingEdge(dut.clk)
            cyc += 1
            av = int(dut.acc_valid.value)
            if av:
                a = int(dut.acc.value)
                for c in range(NCOL):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
            if i < len(ops):
                o = ops[i]
                dut.op_valid.value = 1
                dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(o["x"]))
                dut.op_e_x.value, dut.op_max_e_x.value = o["e_x"] & 0xFF, o["max_e_x"] & 0xFF
                dut.op_x_nan.value, dut.op_set.value, dut.op_clear.value = int(o["nan"]), o["set"], int(o["clear"])
                await Timer(1, unit="ns")
                if int(dut.op_ready.value):
                    i += 1
                    acc_cyc.append(cyc)
                else:
                    stalls += 1
            else:
                dut.op_valid.value = 0
                guard += 1
        t_end = get_sim_time("ns")
        if os.environ.get("CT_WINDOW_OUT"):
            import json

            Path(os.environ["CT_WINDOW_OUT"]).write_text(
                json.dumps({"t_start_ns": t_start, "t_end_ns": t_end, "n_ops": len(ops), "stall_cycles": stalls}))
        got_a = np.array(got, dtype=np.int64).T
        assert got_a.shape == exp.shape, (got_a.shape, exp.shape)
        mism = int((got_a != exp).sum())
        if os.environ.get("CT_MISMATCH_OUT"):
            Path(os.environ["CT_MISMATCH_OUT"]).write_text(f"{mism} of {exp.size}\n")
        assert mism == 0, f"{mism} mismatches"
        n_switch = int(dut.set_switches.value)
        want_switch = 1 if m == 1 else sum(1 for j in range(len(ops)) if j == 0 or ops[j]["set"] != ops[j - 1]["set"])
        assert n_switch == want_switch, (n_switch, want_switch)
        gaps = [b - a for a, b in zip(acc_cyc, acc_cyc[1:], strict=False)]
        same = [ops[j + 1]["set"] == ops[j]["set"] for j in range(len(ops) - 1)]
        sg = [g for g, s_ in zip(gaps, same, strict=True) if s_]
        exp_gap = int(os.environ.get("CT_EXPECT_MIN_GAP", "0"))
        if exp_gap and sg:
            assert min(sg) == exp_gap, f"min cycles/op {min(sg)} != {exp_gap}"
        assert stalls >= n_switch
        dut._log.info(f"bs4 top M={m}: {len(ops)} ops, {n_switch} set switches, {stalls} stalls, 0 mismatches")

except ImportError:
    pass


needs = pytest.mark.skipif(shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator / OpenRAM model missing")


@needs
@pytest.mark.parametrize("m", [1, 16])
def test_bs4_top(m, tmp_path):
    from cocotb_tools.runner import get_results, get_runner

    srcs = [REPO_ROOT / "rtl/include/cim_pkg.sv", REPO_ROOT / "rtl/src/cim_icg.sv",
            REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv", REPO_ROOT / "rtl/src/wset_sram.sv",
            REPO_ROOT / "rtl/src/wset_latch.sv", REPO_ROOT / "rtl/baselines/bitserial_int4sm/bs4_column.sv",
            REPO_ROOT / "rtl/baselines/bitserial_int4sm/bs4_column_top.sv", SRAM_V]
    runner = get_runner("verilator")
    runner.build(includes=[REPO_ROOT / "rtl" / "include"], sources=srcs, hdl_toplevel="bs4_column_top",
                 build_dir=tmp_path / "b", always=True, parameters={"M": m, "STORE": 0, "GUARD": GUARD},
                 build_args=["--timing", "-Wno-fatal"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel="bs4_column_top", test_module="test_bs4_top", test_dir=Path(__file__).parent,
                      build_dir=tmp_path / "b", results_xml=str(tmp_path / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}", "CT_M": str(m),
                                 "CT_NOPS": "3000", "CT_SEED": "3", "CT_EXPECT_MIN_GAP": "3"})
    assert get_results(xml) == (1, 0)
