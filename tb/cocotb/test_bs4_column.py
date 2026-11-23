"""T5.1: INT4 sign-magnitude bit-serial column (lossy D2 surrogate, never INT5/exact) at column level.

Checks the aligned window accumulator against an integer reference of the contract (block sum
sum_r (-1)^(sw^sx) |w||x|, 3 cycles/op, same alignment + W-bit window as bs_column) and the 3-cycle issue gap.
Light test: tiny random + exhaustive-magnitude-pair stimulus (the 1e5-op verification is T5.2).
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

W, GUARD, SB, ROWS = 40, 7, 14, 32
BASE, DMAX = W - SB - 1 - GUARD, W - 1


def _ref_acc(acc, blk, d_raw, clear):
    d = min(d_raw, DMAX) if d_raw >= 0 else d_raw & 0x3F  # RTL: b_d <= clamp, else low 6 bits (ovf flagged)
    term = (blk << (BASE - d)) if d <= BASE else (blk >> (d - BASE))
    s = (0 if clear else acc) + term
    s &= (1 << W) - 1
    return s - (1 << W) if s >> (W - 1) else s


def _pack(vals):  # signed ints in [-7,7] -> {sign, mag[2:0]} per row
    out = 0
    for r, v in enumerate(vals):
        out |= ((1 if v < 0 else 0) << 3 | abs(v)) << (4 * r)
    return out


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import FallingEdge, RisingEdge

    def _make_ops(rng, n_ops):
        ops = []
        for k in range(n_ops):
            if k < 49:  # every magnitude pair 1..7 x 1..7 with sign mixes (lane 31 carries the opposite weight sign)
                wv, xv = [0] * ROWS, [0] * ROWS
                wv[0], xv[0] = (k // 7 + 1) * (-1 if k & 1 else 1), (k % 7 + 1) * (-1 if k & 2 else 1)
                wv[31], xv[31] = -wv[0], xv[0]
            elif k < 53:  # extremes: all lanes +-7 (|block sum| = 1568)
                wv, xv = [7 * (1 if k & 2 else -1)] * ROWS, [7 * (1 if k & 1 else -1)] * ROWS
            else:
                wv = [rng.randint(-7, 7) for _ in range(ROWS)]
                xv = [rng.randint(-7, 7) for _ in range(ROWS)]
            ew, ex = rng.randint(-20, 20), rng.randint(-20, 20)
            mw, mx = ew + rng.randint(0, 6), ex + rng.randint(0, 6)
            ops.append((wv, xv, ew, ex, mw, mx, int(rng.random() < 0.25 or k == 0)))
        return ops

    @cocotb.test()
    async def bs4_column_matches(dut):
        rng = random.Random(int(os.environ.get("B4_SEED", "1")))
        n_ops = int(os.environ.get("B4_NOPS", "300"))
        ops = _make_ops(rng, n_ops)
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        dut.rst_n.value = 0
        dut.op_valid.value = 0
        for sig in ("op_clear", "xbit_bus", "xsgn_bus", "w_bus", "e_w", "e_x", "max_e_w", "max_e_x", "nan_in"):
            getattr(dut, sig).value = 0
        for _ in range(3):
            await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)

        idx, cyc, accept_cycles, exp, got = 0, 0, [], [], []
        mags, shift_n = [0] * ROWS, 9
        acc_model = 0
        while len(got) < n_ops and cyc < 10 * n_ops + 50:
            await FallingEdge(dut.clk)
            will_accept = idx < n_ops and int(dut.op_ready.value) == 1
            if will_accept:
                wv, xv, ew, ex, mw, mx, clr = ops[idx]
                dut.op_valid.value = 1
                dut.e_w.value, dut.e_x.value, dut.max_e_w.value, dut.max_e_x.value = ew, ex, mw, mx
                dut.op_clear.value, dut.nan_in.value = clr, 0
            else:
                dut.op_valid.value = 0
            await RisingEdge(dut.clk)
            cyc += 1
            if int(dut.acc_valid.value) == 1:  # value registered at this edge, readable after it
                pass
            # row-shift-register model (as bs_column_top): loaded on accept, one bit per cycle, LSB first
            if will_accept:
                wv, xv, *_ = ops[idx]
                mags, shift_n = [abs(v) for v in xv], 0
                dut.w_bus.value = _pack(wv)
                dut.xsgn_bus.value = sum(1 << r for r, v in enumerate(xv) if v < 0)
                blk = sum(wv[r] * xv[r] for r in range(ROWS))
                _, _, ew, ex, mw, mx, clr = ops[idx]
                acc_model = _ref_acc(acc_model, blk, mw - ew + mx - ex, clr)
                exp.append(acc_model)
                accept_cycles.append(cyc)
                idx += 1
            else:
                shift_n += 1
            dut.xbit_bus.value = sum(((mags[r] >> shift_n) & 1) << r for r in range(ROWS)) if shift_n < 3 else 0
            await cocotb.triggers.ReadOnly()
            if int(dut.acc_valid.value) == 1:
                v = int(dut.acc.value.to_signed()) if hasattr(dut.acc.value, "to_signed") else int(dut.acc.value.signed_integer)
                got.append(v)
        assert len(got) == n_ops, f"only {len(got)}/{n_ops} results"
        bad = [i for i in range(n_ops) if got[i] != exp[i]]
        assert not bad, f"{len(bad)} mismatches, first op {bad[0]}: got {got[bad[0]]} exp {exp[bad[0]]}"
        gaps = {b - a for a, b in zip(accept_cycles, accept_cycles[1:])}
        assert gaps == {3}, f"issue gaps {gaps} (expected back-to-back 3 cycles/op)"
except ImportError:
    pass


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator missing")


@needs
def test_bs4_column(tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = [
        REPO_ROOT / "rtl/include/cim_pkg.sv",
        REPO_ROOT / "rtl/src/cim_icg.sv",
        REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv",
        REPO_ROOT / "rtl/baselines/bitserial_int4sm/bs4_column.sv",
    ]
    runner = get_runner("verilator")
    runner.build(
        includes=[REPO_ROOT / "rtl" / "include"],
        sources=srcs,
        hdl_toplevel="bs4_column",
        build_dir=tmp_path / "b",
        always=True,
        parameters={"GUARD": GUARD},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="bs4_column",
        test_module="test_bs4_column",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}"},
    )
    assert get_results(xml) == (1, 0)
