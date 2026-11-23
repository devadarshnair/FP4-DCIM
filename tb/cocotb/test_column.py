"""Column RTL (MX) vs the scoreboard reference (== L2), op by op (WP-26).

Env: COLUMN_SUITE = corner | random | capture ; COLUMN_NDOTS (random dot products).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scoreboard import ColumnStim, column_reference, corner_stim, pack, random_stim, row_signals  # noqa: E402

from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

GUARD = 7
WINDOW = 40


def capture_stim(n_dots: int) -> ColumnStim:
    """Dot products from real WP-14 captures (one weight row x one activation row each)."""
    from cimfp4.llm.capture import load_capture

    caps = sorted((REPO_ROOT / "data" / "captures").glob("*/prefill/*.npz"))
    rng = np.random.default_rng(5)
    parts = {k: [] for k in ColumnStim.__dataclass_fields__}
    for i in range(n_dots):
        qw, qa = load_capture(caps[i % len(caps)], "mxfp4")
        r, t = int(rng.integers(0, qw.codes.shape[0])), int(rng.integers(0, qa.codes.shape[0]))
        nb = qw.num_blocks
        if nb > 2**GUARD:
            nb = 2**GUARD
        ew, ex = qw.scale_exp[r, :nb].astype(np.int64), qa.scale_exp[t, :nb].astype(np.int64)
        parts["w"].append(qw.blocked_codes()[r, :nb])
        parts["x"].append(qa.blocked_codes()[t, :nb])
        parts["e_w"].append(ew)
        parts["e_x"].append(ex)
        parts["max_e_w"].append(np.full(nb, ew.max()))
        parts["max_e_x"].append(np.full(nb, ex.max()))
        parts["clear"].append(np.arange(nb) == 0)
        parts["nan"].append(np.zeros(nb, bool))
    return ColumnStim(**{k: np.concatenate(v) for k, v in parts.items()})


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge

    def _stim() -> ColumnStim:
        suite = os.environ.get("COLUMN_SUITE", "corner")
        if suite == "corner":
            return corner_stim()
        if suite == "capture":
            return capture_stim(int(os.environ.get("COLUMN_NDOTS", "40")))
        return random_stim(
            np.random.default_rng(int(os.environ.get("COLUMN_SEED", "0"))),
            int(os.environ.get("COLUMN_NDOTS", "200")),
            max_blocks=2**GUARD,
            nan_rate=0.01,
        )

    @cocotb.test()
    async def column_matches_reference(dut):
        st = _stim()
        exp_acc, exp_nan = column_reference(st, WINDOW, GUARD)
        xp, xp3, xs = row_signals(st.x)
        wb = pack(st.w, 4)
        cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
        dut.rst_n.value = 0
        dut.in_valid.value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        mask = (1 << WINDOW) - 1
        got, nan_got, ovf = [], [], 0
        n = len(st)
        for t in range(n + 3):
            await FallingEdge(dut.clk)
            # sample state produced by the previous rising edge
            if int(dut.acc_valid.value):
                a = int(dut.acc.value) & mask
                got.append(a - (1 << WINDOW) if a >> (WINDOW - 1) else a)
                nan_got.append(bool(int(dut.nan_out.value)))
                ovf |= int(dut.ovf.value)
            if t < n:
                dut.in_valid.value = 1
                dut.acc_clear.value = int(st.clear[t])
                dut.xp_bus.value, dut.xp3_bus.value, dut.xs_bus.value, dut.w_bus.value = (
                    xp[t],
                    xp3[t],
                    xs[t],
                    wb[t],
                )
                dut.e_w.value = int(st.e_w[t]) & 0xFF
                dut.e_x.value = int(st.e_x[t]) & 0xFF
                dut.max_e_w.value = int(st.max_e_w[t]) & 0xFF
                dut.max_e_x.value = int(st.max_e_x[t]) & 0xFF
                dut.nan_in.value = int(st.nan[t])
            else:
                dut.in_valid.value = 0
        got_a = np.array(got[:n], dtype=np.int64)
        assert len(got) >= n, f"only {len(got)} results for {n} ops"
        mism = np.nonzero(got_a != exp_acc)[0]
        assert mism.size == 0, f"{mism.size} acc mismatches, first at op {mism[:5]}"
        assert np.array_equal(np.array(nan_got[:n]), exp_nan), "NaN flag mismatch"
        assert ovf == 0, "overflow/anchor assertion fired"
        dut._log.info(f"{n} ops, 0 mismatches")

except ImportError:
    pass


def _run(suite: str, tmp_path, ndots: int | None = None, seed: int = 0):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = [REPO_ROOT / "rtl/include/cim_pkg.sv", REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv"] + [
        REPO_ROOT / "rtl/src" / f
        for f in ("row_driver.sv", "leaf_mult.sv", "../baselines/digital_mac/generic_leaf.sv", "column.sv")
    ]
    runner = get_runner("verilator")
    runner.build(
        includes=[REPO_ROOT / "rtl" / "include"],
        sources=srcs,
        hdl_toplevel="column",
        build_dir=tmp_path / "b",
        always=True,
        parameters={"GUARD": GUARD},
        build_args=["-Wno-fatal"],
    )
    env = {
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
        "COLUMN_SUITE": suite,
        "COLUMN_SEED": str(seed),
    }
    if ndots:
        env["COLUMN_NDOTS"] = str(ndots)
    xml = runner.test(
        hdl_toplevel="column",
        test_module="test_column",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env=env,
    )
    assert get_results(xml) == (1, 0)


needs_verilator = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")


@needs_verilator
def test_column_corners(tmp_path):
    _run("corner", tmp_path)


@needs_verilator
def test_column_random(tmp_path):
    _run("random", tmp_path, ndots=int(os.environ.get("COLUMN_NDOTS", "200")))


@needs_verilator
@pytest.mark.skipif(not list((REPO_ROOT / "data" / "captures").glob("*/prefill/*.npz")), reason="no captures")
def test_column_capture_tiles(tmp_path):
    _run("capture", tmp_path, ndots=40)


@needs_verilator
@pytest.mark.slow
def test_column_random_1e6_ops(tmp_path):
    # ~1e6 ops: 16,000 dot products x ~64 blocks avg
    _run("random", tmp_path, ndots=16_000, seed=11)
