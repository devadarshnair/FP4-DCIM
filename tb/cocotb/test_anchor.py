"""Tile-anchor detector vs the Hier golden anchor (max_tile e_x) on 10^5 tiles (WP-68)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.utils.provenance import REPO_ROOT

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge

    @cocotb.test()
    async def anchor_matches(dut):
        rng = np.random.default_rng(0)
        cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
        dut.rst_n.value, dut.valid.value = 0, 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        tiles = [rng.integers(-127, 128, int(rng.integers(1, 17))) for _ in range(100_000)]
        stream = [(int(e), j == len(t) - 1) for t in tiles for j, e in enumerate(t)]
        want = [int(t.max()) for t in tiles]
        got = []
        padded = [(e, last, 1) for e, last in stream] + [(0, False, 0)] * 2
        for e, last, v in padded:
            await FallingEdge(dut.clk)
            if int(dut.anchor_valid.value):
                a = int(dut.anchor.value)
                got.append(a - 256 if a > 127 else a)
            dut.valid.value, dut.e_x.value, dut.last.value = v, e & 0xFF, int(last)
        await FallingEdge(dut.clk)
        if int(dut.anchor_valid.value):
            a = int(dut.anchor.value)
            got.append(a - 256 if a > 127 else a)
        assert got == want, (
            f"{sum(g != w for g, w in zip(got, want, strict=False))} mismatches; {len(got)} vs {len(want)}"
        )

except ImportError:
    pass


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_anchor(tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    runner = get_runner("verilator")
    runner.build(
        sources=[REPO_ROOT / "rtl/src/anchor_detect.sv"],
        hdl_toplevel="anchor_detect",
        build_dir=tmp_path / "b",
        always=True,
        build_args=["-Wno-fatal"],
    )
    xml = runner.test(
        hdl_toplevel="anchor_detect",
        test_module="test_anchor",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert get_results(xml) == (1, 0)
