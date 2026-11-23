"""Weight-set stores (OpenRAM SRAM and latch ablation): random writes + 10^4 random set loads (WP-63).

Checks the active weights after every load, and records the set-switch latency in cycles.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.utils.provenance import REPO_ROOT

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
NCOL, M, WORDS = 4, 16, 16

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge

    @cocotb.test()
    async def wset_loads(dut):
        n_loads = int(os.environ.get("WSET_NLOADS", "10000"))
        rng = np.random.default_rng(1)
        mem = rng.integers(0, 2**32, M * WORDS, dtype=np.uint64)
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        dut.rst_n.value, dut.wr_en.value, dut.load_req.value = 0, 0, 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1

        async def write(addr, val):
            await FallingEdge(dut.clk)
            dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, int(addr), int(val)
            await FallingEdge(dut.clk)
            dut.wr_en.value = 0

        for a in range(M * WORDS):
            await write(a, mem[a])
        latencies, bad = [], 0
        for i in range(n_loads):
            if i % 50 == 49:  # occasionally rewrite a word between loads
                a = int(rng.integers(0, M * WORDS))
                mem[a] = rng.integers(0, 2**32, dtype=np.uint64)
                await write(a, mem[a])
            s = int(rng.integers(0, M))
            await FallingEdge(dut.clk)
            dut.load_req.value, dut.load_set.value = 1, s
            await FallingEdge(dut.clk)
            dut.load_req.value = 0
            cyc = 1
            while int(dut.busy.value):
                await FallingEdge(dut.clk)
                cyc += 1
            latencies.append(cyc)
            got = int(dut.w_active.value)
            want = sum(int(mem[s * WORDS + k]) << (32 * k) for k in range(WORDS))
            bad += got != want
        assert bad == 0, f"{bad} wrong set loads"
        dut._log.info(f"set-switch latency cycles: min {min(latencies)} max {max(latencies)}")
        Path(os.environ.get("WSET_REPORT", "/dev/null")).write_text(f"{max(latencies)}\n")

except ImportError:
    pass


def _run(top: str, tmp_path, n_loads: int = 10_000) -> int:
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = [REPO_ROOT / "rtl/src/cim_icg.sv", REPO_ROOT / "rtl/src" / f"{top}.sv"] + (
        [SRAM_V] if top == "wset_sram" else []
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel=top,
        build_dir=tmp_path / "b",
        always=True,
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    report = tmp_path / "latency.txt"
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_wset",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "WSET_NLOADS": str(n_loads),
            "WSET_REPORT": str(report),
        },
    )
    assert get_results(xml) == (1, 0)
    return int(report.read_text())


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")


@needs
@pytest.mark.skipif(not SRAM_V.exists(), reason="sky130 OpenRAM models not installed")
def test_wset_sram(tmp_path):
    lat = _run("wset_sram", tmp_path)
    # cycles from the load_req edge to busy low: 1 request + 8 dual-port reads + 1 capture
    assert lat == 10


@needs
def test_wset_latch(tmp_path):
    assert _run("wset_latch", tmp_path) == 2  # 1 request + 1 parallel copy
