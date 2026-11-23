"""Ping-pong weight-set store (WP-67): 10^4 randomized interleavings of writes, prefetches and op set
requests; every op's active weights must equal the SRAM contents of its set at the time that set was read.
Also measures swap latency (prefetched set) and demand-load latency (not prefetched).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.utils.provenance import REPO_ROOT

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
M, WORDS = 16, 16

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer

    @cocotb.test()
    async def pingpong_interleavings(dut):
        n_steps = int(os.environ.get("PP_STEPS", "10000"))
        rng = np.random.default_rng(3)
        mem = np.zeros(M * WORDS, dtype=np.uint64)
        snap: dict[int, np.ndarray] = {}  # set -> contents captured when its read started
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        for s in ("rst_n", "wr_en", "pf_req", "use_valid"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        for a in range(M * WORDS):  # initial contents
            mem[a] = rng.integers(0, 2**32, dtype=np.uint64)
            await FallingEdge(dut.clk)
            dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, a, int(mem[a])
        await FallingEdge(dut.clk)
        dut.wr_en.value = 0
        was_reading, ops, bad, swap_lat, demand_lat = 0, 0, 0, [], []
        for _ in range(n_steps):
            act = rng.random()
            await FallingEdge(dut.clk)
            dut.wr_en.value = dut.pf_req.value = dut.use_valid.value = 0
            if act < 0.35:  # write a random word (write-while-compute)
                a = int(rng.integers(0, M * WORDS))
                v = int(rng.integers(0, 2**32, dtype=np.uint64))
                dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, a, v
                await Timer(1, unit="ns")
                if int(dut.wr_ready.value):
                    mem[a] = v
            elif act < 0.6:  # prefetch
                dut.pf_req.value, dut.pf_set.value = 1, int(rng.integers(0, M))
            else:  # an op needs set s: hold use_valid until ready, then check the active weights
                s = int(rng.integers(0, M))
                prefetched = bool(int(dut.sh_valid.value)) and int(dut.sh_set.value) == s
                dut.use_valid.value, dut.use_set.value = 1, s
                cyc = 0
                while True:
                    await Timer(1, unit="ns")
                    r = int(dut.reading.value)
                    if r and not was_reading:
                        snap[int(dut.rd_set.value)] = mem[
                            int(dut.rd_set.value) * WORDS : (int(dut.rd_set.value) + 1) * WORDS
                        ].copy()
                    was_reading = r
                    if int(dut.ready.value):
                        break
                    await FallingEdge(dut.clk)
                    cyc += 1
                want = sum(
                    int(snap.get(s, mem[s * WORDS : (s + 1) * WORDS])[k]) << (32 * k) for k in range(WORDS)
                )
                bad += int(dut.w_active.value) != want
                ops += 1
                if cyc:
                    (swap_lat if prefetched else demand_lat).append(cyc)
            # track read starts triggered by prefetches
            await Timer(2, unit="ns")
            r = int(dut.reading.value)
            if r and not was_reading:
                rs = int(dut.rd_set.value)
                snap[rs] = mem[rs * WORDS : (rs + 1) * WORDS].copy()
            was_reading = r
        assert bad == 0, f"{bad} of {ops} ops saw wrong weights"
        dut._log.info(
            f"{ops} ops; swap latency {sorted(set(swap_lat))}; demand latency {sorted(set(demand_lat))}; "
            f"demand_loads {int(dut.demand_loads.value)} swaps {int(dut.swaps.value)}"
        )
        Path(os.environ.get("PP_REPORT", "/dev/null")).write_text(
            f"{max(swap_lat) if swap_lat else 0} {max(demand_lat) if demand_lat else 0}\n"
        )

except ImportError:
    pass


@pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)
def test_pingpong(tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    runner = get_runner("verilator")
    runner.build(
        sources=[REPO_ROOT / "rtl/src/cim_icg.sv", REPO_ROOT / "rtl/src/wset_pingpong.sv", SRAM_V],
        hdl_toplevel="wset_pingpong",
        build_dir=tmp_path / "b",
        always=True,
        build_args=["--timing", "-Wno-fatal", "--public-flat-rw"],
        timescale=("1ns", "1ps"),
    )
    rep = tmp_path / "lat.txt"
    xml = runner.test(
        hdl_toplevel="wset_pingpong",
        test_module="test_pingpong",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": str(REPO_ROOT / "src"), "PP_REPORT": str(rep)},
    )
    assert get_results(xml) == (1, 0)
    swap, demand = (int(v) for v in rep.read_text().split())
    assert swap <= 1, swap  # a prefetched set is swapped in within one cycle
    # worst case: an in-flight prefetch of another set (<= 9 cycles, not aborted) + demand load
    # (decision + 8 dual-port reads + capture + swap <= 11) = 20 cycles
    assert demand <= 20, demand
