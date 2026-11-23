"""NVFP4 mode of the column RTL vs the NV column reference (== L1 Window, tests/golden/test_l2_nv.py) (WP-31).

Suites: `pairs` - all 16×16 scale-significand pairs (u ∈ 0..15: subnormal 0..7 and normal 8..15) on both
halves with random elements; `random` - NVFP4-quantized random dot products (≥10^5 ops in the slow test).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scoreboard import pack, row_signals  # noqa: E402

from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402
from cimfp4.golden.l2_macro import nv_column_reference, nv_ops_from_blocks  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

GUARD, W = 7, 40


def pairs_ops(rng):
    t = 256
    u = np.array([(a, b) for a in range(16) for b in range(16)])
    k_w = rng.integers(-9, 3, (t, 2))
    k_x = rng.integers(-9, 3, (t, 2))
    return {
        "w": rng.integers(0, 16, (t, 32)).astype(np.uint8),
        "x": rng.integers(0, 16, (t, 32)).astype(np.uint8),
        "u_w": np.stack([u[:, 0], u[::-1, 0]], 1),
        "k_w": k_w,
        "u_x": np.stack([u[:, 1], u[::-1, 1]], 1),
        "k_x": k_x,
        "max_k_w": np.full(t, 2),
        "max_k_x": np.full(t, 2),
        "clear": np.ones(t, bool),
    }


def random_ops(rng, n_dots):
    parts = []
    for _ in range(n_dots):
        k = int(rng.choice([32, 512, 2048, 4096]))
        qw = quantize_nvfp4(rng.standard_normal((1, k)) * np.exp(rng.normal(0, 1)))
        qx = quantize_nvfp4(rng.standard_t(3, (1, k)) * np.exp(rng.normal(0, 1)))
        parts.append(nv_ops_from_blocks(qw, qx))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge

    @cocotb.test()
    async def nv_column_matches(dut):
        rng = np.random.default_rng(int(os.environ.get("NV_SEED", "0")))
        ops = (
            pairs_ops(rng)
            if os.environ.get("NV_SUITE") == "pairs"
            else random_ops(rng, int(os.environ.get("NV_NDOTS", "200")))
        )
        exp = nv_column_reference(
            ops["w"],
            ops["x"],
            ops["u_w"],
            ops["k_w"],
            ops["u_x"],
            ops["k_x"],
            ops["max_k_w"],
            ops["max_k_x"],
            ops["clear"],
            W,
            GUARD,
        )
        xp, xp3, xs = row_signals(ops["x"])
        wb = pack(ops["w"], 4)
        cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
        dut.rst_n.value, dut.in_valid.value, dut.nv_mode.value, dut.nan_in.value = 0, 0, 1, 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        got, ovf, mask = [], 0, (1 << W) - 1
        n = len(exp)
        for t in range(n + 3):
            await FallingEdge(dut.clk)
            if int(dut.acc_valid.value):
                a = int(dut.acc.value) & mask
                got.append(a - (1 << W) if a >> (W - 1) else a)
                ovf |= int(dut.ovf.value)
            if t < n:
                dut.in_valid.value, dut.acc_clear.value = 1, int(ops["clear"][t])
                dut.xp_bus.value, dut.xp3_bus.value, dut.xs_bus.value, dut.w_bus.value = (
                    xp[t],
                    xp3[t],
                    xs[t],
                    wb[t],
                )
                dut.e_w.value, dut.e_w1.value = int(ops["k_w"][t, 0]) & 0xFF, int(ops["k_w"][t, 1]) & 0xFF
                dut.e_x.value, dut.e_x1.value = int(ops["k_x"][t, 0]) & 0xFF, int(ops["k_x"][t, 1]) & 0xFF
                dut.u_w0.value, dut.u_w1.value = int(ops["u_w"][t, 0]), int(ops["u_w"][t, 1])
                dut.u_x0.value, dut.u_x1.value = int(ops["u_x"][t, 0]), int(ops["u_x"][t, 1])
                dut.max_e_w.value, dut.max_e_x.value = (
                    int(ops["max_k_w"][t]) & 0xFF,
                    int(ops["max_k_x"][t]) & 0xFF,
                )
            else:
                dut.in_valid.value = 0
        mism = int((np.array(got[:n], np.int64) != exp).sum())
        assert len(got) >= n and mism == 0, f"{mism} mismatches of {n}"
        assert ovf == 0
        dut._log.info(f"NV: {n} ops, 0 mismatches")

except ImportError:
    pass


def _run(suite, tmp_path, ndots=200, seed=0):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        [REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv"]
        + [REPO_ROOT / "rtl/src" / f for f in ("row_driver.sv", "leaf_mult.sv", "column.sv")]
        + [REPO_ROOT / "rtl/baselines/digital_mac/generic_leaf.sv"]
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel="column",
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters={"GUARD": GUARD},
        build_args=["-Wno-fatal"],
    )
    xml = runner.test(
        hdl_toplevel="column",
        test_module="test_nv",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "NV_SUITE": suite,
            "NV_NDOTS": str(ndots),
            "NV_SEED": str(seed),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")


@needs
def test_nv_scale_significand_pairs(tmp_path):
    _run("pairs", tmp_path)


@needs
def test_nv_random(tmp_path):
    _run("random", tmp_path, ndots=200)


@needs
@pytest.mark.slow
def test_nv_random_1e5(tmp_path):
    _run("random", tmp_path, ndots=2000, seed=5)  # ~2000 dots x ~50 ops avg ≈ 1e5 ops
