"""Baseline (a) bit-serial INT5 column group vs the scoreboard (== L2 == L1 Window) (WP-36).

Reuses the column-group cocotb test (same op interface and storage); 5 cycles per block.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_column_top import GUARD, SRAM_V  # noqa: E402

from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

try:
    from test_column_top import column_top_matches  # noqa: F401  (cocotb discovers it here)
except ImportError:
    pass


def _run(m, n_ops, tmp_path, capture=False, seed=0, store=0, pattern=None):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = [
        REPO_ROOT / "rtl/include/cim_pkg.sv",
        REPO_ROOT / "rtl/src/cim_icg.sv",
        REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv",
        REPO_ROOT / "rtl/src/wset_sram.sv",
        REPO_ROOT / "rtl/src/wset_latch.sv",
        REPO_ROOT / "rtl/baselines/bitserial_int5/bs_column.sv",
        REPO_ROOT / "rtl/baselines/bitserial_int5/bs_column_top.sv",
        SRAM_V,
    ]
    runner = get_runner("verilator")
    runner.build(
        includes=[REPO_ROOT / "rtl" / "include"],
        sources=srcs,
        hdl_toplevel="bs_column_top",
        build_dir=tmp_path / "b",
        always=True,
        parameters={"M": m, "STORE": store, "GUARD": GUARD},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    env = {
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
        "CT_M": str(m),
        "CT_NOPS": str(n_ops),
        "CT_SEED": str(seed),
        "CT_CAPTURE": "1" if capture else "0",
        "CT_EXPECT_MIN_GAP": "5",  # numerics.md baseline (a): 5 cycles/op back-to-back
    }
    if pattern:
        env["CT_PATTERN"] = pattern
    xml = runner.test(
        hdl_toplevel="bs_column_top",
        test_module="test_bs_int5",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env=env,
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(),
    reason="verilator / sky130 OpenRAM models missing",
)
HAS_CAP = bool(list((REPO_ROOT / "data" / "captures").glob("*/prefill/*.npz")))


@needs
@pytest.mark.parametrize("m", [1, 16])
def test_bs_int5(m, tmp_path):
    _run(m, int(os.environ.get("CT_NOPS", "5000")), tmp_path)


@needs
@pytest.mark.skipif(not HAS_CAP, reason="no captures")
def test_bs_int5_capture(tmp_path):
    _run(16, 3000, tmp_path, capture=True, seed=3)


@needs
@pytest.mark.slow
def test_bs_int5_1e5(tmp_path):
    _run(16, 100_000, tmp_path, seed=9)


@needs
@pytest.mark.parametrize("store", [0, 1], ids=["SRAM", "LATCH"])
@pytest.mark.parametrize("m", [4, 16])
def test_bs_int5_pattern(store, m, tmp_path):
    """T1.6b: mixed same-set runs + switches; min same-set gap == 5, n_switch == #set changes, 0 mismatches."""
    from test_column_top import MIXED_PATTERN

    _run(m, 2600, tmp_path, seed=11, store=store, pattern=MIXED_PATTERN)
