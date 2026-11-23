"""Baseline (c) DPU column group and the v2 LEAF=GENERIC ablation vs the scoreboard (WP-39)."""

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
    from test_column_top import column_top_matches  # noqa: F401
except ImportError:
    pass

BASE = (
    [
        REPO_ROOT / "rtl/include/cim_pkg.sv",
        REPO_ROOT / "rtl/src/cim_icg.sv",
        REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv",
    ]
    + [
        REPO_ROOT / "rtl/src" / f
        for f in (
            "row_driver.sv",
            "leaf_mult.sv",
            "column.sv",
            "wset_latch.sv",
            "wset_sram.sv",
            "column_top.sv",
        )
    ]
    + [REPO_ROOT / "rtl/baselines/digital_mac/generic_leaf.sv", SRAM_V]
)


def _run(top, params, n_ops, tmp_path, capture=False, seed=0):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = BASE + ([REPO_ROOT / "rtl/baselines/digital_mac/dpu_column.sv"] if top == "dpu_column" else [])
    runner = get_runner("verilator")
    runner.build(
        includes=[REPO_ROOT / "rtl" / "include"],
        sources=srcs,
        hdl_toplevel=top,
        build_dir=tmp_path / "b",
        always=True,
        parameters=params,
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    env = {
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
        "CT_M": str(params.get("M", 16)),
        "CT_NOPS": str(n_ops),
        "CT_SEED": str(seed),
        "CT_CAPTURE": "1" if capture else "0",
    }
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_digital_mac",
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
N = int(os.environ.get("CT_NOPS", "5000"))


@needs
def test_dpu_column(tmp_path):
    _run("dpu_column", {"M": 16, "GUARD": GUARD}, N, tmp_path)


@needs
@pytest.mark.skipif(not HAS_CAP, reason="no captures")
def test_dpu_column_capture(tmp_path):
    _run("dpu_column", {"M": 16, "GUARD": GUARD}, 3000, tmp_path, capture=True, seed=4)


@needs
def test_v2_generic_leaf_ablation(tmp_path):
    _run("column_top", {"M": 16, "GUARD": GUARD, "STORE": 0, "LEAF": 1}, N, tmp_path)


@needs
@pytest.mark.slow
def test_dpu_column_1e5(tmp_path):
    _run("dpu_column", {"M": 16, "GUARD": GUARD}, 100_000, tmp_path, seed=8)
