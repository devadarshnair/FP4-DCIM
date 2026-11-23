"""Baseline (c) DPU macro (WP-72) vs the per-column references (MX and NV), sharing the macro harness."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_macro_top import GUARD, SRAM_V  # noqa: E402

from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

try:
    from test_macro_top import macro_matches  # noqa: F401
except ImportError:
    pass


def _run(fmt, m, tmp_path, src="random", tokens=48, seed=0):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        [REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv"]
        + [
            REPO_ROOT / "rtl/src" / f
            for f in (
                "cim_icg.sv",
                "row_driver.sv",
                "leaf_mult.sv",
                "column.sv",
                "wset_sram.sv",
                "wset_latch.sv",
                "macro_top.sv",
            )
        ]
        + [REPO_ROOT / "rtl/baselines/digital_mac" / f for f in ("generic_leaf.sv", "dpu_macro.sv")]
        + [SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel="dpu_macro",
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters={"M": m, "GUARD": GUARD},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="dpu_macro",
        test_module="test_dpu_macro",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "MT_FMT": fmt,
            "MT_M": str(m),
            "MT_SRC": src,
            "MT_TOKENS": str(tokens),
            "MT_SEED": str(seed),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)


@needs
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_dpu_macro(fmt, tmp_path):
    _run(fmt, 16, tmp_path)


@needs
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_dpu_macro_capture(fmt, tmp_path):
    _run(fmt, 16, tmp_path, src="capture", tokens=24, seed=4)


@needs
@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_dpu_macro_1e5(fmt, tmp_path):
    # WP-72 acceptance: 400 tokens x 16 blocks x 32 columns = 2.05e5 column-ops per format
    # (6400 macro ops), bit-exact vs the L1-equivalent per-column reference, MX and NV.
    _run(fmt, 16, tmp_path, tokens=400, seed=9)
