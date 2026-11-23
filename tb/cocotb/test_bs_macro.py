"""Baseline (a) bit-serial INT5 macro (WP-71) vs the per-column references, sharing the macro harness (MX)."""

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


def _run(m, tmp_path, src="random", tokens=48, seed=0, store=0, pattern=None):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        [REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv"]
        + [REPO_ROOT / "rtl/src" / f for f in ("cim_icg.sv", "wset_sram.sv", "wset_latch.sv")]
        + [REPO_ROOT / "rtl/baselines/bitserial_int5" / f for f in ("bs_column.sv", "bs_macro_top.sv")]
        + [SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel="bs_macro_top",
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters={"M": m, "GUARD": GUARD, "STORE": store},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="bs_macro_top",
        test_module="test_bs_macro",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "MT_FMT": "mxfp4",
            "MT_M": str(m),
            "MT_SRC": src,
            "MT_TOKENS": str(tokens),
            "MT_SEED": str(seed),
            "MT_EXPECT_MIN_GAP": "5",  # numerics.md baseline (a): 5 cycles/op back-to-back
            **({"MT_PATTERN": pattern} if pattern else {}),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)


@needs
@pytest.mark.parametrize("m", [1, 16])
def test_bs_macro(m, tmp_path):
    _run(m, tmp_path)


@needs
def test_bs_macro_capture(tmp_path):
    _run(16, tmp_path, src="capture", tokens=24, seed=3)


@needs
@pytest.mark.slow
def test_bs_macro_1e5(tmp_path):
    _run(16, tmp_path, tokens=400, seed=9)  # 400 tokens x 16 blocks x 32 cols ≈ 2e5 column-ops


@needs
@pytest.mark.parametrize("store", [0, 1], ids=["SRAM", "LATCH"])
@pytest.mark.parametrize("m", [4, 16])
def test_bs_macro_pattern(store, m, tmp_path):
    """T1.6b: mixed same-set runs + switches (random clear/e_x/max_e_x/NaN); min same-set gap == 5."""
    from test_macro_top import MIXED_PATTERN

    _run(m, tmp_path, tokens=8, seed=11, store=store, pattern=MIXED_PATTERN)
