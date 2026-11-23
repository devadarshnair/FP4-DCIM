"""T5.4: D2 INT4-SM bit-serial macro (bs4_macro_top; LOSSY surrogate, never INT5/exact) vs the golden L2 INT4-SM
column model (BitSerialInt4Column) through the shared macro harness (MX stimulus, codes read as {sign, mag[2:0]})."""

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
        + [REPO_ROOT / "rtl/baselines/bitserial_int4sm" / f for f in ("bs4_column.sv", "bs4_macro_top.sv")]
        + [SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel="bs4_macro_top",
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters={"M": m, "GUARD": GUARD, "STORE": store},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="bs4_macro_top",
        test_module="test_bs4_macro",
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
            "MT_EXPECT_MIN_GAP": "3",  # INT4-SM: 3 magnitude cycles/op back-to-back
            "MT_ARITH": "int4sm",
            **({"MT_PATTERN": pattern} if pattern else {}),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)


@needs
@pytest.mark.parametrize("m", [1, 16])
def test_bs4_macro(m, tmp_path):
    _run(m, tmp_path)


@needs
def test_bs4_macro_capture(tmp_path):
    _run(16, tmp_path, src="capture", tokens=24, seed=3)


@needs
@pytest.mark.slow
def test_bs4_macro_1e5(tmp_path):
    _run(16, tmp_path, tokens=400, seed=9)  # 400 tokens x 16 blocks x 32 cols ≈ 2e5 column-ops


@needs
@pytest.mark.parametrize("store", [0, 1], ids=["SRAM", "LATCH"])
@pytest.mark.parametrize("m", [4, 16])
def test_bs4_macro_pattern(store, m, tmp_path):
    """T1.6b: mixed same-set runs + switches (random clear/e_x/max_e_x/NaN); min same-set gap == 3."""
    from test_macro_top import MIXED_PATTERN

    _run(m, tmp_path, tokens=8, seed=11, store=store, pattern=MIXED_PATTERN)


@needs
def test_bs4_macro_power_harness(tmp_path):
    """The e21 GLS harness (test_macro_power, MP_FMT=int4sm) on the RTL with an int4sm_programs.py program from
    synthetic fp32 tensors: same checker path as the e21 bs4_macro_mx netlist GLS."""
    pytest.importorskip("cocotb")
    import importlib.util

    import numpy as np
    from cocotb_tools.runner import get_results, get_runner

    from cimfp4.stimulus.writers import save_program

    spec = importlib.util.spec_from_file_location("i4p", REPO_ROOT / "experiments/e21_macro_power/int4sm_programs.py")
    i4p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i4p)
    rng = np.random.default_rng(5)
    wsel = rng.standard_normal((16, 32, 32)) * np.exp(rng.normal(0, 0.5, (16, 32, 1)))
    asel = rng.standard_t(3, (2, 24, 32)) * np.exp(rng.normal(0, 1, (2, 24, 1)))
    prog = save_program(tmp_path / "int4sm_syn0.npz", i4p.int4sm_macro_program(wsel, asel))
    srcs = (
        [REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv"]
        + [REPO_ROOT / "rtl/src" / f for f in ("cim_icg.sv", "wset_sram.sv", "wset_latch.sv")]
        + [REPO_ROOT / "rtl/baselines/bitserial_int4sm" / f for f in ("bs4_column.sv", "bs4_macro_top.sv")]
        + [SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(sources=srcs, hdl_toplevel="bs4_macro_top", build_dir=tmp_path / "b", always=True,
                 includes=[REPO_ROOT / "rtl/include"], parameters={"M": 16, "GUARD": GUARD, "STORE": 0},
                 build_args=["--timing", "-Wno-fatal"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel="bs4_macro_top", test_module="test_macro_power", test_dir=Path(__file__).parent,
                      build_dir=tmp_path / "b", results_xml=str(tmp_path / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
                                 "MP_PROGRAM": str(prog), "MP_FMT": "int4sm", "MP_CLK_PS": "25000"})
    assert get_results(xml) == (1, 0)
