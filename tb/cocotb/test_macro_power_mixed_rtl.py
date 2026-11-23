"""T4.2 mixed-format stream on the RTL macros: the e21 macro power harness (test_macro_power.py, MP_FMT=mixed) drives
one interleaved MX/NV program (writers.macro_program_mixed) through macro_top and dpu_macro at NV_GATE 0 and 1 and
must match the per-op MX/NV references bit-exactly (the same check e21 applies to the gate-level netlists).

  pytest tb/cocotb/test_macro_power_mixed_rtl.py   (Verilator; light, ~1-2 min per build)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_nv_gate as ng  # noqa: E402

from cimfp4.stimulus.writers import macro_program_mixed, macro_program_random, save_program  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402


@ng.needs
@pytest.mark.parametrize("gate", [0, 1])
@pytest.mark.parametrize("top", ["macro_top", "dpu_macro"])
def test_mixed_stream_rtl(top, gate, tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    rng = np.random.default_rng(11)
    mx = macro_program_random(rng, "mxfp4", m=16, sets_used=2, reuse=6)
    nv = macro_program_random(rng, "nvfp4", m=16, sets_used=2, reuse=6)
    prog = save_program(tmp_path / "mixed.npz", macro_program_mixed(mx, nv, chunk=4))
    if os.environ.get("T43_MIXED_PROGRAM"):  # local smoke: a real e21 mixed window (e.g. cada632 real0, 1024 ops)
        prog = Path(os.environ["T43_MIXED_PROGRAM"])
    params = {"M": ng.M, "GUARD": 7, "NV_GATE": gate} | ({"STORE": 0} if top != "dpu_macro" else {})
    runner = get_runner("verilator")
    bd = tmp_path / "b"
    runner.build(sources=ng._sources(), hdl_toplevel=top, build_dir=bd, always=True,
                 includes=[REPO_ROOT / "rtl/include"], parameters=params,
                 build_args=["--timing", "-Wno-fatal"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel=top, test_module="test_macro_power", test_dir=Path(__file__).parent, build_dir=bd,
                      results_xml=str(tmp_path / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
                                 "MP_PROGRAM": str(prog), "MP_FMT": "mixed", "MP_CLK_PS": "25000"})
    assert get_results(xml) == (1, 0)
