"""Post-synthesis gate-level simulation (zero-delay, sky130 cell models) vs the scoreboard (WP-27).

Runs the column-group cocotb test on the mapped netlists from flows/synth/sweep.py. Simulator:
Icarus (sky130 cell models use UDP primitives). Env: GLS_NOPS (default 10^4).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_column_top import SRAM_V  # noqa: E402

from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

try:
    from test_column_top import column_top_matches  # noqa: F401
except ImportError:
    pass

CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
SYNTH = REPO_ROOT / "build" / "synth"
SRAM_GLS = REPO_ROOT / "build/models/sky130_sram_1kbyte_1rw1r_32x256_8_gls.v"  # scripts/patch_sram_model.py
DESIGNS = ["v2_sram_m16", "bs_int5", "dpu", "v2_generic", "v2_latch_m1"]


def _run(design: str, n_ops: int, tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    meta = {r["design"]: r for r in json.loads((SYNTH / "synth.json").read_text())}[design]
    if not SRAM_GLS.exists():
        import subprocess

        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/patch_sram_model.py")], check=True)
    srcs = [CELLS / "primitives.v", CELLS / "sky130_fd_sc_hd.v", SYNTH / f"{design}_netlist.v", SRAM_GLS]
    runner = get_runner("icarus")
    runner.build(
        sources=srcs,
        hdl_toplevel=meta["top"],
        build_dir=tmp_path / "b",
        always=True,
        defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0"},
        timescale=("1ns", "1ps"),
    )
    env = {
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
        "CT_M": str(meta["M"]),
        "CT_NOPS": str(n_ops),
        "CT_SEED": "21",
        "CT_CAPTURE": "0",
    }
    xml = runner.test(
        hdl_toplevel=meta["top"],
        test_module="test_gls",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env=env,
    )
    assert get_results(xml) == (1, 0)


@pytest.mark.skipif(
    shutil.which("iverilog") is None or not (SYNTH / "synth.json").exists() or not SRAM_V.exists(),
    reason="needs iverilog, synthesized netlists (flows/synth/sweep.py) and OpenRAM models",
)
@pytest.mark.slow
@pytest.mark.parametrize("design", DESIGNS)
def test_gls(design, tmp_path):
    _run(design, int(os.environ.get("GLS_NOPS", "10000")), tmp_path)
