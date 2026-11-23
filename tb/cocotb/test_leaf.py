"""Exhaustive 256-case simulation of row_driver + leaf_mult (leaf_unit) vs the product LUT (WP-25)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cimfp4.formats.tables import PRODUCT_X4
from cimfp4.utils.provenance import REPO_ROOT

try:
    import cocotb
    from cocotb.triggers import Timer

    @cocotb.test()
    async def leaf_exhaustive(dut):
        bad = []
        for w in range(16):
            for x in range(16):
                dut.w.value, dut.x.value = w, x
                await Timer(1, unit="ns")
                p = int(PRODUCT_X4[w, x])
                mag, s, oc = int(dut.mag.value), int(dut.s.value), int(dut.oc.value)
                want_oc = ((~abs(p)) & 0x1FF) if p < 0 else abs(p)
                if (mag, s, oc) != (abs(p), int(p < 0), want_oc):
                    bad.append((w, x, mag, s, oc, p))
        assert not bad, bad[:8]

except ImportError:
    pass


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_leaf_rtl_exhaustive(tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    src = [REPO_ROOT / "rtl" / "src" / f for f in ("row_driver.sv", "leaf_mult.sv", "leaf_unit.sv")]
    runner = get_runner("verilator")
    runner.build(sources=src, hdl_toplevel="leaf_unit", build_dir=tmp_path / "b", always=True)
    xml = runner.test(
        hdl_toplevel="leaf_unit",
        test_module="test_leaf",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert get_results(xml) == (1, 0)


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not on PATH")
def test_leaf_formal_equivalence():
    import subprocess

    r = subprocess.run(
        ["yosys", "-q", "tb/formal/leaf_dec_equiv.ys"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
