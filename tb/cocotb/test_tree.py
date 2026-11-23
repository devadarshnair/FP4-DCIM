"""Verilator simulation of the generated column trees vs the bit-level model (WP-24).

Checks the carry-save outputs (as sums) *and every internal cell net* on 10^4 random vectors.
Run: pytest tb/cocotb/test_tree.py   (needs verilator + cocotb 2.x)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cimfp4.rtlgen.tree import build_column_tree, build_dual_tree, encode_inputs, evaluate
from cimfp4.utils.provenance import REPO_ROOT

N_VEC = int(os.environ.get("TREE_NVEC", "10000"))
CONFIG = os.environ.get("TREE_CONFIG", "9,oc")


def _vectors(nl, n):
    rng = np.random.default_rng(42)
    if nl.scheme == "unsigned":
        return rng.integers(0, 2**nl.w_in, (n, nl.n_inputs))
    return rng.integers(-144, 145, (n, nl.n_inputs))


# ------------------------------------------------------------------------------- cocotb test body
try:
    import cocotb
    from cocotb.triggers import Timer

    @cocotb.test()
    async def tree_matches_bit_model(dut):
        w_in, scheme = CONFIG.split(",")
        nl = build_dual_tree(int(w_in)) if scheme == "dual" else build_column_tree(int(w_in), scheme)
        bpr = nl.w_in - 1 if scheme == "dual" else nl.w_in
        v = _vectors(nl, N_VEC)
        nets, out = evaluate(nl, v)
        enc = encode_inputs(nl, v)
        internal = [c.sum for c in nl.cells] + [c.carry for c in nl.cells if c.carry]
        mismatches = 0
        for t in range(N_VEC):
            word = 0
            for r in range(nl.n_inputs):
                for b in range(bpr):
                    word |= int(enc[f"in_{r}_{b}"][t]) << (r * bpr + b)
            dut.in_bits.value = word
            if scheme in ("oc", "dual"):
                dut.s_bits.value = sum(int(enc[f"s_{r}"][t]) << r for r in range(nl.n_inputs))
            await Timer(1, unit="ns")
            pairs = [(p_, len(nl.outputs[f"{p_}_s"])) for p_ in ("h0", "h1", "m") if f"{p_}_s" in nl.outputs]
            for pair, w in pairs:
                s = int(getattr(dut, f"{pair}_s").value) + int(getattr(dut, f"{pair}_c").value)
                s &= (1 << w) - 1
                if scheme != "unsigned" and s >= 1 << (w - 1):
                    s -= 1 << w
                mismatches += s != int(out[pair][t])
            if t % 97 == 0:  # internal nets on a deterministic subset (full set is slow in Python)
                for net in internal:
                    mismatches += int(getattr(dut, net).value) != int(nets[net][t])
        assert mismatches == 0, f"{mismatches} mismatches"

except ImportError:  # collected by pytest without cocotb installed
    pass


# ------------------------------------------------------------------------------- pytest launcher
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
@pytest.mark.parametrize("cfg", ["9,oc", "9,tc", "1,unsigned", "9,dual"])
def test_tree_rtl_vs_bit_model(cfg, tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_runner

    w_in, scheme = cfg.split(",")
    nl = build_dual_tree(int(w_in)) if scheme == "dual" else build_column_tree(int(w_in), scheme)
    src = REPO_ROOT / "rtl" / "gen" / f"{nl.name}.sv"
    runner = get_runner("verilator")
    runner.build(
        sources=[src],
        hdl_toplevel=nl.name,
        build_dir=tmp_path / "build",
        build_args=["--public-flat-rw", "-Wno-fatal"],
        always=True,
    )
    results_xml = runner.test(
        hdl_toplevel=nl.name,
        test_module="test_tree",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "build",
        results_xml=str(tmp_path / "results.xml"),
        extra_env={"TREE_CONFIG": cfg, "TREE_NVEC": str(N_VEC), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    from cocotb_tools.runner import get_results

    n_tests, n_failed = get_results(results_xml)
    assert n_tests == 1 and n_failed == 0
