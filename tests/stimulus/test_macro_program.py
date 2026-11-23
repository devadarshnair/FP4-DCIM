"""Macro-level power programs (writers.macro_program_*) and the macro power harness reference."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.stimulus.writers import load_program, macro_program_from_capture, macro_program_random, save_program
from cimfp4.utils.provenance import REPO_ROOT


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_macro_program_shapes_and_anchors(fmt, tmp_path):
    rng = np.random.default_rng(0)
    quant = quantize_mxfp4 if fmt == "mxfp4" else quantize_nvfp4
    qw = quant(rng.standard_normal((64, 32 * 20)))
    qa = quant(rng.standard_t(3, (40, 32 * 20)))
    p = macro_program_from_capture(qw, qa, rng, m=16, sets_used=2, reuse=8)
    assert p["sets"].shape == (16, 32, 32) and p["x"].shape == (16, 32)
    assert set(p["set"].tolist()) == {0, 1} and p["clear"].all()
    # anchors never below the block exponents (the RTL saturating shift assumes d >= 0)
    assert (p["maxw"][None, :] >= np.maximum(p["e0"], p["e1"] if fmt == "nvfp4" else p["e0"])).all()
    assert (p["max_e_x"] >= p["e_x0"]).all()
    if fmt == "nvfp4":
        assert (p["max_e_x"] >= p["e_x1"]).all() and p["u0"].max() <= 15
    q = load_program(save_program(tmp_path / "p.npz", p))
    assert str(q["fmt"]) == fmt and np.array_equal(q["x"], p["x"])


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_macro_harness_reference_runs(fmt):
    sys.path.insert(0, str(REPO_ROOT / "tb/cocotb"))
    from test_macro_power import expected

    p = macro_program_random(np.random.default_rng(1), fmt, m=16, sets_used=2, reuse=4)
    e = expected(p, fmt)
    assert e.shape == (8, 32) and np.abs(e).max() < 2**39


def test_macro_program_mixed_merges_paired_streams():
    """T4.2: the mixed stream keeps every op of both programs (own order), NV sets moved to S..2S-1, op_nv flags,
    maxw covers both formats, and the harness reference of each op equals its own-format reference."""
    sys.path.insert(0, str(REPO_ROOT / "tb/cocotb"))
    from test_macro_power import expected

    from cimfp4.stimulus.writers import macro_program_mixed

    rng = np.random.default_rng(3)
    mx = macro_program_random(rng, "mxfp4", m=16, sets_used=2, reuse=10)
    nv = macro_program_random(rng, "nvfp4", m=16, sets_used=2, reuse=10)
    p = macro_program_mixed(mx, nv, chunk=3)
    n = len(mx["set"]) + len(nv["set"])
    assert p["x"].shape == (n, 32) and p["op_nv"].sum() == len(nv["set"]) and p["clear"].all()
    assert np.array_equal(p["x"][~p["op_nv"]], mx["x"]) and np.array_equal(p["x"][p["op_nv"]], nv["x"])
    assert np.array_equal(p["set"][~p["op_nv"]], mx["set"]) and np.array_equal(p["set"][p["op_nv"]], nv["set"] + 2)
    assert np.array_equal(p["sets"][2:4], nv["sets"][:2]) and np.array_equal(p["u0"][2:4], nv["u0"][:2])
    assert (p["maxw"] >= mx["maxw"]).all() and (p["maxw"] >= nv["maxw"]).all()
    assert p["op_nv"][:3].sum() == 0 and p["op_nv"][3:6].all()  # chunked interleave
    e = expected(p, "mixed")
    mx2, nv2 = {**mx, "maxw": p["maxw"]}, {**nv, "maxw": p["maxw"], "sets": p["sets"], "set": nv["set"] + 2,
                                              **{k: p[k] for k in ("e0", "e1", "u0", "u1")}}
    assert np.array_equal(e[~p["op_nv"]], expected(mx2, "mxfp4"))
    assert np.array_equal(e[p["op_nv"]], expected(nv2, "nvfp4"))


def test_macro_program_mixed_rejects_too_many_sets():
    from cimfp4.stimulus.writers import macro_program_mixed

    rng = np.random.default_rng(4)
    mx = macro_program_random(rng, "mxfp4", m=16, sets_used=9, reuse=2)
    nv = macro_program_random(rng, "nvfp4", m=16, sets_used=9, reuse=2)
    with pytest.raises(ValueError):
        macro_program_mixed(mx, nv)
