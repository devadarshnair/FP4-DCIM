"""MXFP6 (E3M2) quantizer (MIDAS-protocol rows): grid, RNE ties, saturation, scale rules, torch == numpy."""

import numpy as np
import pytest

from cimfp4.formats.integerize import FORMATS
from cimfp4.formats.mx_quant import quantize_mxfp6_values, rne_e3m2

GRID = np.array([float(m) for m in FORMATS["E3M2"].magnitudes()])


def test_grid_matches_minifloat_definition():
    assert GRID.max() == 28.0 and GRID[1] == 1 / 16 and len(GRID) == 32
    assert np.array_equal(rne_e3m2(GRID), GRID)
    assert np.array_equal(rne_e3m2(-GRID), -GRID)


def test_rne_ties_and_saturation():
    mids = (GRID[:-1] + GRID[1:]) / 2
    q = rne_e3m2(mids)
    lo, hi = GRID[:-1], GRID[1:]
    # tie goes to the neighbour with even mantissa code: index parity in the sorted grid
    idx_lo = np.arange(len(lo))
    expect = np.where(idx_lo % 2 == 0, lo, hi)
    assert np.array_equal(q, expect)
    assert rne_e3m2(np.array([29.0, 1e30, -100.0])).tolist() == [28.0, 28.0, -28.0]


def test_nearest_on_random_values():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 30, 100_000)
    q = rne_e3m2(a)
    d = np.abs(a[:, None] - GRID[None, :]).min(1)
    assert np.allclose(np.abs(q - a), d, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rule,thr", [("ocp_v11", 1.875), ("rceil", 1.75)])
def test_scale_rule_boundaries(rule, thr):
    eps = 2.0**-20
    for frac, bumped in ((thr - eps, False), (thr if rule == "ocp_v11" else thr + eps, True)):
        x = np.zeros(32)
        x[0] = frac * 8.0  # e = 3
        q = quantize_mxfp6_values(x, rule)
        scale = 2.0 ** (3 - 4 + int(bumped))
        assert q[0] == rne_e3m2(np.array([x[0] / scale]))[0] * scale, (rule, frac)
    # v1.1 never saturates the block max more than half an ULP
    rng = np.random.default_rng(1)
    v = rng.standard_normal((2000, 32)) * np.exp(rng.uniform(-8, 8, (2000, 1)))
    amax = np.abs(v).max(1)
    qmax = np.abs(quantize_mxfp6_values(v, "ocp_v11")).max(1)
    assert np.all(np.abs(qmax - amax) <= amax * 2.0**-3 + 1e-300)


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from cimfp4.kernels.torch_quant import fq_mxfp6

    rng = np.random.default_rng(2)
    v = (rng.standard_normal((4000, 64)) * np.exp(rng.uniform(-10, 10, (4000, 1)))).astype(np.float32)
    v[0, :32] = 0.0
    for rule in ("ocp_v11", "rceil", "ocp_floor"):
        ref = quantize_mxfp6_values(v.astype(np.float64), rule)
        got = fq_mxfp6(torch.from_numpy(v).double(), rule).numpy()
        assert np.array_equal(got, ref), rule
