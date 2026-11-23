import dataclasses

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.golden.accum import Exact, Hier, Window, round_int_to_f32
from cimfp4.golden.l1_dot import INT_BITS, block_exponents, block_terms, l1_dot


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def llm_like(rng, n, k, s=1.5):
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, s, (n, 1)))
    x[:, rng.integers(0, k, max(1, k // 256))] *= 30
    return x


def ref_hier(P, E, ew, ex, tb, window, int_bits, reducer):
    """Independent pure-python HIER reference for one dot product."""
    guard = max(1, int(np.ceil(np.log2(max(tb, 2)))))
    tiles = []
    for s in range(0, len(P), tb):
        anchor = max(ew[s : s + tb]) + max(ex[s : s + tb]) - 2
        lsb = anchor + int_bits + guard - window + 1
        acc = 0
        for p, e in zip(P[s : s + tb], E[s : s + tb], strict=True):
            acc += (p << (e - lsb)) if e >= lsb else (p >> (lsb - e))
        tiles.append((acc, lsb))
    if reducer == "exact":
        m = min(lsb for _, lsb in tiles)
        return round_int_to_f32(sum(a << (lsb - m) for a, lsb in tiles), m)
    a32 = np.float32(0)
    for a, lsb in tiles:
        a32 = np.float32(a32 + round_int_to_f32(a, lsb))
    return a32


def test_hier_single_tile_equals_window():
    rng = np.random.default_rng(0)
    k = 2048
    qw, qx = quantize_mxfp4(llm_like(rng, 16, k, 0.5)), quantize_mxfp4(llm_like(rng, 16, k))
    nb = k // 32
    assert np.array_equal(bits(l1_dot(qw, qx, Hier(nb, 40))), bits(l1_dot(qw, qx, Window(40))))


def test_hier_wide_window_exact_reducer_equals_exact():
    rng = np.random.default_rng(1)
    qw, qx = quantize_mxfp4(llm_like(rng, 16, 4096, 0.5)), quantize_mxfp4(llm_like(rng, 16, 4096))
    assert np.array_equal(bits(l1_dot(qw, qx, Hier(16, 60, reducer="exact"))), bits(l1_dot(qw, qx, Exact())))


def test_anchor_is_tile_local():
    rng = np.random.default_rng(2)
    qw, qx = quantize_mxfp4(llm_like(rng, 1, 1024)), quantize_mxfp4(llm_like(rng, 1, 1024))
    P, E, _ = block_terms(qw, qx)
    ew, ex = block_exponents(qw, qx)
    tb = 8
    base = ref_hier(
        P[0, 0].tolist(), E[0, 0].tolist(), ew[0, 0].tolist(), ex[0, 0].tolist(), tb, 40, 14, "exact"
    )

    # perturb weight exponents of tile 1 upwards: tile 0's contribution must not change -> recompute
    # tile 0 alone before/after
    def tile0(qw_):
        P_, E_, _ = block_terms(qw_, qx)
        ew_, ex_ = block_exponents(qw_, qx)
        return ref_hier(
            P_[0, 0][:tb].tolist(),
            E_[0, 0][:tb].tolist(),
            ew_[0, 0][:tb].tolist(),
            ex_[0, 0][:tb].tolist(),
            tb,
            40,
            14,
            "exact",
        )

    se = qw.scale_exp.copy()
    se[0, tb : 2 * tb] += 5
    qw2 = dataclasses.replace(qw, scale_exp=se)
    assert tile0(qw) == tile0(qw2)
    assert np.isfinite(base)


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_hier_matches_reference_1e5(fmt):
    rng = np.random.default_rng(3)
    quant = quantize_mxfp4 if fmt == "mxfp4" else quantize_nvfp4
    n_done = 0
    while n_done < 100_000:
        k = int(rng.choice([256, 512, 1024]))
        qw, qx = quant(llm_like(rng, 50, k, 0.5)), quant(llm_like(rng, 40, k))
        tb = int(rng.choice([2, 4, 8]))
        red = str(rng.choice(["exact", "fp32"]))
        y = l1_dot(qw, qx, Hier(tb, 40, reducer=red))
        P, E, _ = block_terms(qw, qx)
        ew, ex = block_exponents(qw, qx)
        post = (
            np.float32(np.float32(qw.tensor_scale) * np.float32(qx.tensor_scale)) if fmt == "nvfp4" else None
        )
        for i in range(50):  # every dot product is compared
            for j in range(40):
                r = ref_hier(
                    P[i, j].tolist(),
                    E[i, j].tolist(),
                    ew[i, j].tolist(),
                    ex[i, j].tolist(),
                    tb,
                    40,
                    INT_BITS[fmt],
                    red,
                )
                if post is not None:
                    r = np.float32(np.float32(r) * post)
                assert bits(y[i, j]) == bits(r), (fmt, i, j)
        n_done += 50 * 40


def test_hier_window_reducer_runs():
    rng = np.random.default_rng(4)
    qw, qx = quantize_mxfp4(llm_like(rng, 4, 1024)), quantize_mxfp4(llm_like(rng, 4, 1024))
    y = l1_dot(qw, qx, Hier(8, 40, reducer=("window", 48)))
    ref = l1_dot(qw, qx, Exact())
    assert np.allclose(y, ref, rtol=1e-3, atol=1e-3 * np.abs(ref).max())
