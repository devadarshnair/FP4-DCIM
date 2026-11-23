"""MIDAS-like L2 model (WP-37): approximation off == L1; cycle/activity semantics of STCA + TAOR + MEPD."""

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.golden.accum import Exact, accumulate
from cimfp4.golden.l1_dot import INT_BITS, block_exponents, block_terms, l1_dot
from cimfp4.golden.l2_midas_like import (
    FORMATS,
    MidasLikeMacro,
    block_integer_reference,
    decode,
    element_values,
)


def test_decode_matches_formats():
    from cimfp4.formats.codecs import e2m1_to_int_x2
    from cimfp4.formats.integerize import FORMATS as MF

    c4 = np.arange(16)
    assert np.array_equal(element_values(c4, "mxfp4") * 2, e2m1_to_int_x2(c4))
    mags = sorted({abs(v) for v in element_values(np.arange(64), "mxfp6")})
    assert mags == [float(m) for m in MF["E3M2"].magnitudes()]


@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp6"])
def test_random_codes_exact(fmt):
    rng = np.random.default_rng(0)
    nb = FORMATS[fmt]["code_bits"]
    w = rng.integers(0, 1 << nb, (16, 32))
    x = rng.integers(0, 1 << nb, (3000, 32))
    p, act = MidasLikeMacro(fmt=fmt).run(w, x)
    assert np.array_equal(p, block_integer_reference(w, x, fmt))
    m = MidasLikeMacro(fmt=fmt)
    assert (act["cycles"] == m.top - act["c_end"] + 1).all() and (act["cycles"] >= 1).all()


def test_equals_l1_block_terms_and_dot():
    rng = np.random.default_rng(1)
    k = 256
    w = rng.standard_normal((16, k)) * np.exp(rng.uniform(-3, 3, (16, 1)))
    x = rng.standard_normal((40, k)) * np.exp(rng.uniform(-3, 3, (40, 1)))
    qw, qx = quantize_mxfp4(w), quantize_mxfp4(x)
    P_l1, E, anchor = block_terms(qw, qx)  # (N=16, M=40, nb)
    wb, xb = qw.blocked_codes(), qx.blocked_codes()  # (16, nb, 32), (40, nb, 32)
    macro = MidasLikeMacro(fmt="mxfp4")
    P_ml = np.stack([macro.run(wb[:, b], xb[:, b])[0].T for b in range(k // 32)], -1)  # (16, 40, nb)
    assert np.array_equal(P_ml, P_l1)
    ew, ex = block_exponents(qw, qx)
    y_ml = accumulate(P_ml, E, Exact(), anchor_exp=anchor, int_bits=INT_BITS["mxfp4"], ew=ew, ex=ex)
    assert np.array_equal(y_ml, l1_dot(qw, qx, Exact()))


def test_early_stop_and_gating_counts():
    m = MidasLikeMacro(fmt="mxfp4")
    top = m.top
    assert top == 7
    # all elements at the max exponent (codes 0b0110/0b0111: E=3): only esum=6 -> TAO ∈ {0,1} -> 2 cycles
    w = np.full((16, 32), 0b0111)
    x = np.full((1, 32), 0b0110)
    p, act = m.run(w, x)
    assert act["cycles"][0] == 2 and act["c_end"][0] == 6
    assert act["valid_lca_cycles"][0] == 16 * 32 * 2  # every LCA valid for mbit+1 cycles
    assert p[0, 0] == 32 * 3 * 2 * 2 ** (6 - 2)  # Σ M_x·M_w·2^(esum-2) = 32·(2·3)·16
    # one small element anywhere stretches the macro to its esum (MEPD = min over the macro)
    x2 = x.copy()
    x2[0, 5] = 0b0011  # E=1, m=1
    _, act2 = m.run(w, x2)
    assert act2["c_end"][0] == 4 and act2["cycles"][0] == 4
    # without MEPD the full range runs every op
    _, act3 = MidasLikeMacro(fmt="mxfp4", early_stop=False).run(w, x)
    assert act3["cycles"][0] == top - 2 + 1
    # zeros do not hold the early stop by default
    xz = x.copy()
    xz[0, 0] = 0
    _, actz = m.run(w, xz)
    assert actz["c_end"][0] == 6
    _, actzh = MidasLikeMacro(fmt="mxfp4", zero_holds_early_stop=True).run(w, xz)
    assert actzh["c_end"][0] == 4  # zero decodes as E_eff=1 -> esum 4


def test_each_lca_valid_for_mbit_plus_one_cycles():
    rng = np.random.default_rng(3)
    for fmt in ("mxfp4", "mxfp6"):
        nb, mb = FORMATS[fmt]["code_bits"], FORMATS[fmt]["mbits"]
        w = rng.integers(0, 1 << nb, (16, 32))
        x = rng.integers(0, 1 << nb, (200, 32))
        _, act = MidasLikeMacro(fmt=fmt, zero_holds_early_stop=True).run(w, x)
        assert (act["valid_lca_cycles"] == 16 * 32 * (mb + 1)).all()
        assert (act["valid_lca_cycles"] <= act["lca_cycles_total"]).all()
        s, mant, e = decode(np.arange(1 << nb), fmt)
        assert mant.max() == (1 << (mb + 1)) - 1 and e.min() == 1
