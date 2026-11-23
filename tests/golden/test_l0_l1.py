from fractions import Fraction

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.formats.types import FP4Block
from cimfp4.golden.accum import (
    AlignTrunc,
    Approx,
    Exact,
    Fixed,
    FP32Seq,
    Window,
    accumulate,
    round_int_to_f32,
)
from cimfp4.golden.l0_ref import l0_dot, spread_ok
from cimfp4.golden.l1_dot import block_terms, l1_dot


def llm_like(rng, n, k, row_spread=2.0):
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, row_spread, (n, 1)))
    x[:, rng.integers(0, k, max(1, k // 512))] *= 30.0  # outlier channels
    return x.astype(np.float32)


def as_f32_bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


# ------------------------------------------------------------------ exact rounding helper
def test_round_int_to_f32_matches_fraction_reference():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(-(2**62), 2**62)) >> int(rng.integers(0, 60))
        e = int(rng.integers(-180, 100))
        exact = Fraction(n) * Fraction(2) ** e
        got = round_int_to_f32(n, e)
        # reference: correctly rounded via fractions -> nearest fp32 by checking neighbours
        if abs(exact) > Fraction(float(np.finfo(np.float32).max)):
            continue
        f = np.float32(float(exact))  # candidate (may be double-rounded); fix up by neighbour search
        cands = [np.nextafter(f, np.float32(-np.inf)), f, np.nextafter(f, np.float32(np.inf))]
        cands = [c for c in cands if np.isfinite(c)]
        best = min(cands, key=lambda c: (abs(Fraction(float(c)) - exact), int(c.view(np.uint32)) & 1))
        assert got == best, (n, e)


def test_round_int_to_f32_ties_to_even_and_overflow():
    assert round_int_to_f32((1 << 24) + 1, 0) == np.float32(2**24)  # tie -> even
    assert round_int_to_f32((1 << 24) + 3, 0) == np.float32(2**24 + 4)
    assert round_int_to_f32(1, 128) == np.float32(np.inf)
    assert round_int_to_f32(1, -150) == np.float32(0.0)  # half of min subnormal, tie -> even (0)
    assert round_int_to_f32(3, -151) == np.float32(2.0**-149)


# ------------------------------------------------------------------ L1(EXACT) == round_fp32(L0)
@pytest.mark.slow
@pytest.mark.parametrize("k", [32, 256, 4096, 8192])
def test_mx_l1_exact_equals_l0_rounded(k):
    rng = np.random.default_rng(k)
    n = m = 160  # 4 × 25,600 = 102,400 dot products across the parametrization
    qw = quantize_mxfp4(llm_like(rng, n, k), "ocp_v11")
    qx = quantize_mxfp4(llm_like(rng, m, k), "ocp_v11")
    assert spread_ok(qw, qx).all()
    y1 = l1_dot(qw, qx, Exact())
    y0 = l0_dot(qw, qx).astype(np.float32)
    assert np.array_equal(as_f32_bits(y1), as_f32_bits(y0))


@pytest.mark.slow
@pytest.mark.parametrize("k", [16, 512, 8192])
def test_nv_l1_exact_vs_l0(k):
    rng = np.random.default_rng(100 + k)
    xw, xx = llm_like(rng, 96, k), llm_like(rng, 96, k)
    # (a) power-of-two tensor scales -> bit-exact
    pw = 2688.0 * 2.0 ** np.ceil(np.log2(np.abs(xw).max() / 2688.0))
    px = 2688.0 * 2.0 ** np.ceil(np.log2(np.abs(xx).max() / 2688.0))
    qw, qx = quantize_nvfp4(xw, tensor_amax=pw), quantize_nvfp4(xx, tensor_amax=px)
    assert np.array_equal(as_f32_bits(l1_dot(qw, qx)), as_f32_bits(l0_dot(qw, qx).astype(np.float32)))
    # (b) arbitrary fp32 tensor scales -> within 1 fp32 ulp (L0 rounds twice in float64)
    qw, qx = quantize_nvfp4(xw), quantize_nvfp4(xx)
    y1, y0 = l1_dot(qw, qx), l0_dot(qw, qx).astype(np.float32)
    ulp = np.abs(np.spacing(y0))
    assert (np.abs(y1 - y0) <= ulp).all()


def test_l0_falls_back_when_spread_too_large():
    # two blocks whose scale exponents differ by 100 -> float64 path not exact
    codes = np.full((1, 64), 0b0111, np.uint8)  # +6 everywhere
    qw = FP4Block(
        codes=codes, scale_exp=np.array([[100, -100]], np.int16), fmt="mxfp4", block=32, scale_rule="test"
    )
    qx = FP4Block(
        codes=codes.copy(), scale_exp=np.array([[0, 0]], np.int16), fmt="mxfp4", block=32, scale_rule="test"
    )
    assert not spread_ok(qw, qx).all()
    exact = Fraction(32 * 36) * (Fraction(2) ** 100 + Fraction(2) ** -100)
    assert l0_dot(qw, qx)[0, 0] == float(exact)
    assert l1_dot(qw, qx)[0, 0] == np.float32(float(exact))


# ------------------------------------------------------------------ accumulation modes
@pytest.fixture(scope="module")
def mx_terms():
    rng = np.random.default_rng(7)
    k = 4096
    qw = quantize_mxfp4(llm_like(rng, 48, k, 0.5), "ocp_v11")
    qx = quantize_mxfp4(llm_like(rng, 48, k, 2.0), "ocp_v11")
    return qw, qx, l1_dot(qw, qx, Exact())


def rel_err(y, ref):
    return np.abs(y.astype(np.float64) - ref) / np.maximum(np.abs(ref), 1e-30)


def test_fp32_seq_close_to_exact(mx_terms):
    qw, qx, ref = mx_terms
    assert np.median(rel_err(l1_dot(qw, qx, FP32Seq()), ref)) < 1e-5


def test_window40_is_exact_on_llm_like_data(mx_terms):
    # a 40-bit window anchored at max e_w + max e_x is exact on simulated data
    qw, qx, ref = mx_terms
    y = l1_dot(qw, qx, Window(40))
    assert np.array_equal(as_f32_bits(y), as_f32_bits(ref))


def test_narrow_window_loses_precision_but_never_overflows(mx_terms):
    qw, qx, ref = mx_terms
    y = l1_dot(qw, qx, Window(20))  # raises if the anchor analysis allowed an overflow
    assert rel_err(y, ref).max() > 0


def test_blueprint_fixed24_has_large_error(mx_terms):
    # blueprint: 24-bit register after the shared-exponent shift. Error must be > 0; the
    # report measured up to 0.6 relative - we assert only that it is clearly worse than FP32.
    qw, qx, ref = mx_terms
    y = l1_dot(qw, qx, Fixed(24, frac=8))
    assert rel_err(y, ref).max() > 1e-2


def test_align_trunc_close(mx_terms):
    qw, qx, ref = mx_terms
    assert rel_err(l1_dot(qw, qx, AlignTrunc(35)), ref).max() < 1e-6


def test_approx_hook_receives_terms(mx_terms):
    qw, qx, ref = mx_terms
    hook = Approx(lambda P, E: np.sum(np.ldexp(P.astype(np.float64), E.astype(np.int32)), axis=-1))
    assert np.allclose(l1_dot(qw, qx, hook), ref, rtol=1e-6)


def test_fixed_wraps_or_saturates():
    P, E = np.array([[4608, 4608]]), np.array([[0, 0]])
    assert accumulate(P, E, Fixed(14, 0, saturate=True))[0] == 8191
    assert accumulate(P, E, Fixed(14, 0, saturate=False))[0] == 9216 - 16384


def test_block_terms_shapes_and_ranges(mx_terms):
    qw, qx, _ = mx_terms
    P, E, anchor = block_terms(qw, qx)
    assert P.shape == (48, 48, 128) and np.abs(P).max() <= 4608
    assert (E <= anchor[..., None]).all()


def test_operand_checks():
    rng = np.random.default_rng(0)
    a = quantize_mxfp4(rng.standard_normal((2, 64)))
    b = quantize_nvfp4(rng.standard_normal((2, 64)))
    with pytest.raises(ValueError):
        l1_dot(a, b)
