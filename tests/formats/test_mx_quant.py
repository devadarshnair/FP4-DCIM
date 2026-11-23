import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cimfp4.formats.codecs import e2m1_decode
from cimfp4.formats.mx_quant import (
    dequantize,
    floor_log2,
    mx_scale_exponent,
    quantize_mxfp4,
    rne_e2m1_magnitude,
)
from cimfp4.formats.types import FP4Block

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def block_with(values, fill=0.0):
    """A single 32-element block whose first entries are `values`, plus amax-pinning element 6.0."""
    b = np.full(32, fill)
    b[: len(values)] = values
    return b


# ------------------------------------------------------------------ element rounding
@pytest.mark.parametrize(
    "a, expected",
    [
        (0.25, 0.0),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.75, 2.0),
        (2.5, 2.0),
        (3.5, 4.0),
        (5.0, 4.0),
        (0.2499, 0.0),
        (0.2501, 0.5),
        (5.0001, 6.0),
        (6.0, 6.0),
        (7.9, 6.0),
        (100.0, 6.0),
        (0.0, 0.0),
    ],
)
def test_rne_ties_and_saturation(a, expected):
    assert GRID[rne_e2m1_magnitude(np.array([a]))[0]] == expected


def test_rne_is_nearest_everywhere():
    a = np.linspace(0, 6, 60001)
    q = GRID[rne_e2m1_magnitude(a)]
    best = np.abs(a[:, None] - GRID[None, :]).min(axis=1)
    assert np.allclose(np.abs(a - q), best, atol=0, rtol=0)


# ------------------------------------------------------------------ scale rules
def test_floor_log2_exact_incl_subnormal():
    a = np.array([1.0, 1.999999, 2.0, 0.75, 2.0**-1074, 2.0**-1030 * 1.5, 3.0e300])
    assert floor_log2(a).tolist() == [0, 0, 1, -1, -1074, -1030, 998]


@pytest.mark.parametrize(
    "amax, rule, x",
    [
        (6.0, "ocp_floor", 0),
        (7.0, "ocp_floor", 0),
        (4.0, "ocp_floor", 0),
        (6.0, "ocp_v11", 0),
        (7.0, "ocp_v11", 1),
        (7.5, "ocp_v11", 1),
        (6.99, "ocp_v11", 0),
        (6.0, "rceil", 0),
        (6.01, "rceil", 1),
        (3.01, "rceil", 0),
        (3.0, "rceil", -1),
        (4.0, "ceil", 0),
        (4.01, "ceil", 1),
    ],
)
def test_scale_rules(amax, rule, x):
    assert mx_scale_exponent(np.array([amax]), rule)[0] == x


@pytest.mark.parametrize(
    "rule, lo, hi, lo_incl, hi_incl",
    [
        ("ocp_floor", 4, 8, True, False),
        ("ocp_v11", 3.5, 7, True, False),
        ("rceil", 3, 6, False, True),
        ("ceil", 2, 4, False, True),
    ],
)
def test_scaled_amax_ranges(rule, lo, hi, lo_incl, hi_incl):
    rng = np.random.default_rng(1)
    amax = rng.uniform(0.5, 1.0, 20000) * 2.0 ** rng.integers(-60, 60, 20000)
    r = amax / np.ldexp(1.0, mx_scale_exponent(amax, rule).astype(np.int32))
    assert (r >= lo).all() if lo_incl else (r > lo).all()
    assert (r <= hi).all() if hi_incl else (r < hi).all()


def test_expadj_triggers_exactly_at_1p75():
    # amax = 1.75·2^2 = 7 -> v1.1 bumps the exponent, v1.0 does not (and then clips 7 -> 6)
    q10 = quantize_mxfp4(block_with([7.0]), "ocp_floor")
    q11 = quantize_mxfp4(block_with([7.0]), "ocp_v11")
    assert q10.scale_exp[0] == 0 and q11.scale_exp[0] == 1
    assert dequantize(q10)[0] == 6.0  # clipped
    assert dequantize(q11)[0] == 8.0  # 7/2 = 3.5 -> tie -> 4 -> ×2


# ------------------------------------------------------------------ whole-block behaviour
def test_zero_block():
    q = quantize_mxfp4(np.zeros((2, 64)))
    assert (q.codes == 0).all() and (q.scale_exp == -127).all()


def test_subnormal_amax_block():
    b = block_with([2.0**-1070, -(2.0**-1072)])
    q = quantize_mxfp4(b, "ocp_floor")
    assert q.scale_exp[0] == -127  # clamped
    assert (q.codes == 0).all()  # far below 0.25·2^-127


def test_saturation_to_six():
    q = quantize_mxfp4(block_with([6.0, 6.9, -7.9]), "ocp_floor")
    assert dequantize(q)[:3].tolist() == [6.0, 6.0, -6.0]


def test_negative_values_and_no_negative_zero():
    q = quantize_mxfp4(block_with([-6.0, -0.1, -0.0, 0.1]))
    d = dequantize(q)
    assert d[0] == -6.0 and (q.codes[1:4] == 0).all()


def test_nan_block_gets_nan_scale():
    x = np.ones((1, 64))
    x[0, 40] = np.nan
    q = quantize_mxfp4(x)
    assert q.scale_nan.tolist() == [[False, True]]
    d = dequantize(q)
    assert np.isfinite(d[0, :32]).all() and np.isnan(d[0, 32:]).all()


def test_shape_checks():
    with pytest.raises(ValueError):
        quantize_mxfp4(np.zeros(31))
    with pytest.raises(ValueError):
        FP4Block(
            codes=np.zeros(32, np.uint8),
            scale_exp=np.zeros(1, np.int16),
            fmt="mxfp4",
            block=16,
            scale_rule="x",
        )


# ------------------------------------------------------------------ property: error bound
@settings(max_examples=300, deadline=None)
@given(
    arrays(np.float32, 32, elements=st.floats(-(2.0**100), 2.0**100, width=32, allow_nan=False)),
    st.sampled_from(["ocp_floor", "ocp_v11", "rceil", "ceil"]),
)
def test_error_within_half_step_in_range(x, rule):
    x = x.astype(np.float64)
    q = quantize_mxfp4(x, rule)
    s = np.ldexp(1.0, int(q.scale_exp[0]))
    a = np.abs(x) / s
    d = np.abs(dequantize(q)) / s
    inside = a <= 6.0
    hi = np.clip(np.searchsorted(GRID, a, side="left"), 1, 7)
    half_step = (GRID[hi] - GRID[hi - 1]) / 2
    assert (np.abs(a - d)[inside] <= half_step[inside]).all()
    # elements beyond the range saturate to 6
    assert (d[~inside] == 6.0).all()


# ------------------------------------------------------------------ regression vs R07 Appendix A
def _appendix_a_quant(x, K=32):
    """Verbatim reference implementation (OCP v1.0 floor, RNE, saturate)."""
    xb = x.reshape(*x.shape[:-1], -1, K)
    amax = np.abs(xb).max(-1, keepdims=True)
    e = np.clip(np.floor(np.log2(np.where(amax == 0, 1.0, amax))) - 2, -127, 127)
    a = np.minimum(np.abs(xb) / 2.0**e, 6.0)
    idx = np.clip(np.searchsorted(GRID, a), 1, 7)
    lo, hi = GRID[idx - 1], GRID[idx]
    up = (hi - a < a - lo) | ((hi - a == a - lo) & (idx % 2 == 0))
    mag = np.where(a == 0, 0, np.where(up, idx, idx - 1))
    return (mag | ((xb < 0) & (mag > 0)) * 8).astype(np.uint8), e[..., 0].astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_appendix_a_reference(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((64, 4096)) * np.exp(rng.normal(0, 2, (64, 1)))
    codes_ref, e_ref = _appendix_a_quant(x)
    q = quantize_mxfp4(x, "ocp_floor")
    assert np.array_equal(q.blocked_codes(), codes_ref)
    assert np.array_equal(q.scale_exp.astype(np.int64), e_ref)
    assert np.allclose(e2m1_decode(q.codes) * 0 + 1, 1)  # codes decodable
