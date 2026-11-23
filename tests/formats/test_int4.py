"""D2 INT4 surrogate contract tests (docs/spec/d2_int4_contract.md v1.1 §6)."""

import numpy as np
import pytest

from cimfp4.formats.codecs import e2m1_to_int_x2
from cimfp4.formats.int4 import MAG_MAX, mxint4_scale_exponent, quantize_mxint4, recode_e2m1_to_int4


def _random(seed, n=4000):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 64)) * np.exp(rng.uniform(-20, 20, (n, 1)))
    v[0, :32] = 0.0  # all-zero block
    v[1, 5] = 7.0 * 2.0**-3  # amax exactly 7·2^X
    v[2, 7] = 7.000001 * 2.0**4  # just above 7·2^X
    v[3, 9] = 7.9 * 2.0**2  # saturates under floor_sat
    return v


def _scale(q):
    return np.repeat(np.ldexp(1.0, q.scale_exp.astype(np.int32)), 32, axis=-1)


def test_ceil_rule_never_clips_and_is_minimal():
    v = _random(0)
    q = quantize_mxint4(v, "ceil")
    assert np.abs(q.m).max() <= MAG_MAX
    amax = np.abs(v.reshape(-1, 2, 32)).max(-1)
    x, pos = q.scale_exp, amax > 0
    assert np.all(amax[pos] <= MAG_MAX * np.ldexp(1.0, x[pos].astype(np.int32)))
    assert np.all(amax[pos] > MAG_MAX * np.ldexp(1.0, (x[pos] - 1).astype(np.int32)))
    assert np.all(np.abs(q.dequantize() - v) <= 0.5 * _scale(q) + 1e-300)


def test_floor_sat_rule_scale_and_error_bound():
    v = _random(1)
    q = quantize_mxint4(v)  # primary rule
    amax = np.abs(v.reshape(-1, 2, 32)).max(-1)
    pos = amax > 0
    s = np.ldexp(1.0, q.scale_exp.astype(np.int32))
    assert np.all((amax[pos] / s[pos] >= 4) & (amax[pos] / s[pos] < 8))  # X = floor(log2 amax) - 2
    err, step = np.abs(q.dequantize() - v), _scale(q)
    sat = np.abs(v) > 7.5 * step
    assert np.all(err[~sat] <= 0.5 * step[~sat] + 1e-300)
    assert np.all(err[sat] <= 1.0 * step[sat]) and np.all(np.abs(q.m.astype(int))[sat] == MAG_MAX)
    assert np.abs(q.m).max() <= MAG_MAX and q.scale_exp[0, 0] == -127 and q.m[0, :32].max() == 0


def test_boundaries_and_rne():
    assert mxint4_scale_exponent(np.array([7.0]), "ceil") == 0
    assert mxint4_scale_exponent(np.array([7.0 + 1e-9]), "ceil") == 1
    assert mxint4_scale_exponent(np.array([7.9]), "floor_sat") == 0
    assert mxint4_scale_exponent(np.array([3.9]), "floor_sat") == -1
    blk = np.zeros(32)
    blk[:7] = [7.0, 0.5, 1.5, 2.5, -0.5, -3.5, 7.6]  # X = 0 under floor_sat: ties -> even, 7.6 saturates
    assert quantize_mxint4(blk).m[:7].tolist() == [7, 0, 2, 2, 0, -4, 7]
    with pytest.raises(ValueError):
        mxint4_scale_exponent(np.array([1.0]), "bogus")


def test_recode_tables_exact():
    codes = np.arange(16)
    i2 = e2m1_to_int_x2(codes)
    m, extra = recode_e2m1_to_int4(codes, "clip")
    assert extra == 0 and m.tolist() == np.clip(i2, -7, 7).tolist()
    m, extra = recode_e2m1_to_int4(codes, "half")
    assert extra == 1
    assert np.abs(m[:8]).tolist() == [0, 0, 1, 2, 2, 3, 4, 6]  # RNE(2v/2) for 2v = 0,1,2,3,4,6,8,12
    assert np.abs(m).max() <= MAG_MAX


def test_block_sums_exact():
    v = _random(2, 200)
    qx, qw = quantize_mxint4(v[:100]), quantize_mxint4(v[100:])
    mx_, mw_ = (q.m.reshape(100, 2, 32).astype(np.int64) for q in (qx, qw))
    s = np.einsum("tbk,nbk->tnb", mx_, mw_)
    assert np.abs(s).max() <= 32 * 49  # 12-bit signed block sum (contract §4)
    ref = qx.dequantize() @ qw.dequantize().T
    terms = s * np.ldexp(1.0, (qx.scale_exp[:, None, :] + qw.scale_exp[None, :, :]).astype(np.int32))
    assert np.allclose(terms.sum(-1), ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("rule", ["floor_sat", "ceil"])
def test_torch_matches_numpy(rule):
    torch = pytest.importorskip("torch")
    from cimfp4.formats.int4 import fq_mxint4

    v = _random(3).astype(np.float32).astype(np.float64)
    assert np.array_equal(fq_mxint4(torch.from_numpy(v), rule).numpy(), quantize_mxint4(v, rule).dequantize())
