import numpy as np

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.stimulus import stats as S


def test_gaussian_floor_zero_fraction_matches_prototype():
    # experiments/prelim/fp4_stats.py (R02 §7.2): Gaussian, MXFP4 OCP v1.0 floor -> 8.8 % zeros
    rng = np.random.default_rng(0)
    q = quantize_mxfp4(rng.standard_normal((4096, 1024)), "ocp_floor")
    zero = S.code_hist(q)[[0, 8]].sum()
    assert abs(zero * 100 - 8.8) <= 0.3


def test_zero_fractions_exact_against_bruteforce():
    rng = np.random.default_rng(1)
    qw = quantize_mxfp4(rng.standard_normal((8, 64)) * rng.integers(0, 2, (8, 64)))
    qa = quantize_mxfp4(rng.standard_normal((6, 64)))
    z = S.zero_fractions(qw, qa)
    zw, za = (qw.codes & 7) == 0, (qa.codes & 7) == 0
    brute = (zw[:, None, :] | za[None, :, :]).mean()
    assert np.isclose(z["p_prod_zero"], brute)


def test_sign_balance_and_partials():
    rng = np.random.default_rng(2)
    qw, qa = quantize_mxfp4(rng.standard_normal((32, 256))), quantize_mxfp4(rng.standard_normal((32, 256)))
    rows, cols = S.sample_pairs(32, 32, 500, rng)
    prod = S.block_products(qw, qa, rows, cols)
    sb = S.sign_balance(prod)
    assert np.isclose(sb["mean_pos"] + sb["mean_neg"] + sb["mean_zero"], 32)
    assert 0.4 < sb["neg_share_of_nonzero"] < 0.6  # independent symmetric data
    bp = S.block_partials(prod)
    assert bp["abs_sum_pct"][100] <= 4608 and bp["bits_needed_p999"] <= 14


def test_exponent_spans_and_toggles_and_bitplanes():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, 128)) * np.repeat(2.0 ** rng.integers(-3, 4, (16, 4)), 32, axis=1)
    qa = quantize_mxfp4(x)
    qw = quantize_mxfp4(rng.standard_normal((4, 128)))
    rows, cols = S.sample_pairs(4, 16, 50, rng)
    spans = S.exponent_spans(qw, qa, rows, cols)
    assert spans.min() >= 0 and spans.max() <= 12
    t = S.toggle_rate(qa)
    assert 0 < t["code"] < 1 and 0 < t["mag_sm"] < 1
    bp = S.bitplane_density(qa)
    assert bp.shape == (4,) and (bp >= 0).all() and (bp <= 1).all()


def test_nvfp4_scale_exponents():
    rng = np.random.default_rng(4)
    q = quantize_nvfp4(rng.standard_normal((4, 64)))
    e = S.scale_exponents(q)
    assert e.shape == (4, 4) and (e > -20).all()
