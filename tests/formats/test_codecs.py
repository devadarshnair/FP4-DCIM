import numpy as np
import pytest

from cimfp4.formats import codecs as C
from cimfp4.formats import tables as T

ALL4 = np.arange(16, dtype=np.uint8)
ALL8 = np.arange(256, dtype=np.uint8)


# ---------------------------------------------------------------- E2M1
def test_e2m1_value_table_matches_spec():
    expected = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert C.e2m1_decode(ALL4[:8]).tolist() == expected
    assert C.e2m1_decode(ALL4[8:]).tolist() == [-v for v in expected]


def test_e2m1_formula_normal_and_subnormal():
    s, e, m = C.e2m1_fields(ALL4)
    formula = np.where(e > 0, 2.0 ** (e - 1) * (1 + m / 2), m / 2) * np.where(s, -1, 1)
    assert np.array_equal(formula, C.e2m1_decode(ALL4))


def test_e2m1_roundtrip_all_codes_including_signed_zero():
    vals = C.e2m1_decode(ALL4)
    assert np.array_equal(C.e2m1_encode_exact(vals), ALL4)
    assert np.signbit(C.e2m1_decode(np.uint8(8)))  # -0 keeps its sign
    assert C.e2m1_to_int_x2(np.uint8(8)) == 0


def test_e2m1_encode_rejects_off_grid():
    with pytest.raises(ValueError):
        C.e2m1_encode_exact([2.5])


def test_e2m1_magnitude_decomposition():
    m, e = C.e2m1_mag_decomp(ALL4)
    assert set(m[ALL4 & 7 != 0].tolist()) == {1, 3}
    assert np.array_equal(m * 2**e, np.abs(C.e2m1_to_int_x2(ALL4)))


def test_codes_out_of_range_rejected():
    with pytest.raises(ValueError):
        C.e2m1_decode([16])
    with pytest.raises(TypeError):
        C.e2m1_decode([1.0])


# ---------------------------------------------------------------- product table
def test_product_table_properties():
    assert T.PRODUCT_X4.shape == (16, 16)
    assert T.PRODUCT_MAG_X4.max() == 144
    assert len(T.DISTINCT_MAGS_X4) == 19  # incl. zero
    popcount = np.vectorize(lambda v: bin(int(v)).count("1"))(T.PRODUCT_MAG_X4)
    assert popcount.max() <= 2
    assert T.PRODUCT_MAG_BITS == 8
    # products ×4 equal float products ×4 exactly
    vals = C.e2m1_decode(ALL4)
    assert np.array_equal(T.PRODUCT_X4, (np.outer(vals, vals) * 4).astype(np.int64))


def test_product_mantissa_exponent_structure():
    m, e = T.product_mantissa_exponent()
    assert set(np.unique(m).tolist()) == {0, 1, 3, 9}
    assert e.max() == 6
    assert np.array_equal(m * 2**e, T.PRODUCT_MAG_X4)


def test_block_sum_widths_correct_blueprint():
    assert T.block_sum_bits(32) == (13, 14)  # MXFP4 (blueprint said 12/13)
    assert T.block_sum_bits(16) == (12, 13)  # NVFP4


# ---------------------------------------------------------------- UE8M0
def test_ue8m0():
    assert np.isnan(C.ue8m0_decode(np.uint8(0xFF)))
    assert C.ue8m0_decode(np.uint8(127)) == 1.0
    assert C.ue8m0_decode(np.uint8(0)) == 2.0**-127
    assert C.ue8m0_decode(np.uint8(254)) == 2.0**127
    assert np.array_equal(C.ue8m0_to_exp(C.ue8m0_encode(np.arange(-127, 128))), np.arange(-127, 128))
    with pytest.raises(ValueError):
        C.ue8m0_encode([128])
    with pytest.raises(ValueError):
        C.ue8m0_to_exp([0xFF])


# ---------------------------------------------------------------- E4M3
def test_e4m3_landmarks():
    assert C.e4m3_decode(np.uint8(0x7E)) == 448.0
    assert C.e4m3_decode(np.uint8(0x01)) == 2.0**-9
    assert C.e4m3_decode(np.uint8(0x38)) == 1.0
    assert np.isnan(C.e4m3_decode(np.uint8(0x7F))) and np.isnan(C.e4m3_decode(np.uint8(0xFF)))
    assert np.isfinite(C.E4M3_VALUES).sum() == 254


def test_e4m3_roundtrip_all_finite_codes():
    finite = ALL8[np.isfinite(C.E4M3_VALUES)]
    back = C.e4m3_encode(C.e4m3_decode(finite))
    # -0 (0x80) encodes back to 0x80; everything else must round-trip exactly
    assert np.array_equal(back, finite)


def test_e4m3_rne_and_saturation():
    # midpoint between 1.0 (0x38, m=0) and 1.125 (0x39, m=1) -> even (1.0)
    assert C.e4m3_encode(1.0625) == 0x38
    # midpoint between 1.125 and 1.25 (m=2) -> even (1.25)
    assert C.e4m3_encode(1.1875) == 0x3A
    assert C.e4m3_encode(1e6) == 0x7E
    assert C.e4m3_encode(-1e6) == 0xFE
    assert C.e4m3_encode(1e6, saturate=False) == 0x7F
    assert C.e4m3_encode(2.0**-11) == 0x00  # below half the min subnormal -> 0


def test_e4m3_significand():
    finite = ALL8[np.isfinite(C.E4M3_VALUES)]
    u, k = C.e4m3_significand(finite)
    assert u.max() == 15 and u.min() == 0
    assert np.array_equal(u * np.ldexp(1.0, k.astype(np.int32)), np.abs(C.e4m3_decode(finite)))


def test_e4m3_matches_torch_float8():
    torch = pytest.importorskip("torch")
    ours = C.E4M3_VALUES
    theirs = torch.from_numpy(ALL8.copy()).view(torch.float8_e4m3fn).to(torch.float64).numpy()
    assert np.array_equal(np.isnan(ours), np.isnan(theirs))
    assert np.array_equal(ours[~np.isnan(ours)], theirs[~np.isnan(theirs)])
    # rounding agrees with torch's cast on random reals (torch saturates? no - compare in-range only)
    rng = np.random.default_rng(0)
    x = rng.uniform(-440, 440, 100_000) * 2.0 ** rng.integers(-12, 1, 100_000)
    t = torch.from_numpy(x).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.array_equal(C.e4m3_encode(x), t)
