"""D2 bit-serial INT4-SM column model == direct integer block dot (contract §4)."""

import numpy as np
import pytest

from cimfp4.formats.int4 import quantize_mxint4
from cimfp4.golden.l2_bitserial_int4 import BitSerialInt4Column, mag_bits


def test_random_magnitudes_exact():
    rng = np.random.default_rng(0)
    w = rng.integers(-7, 8, (5000, 32))
    x = rng.integers(-7, 8, (5000, 32))
    sums, tog = BitSerialInt4Column().block_sums(w, x, count_toggles=True)
    assert np.array_equal(sums, (w * x).sum(1))
    assert tog["tree_evals"] == 5000 * 3 * 3 * 2  # 3 cycles × 3 bit-columns × 2 polarities
    assert np.abs(sums).max() <= 32 * 49


def test_extremes_and_real_quantized_blocks():
    col = BitSerialInt4Column()
    for w, x in ((np.full((1, 32), 7), np.full((1, 32), -7)), (np.full((1, 32), -7), np.full((1, 32), -7)),
                 (np.zeros((1, 32), int), np.full((1, 32), 7))):
        assert col.block_sums(w, x)[0][0] == int((w * x).sum())
    rng = np.random.default_rng(1)
    qw = quantize_mxint4(rng.standard_normal((64, 32)) * 0.05)
    qx = quantize_mxint4(rng.standard_normal((64, 32)) * np.exp(rng.uniform(-3, 3, (64, 1))))
    sums, _ = col.block_sums(qw.m.astype(np.int64), qx.m.astype(np.int64))
    assert np.array_equal(sums, (qw.m.astype(np.int64) * qx.m.astype(np.int64)).sum(1))


def test_range_check():
    with pytest.raises(ValueError):
        mag_bits(np.array([8]))
