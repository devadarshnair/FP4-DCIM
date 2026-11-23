"""T1.5: NVFP4 E4M3 block-scale NaN / zero / subnormal / reserved-code rules (numerics.md)."""

import numpy as np
import pytest

from cimfp4.formats import codecs as C
from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.golden.accum import Window
from cimfp4.golden.l1_dot import l1_dot
from cimfp4.golden.l2_macro import nv_column_nan, nv_ops_from_blocks

ALL = np.arange(256, dtype=np.uint8)


def test_nan_and_reserved_partition():
    nan = np.flatnonzero(C.e4m3_scale_nan(ALL))
    assert list(nan) == [0x7F, 0xFF]
    res = np.flatnonzero(C.e4m3_scale_reserved(ALL))
    assert list(res) == list(range(0x80, 0xFF))
    assert not (C.e4m3_scale_nan(ALL) & C.e4m3_scale_reserved(ALL)).any()
    assert np.array_equal(np.isnan(C.e4m3_decode(ALL)), C.e4m3_scale_nan(ALL))


def test_fields_zero_subnormal_normal():
    u, k, nan = C.e4m3_scale_fields(np.array([0x00, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x7F], np.uint8))
    assert list(nan) == [0, 0, 0, 0, 0, 0, 1]
    assert list(u[:3]) == [0, 1, 7] and list(k[:3]) == [-9, -9, -9]  # zero, subnormals
    assert (u[3], k[3]) == (8, -9)  # smallest normal 2^-6 = 8*2^-9
    assert u[4] * 2.0 ** int(k[4]) == 1.0 and u[5] * 2.0 ** int(k[5]) == 448.0
    assert (u[6], k[6]) == (0, 0)  # NaN: don't-care, canonicalised
    finite = np.arange(0x7F, dtype=np.uint8)
    uf, kf, _ = C.e4m3_scale_fields(finite)
    assert np.array_equal(uf * np.ldexp(1.0, kf.astype(np.int32)), C.e4m3_decode(finite))


def test_reserved_codes_rejected():
    for c in (0x80, 0x81, 0xFE):
        with pytest.raises(ValueError):
            C.e4m3_scale_fields(np.uint8(c))


def _pair(seed=0, n=3, m=2, k=64):
    rng = np.random.default_rng(seed)
    return quantize_nvfp4(rng.standard_normal((n, k))), quantize_nvfp4(rng.standard_normal((m, k)))


def _with_code(q, row, blk, code):
    sc = q.scale_e4m3.copy()
    sc[row, blk] = code
    return type(q)(**{**q.__dict__, "scale_e4m3": sc})


@pytest.mark.parametrize("code", [0x7F, 0xFF])
@pytest.mark.parametrize("side", ["w", "x"])
def test_l1_nan_scale_gives_nan_only_for_affected_dots(code, side):
    qw, qx = _pair()
    base = l1_dot(qw, qx, Window(40, guard=7))
    assert np.isfinite(base).all()
    if side == "w":
        y = l1_dot(_with_code(qw, 1, 2, code), qx, Window(40, guard=7))
        expect = np.zeros(base.shape, bool)
        expect[1, :] = True
    else:
        y = l1_dot(qw, _with_code(qx, 0, 3, code), Window(40, guard=7))
        expect = np.zeros(base.shape, bool)
        expect[:, 0] = True
    assert np.array_equal(np.isnan(y), expect)
    assert np.array_equal(y[~expect], base[~expect])  # unaffected dots bit-identical


def test_zero_scale_block_contributes_exactly_zero():
    qw, qx = _pair(1)
    z = _with_code(qw, 0, 1, 0x00)
    # same as zeroing that block's element codes
    codes = qw.codes.copy()
    codes[0, 16:32] = 0
    ref = type(qw)(**{**qw.__dict__, "codes": codes, "scale_e4m3": z.scale_e4m3})
    a, b = l1_dot(z, qx, Window(40, guard=7)), l1_dot(ref, qx, Window(40, guard=7))
    assert np.isfinite(a).all() and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_subnormal_scale_is_exact():
    qw, qx = _pair(2)
    for code in (0x01, 0x07):
        y = l1_dot(_with_code(qw, 2, 0, code), qx)  # Exact mode
        assert np.isfinite(y).all()


def test_l2_sticky_nan_flag():
    qw, qx = _pair(3, n=1, m=1, k=128)  # 8 NV blocks -> 4 column ops
    qw = _with_code(qw, 0, 3, 0x7F)  # block 3 -> op 1, half 1
    ops = nv_ops_from_blocks(qw, qx)
    assert list(ops["w_nan"]) == [False, True, False, False]
    assert not ops["x_nan"].any()
    assert list(nv_column_nan(ops["w_nan"], ops["x_nan"], ops["clear"])) == [False, True, True, True]
    # a new dot product (acc_clear) clears the flag
    clear = np.array([True, False, True, False])
    assert list(nv_column_nan(ops["w_nan"], ops["x_nan"], clear)) == [False, True, False, False]
