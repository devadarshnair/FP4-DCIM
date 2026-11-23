import sys

import numpy as np
import pytest

from cimfp4.formats.codecs import e2m1_to_int_x2
from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.golden.accum import Window
from cimfp4.golden.l1_dot import l1_dot
from cimfp4.golden.l2_bitserial import BitSerialColumn, int5_bits
from cimfp4.golden.l2_macro import run_gemv
from cimfp4.utils.provenance import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tb" / "cocotb"))
from scoreboard import column_reference, random_stim  # noqa: E402


def test_int5_bits_roundtrip():
    v = np.arange(-16, 16)
    b = int5_bits(v).astype(np.int64)
    back = (b * (1 << np.arange(5))).sum(-1)
    back = np.where(back >= 16, back - 32, back)
    assert np.array_equal(back, v)


def test_block_sums_exact_exhaustive_pairs():
    # every (w, x) code pair in one row, others zero -> block sum = single product
    col = BitSerialColumn()
    w = np.zeros((256, 32), np.uint8)
    x = np.zeros((256, 32), np.uint8)
    w[:, 0] = np.repeat(np.arange(16), 16)
    x[:, 0] = np.tile(np.arange(16), 16)
    s, _ = col.block_sums(w, x)
    assert np.array_equal(s, e2m1_to_int_x2(w[:, 0]) * e2m1_to_int_x2(x[:, 0]))


@pytest.mark.slow
def test_bitserial_equals_l1_window_1e5_ops():
    col = BitSerialColumn()
    rng = np.random.default_rng(0)
    n = 0
    while n < 100_000:
        st = random_stim(rng, 300, max_blocks=128)
        s, _ = col.block_sums(st.w, st.x)
        acc = col.accumulate(s, st.e_w, st.e_x, st.max_e_w, st.max_e_x, st.clear)
        ref, _ = column_reference(st)  # == L2 v2 column == L1 Window
        assert np.array_equal(acc, ref)
        n += len(st)


def test_bitserial_equals_v2_on_gemv():
    rng = np.random.default_rng(1)
    qw, qx = quantize_mxfp4(rng.standard_normal((1, 4096))), quantize_mxfp4(rng.standard_t(3, (1, 4096)))
    st_w = qw.blocked_codes()[0]
    st_x = qx.blocked_codes()[0]
    col = BitSerialColumn(guard=7)
    s, _ = col.block_sums(st_w, st_x)
    nb = qw.num_blocks
    ew, ex = qw.scale_exp[0].astype(np.int64), qx.scale_exp[0].astype(np.int64)
    acc = col.accumulate(s, ew, ex, np.full(nb, ew.max()), np.full(nb, ex.max()), np.arange(nb) == 0)
    y2, mac = run_gemv(qw, qx)
    assert int(acc[-1]) == int(mac.acc[0])
    assert np.isclose(float(y2[0]), float(l1_dot(qw, qx, Window(40))[0, 0]))


def test_toggle_ratio_vs_v2_weight_stationary():
    """Structural zero-delay toggle count, weight-stationary (static weights), bit-serial datapath incl.
    AND nodes + popcount trees, v2 incl. leaf outputs + tree + CPA output. R03 claimed ~2x; measured
    1.17-1.31x on synthetic and real Qwen tiles. Regression guard on the synthetic case.
    The combiner/accumulator of the bit-serial design (~1.2 toggles/MAC) is excluded here.
    """
    from cimfp4.rtlgen.tree import build_column_tree, evaluate, toggle_counts

    rng = np.random.default_rng(0)
    w = quantize_mxfp4(rng.standard_normal((1, 32))).codes[0]
    x = quantize_mxfp4(rng.standard_t(3, (400, 32))).codes
    ws = np.repeat(w[None], len(x), 0)
    _, tog_bs = BitSerialColumn().block_sums(ws, x, count_toggles=True)
    bs = tog_bs["and_nodes"] + tog_bs["tree_cells"]
    nl = build_column_tree(9, "oc")
    nets, out = evaluate(nl, e2m1_to_int_x2(ws) * e2m1_to_int_x2(x))
    tc = toggle_counts(nl, nets)
    ratio = bs / (tc["inputs"] + tc["cells_total"])
    assert 0.95 < ratio < 1.45, ratio
