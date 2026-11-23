import itertools

import numpy as np
import pytest

from cimfp4.rtlgen.tree import build_column_tree, build_dual_tree, emit_sv, evaluate, toggle_counts
from cimfp4.utils.provenance import REPO_ROOT

CONFIGS = [(9, "oc"), (9, "tc"), (1, "unsigned"), (5, "unsigned")]


def random_inputs(rng, w_in, scheme, t, rows=32):
    if scheme == "unsigned":
        return rng.integers(0, 2**w_in, (t, rows))
    return rng.integers(-144, 145, (t, rows))


@pytest.mark.parametrize("w_in, scheme", [(3, "tc"), (3, "oc"), (2, "unsigned")])
def test_exhaustive_four_input_trees(w_in, scheme):
    lo, hi = (0, 2**w_in - 1) if scheme == "unsigned" else (-(2 ** (w_in - 1)) + 1, 2 ** (w_in - 1) - 1)
    nl = build_column_tree(w_in, scheme, rows=4, max_abs=None if scheme == "unsigned" else hi)
    vals = np.array(list(itertools.product(range(lo, hi + 1), repeat=4)))
    _, out = evaluate(nl, vals)
    assert np.array_equal(out["m"], vals.sum(1))
    assert np.array_equal(out["h0"], vals[:, :2].sum(1)) and np.array_equal(out["h1"], vals[:, 2:].sum(1))


@pytest.mark.slow
@pytest.mark.parametrize("w_in, scheme", CONFIGS)
def test_random_32_input_trees(w_in, scheme):
    rng = np.random.default_rng(w_in)
    nl = build_column_tree(w_in, scheme)
    for _ in range(10):  # 10 × 10^4 = 10^5 vectors
        v = random_inputs(rng, w_in, scheme, 10_000)
        _, out = evaluate(nl, v)
        assert np.array_equal(out["m"], v.sum(1))
        assert np.array_equal(out["h0"], v[:, :16].sum(1)) and np.array_equal(out["h1"], v[:, 16:].sum(1))


def test_corner_cases_fp4_leaf():
    nl = build_column_tree(9, "oc")
    corners = np.array(
        [[144] * 32, [-144] * 32, [144, -144] * 16, [0] * 32, [-1] * 32, [1] * 32, [-144] * 16 + [144] * 16]
    )
    _, out = evaluate(nl, corners)
    assert np.array_equal(out["m"], corners.sum(1))


def test_widths_and_cell_counts_reasonable():
    nl = build_column_tree(9, "tc")
    assert len(nl.outputs["m_s"]) == 14 and len(nl.outputs["h0_s"]) == 13
    assert 240 <= nl.n_fa <= 320  # R03 §4: ~276 FA for a single 32-input 9b tree with SEE


def test_toggle_counts():
    rng = np.random.default_rng(1)
    nl = build_column_tree(9, "oc")
    val, _ = evaluate(nl, random_inputs(rng, 9, "oc", 200))
    t = toggle_counts(nl, val)
    assert t["cells_total"] > 0 and t["inputs"] > 0


@pytest.mark.parametrize("w_in, scheme", [(9, "oc"), (9, "tc"), (1, "unsigned")])
def test_committed_rtl_up_to_date(w_in, scheme):
    nl = build_column_tree(w_in, scheme)
    assert (REPO_ROOT / "rtl" / "gen" / f"{nl.name}.sv").read_text() == emit_sv(nl)


def test_dual_polarity_tree_variant_c():
    rng = np.random.default_rng(7)
    nl = build_dual_tree(9)
    v = np.concatenate(
        [
            rng.integers(-144, 145, (20000, 32)),
            np.array([[144] * 32, [-144] * 32, [144, -144] * 16, [0] * 32]),
        ]
    )
    _, out = evaluate(nl, v)
    assert np.array_equal(out["m"], v.sum(1))
    assert 450 <= nl.n_fa <= 540  # R03: 504 FA + 36 HA for the dual-polarity column
    assert (REPO_ROOT / "rtl" / "gen" / f"{nl.name}.sv").read_text() == emit_sv(nl)
