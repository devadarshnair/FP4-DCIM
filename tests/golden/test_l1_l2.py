from pathlib import Path

import numpy as np
import pytest

from cimfp4.formats.mx_quant import quantize_mxfp4
from cimfp4.golden.accum import Window
from cimfp4.golden.l1_dot import l1_dot
from cimfp4.golden.l2_macro import L2Macro, popcount_tc, run_gemv
from cimfp4.golden.transactions import MacroOp, WeightWrite
from cimfp4.rtlgen.pkg import CimParams
from cimfp4.utils.provenance import REPO_ROOT


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def llm_like(rng, n, k, s=1.5):
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, s, (n, 1)))
    x[:, rng.integers(0, k, max(1, k // 512))] *= 30
    return x


def test_popcount_tc():
    assert popcount_tc(np.array([-1]), 9)[0] == 9
    assert popcount_tc(np.array([144, 0, -144]), 9).tolist() == [2, 0, bin((-144) & 511).count("1")]


@pytest.mark.slow
def test_l2_equals_l1_window_on_1e5_ops():
    rng = np.random.default_rng(0)
    params = CimParams()
    ops = 0
    while ops < 100_000:
        k = int(rng.choice([256, 1024, 4096]))
        qw = quantize_mxfp4(llm_like(rng, 32, k, 0.5))
        qx = quantize_mxfp4(llm_like(rng, 1, k, 2.0))
        y2, mac = run_gemv(qw, qx, params)
        y1 = l1_dot(qw, qx, Window(params.window))[:, 0]
        assert np.array_equal(bits(y2), bits(y1))
        ops += mac.cnt.ops


@pytest.mark.parametrize("n", [1, 7, 32])
def test_l2_equals_l1_window_small(n):
    rng = np.random.default_rng(n)
    qw, qx = quantize_mxfp4(llm_like(rng, n, 2048)), quantize_mxfp4(llm_like(rng, 1, 2048))
    y2, mac = run_gemv(qw, qx)
    assert np.array_equal(bits(y2), bits(l1_dot(qw, qx, Window(40))[:, 0]))
    assert mac.cnt.set_switches == 64  # 64 blocks, sets streamed (one switch per op)
    assert mac.cnt.weight_bits_written == 64 * (4 * 32 * 32 + 8 * 32)


CAP = sorted((REPO_ROOT / "data" / "captures").glob("*/prefill/*down_proj.npz"))


@pytest.mark.skipif(not CAP, reason="no WP-14 captures present")
def test_l2_equals_l1_window_on_real_capture_tiles():
    from cimfp4.llm.capture import load_capture

    qw, qa = load_capture(CAP[0], "mxfp4")
    rng = np.random.default_rng(1)
    for _ in range(4):
        r0 = int(rng.integers(0, qw.codes.shape[0] - 32))
        t = int(rng.integers(0, qa.codes.shape[0]))
        import dataclasses

        w = dataclasses.replace(
            qw,
            codes=qw.codes[r0 : r0 + 32],
            scale_exp=qw.scale_exp[r0 : r0 + 32],
            scale_nan=qw.scale_nan[r0 : r0 + 32],
        )
        x = dataclasses.replace(
            qa,
            codes=qa.codes[t : t + 1],
            scale_exp=qa.scale_exp[t : t + 1],
            scale_nan=qa.scale_nan[t : t + 1],
        )
        y2, _ = run_gemv(w, x)
        assert np.array_equal(bits(y2), bits(l1_dot(w, x, Window(40))[:, 0]))


def test_toggle_counters_match_bruteforce_recount():
    rng = np.random.default_rng(2)
    p = CimParams()
    mac = L2Macro(params=p, guard=6, keep_trace=True)
    for s in range(4):
        mac.write(WeightWrite(s, rng.integers(0, 16, (32, 32)).astype(np.uint8), rng.integers(-3, 3, 32)))
    max_e_w = np.full(32, 3)
    for c in range(64):
        x = quantize_mxfp4(rng.standard_normal((1, 32)))
        mac.step(
            MacroOp(
                x.codes[0],
                int(np.clip(x.scale_exp[0, 0], -10, 3)),
                set_idx=int(rng.integers(0, 4)),
                max_e_x=3,
                max_e_w=max_e_w,
                acc_clear=(c == 0),
            )
        )
    tr = mac.trace
    zero = {
        "bcast": np.zeros((32, 3), np.int64),
        "leaf": np.zeros((32, 32), np.int64),
        "tree": np.zeros(32, np.int64),
        "acc": np.zeros(32, np.int64),
        "latch": np.zeros((32, 32), np.uint8),
    }
    prev, cnt = zero, dict.fromkeys(("bcast", "leaf", "tree", "acc", "latch"), 0)
    for t in tr:
        b = t["bcast"] ^ prev["bcast"]
        cnt["bcast"] += sum(
            bin(int(v) & m).count("1") for col, m in ((0, 15), (1, 63), (2, 1)) for v in b[:, col]
        )
        cnt["leaf"] += sum(bin(int(v) & 511).count("1") for v in (t["leaf"] ^ prev["leaf"]).ravel())
        cnt["tree"] += sum(bin(int(v) & (2**14 - 1)).count("1") for v in (t["tree"] ^ prev["tree"]))
        cnt["acc"] += sum(bin(int(v) & (2**40 - 1)).count("1") for v in (t["acc"] ^ prev["acc"]))
        cnt["latch"] += sum(bin(int(v)).count("1") for v in (t["latch"] ^ prev["latch"]).ravel())
        prev = t
    c = mac.cnt
    assert (c.row_bcast, c.leaf_out, c.tree_out, c.acc_reg, c.weight_latch) == (
        cnt["bcast"],
        cnt["leaf"],
        cnt["tree"],
        cnt["acc"],
        cnt["latch"],
    )
    assert c.ops == 64 and 0 < c.set_switches <= 64


def test_invalid_anchor_rejected():
    mac = L2Macro()
    mac.write(WeightWrite(0, np.zeros((32, 32), np.uint8), np.full(32, 5)))
    with pytest.raises(ValueError):
        mac.step(MacroOp(np.zeros(32, np.uint8), 0, 0, max_e_x=0, max_e_w=np.zeros(32, int)))


def test_paths_exist():
    assert Path(REPO_ROOT / "rtl/include/cim_pkg.sv").exists()
