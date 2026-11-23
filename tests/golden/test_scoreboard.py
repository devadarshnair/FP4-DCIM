"""The cocotb column scoreboard reference must equal the L2 macro model (so RTL == scoreboard => RTL == L2)."""

import sys

import numpy as np

from cimfp4.golden.l2_macro import L2Macro
from cimfp4.golden.transactions import MacroOp, WeightWrite
from cimfp4.rtlgen.pkg import CimParams
from cimfp4.utils.provenance import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tb" / "cocotb"))
from scoreboard import column_reference, corner_stim, random_stim  # noqa: E402


def l2_column(st, guard=7):
    mac = L2Macro(params=CimParams(weight_sets=2), guard=guard)
    acc, nan = [], []
    for t in range(len(st)):
        s = t % 2
        w = np.zeros((32, 32), np.uint8)
        w[:, 0] = st.w[t]
        mac.write(WeightWrite(s, w, np.full(32, st.e_w[t]), w_scale_nan=np.full(32, False)))
        mac.step(
            MacroOp(
                st.x[t],
                int(st.e_x[t]),
                s,
                max_e_x=int(st.max_e_x[t]),
                max_e_w=np.full(32, st.max_e_w[t]),
                acc_clear=bool(st.clear[t]),
                x_scale_nan=bool(st.nan[t]),
            )
        )
        acc.append(int(mac.acc[0]))
        nan.append(bool(mac.nan[0]))
    return np.array(acc), np.array(nan)


def test_scoreboard_equals_l2_random():
    st = random_stim(np.random.default_rng(3), 20, max_blocks=64, nan_rate=0.05)
    a_ref, n_ref = column_reference(st)
    a_l2, n_l2 = l2_column(st)
    assert np.array_equal(a_ref, a_l2) and np.array_equal(n_ref, n_l2)


def test_scoreboard_equals_l2_corners():
    st = corner_stim()
    a_ref, n_ref = column_reference(st)
    a_l2, n_l2 = l2_column(st)
    assert np.array_equal(a_ref, a_l2) and np.array_equal(n_ref, n_l2)
