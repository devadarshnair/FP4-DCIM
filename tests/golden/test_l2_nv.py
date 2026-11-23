import numpy as np

from cimfp4.formats.nv_quant import quantize_nvfp4
from cimfp4.golden.accum import Window, round_int_to_f32
from cimfp4.golden.l1_dot import l1_dot
from cimfp4.golden.l2_macro import nv_column_reference, nv_ops_from_blocks


def test_nv_column_model_equals_l1_window():
    rng = np.random.default_rng(0)
    guard, n_ok = 7, 0
    for _ in range(40):
        k = int(rng.choice([32, 256, 1024, 2048]))
        xw = rng.standard_normal((4, k)) * np.exp(rng.normal(0, 1, (4, 1)))
        xx = rng.standard_t(3, (3, k)) * np.exp(rng.normal(0, 1, (3, 1)))
        qw, qx = quantize_nvfp4(xw), quantize_nvfp4(xx)
        y1 = l1_dot(qw, qx, Window(40, guard=guard))
        for r in range(4):
            for c in range(3):
                ops = nv_ops_from_blocks(qw, qx, r, c)
                acc = nv_column_reference(
                    ops["w"],
                    ops["x"],
                    ops["u_w"],
                    ops["k_w"],
                    ops["u_x"],
                    ops["k_x"],
                    ops["max_k_w"],
                    ops["max_k_x"],
                    ops["clear"],
                    40,
                    guard,
                )
                lsb = int(ops["max_k_w"][0] + ops["max_k_x"][0] - 2 + 21 + guard - 40 + 1)
                v = round_int_to_f32(int(acc[-1]), lsb)
                v = np.float32(v * np.float32(np.float32(qw.tensor_scale) * np.float32(qx.tensor_scale)))
                assert np.float32(y1[r, c]).view(np.uint32) == v.view(np.uint32)
                n_ok += 1
    assert n_ok == 480
