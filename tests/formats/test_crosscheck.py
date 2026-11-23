"""WP-07: our OCP-exact quantizers vs torchao. Library divergences are measured, not hidden.

MX: dequantized values and scales must be bit-identical to torchao `to_mx` for the matching scale rule
(FLOOR<->ocp_floor, EVEN<->ocp_v11, RCEIL<->rceil, CEIL<->ceil). NVFP4: torchao applies the reciprocal as
(1/s_tensor)/s_block (MSLK kernel order) whereas our `recip_mul` follows the NVIDIA paper order
1/(s_block·s_tensor); the disagreement rate is measured and must be ≤ 1e-4 (near-tie cases only).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
from torchao.prototype.mx_formats.config import ScaleCalculationMode  # noqa: E402
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4  # noqa: E402

from cimfp4.formats.mx_quant import dequantize, quantize_mxfp4  # noqa: E402
from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402

RULES = {
    "ocp_floor": ScaleCalculationMode.FLOOR,
    "ocp_v11": ScaleCalculationMode.EVEN,
    "rceil": ScaleCalculationMode.RCEIL,
    "ceil": ScaleCalculationMode.CEIL,
}
FP4 = getattr(torch, "float4_e2m1fn_x2", "fp4_e2m1")


def llm_like(rng, n, k):
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, 2, (n, 1)))
    x[:, rng.integers(0, k, max(1, k // 256))] *= 40
    return x.astype(np.float32)


def torchao_mx_dequant(x: np.ndarray, mode) -> tuple[np.ndarray, np.ndarray]:
    s, d = mx.to_mx(torch.from_numpy(x), FP4, 32, scaling_mode=mode)
    exp = s.view(torch.uint8).numpy().astype(np.int64) - 127
    vals = f4_unpacked_to_f32(unpack_uint4(d.view(torch.uint8))).numpy().astype(np.float64)
    vals = vals.reshape(*x.shape[:-1], -1, 32) * np.ldexp(1.0, exp.astype(np.int32))[..., None]
    return vals.reshape(x.shape), exp


@pytest.mark.parametrize("rule", list(RULES))
def test_mx_matches_torchao(rule):
    rng = np.random.default_rng(hash(rule) % 2**32)
    x = llm_like(rng, 512, 2048)  # 1e6 elements per rule (1e7 in the slow test)
    tv, te = torchao_mx_dequant(x, RULES[rule])
    q = quantize_mxfp4(x, rule)
    assert np.array_equal(q.scale_exp.astype(np.int64), te)
    assert np.array_equal(dequantize(q), tv)  # -0 == +0 in value comparison


@pytest.mark.slow
@pytest.mark.parametrize("rule", ["ocp_floor", "ocp_v11"])
def test_mx_matches_torchao_1e7(rule):
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = llm_like(rng, 1024, 2048)
        tv, te = torchao_mx_dequant(x, RULES[rule])
        q = quantize_mxfp4(x, rule)
        assert np.array_equal(q.scale_exp.astype(np.int64), te) and np.array_equal(dequantize(q), tv)


def test_nvfp4_vs_torchao_two_level():
    """torchao clamps block scales to the smallest *normal* E4M3 (2^-6) - no subnormal scales - whereas
    our quantizer follows the NVIDIA recipe (RNE into the subnormal range, possibly 0). On blocks whose
    scale is in the normal range both must agree bitwise (scale codes) and nearly always on elements
    (reciprocal-order near ties); the clamped-block fraction is reported, not hidden."""
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

    from cimfp4.formats.codecs import e2m1_decode, e4m3_decode

    rng = np.random.default_rng(3)
    x = llm_like(rng, 512, 2048)
    q = quantize_nvfp4(x, "recip_mul")
    ts = torch.tensor(q.tensor_scale, dtype=torch.float32)
    scales, data = nvfp4_quantize(torch.from_numpy(x), 16, per_tensor_scale=ts)
    t_scale = scales.view(torch.uint8).numpy().reshape(q.scale_e4m3.shape)
    raw = data.view(torch.uint8)
    t_vals = (
        f4_unpacked_to_f32(raw) if raw.shape[-1] == x.shape[-1] else f4_unpacked_to_f32(unpack_uint4(raw))
    )
    t_vals = t_vals.numpy().reshape(q.codes.shape)
    normal = e4m3_decode(q.scale_e4m3) >= 2.0**-6  # our scale in the normal E4M3 range
    clamped_fraction = 1 - normal.mean()
    assert np.array_equal(t_scale[normal], q.scale_e4m3[normal])
    elem_normal = np.repeat(normal, 16, axis=-1)
    disagree = np.mean(t_vals[elem_normal] != e2m1_decode(q.codes)[elem_normal])
    assert disagree <= 1e-4, disagree
    print(
        f"NVFP4 vs torchao: clamped-scale blocks {clamped_fraction:.3%}, element disagreement {disagree:.2e}"
    )
