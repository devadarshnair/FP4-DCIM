import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402
from cimfp4.golden.accum import Exact  # noqa: E402
from cimfp4.golden.l1_dot import l1_dot  # noqa: E402
from cimfp4.kernels.fakequant import dequantize_torch, fakequant_matmul  # noqa: E402
from cimfp4.kernels.l1_torch import l1_exact_torch  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def with_spread(q, rng, spread):
    """Replace MX block scale exponents with values uniform in [-spread, spread]."""
    e = rng.integers(-spread, spread + 1, q.scale_exp.shape).astype(np.int16)
    return dataclasses.replace(q, scale_exp=e)


def gauss(rng, n, k):
    return rng.standard_normal((n, k)).astype(np.float32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [32, 512, 4096])
def test_torch_exact_matches_numpy_l1_mx(device, k):
    rng = np.random.default_rng(k)
    qw = quantize_mxfp4(gauss(rng, 24, k) * np.exp(rng.normal(0, 2, (24, 1))))
    qx = quantize_mxfp4(gauss(rng, 20, k) * np.exp(rng.normal(0, 2, (20, 1))))
    assert np.array_equal(bits(l1_exact_torch(qw, qx, device)), bits(l1_dot(qw, qx, Exact())))


@pytest.mark.parametrize("device", DEVICES)
def test_torch_exact_nv_within_one_ulp(device):
    rng = np.random.default_rng(3)
    qw, qx = quantize_nvfp4(gauss(rng, 16, 1024)), quantize_nvfp4(gauss(rng, 12, 1024))
    ref = l1_dot(qw, qx, Exact())
    got = l1_exact_torch(qw, qx, device)
    assert (np.abs(got - ref) <= np.abs(np.spacing(ref))).all()


def test_torch_exact_falls_back_on_extreme_spread():
    rng = np.random.default_rng(4)
    qw = with_spread(quantize_mxfp4(gauss(rng, 4, 256)), rng, 100)
    qx = quantize_mxfp4(gauss(rng, 3, 256))
    assert np.array_equal(bits(l1_exact_torch(qw, qx)), bits(l1_dot(qw, qx, Exact())))


def test_dequantize_torch_matches_numpy():
    from cimfp4.formats.mx_quant import dequantize

    rng = np.random.default_rng(5)
    q = quantize_mxfp4(gauss(rng, 8, 128))
    assert np.array_equal(dequantize_torch(q, dtype=torch.float64).numpy(), dequantize(q))


@pytest.mark.parametrize("device", DEVICES)
def test_fakequant_bitwise_at_small_spread(device):
    # At realistic (±2) block-exponent spreads the fp32 fake-quant GEMM is bit-exact
    rng = np.random.default_rng(6)
    k = 4096
    qw = with_spread(quantize_mxfp4(gauss(rng, 64, k)), rng, 2)
    qx = with_spread(quantize_mxfp4(gauss(rng, 64, k)), rng, 2)
    got = fakequant_matmul(qw, qx, device).cpu().numpy()
    ref = l1_dot(qw, qx, Exact())
    assert np.array_equal(bits(got), bits(ref))


@pytest.mark.parametrize("device", DEVICES)
def test_fakequant_close_at_large_spread(device):
    rng = np.random.default_rng(7)
    k = 4096
    qw = with_spread(quantize_mxfp4(gauss(rng, 64, k)), rng, 10)
    qx = with_spread(quantize_mxfp4(gauss(rng, 64, k)), rng, 10)
    got = fakequant_matmul(qw, qx, device).cpu().numpy().astype(np.float64)
    ref = l1_dot(qw, qx, Exact()).astype(np.float64)
    denom = (
        np.abs(dequantize_torch(qw, dtype=torch.float64).numpy())
        @ np.abs(dequantize_torch(qx, dtype=torch.float64).numpy()).T
    )
    assert (np.abs(got - ref) / denom).max() <= 1.3e-6
