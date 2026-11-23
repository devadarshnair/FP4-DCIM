import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cimfp4.formats.mx_quant import dequantize, quantize_mxfp4, quantize_mxfp8_values  # noqa: E402
from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402
from cimfp4.kernels.torch_quant import fq_mxfp4, fq_mxfp8, fq_nvfp4  # noqa: E402

DEV = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(seed, n=256, k=2048):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, 2, (n, 1)))
    x[:, rng.integers(0, k, 8)] *= 30
    return x.astype(np.float32)


@pytest.mark.parametrize("dev", DEV)
@pytest.mark.parametrize("rule", ["ocp_floor", "ocp_v11", "rceil", "ceil"])
def test_fq_mxfp4_bitwise(dev, rule):
    x = data(1)
    got = fq_mxfp4(torch.from_numpy(x).to(dev), rule).cpu().numpy()
    assert np.array_equal(got, dequantize(quantize_mxfp4(x, rule)).astype(np.float32))


@pytest.mark.parametrize("dev", DEV)
@pytest.mark.parametrize("conv", ["threshold", "hw"])
def test_fq_nvfp4_bitwise(dev, conv):
    x = data(2)
    got = fq_nvfp4(torch.from_numpy(x).to(dev), conv).cpu().numpy().astype(np.float64)
    ref = dequantize(quantize_nvfp4(x, conv))
    assert np.array_equal(got.astype(np.float32), ref.astype(np.float32))


@pytest.mark.parametrize("dev", DEV)
def test_fq_mxfp8_bitwise(dev):
    x = data(3)
    got = fq_mxfp8(torch.from_numpy(x).to(dev)).cpu().numpy()
    assert np.array_equal(got, quantize_mxfp8_values(x).astype(np.float32))


def test_mxfp8_error_small():
    x = data(4).astype(np.float64)
    q = quantize_mxfp8_values(x)
    rel = np.abs(q - x) / np.maximum(np.abs(x), 1e-30)
    assert np.median(rel) < 0.04


# ---- B2: float32 compute path of fq_mxfp4 is bitwise equal to the float64 path -----------------------------
def _edge_tensor(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    fi = torch.finfo(dtype)
    rows.append(torch.zeros(2048))
    rows.append(torch.full((2048,), float(fi.tiny)))
    rows.append(torch.full((2048,), float(fi.max)))
    rows.append(-torch.full((2048,), float(fi.max)))
    rows.append(torch.randn(2048, generator=g) * float(fi.tiny))  # subnormal-scale blocks
    rows.append(torch.randn(2048, generator=g) * float(fi.max) / 8)  # extreme but finite
    rows.append(torch.randn(2048, generator=g) * 2.0 ** -120)
    rows.append(torch.randn(2048, generator=g) * 2.0 ** 100)
    sub = torch.randn(2048, generator=g) * 2.0 ** -140  # fp32 subnormals
    rows.append(sub)
    mixed = torch.randn(2048, generator=g)
    mixed[::7] = 0
    mixed[3::64] = float(fi.max)
    rows.append(mixed)
    # exact ties/thresholds: block amax at 1.75/1.5/1.0 * 2^k and E2M1 midpoints
    tie = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 1.5, 1.0, 0.5, 0.0], dtype=torch.float32)
    for k in (-130, -20, 0, 40):
        rows.append((tie.repeat(2048 // 12 + 1)[:2048] * 2.0**k))
    rows.append(torch.randn(2048, generator=g) * torch.exp(torch.randn(2048, generator=g) * 8))
    return torch.stack(rows).to(dtype)


@pytest.mark.parametrize("dev", DEV)
@pytest.mark.parametrize("rule", ["ocp_floor", "ocp_v11", "rceil", "ceil"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fq_mxfp4_float32_path_bitwise_vs_float64(dev, rule, dtype):
    x = _edge_tensor(dtype).to(dev)
    x = torch.nan_to_num(x, posinf=float(torch.finfo(dtype).max), neginf=-float(torch.finfo(dtype).max))
    got = fq_mxfp4(x, rule)
    ref = fq_mxfp4(x.to(torch.float64), rule).to(dtype)
    assert got.dtype == dtype
    assert torch.equal(got.view(torch.int16 if dtype != torch.float32 else torch.int32),
                       ref.view(torch.int16 if dtype != torch.float32 else torch.int32))


@pytest.mark.parametrize("dev", DEV)
def test_fq_mxfp4_float32_path_random(dev):
    for seed in range(5):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn(64, 1024, generator=g) * torch.exp(torch.randn(64, 1, generator=g) * 3)).to(dtype).to(dev)
            x = torch.nan_to_num(x, posinf=float(torch.finfo(dtype).max), neginf=-float(torch.finfo(dtype).max))
            for rule in ("ocp_floor", "ocp_v11", "rceil", "ceil"):
                assert torch.equal(fq_mxfp4(x, rule), fq_mxfp4(x.double(), rule).to(dtype))
