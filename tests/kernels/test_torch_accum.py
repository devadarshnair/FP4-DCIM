import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402
from cimfp4.golden.accum import AlignTrunc, Exact, FP32Seq, Hier, Window  # noqa: E402
from cimfp4.golden.l1_dot import l1_dot  # noqa: E402
from cimfp4.kernels.torch_accum import linear_l1, mx_int, nv_int  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODES = [
    Exact(),
    FP32Seq(),
    Window(40),
    Hier(4, 40, reducer="fp32"),
    Hier(4, 40, reducer="exact"),
    AlignTrunc(35),
]


def data(seed, n, k, s):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)) * np.exp(rng.normal(0, s, (n, 1)))
    x[:, rng.integers(0, k, 4)] *= 20
    return x.astype(np.float32)


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: type(m).__name__ + str(getattr(m, "reducer", "")))
def test_mx_linear_l1_matches_numpy(mode):
    x, w = data(1, 24, 1024, 1.5), data(2, 20, 1024, 0.5)
    wi, wm = mx_int(torch.from_numpy(w).to(DEV))
    got = linear_l1(torch.from_numpy(x).to(DEV), wi, wm, "mxfp4", mode).cpu().numpy()
    ref = l1_dot(quantize_mxfp4(x, "ocp_v11"), quantize_mxfp4(w, "ocp_v11"), mode)  # (T, N)
    assert np.array_equal(bits(got), bits(ref))


@pytest.mark.parametrize(
    "mode", [Exact(), Window(40), Hier(4, 40, reducer="fp32")], ids=lambda m: type(m).__name__
)
def test_nv_linear_l1_matches_numpy(mode):
    x, w = data(3, 16, 1024, 1.0), data(4, 12, 1024, 0.5)
    wi, uw, kw, ws = nv_int(torch.from_numpy(w).to(DEV))
    got = linear_l1(torch.from_numpy(x).to(DEV), wi, (uw, kw), "nvfp4", mode, w_tensor_scale=ws).cpu().numpy()
    ref = l1_dot(quantize_nvfp4(x, "hw"), quantize_nvfp4(w, "hw"), mode)  # (T, N)
    assert np.allclose(got, ref, rtol=2e-7, atol=0)  # tensor-scale product order may differ by <=1 ulp
