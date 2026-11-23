"""Fast exact float64-GEMM path == slow block-term path, bitwise, incl. forced fallbacks (WP-16 speed-up)."""

import pytest

torch = pytest.importorskip("torch")

from cimfp4.golden.accum import AlignTrunc, Exact, FP32Seq, Hier, Window  # noqa: E402
from cimfp4.kernels.torch_accum import linear_l1, mx_int, nv_int  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _data(seed, t=96, n=80, k=1024, wide=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, k, generator=g) * torch.exp(torch.randn(t, 1, generator=g))
    w = torch.randn(n, k, generator=g) * 0.05
    x[:, :64] *= 8.0  # outlier channels
    if wide:  # rows / channels with huge block-exponent spread -> must fall back
        x[3, 32:64] *= 2.0**-30
        x[7, ::97] *= 2.0**25
        w[5, 128:160] *= 2.0**-28
    return x.to(DEV), w.to(DEV)


MODES = [Exact(), Window(40), Hier(16, 40, reducer="fp32"), Hier(16, 40, reducer="exact"),
         Hier(16, 30, reducer="fp32"), Hier(16, 24, reducer="fp32"), Hier(4, 36, reducer="fp32"),
         Hier(16, 24, reducer="exact"), Window(28), FP32Seq(), AlignTrunc(35), AlignTrunc(20)]


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("mode", MODES, ids=repr)
def test_fast_equals_slow(fmt, wide, mode):
    x, w = _data(1 + wide, wide=wide)
    if fmt == "mxfp4":
        wi, wm = mx_int(w)
        args = (wi, wm, fmt)
        ws = 1.0
    else:
        wi, uw, kw, ws = nv_int(w)
        args = (wi, (uw, kw), fmt)
    if isinstance(mode, Hier) and fmt == "nvfp4":  # same Kt
        mode = Hier(mode.tile_blocks * 2, mode.window, mode.guard, mode.reducer)
    try:
        slow = linear_l1(x, *args, mode, w_tensor_scale=ws, fast=False)
    except AssertionError:
        with pytest.raises(AssertionError):
            linear_l1(x, *args, mode, w_tensor_scale=ws, fast=True)
        return
    st = {}
    fast = linear_l1(x, *args, mode, w_tensor_scale=ws, fast=True, stats=st)
    assert torch.equal(fast, slow), (fmt, mode, (fast != slow).sum().item())
    # NVFP4 cannot represent these spreads (E4M3 + tensor scale clip them to 0);
    # FP32Seq/AlignTrunc have no GEMM path
    if wide and fmt == "mxfp4" and isinstance(mode, (Exact, Window, Hier)):
        assert st["fast_rows"] < st["rows"]  # the wide rows were routed to the slow path
