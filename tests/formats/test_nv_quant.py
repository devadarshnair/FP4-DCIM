import numpy as np
import pytest

from cimfp4.formats.codecs import E4M3_MIN_SUBNORMAL, e4m3_decode
from cimfp4.formats.mx_quant import dequantize
from cimfp4.formats.nv_quant import nv_tensor_scale, quantize_nvfp4

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
MIDPOINTS = (GRID[1:] + GRID[:-1]) / 2


def llm_like(rng, shape):
    x = rng.standard_normal(shape) * np.exp(rng.normal(0, 1.5, (shape[0], 1)))
    x[:, rng.integers(0, shape[1], 4)] *= 50  # outlier channels
    return x.astype(np.float32)


def test_tensor_scale():
    assert nv_tensor_scale(2688.0) == np.float32(1.0)
    assert nv_tensor_scale(0.0) == np.float32(1.0)
    with pytest.raises(ValueError):
        nv_tensor_scale(np.inf)


@pytest.mark.parametrize("conv", ["recip_mul", "threshold"])
def test_block_max_maps_near_six_and_roundtrip_error(conv):
    rng = np.random.default_rng(0)
    x = llm_like(rng, (64, 1024))
    q = quantize_nvfp4(x, conv)
    d = dequantize(q)
    s = e4m3_decode(q.scale_e4m3) * q.tensor_scale  # per-block decode scale
    xb = x.reshape(64, -1, 16).astype(np.float64)
    a = np.abs(xb) / s[..., None]
    # in-range elements: error ≤ half a grid step (plus tiny fp32 slack for recip_mul double rounding)
    hi = np.clip(np.searchsorted(GRID, np.minimum(a, 6), side="left"), 1, 7)
    half = (GRID[hi] - GRID[hi - 1]) / 2
    err = np.abs(np.abs(d.reshape(xb.shape)) / s[..., None] - np.minimum(a, 6))
    assert (err <= half * (1 + 1e-6)).all()
    # E4M3 RNE of a *normal* block scale (rel. half-step 1/16) keeps amax_b/scale in [6/(17/16), 6/(15/16)].
    # Blocks tiny relative to the tensor amax get subnormal E4M3 scales with coarser steps (real NVFP4
    # behaviour) - excluded from this bound.
    normal = (q.scale_e4m3 >> 3) & 0xF > 0
    ratio = np.abs(xb).max(-1) / s
    assert normal.mean() > 0.9
    assert (ratio[normal] >= 6 / (17 / 16) * (1 - 1e-6)).all()
    assert (ratio[normal] <= 6 / (15 / 16) * (1 + 1e-6)).all()


def test_conventions_agree_except_near_ties():
    rng = np.random.default_rng(1)
    x = llm_like(rng, (256, 2048))
    qa, qb = quantize_nvfp4(x, "recip_mul"), quantize_nvfp4(x, "threshold")
    assert np.array_equal(qa.scale_e4m3, qb.scale_e4m3)
    diff = qa.codes != qb.codes
    # every disagreement must sit within fp32 rounding distance of a rounding midpoint
    s = e4m3_decode(qa.scale_e4m3).astype(np.float64) * qa.tensor_scale
    a = (np.abs(x.reshape(256, -1, 16)).astype(np.float64) / s[..., None]).reshape(x.shape)
    near_tie = np.abs(a[..., None] - MIDPOINTS).min(-1) <= 4 * np.finfo(np.float32).eps * np.maximum(a, 1)
    assert (~diff | near_tie).all()
    assert diff.mean() < 1e-4


def test_enumerated_tie_cases_differ_only_by_convention():
    # an exact midpoint in the x/s domain: both conventions see an exact tie -> same RNE result
    s_dec = nv_tensor_scale(2688.0)  # = 1.0
    blk = np.zeros(16, np.float32)
    blk[0] = 6.0  # scale_b = 1.0 exactly
    blk[1:8] = MIDPOINTS
    qa, qb = quantize_nvfp4(blk, "recip_mul", 2688.0), quantize_nvfp4(blk, "threshold", 2688.0)
    assert s_dec == 1.0 and e4m3_decode(qa.scale_e4m3)[0] == 1.0
    assert np.array_equal(qa.codes, qb.codes)
    assert GRID[qa.codes[1:8] & 7].tolist() == [0, 1, 1, 2, 2, 4, 4]


def test_e4m3_subnormal_block_scales():
    # small blocks relative to the tensor amax -> block scale in the E4M3 subnormal range
    x = np.zeros((1, 64), np.float32)
    x[0, :16] = 2688.0  # sets s_dec = 1
    x[0, 16:32] = 6 * 2.0**-8  # s_dec_b·s_enc = 2^-8 -> subnormal E4M3 (min normal 2^-6)
    x[0, 32:48] = 6 * E4M3_MIN_SUBNORMAL
    x[0, 48:64] = 6 * 2.0**-12  # underflows to scale 0 -> codes 0
    for conv in ("recip_mul", "threshold"):
        q = quantize_nvfp4(x, conv)
        sc = e4m3_decode(q.scale_e4m3)[0]
        assert sc[1] == 2.0**-8 and sc[2] == E4M3_MIN_SUBNORMAL and sc[3] == 0
        assert (q.codes[0, 16:48] == 0b0111).all()  # +6
        assert (q.codes[0, 48:] == 0).all()


def test_static_tensor_amax_saturates():
    x = np.full((1, 16), 100.0, np.float32)
    q = quantize_nvfp4(x, tensor_amax=1.0)  # calibrated amax far too small
    assert e4m3_decode(q.scale_e4m3)[0, 0] == 448.0
    assert np.allclose(dequantize(q), 1.0)  # clamps to 6·448·(1/2688) = 1.0


def test_input_validation():
    with pytest.raises(ValueError):
        quantize_nvfp4(np.zeros(15))
    with pytest.raises(ValueError):
        quantize_nvfp4(np.array([np.nan] * 16))
    with pytest.raises(ValueError):
        quantize_nvfp4(np.zeros(16), convention="bogus")


def test_hw_convention_close_to_threshold():
    rng = np.random.default_rng(9)
    x = llm_like(rng, (256, 2048))
    qa, qh = quantize_nvfp4(x, "threshold"), quantize_nvfp4(x, "hw")
    scale_diff = np.mean(qa.scale_e4m3 != qh.scale_e4m3)
    assert scale_diff < 2e-3, scale_diff  # only fp32 double-rounding boundary cases
    same = np.repeat(qa.scale_e4m3 == qh.scale_e4m3, 16, axis=-1)
    assert np.array_equal(qa.codes[same], qh.codes[same])
