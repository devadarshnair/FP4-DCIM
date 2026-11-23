"""GPU fake-quantizers (torch), bit-identical to the numpy reference quantizers in cimfp4.formats.

Return *dequantized* tensors (fake-quant) for LLM evaluation; computations run in float64 so that the
power-of-two scaling and the tie detection are exact (bf16/fp32 inputs are exact in float64).
Parity with the numpy reference is tested in tests/kernels/test_torch_quant.py.
"""

from __future__ import annotations

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 for bf16/fp16/fp32 inputs (power-of-two scaling, RNE and min/abs are exact there; bitwise
    equality with the float64 path is tested), float64 only for float64 inputs."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _rne_e2m1(a: torch.Tensor) -> torch.Tensor:
    """Non-negative float tensor (float32 or float64) -> nearest E2M1 magnitude (RNE, saturate at 6), as values."""
    grid = torch.tensor(E2M1_GRID, dtype=a.dtype, device=a.device)
    a = torch.clamp(a, max=6.0)
    hi = torch.clamp(torch.bucketize(a, grid, right=False), 1, 7)
    lo = hi - 1
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    up = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return grid[torch.where(up, hi, lo)]


def _floor_log2(a: torch.Tensor) -> torch.Tensor:
    _, e = torch.frexp(torch.where(a > 0, a, torch.ones_like(a)))
    return (e - 1).to(torch.int64)


def fq_mxfp4(x: torch.Tensor, rule: str = "ocp_v11") -> torch.Tensor:
    """MXFP4 fake-quant along the last axis (block 32). Matches formats.mx_quant.quantize_mxfp4."""
    shape, dtype = x.shape, x.dtype
    v = x.to(_work_dtype(x)).reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = v.abs().amax(-1)
    e = _floor_log2(amax)
    frac = amax / torch.ldexp(torch.ones_like(amax), e)
    adj = {
        "ocp_floor": torch.zeros_like(e),
        "ocp_v11": (frac >= 1.75).to(torch.int64),
        "rceil": (frac > 1.5).to(torch.int64),
        "ceil": (frac > 1.0).to(torch.int64),
    }[rule]
    xexp = torch.clamp(torch.where(amax > 0, e - 2 + adj, torch.full_like(e, -127)), -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    q = _rne_e2m1(v.abs() / s) * torch.sign(v)
    return (q * s).reshape(shape).to(dtype)


def _e4m3_rne_satfinite(a: torch.Tensor) -> torch.Tensor:
    """float64 -> nearest E4M3 value (RNE, satfinite at ±448), via torch's float8 cast after clamping."""
    return torch.clamp(a, -448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float64)


def fq_nvfp4(x: torch.Tensor, convention: str = "hw", tensor_amax: float | None = None) -> torch.Tensor:
    """NVFP4 fake-quant (block 16, E4M3 scale, fp32 tensor scale). Matches formats.nv_quant (threshold/hw)."""
    if convention not in ("threshold", "hw"):
        raise ValueError("torch NVFP4 supports the exact-division conventions (threshold, hw)")
    shape, dtype = x.shape, x.dtype
    v32 = x.to(torch.float32)
    t_amax = torch.tensor(float(v32.abs().max()) if tensor_amax is None else tensor_amax, dtype=torch.float32)
    s_dec = (
        torch.tensor(1.0, dtype=torch.float32)
        if t_amax == 0
        else (t_amax / torch.tensor(2688.0)).to(torch.float32)
    )
    s_enc = (torch.tensor(1.0, dtype=torch.float32) / s_dec).to(torch.float32)
    vb = v32.reshape(*shape[:-1], shape[-1] // 16, 16)
    amax_b = vb.abs().amax(-1)
    if convention == "hw":
        c = (s_enc / torch.tensor(6.0, dtype=torch.float32)).to(torch.float32)
        scale = _e4m3_rne_satfinite(amax_b.to(torch.float64) * float(c))
    else:
        s_dec_b = (amax_b / torch.tensor(6.0, dtype=torch.float32)).to(torch.float32)
        scale = _e4m3_rne_satfinite((s_dec_b * s_enc).to(torch.float32).to(torch.float64))
    d64 = scale * float(s_dec)
    a = torch.where(
        d64[..., None] > 0,
        vb.abs().to(torch.float64) / d64[..., None],
        torch.zeros_like(vb, dtype=torch.float64),
    )
    q = _rne_e2m1(a) * torch.sign(vb.to(torch.float64))
    return (q * d64[..., None]).reshape(shape).to(dtype)


def fq_mxfp8(x: torch.Tensor, rule: str = "ocp_v11") -> torch.Tensor:
    """MXFP8 (E4M3) fake-quant along the last axis. Matches formats.mx_quant.quantize_mxfp8_values."""
    shape, dtype = x.shape, x.dtype
    v = x.to(torch.float64).reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = v.abs().amax(-1)
    e = _floor_log2(amax)
    frac = amax / torch.ldexp(torch.ones_like(amax), e)
    adj = {
        "ocp_floor": torch.zeros_like(e),
        "ocp_v11": (frac >= 1.9375).to(torch.int64),
        "rceil": (frac > 1.75).to(torch.int64),
    }[rule]
    xexp = torch.clamp(torch.where(amax > 0, e - 8 + adj, torch.full_like(e, -127)), -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    return (_e4m3_rne_satfinite(v / s) * s).reshape(shape).to(dtype)


def _rne_e3m2(a: torch.Tensor) -> torch.Tensor:
    """float64 -> nearest E3M2 value (RNE, saturate at ±28). Matches formats.mx_quant.rne_e3m2."""
    mag = a.abs()
    e = torch.clamp(_floor_log2(mag), min=-2)
    quantum = torch.ldexp(torch.ones_like(mag), e - 2)
    q = torch.clamp(torch.round(mag / quantum) * quantum, max=28.0)  # torch.round = half to even
    return torch.where(mag > 0, q, torch.zeros_like(q)) * torch.sign(a)


def fq_mxfp6(x: torch.Tensor, rule: str = "ocp_v11") -> torch.Tensor:
    """MXFP6 (E3M2) fake-quant along the last axis. Matches formats.mx_quant.quantize_mxfp6_values."""
    shape, dtype = x.shape, x.dtype
    v = x.to(torch.float64).reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = v.abs().amax(-1)
    e = _floor_log2(amax)
    frac = amax / torch.ldexp(torch.ones_like(amax), e)
    adj = {
        "ocp_floor": torch.zeros_like(e),
        "ocp_v11": (frac >= 1.875).to(torch.int64),
        "rceil": (frac > 1.75).to(torch.int64),
    }[rule]
    xexp = torch.clamp(torch.where(amax > 0, e - 4 + adj, torch.full_like(e, -127)), -127, 127)
    s = torch.ldexp(torch.ones_like(amax), xexp)[..., None]
    return (_rne_e3m2(v / s) * s).reshape(shape).to(dtype)
