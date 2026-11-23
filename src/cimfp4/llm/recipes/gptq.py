"""GPTQ for MXFP4 / NVFP4 block-quantized weights (WP-17; Frantar et al., ICLR 2023, adapted).

Columns (input channels, the K axis = macro rows) are quantized left to right; the block scale of each
32-column (MX) / 16-column (NV) group is fixed when the group starts, from the *current* (error-updated)
weights, using the same OCP / NVFP4 rules as cimfp4.formats. The quantization error of each column is
propagated to the not-yet-quantized columns through the upper Cholesky factor of H⁻¹ (H = 2XᵀX + λI).
Output weights are on the exact FP4 grid, so CIMLinear's RTN weight re-quantization reproduces them.

Hessians are collected in a single calibration pass over the unquantized model (non-sequential GPTQ;
disclosed). Calibration data, sample count and seed are recorded by the experiment.
"""

from __future__ import annotations

import math

import torch

from ...kernels.torch_quant import _e4m3_rne_satfinite, _floor_log2, _rne_e2m1


def _mx_block_scale(wblk: torch.Tensor, rule: str) -> torch.Tensor:
    """(rows, 32) -> per-row power-of-two scale (rows, 1), OCP rule."""
    amax = wblk.abs().amax(-1)
    e = _floor_log2(amax)
    frac = amax / torch.ldexp(torch.ones_like(amax), e)
    adj = {
        "ocp_floor": torch.zeros_like(e),
        "ocp_v11": (frac >= 1.75).long(),
        "rceil": (frac > 1.5).long(),
        "ceil": (frac > 1.0).long(),
    }[rule]
    x = torch.clamp(torch.where(amax > 0, e - 2 + adj, torch.full_like(e, -127)), -127, 127)
    return torch.ldexp(torch.ones_like(amax), x)[:, None]


def _nv_block_scale(wblk: torch.Tensor, s_dec: float, s_enc: float, four_over_six: bool = False):
    """(rows, 16) -> per-row decode scale (rows, 1) = e4m3(amax·c)·s_dec  (hw convention)."""
    amax = wblk.abs().amax(-1)
    c = float(torch.tensor(s_enc, dtype=torch.float32) / torch.tensor(6.0, dtype=torch.float32))
    sc = _e4m3_rne_satfinite(amax * c) * s_dec
    if not four_over_six:
        return sc[:, None], None
    c4 = float(torch.tensor(s_enc, dtype=torch.float32) / torch.tensor(4.0, dtype=torch.float32))
    sc4 = _e4m3_rne_satfinite(amax * c4) * s_dec
    return sc[:, None], sc4[:, None]


def _q(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    a = torch.where(s > 0, w.abs() / s, torch.zeros_like(w))
    return _rne_e2m1(a) * torch.sign(w) * s


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    h: torch.Tensor,
    fmt: str = "mxfp4",
    rule: str = "ocp_v11",
    percdamp: float = 0.01,
    blocksize: int = 128,
    four_over_six: bool = False,
) -> torch.Tensor:
    """w (rows, K) float, h (K, K) = Σ 2xxᵀ. Returns quantized-dequantized weights (float64 on w.device)."""
    w = w.to(torch.float64).clone()
    h = h.to(torch.float64).clone()
    rows, k = w.shape
    g = 32 if fmt == "mxfp4" else 16
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(h))
    h += damp * torch.eye(k, dtype=h.dtype, device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q_all = torch.zeros_like(w)
    s_dec = s_enc = None
    if fmt == "nvfp4":
        t_amax = float(w.abs().max())
        s_dec = (
            1.0 if t_amax == 0 else float(torch.tensor(t_amax, dtype=torch.float32) / torch.tensor(2688.0))
        )
        s_enc = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(s_dec, dtype=torch.float32))
    scale = scale4 = None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % g == 0:  # new quantization group: fix its scale from the current weights
                blk = torch.cat([w1[:, i:], w[:, i2:]], dim=1)[:, :g]
                if fmt == "mxfp4":
                    scale, scale4 = _mx_block_scale(blk, rule), None
                else:
                    scale, scale4 = _nv_block_scale(blk, s_dec, s_enc, four_over_six)
            wc = w1[:, i : i + 1]
            qc = _q(wc, scale)
            if scale4 is not None:  # Four-over-Six: per row pick the scale with lower error
                qc4 = _q(wc, scale4)
                qc = torch.where((qc4 - wc).abs() < (qc - wc).abs(), qc4, qc)
            q1[:, i : i + 1] = qc
            e = (wc - qc) / hinv1[i, i]
            w1[:, i:] -= e @ hinv1[i : i + 1, i:]
            err1[:, i : i + 1] = e
        q_all[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q_all


def rtn_quantize(w: torch.Tensor, fmt: str = "mxfp4", rule: str = "ocp_v11") -> torch.Tensor:
    from ...kernels.torch_quant import fq_mxfp4, fq_nvfp4

    return (fq_mxfp4(w.float(), rule) if fmt == "mxfp4" else fq_nvfp4(w.float(), "hw")).to(torch.float64)


def layer_error(w: torch.Tensor, wq: torch.Tensor, x: torch.Tensor) -> float:
    """Output reconstruction error ||xWᵀ - xWqᵀ||² / ||xWᵀ||²."""
    y, yq = x.double() @ w.double().T, x.double() @ wq.double().T
    return float(((y - yq) ** 2).sum() / (y**2).sum())


def _nan_check(t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise FloatingPointError("non-finite values in GPTQ")


__all__ = ["gptq_quantize", "rtn_quantize", "layer_error", "math"]
