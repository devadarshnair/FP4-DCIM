"""CIMLinear: an nn.Linear whose matmul follows the macro's arithmetic (WP-11).

Weights are quantized once (MXFP4/NVFP4 along the input dimension K, i.e. the macro rows); activations
are quantized on every call along K. backend="fakequant" multiplies the dequantized operands with an
fp32 GEMM (TF32 off), which equals L1 Exact at realistic block-exponent spreads (WP-06).

NVFP4 activation tensor scale (T6.6): cfg.nv_act_scale = "dynamic" takes the amax of the whole activation
tensor of each call (batch-dependent); "static" uses the calibrated per-layer amax `act_amax` (set by
llm.act_calib), which is what the `act_quant_nv` register implements, so every token is quantized
independently of the other tokens and samples in the batch.
"""

from __future__ import annotations

import torch
from torch import nn

from ..kernels.fakequant import no_tf32
from ..kernels.torch_quant import fq_mxfp4, fq_mxfp6, fq_mxfp8, fq_nvfp4
from .config import CIMConfig

NV_ACT_SCALES = ("dynamic", "static")


def fake_quant(x: torch.Tensor, cfg: CIMConfig, tensor_amax: float | None = None) -> torch.Tensor:
    """tensor_amax: NVFP4 static tensor amax (None = dynamic amax of x); ignored by the other formats."""
    if cfg.fmt == "mxfp4":
        return fq_mxfp4(x, cfg.scale_rule)
    if cfg.fmt == "nvfp4":
        return fq_nvfp4(x, cfg.nv_convention, tensor_amax)
    if cfg.fmt == "mxfp6":
        return fq_mxfp6(x, cfg.scale_rule)
    if cfg.fmt == "mxfp8":  # E4M3 elements (MIDAS Fig. 1(b) 8-bit point; e25)
        return fq_mxfp8(x, cfg.scale_rule)
    if cfg.fmt == "mxint4":  # D2 surrogate MXINT4-SM (docs/spec/d2_int4_contract.md)
        from ..formats.int4 import fq_mxint4

        return fq_mxint4(x)
    if cfg.fmt == "bf16":
        return x
    raise ValueError(cfg.fmt)


class CIMLinear(nn.Linear):
    """Drop-in replacement for nn.Linear (keeps the subclass relation so capture hooks still apply)."""

    def __init__(self, base: nn.Linear, cfg: CIMConfig, transpose_weight: bool = False):
        w = base.weight.detach()
        if transpose_weight:  # HF Conv1D stores (in, out)
            w = w.t()
        out_f, in_f = w.shape
        block = 16 if cfg.fmt == "nvfp4" else 32
        if cfg.backend == "l1" and cfg.fmt not in ("mxfp4", "nvfp4"):
            raise ValueError(f"l1 backend models the FP4 macro only (got {cfg.fmt})")
        if cfg.fmt != "bf16" and in_f % block:
            raise ValueError(f"in_features {in_f} not a multiple of the {cfg.fmt} block {block}")
        if cfg.nv_act_scale not in NV_ACT_SCALES:
            raise ValueError(f"nv_act_scale {cfg.nv_act_scale!r} not in {NV_ACT_SCALES}")
        super().__init__(in_f, out_f, bias=base.bias is not None, device=w.device, dtype=w.dtype)
        self.cfg = cfg
        self.act_amax: float | None = None  # static NVFP4 activation tensor amax (llm.act_calib)
        self._calib_amax: torch.Tensor | None = None  # running max |x| while calibrating
        self._sat: tuple | None = None  # (max |x|/act_amax, n elements > act_amax, n elements)
        with torch.no_grad():
            wq = fake_quant(w.float(), cfg) if (cfg.quantize_weights and cfg.fmt != "bf16") else w.float()
            self.weight.copy_(wq.to(w.dtype))
            if base.bias is not None:
                self.bias.copy_(base.bias.detach())
            if cfg.backend == "l1":  # integer operands for the exact accumulation path
                from ..kernels.torch_accum import mx_int, nv_int

                if cfg.fmt == "mxfp4":
                    wi, wm = mx_int(w.float(), cfg.scale_rule)
                    self._l1 = (wi, wm, 1.0)
                else:
                    wi, uw, kw, ws = nv_int(w.float(), cfg.nv_convention)
                    self._l1 = (wi, (uw, kw), ws)

    @property
    def static_act_scale(self) -> bool:
        return self.cfg.fmt == "nvfp4" and self.cfg.nv_act_scale == "static"

    def _act_tensor_amax(self, x: torch.Tensor) -> float | None:
        """Static NVFP4 activation amax (None = dynamic); records the clipping statistics of this call."""
        if not self.static_act_scale:
            return None
        if self.act_amax is None:
            raise RuntimeError("static NVFP4 activation scale not calibrated (cimfp4.llm.act_calib)")
        with torch.no_grad():
            ax = x.detach().abs()
            r, over = (ax.amax() / self.act_amax).float(), (ax > self.act_amax).sum()
            if self._sat is None:
                self._sat = (r, over, ax.numel())
            else:
                self._sat = (torch.maximum(self._sat[0], r), self._sat[1] + over, self._sat[2] + ax.numel())
        return self.act_amax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self._calib_amax is not None:  # calibration pass: record max |x|, activations not quantized
            with torch.no_grad():
                self._calib_amax = torch.maximum(self._calib_amax, x.detach().abs().amax().float())
            with no_tf32():
                y = torch.nn.functional.linear(
                    x.float(), self.weight.float(), None if self.bias is None else self.bias.float()
                )
            return y.to(x.dtype)
        if self.cfg.backend == "l1":
            from ..golden.accum import Exact
            from ..kernels.torch_accum import linear_l1

            wi, wm, ws = self._l1
            y = linear_l1(
                x.float(),
                wi,
                wm,
                self.cfg.fmt,
                self.cfg.accum or Exact(),
                self.cfg.scale_rule,
                self.cfg.nv_convention,
                ws,
                x_tensor_amax=self._act_tensor_amax(x),
            )
            if self.bias is not None:
                y = y + self.bias.float()
            return y.to(x.dtype)
        if self.cfg.backend != "fakequant":
            raise NotImplementedError(self.cfg.backend)
        xq = (
            fake_quant(x.float(), self.cfg, self._act_tensor_amax(x))
            if (self.cfg.quantize_acts and self.cfg.fmt != "bf16")
            else x.float()
        )
        with no_tf32():
            y = torch.nn.functional.linear(
                xq, self.weight.float(), None if self.bias is None else self.bias.float()
            )
        return y.to(x.dtype)
