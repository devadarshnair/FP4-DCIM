"""LLM-in-the-loop configuration (WP-11). Every field is recorded in run configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CIMConfig:
    # "mxfp4" | "nvfp4" | "mxfp6" (E3M2, MIDAS-protocol rows) | "mxfp8" (E4M3; scale rules ocp_floor/v11/rceil)
    # | "mxint4" (D2 MXINT4-SM) | "bf16" (none)
    fmt: str = "mxfp4"
    scale_rule: str = "ocp_v11"  # MX scale rule (ocp_floor | ocp_v11 | rceil | ceil)
    nv_convention: str = "hw"  # NVFP4 element/scale convention (threshold | hw)
    # NVFP4 activation tensor scale (T6.6): "dynamic" = amax of the whole activation tensor per
    # call (all tokens and samples of the batch; the pre-T6.6 rows) | "static" = calibrated per-layer amax
    # (llm.act_calib), i.e. the fixed `act_quant_nv` config register c = fp32(s_enc/6). Weights always use
    # their own (static) tensor amax.
    nv_act_scale: str = "dynamic"
    quantize_weights: bool = True
    quantize_acts: bool = True  # W4A4 when True; W4A16 when False
    backend: str = "fakequant"  # "fakequant" (fp32 GEMM, == L1 Exact at realistic spreads) | "l1"
    accum: object = None  # l1 backend: golden.accum mode (Exact, FP32Seq, Window, Hier, AlignTrunc)
    attn_policy: str = "mxfp8"  # "mxfp8" (default) | "bf16" | "mxfp4_macro"
    exclude: tuple = field(default=("lm_head", "embed", "wte", "wpe"))
    # per-module format overrides: ((name-substring, fmt), ...), first match wins. MIDAS Table II GPT2-L:
    # (("attn.", "mxfp6"),) = MXFP6 attention projections, MXFP4 elsewhere.
    fmt_overrides: tuple = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["accum"] = repr(self.accum)
        return d
