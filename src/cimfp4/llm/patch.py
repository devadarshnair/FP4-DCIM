"""Patch a HF causal LM: replace Linear / Conv1D layers with CIMLinear and set the attention policy."""

from __future__ import annotations

from dataclasses import replace

from torch import nn

from .attn_quant import install
from .cim_linear import CIMLinear
from .config import CIMConfig


def patch_model(model: nn.Module, cfg: CIMConfig) -> dict:
    """In-place patch. Returns {"patched": [...], "skipped": [...]} for the run record."""
    patched, skipped = [], []

    def cfg_for(full: str) -> CIMConfig:
        for tok, fmt in cfg.fmt_overrides:
            if tok in full:
                return replace(cfg, fmt=fmt, fmt_overrides=())
        return cfg
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if any(tok in full for tok in cfg.exclude):
                if isinstance(child, nn.Linear) or type(child).__name__ == "Conv1D":
                    skipped.append(full)
                continue
            if isinstance(child, CIMLinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, CIMLinear(child, cfg_for(full)))
                patched.append(full)
            elif type(child).__name__ == "Conv1D":  # GPT-2: weight (in, out)
                lin = nn.Linear(
                    child.weight.shape[0],
                    child.weight.shape[1],
                    bias=True,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                lin.weight.data = child.weight.data.t().contiguous()
                lin.bias.data = child.bias.data
                setattr(mod, child_name, CIMLinear(lin, cfg_for(full)))
                patched.append(full)
    if cfg.fmt != "bf16" or cfg.attn_policy != "bf16":
        install(model, cfg.attn_policy)
    return {"patched": patched, "skipped": skipped, "config": cfg.as_dict()}
