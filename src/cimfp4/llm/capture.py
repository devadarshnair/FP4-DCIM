"""Capture real Linear-layer operands from a HF model and quantize them (WP-14).

Forward hooks on selected ``nn.Linear`` modules record the *input activations* (flattened to
(tokens, in_features)); weights are read directly. Both are quantized with cimfp4.formats
(MXFP4 and/or NVFP4) and saved as .npz - the real operand streams for statistics (WP-15),
L2/RTL stimulus (WP-23+) and switching-activity power (WP-28).

Weights are (out_features, in_features) = (N, K); activations (T, K): the dot-product axis K is the
last axis of both, matching FP4Block's blocked axis.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..formats.mx_quant import quantize_mxfp4
from ..formats.nv_quant import quantize_nvfp4
from ..formats.types import FP4Block

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def select_linears(model: nn.Module, layers: Iterable[int], projs: Iterable[str] = PROJ_NAMES) -> dict:
    """{qualified_name: module} for decoder layers ``layers`` and projection names ``projs``."""
    layers, projs = set(layers), set(projs)
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"layers\.(\d+)\..*\.(\w+)$", name)
        if m and int(m.group(1)) in layers and m.group(2) in projs:
            out[name] = mod
    return out


@dataclass
class ActivationRecorder:
    """Accumulates the inputs seen by each hooked Linear (as float32 numpy, tokens × K)."""

    modules: dict
    max_rows: int | None = None
    inputs: dict = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def __enter__(self) -> ActivationRecorder:
        for name, mod in self.modules.items():
            self.inputs[name] = []
            self._handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, name):
        def fn(_mod, args, _out):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu().numpy()
            self.inputs[name].append(x)

        return fn

    def stacked(self, name: str) -> np.ndarray:
        x = np.concatenate(self.inputs[name], axis=0)
        return x if self.max_rows is None else x[: self.max_rows]


def quantize_operand(x: np.ndarray, fmt: str) -> FP4Block:
    if fmt == "mxfp4":
        return quantize_mxfp4(x, "ocp_v11")
    if fmt == "nvfp4":
        return quantize_nvfp4(x, "threshold")
    raise ValueError(fmt)


def block_to_arrays(q: FP4Block, prefix: str) -> dict:
    d = {f"{prefix}_codes": q.codes, f"{prefix}_scale_exp": q.scale_exp}
    if q.fmt == "nvfp4":
        d[f"{prefix}_scale_e4m3"] = q.scale_e4m3
        d[f"{prefix}_tensor_scale"] = np.float32(q.tensor_scale)
    return d


def arrays_to_block(d, prefix: str, fmt: str) -> FP4Block:
    bs = 32 if fmt == "mxfp4" else 16
    return FP4Block(
        codes=d[f"{prefix}_codes"],
        scale_exp=d[f"{prefix}_scale_exp"],
        fmt=fmt,
        block=bs,
        scale_rule="ocp_v11" if fmt == "mxfp4" else "nv_threshold",
        scale_e4m3=d.get(f"{prefix}_scale_e4m3") if fmt == "nvfp4" else None,
        tensor_scale=float(d[f"{prefix}_tensor_scale"]) if fmt == "nvfp4" else None,
    )


def save_capture(
    path: Path,
    name: str,
    weight: np.ndarray,
    acts: np.ndarray,
    fmts=("mxfp4", "nvfp4"),
    phase: str = "prefill",
) -> Path:
    """Quantize weight (N,K) and activations (T,K) in each format and save one .npz."""
    arrays = {"weight_shape": np.array(weight.shape), "act_shape": np.array(acts.shape)}
    for fmt in fmts:
        arrays |= block_to_arrays(quantize_operand(weight, fmt), f"{fmt}_w")
        arrays |= block_to_arrays(quantize_operand(acts, fmt), f"{fmt}_a")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, module=np.array(name), phase=np.array(phase), **arrays)
    return path


def load_capture(path: Path, fmt: str) -> tuple[FP4Block, FP4Block]:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return arrays_to_block(d, f"{fmt}_w", fmt), arrays_to_block(d, f"{fmt}_a", fmt)
