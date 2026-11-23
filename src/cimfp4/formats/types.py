"""Shared data contracts. Quantization happens once (cimfp4.formats); every golden level and the RTL
stimulus writers consume the same FP4Block, so a mismatch can never come from two quantizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Fmt = Literal["mxfp4", "nvfp4"]
BLOCK_SIZE: dict[str, int] = {"mxfp4": 32, "nvfp4": 16}


@dataclass(frozen=True)
class FP4Block:
    """Canonical block-quantized FP4 tensor; the *last axis* is blocked.

    codes        uint8 in [0, 15], unpacked E2M1 (S E1 E0 M), shape (..., K)
    scale_exp    MX: int16 unbiased UE8M0 exponent in [-127, 127], shape (..., K // block).
                 NVFP4: unused (zeros).
    scale_nan    MX: bool, block scale is NaN (0xFF) because the block contained NaN/Inf.
    scale_e4m3   NVFP4: uint8 E4M3 bit patterns, shape (..., K // 16); None for MX.
    tensor_scale NVFP4: float32 per-tensor scale (dequant = code · e4m3 · tensor_scale); None for MX.
    scale_rule   provenance string, e.g. "ocp_v11", "ocp_floor", "rceil", "ceil", "nv_recip_mul".
    """

    codes: np.ndarray
    scale_exp: np.ndarray
    fmt: Fmt
    block: int
    scale_rule: str
    scale_nan: np.ndarray = field(default=None)  # type: ignore[assignment]
    scale_e4m3: np.ndarray | None = None
    tensor_scale: float | None = None

    def __post_init__(self) -> None:
        if self.block != BLOCK_SIZE[self.fmt]:
            raise ValueError(f"{self.fmt} requires block {BLOCK_SIZE[self.fmt]}, got {self.block}")
        if self.codes.dtype != np.uint8 or (self.codes.size and self.codes.max() > 15):
            raise ValueError("codes must be uint8 in [0, 15]")
        k = self.codes.shape[-1]
        if k % self.block:
            raise ValueError(f"last axis {k} not a multiple of block {self.block}")
        nb_shape = (*self.codes.shape[:-1], k // self.block)
        if self.scale_nan is None:
            object.__setattr__(self, "scale_nan", np.zeros(nb_shape, dtype=bool))
        if self.fmt == "mxfp4":
            if self.scale_exp.shape != nb_shape:
                raise ValueError(f"scale_exp shape {self.scale_exp.shape} != {nb_shape}")
        else:
            if self.scale_e4m3 is None or self.scale_e4m3.shape != nb_shape:
                raise ValueError("nvfp4 requires scale_e4m3 with one entry per block")
            if self.tensor_scale is None:
                raise ValueError("nvfp4 requires tensor_scale")

    @property
    def num_blocks(self) -> int:
        return self.codes.shape[-1] // self.block

    def blocked_codes(self) -> np.ndarray:
        """codes reshaped to (..., n_blocks, block)."""
        return self.codes.reshape(*self.codes.shape[:-1], self.num_blocks, self.block)
