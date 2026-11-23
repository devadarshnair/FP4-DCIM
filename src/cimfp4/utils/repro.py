"""Reproduction contract for fake-quant PPL rows (E03 repro check).

A recomputed token-PPL is compared with its reference run under the strictest criterion the arithmetic allows:

* ``bitwise`` - exact float equality. Required for every row on the *same platform* (GPU model, torch build,
  CUDA runtime as recorded in the run's env.json) and, on any platform, for formats whose fake-quant GEMM is
  exact (MXFP4: power-of-two scales, every product and partial sum representable in fp32, so the result is
  independent of the kernel's summation order; we measured 100 % bit-stable outputs).
* ``rel_tol`` - |ppl / ref - 1| <= NV_CROSS_PLATFORM_RTOL. Only for NVFP4 rows on a *different* platform: the
  E4M3 x fp32 tensor scale makes the fp32 GEMM round, so a different GEMM kernel (other GPU / cuBLAS build)
  sums in another order; the ~1e-7 relative output differences flip next-layer element/scale codes and
  compound to ~0.1 % PPL (L4 vs RTX-4060: 0.141 %). Portable NVFP4 numbers come from the order-independent L1
  path instead.

Any other format is treated as ``bitwise`` until its order-independence (or a tolerance) has been established.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Tolerance: 0.25 % relative token-PPL across platforms for NVFP4 fake-quant rows.
NV_CROSS_PLATFORM_RTOL = 2.5e-3
ORDER_INDEPENDENT_FMTS = frozenset({"mxfp4"})
TOLERANCED_FMTS = frozenset({"nvfp4"})
PLATFORM_KEYS = ("gpu", "torch", "cuda")


def platform_of(env: Mapping) -> tuple:
    """The parts of an env.json record that select the GEMM kernels (GPU model, torch build, CUDA runtime)."""
    # str(): torch.__version__ is a TorchVersion (str subclass) that yaml/json run records cannot serialise
    return tuple(None if env.get(k) is None else str(env.get(k)) for k in PLATFORM_KEYS)


@dataclass(frozen=True)
class Verdict:
    criterion: str  # "bitwise" | "rel_tol"
    same_platform: bool
    rel_diff: float
    passed: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def check_reproduction(fmt: str, ppl: float, ref_ppl: float, platform: tuple, ref_platform: tuple) -> Verdict:
    """Apply the reproduction contract to one recomputed row."""
    known = None not in platform and None not in ref_platform
    same = known and tuple(platform) == tuple(ref_platform)
    rel = abs(ppl / ref_ppl - 1.0)
    # The tolerance needs positive evidence of a platform change: an unrecorded platform stays bitwise.
    if same or not known or fmt in ORDER_INDEPENDENT_FMTS or fmt not in TOLERANCED_FMTS:
        return Verdict("bitwise", same, rel, ppl == ref_ppl)
    return Verdict("rel_tol", same, rel, rel <= NV_CROSS_PLATFORM_RTOL)
