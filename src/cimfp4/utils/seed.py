"""Deterministic seeding for numpy, python and (if installed) torch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int = 0, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed every RNG we use and return a fresh numpy Generator.

    torch is optional: the formats/golden layers must work without it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by third-party code
    try:
        import torch
    except ImportError:
        return np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
