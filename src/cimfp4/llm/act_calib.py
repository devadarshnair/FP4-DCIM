"""Static per-layer NVFP4 activation tensor scale (T6.6).

The `act_quant_nv` RTL block takes the activation tensor scale from a configuration register c =
fp32(s_enc/6), s_enc = 2688/amax (docs/spec/numerics.md), i.e. a fixed value per layer that is known before
the data arrives. Evaluating NVFP4 with the amax of each activation tensor (the pre-T6.6 "dynamic" rows) is
an oracle the hardware cannot realise: it sees all tokens of the sequence (including future ones) and every
sample of the batch, so the result depends on batch composition.

Calibration protocol (max calibration): the patched model (weights already quantized, attention policy
installed) runs over `nsamples` random WikiText-2 *train* windows (the e07 GPTQ calibration recipe, disjoint
from the WikiText-2 test, C4 and lm-eval data); during calibration the CIMLinear activations are not
quantized, and each NVFP4 CIMLinear records max |x| over all calibration tokens. That fp32 value is its
static `act_amax`. Activations above it at evaluation time saturate (block scale clamps at 448, elements at
±6), as in hardware; `saturation_stats` reports how often that happened.
"""

from __future__ import annotations

import torch
from torch import nn

from .cim_linear import CIMLinear

CALIB_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1", "train")


def calib_batches(
    tok, nsamples: int, seqlen: int, seed: int, revision: str | None = None
) -> list[torch.Tensor]:
    """`nsamples` random `seqlen`-token windows of WikiText-2 train (same recipe as e07 `calib_batches`)."""
    from datasets import load_dataset

    repo, name, split = CALIB_DATASET
    text = "\n\n".join(load_dataset(repo, name, split=split, revision=revision)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seqlen - 1, (nsamples,), generator=g)
    return [ids[s : s + seqlen][None] for s in starts.tolist()]


def _static_nv_linears(model: nn.Module) -> dict[str, CIMLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, CIMLinear) and m.static_act_scale}


@torch.no_grad()
def calibrate_nv_act_amax(model: nn.Module, batches, device: str = "cuda") -> dict[str, float]:
    """Max calibration of every static-scale NVFP4 CIMLinear -> {module name: amax (fp32 value)}."""
    mods = _static_nv_linears(model)
    if not mods:
        raise ValueError("no CIMLinear with fmt=nvfp4 and nv_act_scale='static' in the model")
    for m in mods.values():
        m._calib_amax = torch.zeros((), dtype=torch.float32, device=m.weight.device)
    try:
        for b in batches:
            model(b.to(device))
        amax = {n: float(m._calib_amax) for n, m in mods.items()}
    finally:
        for m in mods.values():
            m._calib_amax = None
    bad = [n for n, a in amax.items() if not (a > 0 and torch.isfinite(torch.tensor(a)))]
    if bad:
        raise ValueError(f"non-positive or non-finite calibrated amax in {bad[:5]}")
    return amax


def apply_nv_act_amax(model: nn.Module, amax: dict[str, float]) -> int:
    """Set the static activation amax of every static-scale NVFP4 CIMLinear; every one must be covered."""
    mods = _static_nv_linears(model)
    missing = sorted(set(mods) - set(amax))
    extra = sorted(set(amax) - set(mods))
    if missing or extra:
        raise KeyError(f"calibration does not match the model: missing {missing[:5]}, extra {extra[:5]}")
    for n, m in mods.items():
        m.act_amax = float(amax[n])
        m._sat = None
    return len(mods)


def saturation_stats(model: nn.Module) -> dict:
    """Clipping of the static scale since `apply_nv_act_amax`: per layer max |x|/amax and element counts."""
    per, n_over, n_tot, worst = {}, 0, 0, 0.0
    for n, m in _static_nv_linears(model).items():
        if m._sat is None:
            continue
        r, over, tot = float(m._sat[0]), int(m._sat[1]), int(m._sat[2])
        per[n] = {"max_ratio": r, "n_over": over, "n": tot}
        n_over, n_tot, worst = n_over + over, n_tot + tot, max(worst, r)
    return {
        "layers": len(per),
        "layers_over": sum(v["max_ratio"] > 1.0 for v in per.values()),
        "max_ratio": worst,
        "frac_elems_over": n_over / n_tot if n_tot else 0.0,
        "per_layer": per,
    }
