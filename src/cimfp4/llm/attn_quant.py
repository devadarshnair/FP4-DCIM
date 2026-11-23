"""Attention operand quantization policy (WP-11).

Registers a HF attention implementation "cim_eager" that quantizes the BMM operands:
  QK^T : Q and K along head_dim (the K-dimension of QK^T)
  PV   : P (softmax probabilities) and V along the key/token axis (the K-dimension of PV)
policy "mxfp8" (default; MIDAS kept attention in MXFP8/6), "mxfp4_macro" (attention on the macro in
MXFP4), "bf16" (no quantization). Softmax and masking stay in fp32.
"""

from __future__ import annotations

import torch

from ..kernels.torch_quant import fq_mxfp4, fq_mxfp8

_POLICY = {"value": "mxfp8"}


def _fq(x: torch.Tensor, dim: int) -> torch.Tensor:
    pol = _POLICY["value"]
    if pol == "bf16":
        return x
    xt = x.transpose(dim, -1)
    pad = (-xt.shape[-1]) % 32
    if pad:
        xt = torch.nn.functional.pad(xt, (0, pad))
    q = fq_mxfp8(xt.float()) if pol == "mxfp8" else fq_mxfp4(xt.float())
    if pad:
        q = q[..., : xt.shape[-1] - pad]
    return q.transpose(dim, -1).to(x.dtype)


def _apply_mask(scores: torch.Tensor, attention_mask, kv_len: int) -> torch.Tensor:
    """Apply a HF attention mask of any convention to fp32 scores (B, H, Tq, Tk).

    bool (True = attend, the sdpa convention) -> masked_fill; float (additive 0/-inf, eager) -> add;
    None -> causal (HF's sdpa mask builder returns None for purely causal batches and relies on
    is_causal; treating None as "no mask" made attention bidirectional and PPL ~1e4).
    """
    if attention_mask is None:
        tq, tk = scores.shape[-2], scores.shape[-1]
        q_pos = torch.arange(tk - tq, tk, device=scores.device)[:, None]
        k_pos = torch.arange(tk, device=scores.device)[None, :]
        return scores.masked_fill(k_pos > q_pos, float("-inf"))
    m = attention_mask[..., :kv_len]
    if m.dtype == torch.bool:
        return scores.masked_fill(~m, float("-inf"))
    return scores + m.float()


def cim_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with quantized BMM operands. Shapes: (B, H, T, D); GQA keys/values repeated."""
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    q, k = _fq(query, -1), _fq(key, -1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scaling
    scores = _apply_mask(scores, attention_mask, key.shape[-2])
    probs = torch.softmax(scores, dim=-1)
    p, v = _fq(probs, -1), _fq(value, -2)
    out = torch.matmul(p.float(), v.float()).to(query.dtype)
    return out.transpose(1, 2).contiguous(), probs


def install(model, policy: str = "mxfp8") -> None:
    """Register and select the cim_eager attention implementation with the given policy."""
    from transformers import AttentionInterface

    if policy not in ("mxfp8", "bf16", "mxfp4_macro"):
        raise ValueError(policy)
    _POLICY["value"] = policy
    AttentionInterface.register("cim_eager", cim_eager_attention)
    try:  # transformers >= 4.53: build an explicit additive (eager) mask for our implementation
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

        ALL_MASK_ATTENTION_FUNCTIONS.register("cim_eager", eager_mask)
    except ImportError:  # older transformers: _apply_mask handles bool / None masks
        pass
    model.config._attn_implementation = "cim_eager"
