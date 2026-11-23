"""Perplexity harness (WP-12) (WikiText-2 test split, fixed sequence length, non-overlapping windows).

WikiText-2 raw test, documents joined with "\\n\\n", tokenizer default add_special_tokens behaviour
(recorded), non-overlapping windows of `seqlen` (stride = seqlen), final partial window dropped, loss over
all predicted tokens of each window. Reports token-PPL and word-PPL; word count = whitespace split of the
raw text *covered by the evaluated windows* (proportional share), see `word_ppl`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PPLResult:
    token_ppl: float
    word_ppl: float
    nll_sum: float
    n_tokens: int  # predicted tokens
    n_windows: int
    n_words: float  # words attributed to the evaluated tokens
    bos_added: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def word_ppl(nll_sum: float, n_words: float) -> float:
    """Word-level perplexity from the summed token NLL (nats): exp(NLL / #words)."""
    return math.exp(nll_sum / n_words)


def token_ppl(nll_sum: float, n_tokens: int) -> float:
    return math.exp(nll_sum / n_tokens)


def wikitext2_text(split: str = "test", revision: str | None = None) -> tuple[str, str]:
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    rev = revision or dataset_info("Salesforce/wikitext").sha
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, revision=rev)
    return "\n\n".join(ds["text"]), rev


@torch.no_grad()
def perplexity(
    model, tokenizer, text: str, seqlen: int = 2048, max_windows: int | None = None, device: str = "cuda"
) -> PPLResult:
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    bos = tokenizer.bos_token_id
    bos_added = bool(
        bos is not None
        and len(ids)
        and ids[0].item() == bos
        and not text.startswith(tokenizer.bos_token or "\0")
    )
    n_win = len(ids) // seqlen
    if max_windows is not None:
        n_win = min(n_win, max_windows)
    nll_sum, n_tok = 0.0, 0
    for w in range(n_win):
        chunk = ids[w * seqlen : (w + 1) * seqlen][None].to(device)
        logits = model(chunk).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum")
        nll_sum += float(loss)
        n_tok += seqlen - 1
    total_words = len(text.split())
    n_words = total_words * (n_win * seqlen) / len(ids)  # words attributed to the evaluated span
    return PPLResult(
        token_ppl(nll_sum, n_tok), word_ppl(nll_sum, n_words), nll_sum, n_tok, n_win, n_words, bos_added
    )
