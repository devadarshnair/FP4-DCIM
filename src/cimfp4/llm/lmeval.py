"""Zero-shot downstream accuracy (lm-eval-harness) and C4 perplexity on CIM-patched models (WP-19).

`run_lmeval` wraps an already-patched in-memory HF model in lm-eval's HFLM, so every Linear runs through
CIMLinear and attention through `cim_eager` exactly as in the PPL harness. Results carry the lm-eval version
and the per-task config hashes for provenance.

C4 follows the GPTQ protocol: allenai/c4 `en/c4-validation.00000-of-00008.json.gz`, the first 1100 documents
joined with spaces, evaluated on the first 256 non-overlapping windows of `seqlen` tokens.
"""

from __future__ import annotations

import hashlib
import json

# Task -> metric reported in T2 (acc_norm where the literature standardly reports it).
TASKS = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "lambada_openai": "acc",
}

C4_REPO = "allenai/c4"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"
C4_DOCS = 1100
C4_WINDOWS = 256


def c4_text(revision: str | None = None) -> tuple[str, str]:
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    rev = revision or dataset_info(C4_REPO).sha
    ds = load_dataset(C4_REPO, data_files={"validation": C4_FILE}, split="validation", revision=rev)
    return " ".join(ds["text"][:C4_DOCS]), rev


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_lmeval(
    model,
    tokenizer,
    tasks: dict[str, str] | None = None,
    batch_size: int | str = 8,
    limit: int | None = None,
    num_fewshot: int = 0,
    max_length: int | None = None,
    seed: int = 0,
) -> dict:
    """Evaluate `model` (patched or not) on lm-eval tasks.

    Returns {task: {metric, value, stderr, n[, perplexity]}, "_meta": {version, hashes, ...}}.
    """
    import lm_eval
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or TASKS
    kw = {"max_length": max_length} if max_length else {}
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, **kw)
    res = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    out: dict = {}
    for task, metric in tasks.items():
        r = res["results"][task]
        out[task] = {
            "metric": metric,
            "value": r.get(f"{metric},none"),
            "stderr": r.get(f"{metric}_stderr,none"),
            "n": (res.get("n-samples", {}).get(task) or {}).get("effective"),
        }
        if task == "lambada_openai":
            out[task]["perplexity"] = r.get("perplexity,none")
    out["_meta"] = {
        "lm_eval_version": lm_eval.__version__,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "task_versions": res.get("versions", {}),
        "task_hashes": res.get("task_hashes") or {t: _hash(res.get("configs", {}).get(t)) for t in tasks},
    }
    return out
