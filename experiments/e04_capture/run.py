"""e04 - capture quantized Linear operands from a real LLM (WP-14).

Layers {0, L/2, L-1} x {q,k,v,o,gate,up,down}; 4 prefill windows x 512 tokens of WikiText-2 test,
plus 64 greedy decode steps. Writes data/captures/<model>@<rev>/{prefill,decode}/*.npz + manifest.json.

Run: python experiments/e04_capture/run.py --model qwen2.5-0.5b [--paper]
"""

from __future__ import annotations

import argparse
import resource

import numpy as np
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimfp4.llm.capture import ActivationRecorder, save_capture, select_linears
from cimfp4.stimulus.manifest import verify_manifest, write_manifest
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run
from cimfp4.utils.seed import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    seed_all(0)
    spec = yaml.safe_load((REPO_ROOT / "experiments/configs/models.yaml").read_text())[args.model]
    rev = spec["revision"] or model_info(spec["hf_id"]).sha
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        **vars(args),
        "hf_id": spec["hf_id"],
        "revision": rev,
        "license": spec["license"],
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "dtype": "float32",
        "device": dev,
    }

    with start_run("e04_capture", cfg, paper=args.paper) as run:
        tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=rev)
        model = (
            AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=rev, torch_dtype=torch.float32)
            .to(dev)
            .eval()
        )
        n_layers = model.config.num_hidden_layers
        layers = sorted({0, n_layers // 2, n_layers - 1})
        mods = select_linears(model, layers)
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
        out_root = REPO_ROOT / "data" / "captures" / f"{args.model}@{rev[:12]}"
        files = []

        # prefill: non-overlapping windows evenly spread over the test set
        starts = np.linspace(0, len(ids) - args.seqlen - 1, args.windows).astype(int)
        with torch.no_grad(), ActivationRecorder(mods) as rec:
            for s in starts:
                model(ids[s : s + args.seqlen][None].to(dev))
        for name, mod in mods.items():
            files.append(
                save_capture(
                    out_root / "prefill" / f"{name}.npz",
                    name,
                    mod.weight.detach().float().cpu().numpy(),
                    rec.stacked(name),
                    phase="prefill",
                )
            )
        del rec

        # decode: greedy continuation of the first window, capturing one row per step
        with torch.no_grad():
            prompt = ids[starts[0] : starts[0] + args.seqlen][None].to(dev)
            out = model(prompt, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            with ActivationRecorder(mods) as rec:
                for _ in range(args.decode_steps):
                    out = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        for name, mod in mods.items():
            files.append(
                save_capture(
                    out_root / "decode" / f"{name}.npz",
                    name,
                    mod.weight.detach().float().cpu().numpy(),
                    rec.stacked(name),
                    phase="decode",
                )
            )

        write_manifest(out_root, {**cfg, "layers": layers, "modules": sorted(mods)}, files)
        problems = verify_manifest(out_root)
        total = sum(f.stat().st_size for f in files)
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        run.log_metrics(
            {
                "n_modules": len(mods),
                "n_files": len(files),
                "total_bytes": total,
                "total_gb": total / 1e9,
                "peak_rss_gib": peak_rss_gib,
                "manifest_ok": not problems,
                "capture_dir": str(out_root.relative_to(REPO_ROOT)),
            }
        )
        assert not problems, problems
        assert total <= 2e9, "capture exceeds 2 GB budget"
        print(f"{len(files)} files, {total / 1e6:.1f} MB, peak RSS {peak_rss_gib:.2f} GiB -> {out_root}")


if __name__ == "__main__":
    main()
