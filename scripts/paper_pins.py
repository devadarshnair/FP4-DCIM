"""Pinned run directories for the paper generators (T8.2). Pins live in experiments/configs/paper_runs.yaml:
``figures_tables`` (runs read by gen_t2/gen_t89/gen_fig_boundary) and ``numbers`` (every exported run of gen_numbers.py).
A generator never follows ``latest``: re-pinning is an explicit edit of the yaml (``gen_numbers.py --write-pins``)."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PINS_FILE = ROOT / "experiments" / "configs" / "paper_runs.yaml"


def load(section: str) -> dict[str, str]:
    return yaml.safe_load(PINS_FILE.read_text())[section]


def run_dir(exp: str, section: str = "figures_tables") -> Path:
    d = RESULTS / exp / load(section)[exp]
    if not d.is_dir():
        raise SystemExit(f"pinned run missing: {d}")
    return d
