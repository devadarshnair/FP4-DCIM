"""Small IO helpers that understand numpy types."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def save_csv(path: Path, rows: Iterable[Mapping]) -> None:
    """Write a list of dicts (or a pandas DataFrame) to CSV; columns = union of keys in first-seen order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(rows, "to_csv"):
        rows.to_csv(path, index=False)
        return
    rows = list(rows)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _default(v) if isinstance(v, np.generic) else v for k, v in r.items()})


def load_csv(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))
