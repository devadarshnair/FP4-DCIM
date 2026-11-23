"""Run logging with provenance. Every number in the paper is produced by a Run.

Layout:  results/<exp>/<run_id>/{config.yaml, env.json, git.txt, [git_diff.patch, untracked.sha256 if dirty], metrics.json, *.csv, ...}
         results/<exp>/latest -> <run_id>     (symlink, updated on success)
         results/index.csv                     (one row per run, append-only)

Usage::

    with start_run("e01_integerization", config={"formats": [...]}) as run:
        run.save_csv("integerize.csv", rows)
        run.log_metrics({"e2m1_int_bits": 5})

``paper=True`` refuses to run on a dirty git tree, so paper numbers are always tied to a commit.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

from .io import save_csv, save_json
from .provenance import REPO_ROOT, env_info, git_info, save_dirty_state

RESULTS = REPO_ROOT / "results"
INDEX_FIELDS = ["exp", "run_id", "utc", "git_sha", "dirty", "paper", "status", "path"]


class DirtyTreeError(RuntimeError):
    pass


class Run:
    def __init__(self, exp: str, config: Mapping, root: Path, paper: bool) -> None:
        self.exp, self.config, self.root, self.paper = exp, dict(config), Path(root), paper
        self.git = git_info()
        if paper and self.git["dirty"]:
            raise DirtyTreeError(f"--paper run refused: uncommitted changes in {self.git['dirty_files'][:5]}")
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{stamp}_{self.git['short'] or 'nogit'}"
        self.dir = self.root / exp / self.run_id
        self.metrics: dict = {}
        self.status = "running"

    # -- context manager ---------------------------------------------------------------------
    def __enter__(self) -> Run:
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=True))
        save_json(self.dir / "env.json", env_info())
        g = self.git
        (self.dir / "git.txt").write_text(
            f"sha: {g['sha']}\nbranch: {g['branch']}\ndirty: {g['dirty']}\n"
            + "".join(f"  M {f}\n" for f in g["dirty_files"])
        )
        if g["dirty"]:  # T0.9: a dirty-tree result must be reconstructible
            save_dirty_state(self.dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.status = "ok" if exc_type is None else f"failed:{exc_type.__name__}"
        save_json(self.dir / "metrics.json", self.metrics)
        self._append_index()
        if exc_type is None:
            link = self.root / self.exp / "latest"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(self.run_id, link)

    # -- logging -----------------------------------------------------------------------------
    def log_metrics(self, metrics: Mapping) -> None:
        self.metrics.update(metrics)
        save_json(self.dir / "metrics.json", self.metrics)

    def save_csv(self, name: str, rows: Iterable[Mapping]) -> Path:
        path = self.dir / name
        save_csv(path, rows)
        return path

    def path(self, name: str) -> Path:
        return self.dir / name

    def _append_index(self) -> None:
        index = self.root / "index.csv"
        new = not index.exists()
        with index.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "exp": self.exp,
                    "run_id": self.run_id,
                    "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "git_sha": self.git["sha"],
                    "dirty": self.git["dirty"],
                    "paper": self.paper,
                    "status": self.status,
                    "path": str(self.dir.relative_to(self.root)),
                }
            )


def start_run(exp: str, config: Mapping | None = None, *, paper: bool = False, root: Path = RESULTS) -> Run:
    return Run(exp, config or {}, root, paper)
