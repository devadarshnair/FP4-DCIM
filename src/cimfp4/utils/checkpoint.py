"""Row-level checkpoints for long experiment sweeps (resume after a pause / kill / crash).

Each finished row is appended (flushed + fsynced) to build/checkpoints/<exp>.jsonl together with a tag =
hash(sweep-config key + last git commit touching the code paths that determine the result). On restart,
rows whose tag matches are reused, so a resumed sweep never mixes results from different code/configs.
Commit code changes before resuming; rows from uncommitted edits are tagged with the previous commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .provenance import REPO_ROOT

CKPT_DIR = REPO_ROOT / "build" / "checkpoints"


def _code_rev(paths: list[str]) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() or "unknown"


class RowCheckpoint:
    def __init__(self, exp: str, cfg_key: dict, code_paths: list[str], directory: Path | None = None):
        self.path = (directory or CKPT_DIR) / f"{exp}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({"cfg": cfg_key, "rev": _code_rev(code_paths)}, sort_keys=True, default=str)
        self.tag = hashlib.sha256(blob.encode()).hexdigest()[:16]
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line from a kill mid-write
                    continue
                if rec.get("tag") == self.tag:
                    self._rows[rec["key"]] = rec["row"]

    @staticmethod
    def _k(key: tuple) -> str:
        return json.dumps(list(key), default=str)

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, *key) -> dict | None:
        return self._rows.get(self._k(key))

    def put(self, row: dict, *key) -> None:
        k = self._k(key)
        self._rows[k] = row
        with self.path.open("a") as f:
            f.write(json.dumps({"tag": self.tag, "key": k, "row": row}, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
