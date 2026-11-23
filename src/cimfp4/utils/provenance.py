"""Provenance capture: git state and software/hardware environment for every result."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info(cwd: Path = REPO_ROOT) -> dict:
    """{'sha', 'short', 'branch', 'dirty', 'dirty_files'} - dirty ignores untracked files under results/."""
    status = [
        line
        for line in _git("status", "--porcelain", cwd=cwd).splitlines()
        if line and not line[3:].startswith(("results/", "data/"))
    ]
    return {
        "sha": _git("rev-parse", "HEAD", cwd=cwd).strip(),
        "short": _git("rev-parse", "--short", "HEAD", cwd=cwd).strip(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd).strip(),
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in status],
    }


_IGNORED_PREFIXES = ("results/", "data/")


def dirty_state(cwd: Path = REPO_ROOT) -> dict:
    """{'diff': unified diff of tracked changes vs HEAD, 'untracked': {path: sha256}}.

    Paths under results/ and data/ are ignored, matching ``git_info``'s notion of dirty.
    """
    diff = _git("diff", "HEAD", "--", ".", ":(exclude)results", ":(exclude)data", cwd=cwd)
    untracked: dict[str, str] = {}
    for rel in sorted(_git("ls-files", "--others", "--exclude-standard", cwd=cwd).splitlines()):
        if not rel or rel.startswith(_IGNORED_PREFIXES):
            continue
        path = cwd / rel
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            untracked[rel] = h.hexdigest()
        except OSError:
            untracked[rel] = "unreadable"
    return {"diff": diff, "untracked": untracked}


def save_dirty_state(dest: Path, cwd: Path = REPO_ROOT) -> None:
    """Write ``git_diff.patch`` and ``untracked.sha256`` into ``dest`` (call only for a dirty tree)."""
    state = dirty_state(cwd)
    (dest / "git_diff.patch").write_text(state["diff"])
    (dest / "untracked.sha256").write_text(
        "".join(f"{h}  {rel}\n" for rel, h in state["untracked"].items())
    )


def env_info() -> dict:
    """Interpreter, key package versions, OS, CPU/GPU."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }
    for mod in ("numpy", "torch", "transformers", "cocotb"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - absence is information
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info
