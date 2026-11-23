#!/usr/bin/env python3
"""Run a heavy command under a memory guard (useful on small-RAM hosts where /tmp is a RAM tmpfs).

- Refuses to start unless MemAvailable >= --min-start-mb.
- Polls every --interval s: kills the whole process group (SIGTERM, then SIGKILL) if system MemAvailable
  drops below --floor-mb or the job's process-tree RSS exceeds --max-rss-mb.
- Logs peak RSS / min MemAvailable to stderr and to --log (append) for the run record.
Exit code: the command's, or 137 if killed by the guard.

Usage: python scripts/safe_run.py [--max-rss-mb 3500] [--floor-mb 900] -- <command ...>
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def tree_rss_mb(root_pid: int) -> int:
    """Sum RSS of root_pid and all descendants (via /proc)."""
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            status = (d / "status").read_text()
            r = next((int(ln.split()[1]) for ln in status.splitlines() if ln.startswith("VmRSS:")), 0)
        except (OSError, ValueError, IndexError):
            continue
        pid = int(d.name)
        children.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(children.get(p, []))
    return total // 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rss-mb", type=int, default=3500)
    ap.add_argument("--floor-mb", type=int, default=900)
    ap.add_argument("--min-start-mb", type=int, default=2500)
    ap.add_argument("--interval", type=float, default=1.0)
    # Accepted for command-line compatibility; ignored by this standalone runner.
    ap.add_argument("--lane", choices=("cpu", "gpu"), default="cpu", help="ignored (compatibility)")
    ap.add_argument("--reserve-mb", type=int, default=None, help="ignored (compatibility)")
    ap.add_argument("--vram-gb", type=float, default=2.0, help="ignored (compatibility)")
    ap.add_argument("--log", type=Path, default=Path.home() / "tmp" / "safe_run.log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    avail = mem_available_mb()
    if avail < a.min_start_mb:
        print(f"[safe_run] refusing to start: MemAvailable {avail} MB < {a.min_start_mb} MB", file=sys.stderr)
        return 75
    proc = subprocess.Popen(cmd, start_new_session=True)

    def _forward(signum, _frame):  # pause/stop: take the whole job down, then exit
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"[safe_run] stopped by signal {signum}: {' '.join(cmd)[:160]}", file=sys.stderr)
        sys.exit(143)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGHUP, _forward)
    peak, low, reason = 0, avail, ""
    try:
        while proc.poll() is None:
            time.sleep(a.interval)
            rss, avail = tree_rss_mb(proc.pid), mem_available_mb()
            peak, low = max(peak, rss), min(low, avail)
            if avail < a.floor_mb:
                reason = f"system MemAvailable {avail} MB < floor {a.floor_mb} MB"
            elif rss > a.max_rss_mb:
                reason = f"job RSS {rss} MB > cap {a.max_rss_mb} MB"
            if reason:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
        raise
    rc = 137 if reason else proc.returncode
    msg = (f"[safe_run] rc={rc} peak_rss={peak} MB min_avail={low} MB"
           + (f" KILLED: {reason}" if reason else "") + f" cmd={' '.join(cmd)[:160]}")
    print(msg, file=sys.stderr)
    a.log.parent.mkdir(parents=True, exist_ok=True)
    with a.log.open("a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
