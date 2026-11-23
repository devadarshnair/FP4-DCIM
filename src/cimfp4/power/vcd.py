"""Streaming VCD utilities for the power flow (WP-28).

`slice_vcd` keeps the header, emits a `$dumpvars` block with every signal's value at t0 (so the
window starts from the true state), then copies value changes in [t0, t1). This restricts OpenSTA's
activity annotation to the compute phase (weight loading excluded), identically for all designs.
"""

from __future__ import annotations

import re
from pathlib import Path

_TS = re.compile(r"^#(\d+)\s*$")


def _timescale_ps(header: str) -> float:
    m = re.search(r"\$timescale\s+(\d+)\s*(fs|ps|ns|us|ms|s)\s+\$end", header, re.S)
    mul = {"fs": 1e-3, "ps": 1.0, "ns": 1e3, "us": 1e6, "ms": 1e9, "s": 1e12}
    return int(m.group(1)) * mul[m.group(2)] if m else 1.0


def slice_vcd(src: Path, dst: Path, t0_ns: float, t1_ns: float, rebase: bool = False) -> dict:
    """Window [t0_ns, t1_ns) of `src` into `dst`. Returns stats (signals, changes kept, duration_ns).

    rebase=True shifts the window to start at time 0. OpenSTA read_vcd divides toggle counts by the
    time from 0 to the last timestamp, so an un-rebased window that starts at t0 > 0 under-counts
    activity by (t1 - t0) / t1 (OpenSTA averages toggles from time 0). Default False keeps the e13 runs reproducible."""
    src, dst = Path(src), Path(dst)
    header_lines, ids = [], []
    state: dict[str, str] = {}
    with src.open() as f:
        for line in f:
            header_lines.append(line)
            m = re.match(r"\s*\$var\s+\S+\s+\d+\s+(\S+)\s", line)
            if m:
                ids.append(m.group(1))
            if line.startswith("$enddefinitions"):
                break
        header = "".join(header_lines)
        unit_ps = _timescale_ps(header)
        t0, t1 = int(t0_ns * 1e3 / unit_ps), int(t1_ns * 1e3 / unit_ps)
        kept, now, started = 0, 0, False
        with dst.open("w") as out:
            out.write(header)

            off = t0 if rebase else 0

            def emit_state():
                out.write(f"#{t0 - off}\n$dumpvars\n")
                for k, v in state.items():
                    out.write(f"{v}{k}\n" if len(v) == 1 else f"{v} {k}\n")
                out.write("$end\n")

            for line in f:
                m = _TS.match(line)
                if m:
                    now = int(m.group(1))
                    if now >= t1:
                        break
                    if now >= t0 and not started:
                        emit_state()
                        started = True
                    if started and now > t0:
                        out.write(f"#{now - off}\n")
                    continue
                s = line.strip()
                if not s or s.startswith("$"):
                    continue
                if s[0] in "bBrR":
                    val, key = s.split()
                else:
                    val, key = s[0], s[1:]
                state[key] = val
                if started and now >= t0:
                    out.write(line)
                    kept += 1
            if not started:
                emit_state()
            out.write(f"#{t1 - off}\n")
    return {"signals": len(ids), "changes_kept": kept, "duration_ns": t1_ns - t0_ns}
