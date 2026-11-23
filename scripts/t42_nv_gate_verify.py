"""T4.2 verification of NV_GATE (verification only; synthesis / STA / power are separate steps).

    python scripts/safe_run.py --max-rss-mb 3500 -- python scripts/t42_nv_gate_verify.py

1. Lint (verilator --lint-only -Wall, SRAM model on the include path) of column_top, dpu_column, dpu_macro,
   macro_top at NV_GATE=0 and 1.
2. tb/cocotb/test_nv_gate.py: directed tests 1-8 (NV_GATE=0/1 vs golden, traces bit-identical) on macro_top
   and dpu_macro, plus test 11 (VCD: zero r_h* toggles in MX ops).
Results: results/nv_gate_verify/<run_id>/ (metrics.json, lint logs, pytest log, per-scenario summaries).
Spec: docs/spec/nv_gate.md "T4.2 directed tests".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

SRAM_DIR = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog"
TOPS = ["column_top", "dpu_column", "dpu_macro", "macro_top"]


def lint(top: str, gate: int, log: Path) -> int:
    src = "rtl/baselines/digital_mac" if top.startswith("dpu") else "rtl/src"
    cmd = [
        "verilator", "--lint-only", "-Wall", f"-GNV_GATE={gate}", "--top-module", top,
        "-Irtl/include", "-Irtl/gen", "-y", "rtl/src", "-y", "rtl/gen", "-y", "rtl/baselines/bitserial_int5",
        "-y", "rtl/baselines/digital_mac", "-y", str(SRAM_DIR), "--timing",
        "rtl/lint/waivers.vlt", "rtl/include/cim_pkg.sv", f"{src}/{top}.sv",
    ]  # fmt: skip
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    log.write_text(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    return r.returncode


def main() -> int:
    with start_run(
        "nv_gate_verify", {"tops": TOPS, "gates": [0, 1], "tests": "tb/cocotb/test_nv_gate.py"}, paper=True
    ) as run:
        lint_rc = {}
        for top in TOPS:
            for gate in (0, 1):
                rc = lint(top, gate, run.dir / f"lint_{top}_g{gate}.log")
                lint_rc[f"{top}/NV_GATE={gate}"] = rc
                print("lint", top, gate, rc, flush=True)
        env = dict(
            os.environ,
            NG_SUMMARY=str(run.dir / "directed_summary.json"),
            NG_SUMMARY_VCD=str(run.dir / "vcd_summary.json"),
        )
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-rs", "-p", "no:cacheprovider",
             f"--basetemp={REPO_ROOT / 'build/pytest_t42'}", "tb/cocotb/test_nv_gate.py"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )  # fmt: skip
        (run.dir / "pytest.log").write_text(r.stdout + r.stderr)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        print(last, flush=True)
        dsum = (
            json.loads((run.dir / "directed_summary.json").read_text())
            if (run.dir / "directed_summary.json").exists()
            else {}
        )
        vsum = (
            json.loads((run.dir / "vcd_summary.json").read_text())
            if (run.dir / "vcd_summary.json").exists()
            else {}
        )
        for v in vsum.values():
            v.pop("vcd", None)
        ok = r.returncode == 0 and all(v == 0 for v in lint_rc.values())
        run.log_metrics({"lint_rc": lint_rc, "pytest_rc": r.returncode, "pytest_summary": last,
                         "directed": dsum, "vcd_test11": vsum, "all_pass": ok})  # fmt: skip
        print(run.dir, "ALL PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
