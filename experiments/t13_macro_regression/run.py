"""T1.3: v2 macro bit-exact regression, >= 1e6 column-ops per format (MXFP4, NVFP4).

Each chunk = one Verilator build + one cocotb test (tb/cocotb/test_macro_top.py::test_macro_large) comparing every
(op, column) accumulator against the per-column L2/L1 reference (column_reference / nv_column_reference).
1 op = 32 column-ops. Run under safe_run (this driver is the single heavy job; it does not nest safe_run):
  python scripts/safe_run.py --max-rss-mb 3000 -- python experiments/t13_macro_regression/run.py --paper
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

MIXED = "0,0,0,1,1,2,0,0,3,3,3,3,1"
# name, M, store(0 SRAM/1 LATCH), src, tokens, seed, pattern, nops  (ops = tokens*M, or nops for pattern)
CHUNKS = [
    ("random_sram", 16, 0, "random", 1200, 101, None, 19200),
    ("capture_sram", 16, 0, "capture", 600, 102, None, 9600),
    ("random_latch", 16, 1, "random", 600, 103, None, 9600),
    ("pattern_sram", 4, 0, "random", 8, 104, MIXED, 9600),
]
FMTS = ["mxfp4", "nvfp4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    cfg = dict(task="T1.3", formats=FMTS, scale=a.scale,
               chunks=[dict(name=c[0], M=c[1], store=c[2], src=c[3], tokens=c[4], seed=c[5], pattern=c[6], ops=c[7]) for c in CHUNKS],
               test="tb/cocotb/test_macro_top.py::test_macro_large", column_ops_per_op=32)
    with start_run("macro_regression", cfg, paper=a.paper) as run:
        rows, tot = [], {f: 0 for f in FMTS}
        t_all = time.time()
        for fmt in FMTS:
            for name, m, store, src, tokens, seed, pat, nops in CHUNKS:
                tokens_s = max(1, int(tokens * a.scale))
                nops_s = int(nops * a.scale)
                ops = nops_s if pat else tokens_s * m
                xml = run.dir / f"junit_{fmt}_{name}.xml"
                env = dict(os.environ, T13_FMT=fmt, T13_M=str(m), T13_STORE=str(store), T13_SRC=src,
                           T13_TOKENS=str(tokens_s), T13_SEED=str(seed), MT_NOPS=str(nops_s),
                           PYTHONPATH=f"{REPO_ROOT / 'src'}")
                if pat:
                    env["T13_PATTERN"] = pat
                t0 = time.time()
                p = subprocess.run(
                    [sys.executable, "-m", "pytest", "tb/cocotb/test_macro_top.py::test_macro_large", "-v", "-rs",
                     "-p", "no:cacheprovider", f"--junitxml={xml}"],
                    cwd=REPO_ROOT, env=env, capture_output=True, text=True)
                (run.dir / f"pytest_{fmt}_{name}.log").write_text(p.stdout + p.stderr)
                r = ET.parse(xml).getroot()
                r = r if r.tag == "testsuite" else r[0]
                fail, err, skip, n = (int(r.get(k, 0)) for k in ("failures", "errors", "skipped", "tests"))
                ok = p.returncode == 0 and n == 1 and fail == err == skip == 0
                rows.append(dict(fmt=fmt, chunk=name, M=m, store=store, src=src, seed=seed, pattern=pat or "",
                                 ops=ops, column_ops=ops * 32, tests=n, failures=fail, errors=err, skipped=skip,
                                 returncode=p.returncode, seconds=round(time.time() - t0, 1), ok=ok))
                print(rows[-1], flush=True)
                if ok:
                    tot[fmt] += ops * 32
        run.save_csv("chunks.csv", rows)
        m = dict(column_ops_per_format=tot, total_seconds=round(time.time() - t_all, 1),
                 chunks=len(rows), all_ok=all(r["ok"] for r in rows),
                 mismatches=0 if all(r["ok"] for r in rows) else "see chunks.csv",
                 skipped=sum(r["skipped"] for r in rows))
        run.log_metrics(m)
        if not m["all_ok"] or min(tot.values()) < 1_000_000 * a.scale:
            raise SystemExit(f"T1.3 FAILED: {m}")


if __name__ == "__main__":
    main()
