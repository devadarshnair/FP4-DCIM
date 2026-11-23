"""T5.2: INT4-SM column (rtl/baselines/bitserial_int4sm/bs4_column.sv) verification, >= 1e5 randomized ops.

Chunks (one Verilator build + one cocotb test each, tb/cocotb/test_bs4_verify.py): 1 directed chunk (explicit -0,
+-7 extremes, window-edge d sweep incl. clamp at DMAX / negative d, accumulator overflow, NaN) + N random chunks with
recorded seeds (alternating back-to-back and 15 % idle-gap issue).  Then 4 mutation self-checks on a scratch copy of the
RTL: each must be caught by the directed chunk.  Run under safe_run (this driver is the single heavy job, no nesting):
  python scripts/safe_run.py --lane cpu --max-rss-mb 1500 -- python experiments/t52_bs4_verify/run.py [--paper]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

RTL = REPO_ROOT / "rtl/baselines/bitserial_int4sm/bs4_column.sv"
SEEDS = [5201, 5202, 5203, 5204, 5205, 5206]
NOPS = 25000
MUTATIONS = [  # (name, old, new) applied to a scratch copy of the RTL
    ("polarity_swap", "(p == 1 ? neg : ~neg)", "(p == 1 ? ~neg : neg)"),
    ("ignore_x_sign", "w_bus[4*r+3] ^ xsgn_bus[r]", "w_bus[4*r+3]"),
    ("clamp_removed", "(d_raw > $signed(10'(DMAX))) ? 6'(DMAX) : d_raw[5:0]", "d_raw[5:0]"),
    ("dneg_not_flagged", "b_dneg <= d_raw[9];", "b_dneg <= 1'b0;"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_chunk(run, name, mode, seed, nops, gap, rtl=None):
    out = run.dir / f"chunk_{name}.json"
    env = dict(os.environ, B4V_MODE=mode, B4V_SEED=str(seed), B4V_NOPS=str(nops), B4V_GAP=str(gap), B4V_OUT=str(out),
               PYTHONPATH=str(REPO_ROOT / "src"))
    if rtl:
        env["B4V_RTL"] = str(rtl)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "tb/cocotb/test_bs4_verify.py", "-q", "-p", "no:cacheprovider"],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    (run.dir / f"pytest_{name}.log").write_text(p.stdout + p.stderr)
    res = json.loads(out.read_text()) if out.exists() else {}
    return p.returncode, res, round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    nops = int(NOPS * a.scale)
    cfg = dict(task="T5.2", rtl=str(RTL.relative_to(REPO_ROOT)), rtl_sha256=sha(RTL), seeds=SEEDS, ops_per_random_chunk=nops,
               gap_prob_alternating=[0.0, 0.15], tb="tb/cocotb/test_bs4_verify.py",
               golden="cimfp4.golden.l2_bitserial_int4.BitSerialInt4Column", mutations=[m[0] for m in MUTATIONS])
    with start_run("t52_bs4_verify", cfg, paper=a.paper) as run:
        rows, ops_tot, mm_tot, cov = [], 0, 0, {}
        plan = [("directed", "directed", 5200, 0, 0.0)] + [
            (f"random_s{s}", "random", s, nops, 0.0 if i % 2 == 0 else 0.15) for i, s in enumerate(SEEDS)]
        for name, mode, seed, n, gap in plan:
            rc, res, secs = run_chunk(run, name, mode, seed, n, gap)
            ok = rc == 0 and res.get("mismatch_total", 1) == 0
            rows.append(dict(chunk=name, mode=mode, seed=seed, gap_prob=gap, ops=res.get("n_ops", 0), returncode=rc,
                             mismatch_total=res.get("mismatch_total", "n/a"), **{f"mm_{k}": v for k, v in res.get("mismatches", {}).items()},
                             seconds=secs, ok=ok))
            print(rows[-1], flush=True)
            if res:
                ops_tot += res["n_ops"]
                mm_tot += res["mismatch_total"]
                for k, v in res["coverage"].items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            cov.setdefault(k, {}).setdefault(kk, 0)
                            cov[k][kk] += vv
                    else:
                        cov[k] = cov.get(k, 0) + v
        run.save_csv("chunks.csv", rows)
        muts = []
        scratch = Path(tempfile.mkdtemp(prefix="b4mut_", dir=os.environ.get("TMPDIR")))
        base = RTL.read_text()
        for mname, old, new in MUTATIONS:
            assert base.count(old) == 1, mname
            mp = scratch / f"{mname}.sv"
            mp.write_text(base.replace(old, new))
            rc, res, secs = run_chunk(run, f"mut_{mname}", "directed", 5200, 0, 0.0, rtl=mp)
            muts.append(dict(mutation=mname, caught=rc != 0, mismatches=res.get("mismatches")))
            print(muts[-1], flush=True)
        n_rand = sum(r["ops"] for r in rows if r["mode"] == "random")
        m = dict(total_ops=ops_tot, random_ops=n_rand, directed_ops=ops_tot - n_rand, mismatches=mm_tot,
                 all_chunks_ok=all(r["ok"] for r in rows), coverage_total=cov, mutations=muts,
                 all_mutations_caught=all(x["caught"] for x in muts),
                 reference_checks=["blk vs integer ref", "blk vs golden BitSerialInt4Column", "golden vs integer ref",
                                   "acc", "ovf", "nan_out", "issue gap"])
        run.log_metrics(m)
        if not (m["all_chunks_ok"] and mm_tot == 0 and n_rand >= 100_000 * a.scale and m["all_mutations_caught"]):
            raise SystemExit(f"T5.2 FAILED: {m}")


if __name__ == "__main__":
    main()
