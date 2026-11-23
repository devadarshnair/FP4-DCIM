import re, subprocess, sys
from cimfp4.utils.runlog import start_run

JOBS = {
 "dpu_1e5": ["tb/cocotb/test_dpu_macro.py", "-k", "1e5"],
 "bs_1e5": ["tb/cocotb/test_bs_macro.py", "-k", "1e5"],
}
for name, args in JOBS.items():
    cmd = ["python", "-m", "pytest", *args, "-v", "-rs", "-p", "no:cacheprovider"]
    cfg = {"job": name, "cmd": " ".join(cmd), "seed": 9, "tokens": 400, "M": 16,
           "purpose": "T1.1 provenance rerun (WP-72 DPU 1e5 / INT5 bs 1e5 post-B15), clean tree"}
    with start_run("macro_regression", cfg, paper=True) as run:
        r = subprocess.run(cmd, capture_output=True, text=True)
        run.path(f"pytest_{name}.log").write_text(r.stdout + r.stderr)
        m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|deselected)", r.stdout.splitlines()[-1])}
        m.setdefault("skipped", 0)
        run.log_metrics({name: {**m, "column_ops_per_format": 204800, "returncode": r.returncode}})
        print(name, m, r.returncode, run.dir)
        if r.returncode: sys.exit(1)
