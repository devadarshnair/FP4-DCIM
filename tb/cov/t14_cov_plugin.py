"""pytest plugin (T1.4): run the existing cocotb regressions unchanged, but build the Verilator models with
functional-coverage monitors bound in and write one coverage .dat per test.

Usage:  T14_COV_DIR=<dir> PYTHONPATH=tb/cov python -m pytest -p t14_cov_plugin tb/cocotb/<test>.py -k ...

It wraps ``cocotb_tools.runner.get_runner`` (every testbench imports it inside its ``_run``): ``build`` gets
the monitor/bind sources for its ``hdl_toplevel`` plus ``--coverage-user --coverage-line``; ``test`` gets
``+verilator+coverage+file+<T14_COV_DIR>/<test id>.dat``. Test files, RTL and scoreboards are not touched.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cocotb_tools.runner as _r

COV = Path(__file__).resolve().parent
_COLUMN = ["cov_column.sv", "bind_column.sv"]
_BS = ["cov_bs_column.sv", "bind_bs_column.sv", "cov_ctrl.sv"]
BINDS = {
    "column": _COLUMN,
    "column_top": _COLUMN + ["cov_ctrl.sv", "bind_column_top.sv"],
    "dpu_column": _COLUMN + ["cov_ctrl.sv", "bind_column_top.sv"],
    "macro_top": _COLUMN + ["cov_ctrl.sv", "bind_macro_top.sv"],
    "dpu_macro": _COLUMN + ["cov_ctrl.sv", "bind_macro_top.sv"],
    "bs_column_top": _BS + ["bind_bs_column_top.sv"],
    "bs_macro_top": _BS + ["bind_bs_macro_top.sv"],
}
_orig_get_runner = _r.get_runner


def _test_id() -> str:
    cur = os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" ")[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", cur.split("/")[-1])


def _get_runner(sim: str):
    runner = _orig_get_runner(sim)
    build, test = runner.build, runner.test

    def cov_build(*args, **kw):
        top = kw.get("hdl_toplevel")
        if top in BINDS:
            kw["sources"] = list(kw.get("sources", [])) + [COV / f for f in BINDS[top]]
            kw["build_args"] = list(kw.get("build_args", [])) + ["--coverage-user", "--coverage-line"]
        return build(*args, **kw)

    def cov_test(*args, **kw):
        if kw.get("hdl_toplevel") in BINDS:
            out = Path(os.environ["T14_COV_DIR"]) / f"{_test_id()}.dat"
            kw["plusargs"] = list(kw.get("plusargs", []) or []) + [f"+verilator+coverage+file+{out}"]
        return test(*args, **kw)

    runner.build, runner.test = cov_build, cov_test
    return runner


_r.get_runner = _get_runner
