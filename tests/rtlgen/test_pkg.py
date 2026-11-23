import re
import shutil
import subprocess

import pytest

from cimfp4.rtlgen.pkg import CimParams, emit_sv, signed_bits, write_all
from cimfp4.utils.provenance import REPO_ROOT

# R03 §8 expected widths (hand-checked), including the NV exponent-sum correction (6 b)
EXPECTED = {
    "ROW_MAG_BITS": 4,
    "ROW_MAG3_BITS": 6,
    "ROW_BCAST_WIRES": 11,
    "PROD_MAG_BITS": 8,
    "PROD_TC_BITS": 9,
    "MX_SUM_U_BITS": 13,
    "MX_SUM_S_BITS": 14,
    "NV_SUM_U_BITS": 12,
    "NV_SUM_S_BITS": 13,
    "MX_EXP_SUM_BITS": 9,
    "NV_SCALE_SIG_BITS": 4,
    "NV_SCALE_PROD_BITS": 8,
    "NV_TERM_BITS": 20,
    "NV_EXP_SUM_BITS": 6,
    "WINDOW_BITS": 40,
    "PIPE_FLOPS_PER_COL_MX": 28,
    "SET_IDX_BITS": 4,
    "SHIFT_BITS": 6,
}


def test_signed_bits():
    assert signed_bits(-144, 144) == 9
    assert signed_bits(-256, 255) == 9 and signed_bits(-257, 0) == 10
    assert signed_bits(-20, 8) == 6


def test_derived_widths_match_spec():
    d = CimParams().derived
    for k, v in EXPECTED.items():
        assert d[k] == v, k


def test_generated_package_matches_python_constants(tmp_path):
    p = CimParams()
    sv = emit_sv(p)
    got = {m.group(1): int(m.group(2)) for m in re.finditer(r"localparam int (\w+) = (-?\d+);", sv)}
    for k, v in p.derived.items():
        if not k.startswith("_"):
            assert got[k] == v, k
    assert got["WEIGHT_SETS"] == p.weight_sets


def test_committed_package_is_up_to_date(tmp_path):
    write_all(root=tmp_path)
    assert (tmp_path / "rtl/include/cim_pkg.sv").read_text() == (
        REPO_ROOT / "rtl/include/cim_pkg.sv"
    ).read_text()


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_package_lints_clean():
    r = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "rtl/lint/waivers.vlt", "rtl/include/cim_pkg.sv"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_parameter_changes_propagate():
    p = CimParams(weight_sets=64, window=48)
    assert p.derived["SET_IDX_BITS"] == 6 and p.derived["WINDOW_BITS"] == 48
