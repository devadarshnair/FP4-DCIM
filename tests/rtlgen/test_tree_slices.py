"""T1.4: tree slicing for compositional formal, and the algebraic function-extraction checker."""

import shutil
import subprocess

import pytest

from cimfp4.rtlgen.alg_extract import extract, poly_spec
from cimfp4.rtlgen.tree import build_column_tree, emit_sv
from cimfp4.rtlgen.tree_slices import emit_composed_sv, emit_slice_check_sv, emit_slice_sv


@pytest.mark.parametrize("w_in, scheme", [(9, "oc"), (9, "tc"), (1, "unsigned")])
def test_slices_partition_cells(w_in, scheme):
    nl = build_column_tree(w_in, scheme)
    spans = sorted(nl.slices.values())
    assert spans[0][0] == 0 and spans[-1][1] == len(nl.cells)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))  # contiguous, disjoint
    for s in nl.slices:  # emitters check slice closure (no dangling nets) by assertion
        assert f"module {nl.name}_slice_{s}" in emit_slice_sv(nl, s)
        assert "assign ok" in emit_slice_check_sv(nl, s)
    assert f"module {nl.name}_composed" in emit_composed_sv(nl)


def _spec(w_in, scheme, rows):
    t = []
    for r in rows:
        for b in range(w_in):
            k = -(1 << b) if scheme != "unsigned" and b == w_in - 1 else 1 << b
            t.append(((f"in_bits[{r * w_in + b}]",), k))
        if scheme == "oc":
            t.append(((f"s_bits[{r}]",), 1))
    return t


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not on PATH")
@pytest.mark.parametrize("mutate", [False, True])
def test_alg_extract_monolith(tmp_path, mutate):
    nl = build_column_tree(9, "oc")
    sv = emit_sv(nl)
    if mutate:  # first FA sum a ^ b ^ c -> a ^ b | c
        i = sv.index(" ^ ", sv.index(" ^ ") + 1)
        sv = sv[:i] + " | " + sv[i + 3 :]
    (tmp_path / "t.sv").write_text(sv)
    js = tmp_path / "t.json"
    subprocess.run(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog -sv {tmp_path / 't.sv'}; hierarchy -top {nl.name}; proc; flatten; "
            f"techmap; opt_clean; write_json {js}",
        ],
        check=True,
    )
    w = len(nl.outputs["m_s"])
    got, _ = extract(js, nl.name, [(p, i, 1 << i) for p in ("m_s", "m_c") for i in range(w)], w)
    assert (got == poly_spec(_spec(9, "oc", range(32)), w)) is (not mutate)
