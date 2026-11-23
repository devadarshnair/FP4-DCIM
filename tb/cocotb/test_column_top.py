"""Column group with storage (WP-64): bit-exact vs the scoreboard (== L2) with set switches.

Configs: STORE ∈ {SRAM, LATCH} × M ∈ {1, 16}. Weight block b of a dot product lives in set b mod M
(K-blocks stored in successive sets), so M=16 switches sets on every op (single-buffer stall).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scoreboard import ColumnStim, column_reference  # noqa: E402

from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
NCOL, GUARD, W = 4, 7, 40


def capture_program(rng, m, n_ops):
    """Weight sets and activations taken from real WP-14 captures (MXFP4)."""
    from cimfp4.llm.capture import load_capture

    caps = sorted((REPO_ROOT / "data" / "captures").glob("*/prefill/*.npz"))
    qw, qa = load_capture(caps[int(rng.integers(0, len(caps)))], "mxfp4")
    r0 = int(rng.integers(0, qw.codes.shape[0] - NCOL))
    nb = min(qw.num_blocks, m)
    sets = np.zeros((m, NCOL, 32), np.uint8)
    ew = np.zeros((m, NCOL), np.int64)
    for s in range(m):
        b = s % nb
        sets[s] = qw.blocked_codes()[r0 : r0 + NCOL, b]
        ew[s] = qw.scale_exp[r0 : r0 + NCOL, b]
    maxw = ew.max(axis=0)
    ops = []
    while len(ops) < n_ops:
        t = int(rng.integers(0, qa.codes.shape[0]))
        ex = qa.scale_exp[t, :nb].astype(int)
        for b in range(nb):
            ops.append(
                dict(
                    x=qa.blocked_codes()[t, b],
                    e_x=int(ex[b]),
                    max_e_x=int(ex.max()),
                    set=b % m,
                    clear=b == 0,
                    nan=False,
                )
            )
    return sets, ew, maxw, ops[:n_ops]


def make_program(rng, m, n_ops):
    """Weight sets (codes + exponents) and an op list of dot products."""
    sets = quantize_mxfp4(rng.standard_normal((m * NCOL, 32))).codes.reshape(m, NCOL, 32)
    ew = rng.integers(-6, 3, (m, NCOL))
    maxw = ew.max(axis=0)
    ops = []
    while len(ops) < n_ops:
        nb = int(rng.integers(1, min(2**GUARD, 64) + 1))
        q = quantize_mxfp4(rng.standard_t(3, (1, nb * 32)) * np.exp(rng.normal(0, 1)))
        ex = np.clip(q.scale_exp[0].astype(int) + rng.integers(-3, 4), -127, 127)
        for b in range(nb):
            ops.append(
                dict(
                    x=q.codes[0, b * 32 : (b + 1) * 32],
                    e_x=int(ex[b]),
                    max_e_x=int(ex.max()),
                    set=b % m,
                    clear=b == 0,
                    nan=False,
                )
            )
    return sets, ew, maxw, ops[:n_ops]  # a truncated last dot product is fine: compared op by op


def pattern_program(rng, m, n_ops, pattern):
    """Mixed weight-set pattern (T1.6b): `pattern` (set ids < m) repeated; random clear / e_x / max_e_x / NaN.

    Exercises back-to-back same-set runs and set switches in one stream (regression for the back-to-back fix).
    """
    assert max(pattern) < m, (pattern, m)
    sets = quantize_mxfp4(rng.standard_normal((m * NCOL, 32))).codes.reshape(m, NCOL, 32)
    ew = rng.integers(-6, 3, (m, NCOL))
    maxw = ew.max(axis=0)
    ops = []
    for j in range(n_ops):
        q = quantize_mxfp4(rng.standard_t(3, (1, 32)) * np.exp(rng.normal(0, 1)))
        e_x = int(rng.integers(-8, 9))
        ops.append(
            dict(
                x=q.codes[0],
                e_x=e_x,
                max_e_x=e_x + int(rng.integers(0, 4)),
                set=int(pattern[j % len(pattern)]),
                clear=bool(rng.random() < 0.3),
                nan=bool(rng.random() < 0.05),
            )
        )
    return sets, ew, maxw, ops


def expected(sets, ew, maxw, ops):
    out = []
    for c in range(NCOL):
        st = ColumnStim(
            w=np.array([sets[o["set"], c] for o in ops], np.uint8),
            x=np.array([o["x"] for o in ops], np.uint8),
            e_w=np.array([ew[o["set"], c] for o in ops]),
            e_x=np.array([o["e_x"] for o in ops]),
            max_e_w=np.full(len(ops), maxw[c]),
            max_e_x=np.array([o["max_e_x"] for o in ops]),
            clear=np.array([o["clear"] for o in ops]),
            nan=np.array([o["nan"] for o in ops]),
        )
        out.append(column_reference(st, W, GUARD)[0])
    return np.stack(out, axis=1)  # (n_ops, NCOL)


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer

    @cocotb.test()
    async def column_top_matches(dut):
        m, n_ops = int(os.environ["CT_M"]), int(os.environ["CT_NOPS"])
        rng = np.random.default_rng(int(os.environ.get("CT_SEED", "0")))
        if os.environ.get("CT_PROGRAM"):
            from cimfp4.stimulus.writers import load_program

            pg = load_program(os.environ["CT_PROGRAM"])
            sets, ew, maxw = pg["sets"], pg["ew"], pg["maxw"]
            ops = [
                dict(
                    x=pg["x"][i],
                    e_x=int(pg["e_x"][i]),
                    max_e_x=int(pg["max_e_x"][i]),
                    set=int(pg["set"][i]),
                    clear=bool(pg["clear"][i]),
                    nan=False,
                )
                for i in range(len(pg["x"]))
            ]
            m = sets.shape[0]
        elif os.environ.get("CT_PATTERN"):
            pat = [int(v) for v in os.environ["CT_PATTERN"].split(",")]
            sets, ew, maxw, ops = pattern_program(rng, m, n_ops, pat)
        else:
            prog = capture_program if os.environ.get("CT_CAPTURE") == "1" else make_program
            sets, ew, maxw, ops = prog(rng, m, n_ops)
        exp = expected(sets, ew, maxw, ops)
        cocotb.start_soon(Clock(dut.clk, int(os.environ.get("CT_CLK_PS", "10000")), unit="ps").start())
        for s in ("rst_n", "wr_en", "ew_wr_en", "maxw_wr_en", "op_valid"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        # load weight codes, exponents and anchors
        for s in range(m):
            for c in range(NCOL):
                for k in range(4):
                    word = sum(int(sets[s, c, 8 * k + j]) << (4 * j) for j in range(8))
                    await FallingEdge(dut.clk)
                    dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, s * 16 + c * 4 + k, word
                await FallingEdge(dut.clk)
                dut.wr_en.value = 0
                dut.ew_wr_en.value, dut.ew_wr_set.value, dut.ew_wr_col.value = 1, s, c
                dut.ew_wr_val.value, dut.ew_wr_nan.value = int(ew[s, c]) & 0xFF, 0
        for c in range(NCOL):
            await FallingEdge(dut.clk)
            dut.ew_wr_en.value = 0
            dut.maxw_wr_en.value, dut.maxw_wr_col.value, dut.maxw_wr_val.value = 1, c, int(maxw[c]) & 0xFF
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0
        # run ops
        from cocotb.utils import get_sim_time

        t_start = get_sim_time("ns")
        got = [[] for _ in range(NCOL)]
        i, stalls, guard = 0, 0, 0
        mask = (1 << W) - 1
        cyc, acc_cyc = 0, []
        while i < len(ops) or guard < 16:  # drain: bit-serial latency is 5 + 3 cycles
            await FallingEdge(dut.clk)
            cyc += 1
            av = int(dut.acc_valid.value)
            if av:
                a = int(dut.acc.value)
                for c in range(NCOL):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
            if i < len(ops):
                o = ops[i]
                dut.op_valid.value = 1
                dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(o["x"]))
                dut.op_e_x.value, dut.op_max_e_x.value = o["e_x"] & 0xFF, o["max_e_x"] & 0xFF
                dut.op_x_nan.value, dut.op_set.value, dut.op_clear.value = int(o["nan"]), o["set"], int(o["clear"])
                # op_ready is sampled CT_READY_NS after the falling edge (default 1 ns; T3.3 routed-SDF GLS uses a
                # larger value because the port->op_ready path carries routed input-buffer and wire delays)
                await Timer(float(os.environ.get("CT_READY_NS", "1")), unit="ns")
                if int(dut.op_ready.value):
                    i += 1
                    acc_cyc.append(cyc)
                else:
                    stalls += 1
            else:
                dut.op_valid.value = 0
                guard += 1
        t_end = get_sim_time("ns")
        if os.environ.get("CT_WINDOW_OUT"):
            import json

            Path(os.environ["CT_WINDOW_OUT"]).write_text(
                json.dumps(
                    {"t_start_ns": t_start, "t_end_ns": t_end, "n_ops": len(ops), "stall_cycles": stalls}
                )
            )
        got_a = np.array(got, dtype=np.int64).T
        assert got_a.shape == exp.shape, (got_a.shape, exp.shape)
        mism = int((got_a != exp).sum())
        assert mism == 0, f"{mism} mismatches"
        n_switch = int(dut.set_switches.value)
        want_switch = (
            1 if m == 1 else sum(1 for j in range(len(ops)) if j == 0 or ops[j]["set"] != ops[j - 1]["set"])
        )
        assert n_switch == want_switch, (n_switch, want_switch)
        if os.environ.get("CT_PATTERN"):  # mixed pattern must contain both same-set runs and switches
            assert want_switch > 1 and any(
                ops[j + 1]["set"] == ops[j]["set"] for j in range(len(ops) - 1)
            ), "pattern lacks same-set runs or switches"
        gaps = [b - a for a, b in zip(acc_cyc, acc_cyc[1:], strict=False)]
        same = [ops[j + 1]["set"] == ops[j]["set"] for j in range(len(ops) - 1)]
        sg = [g for g, s_ in zip(gaps, same, strict=True) if s_]
        exp_gap = int(os.environ.get("CT_EXPECT_MIN_GAP", "0"))
        if exp_gap:  # steady state = consecutive ops on the same weight set (set switches are load stalls)
            assert min(gaps) >= exp_gap, f"gap {min(gaps)} < {exp_gap}"
            if m == 1:
                assert len(sg) == len(gaps)
            if sg:
                assert min(sg) == exp_gap, f"min cycles/op {min(sg)} != {exp_gap}"
        dut._log.info(f"cycles/op: same-set min {min(sg) if sg else -1} (n={len(sg)}), mean all {sum(gaps) / len(gaps):.3f}")
        assert stalls >= n_switch  # every switch stalls the single buffer
        dut._log.info(f"M={m}: {len(ops)} ops, {n_switch} set switches, {stalls} stall cycles, 0 mismatches")

except ImportError:
    pass


def _run(store, m, n_ops, tmp_path, seed=0, pattern=None):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        [
            REPO_ROOT / "rtl/include/cim_pkg.sv",
            REPO_ROOT / "rtl/src/cim_icg.sv",
            REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv",
        ]
        + [
            REPO_ROOT / "rtl/src" / f
            for f in (
                "row_driver.sv",
                "leaf_mult.sv",
                "column.sv",
                "wset_latch.sv",
                "wset_sram.sv",
                "column_top.sv",
            )
        ]
        + [SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(
        includes=[REPO_ROOT / "rtl" / "include"],
        sources=srcs,
        hdl_toplevel="column_top",
        build_dir=tmp_path / "b",
        always=True,
        parameters={"M": m, "STORE": 0 if store == "SRAM" else 1, "GUARD": GUARD},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="column_top",
        test_module="test_column_top",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "CT_M": str(m),
            "CT_NOPS": str(n_ops),
            "CT_SEED": str(seed),
            **({"CT_PATTERN": pattern} if pattern else {}),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(),
    reason="verilator / sky130 OpenRAM models missing",
)


@needs
@pytest.mark.parametrize("store", ["SRAM", "LATCH"])
@pytest.mark.parametrize("m", [1, 16])
def test_column_top(store, m, tmp_path):
    _run(store, m, int(os.environ.get("CT_NOPS", "10000")), tmp_path)


@needs
@pytest.mark.slow
@pytest.mark.parametrize("store", ["SRAM", "LATCH"])
@pytest.mark.parametrize("m", [1, 16])
def test_column_top_1e5(store, m, tmp_path):
    _run(store, m, 100_000, tmp_path, seed=7)


MIXED_PATTERN = "0,0,0,1,1,2,0,0,3,3,3,3,1"


@needs
@pytest.mark.parametrize("store", ["SRAM", "LATCH"])
def test_column_top_pattern(store, tmp_path):
    """T1.6b: mixed set-pattern stream (v2 column; random clear/e_x/max_e_x/NaN)."""
    _run(store, 4, 2600, tmp_path, seed=11, pattern=MIXED_PATTERN)
