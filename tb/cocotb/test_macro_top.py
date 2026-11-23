"""v2 macro (32 columns, shared row drivers, 8 weight-set stores) vs the per-column reference models (WP-32).

MX: column_reference (== L2 == L1 Window); NV: nv_column_reference (== L1 Window). Weight block b of K
lives in set b (nb ≤ M); each token runs nb ops (clear on block 0). Env: MT_FMT (mxfp4|nvfp4), MT_M,
MT_TOKENS, MT_SRC (random|capture), MT_SEED.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scoreboard import ColumnStim, column_reference, int4sm_column_reference  # noqa: E402

from cimfp4.formats.codecs import e4m3_significand  # noqa: E402
from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.formats.nv_quant import quantize_nvfp4  # noqa: E402
from cimfp4.golden.l2_macro import nv_column_reference  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
GUARD, W = 7, 40


def operands(fmt, m, tokens, src, rng):
    k = 32 * m
    if src == "capture":
        from cimfp4.llm.capture import load_capture

        caps = sorted((REPO_ROOT / "data/captures").glob("*/prefill/*.npz"))
        qw, qa = load_capture(caps[int(rng.integers(0, len(caps)))], fmt)
        r0 = int(rng.integers(0, qw.codes.shape[0] - 32))
        ts = rng.integers(0, qa.codes.shape[0], tokens)
        import dataclasses

        def sl(q, rows, cols):
            nbk = 32 if fmt == "mxfp4" else 16
            return dataclasses.replace(
                q,
                codes=q.codes[rows][:, :cols],
                scale_exp=q.scale_exp[rows][:, : cols // nbk],
                scale_nan=q.scale_nan[rows][:, : cols // nbk],
                scale_e4m3=None if q.scale_e4m3 is None else q.scale_e4m3[rows][:, : cols // nbk],
            )

        return sl(qw, slice(r0, r0 + 32), k), sl(qa, ts, k)
    quant = quantize_mxfp4 if fmt == "mxfp4" else quantize_nvfp4
    xw = rng.standard_normal((32, k)) * np.exp(rng.normal(0, 0.5, (32, 1)))
    xa = rng.standard_t(3, (tokens, k)) * np.exp(rng.normal(0, 1, (tokens, 1)))
    return quant(xw), quant(xa)


def build(fmt, qw, qa, m):
    """Weights per (set, column) + op list + expected acc per (op, column)."""
    nb = qw.codes.shape[1] // 32  # 32-row ops per dot product
    sets = qw.codes.reshape(32, nb, 32).transpose(1, 0, 2)  # (set=b, col, row)
    if fmt == "mxfp4":
        e0 = qw.scale_exp.astype(np.int64).T  # (b, col)
        e1 = np.zeros_like(e0)
        u0 = u1 = np.ones_like(e0)
        xe0 = qa.scale_exp.astype(np.int64)  # (t, b)
        xe1 = np.zeros_like(xe0)
        xu0 = xu1 = np.ones_like(xe0)
        maxw = qw.scale_exp.max(axis=1).astype(np.int64)
        maxx = qa.scale_exp.max(axis=1).astype(np.int64)
    else:
        uw, kw = e4m3_significand(qw.scale_e4m3)  # (32, 2nb)
        ux, kx = e4m3_significand(qa.scale_e4m3)
        e0, e1 = kw[:, 0::2].T, kw[:, 1::2].T
        u0, u1 = uw[:, 0::2].T, uw[:, 1::2].T
        xe0, xe1, xu0, xu1 = kx[:, 0::2], kx[:, 1::2], ux[:, 0::2], ux[:, 1::2]
        maxw, maxx = kw.max(axis=1), kx.max(axis=1)
    ops = []
    for t in range(qa.codes.shape[0]):
        for b in range(nb):
            ops.append(
                dict(
                    x=qa.codes[t, 32 * b : 32 * (b + 1)],
                    set=b,
                    clear=b == 0,
                    t=t,
                    b=b,
                    e_x0=int(xe0[t, b]),
                    e_x1=int(xe1[t, b]),
                    u_x0=int(xu0[t, b]),
                    u_x1=int(xu1[t, b]),
                    max_e_x=int(maxx[t]),
                )
            )
    if os.environ.get("MT_PATTERN"):  # T1.6b: mixed set pattern, random clear / e_x / max_e_x / NaN
        pat = [int(v) for v in os.environ["MT_PATTERN"].split(",")]
        assert max(pat) < nb, (pat, nb)
        prng = np.random.default_rng(int(os.environ.get("MT_SEED", "0")) + 1000)
        n_pat = int(os.environ.get("MT_NOPS", "2600"))
        base = ops
        ops = []
        for j in range(n_pat):
            b0 = base[int(prng.integers(0, len(base)))]
            b = pat[j % len(pat)]
            t = b0["t"]
            o = dict(b0, set=b, b=b, x=qa.codes[t, 32 * b : 32 * (b + 1)], clear=bool(prng.random() < 0.3))
            o["e_x0"], o["e_x1"] = int(xe0[t, b]), int(xe1[t, b])
            o["u_x0"], o["u_x1"] = int(xu0[t, b]), int(xu1[t, b])
            if fmt == "mxfp4":
                o["e_x0"] += int(prng.integers(-3, 4))
                o["max_e_x"] = o["e_x0"] + int(prng.integers(0, 4))
                o["nan"] = bool(prng.random() < 0.05)
            else:
                o["max_e_x"] = int(maxx[t]) + int(prng.integers(0, 4))
                o["nan"] = False
            ops.append(o)
    n = len(ops)
    exp = np.zeros((n, 32), np.int64)
    for c in range(32):
        w = np.array([sets[o["b"], c] for o in ops], np.uint8)
        x = np.array([o["x"] for o in ops], np.uint8)
        clr = np.array([o["clear"] for o in ops])
        if fmt == "mxfp4":
            st = ColumnStim(
                w=w,
                x=x,
                e_w=np.array([e0[o["b"], c] for o in ops]),
                e_x=np.array([o["e_x0"] for o in ops]),
                max_e_w=np.full(n, maxw[c]),
                max_e_x=np.array([o["max_e_x"] for o in ops]),
                clear=clr,
                nan=np.array([bool(o.get("nan", False)) for o in ops]),
            )
            if os.environ.get("MT_ARITH") == "int4sm":  # T5.4: same stimulus, codes read as {sign, mag[2:0]}
                exp[:, c] = int4sm_column_reference(st, W, GUARD)
            else:
                exp[:, c] = column_reference(st, W, GUARD)[0]
        else:
            exp[:, c] = nv_column_reference(
                w,
                x,
                np.array([[u0[o["b"], c], u1[o["b"], c]] for o in ops]),
                np.array([[e0[o["b"], c], e1[o["b"], c]] for o in ops]),
                np.array([[o["u_x0"], o["u_x1"]] for o in ops]),
                np.array([[o["e_x0"], o["e_x1"]] for o in ops]),
                np.full(n, maxw[c]),
                np.array([o["max_e_x"] for o in ops]),
                clr,
                W,
                GUARD,
            )
    return dict(sets=sets, e0=e0, e1=e1, u0=u0, u1=u1, maxw=maxw, nb=nb), ops, exp


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer

    @cocotb.test()
    async def macro_matches(dut):
        fmt, m = os.environ["MT_FMT"], int(os.environ["MT_M"])
        rng = np.random.default_rng(int(os.environ.get("MT_SEED", "0")))
        qw, qa = operands(
            fmt, m, int(os.environ.get("MT_TOKENS", "64")), os.environ.get("MT_SRC", "random"), rng
        )
        wt, ops, exp = build(fmt, qw, qa, m)
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        for s in ("rst_n", "wr_en", "sc_wr_en", "maxw_wr_en", "op_valid"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        for b in range(wt["nb"]):
            for c in range(32):
                for k in range(4):
                    word = sum(int(wt["sets"][b, c, 8 * k + j]) << (4 * j) for j in range(8))
                    await FallingEdge(dut.clk)
                    dut.wr_en.value, dut.wr_group.value = 1, c // 4
                    dut.wr_addr.value, dut.wr_data.value = b * 16 + (c % 4) * 4 + k, word
                await FallingEdge(dut.clk)
                dut.wr_en.value = 0
                dut.sc_wr_en.value, dut.sc_wr_set.value, dut.sc_wr_col.value = 1, b, c
                dut.sc_wr_e0.value, dut.sc_wr_e1.value = (
                    int(wt["e0"][b, c]) & 0xFF,
                    int(wt["e1"][b, c]) & 0xFF,
                )
                dut.sc_wr_u0.value, dut.sc_wr_u1.value, dut.sc_wr_nan.value = (
                    int(wt["u0"][b, c]),
                    int(wt["u1"][b, c]),
                    0,
                )
        for c in range(32):
            await FallingEdge(dut.clk)
            dut.sc_wr_en.value = 0
            dut.maxw_wr_en.value, dut.maxw_wr_col.value, dut.maxw_wr_val.value = (
                1,
                c,
                int(wt["maxw"][c]) & 0xFF,
            )
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0
        got = [[] for _ in range(32)]
        i, drain, mask = 0, 0, (1 << W) - 1
        cyc, acc_cyc = 0, []  # cycle index of every accepted op (back-to-back issue measurement)
        while i < len(ops) or drain < 16:  # bit-serial baselines need a longer drain
            await FallingEdge(dut.clk)
            cyc += 1
            av = int(dut.acc_valid.value)
            if av:
                a = int(dut.acc.value)
                for c in range(32):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
            if i < len(ops):
                o = ops[i]
                dut.op_valid.value, dut.op_nv.value = 1, int(fmt == "nvfp4")
                dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(o["x"]))
                dut.op_e_x0.value, dut.op_e_x1.value = o["e_x0"] & 0xFF, o["e_x1"] & 0xFF
                dut.op_u_x0.value, dut.op_u_x1.value = o["u_x0"], o["u_x1"]
                dut.op_max_e_x.value, dut.op_x_nan.value = o["max_e_x"] & 0xFF, int(o.get("nan", False))
                dut.op_set.value, dut.op_clear.value = o["set"], int(o["clear"])
                await Timer(1, unit="ns")
                if int(dut.op_ready.value):
                    i += 1
                    acc_cyc.append(cyc)
            else:
                dut.op_valid.value = 0
                drain += 1
        gaps = np.diff(acc_cyc)
        # steady state = consecutive ops on the SAME weight set; a set switch costs a weight-set load
        # (>= 17 cycles) and is a workload property, not the per-op issue rate.
        same = np.array([ops[j + 1]["set"] == ops[j]["set"] for j in range(len(ops) - 1)], bool)
        exp_gap = int(os.environ.get("MT_EXPECT_MIN_GAP", "0"))
        if exp_gap:
            assert int(gaps.min()) >= exp_gap, f"gap {int(gaps.min())} < {exp_gap}"
            if int(os.environ.get("MT_M", "1")) == 1:
                assert same.all(), "M=1 must be all same-set"
            if same.any():
                assert int(gaps[same].min()) == exp_gap, f"min cycles/op {int(gaps[same].min())} != {exp_gap}"
        dut._log.info(
            f"cycles/op: min gap {int(gaps.min())}, same-set min {int(gaps[same].min()) if same.any() else -1}, "
            f"same-set n={int(same.sum())}, mean {float(gaps.mean()):.3f} (set stalls included)"
        )
        g = np.array(got, np.int64).T
        assert g.shape == exp.shape, (g.shape, exp.shape)
        mism = int((g != exp).sum())
        assert mism == 0, f"{mism} mismatches"
        assert int(dut.ovf.value) == 0
        if os.environ.get("MT_PATTERN"):
            n_sw = int(dut.set_switches.value)
            want = sum(1 for j in range(len(ops)) if j == 0 or ops[j]["set"] != ops[j - 1]["set"])
            assert n_sw == want, (n_sw, want)
            assert same.any() and not same.all(), "pattern lacks same-set runs or switches"
        dut._log.info(
            f"macro {fmt} M={m}: {len(ops)} ops x 32 columns, 0 mismatches, "
            f"{int(dut.set_switches.value)} set switches"
        )

except ImportError:
    pass


def _run(fmt, m, store, tmp_path, src="random", tokens=64, seed=0, pattern=None):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        [REPO_ROOT / "rtl/gen/col_tree_oc_w9_r32.sv"]
        + [
            REPO_ROOT / "rtl/src" / f
            for f in (
                "cim_icg.sv",
                "row_driver.sv",
                "leaf_mult.sv",
                "column.sv",
                "wset_sram.sv",
                "wset_latch.sv",
                "macro_top.sv",
            )
        ]
        + [REPO_ROOT / "rtl/baselines/digital_mac/generic_leaf.sv", SRAM_V]
    )
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel="macro_top",
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters={"M": m, "GUARD": GUARD, "STORE": store},
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel="macro_top",
        test_module="test_macro_top",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "MT_FMT": fmt,
            "MT_M": str(m),
            "MT_SRC": src,
            "MT_TOKENS": str(tokens),
            "MT_SEED": str(seed),
            **({"MT_PATTERN": pattern} if pattern else {}),
        },
    )
    assert get_results(xml) == (1, 0)


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)
HAS_CAP = bool(list((REPO_ROOT / "data/captures").glob("*/prefill/*.npz")))


@needs
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("m", [1, 16])
def test_macro_sram(fmt, m, tmp_path):
    _run(fmt, m, 0, tmp_path)


@needs
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_macro_latch(fmt, tmp_path):
    _run(fmt, 16, 1, tmp_path)


@needs
@pytest.mark.skipif(not HAS_CAP, reason="no captures")
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_macro_capture(fmt, tmp_path):
    _run(fmt, 16, 0, tmp_path, src="capture", tokens=32, seed=2)


MIXED_PATTERN = "0,0,0,1,1,2,0,0,3,3,3,3,1"


@needs
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("store", [0, 1], ids=["SRAM", "LATCH"])
def test_macro_pattern(fmt, store, tmp_path):
    """T1.6b: mixed set-pattern stream (v2 macro; random clear/e_x/max_e_x, NaN for MX)."""
    _run(fmt, 4, store, tmp_path, tokens=8, seed=11, pattern=MIXED_PATTERN)


@needs
@pytest.mark.skipif("T13_FMT" not in os.environ, reason="T13 large regression: driver-only (env T13_*)")
def test_macro_large(tmp_path):
    """T1.3: >= 1e6 column-ops per format; parameters from env (driven by experiments/t13_macro_regression)."""
    e = os.environ
    _run(
        e["T13_FMT"],
        int(e["T13_M"]),
        int(e.get("T13_STORE", "0")),
        tmp_path,
        src=e.get("T13_SRC", "random"),
        tokens=int(e["T13_TOKENS"]),
        seed=int(e["T13_SEED"]),
        pattern=e.get("T13_PATTERN"),
    )
