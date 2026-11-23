"""T1.4 directed test: per-op MX/NV format toggling, weight-scale NaN and NaN propagation at macro level.

Closes coverage bins that the format-homogeneous macro regressions cannot hit (docs/spec/coverage_plan.md):
consecutive accepted ops with different `op_nv` (with and without a set switch, back-to-back), a weight-scale
NaN bit in the scale register file, and the `nan_out` flag of every column (the other macro tests check
`acc` only). Each dot product (clear .. next clear) is format-homogeneous; dot products of both formats
share the weight sets, whose scale records {e0, e1, u0, u1} are read as UE8M0 e0 (MX) or as (k, u) pairs
(NV). Expected acc: `column_reference` (MX) / `nv_column_reference` (NV) per dot product. Expected NaN
(MX and NV, T1.5): sticky OR over the dot product of op_x_nan | weight-scale NaN (`nv_column_nan` for NV;
weight-scale NaN records are written from E4M3 0x7F/0xFF via `rf_record`, which rejects reserved 0x80..0xFE).

Scale corners (T1.3 gaps): MX block exponents ±8 around 0 and a saturated anchor (d >= W-1); NV k at both
ends of [-9, 5]; E4M3 significands 0 (zero scale), 1..7 (subnormal) and 15 (max); weight-scale NaN (MX).
UE8M0 0xFE (+127) and 0x00 (-127) activation exponents in "e8" dot products (weights: test_column corners).

Env: MM_DOTS (default 300), MM_SEED, MM_MX_ONLY. Tops: macro_top (v2), dpu_macro (baseline (c), generic leaf),
bs_macro_top (baseline (a) INT5, MX-only stream). Also: weight UE8M0 0xFE/0x00 in column XCOL, "far" anchors
(d = 48 / 39 / 30) and one 130-block dot product that fills the top window bits.
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

from cimfp4.formats.codecs import e4m3_scale_fields  # noqa: E402
from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.golden.l2_macro import nv_column_nan, nv_column_reference  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

SRAM_V = Path.home() / ".ciel/sky130A/libs.ref/sky130_sram_macros/verilog/sky130_sram_1kbyte_1rw1r_32x256_8.v"
GUARD, W, M, NCOL = 7, 40, 4, 32
NAN_SET, NAN_COL = 1, 5  # one weight-scale NaN record
XCOL = 31  # column with UE8M0 +-127 weight exponents (its anchor is 127: all its other terms saturate)
FAR_EX = (-8, 1, 10)  # "far" MX ops, anchor 40: d = 48 (saturated), 39 (= W-1), 30 (right shift)
XSET = M - 1  # "extreme" set: all weights +6, e0 = e1 = 2 (column max), u0 = u1 = 15
P6, N6 = 0b0111, 0b1111
ZERO_COL, ZERO_K = 8, 4  # column whose max-k anchor comes only from a zero-scale (E4M3 0x00, u=0) block
K9_COL = 7  # column whose scale-RF k fields are all -9 (subnormal / zero E4M3 scales)
NAN_REC = [
    (1, 5),
    (0, 10),
    (1, 11),
    (2, 12),
    (0, 13),
    (2, 14),
    (1, 15),
]  # (set, col) weight-scale NaN records


def rf_record(code0, code1):
    """Stimulus WRITER for one scale-RF record of a 32-element block (two 16-element E4M3 block scales).

    Returns (k0, k1, u0, u1, nan). `nan` = OR of the two NaN states (E4M3 0x7F/0xFF), the value to drive on
    `sc_wr_nan`; NaN blocks store u = k = 0. Reserved codes 0x80..0xFE are REJECTED (ValueError from
    `e4m3_scale_fields`): the RF keeps magnitudes only, so a sign would be dropped silently (numerics.md).
    """
    u, k, n = e4m3_scale_fields(np.array([code0, code1], np.uint8))
    return int(k[0]), int(k[1]), int(u[0]), int(u[1]), bool(n.any())


def anchor_maxw(e0, e1):
    """Static per-column max_e_w writer rule (numerics.md): max over ALL blocks of the column, zero-scale
    blocks (u = 0) and NaN blocks (stored k = 0) included; u is deliberately not consulted."""
    return np.maximum(e0, e1).max(axis=0)


def program(rng, n_dots, mx_only=False):
    codes = quantize_mxfp4(rng.standard_normal((M * NCOL, 32))).codes.reshape(M, NCOL, 32)
    e0, e1 = rng.integers(-6, 3, (M, NCOL)), rng.integers(-6, 3, (M, NCOL))
    u0, u1 = rng.integers(0, 16, (M, NCOL)), rng.integers(0, 16, (M, NCOL))
    wnan = np.zeros((M, NCOL), bool)
    codes[XSET], e0[XSET], e1[XSET], u0[XSET], u1[XSET] = P6, 2, 2, 15, 15
    e0[XSET, XCOL], e0[0, XCOL] = 127, -127  # UE8M0 0xFE / 0x00 weight exponents (column XCOL only)
    # NV scale corners: k = -9 for ~12 % of the blocks (u = 0 zero scale or 1..7 subnormal), one column all -9
    m9 = rng.random((2, M, NCOL)) < 0.12
    for h, ee, uu in ((0, e0, u0), (1, e1, u1)):
        ee[m9[h] & (np.arange(M)[:, None] != XSET) & (np.arange(NCOL) != XCOL)] = -9
        uu[m9[h]] = np.minimum(uu[m9[h]], 7)  # k = -9 <=> E4M3 code 0x00..0x07 (u = 0..7)
    e0[:, K9_COL], e1[:, K9_COL] = -9, -9
    u0[:, K9_COL], u1[:, K9_COL] = rng.integers(0, 8, M), rng.integers(0, 8, M)
    u0[0, K9_COL] = 0  # one zero-scale block; its k = -9 still anchors the column
    e0[:, ZERO_COL], e1[:, ZERO_COL] = rng.integers(-6, 4, M), rng.integers(-6, 4, M)
    e0[0, ZERO_COL], u0[0, ZERO_COL] = ZERO_K, 0  # zero-scale block with the largest k of the column
    # weight-scale NaN records, realised through the writer from E4M3 codes 0x7F / 0xFF in one or both blocks
    for j, (ns, nc) in enumerate(NAN_REC):
        codes_e4 = [(0x7F, 0x38), (0x30, 0xFF), (0xFF, 0x7F)][j % 3]
        e0[ns, nc], e1[ns, nc], u0[ns, nc], u1[ns, nc], wnan[ns, nc] = rf_record(*codes_e4)
        assert wnan[ns, nc]
    # the writer anchor includes zero-scale blocks: column ZERO_COL is anchored by a u = 0 block
    maxw = anchor_maxw(e0, e1)
    assert maxw[ZERO_COL] == ZERO_K and u0[0, ZERO_COL] == 0 and maxw[K9_COL] == -9
    ops, dots = [], []
    set_now = 0
    for _ in range(n_dots):
        nv = (not mx_only) and bool(rng.random() < 0.5)
        kind = rng.random()
        if kind < 0.12:  # extreme dot product: |block sum| = 4608 (MX) / |half sums| = 2304, u = 15 x 15 (NV)
            start = len(ops)
            far = (
                not nv
            ) and kind < 0.04  # MX, anchor 40 above the block: d saturates, the term floors to -1
            e8 = (not nv) and 0.04 <= kind < 0.07  # MX UE8M0 extremes: e_x = 0xFE (+127) / 0x00 (-127)
            for b in range(int(rng.integers(1, 4))):
                x = np.full(32, P6 if rng.random() < 0.5 else N6, np.uint8)
                if far:
                    x[:] = N6
                ops.append(
                    dict(
                        x=x,
                        nv=nv,
                        set=XSET,
                        clear=b == 0,
                        e_x0=FAR_EX[b % 3]
                        if far
                        else (5 if nv else 8),  # NV k_x = 5: largest legal E4M3 exponent
                        e_x1=5 if nv else 0,
                        u_x0=15 if nv else 1,
                        u_x1=15 if nv else 1,
                        max_e_x=40 if far else (5 if nv else 8),
                        nan=bool(nv and rng.random() < 0.15),
                    )
                )
                if e8:
                    ops[-1]["e_x0"], ops[-1]["max_e_x"] = (127 if b % 2 == 0 else -127), 127
            set_now = XSET
            dots.append((start, len(ops), nv))
            continue
        nb = int(rng.integers(1, 5))
        lo, hi = (-9, 6) if nv else (-8, 9)  # NV k in [-9, 5] (E4M3, numerics.md); MX UE8M0 offsets
        ex0 = rng.integers(lo, hi, nb)
        ex1 = rng.integers(lo, hi, nb) if nv else np.zeros(nb, np.int64)
        mx = int(max(ex0.max(), ex1.max() if nv else -128)) + int(rng.integers(0, 4))
        start = len(ops)
        for b in range(nb):
            if rng.random() < 0.4:  # keep the set (same-set back-to-back) or switch
                set_now = int(rng.integers(0, M))
            q = quantize_mxfp4(rng.standard_t(3, (1, 32)) * np.exp(rng.normal(0, 1)))
            ops.append(
                dict(
                    x=q.codes[0],
                    nv=nv,
                    set=set_now,
                    clear=b == 0,
                    e_x0=int(ex0[b]),
                    e_x1=int(ex1[b]),
                    u_x0=int(rng.integers(0, 16)) if nv else 1,
                    u_x1=int(rng.integers(0, 16)) if nv else 1,
                    max_e_x=mx,
                    nan=bool(rng.random() < 0.05),
                )
            )
        dots.append((start, len(ops), nv))
    # one long MX dot product of maximal blocks at d = 0: |acc| reaches 130 x 4608 x 2^18 > 2^(W-3)
    start = len(ops)
    for b in range(130):
        ops.append(
            dict(
                x=np.full(32, P6, np.uint8),
                nv=False,
                set=XSET,
                clear=b == 0,
                e_x0=8,
                e_x1=0,
                u_x0=1,
                u_x1=1,
                max_e_x=8,
                nan=False,
            )
        )
    dots.append((start, len(ops), False))
    wt = dict(codes=codes, e0=e0, e1=e1, u0=u0, u1=u1, wnan=wnan, maxw=maxw)
    return wt, ops, dots


def expected(wt, ops, dots):
    n = len(ops)
    acc = np.zeros((n, NCOL), np.int64)
    nan = np.zeros((n, NCOL), bool)
    for a, b, nv in dots:
        sl = ops[a:b]
        k = b - a
        s = np.array([o["set"] for o in sl])
        x = np.array([o["x"] for o in sl], np.uint8)
        clr = np.array([o["clear"] for o in sl])
        mxx = np.array([o["max_e_x"] for o in sl])
        for c in range(NCOL):
            w = wt["codes"][s, c]
            if nv:
                nan[a:b, c] = nv_column_nan(wt["wnan"][s, c], np.array([o["nan"] for o in sl]), clr)
                acc[a:b, c] = nv_column_reference(
                    w,
                    x,
                    np.stack([wt["u0"][s, c], wt["u1"][s, c]], 1),
                    np.stack([wt["e0"][s, c], wt["e1"][s, c]], 1),
                    np.array([[o["u_x0"], o["u_x1"]] for o in sl]),
                    np.array([[o["e_x0"], o["e_x1"]] for o in sl]),
                    np.full(k, wt["maxw"][c]),
                    mxx,
                    clr,
                    W,
                    GUARD,
                )
            else:
                st = ColumnStim(
                    w=w,
                    x=x,
                    e_w=wt["e0"][s, c],
                    e_x=np.array([o["e_x0"] for o in sl]),
                    max_e_w=np.full(k, wt["maxw"][c]),
                    max_e_x=mxx,
                    clear=clr,
                    nan=np.array([o["nan"] for o in sl]) | wt["wnan"][s, c],
                )
                acc[a:b, c], nan[a:b, c] = column_reference(st, W, GUARD)
    return acc, nan


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer

    @cocotb.test()
    async def macro_mixed_matches(dut):
        rng = np.random.default_rng(int(os.environ.get("MM_SEED", "0")))
        mx_only = os.environ.get("MM_MX_ONLY") == "1"
        wt, ops, dots = program(rng, int(os.environ.get("MM_DOTS", "300")), mx_only)
        exp, exp_nan = expected(wt, ops, dots)
        is_mx = np.zeros(len(ops), bool)
        for a, b, nv in dots:
            is_mx[a:b] = not nv
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        for s in ("rst_n", "wr_en", "sc_wr_en", "maxw_wr_en", "op_valid"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        for st in range(M):
            for c in range(NCOL):
                for k in range(4):
                    word = sum(int(wt["codes"][st, c, 8 * k + j]) << (4 * j) for j in range(8))
                    await FallingEdge(dut.clk)
                    dut.wr_en.value, dut.wr_group.value = 1, c // 4
                    dut.wr_addr.value, dut.wr_data.value = st * 16 + (c % 4) * 4 + k, word
                await FallingEdge(dut.clk)
                dut.wr_en.value = 0
                dut.sc_wr_en.value, dut.sc_wr_set.value, dut.sc_wr_col.value = 1, st, c
                dut.sc_wr_e0.value, dut.sc_wr_e1.value = (
                    int(wt["e0"][st, c]) & 0xFF,
                    int(wt["e1"][st, c]) & 0xFF,
                )
                dut.sc_wr_u0.value, dut.sc_wr_u1.value = int(wt["u0"][st, c]), int(wt["u1"][st, c])
                dut.sc_wr_nan.value = int(wt["wnan"][st, c])
        for c in range(NCOL):
            await FallingEdge(dut.clk)
            dut.sc_wr_en.value = 0
            dut.maxw_wr_en.value, dut.maxw_wr_col.value, dut.maxw_wr_val.value = (
                1,
                c,
                int(wt["maxw"][c]) & 0xFF,
            )
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0
        got = [[] for _ in range(NCOL)]
        gnan = [[] for _ in range(NCOL)]
        i, drain, mask = 0, 0, (1 << W) - 1
        prev_fire_nv, toggles_b2b, fired_last = None, 0, False
        while i < len(ops) or drain < 16:
            await FallingEdge(dut.clk)
            av = int(dut.acc_valid.value)
            if av:
                a, nf = int(dut.acc.value), int(dut.nan_out.value)
                for c in range(NCOL):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
                        gnan[c].append(bool((nf >> c) & 1))
            if i < len(ops):
                o = ops[i]
                dut.op_valid.value, dut.op_nv.value = 1, int(o["nv"])
                dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(o["x"]))
                dut.op_e_x0.value, dut.op_e_x1.value = o["e_x0"] & 0xFF, o["e_x1"] & 0xFF
                dut.op_u_x0.value, dut.op_u_x1.value = o["u_x0"], o["u_x1"]
                dut.op_max_e_x.value, dut.op_x_nan.value = o["max_e_x"] & 0xFF, int(o["nan"])
                dut.op_set.value, dut.op_clear.value = o["set"], int(o["clear"])
                await Timer(1, unit="ns")
                if int(dut.op_ready.value):
                    if fired_last and prev_fire_nv is not None and prev_fire_nv != o["nv"]:
                        toggles_b2b += 1
                    prev_fire_nv, fired_last = o["nv"], True
                    i += 1
                else:
                    fired_last = False
            else:
                dut.op_valid.value = 0
                drain += 1
        g = np.array(got, np.int64).T
        gn = np.array(gnan, bool).T
        assert g.shape == exp.shape, (g.shape, exp.shape)
        mism = int((g != exp).sum())
        nan_mism = int((gn != exp_nan).sum())
        nv_nan = exp_nan[~is_mx]
        assert mism == 0, f"{mism} acc mismatches"
        assert nan_mism == 0, f"{nan_mism} NaN-flag mismatches (MX + NV)"
        # NV NaN stimulus is non-trivial: NaN and non-NaN columns, and stickiness across ops
        assert mx_only or (nv_nan.any() and not nv_nan.all()), "NV nan_out never / always set"
        nv_idx = [j for a, b, nv in dots if nv for j in range(a, b)]
        sticky = sum(
            1
            for a, b, nv in dots
            if nv
            for j in range(a + 1, b)
            if (
                exp_nan[j] & exp_nan[j - 1] & ~np.array([ops[j]["nan"]] * NCOL) & ~wt["wnan"][ops[j]["set"]]
            ).any()
        )
        assert mx_only or (sticky > 0 and len(nv_idx) > 0), "no sticky NV NaN carried across ops"
        assert int(dut.ovf.value) == 0
        n_tog = sum(1 for j in range(1, len(ops)) if ops[j]["nv"] != ops[j - 1]["nv"])
        assert mx_only or (n_tog > 0 and toggles_b2b > 0), (n_tog, toggles_b2b)
        assert exp_nan[is_mx, NAN_COL].any() and not exp_nan[is_mx][:, NAN_COL].all()
        dut._log.info(
            f"mixed MX/NV: {len(ops)} ops x 32 columns, {len(dots)} dot products, {n_tog} format toggles "
            f"({toggles_b2b} back-to-back), 0 acc / 0 NaN mismatches (NV nan_out checked, {sticky} sticky carries)"
        )

except ImportError:
    pass


def _run(top, tmp_path, n_dots=300, seed=0):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = (
        (
            [REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv"]
            + [REPO_ROOT / "rtl/src" / f for f in ("cim_icg.sv", "wset_sram.sv", "wset_latch.sv")]
            + [REPO_ROOT / "rtl/baselines/bitserial_int5" / f for f in ("bs_column.sv", "bs_macro_top.sv")]
            + [SRAM_V]
        )
        if top == "bs_macro_top"
        else (
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
            + [REPO_ROOT / "rtl/baselines/digital_mac" / f for f in ("generic_leaf.sv", "dpu_macro.sv")]
            + [SRAM_V]
        )
    )
    params = {"M": M, "GUARD": GUARD} | ({"STORE": 0} if top != "dpu_macro" else {})
    runner = get_runner("verilator")
    runner.build(
        sources=srcs,
        hdl_toplevel=top,
        build_dir=tmp_path / "b",
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters=params,
        build_args=["--timing", "-Wno-fatal"],
        timescale=("1ns", "1ps"),
    )
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_macro_mixed",
        test_dir=Path(__file__).parent,
        build_dir=tmp_path / "b",
        results_xml=str(tmp_path / "r.xml"),
        extra_env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
            "MM_DOTS": str(n_dots),
            "MM_SEED": str(seed),
            "MM_MX_ONLY": "1" if top == "bs_macro_top" else "0",  # baseline (a) is MXFP4 only
        },
    )
    assert get_results(xml) == (1, 0)


def test_writer_rejects_reserved_e4m3():
    """Writer stance (T1.5): reserved E4M3 scale codes 0x80..0xFE never reach the scale RF."""
    for c in (0x80, 0x81, 0xC0, 0xFE):
        with pytest.raises(ValueError):
            rf_record(0x38, c)
    assert rf_record(0x7F, 0x00)[4] and rf_record(0xFF, 0x38)[4] and not rf_record(0x00, 0x07)[4]
    assert rf_record(0x00, 0x07)[:4] == (-9, -9, 0, 7)  # zero and subnormal scales: k = -9


needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)


@needs
@pytest.mark.parametrize("top", ["macro_top", "dpu_macro", "bs_macro_top"])
def test_macro_mixed(top, tmp_path):
    _run(top, tmp_path, n_dots=int(os.environ.get("MM_DOTS", "300")))
