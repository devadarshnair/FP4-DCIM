"""T5.2: INT4-SM bit-serial column verification (rtl/baselines/bitserial_int4sm/bs4_column.sv).

Chunked, env-parameterised cocotb test (driver: experiments/t52_bs4_verify/run.py).  Per op it checks
  (1) the RTL block sum (internal `blk`, sampled when `b_valid`) against an independent integer reference
      sum_r (-1)^(sw^sx) |w||x|  AND against the golden L2 model BitSerialInt4Column.block_sums on the same ops,
  (2) the window accumulator `acc`, `ovf` and `nan_out` against the integer reference of the contract
      (alignment by d = max_e_w - e_w + max_e_x - e_x, W-bit window, clamp at W-1, ovf/NaN sticky),
  (3) the issue gap (3 cycles back-to-back; >= 3 with idle gaps).
Stimulus: sign-magnitude (sign, mag) pairs, so -0 (sign=1, mag=0) is driven explicitly on weights and activations.
Env: B4V_MODE=directed|random, B4V_SEED, B4V_NOPS, B4V_GAP (idle-gap probability), B4V_OUT (json path).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

W, GUARD, SB, ROWS = 40, 7, 14, 32
BASE, DMAX = W - SB - 1 - GUARD, W - 1
MASK = (1 << W) - 1


def _wrap(s):
    s &= MASK
    return s - (1 << W) if s >> (W - 1) else s


def ref_step(acc, ovf, nanf, blk, d_raw, clear, nan):
    """Integer reference of stage C (mirrors the contract, not the RTL structure)."""
    d = min(d_raw, DMAX) if d_raw >= 0 else d_raw & 0x3F  # negative d: low 6 bits + ovf flagged
    term = (blk << (BASE - d)) if d <= BASE else (blk >> (d - BASE))
    s = (0 if clear else acc) + term
    over = not (-(1 << (W - 1)) <= s < (1 << (W - 1)))
    return (_wrap(s), (0 if clear else ovf) | int(over) | int(d_raw < 0), (0 if clear else nanf) | nan)


def pack_w(sg, mg):
    out = 0
    for r in range(ROWS):
        out |= ((sg[r] << 3) | mg[r]) << (4 * r)
    return out


def _exps_for_d(rng, d):
    a = rng.randint(max(-255, d - 255), min(255, d + 255))
    b = d - a

    def pair(delta):
        ew = rng.randint(max(-128, -128 - delta), min(127, 127 - delta))
        return ew, ew + delta

    ew, mw = pair(a)
    ex, mx = pair(b)
    return ew, ex, mw, mx


EDGE_D = [0, 1, 2, BASE - 1, BASE, BASE + 1, DMAX - 1, DMAX, DMAX + 1, DMAX + 2, 63, 64, 100, 300, 510,
          -1, -2, -22, -63, -64, -65, -300, -510]


def _lane_profile(rng, prof):
    """one lane -> (sign, mag); the sign is random even for mag 0, so -0 occurs naturally."""
    if prof == "uniform":
        return rng.getrandbits(1), rng.randint(0, 7)
    if prof == "sparse":
        return rng.getrandbits(1), (0 if rng.random() < 0.7 else rng.randint(1, 7))
    if prof == "extreme":
        return rng.getrandbits(1), (7 if rng.random() < 0.8 else rng.randint(0, 7))
    if prof == "negzero":  # half the lanes are -0, rest uniform
        return (1, 0) if rng.random() < 0.5 else (rng.getrandbits(1), rng.randint(1, 7))
    if prof == "all7":
        return rng.getrandbits(1), 7
    return rng.getrandbits(1), rng.choice((0, 7))  # "zero7"


def make_random_ops(rng, n):
    profs = ("uniform", "sparse", "extreme", "negzero", "zero7", "all7")
    ops = []
    for k in range(n):
        pw, px = rng.choice(profs), rng.choice(profs)
        w = [_lane_profile(rng, pw) for _ in range(ROWS)]
        x = [_lane_profile(rng, px) for _ in range(ROWS)]
        u = rng.random()
        if u < 0.55:
            ew, ex = rng.randint(-20, 20), rng.randint(-20, 20)
            mw, mx = ew + rng.randint(0, 6), ex + rng.randint(0, 6)
        elif u < 0.9:
            ew, ex, mw, mx = _exps_for_d(rng, rng.choice(EDGE_D))
        else:
            ew, ex, mw, mx = (rng.randint(-128, 127) for _ in range(4))
        ops.append(dict(ws=[a for a, _ in w], wm=[b for _, b in w], xs=[a for a, _ in x], xm=[b for _, b in x],
                        ew=ew, ex=ex, mw=mw, mx=mx, clr=int(rng.random() < 0.25 or k == 0), nan=int(rng.random() < 0.02)))
    return ops


def make_directed_ops():
    rng = random.Random(5200)
    ops = []

    def add(ws, wm, xs, xm, d0=True, clr=1, nan=0, exps=None):
        e = exps or (0, 0, BASE // 2, BASE - BASE // 2) if d0 else exps  # d = BASE: window sum is the block sum
        ops.append(dict(ws=ws, wm=wm, xs=xs, xm=xm, ew=e[0], ex=e[1], mw=e[2], mx=e[3], clr=clr, nan=nan))

    rv = lambda: [rng.randint(1, 7) for _ in range(ROWS)]  # noqa: E731
    rs = lambda: [rng.getrandbits(1) for _ in range(ROWS)]  # noqa: E731
    z, o = [0] * ROWS, [1] * ROWS
    alt = [r & 1 for r in range(ROWS)]
    # --- explicit -0 (sign=1, mag=0) ---
    for _ in range(8):
        add(o, z, rs(), rv())  # all weights -0
        add(rs(), rv(), o, z)  # all activations -0
        add(o, z, o, z)  # both -0
        add(alt, z, rs(), rv())  # mixed -0/+0 weights
        add(rs(), rv(), alt, z)
        add(o, z, rs(), [7] * ROWS)  # -0 weights vs +-7
        add(rs(), [7] * ROWS, o, z)
    for r in range(ROWS):  # single -0 lane inside an otherwise +-7 block, weight and activation side
        wm, xm = [7] * ROWS, [7] * ROWS
        wm[r] = 0
        add([1 if i == r else 0 for i in range(ROWS)], wm, z, xm)
        add(z, [7] * ROWS, [1 if i == r else 0 for i in range(ROWS)], xm[:r] + [0] + xm[r + 1:])
    add(z, z, z, z)  # all +0
    add(o, z, z, z)
    # --- extremes +-7 ---
    for sw in (0, 1):
        for sx in (0, 1):
            add([sw] * ROWS, [7] * ROWS, [sx] * ROWS, [7] * ROWS)  # |block sum| = 1568
    add(alt, [7] * ROWS, z, [7] * ROWS)  # 16 +, 16 - lanes: sum 0
    add([0] * 16 + [1] * 16, [7] * ROWS, z, [7] * ROWS)
    add(rs(), [7] * ROWS, rs(), [7] * ROWS)
    for m in range(8):
        add(z, [m] * ROWS, z, [7] * ROWS)
        add(o, [7] * ROWS, z, [m] * ROWS)
    for r in (0, 1, 15, 16, 30, 31):  # single active lane at the extremes
        for s in (0, 1):
            add([s if i == r else 0 for i in range(ROWS)], [7 if i == r else 0 for i in range(ROWS)], z, [7 if i == r else 0 for i in range(ROWS)])
    # --- window edges / saturation: every edge d with extreme / small blocks, clear then accumulate ---
    for d in EDGE_D:
        for (ws, xs, mag) in ((0, 0, 7), (1, 0, 7), (0, 0, 1), (1, 0, 1), (0, 0, 0)):
            ex_ = _exps_for_d(rng, d)
            add([ws] * ROWS, [mag] * ROWS, [xs] * ROWS, [mag] * ROWS, exps=ex_, clr=1)
            add([ws] * ROWS, [mag] * ROWS, [xs] * ROWS, [mag] * ROWS, exps=_exps_for_d(rng, d), clr=0)  # accumulate on top
    # --- accumulator overflow (d=0, +1568<<18 per op): positive then negative, sticky ovf, then clear ---
    for sgn in (0, 1):
        add([sgn] * ROWS, [7] * ROWS, z, [7] * ROWS, exps=(0, 0, 0, 0), clr=1)
        for _ in range(1500):
            add([sgn] * ROWS, [7] * ROWS, z, [7] * ROWS, exps=(0, 0, 0, 0), clr=0)
        add(z, z, z, z, exps=(0, 0, 0, 0), clr=0)  # ovf must stay sticky
        add(z, z, z, z, exps=(0, 0, 0, 0), clr=1)  # ... and clear
    # --- NaN sticky, cleared by op_clear ---
    for k in range(6):
        add(rs(), rv(), rs(), rv(), nan=int(k == 1), clr=int(k == 0 or k == 4))
    return ops


try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

    def _sgn(sig):
        v = sig.value
        return int(v.to_signed()) if hasattr(v, "to_signed") else int(v.signed_integer)

    @cocotb.test()
    async def bs4_verify(dut):
        mode = os.environ.get("B4V_MODE", "random")
        seed = int(os.environ.get("B4V_SEED", "1"))
        gap_p = float(os.environ.get("B4V_GAP", "0"))
        rng = random.Random(seed)
        ops = make_directed_ops() if mode == "directed" else make_random_ops(rng, int(os.environ.get("B4V_NOPS", "1000")))
        n_ops = len(ops)
        grng = random.Random(seed + 1)
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        dut.rst_n.value = 0
        dut.op_valid.value = 0
        for sig in ("op_clear", "xbit_bus", "xsgn_bus", "w_bus", "e_w", "e_x", "max_e_w", "max_e_x", "nan_in"):
            getattr(dut, sig).value = 0
        for _ in range(3):
            await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)

        idx, cyc, accepts = 0, 0, []
        mags, shift_n = [0] * ROWS, 9
        blk_rtl, acc_rtl, ovf_rtl, nan_rtl = [], [], [], []
        gap_now = False
        while len(acc_rtl) < n_ops and cyc < 12 * n_ops + 100:
            await FallingEdge(dut.clk)
            gap_now = gap_p > 0 and grng.random() < gap_p
            will_accept = idx < n_ops and int(dut.op_ready.value) == 1 and not gap_now
            if will_accept:
                o = ops[idx]
                dut.op_valid.value = 1
                dut.e_w.value, dut.e_x.value, dut.max_e_w.value, dut.max_e_x.value = o["ew"], o["ex"], o["mw"], o["mx"]
                dut.op_clear.value, dut.nan_in.value = o["clr"], o["nan"]
            else:
                dut.op_valid.value = 0
            await RisingEdge(dut.clk)
            cyc += 1
            if will_accept:
                o = ops[idx]
                mags, shift_n = o["xm"], 0
                dut.w_bus.value = pack_w(o["ws"], o["wm"])
                dut.xsgn_bus.value = sum(s << r for r, s in enumerate(o["xs"]))
                accepts.append(cyc)
                idx += 1
            else:
                shift_n += 1
            dut.xbit_bus.value = sum(((mags[r] >> shift_n) & 1) << r for r in range(ROWS)) if shift_n < 3 else 0
            await ReadOnly()
            if int(dut.b_valid.value) == 1:
                blk_rtl.append(_sgn(dut.blk))
            if int(dut.acc_valid.value) == 1:
                acc_rtl.append(_sgn(dut.acc))
                ovf_rtl.append(int(dut.ovf.value))
                nan_rtl.append(int(dut.nan_out.value))
        assert len(acc_rtl) == n_ops and len(blk_rtl) == n_ops, f"results {len(acc_rtl)}/{len(blk_rtl)} of {n_ops}"

        import numpy as np

        from cimfp4.golden.l2_bitserial_int4 import BitSerialInt4Column

        sw = np.array([[(-1 if s else 1) * m for s, m in zip(o["ws"], o["wm"])] for o in ops], np.int64)
        sx = np.array([[(-1 if s else 1) * m for s, m in zip(o["xs"], o["xm"])] for o in ops], np.int64)
        gold = np.concatenate([BitSerialInt4Column().block_sums(sw[i:i + 4000], sx[i:i + 4000])[0] for i in range(0, n_ops, 4000)])
        ref_blk = (sw * sx).sum(axis=1)
        acc, ovf, nanf = 0, 0, 0
        mm = dict(blk_vs_int_ref=0, blk_vs_golden=0, golden_vs_int_ref=0, acc=0, ovf=0, nan=0)
        ex = []
        cov = dict(w_negzero_lanes=0, x_negzero_lanes=0, ops_with_w_negzero=0, ops_with_x_negzero=0,
                   ops_all_lanes_pm7_both=0, ops_abs_blk_1568=0, d_hist={"neg": 0, "0..BASE-1": 0, "BASE": 0, "BASE+1..DMAX": 0,
                                                                         ">DMAX(sat)": 0}, ops_ovf_set=0, ops_nan_set=0, clear_ops=0)
        for i, o in enumerate(ops):
            d_raw = o["mw"] - o["ew"] + o["mx"] - o["ex"]
            acc, ovf, nanf = ref_step(acc, ovf, nanf, int(ref_blk[i]), d_raw, o["clr"], o["nan"])
            bad = []
            if blk_rtl[i] != ref_blk[i]:
                mm["blk_vs_int_ref"] += 1; bad.append("blk_int")
            if blk_rtl[i] != gold[i]:
                mm["blk_vs_golden"] += 1; bad.append("blk_golden")
            if gold[i] != ref_blk[i]:
                mm["golden_vs_int_ref"] += 1; bad.append("golden_int")
            if acc_rtl[i] != acc:
                mm["acc"] += 1; bad.append("acc")
            if ovf_rtl[i] != ovf:
                mm["ovf"] += 1; bad.append("ovf")
            if nan_rtl[i] != nanf:
                mm["nan"] += 1; bad.append("nan")
            if bad and len(ex) < 20:
                ex.append(dict(op=i, kinds=bad, blk_rtl=blk_rtl[i], blk_ref=int(ref_blk[i]), blk_golden=int(gold[i]),
                               acc_rtl=acc_rtl[i], acc_ref=acc, ovf=(ovf_rtl[i], ovf), nan=(nan_rtl[i], nanf), d_raw=d_raw))
            nzw = sum(1 for s, m in zip(o["ws"], o["wm"]) if s and m == 0)
            nzx = sum(1 for s, m in zip(o["xs"], o["xm"]) if s and m == 0)
            cov["w_negzero_lanes"] += nzw; cov["x_negzero_lanes"] += nzx
            cov["ops_with_w_negzero"] += nzw > 0; cov["ops_with_x_negzero"] += nzx > 0
            cov["ops_all_lanes_pm7_both"] += all(m == 7 for m in o["wm"]) and all(m == 7 for m in o["xm"])
            cov["ops_abs_blk_1568"] += abs(int(ref_blk[i])) == 1568
            k = "neg" if d_raw < 0 else "0..BASE-1" if d_raw < BASE else "BASE" if d_raw == BASE else "BASE+1..DMAX" if d_raw <= DMAX else ">DMAX(sat)"
            cov["d_hist"][k] += 1
            cov["ops_ovf_set"] += ovf; cov["ops_nan_set"] += nanf; cov["clear_ops"] += o["clr"]
        gaps = sorted({b - a for a, b in zip(accepts, accepts[1:])})
        h = hashlib.sha256()
        h.update(np.asarray(blk_rtl, np.int64).tobytes()); h.update(np.asarray(acc_rtl, np.int64).tobytes())
        res = dict(mode=mode, seed=seed, gap_prob=gap_p, n_ops=n_ops, mismatches=mm, mismatch_total=sum(mm.values()),
                   examples=ex, issue_gaps_seen=gaps, coverage=cov, rtl_results_sha256=h.hexdigest(),
                   stimulus_sha256=hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest(),
                   blk_min=int(min(blk_rtl)), blk_max=int(max(blk_rtl)))
        Path(os.environ["B4V_OUT"]).write_text(json.dumps(res, indent=1))
        assert sum(mm.values()) == 0, f"mismatches {mm}; first {ex[:2]}"
        assert (gaps == [3]) if gap_p == 0 else (min(gaps) >= 3), f"issue gaps {gaps}"
except ImportError:
    pass


needs = pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator missing")


@needs
def test_bs4_verify(tmp_path):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    srcs = [REPO_ROOT / "rtl/include/cim_pkg.sv", REPO_ROOT / "rtl/src/cim_icg.sv",
            REPO_ROOT / "rtl/gen/col_tree_unsigned_w1_r32.sv", Path(os.environ.get("B4V_RTL", REPO_ROOT / "rtl/baselines/bitserial_int4sm/bs4_column.sv"))]
    runner = get_runner("verilator")
    runner.build(includes=[REPO_ROOT / "rtl" / "include"], sources=srcs, hdl_toplevel="bs4_column",
                 build_dir=tmp_path / "b", always=True, parameters={"GUARD": GUARD},
                 build_args=["--timing", "-Wno-fatal"], timescale=("1ns", "1ps"))
    xml = runner.test(hdl_toplevel="bs4_column", test_module="test_bs4_verify", test_dir=Path(__file__).parent,
                      build_dir=tmp_path / "b", results_xml=str(tmp_path / "r.xml"),
                      extra_env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}"})
    assert get_results(xml) == (1, 0)
