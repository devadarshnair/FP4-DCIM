"""T4.2 verification of NV_GATE (docs/spec/nv_gate.md, "T4.2 directed tests"): NV_GATE=1 vs NV_GATE=0 vs golden.

Directed scenarios on macro_top (v2, fp4 leaf) and dpu_macro (generic leaf), each checked bit-exact (acc and nan_out
of all 32 columns, per accepted op) against the golden model, and the full per-cycle output traces
(acc, acc_valid, nan_out, ovf, op_ready) of NV_GATE=1 compared bit-identical to NV_GATE=0:

  t1 toggle        MX/NV/MX/NV every op (one-op dot products, sets change too)
  t2 long_mx       a long MX run (same set), then NV (the first NV cycle loads the gated flops), then MX
  t3 idle_toggle   op_nv (and op_clear, junk operands) toggled with op_valid=0 between valid ops
  t4 phase_fall / phase_rise   op_nv launched on the falling edge vs 1 ns after the rising edge (ICG latch phase)
  t5 set_switch    set changes on (almost) every op with toggling format; op_nv wiggles while op_ready=0
                   (load_pending / busy) and takes its true value only in the accepting cycle
  t6 reset_mid     reset with op_nv=1 and a junk op presented, then MX, then NV
  t7 clear_trans   acc_clear coincident with MX<->NV transitions and with same-format ops
  t8 mixed_random  random per-op format
Test 11 (zero r_h* toggles in MX ops, VCD) is `test_nv_gate_vcd` (separate Python-side check of a Verilator VCD).

Env: NG_TRACE (json output path, written by the cocotb side), NG_SEED, NG_SCALE (ops multiplier).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_macro_mixed as mm  # noqa: E402

from cimfp4.formats.mx_quant import quantize_mxfp4  # noqa: E402
from cimfp4.utils.provenance import REPO_ROOT  # noqa: E402

NCOL, W, M = mm.NCOL, mm.W, mm.M
SEED = int(os.environ.get("NG_SEED", "7"))
SCALE = int(os.environ.get("NG_SCALE", "1"))


def mkop(rng, nv, set_, clear, nan=False):
    q = quantize_mxfp4(rng.standard_t(3, (1, 32)) * np.exp(rng.normal(0, 1)))
    lo, hi = (-9, 6) if nv else (-8, 9)
    return dict(
        x=q.codes[0],
        nv=nv,
        set=set_,
        clear=clear,
        e_x0=int(rng.integers(lo, hi)),
        e_x1=int(rng.integers(lo, hi)) if nv else 0,
        u_x0=int(rng.integers(0, 16)) if nv else 1,
        u_x1=int(rng.integers(0, 16)) if nv else 1,
        max_e_x=int(hi),
        nan=nan,
    )


def finish(ops, dots_fmt):
    """dots_fmt: list of (start, stop, nv); max_e_x of an op = max over its dot (writer anchor rule)."""
    for a, b, nv in dots_fmt:
        mx = max(max(o["e_x0"], o["e_x1"] if nv else -128) for o in ops[a:b])
        for o in ops[a:b]:
            o["max_e_x"] = mx
    return ops, dots_fmt


def build(rng, kind):
    """Return dict(ops, dots, opt) for scenario `kind`; opt are driver options."""
    n = 24 * SCALE
    ops, dots, opt = [], [], {}

    def dot(nv, length, set_, nan_p=0.05):
        a = len(ops)
        for j in range(length):
            ops.append(mkop(rng, nv, set_, j == 0, bool(rng.random() < nan_p)))
        dots.append((a, len(ops), nv))

    if kind == "toggle":
        for i in range(2 * n):
            dot(bool(i % 2), 1, int(rng.integers(0, M - 1)))
    elif kind == "long_mx":
        for _ in range(max(1, SCALE)):
            dot(False, 40, 1)
            dot(True, 3, 1)
            dot(False, 20, 1)
            dot(True, 1, 1)
    elif kind == "idle_toggle":
        opt["idle"] = True
        for i in range(n):
            dot(bool(rng.random() < 0.5), int(rng.integers(1, 3)), int(rng.integers(0, M - 1)))
    elif kind in ("phase_fall", "phase_rise"):
        opt["phase"] = "rise" if kind == "phase_rise" else "fall"
        for i in range(n):
            dot(bool(i % 2) if i < n // 2 else bool(rng.random() < 0.5), int(rng.integers(1, 3)), 0)
    elif kind == "set_switch":
        opt["wiggle"] = True
        for i in range(2 * n):
            dot(bool(i % 2), 1, i % (M - 1))
    elif kind == "reset_mid":
        opt["reset_at"] = 3  # after dot 3 the stream is drained and reset asserted with op_nv=1
        for i in range(3):
            dot(i % 2 == 1, 2, 0)
        for nv in (False, False, True, False, True):  # post-reset: MX, then NV
            dot(nv, 2, 0)
        opt["reset_dots"] = 3
    elif kind == "clear_trans":
        # clear lands on the first op of every dot; dots 1..3 ops long, formats change on ~60 % of boundaries
        nv = False
        for i in range(n):
            if rng.random() < 0.6:
                nv = not nv
            dot(nv, int(rng.integers(1, 4)), int(rng.integers(0, 2)), nan_p=0.0)
        opt["idle_clear"] = True
    elif kind == "mixed_random":
        for i in range(4 * n):
            dot(bool(rng.random() < 0.5), int(rng.integers(1, 4)), int(rng.integers(0, M - 1)))
    else:
        raise ValueError(kind)
    return dict(ops=ops, dots=dots, opt=opt)


KINDS = [
    "toggle",
    "long_mx",
    "idle_toggle",
    "phase_fall",
    "phase_rise",
    "set_switch",
    "reset_mid",
    "clear_trans",
    "mixed_random",
]

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

    TRACE: dict = {}

    def _dump():
        path = os.environ.get("NG_TRACE")
        if path:
            Path(path).write_text(json.dumps(TRACE))

    async def load(dut, wt):
        for s in ("wr_en", "sc_wr_en", "maxw_wr_en", "op_valid", "op_nv", "op_clear", "op_x_nan"):
            getattr(dut, s).value = 0
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
            dut.maxw_wr_en.value, dut.maxw_wr_col.value = 1, c
            dut.maxw_wr_val.value = int(wt["maxw"][c]) & 0xFF
        await FallingEdge(dut.clk)
        dut.maxw_wr_en.value = 0

    def drive(dut, o, nv=None):
        dut.op_valid.value, dut.op_nv.value = 1, int(o["nv"] if nv is None else nv)
        dut.op_x.value = sum(int(v) << (4 * r) for r, v in enumerate(o["x"]))
        dut.op_e_x0.value, dut.op_e_x1.value = o["e_x0"] & 0xFF, o["e_x1"] & 0xFF
        dut.op_u_x0.value, dut.op_u_x1.value = o["u_x0"], o["u_x1"]
        dut.op_max_e_x.value, dut.op_x_nan.value = o["max_e_x"] & 0xFF, int(o["nan"])
        dut.op_set.value, dut.op_clear.value = o["set"], int(o["clear"])

    def junk(dut, rng):
        dut.op_valid.value = 0
        dut.op_nv.value = int(rng.integers(0, 2))
        dut.op_x.value = (
            int(rng.integers(0, 1 << 62))
            | (int(rng.integers(0, 1 << 62)) << 62)
            | (int(rng.integers(0, 1 << 4)) << 124)
        )
        dut.op_e_x0.value, dut.op_e_x1.value = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        dut.op_u_x0.value, dut.op_u_x1.value = int(rng.integers(0, 16)), int(rng.integers(0, 16))
        dut.op_max_e_x.value = int(rng.integers(0, 256))
        dut.op_x_nan.value, dut.op_set.value = int(rng.integers(0, 2)), int(rng.integers(0, M))
        dut.op_clear.value = int(rng.integers(0, 2))

    async def run_ops(dut, ops, opt, rng, rec):
        """Drive `ops` with the options of the scenario; returns per-column per-accepted-op got lists."""
        got, gnan = [[] for _ in range(NCOL)], [[] for _ in range(NCOL)]
        i, drain, mask = 0, 0, (1 << W) - 1
        rise = opt.get("phase") == "rise"
        idle_left = 0
        while i < len(ops) or drain < 16:
            await FallingEdge(dut.clk)
            av = int(dut.acc_valid.value)
            a, nf = int(dut.acc.value), int(dut.nan_out.value)
            rec.append((a, av, nf, int(dut.ovf.value), int(dut.op_ready.value)))
            if av:
                for c in range(NCOL):
                    if (av >> c) & 1:
                        v = (a >> (c * W)) & mask
                        got[c].append(v - (1 << W) if v >> (W - 1) else v)
                        gnan[c].append(bool((nf >> c) & 1))
            if rise:
                await RisingEdge(dut.clk)
                await Timer(1, unit="ns")
            if i < len(ops):
                if (opt.get("idle") or opt.get("idle_clear")) and idle_left == 0 and rng.random() < 0.4:
                    idle_left = int(rng.integers(1, 4))
                if idle_left:
                    idle_left -= 1
                    junk(dut, rng)
                    if not opt.get("idle_clear"):
                        dut.op_clear.value = 0
                    continue
                o = ops[i]
                if opt.get("wiggle"):
                    drive(
                        dut, o, nv=int(rng.integers(0, 2))
                    )  # random format while the op is still not accepted
                    await Timer(1, unit="ns")
                    if int(dut.op_ready.value):
                        drive(dut, o)  # accepting cycle: true format
                        await Timer(1, unit="ns")
                        i += 1
                    continue
                drive(dut, o)
                await Timer(1, unit="ns")
                if int(dut.op_ready.value):
                    i += 1
            else:
                dut.op_valid.value = 0
                drain += 1
        return np.array(got, np.int64).T, np.array(gnan, bool).T

    def _make(kind):
        @cocotb.test(name=f"t_{kind}")
        async def _t(dut):
            rng = np.random.default_rng(SEED + KINDS.index(kind))
            wt, _, _ = mm.program(np.random.default_rng(SEED), 0)
            sc = build(rng, kind)
            ops, dots, opt = sc["ops"], sc["dots"], sc["opt"]
            finish(ops, dots)
            cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
            for s in ("rst_n", "wr_en", "sc_wr_en", "maxw_wr_en", "op_valid", "op_nv", "op_clear"):
                getattr(dut, s).value = 0
            await ClockCycles(dut.clk, 2)
            dut.rst_n.value = 1
            await load(dut, wt)
            rec: list = []
            split = opt.get("reset_dots")
            if split is None:
                parts = [(ops, dots)]
            else:
                cut = dots[split][0]
                parts = [
                    (ops[:cut], dots[:split]),
                    (ops[cut:], [(a - cut, b - cut, nv) for a, b, nv in dots[split:]]),
                ]
            exp_all, nexp_all, got_all, ngot_all = [], [], [], []
            for pi, (pops, pdots) in enumerate(parts):
                if pi == 1:  # reset mid-stream with op_nv = 1 and a junk op on the inputs
                    await FallingEdge(dut.clk)
                    dut.rst_n.value = 0
                    junk(dut, rng)
                    dut.op_nv.value, dut.op_valid.value = 1, 1
                    await ClockCycles(dut.clk, 3)
                    await FallingEdge(dut.clk)
                    dut.rst_n.value = 1
                    dut.op_valid.value = 0
                    await ClockCycles(dut.clk, 2)
                    await load(dut, wt)  # weight stores survive reset or not: rewrite either way
                g, gn = await run_ops(dut, pops, opt if pi == 0 or "reset_at" not in opt else {}, rng, rec)
                e, en = mm.expected(wt, pops, pdots)
                got_all.append(g), ngot_all.append(gn), exp_all.append(e), nexp_all.append(en)
            g, gn, e, en = np.vstack(got_all), np.vstack(ngot_all), np.vstack(exp_all), np.vstack(nexp_all)
            assert g.shape == e.shape, (g.shape, e.shape)
            assert int((g != e).sum()) == 0, f"{int((g != e).sum())} acc mismatches vs golden"
            assert int((gn != en).sum()) == 0, f"{int((gn != en).sum())} nan_out mismatches vs golden"
            n_tog = sum(1 for j in range(1, len(ops)) if ops[j]["nv"] != ops[j - 1]["nv"])
            digest = hashlib.sha256(json.dumps(rec).encode()).hexdigest()
            TRACE[kind] = dict(ops=len(ops), toggles=n_tog, cycles=len(rec), sha256=digest, trace=rec)
            _dump()
            dut._log.info(
                f"{kind}: {len(ops)} ops, {n_tog} format toggles, {len(rec)} cycles, 0 mismatches vs golden"
            )

        return _t

    for _k in KINDS:
        globals()[f"t_{_k}"] = _make(_k)

    @cocotb.test()
    async def t_vcd_mx_after_nv(dut):
        """Test 11 stimulus: NV ops, then a long MX run (NV flops must be quiet), then NV again; markers in a json."""
        rng = np.random.default_rng(SEED + 99)
        wt, _, _ = mm.program(np.random.default_rng(SEED), 0)
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        for s in ("rst_n", "wr_en", "sc_wr_en", "maxw_wr_en", "op_valid", "op_nv", "op_clear"):
            getattr(dut, s).value = 0
        await ClockCycles(dut.clk, 2)
        dut.rst_n.value = 1
        await load(dut, wt)
        ops, dots = [], []
        for nv, ln in ((True, 6), (False, 40), (True, 6)):
            a = len(ops)
            ops += [mkop(rng, nv, 1, j == 0) for j in range(ln)]
            dots.append((a, len(ops), nv))
        finish(ops, dots)
        marks = {}
        for si, (a, b, nv) in enumerate(dots):
            await FallingEdge(dut.clk)
            marks[f"seg{si}_start_ns"] = int(cocotb.utils.get_sim_time("ns"))
            g, gn = await run_ops(dut, ops[a:b], {}, rng, [])
            marks[f"seg{si}_end_ns"] = int(cocotb.utils.get_sim_time("ns"))
            e, en = mm.expected(wt, ops[a:b], [(0, b - a, nv)])
            assert (g == e).all() and (gn == en).all()
        path = os.environ.get("NG_MARKS")
        if path:
            Path(path).write_text(json.dumps(marks))

except ImportError:
    pass


# ---------------------------------------------------------------- pytest drivers
SRAM_V = mm.SRAM_V
needs = pytest.mark.skipif(
    shutil.which("verilator") is None or not SRAM_V.exists(), reason="verilator/OpenRAM missing"
)


def _sources():
    return (
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


def _run(top, gate, tmp_path, trace_name, waves=False, extra_env=None, tests=None):
    pytest.importorskip("cocotb")
    from cocotb_tools.runner import get_results, get_runner

    params = {"M": M, "GUARD": mm.GUARD, "NV_GATE": gate} | ({"STORE": 0} if top != "dpu_macro" else {})
    runner = get_runner("verilator")
    bd = tmp_path / f"b_{top}_{gate}"
    runner.build(
        sources=_sources(),
        hdl_toplevel=top,
        build_dir=bd,
        always=True,
        includes=[REPO_ROOT / "rtl/include"],
        parameters=params,
        build_args=["--timing", "-Wno-fatal"] + (["--trace"] if waves else []),
        timescale=("1ns", "1ps"),
        waves=waves,
    )
    env = {
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{Path(__file__).parent}",
        "NG_TRACE": str(tmp_path / trace_name),
    } | (extra_env or {})
    kw = {"testcase": tests} if tests else {}
    xml = runner.test(
        hdl_toplevel=top,
        test_module="test_nv_gate",
        test_dir=Path(__file__).parent,
        build_dir=bd,
        results_xml=str(tmp_path / f"r_{top}_{gate}.xml"),
        extra_env=env,
        waves=waves,
        **kw,
    )
    return get_results(xml), bd


@needs
@pytest.mark.parametrize("top", ["macro_top", "dpu_macro"])
def test_nv_gate_directed(top, tmp_path):
    """Tests 1-8: NV_GATE=0 and 1 both bit-exact vs golden; per-cycle traces of NV_GATE=1 identical to NV_GATE=0."""
    names = [f"t_{k}" for k in KINDS]
    res = {}
    for gate in (0, 1):
        (tot, failed), _ = _run(top, gate, tmp_path, f"trace_{gate}.json", tests=names)
        assert (tot, failed) == (len(KINDS), 0), (gate, tot, failed)
        res[gate] = json.loads((tmp_path / f"trace_{gate}.json").read_text())
    summary = {}
    for k in KINDS:
        a, b = res[0][k], res[1][k]
        assert a["trace"] == b["trace"], f"{top}/{k}: NV_GATE=1 trace differs from NV_GATE=0"
        assert a["toggles"] > 0 or k == "long_mx" or k.startswith("phase")
        summary[k] = {x: a[x] for x in ("ops", "toggles", "cycles", "sha256")}
    out = os.environ.get("NG_SUMMARY")
    if out:
        p = Path(out)
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur[top] = summary
        p.write_text(json.dumps(cur, indent=1))


def parse_vcd_toggles(vcd: Path, name_re: str, windows: dict[str, tuple[int, int]]):
    """Count value changes (after the initial dump) of signals whose hierarchical name matches `name_re`, per window
    (time in ns). Returns ({window: n_changes}, number_of_signals)."""
    scale, ids, scope, t = 1.0, {}, [], 0
    counts = {w: 0 for w in windows}
    started = False
    ts = None
    with open(vcd) as f:
        for line in f:
            line = line.strip()
            if line.startswith("$timescale"):
                m = re.search(r"(\d+)\s*(ps|ns|us|fs)", line) or re.search(
                    r"(\d+)(ps|ns|us|fs)", f.readline()
                )
                ts = {"fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3}[m.group(2)] * int(m.group(1))
            elif line.startswith("$scope"):
                scope.append(line.split()[2])
            elif line.startswith("$upscope"):
                scope.pop()
            elif line.startswith("$var"):
                p = line.split()
                full = ".".join(scope + [p[4]])
                if re.search(name_re, full):
                    ids.setdefault(p[3], []).append(full)
            elif line.startswith("$enddefinitions"):
                started = True
            elif started and line.startswith("#"):
                t = int(line[1:]) * (ts or 1.0)
            elif started and line and line[0] in "01xzbB":
                ident = line.split()[-1] if line[0] in "bB" else line[1:]
                if ident in ids:
                    for w, (a, b) in windows.items():
                        if a <= t < b:
                            counts[w] += 1
    return counts, sum(len(v) for v in ids.values())


@needs
def test_nv_gate_vcd(tmp_path):
    """Test 11: zero toggles of r_h* (and of the NV clock gate) during MX ops, NV_GATE=1 (and =0), from a VCD."""
    summary = {}
    (Path(__file__).parent / "dump.vcd").unlink(missing_ok=True)
    for gate in (1, 0):
        marks = tmp_path / f"marks_{gate}.json"
        (tot, failed), bd = _run(
            "macro_top",
            gate,
            tmp_path,
            f"vtrace_{gate}.json",
            waves=True,
            extra_env={"NG_MARKS": str(marks)},
            tests=["t_vcd_mx_after_nv"],
        )
        assert (tot, failed) == (1, 0)
        mk = json.loads(marks.read_text())
        dump = Path(__file__).parent / "dump.vcd"  # Verilator writes it into the cocotb test dir
        assert dump.exists(), "no VCD produced"
        vcd = tmp_path / f"nv_gate_{gate}.vcd"
        shutil.move(str(dump), vcd)
        win = {
            "nv_seg0": (mk["seg0_start_ns"], mk["seg0_end_ns"]),
            # MX segment, excluding the first 3 ns for the last NV result drain skew (pipeline is 2 stages deep)
            "mx_seg1": (mk["seg1_start_ns"] + 30, mk["seg1_end_ns"] - 30),
            "nv_seg2": (mk["seg2_start_ns"], mk["seg2_end_ns"]),
        }
        rh, nsig = parse_vcd_toggles(vcd, r"g_col.*u_col\.r_h[01][sc]$", win)
        rd, nsig_d = parse_vcd_toggles(vcd, r"g_col.*u_col\.(r_d1|r_u0|r_u1)$", win)
        gc = (
            parse_vcd_toggles(vcd, r"g_col.*u_col\.g_nvff\.clk_nv$", win)
            if gate
            else ({k: 0 for k in win}, 0)
        )
        summary[f"NV_GATE={gate}"] = dict(
            r_h=rh,
            r_h_signals=nsig,
            r_d1_u=rd,
            r_d1_u_signals=nsig_d,
            clk_nv=gc[0],
            clk_nv_signals=gc[1],
            windows_ns=win,
            vcd=str(vcd),
        )
        assert nsig == 32 * 4, nsig
        assert rh["mx_seg1"] == 0 and rd["mx_seg1"] == 0, summary
        assert rh["nv_seg0"] > 0 and rh["nv_seg2"] > 0, summary  # sanity: they do toggle in NV ops
        if gate:
            assert gc[1] == 32 and gc[0]["mx_seg1"] == 0 and gc[0]["nv_seg0"] > 0, summary
    out = os.environ.get("NG_SUMMARY_VCD")
    if out:
        Path(out).write_text(json.dumps(summary, indent=1))
