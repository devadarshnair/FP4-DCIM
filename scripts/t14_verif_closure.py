"""T1.4 verification closure: formal obligations and functional coverage, each as a provenance run.

    python scripts/t14_verif_closure.py formal     # leaf miters + compositional tree proofs + mutations
    python scripts/t14_verif_closure.py coverage   # existing regressions (small seeds) with coverage monitors

Run both through scripts/safe_run.py (machine-wide heavy slot). Results: results/verif_closure/<run_id>/.
Plan, bins, waivers: docs/spec/coverage_plan.md (bins), tb/cov/waivers.yaml (machine-readable waivers).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from cimfp4.rtlgen.alg_extract import extract, poly_spec
from cimfp4.rtlgen.leaf_spec import emit_leaf_spec_sv
from cimfp4.rtlgen.tree import build_column_tree, build_dual_tree, emit_sv
from cimfp4.rtlgen.tree_slices import (
    emit_composed_sv,
    emit_slice_check_sv,
    emit_slice_sv,
    link_ys,
    slice_check_ys,
)
from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

TREES = {
    "oc": (9, "oc", "v2 column (column.sv)"),
    "unsigned": (1, "unsigned", "INT5 bit-serial (bs_column.sv)"),
    "tc": (9, "tc", "TC ablation (not instantiated)"),
}
SAT_HALF_TIMEOUT_S = 600


def _yosys(args: list[str], log: Path, timeout: int | None = None) -> tuple[str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(
            ["yosys", "-q", "-l", str(log), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t0
    txt = log.read_text() if log.exists() else ""
    ok = r.returncode == 0 and "SUCCESS!" in txt
    return ("PASS" if ok else "FAIL"), time.time() - t0


def _tree_terms(nl, xport: str, sport: str, rows: range, rel: bool = True) -> list:
    """Spec Σ_r val(r) as polynomial terms over the port bits of `rows` (row index relative to the port)."""
    terms = []
    bpr = nl.w_in - 1 if nl.scheme == "dual" else nl.w_in
    for j, r in enumerate(rows):
        i = j if rel else r
        for b in range(bpr):
            v = f"{xport}[{i * bpr + b}]"
            if nl.scheme == "dual":  # mag·(1 - 2s)
                terms += [((v,), 1 << b), ((v, f"{sport}[{i}]"), -(1 << (b + 1)))]
            elif nl.scheme != "unsigned" and b == nl.w_in - 1:
                terms.append(((v,), -(1 << b)))
            else:
                terms.append(((v,), 1 << b))
        if nl.scheme == "oc":
            terms.append(((f"{sport}[{i}]",), 1))
    return terms


def _alg(
    json_path: Path, top: str, outs: list[tuple[str, str]], w: int, spec_terms: list
) -> tuple[str, dict]:
    o = [(p, i, 1 << i) for p in outs for i in range(w)]
    got, st = extract(json_path, top, o, w)
    return ("PASS" if got == poly_spec(spec_terms, w) else "FAIL"), st


def _write_json(srcs: list[Path], top: str, out: Path, log: Path) -> None:
    cmd = (
        f"read_verilog -sv {' '.join(map(str, srcs))}; hierarchy -top {top}; "
        f"proc; flatten; techmap; opt_clean; write_json {out}"
    )
    r = subprocess.run(
        ["yosys", "-q", "-l", str(log), "-p", cmd], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if r.returncode:
        raise RuntimeError(r.stderr)


def _mutate_first_fa_sum(sv: str) -> str:
    """First FA sum `a ^ b ^ c` -> `a ^ b | c` (a real functional change)."""
    m = re.search(r"assign (n\d+) = ([^;]+?) \^ ([^;]+?) \^ ([^;]+?);", sv)
    assert m, "no FA found"
    return (
        sv[: m.start()] + f"assign {m.group(1)} = {m.group(2)} ^ {m.group(3)} | {m.group(4)};" + sv[m.end() :]
    )


def formal() -> int:
    cfg = {
        "part": "formal",
        "tool": "yosys sat (MiniSAT) + algebraic function extraction (alg_extract.py)",
        "sat_half_timeout_s": SAT_HALF_TIMEOUT_S,
        "trees": list(TREES) + ["dual (mono ALG only)"],
    }
    rows: list[dict] = []
    with start_run("verif_closure", cfg) as run:
        d = run.dir / "formal"
        (d / "gen").mkdir(parents=True)
        (d / "logs").mkdir()
        yv = subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip()

        def rec(ob, kind, scope, status, secs=0.0, **extra):
            rows.append(
                {
                    "obligation": ob,
                    "method": kind,
                    "scope": scope,
                    "status": status,
                    "seconds": round(secs, 2),
                    **{k: v for k, v in extra.items()},
                }
            )
            print(f"{ob:34s} {kind:10s} {status:8s} {secs:7.1f}s {extra}", flush=True)

        # ---- leaf miters (existing scripts, exhaustive over the 8 input bits by SAT) ----
        spec = REPO_ROOT / "tb/formal/gen/leaf_spec.sv"
        rec(
            "LEAF_SPEC_REGEN",
            "python",
            "tb/formal/gen/leaf_spec.sv == emit_leaf_spec_sv() (cimfp4.formats.tables)",
            "PASS" if spec.read_text() == emit_leaf_spec_sv() else "FAIL",
        )
        for ob, ys, scope in [
            ("LEAF_FP4", "tb/formal/leaf_dec_equiv.ys", "row_driver+leaf_mult (v2 leaf) == 256-entry LUT"),
            (
                "LEAF_GENERIC",
                "tb/formal/generic_leaf_equiv.ys",
                "generic_unit (DPU / INT5-style leaf) == LUT",
            ),
        ]:
            st, t = _yosys([ys], d / "logs" / f"{ob}.log")
            rec(ob, "sat", scope, st, t)

        # ---- trees ----
        for key, (w_in, scheme, used_by) in TREES.items():
            nl = build_column_tree(w_in, scheme)
            rtl = REPO_ROOT / "rtl/gen" / f"{nl.name}.sv"
            rec(
                f"REGEN_{key}",
                "python",
                f"{rtl.relative_to(REPO_ROOT)} == emit_sv(build_column_tree) [{used_by}]",
                "PASS" if rtl.read_text() == emit_sv(nl) else "FAIL",
            )
            files = {}
            for s in ("h0", "h1", "merge"):
                files[s] = d / "gen" / f"{nl.name}_slice_{s}.sv"
                files[s].write_text(emit_slice_sv(nl, s))
                (d / "gen" / f"{nl.name}_slice_{s}_chk.sv").write_text(emit_slice_check_sv(nl, s))
            comp = d / "gen" / f"{nl.name}_composed.sv"
            comp.write_text(emit_composed_sv(nl))
            # LINK: committed monolith == composition of the slices
            ys = d / f"LINK_{key}.ys"
            ys.write_text(
                link_ys(
                    nl,
                    str(rtl.relative_to(REPO_ROOT)),
                    [str(files[s]) for s in ("h0", "h1", "merge")] + [str(comp)],
                )
            )
            st, t = _yosys([str(ys)], d / "logs" / f"LINK_{key}.log")
            rec(f"LINK_{key}", "sat", "rtl/gen monolith == h0 ∘ h1 ∘ merge (structural miter)", st, t)
            # SAT slice contracts
            for s in ("merge", "h0", "h1"):
                ob = f"S_{key}_{s}"
                if s != "merge" and scheme != "unsigned" and not (key == "oc" and s == "h0"):
                    rec(
                        ob,
                        "sat",
                        "16-row half vs Σ (mod 2^w)",
                        "NOT_RUN",
                        note="same structure as S_oc_h0 (SAT timeout); see A_*",
                    )
                    continue
                ys = d / f"{ob}.ys"
                ys.write_text(
                    slice_check_ys(nl, s, [str(files[s]), str(d / "gen" / f"{nl.name}_slice_{s}_chk.sv")])
                )
                st, t = _yosys(
                    [str(ys)], d / "logs" / f"{ob}.log", timeout=SAT_HALF_TIMEOUT_S if s != "merge" else None
                )
                rec(
                    ob,
                    "sat",
                    "merge: m_s+m_c ≡ a+b+c+d" if s == "merge" else "half: s+c ≡ Σ val(r) (mod 2^w)",
                    st,
                    t,
                )
            # algebraic proofs of the half slices (contract identical to S_*_h*) and of the monolith
            w = len(nl.rows["h0_s_full"])
            for s, rng in (("h0", range(0, 16)), ("h1", range(16, 32))):
                js = d / "gen" / f"{nl.name}_slice_{s}.json"
                _write_json([files[s]], f"{nl.name}_slice_{s}", js, d / "logs" / f"json_{key}_{s}.log")
                t0 = time.time()
                st, stat = _alg(js, f"{nl.name}_slice_{s}", ["os", "oc"], w, _tree_terms(nl, "x", "sb", rng))
                rec(f"A_{key}_{s}", "alg", "half: s+c ≡ Σ val(r) (mod 2^w)", st, time.time() - t0, **stat)
            js = d / "gen" / f"{nl.name}_mono.json"
            _write_json([rtl], nl.name, js, d / "logs" / f"json_{key}_mono.log")
            wm, wh = len(nl.outputs["m_s"]), len(nl.outputs["h0_s"])
            t0 = time.time()
            st, stat = _alg(
                js, nl.name, ["m_s", "m_c"], wm, _tree_terms(nl, "in_bits", "s_bits", range(32), rel=False)
            )
            rec(
                f"A_{key}_mono_m",
                "alg",
                "rtl/gen monolith: m_s+m_c ≡ Σ_32 val(r) (mod 2^w)",
                st,
                time.time() - t0,
                **stat,
            )
            for h, rng in ((0, range(0, 16)), (1, range(16, 32))):
                t0 = time.time()
                st, stat = _alg(
                    js,
                    nl.name,
                    [f"h{h}_s", f"h{h}_c"],
                    wh,
                    _tree_terms(nl, "in_bits", "s_bits", rng, rel=False),
                )
                rec(
                    f"A_{key}_mono_h{h}",
                    "alg",
                    f"rtl/gen monolith: h{h} ≡ Σ_16 val(r) (mod 2^w_half)",
                    st,
                    time.time() - t0,
                    **stat,
                )
            # mutation sanity on the v2 tree: every method must reject a one-gate change
            if key == "oc":
                mut = d / "gen" / f"{nl.name}_slice_h0_MUT.sv"
                mut.write_text(_mutate_first_fa_sum(files["h0"].read_text()))
                js = d / "gen" / f"{nl.name}_slice_h0_MUT.json"
                _write_json([mut], f"{nl.name}_slice_h0", js, d / "logs" / "json_MUT.log")
                st, stat = _alg(
                    js, f"{nl.name}_slice_h0", ["os", "oc"], w, _tree_terms(nl, "x", "sb", range(16))
                )
                rec(
                    "MUT_ALG_h0",
                    "alg",
                    "mutated h0 (one FA sum) must FAIL",
                    "PASS" if st == "FAIL" else "FAIL",
                    0,
                    raw=st,
                )
                ys = d / "MUT_LINK.ys"
                ys.write_text(
                    link_ys(
                        nl,
                        str(rtl.relative_to(REPO_ROOT)),
                        [str(mut), str(files["h1"]), str(files["merge"]), str(comp)],
                    )
                )
                st, t = _yosys([str(ys)], d / "logs" / "MUT_LINK.log", timeout=600)
                rec(
                    "MUT_LINK",
                    "sat",
                    "mutated h0 in the composition must FAIL",
                    "PASS" if st == "FAIL" else "FAIL",
                    t,
                    raw=st,
                )
                mm = d / "gen" / f"{nl.name}_slice_merge_MUT.sv"
                mm.write_text(_mutate_first_fa_sum(files["merge"].read_text()))
                ys = d / "MUT_S_merge.ys"
                ys.write_text(
                    slice_check_ys(nl, "merge", [str(mm), str(d / "gen" / f"{nl.name}_slice_merge_chk.sv")])
                )
                st, t = _yosys([str(ys)], d / "logs" / "MUT_S_merge.log", timeout=600)
                rec(
                    "MUT_S_merge",
                    "sat",
                    "mutated merge slice must FAIL",
                    "PASS" if st == "FAIL" else "FAIL",
                    t,
                    raw=st,
                )

        # dual-polarity ablation tree: monolith by algebra only (not instantiated by any design)
        nl = build_dual_tree(9)
        rtl = REPO_ROOT / "rtl/gen" / f"{nl.name}.sv"
        rec(
            "REGEN_dual",
            "python",
            "rtl/gen dual == emit_sv(build_dual_tree)",
            "PASS" if rtl.read_text() == emit_sv(nl) else "FAIL",
        )
        js = d / "gen" / f"{nl.name}_mono.json"
        _write_json([rtl], nl.name, js, d / "logs" / "json_dual_mono.log")
        t0 = time.time()
        st, stat = _alg(
            js,
            nl.name,
            ["m_s", "m_c"],
            len(nl.outputs["m_s"]),
            _tree_terms(nl, "in_bits", "s_bits", range(32), rel=False),
        )
        rec("A_dual_mono_m", "alg", "dual monolith: m ≡ Σ mag·(1−2s)", st, time.time() - t0, **stat)

        with open(d / "obligations.csv", "w", newline="") as f:
            keys = sorted(
                {k for r in rows for k in r},
                key=lambda k: (k not in ("obligation", "method", "scope", "status", "seconds"), k),
            )
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        counts = defaultdict(int)
        for r in rows:
            counts[r["status"]] += 1
        required = [r for r in rows if r["status"] not in ("NOT_RUN", "TIMEOUT")]
        run.log_metrics(
            {
                "yosys": yv,
                "status_counts": dict(counts),
                "obligations": {r["obligation"]: r["status"] for r in rows},
                "all_required_pass": all(r["status"] == "PASS" for r in required),
            }
        )
        print(run.dir, dict(counts))
        return 0 if all(r["status"] == "PASS" for r in required) else 1


# --------------------------------------------------------------------------------------------- coverage
#: (pytest node, env overrides, design family): small seeds of the existing regressions + T1.4 directed
COV_JOBS = [
    ("tb/cocotb/test_column.py::test_column_corners", {}, "v2"),
    ("tb/cocotb/test_column.py::test_column_random", {"COLUMN_NDOTS": "100"}, "v2"),
    ("tb/cocotb/test_column.py::test_column_capture_tiles", {}, "v2"),
    ("tb/cocotb/test_nv.py::test_nv_scale_significand_pairs", {}, "v2"),
    ("tb/cocotb/test_nv.py::test_nv_random", {"NV_NDOTS": "100"}, "v2"),
    ("tb/cocotb/test_column_top.py::test_column_top[16-SRAM]", {"CT_NOPS": "2000"}, "v2"),
    ("tb/cocotb/test_column_top.py::test_column_top_pattern[SRAM]", {}, "v2"),
    ("tb/cocotb/test_macro_top.py::test_macro_sram[16-mxfp4]", {}, "v2"),
    ("tb/cocotb/test_macro_top.py::test_macro_sram[16-nvfp4]", {}, "v2"),
    ("tb/cocotb/test_macro_top.py::test_macro_pattern[SRAM-mxfp4]", {}, "v2"),
    ("tb/cocotb/test_macro_top.py::test_macro_pattern[SRAM-nvfp4]", {}, "v2"),
    ("tb/cocotb/test_macro_mixed.py::test_macro_mixed[macro_top]", {}, "v2"),
    ("tb/cocotb/test_dpu_macro.py::test_dpu_macro[mxfp4]", {}, "dpu"),
    ("tb/cocotb/test_dpu_macro.py::test_dpu_macro[nvfp4]", {}, "dpu"),
    ("tb/cocotb/test_macro_mixed.py::test_macro_mixed[dpu_macro]", {}, "dpu"),
    ("tb/cocotb/test_bs_int5.py::test_bs_int5[16]", {"CT_NOPS": "2000"}, "int5"),
    ("tb/cocotb/test_bs_int5.py::test_bs_int5_pattern[4-SRAM]", {}, "int5"),
    ("tb/cocotb/test_bs_macro.py::test_bs_macro[16]", {}, "int5"),
    ("tb/cocotb/test_bs_macro.py::test_bs_macro_pattern[4-SRAM]", {}, "int5"),
    ("tb/cocotb/test_macro_mixed.py::test_macro_mixed[bs_macro_top]", {}, "int5"),
]


def parse_dat(path: Path) -> tuple[dict[tuple[str, str], int], dict[str, list[int]]]:
    """Verilator coverage.dat -> ({(module page, cover label): hits summed over instances},
    {file: [points, points hit]} for v_line points)."""
    user: dict[tuple[str, str], int] = defaultdict(int)
    lines: dict[str, dict[int, int]] = defaultdict(dict)
    for ln in path.read_text(errors="replace").splitlines():
        if not ln.startswith("C '"):
            continue
        body, cnt = ln[3:].rsplit("' ", 1)
        kv = dict(p.split("\x02", 1) for p in body.split("\x01") if "\x02" in p)
        page = kv.get("page", "")
        if page.startswith("v_user/"):
            label = kv.get("o", "") or kv.get("O", "")
            mod = page[len("v_user/") :].split("__")[0]  # Verilator suffixes parameter specialisations
            user[(mod, label)] += int(cnt)
        elif page.startswith("v_line/"):
            f = kv.get("f", "")
            ln_no = int(kv.get("l", "0"))
            lines[f][ln_no] = lines[f].get(ln_no, 0) + int(cnt)
    lsum = {f: [len(v), sum(1 for c in v.values() if c)] for f, v in lines.items()}
    return dict(user), lsum


def coverage(only: str | None = None) -> int:
    jobs = [j for j in COV_JOBS if not only or only in j[0]]  # --only: re-run an affected subset (partial run)
    waivers = yaml.safe_load((REPO_ROOT / "tb/cov/waivers.yaml").read_text()) or {}
    cfg = {
        "part": "coverage",
        "jobs": [{"node": n, "env": e, "family": f} for n, e, f in jobs],
        "only": only,
        "monitors": "tb/cov/*.sv (bind), Verilator --coverage-user --coverage-line",
        "waivers": "tb/cov/waivers.yaml",
    }
    with start_run("verif_closure", cfg) as run:
        covd = run.dir / "coverage" / "dat"
        covd.mkdir(parents=True)
        logs = run.dir / "coverage" / "logs"
        logs.mkdir()
        job_rows = []
        for node, env, fam in jobs:
            t0 = time.time()
            e = dict(
                os.environ,
                T14_COV_DIR=str(covd),
                PYTHONPATH=f"{REPO_ROOT / 'tb/cov'}:{os.environ.get('PYTHONPATH', '')}",
                **env,
            )
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    f"--basetemp={REPO_ROOT / 'build/pytest_t14'}",
                    "-p",
                    "t14_cov_plugin",
                    node,
                    "-q",
                    "-rs",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=REPO_ROOT,
                env=e,
                capture_output=True,
                text=True,
            )
            tid = re.sub(r"[^A-Za-z0-9_.-]+", "_", node.split("/")[-1])
            (logs / f"{tid}.log").write_text(r.stdout + r.stderr)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            job_rows.append(
                {
                    "node": node,
                    "family": fam,
                    "rc": r.returncode,
                    "summary": last,
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(f"{node}: rc={r.returncode} {last} ({time.time() - t0:.0f}s)", flush=True)
        # aggregate
        hits: dict[tuple[str, str, str], int] = defaultdict(int)  # (family, module, label)
        who: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        fam_of = {re.sub(r"[^A-Za-z0-9_.-]+", "_", n.split("/")[-1]): f for n, _, f in COV_JOBS}
        for dat in sorted(covd.glob("*.dat")):
            fam = fam_of.get(dat.stem, "?")
            user, _ = parse_dat(dat)
            for (mod, label), c in user.items():
                hits[(fam, mod, label)] += c
                if c:
                    who[(fam, mod, label)].append(dat.stem)
        # line coverage: union of hit lines across all runs
        line_hits: dict[str, dict[tuple, int]] = defaultdict(dict)  # file -> {(page kind, line, point): hits}
        for dat in sorted(covd.glob("*.dat")):
            for ln in dat.read_text(errors="replace").splitlines():
                if not ln.startswith("C '"):
                    continue
                body, cnt = ln[3:].rsplit("' ", 1)
                kv = dict(p.split("\x02", 1) for p in body.split("\x01") if "\x02" in p)
                pg = kv.get("page", "")
                if pg.startswith(("v_line/", "v_branch/")):
                    k = (pg.split("/")[0], int(kv.get("l", "0")), kv.get("n", ""))
                    line_hits[kv.get("f", "")][k] = line_hits[kv.get("f", "")].get(k, 0) + int(cnt)
        bins = []
        for (fam, mod, label), c in sorted(hits.items()):
            illegal = label.startswith("ILL_")
            wv = (waivers.get(fam, {}) or {}).get(label) or (waivers.get("all", {}) or {}).get(label)
            if illegal:
                status = "PASS(0 hits)" if c == 0 else "ILLEGAL_HIT"
            elif c > 0:
                status = "HIT"
            elif wv:
                status = "WAIVED"
            else:
                status = "UNHIT"
            bins.append(
                {
                    "family": fam,
                    "module": mod,
                    "bin": label,
                    "hits": c,
                    "status": status,
                    "tests": " ".join(sorted(set(who[(fam, mod, label)]))),
                    "waiver": wv or "",
                }
            )
        with open(run.dir / "coverage" / "bins.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(bins[0]))
            wr.writeheader()
            wr.writerows(bins)
        with open(run.dir / "coverage" / "jobs.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(job_rows[0]))
            wr.writeheader()
            wr.writerows(job_rows)
        lc = []
        for fpath, d in sorted(line_hits.items()):
            rel = fpath.replace(str(REPO_ROOT) + "/", "")
            if rel.startswith(("rtl/", "tb/cov")):
                lc.append({"file": rel, "line_branch_points": len(d), "hit": sum(1 for v in d.values() if v)})
        if lc:
            with open(run.dir / "coverage" / "line_coverage.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(lc[0]))
                wr.writeheader()
                wr.writerows(lc)
        cnt = defaultdict(int)
        for b in bins:
            cnt[b["status"]] += 1
        legal = [b for b in bins if not b["bin"].startswith("ILL_")]
        summary = {
            "jobs_passed": sum(1 for j in job_rows if j["rc"] == 0),
            "jobs": len(job_rows),
            "bins_legal": len(legal),
            "bins_hit": sum(1 for b in legal if b["status"] == "HIT"),
            "bins_waived": sum(1 for b in legal if b["status"] == "WAIVED"),
            "bins_unhit": [f"{b['family']}:{b['bin']}" for b in legal if b["status"] == "UNHIT"],
            "illegal_bins": sum(1 for b in bins if b["bin"].startswith("ILL_")),
            "illegal_hit": [f"{b['family']}:{b['bin']}" for b in bins if b["status"] == "ILLEGAL_HIT"],
            "status_counts": dict(cnt),
            "line_coverage": {r["file"]: f"{r['hit']}/{r['line_branch_points']}" for r in lc},
        }
        summary["closed"] = (
            summary["jobs_passed"] == summary["jobs"]
            and not summary["bins_unhit"]
            and not summary["illegal_hit"]
        )
        run.log_metrics(summary)
        print(run.dir, summary)
        return 0 if summary["closed"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["formal", "coverage"])
    ap.add_argument("--only", help="coverage: run only jobs whose node id contains this substring (partial run)")
    a = ap.parse_args()
    sys.exit(formal() if a.part == "formal" else coverage(a.only))
