#!/usr/bin/env python3
"""T3.3 routed-SDF GLS power of one P&R'd tile on one e13 window (T3.3).

Unlike post_route_power.py (zero-delay VCD of the *pre-placement* netlist annotated by name, so CTS/repair buffers
carry no activity and the Clock group is SDC-assumed), this simulates the **routed** netlist itself (final/nl: CTS
tree, clock gates, hold/repair buffers, antenna diodes) with the flow's own post-route SDF (final/sdf/nom_tt_025C_1v80:
OpenSTA cell IOPATH delays computed on the SPEF, plus INTERCONNECT wire delays, Icarus `-gspecify -ginterconnect`).
The clock is therefore propagated through the real clock tree (insertion delay and skew come from the netlist + SDF,
not from an SDC clock model), and the VCD carries the activity of every net, including the CTS/ICG nets.
The clock-tree delta race seen in earlier runs does not apply: it was caused by delta-delay ordering in a *zero-delay* simulation of the clock tree.

Power: OpenSTA on the same netlist + final SPEF (nom) + the rebased window VCD, report_power plus
report_activity_annotation (acceptance: >= 99 % of pins annotated from the VCD). Energy bookkeeping as in e13 /
post_route_power.py. The zero-delay reference for the routed/zero factor is post_route_power.py on the same window
(W lane, power_s3/<window>/postroute_power.json); flows/pnr/t33_summary.py forms the per-window factors and the CI.

  python flows/pnr/routed_sdf_gls.py <run dir (s3)> <out dir> --design v2_sram_m16|bs_int5|dpu --window real0
         [--nops N]  (smoke: N random ops from the harness generator instead of an e13 program; no power)
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e13_power"))
sys.path.insert(0, str(ROOT / "flows/synth"))
import run as e13  # noqa: E402
import sweep  # noqa: E402
from cimfp4.power.vcd import slice_vcd  # noqa: E402

CORNER = "nom_tt_025C_1v80"
CLK_PS = 25000  # e13 / post_route_power window clock (40 MHz)
READY_NS = 6  # op_ready sampling point after the falling edge (port hold/input buffers + wires, see test_column_top)


PHYS = re.compile(r"sky130_fd_sc_hd__(decap|fill|tapvpwrvgnd|diode)_\d+$")  # no signal arcs, no SDF CELL


def _san(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", lambda m: f"_{ord(m.group()):02x}_", name)


def rename_instances(src: Path, dst: Path) -> tuple[dict, dict]:
    """Icarus 12's $sdf_annotate cannot address an instance whose (escaped) name contains '.': it splits the SDF
    path on every '.', escaped or not, and also when (DIVIDER /) is used (checked on a 2-cell test). The flattened
    OpenROAD netlist names many instances `\\g_col[1].u_col.x` / `\\clkbuf_leaf_9_u_lanes.g_sram...`, so the SDF of
    those cells was unannotatable. Simulation copy only: rename every escaped *instance* name to a simple identifier
    (non-word chars -> _hh_); nets/ports are untouched, so the testbench and the VCD net names are unchanged.
    Returns (orig->new map, instance cell types by new name)."""
    out, ren, types, seen = [], {}, {}, set()
    pat = re.compile(r"^(\s*)(sky130_\w+)\s+(\\\S+|\w+)(\s+\()")
    lines = src.read_text().splitlines(keepends=True)
    for ln in lines:
        m = pat.match(ln)
        if m:
            name = m.group(3)
            if name.startswith("\\"):
                new = _san(name[1:])
                ren[name[1:]] = new
                ln = f"{m.group(1)}{m.group(2)} {new}{m.group(4)}" + ln[m.end():]
                name = new
            if name in seen:
                raise SystemExit(f"instance rename collision: {name}")
            seen.add(name); types[name] = m.group(2)
        out.append(ln)
    dst.write_text("".join(out))
    return ren, types


def fix_sdf(src: Path, dst: Path, ren: dict) -> dict:
    """Make the OpenSTA routed SDF readable by Icarus 12:
    - `a::b` triples -> full `a:a:b` (Icarus wants min:typ:max);
    - instance paths (CELL INSTANCE and INTERCONNECT endpoints) -> the renamed simulation instances (rename_instances);
    - macro CELLs (sky130_sram*) dropped: the SRAM GLS model has no specify block, and its bus-pin IOPATHs are a
      syntax error for Icarus that aborted the whole file;
    - TIMINGCHECK blocks dropped (Icarus ignores them: 'TIMINGCHECK not supported');
    - INTERCONNECTs driven by a top-level input port dropped: Icarus has no driving module port for them ('Could
      not find intermodpath' / 'Could not find net'), so they were never applied. This removes the clk port -> clock
      root wire delay (~0.4-0.5 ns, common to every sink, so no skew) and the input-port wire delays (tens of ps;
      the testbench drives inputs at the falling edge). Every std-cell IOPATH and every cell->cell / cell->port
      INTERCONNECT is kept."""
    t = re.sub(r"(?<=[( ])(-?[\d.]+)::(-?[\d.]+)(?=\))", r"\1:\1:\2", src.read_text()).rstrip()
    assert t.endswith(")"), "SDF does not end with the DELAYFILE ')'"
    t = t[:-1].rstrip()
    unesc = lambda s: re.sub(r"\\(.)", r"\1", s)

    def inst(path: str) -> str:
        n = unesc(path)
        return ren.get(n, path)

    def pin(path: str) -> str:  # inst.pin (split at the last unescaped '.') or a top-level port
        m = re.match(r"^((?:[^.\\]|\\.)+)\.([^.]+)$", path)
        return f"{inst(m.group(1))}.{m.group(2)}" if m else path

    head, *cells = re.split(r"\n (?=\(CELL\n)", t)
    kept, dropped, nt, ann, port_in, nport = [], [], 0, {}, [], 0
    for c in cells:
        ct = re.search(r'\(CELLTYPE "([^"]+)"\)', c).group(1)
        im = re.search(r"\(INSTANCE ?([^)]*)\)", c)
        if ct.startswith("sky130_sram"):
            dropped.append(ct); continue
        nt += c.count("(TIMINGCHECK")
        c = re.sub(r"\n *\(TIMINGCHECK\n.*?\n  \)(?=\n)", "", c, flags=re.S)
        n0 = c.count("(INTERCONNECT")
        for m in re.finditer(r"\n *\(INTERCONNECT ((?:[^.\s\\]|\\.)+) \S+ \(([-\d.]+):", c):
            port_in.append((m.group(1), float(m.group(2))))
        c = re.sub(r"\n *\(INTERCONNECT (?:[^.\s\\]|\\.)+ \S+ [^\n]*", "", c)
        nport += n0 - c.count("(INTERCONNECT")
        c = re.sub(r"\(INTERCONNECT (\S+) (\S+) ", lambda m: f"(INTERCONNECT {pin(m.group(1))} {pin(m.group(2))} ", c)
        if im and im.group(1):
            new = inst(im.group(1))
            c = c.replace(im.group(0), f"(INSTANCE {new})", 1)
            if "(IOPATH" in c:
                ann[new] = ct
        kept.append(c)
    t = "\n ".join([head] + kept) + "\n)\n"
    dst.write_text(t)
    leftover = len(re.findall(r"\(INSTANCE [^)]*\\", t))
    return {"interconnect": t.count("(INTERCONNECT"), "cells": len(kept) - 1, "iopath": t.count("(IOPATH"),
            "dropped_macro_cells": dropped, "dropped_timingcheck_blocks": nt,
            "dropped_input_port_interconnects": nport,
            "dropped_input_port_max_ns": {p: max(d for q, d in port_in if q == p) for p in ("clk",) if any(q == p for q, _ in port_in)}
            | {"other_ports_max": max((d for q, d in port_in if q != "clk"), default=0.0)}, "escaped_instances_left": leftover,
            "_iopath_cells": ann}


def sram_model(src: Path, dst: Path) -> dict:
    """The OpenRAM GLS model drives dout to X at posedge clk + T_HOLD (1 ns, 'value is arbitrary') and the new read
    data at negedge + DELAY. Its liberty (which STA signed off against) launches dout only from the *falling* clock
    edge (timing_type falling_edge, 0.34-0.48 ns), i.e. dout is stable across the rising edge. With the routed clock
    tree the capture flops behind the ICG see their edge ~1.7 ns after the macro clock pin (v2: 8225.90 -> 8227.62
    ns), so the model's 1 ns posedge X was captured although no liberty arc changes dout there (v2/DPU smoke: acc =
    X). Simulation copy: T_HOLD = half clock period, so the X window is [negedge, negedge + DELAY], consistent with
    the liberty; read/write function and DELAY unchanged."""
    t = src.read_text()
    new = f"parameter T_HOLD = {CLK_PS / 2000:g} ;"
    t2, n = re.subn(r"parameter T_HOLD = 1 ;", new, t)
    if n != 1:
        raise SystemExit(f"{src}: T_HOLD parameter not found")
    dst.write_text(t2)
    return {"src": str(src), "T_HOLD_ns": CLK_PS / 2000, "orig_T_HOLD_ns": 1, "DELAY_ns": 3}


def coverage(types: dict, ann: dict, sim_log: str) -> dict:
    """Std-cell instances (minus decap/fill/tap/diode, which have no timing arcs) that get an IOPATH from the SDF
    and that Icarus did not reject (any SDF ERROR or 'Cannot find' / 'Unable to match' / 'is not a' warning naming
    the instance counts as not annotated)."""
    logic = {n: c for n, c in types.items() if c.startswith("sky130_fd_sc_hd__") and not PHYS.match(c)}
    errs = re.findall(r"^SDF (ERROR|WARNING): .*$", sim_log, flags=re.M)
    bad_lines = [l for l in re.findall(r"^SDF (?:ERROR|WARNING): (.*)$", sim_log, flags=re.M)
                 if "TIMINGCHECK not supported" not in l]
    bad = {n for n in ann if any(re.search(rf"\b{re.escape(n)}\b", l) for l in bad_lines)} if bad_lines else set()
    got = [n for n in logic if n in ann and n not in bad]
    no_arc = sorted({c for n, c in logic.items() if n not in ann})
    return {"std_cells": len(logic), "annotated": len(got), "coverage": len(got) / len(logic) if logic else 0.0,
            "sdf_messages": len(errs), "sdf_bad_messages": len(bad_lines), "sdf_bad_examples": bad_lines[:10],
            "cell_types_without_iopath": no_arc,
            "physical_excluded": sum(1 for c in types.values() if PHYS.match(c))}


def restore_vcd_scopes(vcd: Path, ren: dict, style: str = "escaped") -> int:
    """Put the original escaped instance names back into the VCD $scope lines (header only), so OpenSTA read_vcd
    sees the names of the real netlist. Returns the number of scopes renamed. style "escaped" writes `\\name` (as
    built for T3.3); "plain" writes `name` without the backslash, the spelling OpenSTA read_vcd matches (t43: a
    3-cell check with OpenSTA 3.1.0 left the pins of a `\\u.b[0].x` scope unannotated, `u.b[0].x` annotated)."""
    back = {v: ("\\" if style == "escaped" else "") + k for k, v in ren.items()}
    with vcd.open() as fi:  # already restored (e.g. --power-only on a VCD of an earlier run): no 1.4 GB rewrite
        hit = False
        for ln in fi:
            if ln.startswith("$enddefinitions"):
                break
            if ln.startswith("$scope module ") and ln.split()[2] in back:
                hit = True
                break
        if not hit:
            return 0
    tmp, n = vcd.with_suffix(".tmp"), 0
    with vcd.open() as fi, tmp.open("w") as fo:
        for ln in fi:
            if ln.startswith("$scope module "):
                parts = ln.split()
                if parts[2] in back:
                    parts[2] = back[parts[2]]; n += 1
                    ln = " ".join(parts) + "\n"
            fo.write(ln)
            if ln.startswith("$enddefinitions"):
                break
        while chunk := fi.read(1 << 24):
            fo.write(chunk)
    tmp.replace(vcd)
    return n


def trim_cells(lib: Path, netlist: Path, dst: Path) -> dict:
    """Keep only the include-guarded cell blocks of sky130_fd_sc_hd.v that the netlist uses (closure over the
    cells they instantiate). Text of every kept block is unchanged; this only cuts Icarus compile memory/time
    (the full library with specify blocks needs > 1.5 GB)."""
    t = lib.read_text()
    blocks = re.findall(r"(`ifndef (SKY130_\w+_V)\n.*?`endif\s*// \2\n)", t, flags=re.S)
    mods = {}
    for body, _ in blocks:
        for m in re.findall(r"^module (\w+)", body, flags=re.M):
            mods.setdefault(m, []).append(body)
    head = t[: t.index(blocks[0][0])] if blocks else ""
    used = set(re.findall(r"^\s*(sky130_\w+)\s", netlist.read_text(), flags=re.M))
    keep, todo = [], list(used & set(mods))
    seen = set(todo)
    while todo:
        m = todo.pop()
        for body in mods[m]:
            if body not in keep:
                keep.append(body)
            for sub in re.findall(r"^\s*(sky130_fd_sc_hd__\w+)\s", body, flags=re.M):
                if sub in mods and sub not in seen:
                    seen.add(sub); todo.append(sub)
    order = {b: i for i, (b, _) in enumerate(blocks)}
    dst.write_text(head + "".join(sorted(keep, key=order.get)))
    return {"cell_types_in_netlist": len(used), "lib_modules_kept": len(seen), "blocks_kept": len(keep),
            "not_in_lib": sorted(u for u in used if u not in mods and not u.startswith("sky130_sram"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir"); ap.add_argument("out")
    ap.add_argument("--design", required=True, choices=["v2_sram_m16", "bs_int5", "dpu"])
    ap.add_argument("--window", default="real0"); ap.add_argument("--prog-exp", default="e13_power_rb")
    ap.add_argument("--nops", type=int, default=0, help="smoke: random ops, no program, no power")
    ap.add_argument("--power-only", action="store_true",
                    help="reuse out/window.vcd + sim.log/r.xml/window.json of a finished window GLS: OpenSTA power only")
    ap.add_argument("--sta-mem", default="16g", help="docker --memory of the OpenSTA power step (4000m OOMed on the "
                    "1.4 GB full-hierarchy window VCDs)")
    ap.add_argument("--keep-vcd", action="store_true", help="keep window.vcd (default: sha256 + size only, disk)")
    ap.add_argument("--vcd-scope-style", choices=["escaped", "plain"], default="escaped",
                    help="escaped-instance scope names in the window VCD (restore_vcd_scopes); t43 probes plain")
    ap.add_argument("--annot-diag", action="store_true",
                    help="t43: before the VCD is deleted, list/classify unannotated pins (flows/pnr/t43_annot_diag.py, "
                         "native OpenSTA 3.1.0) -> out/annot_diag/")
    a = ap.parse_args()
    top = {"v2_sram_m16": "column_top", "bs_int5": "bs_column_top", "dpu": "dpu_column"}[a.design]
    rd, out = Path(a.rundir).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    net = rd / f"final/nl/{top}.nl.v"
    spef = rd / f"final/spef/nom/{top}.nom.spef"
    sdf_src = rd / f"final/sdf/{CORNER}/{top}__{CORNER}.sdf"
    for p in (net, spef, sdf_src):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    sdf, net_sim = out / "routed.sdf", out / f"{top}.sim.nl.v"
    ren, itypes = rename_instances(net, net_sim)
    sdf_stats = fix_sdf(sdf_src, sdf, ren)
    ann_cells = sdf_stats.pop("_iopath_cells")
    prog_path = ROOT / "build/power" / a.prog_exp / "programs" / f"{a.window}.npz"

    # --- routed SDF GLS (sky130 cell models with specify blocks: no FUNCTIONAL / UNIT_DELAY)
    from cocotb_tools.runner import get_results, get_runner
    vcd, win = out / "dump.vcd", out / "window.json"
    if a.power_only:
        res, w = power_only_state(a, out, net, sdf_src, spef, sdf_stats, ann_cells, itypes, ren)
        sliced = out / "window.vcd"
    else:
        res, w, sliced = None, None, None
    if res is None:
        res, w, sliced = simulate(a, top, out, rd, net, net_sim, sdf, sdf_src, spef, sdf_stats, ann_cells, itypes, ren,
                                  prog_path, vcd, win, get_results, get_runner)
        if res is None:
            return
    power(a, top, out, net, spef, sliced, res, w, prog_path)


def power_only_state(a, out, net, sdf_src, spef, sdf_stats, ann_cells, itypes, ren):
    """--power-only: rebuild the GLS part of routed_gls.json from the finished window's files (no re-simulation)."""
    from cocotb_tools.runner import get_results
    need = [out / n for n in ("window.vcd", "window.json", "sim.log", "r.xml")]
    for p in need:
        if not p.exists():
            raise SystemExit(f"--power-only: missing {p}")
    slog = (out / "sim.log").read_text(errors="replace")
    cov = coverage(itypes, ann_cells, slog)
    ok = get_results(out / "r.xml") == (1, 0)
    w = json.loads((out / "window.json").read_text())
    res = {"design": a.design, "window": a.window, "netlist": str(net), "sdf": str(sdf_src), "spef": str(spef),
           "sdf_stats": sdf_stats, "clk_ps": CLK_PS, "ready_ns": READY_NS, "gls_pass": ok,
           "mismatches": 0 if ok else None, "instances_renamed": len(ren),
           "sram_model": {"T_HOLD_ns": CLK_PS / 2000, "orig_T_HOLD_ns": 1, "DELAY_ns": 3},
           "sdf_coverage": cov, "accept_sdf_coverage_ge_99pct": cov["coverage"] >= 0.99, **w,
           "power_only": "GLS from the earlier run of this window (sim.log, r.xml, window.json, window.vcd reused)"}
    if not ok or cov["coverage"] < 0.99:
        raise SystemExit(f"--power-only: earlier GLS not usable (pass={ok}, coverage={cov['coverage']:.4f})")
    res["vcd_scopes_restored"] = restore_vcd_scopes(out / "window.vcd", ren, a.vcd_scope_style)
    res["vcd_scope_style"] = a.vcd_scope_style
    return res, w


def simulate(a, top, out, rd, net, net_sim, sdf, sdf_src, spef, sdf_stats, ann_cells, itypes, ren, prog_path, vcd,
             win, get_results, get_runner):
    runner = get_runner("icarus")
    orig = runner._test_command
    runner._test_command = lambda: [["-vcd" if x == "-none" else x for x in c] for c in orig()]
    cells = out / "cells_used.v"
    trim = trim_cells(e13.CELLS / "sky130_fd_sc_hd.v", net_sim, cells)
    sram_sim, sram_note = out / "sram_gls_sim.v", sram_model(e13.SRAM_GLS, out / "sram_gls_sim.v")
    runner.build(sources=[e13.CELLS / "primitives.v", cells, net_sim, sram_sim,
                          ROOT / "flows/power/vcd_dump.v", ROOT / "flows/power/sdf_annotate.v"],
                 hdl_toplevel=top, build_dir=out / "b", always=True,
                 defines={"VCD_FILE": f'"{vcd}"', "VCD_TOP": top, "SDF_FILE": f'"{sdf}"', "SDF_TOP": top},
                 build_args=["-s", "vcd_dump", "-s", "sdf_annotate", "-gspecify", "-ginterconnect"],
                 timescale=("1ns", "1ps"))
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tb/cocotb'}", "CT_WINDOW_OUT": str(win), "CT_M": "16",
           "CT_CLK_PS": str(CLK_PS), "CT_READY_NS": str(READY_NS)}
    env |= {"CT_NOPS": str(a.nops)} if a.nops else {"CT_PROGRAM": str(prog_path), "CT_NOPS": "0"}
    xml = runner.test(hdl_toplevel=top, test_module="test_column_top", test_dir=ROOT / "tb/cocotb",
                      build_dir=out / "b", results_xml=str(out / "r.xml"), extra_env=env, log_file=out / "sim.log")
    slog = (out / "sim.log").read_text(errors="replace") if (out / "sim.log").exists() else ""
    print("\n".join(slog.splitlines()[-25:]))
    cov = coverage(itypes, ann_cells, slog)
    print(f"SDF annotation coverage: {cov['annotated']}/{cov['std_cells']} = {100 * cov['coverage']:.3f} % "
          f"(SDF messages {cov['sdf_messages']}, rejected {cov['sdf_bad_messages']})")
    ok = get_results(xml) == (1, 0)
    w = json.loads(win.read_text()) if win.exists() else {}
    res = {"design": a.design, "window": a.window if not a.nops else f"smoke_nops{a.nops}", "netlist": str(net),
           "sdf": str(sdf_src), "spef": str(spef), "sdf_stats": sdf_stats, "cell_trim": trim, "clk_ps": CLK_PS, "ready_ns": READY_NS,
           "gls_pass": ok, "mismatches": 0 if ok else None, "instances_renamed": len(ren), "sram_model": sram_note,
           "sdf_coverage": cov, "accept_sdf_coverage_ge_99pct": cov["coverage"] >= 0.99, **w}
    if not ok or cov["coverage"] < 0.99:
        (out / "routed_gls.json").write_text(json.dumps(res, indent=2))
        raise SystemExit("routed SDF GLS mismatch (see r.xml / sim.log)" if not ok else
                         f"SDF annotation coverage {cov['coverage']:.4f} < 0.99 (see sim.log)")
    if a.nops:  # smoke stops after the bit-exact routed GLS (--keep-vcd: dump.vcd kept for annotation diagnosis)
        if not a.keep_vcd:
            vcd.unlink(missing_ok=True)
        (out / "routed_gls.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return None, None, None
    sliced = out / "window.vcd"
    res["vcd_slice"] = slice_vcd(vcd, sliced, w["t_start_ns"], w["t_end_ns"], rebase=True)
    vcd.unlink()
    res["vcd_scopes_restored"] = restore_vcd_scopes(sliced, ren, a.vcd_scope_style)
    res["vcd_scope_style"] = a.vcd_scope_style
    return res, w, sliced


def power(a, top, out, net, spef, sliced, res, w, prog_path):
    # --- OpenSTA + SPEF + VCD (all nets, incl. clock tree), annotation coverage
    (out / "sta_routed.tcl").write_text("""read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
set_propagated_clock [all_clocks]
read_spef $::env(SPEF)
read_vcd -scope $::env(SCOPE) $::env(VCD)
set f [open $::env(ANN) w]
puts $f "pins [llength [get_pins -hierarchical *]]"
puts $f "ports [llength [get_ports *]]"
puts $f "internal [llength [get_pins -hierarchical * -filter "direction == internal"]]"
close $f
report_power -digits 6 > $::env(OUT)
report_power -instances [get_cells -hierarchical -filter "ref_name =~ sky130_sram*"] -digits 6 >> $::env(OUT)
exit
""")
    h = str(Path.home())
    senv = {"NETLIST": net, "TOP": top, "LIB": sweep.LIB,
            "SRAM_LIB": sweep.SRAM_DIR / "lib" / (sweep.SRAM + "_TT_1p8V_25C.lib"),
            "SDC": ROOT / "flows/synth/constraints/column.sdc", "CLK_PERIOD_NS": CLK_PS / 1e3, "SPEF": spef,
            "VCD": sliced, "SCOPE": top, "OUT": out / "power.rpt", "ANN": out / "annotation.rpt"}
    cmd = ["docker", "run", "--rm", f"--memory={a.sta_mem}", "-v", f"{ROOT}:{ROOT}", "-v", f"{h}/.ciel:{h}/.ciel",
           "-w", str(ROOT)]
    if not out.is_relative_to(ROOT):  # a second mount of the same path is a docker error
        cmd[-2:-2] = ["-v", f"{out}:{out}"]
    for k, v in senv.items():
        cmd += ["-e", f"{k}={v}"]
    r = subprocess.run(cmd + [e13.IMG, "sta", "-no_splash", "-exit", str(out / "sta_routed.tcl")],
                       capture_output=True, text=True)
    (out / "sta.log").write_text(r.stdout + r.stderr + f"\n[docker/sta exit code {r.returncode}]\n")
    res["sta_rc"], res["sta_mem"] = r.returncode, a.sta_mem
    if r.returncode != 0 or not (out / "annotation.rpt").exists() or not (out / "power.rpt").exists():
        (out / "routed_gls.json").write_text(json.dumps(res, indent=2, default=str))
        raise SystemExit(f"OpenSTA power step failed (rc {r.returncode}; 137 = OOM-killed, see dmesg / --sta-mem)")
    ann = parse_annotation((out / "annotation.rpt").read_text(), r.stdout + r.stderr)
    pw = e13.parse_power((out / "power.rpt").read_text())
    dur_s = (w["t_end_ns"] - w["t_start_ns"]) * 1e-9
    macs = w["n_ops"] * e13.MACS_PER_OP
    prog = dict(__import__("numpy").load(prog_path, allow_pickle=True))
    e_access_pj = yaml.safe_load((ROOT / "flows/pdk/sram_macros.yaml").read_text())["macros"][sweep.SRAM][
        "read_energy_pj_per_word_tt"]
    p_logic = pw["p_total_w"] - pw["p_sram_sta_w"]
    res |= {**pw, "annotation": ann, "macs": macs, "window_ns": dur_s * 1e9, "p_logic_w": p_logic,
            "e_logic_fj_per_mac": p_logic * dur_s / macs * 1e15,
            "e_sram_fj_per_mac": e13.sram_accesses(prog, a.design) * e_access_pj * 1e3 / macs,
            "e_leak_fj_per_mac": pw["p_leakage_w"] * dur_s / macs * 1e15}
    res["e_total_fj_per_mac"] = res["e_logic_fj_per_mac"] + res["e_sram_fj_per_mac"]
    # t43: the gate uses the fraction without liberty-internal pins when available (raw kept as annotated_frac)
    res["accept_annotation_ge_99pct"] = ann.get("annotated_frac_excl_internal", ann.get("annotated_frac", 0)) >= 0.99
    import hashlib
    if a.annot_diag:
        d = subprocess.run([sys.executable, str(ROOT / "flows/pnr/t43_annot_diag.py"), str(net.parents[2]), str(sliced),
                            str(out / "annot_diag"), "--design", a.design], capture_output=True, text=True)
        (out / "annot_diag.log").write_text(d.stdout + d.stderr)
        f = out / "annot_diag" / "annot_diag.json"
        if f.exists():
            j = json.loads(f.read_text())
            res["annot_diag"] = {k: j[k] for k in ("summary", "annotated_frac_3p1", "classes", "mapping_bug_pins",
                                                    "benign_pins") if k in j}
    res["window_vcd_bytes"] = sliced.stat().st_size
    with sliced.open("rb") as f:
        res["window_vcd_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    if not a.keep_vcd:
        sliced.unlink()
    (out / "routed_gls.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


def parse_annotation(counts: str, log: str) -> dict:
    """OpenSTA 2.6 (openlane2 2.3.10) has no report_activity_annotation: read_vcd logs `Annotated N pin activities.`
    and the script writes the design's pin, port and liberty-internal pin counts; annotated_frac = N / (pins + ports)
    (as T3.3 ce6e93e), annotated_frac_excl_internal = N / (pins - internal + ports) (t43). Tap/fill/decap cells have
    no signal pins, so they do not enter the denominator."""
    c = {k: int(v) for k, v in re.findall(r"^(pins|ports|internal) (\d+)$", counts, flags=re.M)}
    m = re.findall(r"Annotated (\d+) pin activities", log)
    n = int(m[-1]) if m else 0
    tot = c.get("pins", 0) + c.get("ports", 0)
    res = {"annotated_pins": n, **c, "annotated_frac": n / tot if tot else 0.0}
    if "internal" in c:  # t43: liberty internal pins (flop IQ/IQ_N, ICG M0) never appear in a VCD; OpenSTA 3.1.0's
        # report_activity_annotation excludes them too (v2 smoke: 2971 = 2 x 1484 flops + 3 ICGs; 2.6 count / this
        # denominator = 3.1.0 vcd / (vcd + unannotated) exactly)
        res["annotated_frac_excl_internal"] = n / (tot - c["internal"]) if tot > c["internal"] else 0.0
    return res


if __name__ == "__main__":
    main()
