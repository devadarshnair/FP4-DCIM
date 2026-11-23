#!/usr/bin/env python3
"""T3.3 VCD-annotation gap diagnosis (t43 job S; results/t33_routed_sdf_gls/*_ce6e93e/NOTE.md).

routed_sdf_gls.py measures annotation as `Annotated N pin activities` / (get_pins -hierarchical * + ports) with the
openlane2 OpenSTA 2.6, which has no report_activity_annotation. This script reads the same routed netlist, libraries
and window VCD with the bundled native OpenSTA 3.1.0 (CIM_STA_BIN), which has `report_activity_annotation
-report_unannotated`, lists every unannotated pin and classifies it:

  macro           pin of an OpenRAM instance (sky130_sram*)
  tie             pin of a conb tie cell (constant; HI/LO)
  clock_gate      pin of an ICG (dlclkp / sdlclkp)
  diode           antenna diode pin
  unconnected     pin with no connection in the netlist instance (e.g. tie LO, unused Q_N)
  port            top-level port
  scope_escaped   instance has an escaped name ('.', '[' ...) and its VCD scope is spelled differently from the
                  name OpenSTA looks for (the scope exists in the VCD with a leading backslash, or not at all): a
                  name-mapping bug
  vcd_present     the VCD has a var for this pin in the matching scope, yet it was not annotated: a mapping bug
  vcd_missing     no VCD scope/var for this instance pin (by cell type in `other_by_cell`)

`--fix-escaped-scopes` rewrites the VCD header scope names of escaped instances to the spelling OpenSTA matches (the
plain hierarchical name, no backslash; checked on a 3-cell example: `\\u.b[0].x` is not matched, `u.b[0].x` is).

  python flows/pnr/t43_annot_diag.py <run dir (s3)> <window.vcd> <out dir> --design dpu [--spef]
"""
import argparse, collections, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flows/synth"))
import sweep  # noqa: E402

TOPS = {"v2_sram_m16": "column_top", "bs_int5": "bs_column_top", "dpu": "dpu_column"}
INST = re.compile(r"^\s*(sky130_\w+)\s+(\\\S+|\w+)\s*\((.*?)\);", re.S | re.M)


def netlist_instances(net: Path) -> dict:
    """instance name (unescaped, as OpenSTA prints it) -> (cell type, set of connected pin names)."""
    out = {}
    for ct, n, body in INST.findall(net.read_text()):
        pins = {p for p, c in re.findall(r"\.(\w+)\s*\(([^()]*)\)", body) if c.strip()}
        out[n[1:] if n.startswith("\\") else n] = (ct, pins, n.startswith("\\"))
    return out


def vcd_scopes(vcd: Path, top: str) -> dict:
    """VCD header: scope path relative to `top` (names joined with '/', exactly as written in the VCD) -> var names."""
    st, out = [], collections.defaultdict(set)
    with vcd.open(errors="replace") as f:
        for ln in f:
            if ln.startswith("$scope"):
                st.append(ln.split()[2])
            elif ln.startswith("$upscope"):
                st.pop()
            elif ln.startswith("$var"):
                p = ln.split()
                if top in st:
                    out["/".join(st[st.index(top) + 1:])].add(p[4])
            elif ln.startswith("$enddefinitions"):
                break
    return out


def fix_escaped_scopes(vcd: Path) -> int:
    """Strip the leading backslash of escaped instance scope names (header only, in place). Returns scopes changed."""
    tmp, n = vcd.with_suffix(".fixtmp"), 0
    with vcd.open() as fi, tmp.open("w") as fo:
        for ln in fi:
            if ln.startswith("$scope module \\"):
                p = ln.split(); p[2] = p[2][1:]; n += 1
                ln = " ".join(p) + "\n"
            fo.write(ln)
            if ln.startswith("$enddefinitions"):
                break
        while chunk := fi.read(1 << 24):
            fo.write(chunk)
    tmp.replace(vcd)
    return n


def run_sta(net: Path, top: str, vcd: Path, spef: Path | None, out: Path) -> str:
    tcl = out / "annot_diag.tcl"
    tcl.write_text(f"""read_liberty {sweep.LIB}
read_liberty {sweep.SRAM_DIR / 'lib' / (sweep.SRAM + '_TT_1p8V_25C.lib')}
read_verilog {net}
link_design {top}
create_clock -name vclk -period 25
{f'read_spef {spef}' if spef else ''}
read_vcd -scope {top} {vcd}
report_activity_annotation -report_unannotated > {out / 'unannotated.rpt'}
puts "pins [llength [get_pins -hierarchical *]] ports [llength [get_ports *]]"
""")
    sta = os.environ.get("CIM_STA_BIN", "sta")
    r = subprocess.run([sta, "-no_splash", "-exit", str(tcl)], capture_output=True, text=True)
    (out / "annot_diag_sta.log").write_text(r.stdout + r.stderr + f"\n[rc {r.returncode}]\n")
    if r.returncode or not (out / "unannotated.rpt").exists():
        raise SystemExit(f"OpenSTA 3.1.0 failed (rc {r.returncode}), see {out / 'annot_diag_sta.log'}")
    return r.stdout + (out / "unannotated.rpt").read_text()


def classify(pins: list[str], inst: dict, scopes: dict) -> dict:
    cls, other = collections.Counter(), collections.Counter()
    ex = collections.defaultdict(list)
    for p in pins:
        if "/" not in p:
            c = "port"
        else:
            i, pin = p.rsplit("/", 1)
            ct, conn, esc = inst.get(i, ("?", set(), False))
            sc_plain, sc_bs = scopes.get(i), scopes.get("\\" + i)
            if ct.startswith("sky130_sram"):
                c = "macro"
            elif "conb" in ct:
                c = "tie"
            elif "dlclkp" in ct:
                c = "clock_gate"
            elif "diode" in ct:
                c = "diode"
            elif pin not in conn:
                c = "unconnected"
            elif esc and sc_plain is None:
                c = "scope_escaped"
            elif sc_plain is not None and pin in sc_plain:
                c = "vcd_present"
            else:
                c = "vcd_missing"; other[ct] += 1
        cls[c] += 1
        if len(ex[c]) < 20:
            ex[c].append(p)
    return {"classes": dict(cls), "examples": dict(ex), "other_by_cell": dict(other.most_common(30))}


def parse_report(txt: str) -> tuple[dict, list[str]]:
    summ = {k: int(v) for k, v in re.findall(r"^(vcd|unannotated|saif|user|propagated)\s+(\d+)\s*$", txt, flags=re.M)}
    m = re.search(r"^Unannotated pins:\n(.*)", txt, flags=re.M | re.S)
    pins = [ln.strip() for ln in m.group(1).splitlines() if ln.startswith(" ") and ln.strip()] if m else []
    return summ, pins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir"); ap.add_argument("vcd"); ap.add_argument("out")
    ap.add_argument("--design", required=True, choices=sorted(TOPS))
    ap.add_argument("--spef", action="store_true", help="also read the nom SPEF (no effect on annotation; slower)")
    ap.add_argument("--fix-escaped-scopes", action="store_true", help="rewrite the VCD scopes first (in place)")
    a = ap.parse_args()
    top = TOPS[a.design]
    rd, vcd, out = Path(a.rundir).resolve(), Path(a.vcd).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    net = rd / f"final/nl/{top}.nl.v"
    res = {"design": a.design, "netlist": str(net), "vcd": str(vcd), "sta": "native OpenSTA 3.1.0 (CIM_STA_BIN)"}
    if a.fix_escaped_scopes:
        res["escaped_scopes_fixed"] = fix_escaped_scopes(vcd)
    txt = run_sta(net, top, vcd, rd / f"final/spef/nom/{top}.nom.spef" if a.spef else None, out)
    summ, pins = parse_report(txt)
    m = re.search(r"pins (\d+) ports (\d+)", txt)
    inst = netlist_instances(net)
    res |= {"summary": summ, "n_unannotated_listed": len(pins),
            "pins_hier": int(m.group(1)) if m else None, "ports": int(m.group(2)) if m else None,
            "annotated_frac_3p1": summ.get("vcd", 0) / max(1, summ.get("vcd", 0) + summ.get("unannotated", 0)),
            "escaped_instances": sum(1 for v in inst.values() if v[2]),
            **classify(pins, inst, vcd_scopes(vcd, top))}
    c = res["classes"]
    res["mapping_bug_pins"] = c.get("scope_escaped", 0) + c.get("vcd_present", 0)
    res["benign_pins"] = sum(c.get(k, 0) for k in ("macro", "tie", "clock_gate", "diode", "unconnected"))
    (out / "unannotated_pins.txt").write_text("\n".join(pins) + "\n")
    (out / "annot_diag.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
