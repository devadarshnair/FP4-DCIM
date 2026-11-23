"""e20 cell caps - layout extraction of the OpenRAM sky130 dual-port bitcell (magic, from the macro GDS
shipped in the PDK) and a schematic-vs-extracted comparison of the capacitance each bitcell presents to
its bitlines and wordlines (ngspice, tt, 1.8 V).

Why: the PDK's macro SPICE netlist is the LVS (schematic) netlist: no wiring, and the transistors carry
no source/drain areas (AD/AS/PD/PS), so junction capacitance is nearly absent. The ratio
k = C(extracted) / C(schematic) per bitline/wordline quantifies what the e20 "devices-only" energy
misses in the array, and the extracted cell is reused by run.py --cell extracted.

Run: python scripts/safe_run.py --max-rss-mb 3000 -- python experiments/e20_sram_energy/cell_caps.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

PDK = Path.home() / ".ciel/sky130A"
MACRO = "sky130_sram_1kbyte_1rw1r_32x256_8"
GDS = PDK / "libs.ref/sky130_sram_macros/gds" / f"{MACRO}.gds"
NETLIST = PDK / "libs.ref/sky130_sram_macros/spice" / f"{MACRO}.spice"
MODELS = PDK / "libs.tech/ngspice/sky130.lib.spice"
CELL = "sky130_fd_bd_sram__openram_dp_cell"
IMG = "ghcr.io/efabless/openlane2:2.3.10"
BUILD = REPO_ROOT / "build/e20_sram_energy/cell_caps"
PINS = ["BL0", "BR0", "BL1", "BR1", "WL0", "WL1", "VDD", "GND"]


def extract(out: Path, cell: str = CELL) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    h = str(Path.home())
    tcl = REPO_ROOT / "experiments/e20_sram_energy/extract_cell.tcl"
    cmd = ["docker", "run", "--rm", "--memory=2g", "-v", f"{h}/.ciel:{h}/.ciel", "-v", f"{REPO_ROOT}:{REPO_ROOT}",
           "-e", f"GDS={GDS}", "-e", f"CELL={cell}", "-e", f"OUT={out}", "-w", str(out), IMG,
           "magic", "-dnull", "-noconsole", "-rcfile", "/dev/null", "-T", str(PDK / "libs.tech/magic/sky130A.tech"), str(tcl)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / f"magic_{cell}.log").write_text(r.stdout + r.stderr)
    sp = out / f"{cell}.ext.spice"
    if not sp.exists():
        raise RuntimeError("magic extraction failed:\n" + (r.stdout + r.stderr)[-3000:])
    return sp


def subckt_ports(text: str, name: str) -> list[str]:
    m = re.search(rf"^\.subckt\s+{name}\s+(.*)$", text, re.M | re.I)
    if not m:
        raise RuntimeError(f"subckt {name} not found")
    return m.group(1).split()


def wrapper(ext_sp: Path, out: Path, cell: str = CELL, name_q: bool = True) -> Path:
    """Extracted cell as `<cell>_ext` with the schematic pin order (port names matched case-insensitively)."""
    t = ext_sp.read_text()
    ports = subckt_ports(t, cell)
    low = {p.lower(): p for p in ports}
    missing = [p for p in PINS if p.lower() not in low]
    if missing:
        raise RuntimeError(f"extracted ports {ports} miss {missing}")

    def node(p: str) -> str:
        lp = p.lower()
        if lp in [q.lower() for q in PINS]:
            return PINS[[q.lower() for q in PINS].index(lp)]
        if lp in ("vpb", "vpwr"):
            return "VDD"
        if "sub" in lp or lp in ("vnb", "vgnd"):
            return "GND"
        return f"int_{p}"
    t = re.sub(r"\s+\*\*FLOATING", "", t)
    # name the storage nodes like the schematic (Q: access device BL0/WL0, Q_bar: BR0/WL0) for .ic
    for bl, qname in (("BL0", "Q"), ("BR0", "Q_bar")) if name_q else ():
        for ln in t.splitlines():
            f = ln.split()
            if f and f[0].startswith("X") and f[2] == "WL0" and bl in (f[1], f[3]):
                nd = f[3] if f[1] == bl else f[1]
                t = re.sub(rf"(?<![\w#]){re.escape(nd)}(?![\w#])", qname, t)
                break
        else:
            raise RuntimeError(f"no {bl}/WL0 access device in the extracted cell")
    body = re.sub(rf"^\.subckt\s+{cell}\b", f".subckt {cell}_raw", t, flags=re.M | re.I)
    body = re.sub(rf"^\.ends\s+{cell}\b", ".ends", body, flags=re.M | re.I)
    body = re.sub(r"^\.end\s*$", "", body, flags=re.M | re.I)
    w = [body, f".subckt {cell}_ext " + " ".join(PINS),
         "Xraw " + " ".join(node(p) for p in ports) + f" {cell}_raw",
         ".ends"]
    out.write_text("\n".join(w) + "\n")
    return out


def cap_deck(cell: str, include: list[Path], pin: str, wd: Path, q: float) -> Path:
    """Charge drawn by `pin` when ramped 0 -> 1.8 V (others at their idle bias) for 10 cells."""
    bias = {"BL0": 1.8, "BR0": 1.8, "BL1": 1.8, "BR1": 1.8, "WL0": 0.0, "WL1": 0.0, "VDD": 1.8, "GND": 0.0}
    lines = ["* cap test", f'.lib "{MODELS}" tt'] + [f'.include "{p}"' for p in include]
    nodes = []
    for p in PINS:
        n = p.lower()
        nodes.append(n)
        if p == pin:
            lines.append(f"Vt {n} 0 PWL(0 0 1n 0 2n 1.8)")
        elif p != "GND":
            lines.append(f"V{n} {n} 0 {bias[p]}")
    nodes[-1] = "0"
    for i in range(10):
        lines.append(f"X{i} " + " ".join(nodes) + f" {cell}")
    lines += [".tran 10p 3n", ".control", "run", "meas tran qq integ i(Vt) from=0.9n to=2.1n", "quit", ".endc", ".end"]
    deck = wd / f"cap_{cell}_{pin}_{q}.sp"
    deck.write_text("\n".join(lines) + "\n")
    return deck


def run_q(deck: Path) -> float:
    r = subprocess.run(["ngspice", "-b", deck.name], cwd=deck.parent, capture_output=True, text=True)
    m = re.search(r"^qq\s+=\s+([-+\d.eE]+)", r.stdout, re.M)
    if not m:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return -float(m.group(1))


def main() -> None:
    cfg = {"gds": str(GDS), "cell": CELL, "magic_image": IMG, "models": f"{MODELS} tt",
           "method": "10 cells, pin ramped 0->1.8 V in 1 ns, other bitlines at 1.8 V, wordlines at 0 V; C = Q/1.8"}
    with start_run("e20_sram_cell_caps", cfg) as run:
        wd = BUILD / run.run_id
        ext = extract(wd)
        wrap = wrapper(ext, wd / f"{CELL}_ext.spice")
        (run.dir / f"{CELL}.ext.spice").write_text(ext.read_text())
        (wd / ".spiceinit").write_text("set ngbehavior=hsa\nset skywaterpdk\nset ng_nomodcheck\noption noinit\n")
        res = {}
        for pin in ("BL0", "BR0", "BL1", "BR1", "WL0", "WL1"):
            c_s = run_q(cap_deck(CELL, [NETLIST], pin, wd, 0)) / 1.8 / 10
            c_e = run_q(cap_deck(f"{CELL}_ext", [wrap], pin, wd, 0)) / 1.8 / 10
            res[pin] = {"c_schematic_ff": c_s * 1e15, "c_extracted_ff": c_e * 1e15, "k": c_e / c_s}
        bl = [res[p]["k"] for p in ("BL0", "BR0", "BL1", "BR1")]
        wl = [res[p]["k"] for p in ("WL0", "WL1")]
        run.save_csv("cell_caps.csv", [{"pin": p, **v} for p, v in res.items()])
        run.log_metrics({"per_pin": res, "k_wire_bitline": sum(bl) / 4, "k_wire_wordline": sum(wl) / 2,
                         "extracted_ports": subckt_ports(ext.read_text(), CELL)})
        # replica and dummy cells too: the replica bitline times sense-enable / write completion, so it must
        # carry the same layout load as the extracted array (run.py --cell-ext substitutes all three)
        parts = [wrap.read_text()]
        for extra in (f"{CELL}_replica", f"{CELL}_dummy"):
            e = extract(wd, extra)
            parts.append(wrapper(e, wd / f"{extra}_ext.spice", extra, name_q=False).read_text())
        allw = wd / "openram_cells_ext.spice"
        allw.write_text("\n".join(parts))
        (run.dir / "openram_cells_ext.spice").write_text(allw.read_text())
        print(res)


if __name__ == "__main__":
    main()
