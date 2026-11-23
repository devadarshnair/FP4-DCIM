"""T1.7 (13#B1c): does Icarus -gspecify + $sdf_annotate filter short pulses (inertial) or pass them (transport)?

Two chained sky130 and2_1 cells (B tied high), SDF IOPATH delay D=0.5 ns each (same SDF style/flags as the e13 SDF
flow: cell sources, -gspecify, -s sdf_annotate, timescale 1ns/1ps). Stimulus: pulses of width W on the input.
Output transitions counted per W. Inertial: pulses < D vanish at gate 1 (0 output edges). Transport: 2 edges.
Run: python experiments/t1_7_icarus_pulse/run.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from cimfp4.utils.provenance import REPO_ROOT
from cimfp4.utils.runlog import start_run

CELLS = Path.home() / ".ciel/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
D = 0.5
WIDTHS = [0.05, 0.1, 0.2, 0.4, 0.49, 0.6, 1.0, 2.0]

NET = """`timescale 1ns/1ps
module two_gate(input a, output y);
  wire m, one;
  assign one = 1'b1;
  sky130_fd_sc_hd__and2_1 u1(.A(a), .B(one), .X(m));
  sky130_fd_sc_hd__and2_1 u2(.A(m), .B(one), .X(y));
endmodule
"""
SDF = f"""(DELAYFILE (SDFVERSION "3.0") (DESIGN "two_gate") (DIVIDER .) (TIMESCALE 1ns)
 (CELL (CELLTYPE "two_gate") (INSTANCE) (DELAY (ABSOLUTE)))
 (CELL (CELLTYPE "sky130_fd_sc_hd__and2_1") (INSTANCE u1) (DELAY (ABSOLUTE (IOPATH A X ({D}:{D}:{D}) ({D}:{D}:{D})))))
 (CELL (CELLTYPE "sky130_fd_sc_hd__and2_1") (INSTANCE u2) (DELAY (ABSOLUTE (IOPATH A X ({D}:{D}:{D}) ({D}:{D}:{D})))))
)
"""


def tb(sdf: str | None) -> str:
    pulses = ""
    t = 10.0
    for w in WIDTHS:
        pulses += f"  #{t - 0:.3f}; a = 1; #{w}; a = 0;\n"
        t = 10.0  # relative spacing between pulses (delays are relative: #)
    ann = f'  initial $sdf_annotate("{sdf}", dut);\n' if sdf else ""
    return f"""`timescale 1ns/1ps
module tb;
  reg a = 0; wire y, m;
  two_gate dut(.a(a), .y(y));
{ann}  assign m = dut.m;
  integer ny = 0, nm = 0;
  always @(y) if ($time > 0) begin ny = ny + 1; $display("Y %0t %b", $time, y); end
  always @(m) if ($time > 0) begin nm = nm + 1; end
  initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, tb);
    #5;
{pulses}    #10; $display("TOTAL_Y_EDGES %0d", ny); $finish;
  end
endmodule
"""


def sim(wd: Path, tag: str, sdf: Path | None) -> list[tuple[float, int]]:
    """Sim one pulse width at a time (separate runs) so counts are attributable."""
    return []


def main() -> None:
    import cimfp4.utils.runlog as rl  # noqa: F401

    with start_run("t1_7_icarus_pulse", {"delay_ns": D, "widths_ns": WIDTHS, "flags": "-gspecify -s tb -s sdf (as e13)"}) as run:
        wd = run.dir
        (wd / "two_gate.v").write_text(NET)
        (wd / "two_gate.sdf").write_text(SDF)
        rows = []
        for mode in ("sdf", "nosdf"):
            for w in WIDTHS:
                global WIDTHS_ONE
                src = tb(str(wd / "two_gate.sdf") if mode == "sdf" else None)
                # single pulse per sim
                src = re.sub(r"(  #10\.000; a = 1;.*\n)+", f"  #10; a = 1; #{w}; a = 0;\n", src)
                f = wd / f"tb_{mode}_{w}.v"
                f.write_text(src)
                exe = wd / f"sim_{mode}_{w}.vvp"
                cmd = ["iverilog", "-g2012", "-gspecify", "-o", str(exe), "-s", "tb",
                       str(CELLS / "primitives.v"), str(CELLS / "sky130_fd_sc_hd.v"), str(wd / "two_gate.v"), str(f)]
                subprocess.run(cmd, check=True, cwd=wd, capture_output=True, text=True)
                out = subprocess.run(["vvp", str(exe)], check=True, cwd=wd, capture_output=True, text=True).stdout
                (wd / f"out_{mode}_{w}.txt").write_text(out)
                edges = [l for l in out.splitlines() if l.startswith("Y ")]
                rows.append({"mode": mode, "pulse_ns": w, "y_edges": len(edges), "y_events": ";".join(edges)})
                print(mode, w, len(edges), edges)
        run.save_csv("pulse_sweep.csv", rows)
        sdfrows = [r for r in rows if r["mode"] == "sdf"]
        sub = [r for r in sdfrows if r["pulse_ns"] < D]
        passed_short = [r["pulse_ns"] for r in sub if r["y_edges"] > 0]
        verdict = "TRANSPORT" if len(passed_short) == len(sub) else ("INERTIAL" if not passed_short else "MIXED")
        run.log_metrics({"verdict": verdict, "short_pulses_passed": passed_short,
                         "long_pulses_ok": all(r["y_edges"] == 2 for r in sdfrows if r["pulse_ns"] >= D + 0.05)})
        print("VERDICT", verdict)


if __name__ == "__main__":
    main()
