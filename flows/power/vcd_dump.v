// Extra simulation root: dumps every net of the DUT to a VCD (zero-delay GLS -> OpenSTA read_vcd).
// Compile with -DVCD_FILE=\"path\" -DVCD_TOP=<top module>, and add `-s vcd_dump` to iverilog.
module vcd_dump;
  initial begin
    $dumpfile(`VCD_FILE);
    $dumpvars(0, `VCD_TOP);
  end
endmodule
