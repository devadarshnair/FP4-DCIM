// e28 / T2.3 measurement wrapper: the per-row E2M1 -> INT5 activation conversion of the bit-serial INT5 baseline
// (rtl/baselines/bitserial_int5/bs_column_top.sv, g_xrow: xe/xm/xd). The three expressions below are copied verbatim
// from there (the driver asserts the text is still present in the baseline source); this module exists only so the
// conversion can be synthesised and power-measured in isolation. It is NOT an additional block of the INT5 macro: in
// the macro netlist the same logic is already counted inside the INT5 energy.
module e28_int5_conv #(parameter int ROWS = 32) (
  input  logic [ROWS*4-1:0] op_x,     // E2M1 codes, row r at [4r +: 4]
  output logic [ROWS*5-1:0] xd_all    // INT5 (2x value), row r at [5r +: 5], two's complement
);
  for (genvar r = 0; r < ROWS; r++) begin : g_xrow
    logic [1:0] xe;
    logic [3:0] xm;
    logic [4:0] xd;
    assign xe = op_x[4*r+1 +: 2];
    assign xm = {2'b00, (xe != 2'b00), op_x[4*r]} << ((xe == 2'b00) ? 2'd0 : xe - 2'd1);
    assign xd = op_x[4*r+3] ? (5'd0 - {1'b0, xm}) : {1'b0, xm};
    assign xd_all[5*r +: 5] = xd;
  end
endmodule : e28_int5_conv
