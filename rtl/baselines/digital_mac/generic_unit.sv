// Row driver + generic leaf for one pair (formal/exhaustive check against the LUT spec).
module generic_unit (
  input  logic [3:0] w,
  input  logic [3:0] x,
  output logic [7:0] mag,
  output logic       s,
  output logic [8:0] oc
);
  logic [3:0] xp;
  logic       xs;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [5:0] xp3;
  /* verilator lint_on UNUSEDSIGNAL */
  row_driver   u_rd (.x(x), .xp(xp), .xp3(xp3), .xs(xs));
  generic_leaf u_gl (.w(w), .xp(xp), .xs(xs), .mag(mag), .s(s), .oc(oc));
endmodule : generic_unit
