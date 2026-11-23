// Row driver + leaf multiplier for one (weight, activation) pair - the unit checked exhaustively
// and proven equivalent to the generated LUT spec (tb/formal/leaf_dec_equiv.ys).
module leaf_unit (
  input  logic [3:0] w,
  input  logic [3:0] x,
  output logic [7:0] mag,
  output logic       s,
  output logic [8:0] oc
);
  logic [3:0] xp;
  logic [5:0] xp3;
  logic       xs;
  row_driver u_rd (.x(x), .xp(xp), .xp3(xp3), .xs(xs));
  leaf_mult  u_lm (.w(w), .xp(xp), .xp3(xp3), .xs(xs), .mag(mag), .s(s), .oc(oc));
endmodule : leaf_unit
