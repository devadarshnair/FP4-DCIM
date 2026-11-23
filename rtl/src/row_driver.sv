// Row driver (docs/spec/numerics.md): decodes one E2M1 activation code and broadcasts
// X' = 2|x| (4 b), 3X' (6 b) and the sign along the row (11 wires) to all compute units.
// 2|x| = A << a with A = {e != 0, m} in 0..3 and a = max(e,1) - 1 in 0..2 (hidden-bit form).
module row_driver (
  input  logic [3:0] x,    // E2M1 code {s, e1, e0, m}
  output logic [3:0] xp,   // 2|x|  in {0,1,2,3,4,6,8,12}
  output logic [5:0] xp3,  // 3*2|x| in {0,...,36}
  output logic       xs    // sign, forced 0 for x = +-0
);
  logic [1:0] e;
  logic [1:0] a_val;
  logic [3:0] a_mant;
  assign e      = x[2:1];
  assign a_mant = {2'b00, (e != 2'b00), x[0]};
  assign a_val  = (e == 2'b00) ? 2'd0 : e - 2'd1;
  assign xp     = a_mant << a_val;
  assign xp3    = {2'b00, xp} + {1'b0, xp, 1'b0};
  assign xs     = x[3] & (xp != 4'd0);
endmodule : row_driver
