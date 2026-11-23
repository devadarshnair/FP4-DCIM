// Generic FP4 MAC leaf (digital MAC baseline and the macro_top LEAF=GENERIC variant): decode both operands
// to INT5 (2v in [-12,12]) and multiply with a plain 5b x 5b two's-complement multiplier - i.e. no
// FP4-specific structure. Same output encoding as leaf_mult (sign-magnitude + ones' complement).
module generic_leaf (
  input  logic [3:0] w,     // E2M1 weight code
  input  logic [3:0] xp,    // 2|x| from the row driver
  input  logic       xs,    // activation sign
  output logic [7:0] mag,
  output logic       s,
  output logic [8:0] oc
);
  logic [1:0]        we;
  logic [3:0]        wm;
  logic signed [4:0] wi, xi;
  logic signed [9:0] p;
  assign we  = w[2:1];
  assign wm  = {2'b00, (we != 2'b00), w[0]} << ((we == 2'b00) ? 2'd0 : we - 2'd1);
  assign wi  = w[3] ? -$signed({1'b0, wm}) : $signed({1'b0, wm});
  assign xi  = xs   ? -$signed({1'b0, xp}) : $signed({1'b0, xp});
  assign p   = wi * xi;                       // generic signed multiplier
  assign s   = p[9];
  assign mag = s ? 8'(-p) : 8'(p);
  assign oc  = s ? ~{1'b0, mag} : {1'b0, mag};
endmodule : generic_leaf
