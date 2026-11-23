// Compute-unit leaf multiplier: static weight decode + AND-OR select/shift.
//   |p*4| = (A_w * X') << a_w,  A_w in {0,1,2,3} selects {0, X', 2X', 3X'}, a_w in {0,1,2}.
// The weight decode depends only on the latched weight, so it toggles only on a weight-set switch.
// Output: sign-magnitude product and its ones'-complement encoding for the variant-B tree
// (tree adds s at the LSB). Magnitude <= 144 (8 b).
module leaf_mult (
  input  logic [3:0] w,     // latched E2M1 weight code
  input  logic [3:0] xp,    // 2|x| from the row driver
  input  logic [5:0] xp3,   // 3*2|x|
  input  logic       xs,    // activation sign (0 for zero)
  output logic [7:0] mag,   // |product * 4|
  output logic       s,     // product sign (0 when mag == 0)
  output logic [8:0] oc     // s ? ~{0,mag} : {0,mag}
);
  // static weight decode (one-hot)
  logic       hw;
  logic       sel1, sel2, sel3, sh0, sh1, sh2;
  assign hw   = w[2] | w[1];
  assign sel1 = ~hw &  w[0];          // A = 1
  assign sel2 =  hw & ~w[0];          // A = 2
  assign sel3 =  hw &  w[0];          // A = 3
  assign sh0  = ~w[2];                // e in {0,1} -> a = 0
  assign sh1  =  w[2] & ~w[1];        // e = 2      -> a = 1
  assign sh2  =  w[2] &  w[1];        // e = 3      -> a = 2
  // AND-OR select of A*X'
  logic [5:0] ax;
  assign ax = ({6{sel1}} & {2'b00, xp})
            | ({6{sel2}} & {1'b0, xp, 1'b0})
            | ({6{sel3}} & xp3);
  // AND-OR shift by a
  assign mag = ({8{sh0}} & {2'b00, ax})
             | ({8{sh1}} & {1'b0, ax, 1'b0})
             | ({8{sh2}} & {ax, 2'b00});
  assign s   = (w[3] ^ xs) & (|mag);
  assign oc  = s ? ~{1'b0, mag} : {1'b0, mag};
endmodule : leaf_mult
