# Generate Verilog variants of the per-weight E2M1 x E2M1 multiplier
def mag2(c):  # |v|*2 for 3-bit magnitude code
    e=(c>>1)&3; m=c&1
    return m if e==0 else (2+m)<<(e-1)
lines=[]
# V0: raw codes both sides, full truth table, sign + 8b magnitude
lines.append("module v0_full(input [3:0] w, input [3:0] x, output s, output reg [7:0] mag);")
lines.append(" assign s = w[3]^x[3];")
lines.append(" always @* case({w[2:0],x[2:0]})")
for a in range(8):
  for b in range(8):
    lines.append(f"  6'd{a*8+b}: mag = 8'd{mag2(a)*mag2(b)};")
lines.append("  default: mag=8'd0; endcase endmodule\n")
# V0s: structured (A,a) form: 2x2 multiply + shift by a_w+a_x
lines.append("""module v0_struct(input [3:0] w, input [3:0] x, output s, output [7:0] mag);
 wire hw = w[2]|w[1], hx = x[2]|x[1];
 wire [1:0] Aw = {hw,w[0]}, Ax = {hx,x[0]};
 wire [1:0] aw = (w[2:1]==2'd3)?2'd2:(w[2:1]==2'd2)?2'd1:2'd0;
 wire [1:0] ax = (x[2:1]==2'd3)?2'd2:(x[2:1]==2'd2)?2'd1:2'd0;
 wire [3:0] P = Aw*Ax;
 assign mag = {4'b0,P} << (aw+ax);
 assign s = w[3]^x[3];
endmodule
""")
# V1: activation pre-decoded per row (broadcast A_x[1:0] + onehot ax[2:0]); weight raw
lines.append("""module v1_rowdec(input [3:0] w, input [1:0] Ax, input [2:0] axo, input sx, output s, output [7:0] mag);
 wire hw = w[2]|w[1];
 wire [1:0] Aw = {hw,w[0]};
 wire [2:0] awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
 wire [3:0] P = Aw*Ax;
 wire [4:0] so = {awo[2]&axo[2], awo[2]&axo[1]|awo[1]&axo[2], awo[2]&axo[0]|awo[1]&axo[1]|awo[0]&axo[2], awo[1]&axo[0]|awo[0]&axo[1], awo[0]&axo[0]};
 assign mag = ({8{so[0]}}&{4'b0,P}) | ({8{so[1]}}&{3'b0,P,1'b0}) | ({8{so[2]}}&{2'b0,P,2'b0}) | ({8{so[3]}}&{1'b0,P,3'b0}) | ({8{so[4]}}&{P,4'b0});
 assign s = w[3]^sx;
endmodule
""")
# V2: activation broadcast as X'=Ax<<ax (4b) and 3X' (6b); weight raw
lines.append("""module v2_bcastX(input [3:0] w, input [3:0] X, input [5:0] X3, input sx, output s, output [7:0] mag);
 wire hw = w[2]|w[1];
 wire s1 = ~hw & w[0], s2 = hw & ~w[0], s3 = hw & w[0];
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [2:0] awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
 assign mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 assign s = w[3]^sx;
endmodule
""")
# V2s: same, weight pre-decoded static selects provided (s1,s2,s3, awo) -> dynamic logic only
lines.append("""module v2_static(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output s, output [7:0] mag);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 assign mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 assign s = sw^sx;
endmodule
""")
# weight static decode alone
lines.append("""module wdec(input [2:0] w, output s1, output s2, output s3, output [2:0] awo);
 wire hw = w[2]|w[1];
 assign s1 = ~hw & w[0]; assign s2 = hw & ~w[0]; assign s3 = hw & w[0];
 assign awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
endmodule
""")
# V3: sign-split outputs from V2_static (pos/neg 8b each)
lines.append("""module v3_split(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output [7:0] pos, output [7:0] neg);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [7:0] mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 wire s = sw^sx;
 assign pos = mag & {8{~s}}; assign neg = mag & {8{s}};
endmodule
""")
# V4: 2's complement 9b output (conditional invert, +1 deferred to tree as carry-in bit 's')
lines.append("""module v4_onescomp(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output [8:0] q, output cin);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [7:0] mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 wire s = sw^sx;
 assign q = {1'b0,mag} ^ {9{s}}; assign cin = s;
endmodule
""")
# V5: bit-serial TSMC-style: per weight 4 (or 5) NOR/AND gates for one activation bit
lines.append("""module v5_bitserial(input [4:0] wq, input xb, output [4:0] pp);
 assign pp = wq & {5{xb}};
endmodule
""")
open("mult.v","w").write("\n".join(lines))
