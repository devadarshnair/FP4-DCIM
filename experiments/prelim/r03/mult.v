module v0_full(input [3:0] w, input [3:0] x, output s, output reg [7:0] mag);
 assign s = w[3]^x[3];
 always @* case({w[2:0],x[2:0]})
  6'd0: mag = 8'd0;
  6'd1: mag = 8'd0;
  6'd2: mag = 8'd0;
  6'd3: mag = 8'd0;
  6'd4: mag = 8'd0;
  6'd5: mag = 8'd0;
  6'd6: mag = 8'd0;
  6'd7: mag = 8'd0;
  6'd8: mag = 8'd0;
  6'd9: mag = 8'd1;
  6'd10: mag = 8'd2;
  6'd11: mag = 8'd3;
  6'd12: mag = 8'd4;
  6'd13: mag = 8'd6;
  6'd14: mag = 8'd8;
  6'd15: mag = 8'd12;
  6'd16: mag = 8'd0;
  6'd17: mag = 8'd2;
  6'd18: mag = 8'd4;
  6'd19: mag = 8'd6;
  6'd20: mag = 8'd8;
  6'd21: mag = 8'd12;
  6'd22: mag = 8'd16;
  6'd23: mag = 8'd24;
  6'd24: mag = 8'd0;
  6'd25: mag = 8'd3;
  6'd26: mag = 8'd6;
  6'd27: mag = 8'd9;
  6'd28: mag = 8'd12;
  6'd29: mag = 8'd18;
  6'd30: mag = 8'd24;
  6'd31: mag = 8'd36;
  6'd32: mag = 8'd0;
  6'd33: mag = 8'd4;
  6'd34: mag = 8'd8;
  6'd35: mag = 8'd12;
  6'd36: mag = 8'd16;
  6'd37: mag = 8'd24;
  6'd38: mag = 8'd32;
  6'd39: mag = 8'd48;
  6'd40: mag = 8'd0;
  6'd41: mag = 8'd6;
  6'd42: mag = 8'd12;
  6'd43: mag = 8'd18;
  6'd44: mag = 8'd24;
  6'd45: mag = 8'd36;
  6'd46: mag = 8'd48;
  6'd47: mag = 8'd72;
  6'd48: mag = 8'd0;
  6'd49: mag = 8'd8;
  6'd50: mag = 8'd16;
  6'd51: mag = 8'd24;
  6'd52: mag = 8'd32;
  6'd53: mag = 8'd48;
  6'd54: mag = 8'd64;
  6'd55: mag = 8'd96;
  6'd56: mag = 8'd0;
  6'd57: mag = 8'd12;
  6'd58: mag = 8'd24;
  6'd59: mag = 8'd36;
  6'd60: mag = 8'd48;
  6'd61: mag = 8'd72;
  6'd62: mag = 8'd96;
  6'd63: mag = 8'd144;
  default: mag=8'd0; endcase endmodule

module v0_struct(input [3:0] w, input [3:0] x, output s, output [7:0] mag);
 wire hw = w[2]|w[1], hx = x[2]|x[1];
 wire [1:0] Aw = {hw,w[0]}, Ax = {hx,x[0]};
 wire [1:0] aw = (w[2:1]==2'd3)?2'd2:(w[2:1]==2'd2)?2'd1:2'd0;
 wire [1:0] ax = (x[2:1]==2'd3)?2'd2:(x[2:1]==2'd2)?2'd1:2'd0;
 wire [3:0] P = Aw*Ax;
 assign mag = {4'b0,P} << ({1'b0,aw}+{1'b0,ax});
 assign s = w[3]^x[3];
endmodule

module v1_rowdec(input [3:0] w, input [1:0] Ax, input [2:0] axo, input sx, output s, output [7:0] mag);
 wire hw = w[2]|w[1];
 wire [1:0] Aw = {hw,w[0]};
 wire [2:0] awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
 wire [3:0] P = Aw*Ax;
 wire [4:0] so = {awo[2]&axo[2], awo[2]&axo[1]|awo[1]&axo[2], awo[2]&axo[0]|awo[1]&axo[1]|awo[0]&axo[2], awo[1]&axo[0]|awo[0]&axo[1], awo[0]&axo[0]};
 assign mag = ({8{so[0]}}&{4'b0,P}) | ({8{so[1]}}&{3'b0,P,1'b0}) | ({8{so[2]}}&{2'b0,P,2'b0}) | ({8{so[3]}}&{1'b0,P,3'b0}) | ({8{so[4]}}&{P,4'b0});
 assign s = w[3]^sx;
endmodule

module v2_bcastX(input [3:0] w, input [3:0] X, input [5:0] X3, input sx, output s, output [7:0] mag);
 wire hw = w[2]|w[1];
 wire s1 = ~hw & w[0], s2 = hw & ~w[0], s3 = hw & w[0];
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [2:0] awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
 assign mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 assign s = w[3]^sx;
endmodule

module v2_static(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output s, output [7:0] mag);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 assign mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 assign s = sw^sx;
endmodule

module wdec(input [2:0] w, output s1, output s2, output s3, output [2:0] awo);
 wire hw = w[2]|w[1];
 assign s1 = ~hw & w[0]; assign s2 = hw & ~w[0]; assign s3 = hw & w[0];
 assign awo = {w[2]&w[1], w[2]&~w[1], ~w[2]};
endmodule

module v3_split(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output [7:0] pos, output [7:0] neg);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [7:0] mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 wire s = sw^sx;
 assign pos = mag & {8{~s}}; assign neg = mag & {8{s}};
endmodule

module v4_onescomp(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output [8:0] q, output cin);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [7:0] mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 wire s = sw^sx;
 assign q = {1'b0,mag} ^ {9{s}}; assign cin = s;
endmodule

module v5_bitserial(input [4:0] wq, input xb, output [4:0] pp);
 assign pp = wq & {5{xb}};
endmodule
module v6_flat(input [6:0] sel, input sw, input [3:0] X, input [5:0] X3, input sx, output s, output [7:0] mag);
 // sel one-hot static: values 1,2,4,8 (X'<<k) and 3,6,12 (3X'<<k)
 assign mag = ({8{sel[0]}}&{4'b0,X}) | ({8{sel[1]}}&{3'b0,X,1'b0}) | ({8{sel[2]}}&{2'b0,X,2'b0}) | ({8{sel[3]}}&{1'b0,X,3'b0})
            | ({8{sel[4]}}&{2'b0,X3}) | ({8{sel[5]}}&{1'b0,X3,1'b0}) | ({8{sel[6]}}&{X3,2'b0});
 assign s = sw^sx;
endmodule
// V7: two's complement 9b output directly (for TC tree), weight static selects
module v7_tc(input s1, input s2, input s3, input [2:0] awo, input sw, input [3:0] X, input [5:0] X3, input sx, output [8:0] q);
 wire [5:0] Y = ({6{s1}}&{2'b0,X}) | ({6{s2}}&{1'b0,X,1'b0}) | ({6{s3}}&X3);
 wire [7:0] mag = ({8{awo[0]}}&{2'b0,Y}) | ({8{awo[1]}}&{1'b0,Y,1'b0}) | ({8{awo[2]}}&{Y,2'b0});
 wire s = sw^sx;
 assign q = s ? (9'd0 - {1'b0,mag}) : {1'b0,mag};
endmodule
// V9: row-level signed pre-computation. Broadcast P1=+X'' N1=-X'' (5b 2's comp), P3=+3X'' N3=-3X'' (7b), X''=(-1)^sx X'.
// Weight sign folded into STATIC one-hot select (sp1,sn1,sp2,sn2,sp3,sn3) -> per-weight logic is a pure AO-mux + shift.
module v9_tcbcast(input sp1, input sn1, input sp2, input sn2, input sp3, input sn3, input [2:0] awo,
                  input [4:0] P1, input [4:0] N1, input [6:0] P3, input [6:0] N3, output [8:0] q);
 wire [6:0] Y = ({7{sp1}}&{{2{P1[4]}},P1}) | ({7{sn1}}&{{2{N1[4]}},N1}) | ({7{sp2}}&{P1[4],P1,1'b0}) | ({7{sn2}}&{N1[4],N1,1'b0})
              | ({7{sp3}}&P3) | ({7{sn3}}&N3);
 assign q = ({9{awo[0]}}&{{2{Y[6]}},Y}) | ({9{awo[1]}}&{Y[6],Y,1'b0}) | ({9{awo[2]}}&{Y,2'b0});
endmodule
