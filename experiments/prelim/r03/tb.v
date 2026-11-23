module tb;
 reg [3:0] w,x; wire s0,s1,s2,s3,s4; wire [7:0] m0,m1,m2,m3,m4,m6; wire s6;
 function [7:0] m2f(input [2:0] c); begin
   case(c) 0:m2f=0; 1:m2f=1; 2:m2f=2; 3:m2f=3; 4:m2f=4; 5:m2f=6; 6:m2f=8; 7:m2f=12; endcase end endfunction
 wire [3:0] Xp = m2f(x[2:0]); wire [5:0] X3 = 3*Xp;
 wire hx = x[2]|x[1]; wire [1:0] Ax={hx,x[0]}; wire [2:0] axo={x[2]&x[1],x[2]&~x[1],~x[2]};
 wire ws1,ws2,ws3; wire [2:0] awo; wdec D(.w(w[2:0]),.s1(ws1),.s2(ws2),.s3(ws3),.awo(awo));
 reg [6:0] sel; always @* case(w[2:0]) 1:sel=7'b0000001; 2:sel=7'b0000010; 3:sel=7'b0010000; 4:sel=7'b0000100; 5:sel=7'b0100000; 6:sel=7'b0001000; 7:sel=7'b1000000; default: sel=0; endcase
 v0_full U0(.w(w),.x(x),.s(s0),.mag(m0));
 v0_struct U1(.w(w),.x(x),.s(s1),.mag(m1));
 v1_rowdec U2(.w(w),.Ax(Ax),.axo(axo),.sx(x[3]),.s(s2),.mag(m2));
 v2_bcastX U3(.w(w),.X(Xp),.X3(X3),.sx(x[3]),.s(s3),.mag(m3));
 v2_static U4(.s1(ws1),.s2(ws2),.s3(ws3),.awo(awo),.sw(w[3]),.X(Xp),.X3(X3),.sx(x[3]),.s(s4),.mag(m4));
 v6_flat U6(.sel(sel),.sw(w[3]),.X(Xp),.X3(X3),.sx(x[3]),.s(s6),.mag(m6));
 integer i,err; initial begin err=0;
  for(i=0;i<256;i=i+1) begin {w,x}=i; #1;
   if(m0!==m2f(w[2:0])*m2f(x[2:0]) || m1!==m0 || m2!==m0 || m3!==m0 || m4!==m0 || m6!==m0) begin err=err+1; $display("ERR w=%b x=%b %d %d %d %d %d %d",w,x,m0,m1,m2,m3,m4,m6); end
  end
  $display("exhaustive check done, errors=%0d",err); $finish; end
endmodule
