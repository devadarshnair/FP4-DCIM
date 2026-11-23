module tb34; reg [3:0] w,x; integer i,err; wire [7:0] pos,neg; wire [8:0] q; wire cin;
 function [7:0] m2f(input [2:0] c); begin case(c) 0:m2f=0;1:m2f=1;2:m2f=2;3:m2f=3;4:m2f=4;5:m2f=6;6:m2f=8;7:m2f=12; endcase end endfunction
 wire [3:0] Xp=m2f(x[2:0]); wire [5:0] X3=3*Xp; wire s1,s2,s3; wire [2:0] awo; wdec D(.w(w[2:0]),.s1(s1),.s2(s2),.s3(s3),.awo(awo));
 v3_split A(.s1(s1),.s2(s2),.s3(s3),.awo(awo),.sw(w[3]),.X(Xp),.X3(X3),.sx(x[3]),.pos(pos),.neg(neg));
 v4_onescomp B(.s1(s1),.s2(s2),.s3(s3),.awo(awo),.sw(w[3]),.X(Xp),.X3(X3),.sx(x[3]),.q(q),.cin(cin));
 wire [7:0] m = m2f(w[2:0])*m2f(x[2:0]); wire s = w[3]^x[3];
 wire signed [9:0] ref = s ? -$signed({2'b0,m}) : $signed({2'b0,m});
 wire signed [9:0] got4 = $signed(q) + cin;
 initial begin err=0; for(i=0;i<256;i=i+1) begin {w,x}=i; #1;
  if ($signed({2'b0,pos})-$signed({2'b0,neg}) !== ref) err=err+1;
  if (got4[8:0] !== ref[8:0]) err=err+1; end
  $display("v3/v4 errors=%0d",err); end endmodule
