module tb9; reg [3:0] w,x; integer i,err; wire [8:0] q;
 function [7:0] m2f(input [2:0] c); begin case(c) 0:m2f=0;1:m2f=1;2:m2f=2;3:m2f=3;4:m2f=4;5:m2f=6;6:m2f=8;7:m2f=12; endcase end endfunction
 wire signed [5:0] Xs = x[3] ? -$signed({2'b0,m2f(x[2:0])}) : $signed({2'b0,m2f(x[2:0])});
 wire signed [7:0] X3s = 3*Xs;
 wire [4:0] P1=Xs[4:0], N1=(-Xs); wire [6:0] P3=X3s[6:0], N3=(-X3s);
 wire hw=w[2]|w[1]; wire A1=~hw&w[0], A2=hw&~w[0], A3=hw&w[0];
 wire [2:0] awo={w[2]&w[1],w[2]&~w[1],~w[2]};
 v9_tcbcast U(.sp1(A1&~w[3]),.sn1(A1&w[3]),.sp2(A2&~w[3]),.sn2(A2&w[3]),.sp3(A3&~w[3]),.sn3(A3&w[3]),.awo(awo),.P1(P1),.N1(N1),.P3(P3),.N3(N3),.q(q));
 wire signed [9:0] ref = (w[3]^x[3]) ? -$signed({2'b0,m2f(w[2:0])*m2f(x[2:0])}) : $signed({2'b0,m2f(w[2:0])*m2f(x[2:0])});
 initial begin err=0; for(i=0;i<256;i=i+1) begin {w,x}=i; #1; if($signed(q)!==ref[8:0] && !($signed(q)==0 && ref==0)) begin err=err+1; $display("E %b %b q=%d ref=%d",w,x,$signed(q),ref); end end
 $display("v9 errors=%0d",err); end endmodule
