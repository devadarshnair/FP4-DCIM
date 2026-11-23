// T1.4 functional-coverage monitor for one bit-serial INT5 column (rtl/baselines/bitserial_int5/bs_column.sv).
module cov_bs_column #(
  parameter int W     = 40,
  parameter int GUARD = 7
) (
  input logic                clk,
  input logic                rst_n,
  input logic                accept,
  input logic                busy,
  input logic [2:0]          bit_i,
  input logic                op_clear,
  input logic                nan_in,
  input logic                a_valid, a_last,
  input logic signed [9:0]   d_raw,
  input logic                b_valid, b_clear, b_nan,
  input logic [5:0]          b_d,
  input logic signed [13:0]  blk,
  input logic                nan_out,
  input logic                ovf
);
  localparam int BASE = W - 14 - 1 - GUARD;
  localparam int DMAX = W - 1;
  logic blk_stage;
  assign blk_stage = a_valid && a_last;  // the edge that registers blk / d
  BS_ACCEPT_IDLE: cover property (@(posedge clk) disable iff (!rst_n) accept && !busy);
  BS_B2B:         cover property (@(posedge clk) disable iff (!rst_n) accept && busy && bit_i == 3'd4);
  BS_CLEAR:       cover property (@(posedge clk) disable iff (!rst_n) accept && op_clear);
  BS_NAN:         cover property (@(posedge clk) disable iff (!rst_n) accept && nan_in);
  BS_N_STICKY:    cover property (@(posedge clk) disable iff (!rst_n) b_valid && !b_clear && nan_out && !b_nan);
  BS_D0:          cover property (@(posedge clk) disable iff (!rst_n) blk_stage && d_raw == 0);
  BS_D_RIGHT:     cover property (@(posedge clk) disable iff (!rst_n) blk_stage && d_raw > BASE && d_raw < DMAX);
  BS_D_SAT:       cover property (@(posedge clk) disable iff (!rst_n) blk_stage && d_raw >= DMAX);
  BS_BLK_MAX:     cover property (@(posedge clk) disable iff (!rst_n) b_valid && blk ==  14'sd4608);
  BS_BLK_MIN:     cover property (@(posedge clk) disable iff (!rst_n) b_valid && blk == -14'sd4608);
  BS_BLK_NEG:     cover property (@(posedge clk) disable iff (!rst_n) b_valid && blk < 0);
  ILL_BS_OVF:     cover property (@(posedge clk) disable iff (!rst_n) ovf);
endmodule : cov_bs_column
