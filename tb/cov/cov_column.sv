// T1.4 functional-coverage monitor for one v2 column (rtl/src/column.sv), bound into every `column`
// instance by tb/cov/bind_column.sv. Coverage only: no assertion, no effect on the DUT.
// Bin names and meanings: docs/spec/coverage_plan.md. Build with Verilator --coverage-user.
// Bins prefixed ILL_ are illegal: the plan requires 0 hits.
module cov_column #(
  parameter int W     = 40,
  parameter int GUARD = 7
) (
  input logic                clk,
  input logic                rst_n,
  input logic                in_valid,
  input logic                acc_clear,
  input logic                nv_mode,
  input logic                nan_in,
  input logic signed [7:0]   e_w, e_x, e_w1, e_x1, max_e_w, max_e_x,
  input logic [3:0]          u_w0, u_w1, u_x0, u_x1,
  input logic signed [9:0]   d_raw, d1_raw,
  input logic                r_valid, r_clear, r_nv, r_nan,
  input logic [5:0]          r_d, r_d1,
  input logic [13:0]         blk_u,
  input logic [12:0]         s0_u, s1_u,
  input logic signed [W-1:0] term,
  input logic signed [W:0]   sum_wide,
  input logic signed [W-1:0] acc,
  input logic                nan_out,
  input logic                ovf
);
  localparam int BASE    = W - 14 - 1 - GUARD;
  localparam int BASE_NV = W - 21 - 1 - GUARD;
  localparam int DMAX    = W - 1;
  localparam longint ONE_BLK = longint'(4608) <<< BASE;  // |largest single MX term| at d = 0

  // history (coverage-private state)
  logic prev_v, last_nv, seen;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_v <= 1'b0; last_nv <= 1'b0; seen <= 1'b0;
    end else begin
      prev_v <= in_valid;
      if (in_valid) begin last_nv <= nv_mode; seen <= 1'b1; end
    end
  end

  logic signed [W:0] s_abs;
  assign s_abs = sum_wide[W] ? -sum_wide : sum_wide;
  logic signed [13:0] blk;
  assign blk = blk_u;
  logic signed [12:0] s0, s1;
  assign s0 = s0_u;
  assign s1 = s1_u;
  logic mx1, nv1, mx2, nv2;
  assign mx1 = in_valid && !nv_mode;
  assign nv1 = in_valid &&  nv_mode;
  assign mx2 = r_valid && !r_nv;
  assign nv2 = r_valid &&  r_nv;

  // ---- format and op sequencing ----
  F_MX:          cover property (@(posedge clk) disable iff (!rst_n) mx1);
  F_NV:          cover property (@(posedge clk) disable iff (!rst_n) nv1);
  F_MX_AFTER_NV: cover property (@(posedge clk) disable iff (!rst_n) mx1 && seen && last_nv);
  F_NV_AFTER_MX: cover property (@(posedge clk) disable iff (!rst_n) nv1 && seen && !last_nv);
  F_TOGGLE_B2B:  cover property (@(posedge clk) disable iff (!rst_n) in_valid && prev_v && (nv_mode != last_nv));
  F_B2B:         cover property (@(posedge clk) disable iff (!rst_n) in_valid && prev_v);
  F_RESUME:      cover property (@(posedge clk) disable iff (!rst_n) in_valid && !prev_v && seen);
  F_CLEAR:       cover property (@(posedge clk) disable iff (!rst_n) in_valid && acc_clear);
  F_CLEAR_B2B:   cover property (@(posedge clk) disable iff (!rst_n) in_valid && acc_clear && prev_v);
  F_ACCUM:       cover property (@(posedge clk) disable iff (!rst_n) in_valid && !acc_clear);
  F_ACCUM_NV:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && !acc_clear);

  // ---- MX scale corners (UE8M0 unbiased: 0x00 = -127, 0xFE = +127) ----
  E_W_MIN: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_w == -8'sd127);
  E_W_MAX: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_w ==  8'sd127);
  E_X_MIN: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_x == -8'sd127);
  E_X_MAX: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_x ==  8'sd127);
  E_W_NEG: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_w < 0);
  E_W_POS: cover property (@(posedge clk) disable iff (!rst_n) mx1 && e_w > 0);

  // ---- NV scale-significand corners (E4M3 significand u in [0,15]; u < 8: subnormal scale) ----
  U_ZERO:   cover property (@(posedge clk) disable iff (!rst_n) nv1 && (u_w0 == 0 || u_x0 == 0 || u_w1 == 0 || u_x1 == 0));
  U_SUB:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && ((u_w0 != 0 && u_w0 < 8) || (u_x0 != 0 && u_x0 < 8)));
  U_MAX:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && ((u_w0 == 15 && u_x0 == 15) || (u_w1 == 15 && u_x1 == 15)));
  U_HALVES_DIFF: cover property (@(posedge clk) disable iff (!rst_n) nv1 && (u_w0 != u_w1 || u_x0 != u_x1));
  K_NEG:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && (e_w < 0 || e_x < 0));
  K_POS:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && (e_w > 0 || e_x > 0));
  K_MAX:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && (e_w == 5 || e_x == 5 || e_w1 == 5 || e_x1 == 5));
  K_MIN:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && (e_w == -9 || e_x == -9 || e_w1 == -9 || e_x1 == -9));

  // ---- anchor / alignment (MX: d of the block; NV: d0, d1 of the halves) ----
  A_D0:        cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw == 0);
  A_LEFT:      cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw > 0 && d_raw < BASE);
  A_BASE:      cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw == BASE);
  A_RIGHT:     cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw > BASE && d_raw < DMAX);
  A_SAT_EQ:    cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw == DMAX);
  A_SAT_GT:    cover property (@(posedge clk) disable iff (!rst_n) mx1 && d_raw > DMAX);
  A_NV_D0:     cover property (@(posedge clk) disable iff (!rst_n) nv1 && d_raw == 0 && d1_raw == 0);
  A_NV_SPLIT:  cover property (@(posedge clk) disable iff (!rst_n) nv1 && d_raw != d1_raw);
  A_NV_RIGHT:  cover property (@(posedge clk) disable iff (!rst_n) nv1 && (d_raw > BASE_NV || d1_raw > BASE_NV));
  A_NV_SAT:    cover property (@(posedge clk) disable iff (!rst_n) nv1 && (d_raw >= DMAX || d1_raw >= DMAX));

  // ---- block / half sums (stage 2) ----
  S_MX_MAX:  cover property (@(posedge clk) disable iff (!rst_n) mx2 && blk ==  14'sd4608);
  S_MX_MIN:  cover property (@(posedge clk) disable iff (!rst_n) mx2 && blk == -14'sd4608);
  S_MX_ZERO: cover property (@(posedge clk) disable iff (!rst_n) mx2 && blk == 0);
  S_MX_NEG:  cover property (@(posedge clk) disable iff (!rst_n) mx2 && blk < 0);
  S_NV_MAX:  cover property (@(posedge clk) disable iff (!rst_n) nv2 && (s0 ==  13'sd2304 || s1 ==  13'sd2304));
  S_NV_MIN:  cover property (@(posedge clk) disable iff (!rst_n) nv2 && (s0 == -13'sd2304 || s1 == -13'sd2304));
  S_NV_NEG:  cover property (@(posedge clk) disable iff (!rst_n) nv2 && (s0 < 0 || s1 < 0));

  // ---- truncation of right-shifted terms (floor toward -inf) ----
  T_FLOOR_M1: cover property (@(posedge clk) disable iff (!rst_n) mx2 && r_d > BASE && blk < 0 && term == -1);
  T_FLOOR_0:  cover property (@(posedge clk) disable iff (!rst_n) mx2 && r_d > BASE && blk > 0 && term == 0);

  // ---- NaN ----
  N_IN:      cover property (@(posedge clk) disable iff (!rst_n) in_valid && nan_in);
  N_IN_NV:   cover property (@(posedge clk) disable iff (!rst_n) nv1 && nan_in);
  N_STICKY:  cover property (@(posedge clk) disable iff (!rst_n) r_valid && !r_clear && nan_out && !r_nan);
  N_CLEARED: cover property (@(posedge clk) disable iff (!rst_n) r_valid && r_clear && nan_out && !r_nan);

  // ---- accumulator window ----
  W_GUARD_USED: cover property (@(posedge clk) disable iff (!rst_n) r_valid && s_abs > (W+1)'(ONE_BLK));
  W_TOP2:       cover property (@(posedge clk) disable iff (!rst_n) r_valid && s_abs >= (W+1)'(longint'(1) <<< (W-3)));
  // (at the sampling edge `acc` still holds the value before this op, `sum_wide` the value after it)
  W_SIGN_FLIP:  cover property (@(posedge clk) disable iff (!rst_n) r_valid && !r_clear
                                && (acc[W-1] != sum_wide[W-1]) && acc != 0 && sum_wide != 0);
  W_CANCEL0:    cover property (@(posedge clk) disable iff (!rst_n) r_valid && !r_clear && acc != 0 && sum_wide == 0);

  // ---- illegal (must stay at 0 hits) ----
  ILL_OVF:  cover property (@(posedge clk) disable iff (!rst_n) ovf);
  ILL_DNEG: cover property (@(posedge clk) disable iff (!rst_n) in_valid && (d_raw < 0 || (nv_mode && d1_raw < 0)));
endmodule : cov_column
