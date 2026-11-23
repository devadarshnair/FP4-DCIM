// Baseline (lossy, MIDAS-inspired): bit-serial sign-magnitude MXINT4-SM DCIM column.
// NOT exact E2M1 and NOT the INT5 control (exact E2M1 integerization needs signed INT5; see
// docs/spec/d2_int4_contract.md).  Operands are sign + 3-bit magnitude m in [0,7] (shared UE8M0 scale, so the
// block exponent has no -2 offset: callers pass e_w/e_x/max_e_* as plain scale exponents).
// Weights stationary: w_bus[4r+3] = sign, w_bus[4r +: 3] = magnitude (same 4 bits/row as E2M1 storage, so the
// same wset store is reused).  Activation magnitude bits arrive LSB-first from shared row shift registers
// (xbit_bus), 3 magnitude cycles per block; the activation sign xsgn_bus is static for the block.
// Per cycle: the one-bit product x_i & w_j is routed by sgn_w ^ sgn_x to a positive or a negative 32-input
// popcount tree (generated w_in=1 tree, same netlist as bs_column) -> 9 trees (3 weight bits x 2 polarities)
// -> carry-save register -> CPAs -> combiner S_i = sum_j (pc+_j - pc-_j) << j -> shift-add block accumulator
// (all cycles positive weight; sign is per-LCA, no MSB negation) -> the same alignment + W-bit window as
// column / bs_column. One block per 3 cycles; interface differs from bs_column only by xsgn_bus (sign-magnitude
// activations; INT5 carries the sign inside the two's-complement bit stream) and the cycle count (3 vs 5).
`include "cim_defs.vh"
module bs4_column #(
  parameter int W     = `CIM_WINDOW_BITS,
  parameter int GUARD = 7
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                op_valid,       // new block (accepted when op_ready)
  output logic                op_ready,       // !busy || last bit: back-to-back issue, 3 cycles/op
  output logic                idle,           // no block in flight (for set-switch control)
  input  logic                op_clear,
  input  logic [`CIM_ROWS-1:0]   xbit_bus,    // current activation magnitude bit per row (shared row shift register)
  input  logic [`CIM_ROWS-1:0]   xsgn_bus,    // activation sign per row (static during a block)
  input  logic [`CIM_ROWS*4-1:0] w_bus,       // latched INT4-SM weights: {sign, mag[2:0]} per row
  input  logic signed [7:0]   e_w, e_x, max_e_w, max_e_x,
  input  logic                nan_in,
  output logic signed [W-1:0] acc,
  output logic                acc_valid,
  output logic                nan_out,
  output logic                ovf
);
  localparam int ROWS = `CIM_ROWS;
  localparam int SB = `CIM_MX_SUM_S_BITS;
  localparam int BASE = W - SB - 1 - GUARD;
  localparam int DMAX = W - 1;

  // weights are stored sign-magnitude: no decode; the magnitude bits and sign feed the array directly

  // activation shift register (row-driver equivalent) and sequencer
  logic [2:0] bit_i;          // current activation magnitude bit (0..2)
  logic       busy, clear_q, nan_q;
  logic signed [7:0] ew_q, ex_q, mw_q, mx_q;
  assign op_ready = !busy || (bit_i == 3'd2);
  assign idle     = !busy;
  // operand registers load once per block: clock-gated (same rule as the static registers of macro_top)
  logic accept, clk_op;
  assign accept = op_ready && op_valid;
  cim_icg u_icg_op (.clk(clk), .en(accept | ~rst_n), .gclk(clk_op));
  always_ff @(posedge clk_op or negedge rst_n) begin
    if (!rst_n) begin
      clear_q <= 1'b0; nan_q <= 1'b0; ew_q <= '0; ex_q <= '0; mw_q <= '0; mx_q <= '0;
    end else if (accept) begin
      clear_q <= op_clear; nan_q <= nan_in; ew_q <= e_w; ex_q <= e_x; mw_q <= max_e_w; mx_q <= max_e_x;
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; bit_i <= '0;
    end else if (accept) begin  // also fires on the last bit of the previous op (busy stays high)
      busy <= 1'b1; bit_i <= '0;
    end else if (busy) begin
      if (bit_i == 3'd2) busy <= 1'b0;
      bit_i <= bit_i + 3'd1;
    end
  end

  // stage A: AND (bitcell NOR) + sign routing + 2 x 3 popcount trees (index 2j = +, 2j+1 = -)
  logic [6*ROWS-1:0] pp;
  logic [6*6-1:0]    ms, mc;
  for (genvar j = 0; j < 3; j++) begin : g_tree
    for (genvar p = 0; p < 2; p++) begin : g_pol
      for (genvar r = 0; r < ROWS; r++) begin : g_and
        logic neg;
        assign neg = w_bus[4*r+3] ^ xsgn_bus[r];
        assign pp[(2*j+p)*ROWS + r] = busy & xbit_bus[r] & w_bus[4*r + j] & (p == 1 ? neg : ~neg);
      end
      /* verilator lint_off PINCONNECTEMPTY */
      col_tree_unsigned_w1_r32 u_pc (.in_bits(pp[(2*j+p)*ROWS +: ROWS]), .h0_s(), .h0_c(), .h1_s(), .h1_c(),
                                     .m_s(ms[6*(2*j+p) +: 6]), .m_c(mc[6*(2*j+p) +: 6]));
      /* verilator lint_on PINCONNECTEMPTY */
    end
  end

  // pipeline register
  logic [6*6-1:0] rs, rc;
  logic       a_valid, a_first, a_last, a_clear, a_nan;
  logic [2:0] a_bit;
  logic signed [7:0] a_ew, a_ex, a_mw, a_mx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_valid <= 1'b0; a_first <= 1'b0; a_last <= 1'b0; a_clear <= 1'b0; a_nan <= 1'b0; a_bit <= '0;
      a_ew <= '0; a_ex <= '0; a_mw <= '0; a_mx <= '0;
      rs <= '0; rc <= '0;
    end else begin
      rs <= ms; rc <= mc;
      a_valid <= busy; a_bit <= bit_i; a_first <= busy && bit_i == 3'd0; a_last <= busy && bit_i == 3'd2;
      a_clear <= clear_q; a_nan <= nan_q; a_ew <= ew_q; a_ex <= ex_q; a_mw <= mw_q; a_mx <= mx_q;
    end
  end

  // stage B: CPAs + combiner + shift-add block accumulator
  logic signed [SB-1:0] s_i, blk_acc, blk_next;
  always_comb begin
    s_i = '0;
    for (int j = 0; j < 3; j++) begin
      s_i = s_i + (SB'(rs[6*(2*j) +: 6] + rc[6*(2*j) +: 6]) << j)
                - (SB'(rs[6*(2*j+1) +: 6] + rc[6*(2*j+1) +: 6]) << j);
    end
  end
  assign blk_next = (a_first ? '0 : blk_acc) + (s_i <<< a_bit);  // sign-magnitude: every cycle weighted +

  logic signed [SB-1:0] blk;
  logic                 b_valid, b_clear, b_nan, b_dneg;
  logic [5:0]           b_d;
  logic signed [9:0]    d_raw;
  assign d_raw = {{2{a_mw[7]}}, a_mw} - {{2{a_ew[7]}}, a_ew} + {{2{a_mx[7]}}, a_mx} - {{2{a_ex[7]}}, a_ex};
  logic clk_b;
  cim_icg u_icg_b (.clk(clk), .en((a_valid && a_last) | ~rst_n), .gclk(clk_b));
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      blk_acc <= '0; b_valid <= 1'b0;
    end else begin
      if (a_valid) blk_acc <= blk_next;
      b_valid <= a_valid && a_last;
    end
  end
  always_ff @(posedge clk_b or negedge rst_n) begin  // block-result stage: once per 3 cycles
    if (!rst_n) begin
      blk <= '0; b_clear <= 1'b0; b_nan <= 1'b0; b_d <= '0; b_dneg <= 1'b0;
    end else if (a_valid && a_last) begin
      blk <= blk_next; b_clear <= a_clear; b_nan <= a_nan; b_dneg <= d_raw[9];
      b_d <= (d_raw > $signed(10'(DMAX))) ? 6'(DMAX) : d_raw[5:0];
    end
  end

  // stage C: alignment + window accumulator (identical to column)
  logic signed [W-1:0] blk_ext, term;
  logic signed [W:0]   sum_wide;
  assign blk_ext = {{(W-SB){blk[SB-1]}}, blk};
  always_comb begin
    if (b_d <= 6'(BASE)) term = blk_ext <<< (6'(BASE) - b_d);
    else                 term = blk_ext >>> (b_d - 6'(BASE));
  end
  assign sum_wide = (b_clear ? {(W+1){1'b0}} : {acc[W-1], acc}) + {term[W-1], term};
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= '0; acc_valid <= 1'b0; nan_out <= 1'b0; ovf <= 1'b0;
    end else begin
      acc_valid <= b_valid;
      if (b_valid) begin
        acc     <= sum_wide[W-1:0];
        nan_out <= (b_clear ? 1'b0 : nan_out) | b_nan;
        ovf     <= (b_clear ? 1'b0 : ovf) | (sum_wide[W] != sum_wide[W-1]) | b_dneg;
      end
    end
  end
endmodule : bs4_column
