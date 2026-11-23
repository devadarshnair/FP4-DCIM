// Baseline: TSMC-style bit-serial INT5 DCIM column, MX mode.
// Weights: E2M1 -> INT5 (2w in [-12,12]) decoded statically from the latched set; activations:
// INT5 (2x) fed LSB-first by the *shared* row shift registers of bs_column_top (one bit per row
// broadcast to all columns, as in TSMC-style DCIM), one bit per cycle over 5 cycles (bit 4 weighted -16). Per cycle, five
// 32-input popcount trees (one per weight bit; generated w_in=1 tree) -> carry-save register ->
// CPAs -> combiner S_i = sum_j +-(pc_j << j) -> shift-add block accumulator -> after bit 4 the block
// sum goes through the same alignment + W-bit window as column. One block per 5 cycles.
`include "cim_defs.vh"
module bs_column #(
  parameter int W     = `CIM_WINDOW_BITS,
  parameter int GUARD = 7
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                op_valid,       // new block (accepted when op_ready)
  output logic                op_ready,       // !busy || last bit: back-to-back issue, 5 cycles/op
  output logic                idle,           // no block in flight (for set-switch control)
  input  logic                op_clear,
  input  logic [`CIM_ROWS-1:0]   xbit_bus,    // current activation bit per row (shared row shift register)
  input  logic [`CIM_ROWS*4-1:0] w_bus,       // latched E2M1 weights
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

  // E2M1 -> INT5 (2v) decode: |2v| = A << a, A = {e!=0, m}, a = max(e,1)-1
  logic [ROWS*5-1:0] w5;   // static: toggles only on a set switch
  for (genvar r = 0; r < ROWS; r++) begin : g_dec
    logic [1:0] we;
    logic [3:0] wm;
    assign we = w_bus[4*r+1 +: 2];
    assign wm = {2'b00, (we != 2'b00), w_bus[4*r]} << ((we == 2'b00) ? 2'd0 : we - 2'd1);
    assign w5[5*r +: 5] = w_bus[4*r+3] ? (5'd0 - {1'b0, wm}) : {1'b0, wm};
  end

  // activation shift register (row-driver equivalent) and sequencer
  logic [2:0] bit_i;          // current activation bit fed this cycle
  logic       busy, clear_q, nan_q;
  logic signed [7:0] ew_q, ex_q, mw_q, mx_q;
  assign op_ready = !busy || (bit_i == 3'd4);
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
      if (bit_i == 3'd4) busy <= 1'b0;
      bit_i <= bit_i + 3'd1;
    end
  end

  // stage A: AND (bitcell NOR) + five popcount trees
  logic [5*ROWS-1:0] pp;
  logic [5*6-1:0]    ms, mc;
  for (genvar j = 0; j < 5; j++) begin : g_tree
    for (genvar r = 0; r < ROWS; r++) begin : g_and
      assign pp[j*ROWS + r] = busy & xbit_bus[r] & w5[5*r + j];
    end
    /* verilator lint_off PINCONNECTEMPTY */
    col_tree_unsigned_w1_r32 u_pc (.in_bits(pp[j*ROWS +: ROWS]), .h0_s(), .h0_c(), .h1_s(), .h1_c(),
                                   .m_s(ms[6*j +: 6]), .m_c(mc[6*j +: 6]));
    /* verilator lint_on PINCONNECTEMPTY */
  end

  // pipeline register
  logic [5*6-1:0] rs, rc;
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
      a_valid <= busy; a_bit <= bit_i; a_first <= busy && bit_i == 3'd0; a_last <= busy && bit_i == 3'd4;
      a_clear <= clear_q; a_nan <= nan_q; a_ew <= ew_q; a_ex <= ex_q; a_mw <= mw_q; a_mx <= mx_q;
    end
  end

  // stage B: CPAs + combiner + shift-add block accumulator
  logic signed [SB-1:0] s_i, blk_acc, blk_next;
  always_comb begin
    s_i = '0;
    for (int j = 0; j < 5; j++) begin
      if (j == 4) s_i = s_i - (SB'(rs[6*j +: 6] + rc[6*j +: 6]) << j);
      else        s_i = s_i + (SB'(rs[6*j +: 6] + rc[6*j +: 6]) << j);
    end
  end
  assign blk_next = (a_first ? '0 : blk_acc) + ((a_bit == 3'd4) ? -(s_i <<< a_bit) : (s_i <<< a_bit));

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
  always_ff @(posedge clk_b or negedge rst_n) begin  // block-result stage: once per 5 cycles
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
endmodule : bs_column
