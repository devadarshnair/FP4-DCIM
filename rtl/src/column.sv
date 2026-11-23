// One weight column of the FP4 DCIM macro, MX mode (docs/spec/numerics.md):
//   32 leaves -> variant-B compressor tree (carry-save) -> pipeline register -> 14 b CPA
//   -> alignment shift -> W-bit window accumulator.
// Stage 1: leaves + tree; stage 2: CPA + align + accumulate. Result latency: 2 clock edges.
// Row inputs (xp/xp3/xs) come from the shared row drivers; w_bus is the latched weight set.
// Written in the Yosys-compatible SystemVerilog subset (no package import, integer parameters).
`include "cim_defs.vh"
module column #(
  parameter int W     = `CIM_WINDOW_BITS,
  parameter int GUARD = 7,               // ceil(log2(blocks per dot product)); must match the golden model
  parameter int LEAF  = `CIM_LEAF_FP4,   // CIM_LEAF_FP4 | CIM_LEAF_GENERIC (generic INT5 leaf ablation)
  parameter int NV_GATE = 0              // 1: clock-gate NV-only pipeline flops + isolate NV-only operands when nv_mode=0 (docs/spec/nv_gate.md)
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      in_valid,
  input  logic                      acc_clear,   // start a new dot product with this op
  input  logic [`CIM_ROWS*4-1:0]    xp_bus,      // 2|x| per row
  input  logic [`CIM_ROWS*6-1:0]    xp3_bus,     // 3*2|x| per row
  input  logic [`CIM_ROWS-1:0]      xs_bus,      // activation sign per row
  input  logic [`CIM_ROWS*4-1:0]    w_bus,       // latched E2M1 weights, row r at [4r +: 4]
  input  logic                      nv_mode,     // 0: MXFP4 (one 32-row block); 1: NVFP4 (two 16-row blocks)
  input  logic signed [7:0]         e_w,         // MX: weight block exponent (UE8M0); NV: k_w of rows 0-15
  input  logic signed [7:0]         e_x,         // MX: activation block exponent;      NV: k_x of rows 0-15
  input  logic signed [7:0]         e_w1,        // NV only: k_w of rows 16-31
  input  logic signed [7:0]         e_x1,        // NV only: k_x of rows 16-31
  input  logic [3:0]                u_w0, u_w1,  // NV only: E4M3 weight-scale significands (rows 0-15 / 16-31)
  input  logic [3:0]                u_x0, u_x1,  // NV only: E4M3 activation-scale significands
  input  logic signed [7:0]         max_e_w,     // anchor: column max weight exponent (static)
  input  logic signed [7:0]         max_e_x,     // anchor: max activation exponent of the dot product
  input  logic                      nan_in,      // either block scale is UE8M0 NaN
  output logic signed [W-1:0]       acc,
  output logic                      acc_valid,
  output logic                      nan_out,
  output logic                      ovf          // must never assert (anchor analysis); for assertions
);
  localparam int ROWS = `CIM_ROWS;
  localparam int SB   = `CIM_MX_SUM_S_BITS;      // 14
  localparam int BASE = W - SB - 1 - GUARD;      // left shift of a d = 0 term (MX)
  localparam int NB   = `CIM_NV_TERM_BITS + 1;  // 21: signed NV half term S*u_w*u_x incl. sign
  localparam int BASE_NV = W - NB - 1 - GUARD;   // left shift of a d = 0 NV half term
  localparam int DMAX = W - 1;                   // saturation of d (any d >= BASE + SB floors to 0/-1)

  // stage 1: leaves + tree
  logic [ROWS*9-1:0] oc_bus;
  logic [ROWS-1:0]   s_bus;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [ROWS*8-1:0] mag_bus;                    // magnitudes are not used by the variant-B tree
  /* verilator lint_on UNUSEDSIGNAL */
  for (genvar r = 0; r < ROWS; r++) begin : g_leaf
    if (LEAF == `CIM_LEAF_FP4) begin : g_fp4
      leaf_mult u_leaf (
        .w(w_bus[4*r +: 4]), .xp(xp_bus[4*r +: 4]), .xp3(xp3_bus[6*r +: 6]), .xs(xs_bus[r]),
        .mag(mag_bus[8*r +: 8]), .s(s_bus[r]), .oc(oc_bus[9*r +: 9])
      );
    end else begin : g_generic
      generic_leaf u_leaf (
        .w(w_bus[4*r +: 4]), .xp(xp_bus[4*r +: 4]), .xs(xs_bus[r]),
        .mag(mag_bus[8*r +: 8]), .s(s_bus[r]), .oc(oc_bus[9*r +: 9])
      );
    end
  end
  /* verilator lint_off UNUSEDSIGNAL */
  logic [ROWS*6-1:0] xp3_seen;                    // the generic leaf does not use 3X'
  assign xp3_seen = xp3_bus;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [12:0] h0_s, h0_c, h1_s, h1_c;            // half-tree outputs (NVFP4 mode)
  logic [SB-1:0] m_s, m_c;
  col_tree_oc_w9_r32 u_tree (
    .in_bits(oc_bus), .s_bits(s_bus),
    .h0_s(h0_s), .h0_c(h0_c), .h1_s(h1_s), .h1_c(h1_c), .m_s(m_s), .m_c(m_c)
  );

  // alignment amounts, computed in stage 1 (exponents are known with the operands)
  logic signed [9:0] d_raw, d1_raw;
  logic [5:0]        d_sat, d1_sat;
  logic [7:0]        u0, u1;
  // NV_GATE=1: operand isolation of the NV-only stage-1 logic (second exponent difference, E4M3 scale-significand
  // multipliers) -- operands are ANDed with nv_mode so the datapath is quiet in MX ops. NV_GATE=0: constant-folds away.
  logic signed [7:0] mxw1, mxx1, ew1_i, ex1_i;
  logic [3:0]        uw0_i, uw1_i, ux0_i, ux1_i;
  assign mxw1  = (NV_GATE != 0) ? (max_e_w & {8{nv_mode}}) : max_e_w;
  assign mxx1  = (NV_GATE != 0) ? (max_e_x & {8{nv_mode}}) : max_e_x;
  assign ew1_i = (NV_GATE != 0) ? (e_w1 & {8{nv_mode}}) : e_w1;
  assign ex1_i = (NV_GATE != 0) ? (e_x1 & {8{nv_mode}}) : e_x1;
  assign uw0_i = (NV_GATE != 0) ? (u_w0 & {4{nv_mode}}) : u_w0;
  assign uw1_i = (NV_GATE != 0) ? (u_w1 & {4{nv_mode}}) : u_w1;
  assign ux0_i = (NV_GATE != 0) ? (u_x0 & {4{nv_mode}}) : u_x0;
  assign ux1_i = (NV_GATE != 0) ? (u_x1 & {4{nv_mode}}) : u_x1;
  assign d_raw  = {{2{max_e_w[7]}}, max_e_w} - {{2{e_w[7]}}, e_w} + {{2{max_e_x[7]}}, max_e_x} - {{2{e_x[7]}}, e_x};
  assign d1_raw = {{2{mxw1[7]}}, mxw1} - {{2{ew1_i[7]}}, ew1_i} + {{2{mxx1[7]}}, mxx1} - {{2{ex1_i[7]}}, ex1_i};
  assign d_sat  = (d_raw  > $signed(10'(DMAX))) ? 6'(DMAX) : d_raw[5:0];
  assign d1_sat = (d1_raw > $signed(10'(DMAX))) ? 6'(DMAX) : d1_raw[5:0];
  assign u0 = uw0_i * ux0_i;                     // 4b x 4b scale-significand product (<= 225)
  assign u1 = uw1_i * ux1_i;

  // pipeline register (carry-save)
  logic [SB-1:0] r_s, r_c;
  logic [12:0]   r_h0s, r_h0c, r_h1s, r_h1c;
  logic [5:0]    r_d, r_d1;
  logic [7:0]    r_u0, r_u1;
  logic          r_valid, r_clear, r_nan, r_dneg, r_nv;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_s <= '0; r_c <= '0; r_d <= '0; r_valid <= 1'b0; r_clear <= 1'b0; r_nan <= 1'b0; r_dneg <= 1'b0;
      r_nv <= 1'b0;
    end else begin
      r_valid <= in_valid; r_clear <= acc_clear; r_nan <= nan_in; r_nv <= nv_mode;
      r_d <= d_sat;
      r_dneg <= d_raw[9] | (nv_mode & d1_raw[9]);
      if (!nv_mode) begin                         // MX: one 14 b carry-save pair
        r_s <= m_s; r_c <= m_c;
      end
    end
  end
  // NV-only registers (74 flops): two half sums, scale products, second shift amount.
  if (NV_GATE != 0) begin : g_nvff
    logic clk_nv;                                 // gated by the ICG: no edges in MX ops (same cell as the static-register gates)
    cim_icg u_icg_nv (.clk(clk), .en(nv_mode), .gclk(clk_nv));
    always_ff @(posedge clk_nv or negedge rst_n) begin
      if (!rst_n) begin
        r_d1 <= '0; r_u0 <= '0; r_u1 <= '0; r_h0s <= '0; r_h0c <= '0; r_h1s <= '0; r_h1c <= '0;
      end else begin
        r_h0s <= h0_s; r_h0c <= h0_c; r_h1s <= h1_s; r_h1c <= h1_c; r_u0 <= u0; r_u1 <= u1; r_d1 <= d1_sat;
      end
    end
  end else begin : g_nvff
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        r_d1 <= '0; r_u0 <= '0; r_u1 <= '0; r_h0s <= '0; r_h0c <= '0; r_h1s <= '0; r_h1c <= '0;
      end else if (nv_mode) begin
        r_h0s <= h0_s; r_h0c <= h0_c; r_h1s <= h1_s; r_h1c <= h1_c; r_u0 <= u0; r_u1 <= u1; r_d1 <= d1_sat;
      end
    end
  end

  // stage 2: CPA + align + accumulate
  logic [SB-1:0]        blk_u;
  logic signed [W-1:0]  blk_ext, term, acc_next, term_nv0, term_nv1;
  logic signed [W:0]    sum_wide;
  assign blk_u   = r_s + r_c;                    // 14 b CPA (modulo 2^14 = exact block sum)
  assign blk_ext = {{(W-SB){blk_u[SB-1]}}, blk_u};
  always_comb begin
    if (r_d <= 6'(BASE)) term = blk_ext <<< (6'(BASE) - r_d);
    else                 term = blk_ext >>> (r_d - 6'(BASE));
  end
  // NV halves: 13 b CPA -> x (u_w*u_x) -> align (each half has its own exponent)
  logic [12:0]          s0_u, s1_u;
  logic signed [NB-1:0] p0, p1;
  logic signed [W-1:0]  p0_ext, p1_ext;
  assign s0_u = r_h0s + r_h0c;
  assign s1_u = r_h1s + r_h1c;
  assign p0 = $signed(s0_u) * $signed({1'b0, r_u0});
  assign p1 = $signed(s1_u) * $signed({1'b0, r_u1});
  assign p0_ext = {{(W-NB){p0[NB-1]}}, p0};
  assign p1_ext = {{(W-NB){p1[NB-1]}}, p1};
  always_comb begin
    if (r_d  <= 6'(BASE_NV)) term_nv0 = p0_ext <<< (6'(BASE_NV) - r_d);
    else                     term_nv0 = p0_ext >>> (r_d - 6'(BASE_NV));
    if (r_d1 <= 6'(BASE_NV)) term_nv1 = p1_ext <<< (6'(BASE_NV) - r_d1);
    else                     term_nv1 = p1_ext >>> (r_d1 - 6'(BASE_NV));
  end
  logic signed [W:0] add_term;
  assign add_term = r_nv ? ({term_nv0[W-1], term_nv0} + {term_nv1[W-1], term_nv1}) : {term[W-1], term};
  assign sum_wide = (r_clear ? {(W+1){1'b0}} : {acc[W-1], acc}) + add_term;
  assign acc_next = sum_wide[W-1:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= '0; acc_valid <= 1'b0; nan_out <= 1'b0; ovf <= 1'b0;
    end else begin
      acc_valid <= r_valid;
      if (r_valid) begin
        acc     <= acc_next;
        nan_out <= (r_clear ? 1'b0 : nan_out) | r_nan;
        ovf     <= (r_clear ? 1'b0 : ovf) | (sum_wide[W] != sum_wide[W-1]) | r_dneg;
      end
    end
  end
endmodule : column
