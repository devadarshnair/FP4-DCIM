// FP4 DCIM macro: 32 rows x 32 columns, shared row drivers, one weight-set store per 4 columns
// (8 x sky130 OpenRAM macros, or the latch ablation), per-(set, column) exponent / NVFP4 scale register
// files, MX and NVFP4 modes, valid/ready op interface with a stalling (single-buffer) set switch.
`include "cim_defs.vh"
module macro_top #(
  parameter int M     = 16,
  parameter int GUARD = 7,
  parameter int STORE = `CIM_STORE_SRAM,
  parameter int LEAF  = `CIM_LEAF_FP4,    // CIM_LEAF_GENERIC: generic INT5 leaf datapath (as in dpu_macro)
  parameter int NV_GATE = 0               // 1: NV clock gating + operand isolation (docs/spec/nv_gate.md)
) (
  input  logic                         clk,
  input  logic                         rst_n,
  // weight-code writes: group g (4 columns each) word address as in column_top
  input  logic                         wr_en,
  input  logic [2:0]                   wr_group,
  input  logic [7:0]                   wr_addr,
  input  logic [31:0]                  wr_data,
  // per-(set, column) scale metadata: MX uses e0 only; NV uses e0/e1 (k) and u0/u1 (significands)
  input  logic                         sc_wr_en,
  input  logic [$clog2(M)-1:0]         sc_wr_set,
  input  logic [4:0]                   sc_wr_col,
  input  logic signed [7:0]            sc_wr_e0, sc_wr_e1,
  input  logic [3:0]                   sc_wr_u0, sc_wr_u1,
  input  logic                         sc_wr_nan,
  input  logic                         maxw_wr_en,
  input  logic [4:0]                   maxw_wr_col,
  input  logic signed [7:0]            maxw_wr_val,
  // ops
  input  logic                         op_valid,
  output logic                         op_ready,
  input  logic                         op_nv,          // 0: MXFP4, 1: NVFP4
  input  logic [`CIM_ROWS*4-1:0]       op_x,
  input  logic signed [7:0]            op_e_x0, op_e_x1, op_max_e_x,
  input  logic [3:0]                   op_u_x0, op_u_x1,
  input  logic                         op_x_nan,
  input  logic [$clog2(M)-1:0]         op_set,
  input  logic                         op_clear,
  output logic [32*`CIM_WINDOW_BITS-1:0] acc,
  output logic [31:0]                  acc_valid,
  output logic [31:0]                  nan_out,
  output logic [31:0]                  ovf,
  output logic [31:0]                  set_switches
);
  localparam int ROWS = `CIM_ROWS;
  localparam int WB   = `CIM_WINDOW_BITS;
  localparam int NG   = 8;  // column groups (4 columns per SRAM macro)

  // shared row drivers
  logic [ROWS*4-1:0] xp_bus;
  logic [ROWS*6-1:0] xp3_bus;
  logic [ROWS-1:0]   xs_bus;
  for (genvar r = 0; r < ROWS; r++) begin : g_row
    row_driver u_rd (.x(op_x[4*r +: 4]), .xp(xp_bus[4*r +: 4]), .xp3(xp3_bus[6*r +: 6]), .xs(xs_bus[r]));
  end

  // weight-set stores (8 groups, switched in parallel)
  logic                  load_req;
  logic [NG-1:0]         busy_g;
  logic [NG*512-1:0]     w_active;
  logic                  busy;
  assign busy = |busy_g;
  for (genvar g = 0; g < NG; g++) begin : g_store
    if (STORE == `CIM_STORE_SRAM) begin : g_sram
      wset_sram #(.NCOL(4), .M(M)) u_store (
        .clk, .rst_n, .wr_en(wr_en & (wr_group == 3'(g)) & ~busy), .wr_addr, .wr_data,
        .load_req, .load_set(op_set), .busy(busy_g[g]), .w_active(w_active[g*512 +: 512]));
    end else begin : g_latch
      wset_latch #(.NCOL(4), .M(M)) u_store (
        .clk, .rst_n, .wr_en(wr_en & (wr_group == 3'(g)) & ~busy), .wr_addr, .wr_data,
        .load_req, .load_set(op_set), .busy(busy_g[g]), .w_active(w_active[g*512 +: 512]));
    end
  end

  // scale register files (clock-gated: change only on writes)
  localparam int SCW = 8 + 8 + 4 + 4 + 1;  // e0, e1, u0, u1, nan
  logic [M*32*SCW-1:0] sc_rf;
  logic [32*8-1:0]     maxw;
  logic clk_sc, clk_maxw;
  cim_icg u_icg_sc   (.clk(clk), .en(sc_wr_en),   .gclk(clk_sc));
  cim_icg u_icg_maxw (.clk(clk), .en(maxw_wr_en), .gclk(clk_maxw));
  always_ff @(posedge clk_sc) begin
    if (sc_wr_en)
      sc_rf[(int'(sc_wr_set)*32 + int'(sc_wr_col))*SCW +: SCW] <= {sc_wr_nan, sc_wr_u1, sc_wr_u0, sc_wr_e1, sc_wr_e0};
  end
  always_ff @(posedge clk_maxw) begin
    if (maxw_wr_en) maxw[int'(maxw_wr_col)*8 +: 8] <= maxw_wr_val;
  end

  // set-switch control (as column_top)
  logic                  cur_valid, load_pending, need_switch, fire;
  logic [$clog2(M)-1:0]  cur_set;
  assign need_switch = op_valid && (!cur_valid || op_set != cur_set);
  assign op_ready    = cur_valid && !busy && !load_pending && op_set == cur_set;
  assign fire        = op_valid && op_ready;
  assign load_req    = need_switch && !busy && !load_pending;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_valid <= 1'b0; cur_set <= '0; load_pending <= 1'b0; set_switches <= '0;
    end else if (load_req) begin
      load_pending <= 1'b1; cur_set <= op_set; cur_valid <= 1'b0; set_switches <= set_switches + 1;
    end else if (load_pending && !busy) begin
      load_pending <= 1'b0; cur_valid <= 1'b1;
    end
  end

  // 32 columns
  for (genvar c = 0; c < 32; c++) begin : g_col
    logic [SCW-1:0] sc;
    assign sc = sc_rf[(int'(cur_set)*32 + c)*SCW +: SCW];
    column #(.GUARD(GUARD), .LEAF(LEAF), .NV_GATE(NV_GATE)) u_col (
      .clk, .rst_n, .in_valid(fire), .acc_clear(op_clear),
      .xp_bus, .xp3_bus, .xs_bus,
      .w_bus(w_active[(c/4)*512 + (c%4)*128 +: 128]),
      .nv_mode(op_nv),
      .e_w(sc[7:0]), .e_w1(sc[15:8]), .u_w0(sc[19:16]), .u_w1(sc[23:20]),
      .e_x(op_e_x0), .e_x1(op_e_x1), .u_x0(op_u_x0), .u_x1(op_u_x1),
      .max_e_w(maxw[c*8 +: 8]), .max_e_x(op_max_e_x),
      .nan_in(op_x_nan | sc[24]),
      .acc(acc[c*WB +: WB]), .acc_valid(acc_valid[c]), .nan_out(nan_out[c]), .ovf(ovf[c]));
  end
endmodule : macro_top
