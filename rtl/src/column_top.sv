// Column group with real weight storage: 32 row drivers, one weight-set store shared by
// NCOL=4 columns (OpenRAM SRAM, or the latch ablation), a per-set weight-exponent register file and
// a valid/ready op interface. Single buffer: an op whose set differs from the active one first
// triggers a set switch and stalls until it completes. See wset_pingpong for the double-buffered store.
`include "cim_defs.vh"
module column_top #(
  parameter int    NCOL  = 4,
  parameter int    M     = 16,
  parameter int    GUARD = 7,
  parameter int    STORE = `CIM_STORE_SRAM,  // CIM_STORE_SRAM | CIM_STORE_LATCH
  parameter int    LEAF  = `CIM_LEAF_FP4,    // CIM_LEAF_FP4 | CIM_LEAF_GENERIC (generic INT5 leaf, as in dpu_column)
  parameter int    NV_GATE = 0               // see column.sv / docs/spec/nv_gate.md
) (
  input  logic                      clk,
  input  logic                      rst_n,
  // weight-code writes (32-bit words, see docs/spec/numerics.md section Storage)
  input  logic                      wr_en,
  input  logic [7:0]                wr_addr,
  input  logic [31:0]               wr_data,
  // weight exponent writes: e_w for (set, column), and static per-column anchors max_e_w
  input  logic                      ew_wr_en,
  input  logic [$clog2(M)-1:0]      ew_wr_set,
  input  logic [$clog2(NCOL)-1:0]   ew_wr_col,
  input  logic signed [7:0]         ew_wr_val,
  input  logic                      ew_wr_nan,
  input  logic                      maxw_wr_en,
  input  logic [$clog2(NCOL)-1:0]   maxw_wr_col,
  input  logic signed [7:0]         maxw_wr_val,
  // compute ops
  input  logic                      op_valid,
  output logic                      op_ready,
  input  logic [`CIM_ROWS*4-1:0]    op_x,        // E2M1 codes, row r at [4r +: 4]
  input  logic signed [7:0]         op_e_x,
  input  logic signed [7:0]         op_max_e_x,
  input  logic                      op_x_nan,
  input  logic [$clog2(M)-1:0]      op_set,
  input  logic                      op_clear,
  output logic [NCOL*`CIM_WINDOW_BITS-1:0] acc,
  output logic [NCOL-1:0]           acc_valid,
  output logic [NCOL-1:0]           nan_out,
  output logic [NCOL-1:0]           ovf,
  output logic [31:0]               set_switches   // event counter (power proxy / sanity)
);
  localparam int ROWS = `CIM_ROWS;
  localparam int WB   = `CIM_WINDOW_BITS;
  // row drivers
  logic [ROWS*4-1:0] xp_bus;
  logic [ROWS*6-1:0] xp3_bus;
  logic [ROWS-1:0]   xs_bus;
  for (genvar r = 0; r < ROWS; r++) begin : g_row
    row_driver u_rd (.x(op_x[4*r +: 4]), .xp(xp_bus[4*r +: 4]), .xp3(xp3_bus[6*r +: 6]), .xs(xs_bus[r]));
  end

  // weight-set store
  logic                  load_req, busy;
  logic [$clog2(M)-1:0]  load_set;
  logic [NCOL*128-1:0]   w_active;
  if (STORE == `CIM_STORE_SRAM) begin : g_sram
    wset_sram #(.NCOL(NCOL), .M(M)) u_store (
      .clk, .rst_n, .wr_en(wr_en & ~busy), .wr_addr, .wr_data,
      .load_req, .load_set, .busy, .w_active);
  end else begin : g_latch
    wset_latch #(.NCOL(NCOL), .M(M)) u_store (
      .clk, .rst_n, .wr_en(wr_en & ~busy), .wr_addr, .wr_data,
      .load_req, .load_set, .busy, .w_active);
  end

  // exponent register files
  logic [M*NCOL*8-1:0] ew_rf;     // (set, column) -> e_w, flattened for Yosys
  logic [M*NCOL-1:0]   ewn_rf;
  logic [NCOL*8-1:0]   maxw;
  // static registers are clock-gated: they change only on configuration writes
  logic clk_ew, clk_maxw;
  cim_icg u_icg_ew   (.clk(clk), .en(ew_wr_en),   .gclk(clk_ew));
  cim_icg u_icg_maxw (.clk(clk), .en(maxw_wr_en), .gclk(clk_maxw));
  always_ff @(posedge clk_ew) begin
    if (ew_wr_en) begin
      ew_rf[(int'(ew_wr_set)*NCOL + int'(ew_wr_col))*8 +: 8] <= ew_wr_val;
      ewn_rf[int'(ew_wr_set)*NCOL + int'(ew_wr_col)]        <= ew_wr_nan;
    end
  end
  always_ff @(posedge clk_maxw) begin
    if (maxw_wr_en) maxw[maxw_wr_col*8 +: 8] <= maxw_wr_val;
  end

  // set-switch control
  logic                  cur_valid, load_pending;
  logic [$clog2(M)-1:0]  cur_set;
  logic                  need_switch, fire;
  assign need_switch = op_valid && (!cur_valid || op_set != cur_set);
  assign op_ready    = cur_valid && !busy && !load_pending && op_set == cur_set;
  assign fire        = op_valid && op_ready;
  assign load_req    = need_switch && !busy && !load_pending;
  assign load_set    = op_set;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_valid <= 1'b0; cur_set <= '0; load_pending <= 1'b0; set_switches <= '0;
    end else begin
      if (load_req) begin
        load_pending <= 1'b1; cur_set <= op_set; cur_valid <= 1'b0; set_switches <= set_switches + 1;
      end else if (load_pending && !busy) begin
        // the store raises busy on the edge that accepts load_req, so this fires when it falls again
        load_pending <= 1'b0; cur_valid <= 1'b1;
      end
    end
  end

  // columns
  for (genvar c = 0; c < NCOL; c++) begin : g_col
    column #(.GUARD(GUARD), .LEAF(LEAF), .NV_GATE(NV_GATE)) u_col (
      .clk, .rst_n,
      .in_valid(fire), .acc_clear(op_clear),
      .xp_bus, .xp3_bus, .xs_bus,
      .w_bus(w_active[c*128 +: 128]),
      .nv_mode(1'b0), .e_w1(8'sd0), .e_x1(8'sd0), .u_w0(4'd1), .u_w1(4'd1), .u_x0(4'd1), .u_x1(4'd1),
      .e_w(ew_rf[(cur_set*NCOL + c)*8 +: 8]), .e_x(op_e_x), .max_e_w(maxw[c*8 +: 8]), .max_e_x(op_max_e_x),
      .nan_in(op_x_nan | ewn_rf[cur_set*NCOL + c]),
      .acc(acc[c*WB +: WB]), .acc_valid(acc_valid[c]), .nan_out(nan_out[c]), .ovf(ovf[c]));
  end
endmodule : column_top
