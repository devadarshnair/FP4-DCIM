// Bit-serial INT5 baseline column group: identical storage/control to column_top with bit-serial
// INT5 columns (5 cycles per block). Column group with real weight storage: 32 row drivers, one weight-set store shared by
// NCOL=4 columns (OpenRAM SRAM, or the latch ablation), a per-set weight-exponent register file and
// a valid/ready op interface. Single buffer: an op whose set differs from the active one first
// triggers a set switch and stalls until it completes. See wset_pingpong for the double-buffered store.
`include "cim_defs.vh"
module bs_column_top #(
  parameter int    NCOL  = 4,
  parameter int    M     = 16,
  parameter int    GUARD = 7,
  parameter int    STORE = `CIM_STORE_SRAM   // CIM_STORE_SRAM | CIM_STORE_LATCH
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
  localparam int WB   = `CIM_WINDOW_BITS;
  localparam int ROWS = `CIM_ROWS;
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
  logic [NCOL-1:0] col_ready, col_idle;
  assign op_ready    = cur_valid && !busy && !load_pending && op_set == cur_set && (&col_ready);
  assign fire        = op_valid && op_ready;
  assign load_req    = need_switch && !busy && !load_pending && (&col_idle);  // never switch mid-block
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

  // shared activation row shift registers (decode once per row)
  logic [ROWS*5-1:0] xsr;
  logic [ROWS-1:0]   xbit_bus;
  logic              shifting, clk_xsr;
  assign shifting = ~(&col_ready);
  cim_icg u_icg_xsr (.clk(clk), .en(fire | shifting | ~rst_n), .gclk(clk_xsr));
  for (genvar r = 0; r < ROWS; r++) begin : g_xrow
    logic [1:0] xe;
    logic [3:0] xm;
    logic [4:0] xd;
    assign xe = op_x[4*r+1 +: 2];
    assign xm = {2'b00, (xe != 2'b00), op_x[4*r]} << ((xe == 2'b00) ? 2'd0 : xe - 2'd1);
    assign xd = op_x[4*r+3] ? (5'd0 - {1'b0, xm}) : {1'b0, xm};
    always_ff @(posedge clk_xsr or negedge rst_n) begin
      if (!rst_n)        xsr[5*r +: 5] <= '0;
      else if (fire)     xsr[5*r +: 5] <= xd;                         // load INT5 (2x)
      else if (shifting) xsr[5*r +: 5] <= {1'b0, xsr[5*r+1 +: 4]};    // LSB-first
    end
    assign xbit_bus[r] = xsr[5*r];
  end

  // columns
  for (genvar c = 0; c < NCOL; c++) begin : g_col
    bs_column #(.GUARD(GUARD)) u_col (
      .clk, .rst_n,
      .op_valid(fire), .op_ready(col_ready[c]), .idle(col_idle[c]), .op_clear(op_clear), .xbit_bus(xbit_bus),
      .w_bus(w_active[c*128 +: 128]),
      .e_w(ew_rf[(cur_set*NCOL + c)*8 +: 8]), .e_x(op_e_x), .max_e_w(maxw[c*8 +: 8]), .max_e_x(op_max_e_x),
      .nan_in(op_x_nan | ewn_rf[cur_set*NCOL + c]),
      .acc(acc[c*WB +: WB]), .acc_valid(acc_valid[c]), .nan_out(nan_out[c]), .ovf(ovf[c]));
  end
endmodule : bs_column_top
