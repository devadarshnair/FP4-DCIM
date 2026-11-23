// Baseline macro (lossy, MIDAS-inspired): bit-serial sign-magnitude MXINT4-SM columns (bs4_column,
// 3 magnitude cycles per block). NOT exact E2M1 and NOT the INT5 control. Same boundary as bs_macro_top / macro_top
// (32 columns, 8 OpenRAM stores, same capacity / write port / scale RFs / set-switch control); only the shared
// activation row registers and the column differ (as bs4_column_top vs bs_column_top). Codes are {sign, mag[2:0]}
// per row (4 bits/row, store shared); exponents are plain UE8M0. MX-style scaling only (op_nv must be 0; the NV
// ports exist so the macro-level testbench is shared).
`include "cim_defs.vh"
module bs4_macro_top #(
  parameter int M     = 16,
  parameter int GUARD = 7,
  parameter int STORE = `CIM_STORE_SRAM
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
  logic [31:0] col_ready, col_idle;
  assign op_ready    = cur_valid && !busy && !load_pending && op_set == cur_set && (&col_ready);
  assign fire        = op_valid && op_ready;
  assign load_req    = need_switch && !busy && !load_pending && (&col_idle);  // never switch mid-block
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_valid <= 1'b0; cur_set <= '0; load_pending <= 1'b0; set_switches <= '0;
    end else if (load_req) begin
      load_pending <= 1'b1; cur_set <= op_set; cur_valid <= 1'b0; set_switches <= set_switches + 1;
    end else if (load_pending && !busy) begin
      load_pending <= 1'b0; cur_valid <= 1'b1;
    end
  end

  // shared activation row registers: 3-bit magnitude shift register + static sign per row
  logic [ROWS*3-1:0] xsr;
  logic [ROWS-1:0]   xbit_bus, xsgn_bus;
  logic              shifting, clk_xsr;
  assign shifting = ~(&col_ready);
  cim_icg u_icg_xsr (.clk(clk), .en(fire | shifting | ~rst_n), .gclk(clk_xsr));
  for (genvar r = 0; r < ROWS; r++) begin : g_xrow
    always_ff @(posedge clk_xsr or negedge rst_n) begin
      if (!rst_n) begin
        xsr[3*r +: 3] <= '0; xsgn_bus[r] <= 1'b0;
      end else if (fire) begin
        xsr[3*r +: 3] <= op_x[4*r +: 3];                            // magnitude: no decode
        xsgn_bus[r]   <= op_x[4*r+3];                               // sign: held for the 3 cycles of the block
      end else if (shifting) begin
        xsr[3*r +: 3] <= {1'b0, xsr[3*r+1 +: 2]};                   // LSB-first
      end
    end
    assign xbit_bus[r] = xsr[3*r];
  end

  // 32 bit-serial columns
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_nv;  // NV ports are unused by the bit-serial baseline (MXFP4 only)
  assign unused_nv = op_nv ^ (|op_e_x1) ^ (|op_u_x0) ^ (|op_u_x1);
  /* verilator lint_on UNUSEDSIGNAL */
  for (genvar c = 0; c < 32; c++) begin : g_col
    /* verilator lint_off UNUSEDSIGNAL */
    logic [SCW-1:0] sc;  // NV fields (e1, u0, u1) unused in the MX-only baseline
    /* verilator lint_on UNUSEDSIGNAL */
    assign sc = sc_rf[(int'(cur_set)*32 + c)*SCW +: SCW];
    bs4_column #(.GUARD(GUARD)) u_col (
      .clk, .rst_n, .op_valid(fire), .op_ready(col_ready[c]), .idle(col_idle[c]), .op_clear(op_clear), .xbit_bus, .xsgn_bus,
      .w_bus(w_active[(c/4)*512 + (c%4)*128 +: 128]),
      .e_w(sc[7:0]), .e_x(op_e_x0), .max_e_w(maxw[c*8 +: 8]), .max_e_x(op_max_e_x),
      .nan_in(op_x_nan | sc[24]),
      .acc(acc[c*WB +: WB]), .acc_valid(acc_valid[c]), .nan_out(nan_out[c]), .ovf(ovf[c]));
  end
endmodule : bs4_macro_top
