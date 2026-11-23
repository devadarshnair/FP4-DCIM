`include "cim_defs.vh"
module dpu_macro #(
  parameter int M     = 16,
  parameter int GUARD = 7,
  parameter int STORE = `CIM_STORE_SRAM,
  parameter int NV_GATE = 0
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
  // Digital MAC baseline macro: weight-stationary digital MAC lanes (generic FP4 leaf: INT5 decode + 5x5
  // multiplier) with the same tree, window, NV path, capacity and write port as macro_top. The weight SRAM is a
  // *central* buffer placed away from the datapath; its distance enters the power flow through the bus
  // wire model (src/cimfp4/power/boundary.py: length 2*sqrt(area) vs sqrt(area)/2 for macro_top's local stores).
  macro_top #(.M(M), .GUARD(GUARD), .STORE(STORE), .LEAF(`CIM_LEAF_GENERIC), .NV_GATE(NV_GATE)) u_lanes (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(wr_en),
    .wr_group(wr_group),
    .wr_addr(wr_addr),
    .wr_data(wr_data),
    .sc_wr_en(sc_wr_en),
    .sc_wr_set(sc_wr_set),
    .sc_wr_col(sc_wr_col),
    .sc_wr_e0(sc_wr_e0),
    .sc_wr_e1(sc_wr_e1),
    .sc_wr_u0(sc_wr_u0),
    .sc_wr_u1(sc_wr_u1),
    .sc_wr_nan(sc_wr_nan),
    .maxw_wr_en(maxw_wr_en),
    .maxw_wr_col(maxw_wr_col),
    .maxw_wr_val(maxw_wr_val),
    .op_valid(op_valid),
    .op_ready(op_ready),
    .op_nv(op_nv),
    .op_x(op_x),
    .op_e_x0(op_e_x0),
    .op_e_x1(op_e_x1),
    .op_max_e_x(op_max_e_x),
    .op_u_x0(op_u_x0),
    .op_u_x1(op_u_x1),
    .op_x_nan(op_x_nan),
    .op_set(op_set),
    .op_clear(op_clear),
    .acc(acc),
    .acc_valid(acc_valid),
    .nan_out(nan_out),
    .ovf(ovf),
    .set_switches(set_switches));
endmodule : dpu_macro
