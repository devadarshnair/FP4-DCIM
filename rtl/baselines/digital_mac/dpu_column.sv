// Digital MAC baseline, column group: weight-stationary digital MAC lanes. Weights live in a central
// SRAM buffer (central_wbuf: the same OpenRAM macro, placed away from the datapath) and travel over a
// long bus into PE weight registers; datapath = generic FP4 MAC leaves + the same tree and window as
// column. Functionally identical to column_top(STORE=SRAM, LEAF=GENERIC) plus a bus pipeline stage; the
// difference that matters - bus wire length - enters the power flow via flows/pdk/wire_model.yaml.
`include "cim_defs.vh"
module dpu_column #(
  parameter int NCOL  = 4,
  parameter int M     = 16,
  parameter int GUARD = 7,
  parameter int NV_GATE = 0
) (
  input  logic                        clk, rst_n,
  input  logic                        wr_en,
  input  logic [7:0]                  wr_addr,
  input  logic [31:0]                 wr_data,
  input  logic                        ew_wr_en,
  input  logic [$clog2(M)-1:0]        ew_wr_set,
  input  logic [$clog2(NCOL)-1:0]     ew_wr_col,
  input  logic signed [7:0]           ew_wr_val,
  input  logic                        ew_wr_nan,
  input  logic                        maxw_wr_en,
  input  logic [$clog2(NCOL)-1:0]     maxw_wr_col,
  input  logic signed [7:0]           maxw_wr_val,
  input  logic                        op_valid,
  output logic                        op_ready,
  input  logic [`CIM_ROWS*4-1:0]      op_x,
  input  logic signed [7:0]           op_e_x, op_max_e_x,
  input  logic                        op_x_nan,
  input  logic [$clog2(M)-1:0]        op_set,
  input  logic                        op_clear,
  output logic [NCOL*`CIM_WINDOW_BITS-1:0] acc,
  output logic [NCOL-1:0]             acc_valid, nan_out, ovf,
  output logic [31:0]                 set_switches
);
  column_top #(.NCOL(NCOL), .M(M), .GUARD(GUARD), .STORE(`CIM_STORE_SRAM), .LEAF(`CIM_LEAF_GENERIC), .NV_GATE(NV_GATE)) u_lanes (
    .clk(clk), .rst_n(rst_n), .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
    .ew_wr_en(ew_wr_en), .ew_wr_set(ew_wr_set), .ew_wr_col(ew_wr_col), .ew_wr_val(ew_wr_val), .ew_wr_nan(ew_wr_nan),
    .maxw_wr_en(maxw_wr_en), .maxw_wr_col(maxw_wr_col), .maxw_wr_val(maxw_wr_val),
    .op_valid(op_valid), .op_ready(op_ready), .op_x(op_x), .op_e_x(op_e_x), .op_max_e_x(op_max_e_x),
    .op_x_nan(op_x_nan), .op_set(op_set), .op_clear(op_clear),
    .acc(acc), .acc_valid(acc_valid), .nan_out(nan_out), .ovf(ovf), .set_switches(set_switches));
endmodule : dpu_column
