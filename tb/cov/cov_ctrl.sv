// T1.4 functional-coverage monitor for the op / set-switch control of macro_top, column_top and the
// INT5 tops (bs_column_top, bs_macro_top). Bound by tb/cov/bind_*.sv. Coverage only.
// `rf` carries the per-(set, column) scale records; bit NANPOS of each SCW-bit record is the weight NaN.
module cov_ctrl #(
  parameter int M      = 16,
  parameter int NC     = 32,
  parameter int SCW    = 1,
  parameter int NANPOS = 0
) (
  input logic                   clk,
  input logic                   rst_n,
  input logic                   op_valid,
  input logic                   op_ready,
  input logic                   load_req,
  input logic                   load_pending,
  input logic [$clog2(M)-1:0]   op_set,
  input logic [$clog2(M)-1:0]   cur_set,
  input logic                   op_clear,
  input logic                   op_nv,
  input logic                   op_x_nan,
  input logic [M*NC*SCW-1:0]    rf
);
  logic fire;
  assign fire = op_valid && op_ready;
  logic prev_fire, seen, last_nv, since_switch;
  logic [$clog2(M)-1:0] last_set;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_fire <= 1'b0; seen <= 1'b0; last_nv <= 1'b0; last_set <= '0; since_switch <= 1'b0;
    end else begin
      prev_fire <= fire;
      if (fire) begin seen <= 1'b1; last_nv <= op_nv; last_set <= op_set; end
      if (load_req) since_switch <= 1'b1;
      else if (fire) since_switch <= 1'b0;
    end
  end
  logic w_nan;
  always_comb begin
    w_nan = 1'b0;
    for (int c = 0; c < NC; c++) w_nan |= rf[(int'(cur_set) * NC + c) * SCW + NANPOS];
  end

  C_FIRE:              cover property (@(posedge clk) disable iff (!rst_n) fire);
  C_STALL:             cover property (@(posedge clk) disable iff (!rst_n) op_valid && !op_ready);
  C_STALL_LOADING:     cover property (@(posedge clk) disable iff (!rst_n) op_valid && load_pending);
  C_SWITCH:            cover property (@(posedge clk) disable iff (!rst_n) load_req);
  C_SWITCH_AFTER_FIRE: cover property (@(posedge clk) disable iff (!rst_n) load_req && seen);
  C_FIRST_AFTER_SW:    cover property (@(posedge clk) disable iff (!rst_n) fire && since_switch);
  C_B2B_SAME_SET:      cover property (@(posedge clk) disable iff (!rst_n) fire && prev_fire && op_set == last_set);
  C_SET_ZERO:          cover property (@(posedge clk) disable iff (!rst_n) fire && op_set == 0);
  C_SET_MAX:           cover property (@(posedge clk) disable iff (!rst_n) fire && int'(op_set) == M - 1 && M > 1);
  C_CLEAR:             cover property (@(posedge clk) disable iff (!rst_n) fire && op_clear);
  C_X_NAN:             cover property (@(posedge clk) disable iff (!rst_n) fire && op_x_nan);
  C_W_NAN:             cover property (@(posedge clk) disable iff (!rst_n) fire && w_nan);
  C_MX:                cover property (@(posedge clk) disable iff (!rst_n) fire && !op_nv);
  C_NV:                cover property (@(posedge clk) disable iff (!rst_n) fire && op_nv);
  C_NV_TOGGLE:         cover property (@(posedge clk) disable iff (!rst_n) fire && seen && op_nv != last_nv);
  C_NV_TOGGLE_B2B:     cover property (@(posedge clk) disable iff (!rst_n) fire && prev_fire && op_nv != last_nv);
  C_NV_TOGGLE_SAMESET: cover property (@(posedge clk) disable iff (!rst_n) fire && seen && op_nv != last_nv
                                       && op_set == last_set);
endmodule : cov_ctrl
