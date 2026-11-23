// T1.4: op / set-switch control coverage of macro_top (also inside dpu_macro).
bind macro_top cov_ctrl #(.M(M), .NC(32), .SCW(25), .NANPOS(24)) u_cov_ctrl (
  .clk(clk), .rst_n(rst_n), .op_valid(op_valid), .op_ready(op_ready), .load_req(load_req),
  .load_pending(load_pending), .op_set(op_set), .cur_set(cur_set), .op_clear(op_clear), .op_nv(op_nv),
  .op_x_nan(op_x_nan), .rf(sc_rf));
