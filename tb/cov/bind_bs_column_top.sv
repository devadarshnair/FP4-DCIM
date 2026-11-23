// T1.4: INT5 baseline (bs_column_top): control + per-column coverage.
bind bs_column_top cov_ctrl #(.M(M), .NC(NCOL), .SCW(1), .NANPOS(0)) u_cov_ctrl (
  .clk(clk), .rst_n(rst_n), .op_valid(op_valid), .op_ready(op_ready), .load_req(load_req),
  .load_pending(load_pending), .op_set(op_set), .cur_set(cur_set), .op_clear(op_clear), .op_nv(1'b0),
  .op_x_nan(op_x_nan), .rf(ewn_rf));
