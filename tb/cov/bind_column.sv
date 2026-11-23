// T1.4: bind the column coverage monitor into every v2 `column` instance (v2 column/macro and DPU).
bind column cov_column #(.W(W), .GUARD(GUARD)) u_cov (
  .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .acc_clear(acc_clear), .nv_mode(nv_mode), .nan_in(nan_in),
  .e_w(e_w), .e_x(e_x), .e_w1(e_w1), .e_x1(e_x1), .max_e_w(max_e_w), .max_e_x(max_e_x),
  .u_w0(u_w0), .u_w1(u_w1), .u_x0(u_x0), .u_x1(u_x1), .d_raw(d_raw), .d1_raw(d1_raw),
  .r_valid(r_valid), .r_clear(r_clear), .r_nv(r_nv), .r_nan(r_nan), .r_d(r_d), .r_d1(r_d1),
  .blk_u(blk_u), .s0_u(s0_u), .s1_u(s1_u), .term(term), .sum_wide(sum_wide), .acc(acc),
  .nan_out(nan_out), .ovf(ovf));
