// T1.4: bind the bit-serial column coverage monitor into every bs_column instance.
bind bs_column cov_bs_column #(.W(W), .GUARD(GUARD)) u_cov (
  .clk(clk), .rst_n(rst_n), .accept(accept), .busy(busy), .bit_i(bit_i), .op_clear(op_clear), .nan_in(nan_in),
  .a_valid(a_valid), .a_last(a_last), .d_raw(d_raw), .b_valid(b_valid), .b_clear(b_clear), .b_nan(b_nan),
  .b_d(b_d), .blk(blk), .nan_out(nan_out), .ovf(ovf));
