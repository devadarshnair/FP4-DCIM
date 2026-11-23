// e28 / T2.3 [EST] HIER cluster reducer (docs/spec/numerics.md "HIER interface", reducer option `fp32`): one macro pass
// emits per column a W-bit signed window value and lsb_tile (10 b signed); the reducer converts value * 2^lsb to fp32 and
// adds it to a running fp32 accumulator (sequential fp32 adds). The repo has NO RTL for this block: this is a
// measurement model (int->fp32 round-nearest-even, fp32 round-nearest-even adder, subnormals flushed, no Inf/NaN
// handling: out of range for LLM data), written only to cost the block. One instance = one column's reducer.
module e28_hier_reducer #(parameter int W = 40) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                valid,
  input  logic                clear,      // first pass of an output: acc <= value
  input  logic signed [W-1:0] win,
  input  logic signed [9:0]   lsb,
  output logic [31:0]         acc
);
  // ---- int -> fp32 (value = win * 2^lsb), RNE ----
  logic [W-1:0] mag;
  logic         s_c;
  logic [5:0]   p;
  logic [W-1:0] shl;
  logic [23:0]  m_c;
  logic         g_c, st_c, up_c;
  logic [24:0]  m_r;
  logic signed [11:0] e_c;
  logic [31:0]  b;
  always_comb begin
    s_c = win[W-1];
    mag = s_c ? (~win + 1'b1) : win;
    p = 6'd0;
    for (int i = 0; i < W; i++) if (mag[i]) p = 6'(i);
    shl = '0; m_c = '0; g_c = 1'b0; st_c = 1'b0;
    if (p >= 6'd23) begin
      shl  = mag >> (p - 6'd23);
      m_c  = shl[23:0];
      g_c  = (p >= 6'd24) ? mag[p - 6'd24] : 1'b0;
      st_c = 1'b0;
      for (int i = 0; i < W; i++) if (i < int'(p) - 24 && mag[i]) st_c = 1'b1;
    end else begin
      shl = mag << (6'd23 - p);
      m_c = shl[23:0];
    end
    up_c = g_c & (st_c | m_c[0]);
    m_r  = {1'b0, m_c} + {24'd0, up_c};
    e_c  = $signed({6'd0, p}) + $signed({{2{lsb[9]}}, lsb}) + 12'sd127 + (m_r[24] ? 12'sd1 : 12'sd0);
    b = (mag == '0 || e_c <= 12'sd0) ? 32'd0 : {s_c, e_c[7:0], m_r[24] ? m_r[23:1] : m_r[22:0]};
  end

  // ---- fp32 add acc + b (RNE, flush subnormals) ----
  logic [31:0] a, big, sml, sum_f;
  logic        sb_, ss_, op_sub;
  logic [7:0]  eb_, es_;
  logic [23:0] mb_, ms_;
  logic [26:0] mbx, msx, msh;
  logic        sticky;
  logic [7:0]  d;
  logic [27:0] r;
  logic [4:0]  pr;
  logic [26:0] nrm;
  logic [23:0] mo;
  logic        go, so, upo;
  logic [24:0] mr2;
  logic signed [9:0] eo;
  always_comb begin
    a = acc;
    if ({a[30:0]} >= {b[30:0]}) begin big = a; sml = b; end else begin big = b; sml = a; end
    sb_ = big[31]; ss_ = sml[31]; eb_ = big[30:23]; es_ = sml[30:23];
    mb_ = {(eb_ != 8'd0), big[22:0]};
    ms_ = {(es_ != 8'd0), sml[22:0]};
    mbx = {mb_, 3'b000};
    msx = {ms_, 3'b000};
    d   = eb_ - es_;
    sticky = 1'b0;
    if (d > 8'd26) begin msh = '0; sticky = |msx; end
    else begin
      msh = msx >> d;
      for (int i = 0; i < 27; i++) if (i < int'(d) && msx[i]) sticky = 1'b1;
    end
    msh[0] = msh[0] | sticky;
    op_sub = sb_ ^ ss_;
    r = op_sub ? ({1'b0, mbx} - {1'b0, msh}) : ({1'b0, mbx} + {1'b0, msh});
    pr = 5'd0;
    for (int i = 0; i < 28; i++) if (r[i]) pr = 5'(i);
    nrm = '0; go = 1'b0; so = 1'b0;
    if (pr > 5'd26) begin
      nrm = r[27:1]; so = r[0];
    end else begin
      nrm = 27'(r << (5'd26 - pr));
    end
    mo  = nrm[26:3];
    go  = nrm[2];
    so  = so | nrm[1] | nrm[0];
    upo = go & (so | mo[0]);
    mr2 = {1'b0, mo} + {24'd0, upo};
    eo  = $signed({2'b00, eb_}) + $signed({5'b0, pr}) - 10'sd26 + (mr2[24] ? 10'sd1 : 10'sd0);
    if (r == '0 || eo <= 10'sd0) sum_f = 32'd0;
    else sum_f = {sb_, eo[7:0], mr2[24] ? mr2[23:1] : mr2[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)      acc <= '0;
    else if (valid)  acc <= clear ? b : sum_f;
  end
endmodule : e28_hier_reducer
