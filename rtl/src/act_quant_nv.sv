// Online NVFP4 activation quantizer, "hw" convention of cimfp4.formats.nv_quant:
//   scale_b = E4M3_RNE_satfinite(amax_b * c)      (exact product; c = fp32(s_enc/6) config register)
//   q_i     = RNE_E2M1(|x_i| / (scale_b * s_dec)) by exact comparison against the 7 midpoints
//             (n/4)*u*S*2^(k+q), n in {1,3,5,7,10,14,20}; ties -> even code.
// Inputs are finite bf16 (the NV quantizer requires finite values); c and s_dec are normal fp32.
module act_quant_nv (
  input  logic [16*16-1:0] x_bf16,
  /* verilator lint_off UNUSEDSIGNAL */      // sign bits of the (positive) scale registers are unused
  input  logic [31:0]      c_fp32,      // s_enc / 6
  input  logic [31:0]      sdec_fp32,   // per-tensor decode scale
  /* verilator lint_on UNUSEDSIGNAL */
  output logic [16*4-1:0]  codes,
  output logic [7:0]       scale_e4m3
);
  // block amax
  logic [14:0] amax;
  always_comb begin
    amax = '0;
    for (int i = 0; i < 16; i++) if (x_bf16[16*i +: 15] > amax) amax = x_bf16[16*i +: 15];
  end
  // exact product amax * c = N * 2^f
  logic [7:0]         a_sig, ea;
  logic [23:0]        c_sig;
  logic signed [11:0] f;
  logic [31:0]        n;
  assign ea    = amax[14:7];
  assign a_sig = {(ea != 0), amax[6:0]};
  assign c_sig = {1'b1, c_fp32[22:0]};
  assign f     = ((ea == 0) ? -12'sd126 : ($signed({4'd0, ea}) - 12'sd127)) - 12'sd7
               + ($signed({4'd0, c_fp32[30:23]}) - 12'sd127) - 12'sd23;
  assign n     = a_sig * c_sig;
  // RNE to E4M3 (u * 2^k, u in 0..15), satfinite
  logic [5:0]         l;              // msb position of n
  logic signed [11:0] e_val, k;
  logic signed [11:0] sh;             // n >> sh gives u
  /* verilator lint_off UNUSEDSIGNAL */
  logic [63:0]        n64, q, rem, half;   // only q[4:0] is the rounded significand
  /* verilator lint_on UNUSEDSIGNAL */
  logic [4:0]         u_r;
  logic signed [11:0] k_r;
  always_comb begin
    l = 6'd0;
    for (int b = 0; b < 32; b++) if (n[b]) l = 6'(b);
    e_val = $signed({6'd0, l}) + f;              // value in [2^e_val, 2^(e_val+1))
    k     = (e_val >= -12'sd6) ? e_val - 12'sd3 : -12'sd9;
    sh    = k - f;
    n64   = {32'd0, n};
    if (sh <= 0) begin
      q = n64 << (-sh); rem = '0; half = '0;
    end else if (sh > 40) begin
      q = '0; rem = n64; half = 64'd1 << 40;     // far below half an ulp: rounds to 0
    end else begin
      q    = n64 >> sh;
      rem  = n64 & ((64'd1 << sh) - 1);
      half = 64'd1 << (sh - 1);
    end
    u_r = 5'(q);
    if (sh > 0 && (rem > half || (rem == half && q[0]))) u_r = u_r + 5'd1;   // RNE
    k_r = k;
    if (u_r == 5'd16) begin u_r = 5'd8; k_r = k + 12'sd1; end                // carry into next binade
  end
  logic [3:0] e_enc;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [3:0] m_enc;                       // 3-bit mantissa field; bit 3 is always 0
  /* verilator lint_on UNUSEDSIGNAL */
  always_comb begin
    if (n == 32'd0 || u_r == 5'd0) begin
      e_enc = 4'd0; m_enc = 4'd0;
    end else if (u_r >= 5'd8) begin
      if (k_r + 12'sd10 > 12'sd15 || (k_r + 12'sd10 == 12'sd15 && u_r > 5'd14)) begin
        e_enc = 4'd15; m_enc = 4'd6;              // satfinite: 448
      end else begin
        e_enc = 4'(k_r + 12'sd10); m_enc = 4'(u_r - 5'd8);
      end
    end else begin
      e_enc = 4'd0; m_enc = 4'(u_r);              // subnormal (k = -9)
    end
  end
  assign scale_e4m3 = {1'b0, e_enc[3:0], m_enc[2:0]};

  // element thresholds (exact)
  logic [3:0]         u_s;
  logic signed [11:0] k_s, qd;
  logic [23:0]        s_sig;
  assign u_s   = (e_enc == 4'd0) ? {1'b0, m_enc[2:0]} : {1'b1, m_enc[2:0]};
  assign k_s   = ((e_enc == 4'd0) ? 12'sd1 : $signed({8'd0, e_enc})) - 12'sd10;
  assign s_sig = {1'b1, sdec_fp32[22:0]};
  assign qd    = $signed({4'd0, sdec_fp32[30:23]}) - 12'sd127 - 12'sd23;
  localparam int NM [7] = '{1, 3, 5, 7, 10, 14, 20};
  for (genvar i = 0; i < 16; i++) begin : g_el
    logic [7:0]         xe, xs;
    logic signed [11:0] d;
    logic [2:0]         idx;
    assign xe = x_bf16[16*i+7 +: 8];
    assign xs = {(xe != 0), x_bf16[16*i +: 7]};
    // compare 4*|x| = xs*2^(ex+2) with n*u*S*2^(k+q):  xs*2^d  vs  n*u*S,  d = ex + 2 - k - q
    assign d  = ((xe == 0) ? -12'sd126 : ($signed({4'd0, xe}) - 12'sd127)) - 12'sd7 + 12'sd2 - k_s - qd;
    always_comb begin
      logic [95:0] lhs, rhs;
      idx = 3'd0;
      for (int j = 0; j < 7; j++) begin
        lhs = {88'd0, xs};
        rhs = 96'(NM[j]) * 96'(u_s) * 96'(s_sig);
        if (d >= 0) lhs = (d > 60) ? {1'b1, 95'd0} : lhs << d;
        else        rhs = (-d > 60) ? {1'b1, 95'd0} : rhs << (-d);
        if (lhs > rhs || (lhs == rhs && (j % 2 == 1))) idx = idx + 3'd1;
      end
      if (xs == 8'd0 || u_s == 4'd0) idx = 3'd0;
    end
    assign codes[4*i +: 4] = {x_bf16[16*i+15] & (idx != 3'd0), idx};
  end
endmodule : act_quant_nv
