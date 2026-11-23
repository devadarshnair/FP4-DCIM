// Online MXFP4 activation quantizer: one 32-element bf16 block per cycle ->
// 32 E2M1 codes + UE8M0 scale, bit-identical to cimfp4.formats.mx_quant.quantize_mxfp4.
//   E   = floor(log2 amax) (exact, incl. bf16 subnormals)
//   X   = clamp(E - 2 + adj, -127, 127);  adj = [1.m >= 1.75] (OCP v1.1 expAdj) or [1.m > 1.5] (RCEIL)
//   q_i = RNE_E2M1(|x_i| / 2^X), saturating at 6. All-zero block -> X = -127; NaN/Inf -> scale 0xFF, codes 0.
// The element value relative to 2^X is formed exactly with left shifts only (units of 2^-10).
module act_quant_mx (
  input  logic [32*16-1:0] x_bf16,     // element i at [16i +: 16]
  input  logic             mode_rceil, // 0: OCP v1.1 (expAdj), 1: RCEIL
  output logic [32*4-1:0]  codes,
  output logic [7:0]       scale_code, // UE8M0 (X + 127), 0xFF for NaN
  output logic signed [7:0] scale_exp, // X
  output logic             nan
);
  localparam int N = 32;
  // block amax over the 15-bit magnitude fields
  logic [14:0] mag [N];
  logic [N-1:0] nonfinite;
  for (genvar i = 0; i < N; i++) begin : g_mag
    assign mag[i] = x_bf16[16*i +: 15];
    assign nonfinite[i] = (x_bf16[16*i+7 +: 8] == 8'hFF);
  end
  logic [14:0] amax;
  always_comb begin
    amax = '0;
    for (int i = 0; i < N; i++) if (!nonfinite[i] && mag[i] > amax) amax = mag[i];
  end
  assign nan = |nonfinite;

  // E = floor(log2 amax) and the normalized fraction of amax
  logic [7:0]        ea;
  logic [6:0]        ma, frac;
  logic signed [9:0] e_floor;
  logic [2:0]        p;          // msb position of a subnormal mantissa
  assign ea = amax[14:7];
  assign ma = amax[6:0];
  always_comb begin
    p = 3'd0;
    for (int b = 0; b < 7; b++) if (ma[b]) p = 3'(b);
    if (ea != 8'd0) begin
      e_floor = $signed({2'b00, ea}) - 10'sd127;
      frac    = ma;
    end else begin
      e_floor = $signed({7'd0, p}) - 10'sd133;   // value = m * 2^-133
      frac    = 7'(ma << (3'd7 - p));             // bits below the msb, left-aligned
    end
  end
  logic adj;
  assign adj = mode_rceil ? (frac[6] & (|frac[5:0])) : (frac[6] & frac[5]);
  logic signed [9:0] x_raw, x_cl;
  assign x_raw = e_floor - 10'sd2 + {9'd0, adj};
  assign x_cl  = (amax == 15'd0) ? -10'sd127 : (x_raw < -10'sd127) ? -10'sd127 : (x_raw > 10'sd127) ? 10'sd127 : x_raw;
  assign scale_exp  = x_cl[7:0];
  assign scale_code = nan ? 8'hFF : 8'(x_cl + 10'sd127);

  // per element: exact |x|/2^X in units of 2^-10, RNE to the E2M1 grid
  for (genvar i = 0; i < N; i++) begin : g_q
    logic [7:0]        e, sig;
    logic signed [10:0] sh;
    logic [12:0]       v;
    logic [2:0]        idx;
    assign e   = x_bf16[16*i+7 +: 8];
    assign sig = {(e != 8'd0), x_bf16[16*i +: 7]};
    // value = sig * 2^(eraw), eraw = (e==0 ? -126 : e-127) - 7;  v = value * 2^(10 - X)
    assign sh  = ((e == 8'd0) ? -11'sd126 : ($signed({3'b000, e}) - 11'sd127)) - 11'sd7 - $signed({x_cl[9], x_cl}) + 11'sd10;
    assign v   = (sh < 0 || sh > 5) ? 13'd0 : 13'({5'd0, sig} << sh[2:0]);
    always_comb begin
      if (sig == 8'd0 || sh < 0)        idx = 3'd0;   // zero, or < 0.25 (rounds to 0)
      else if (sh > 5)                  idx = 3'd7;   // >= 8: saturate at 6
      else if (v <  13'd256)            idx = 3'd0;   // < 0.25
      else if (v == 13'd256)            idx = 3'd0;   // 0.25 tie -> 0 (even)
      else if (v <  13'd768)            idx = 3'd1;   // 0.5
      else if (v == 13'd768)            idx = 3'd2;   // 0.75 tie -> 1.0 (even)
      else if (v <  13'd1280)           idx = 3'd2;   // 1.0
      else if (v == 13'd1280)           idx = 3'd2;   // 1.25 tie -> 1.0
      else if (v <  13'd1792)           idx = 3'd3;   // 1.5
      else if (v == 13'd1792)           idx = 3'd4;   // 1.75 tie -> 2.0
      else if (v <  13'd2560)           idx = 3'd4;   // 2.0
      else if (v == 13'd2560)           idx = 3'd4;   // 2.5 tie -> 2.0
      else if (v <  13'd3584)           idx = 3'd5;   // 3.0
      else if (v == 13'd3584)           idx = 3'd6;   // 3.5 tie -> 4.0
      else if (v <  13'd5120)           idx = 3'd6;   // 4.0
      else if (v == 13'd5120)           idx = 3'd6;   // 5 tie -> 4.0
      else                              idx = 3'd7;   // 6.0 (saturating)
    end
    assign codes[4*i +: 4] = nan ? 4'd0 : {x_bf16[16*i+15] & (idx != 3'd0), idx};
  end
endmodule : act_quant_mx
