// Integrated clock gate for static registers (weight-exponent RF, anchors, active weights, latch store).
// Synthesis: the sky130 ICG cell (latch + AND, glitch-free). Simulation: equivalent latch-based model.
module cim_icg (
  input  logic clk,
  input  logic en,
  output logic gclk
);
`ifdef SYNTHESIS
  sky130_fd_sc_hd__dlclkp_1 u_icg (.CLK(clk), .GATE(en), .GCLK(gclk));
`else
  logic en_l;
  always_latch if (!clk) en_l = en;
  assign gclk = clk & en_l;
`endif
endmodule : cim_icg
