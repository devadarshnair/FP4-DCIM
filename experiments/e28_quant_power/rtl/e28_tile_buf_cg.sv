// e28 / T2.3 [EST] clock-gated variant of e28_tile_buf: one cim_icg per entry, so only the 136 flops of the entry being
// written are clocked (the macro RTL gates its static registers the same way). Same ports/behaviour as e28_tile_buf.
module e28_tile_buf_cg #(parameter int TILE = 8, parameter int WIDTH = 136) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             wr,
  input  logic [WIDTH-1:0] din,
  output logic [WIDTH-1:0] dout
);
  localparam int DEPTH = 2 * TILE;
  localparam int AW = $clog2(DEPTH);
  logic [WIDTH-1:0] mem [DEPTH];
  logic [AW-1:0] wp;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wp <= '0;
    else if (wr) wp <= (wp == AW'(DEPTH - 1)) ? '0 : wp + 1'b1;
  end
  for (genvar e = 0; e < DEPTH; e++) begin : g_ent
    logic gclk;
    cim_icg u_icg (.clk(clk), .en(wr && (wp == AW'(e))), .gclk(gclk));
    always_ff @(posedge gclk) mem[e] <= din;
  end
  logic [AW-1:0] rp;
  assign rp = wp + AW'(TILE);
  assign dout = mem[rp];
endmodule : e28_tile_buf_cg
