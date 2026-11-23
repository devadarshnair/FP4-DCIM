// e28 / T2.3 measurement model of the anchor tile buffer (docs/spec/numerics.md, WP-68): activation codes + exponent
// of every block of a K-tile are held until the tile anchor is known, so the buffer must hold one tile while the next
// one is written (ping-pong): 2 x TILE entries of 32*4 + 8 = 136 bits, written once and read once per op, flop based
// (Yosys maps the memory to registers + read mux). [EST] organisation: the repo has no RTL for this buffer; this is the
// minimal register-file realisation of the recorded spec; the read port follows the write pointer by TILE slots.
module e28_tile_buf #(parameter int TILE = 8, parameter int WIDTH = 136) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             wr,
  input  logic [WIDTH-1:0] din,
  output logic [WIDTH-1:0] dout        // slot wp+TILE (the same position of the previous tile)
);
  localparam int DEPTH = 2 * TILE;
  localparam int AW = $clog2(DEPTH);
  logic [WIDTH-1:0] mem [DEPTH];
  logic [AW-1:0] wp;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wp <= '0;
    else if (wr) wp <= (wp == AW'(DEPTH - 1)) ? '0 : wp + 1'b1;
  end
  always_ff @(posedge clk) begin
    if (wr) mem[wp] <= din;
  end
  logic [AW-1:0] rp;
  assign rp = wp + AW'(TILE);                  // wraps modulo DEPTH (power of two)
  assign dout = mem[rp];
endmodule : e28_tile_buf
