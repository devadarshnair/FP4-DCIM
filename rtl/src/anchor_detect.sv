// Tile-anchor detector: streams the activation block exponents of one K-tile (one per
// 32-row op, in tile order) and outputs max_e_x of the tile when `last` is seen. The window LSB of
// every macro pass depends only on this tile-local maximum (AccumMode.Hier), so the activation codes
// of one tile must be buffered until the anchor is known: latency = tile_blocks cycles, buffer =
// tile_blocks x (32 x 4 + 8) bits (recorded in docs/spec/numerics.md).
module anchor_detect (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              valid,
  input  logic signed [7:0] e_x,
  input  logic              last,          // final block of the tile
  output logic signed [7:0] anchor,
  output logic              anchor_valid
);
  logic signed [7:0] run_max;
  logic              have;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      run_max <= '0; have <= 1'b0; anchor <= '0; anchor_valid <= 1'b0;
    end else begin
      anchor_valid <= 1'b0;
      if (valid) begin
        if (last) begin
          anchor       <= (!have || e_x > run_max) ? e_x : run_max;
          anchor_valid <= 1'b1;
          have         <= 1'b0;
        end else begin
          run_max <= (!have || e_x > run_max) ? e_x : run_max;
          have    <= 1'b1;
        end
      end
    end
  end
endmodule : anchor_detect
