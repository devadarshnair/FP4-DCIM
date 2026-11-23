// Ablation: the same weight-set store built from standard-cell storage (flip-flops/latches after
// synthesis) - same interface as wset_sram; a set switch takes one cycle.
module wset_latch #(
  parameter int NCOL = 4,
  parameter int M    = 16
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    wr_en,
  input  logic [7:0]              wr_addr,
  input  logic [31:0]             wr_data,
  input  logic                    load_req,
  input  logic [$clog2(M)-1:0]    load_set,
  output logic                    busy,
  output logic [NCOL*32*4-1:0]    w_active
);
  localparam int WORDS = NCOL * 4;
  logic [31:0] mem [0:M*WORDS-1];
  logic        loading;
  logic [$clog2(M)-1:0] set_q;
  assign busy = loading;

  logic clk_wr, clk_act;
  cim_icg u_icg_wr  (.clk(clk), .en(wr_en & ~loading), .gclk(clk_wr));
  cim_icg u_icg_act (.clk(clk), .en(loading | ~rst_n), .gclk(clk_act));
  always_ff @(posedge clk_wr) begin
    if (wr_en && !loading) mem[wr_addr[$clog2(M*WORDS)-1:0]] <= wr_data;
  end
  always_ff @(posedge clk_act or negedge rst_n) begin
    if (!rst_n) w_active <= '0;
    else if (loading) for (int i = 0; i < WORDS; i++) w_active[i * 32 +: 32] <= mem[set_q * WORDS + i];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      loading <= 1'b0; set_q <= '0;
    end else if (!loading && load_req) begin
      loading <= 1'b1; set_q <= load_set;
    end else if (loading) begin
      loading <= 1'b0;
    end
  end
endmodule : wset_latch
