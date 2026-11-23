// Weight-set store backed by one sky130 OpenRAM macro: 4 columns x M=16 sets x 32 rows x 4 b
// = 8192 b = sky130_sram_1kbyte_1rw1r_32x256_8. Word address = set*16 + col*4 + k; word k of a
// column holds rows 8k..8k+7. A set switch reads 16 words through both ports (RW + R) in 8 cycles
// into the active weight latches (single buffer: the column stalls while busy; see wset_pingpong for the double-buffered variant).
module wset_sram #(
  parameter int NCOL = 4,
  parameter int M    = 16
) (
  input  logic                    clk,
  input  logic                    rst_n,
  // external write port (shares SRAM port 0; ignored while a load is in progress)
  input  logic                    wr_en,
  input  logic [7:0]              wr_addr,
  input  logic [31:0]             wr_data,
  // set-switch request
  input  logic                    load_req,
  input  logic [$clog2(M)-1:0]    load_set,
  output logic                    busy,
  output logic [NCOL*32*4-1:0]    w_active
);
  localparam int WORDS = NCOL * 4;            // 16 words per set per group
  localparam int STEPS = WORDS / 2;           // 8 dual-port read cycles

  logic [3:0]  step;                          // 0..STEPS issue, STEPS = last capture
  logic        loading, cap_valid;
  logic [3:0]  cap_step;
  logic [$clog2(M)-1:0] set_q;
  logic [7:0]  a0, a1;
  logic [31:0] d0, d1;
  logic        csb0, web0, csb1;

  assign a0   = loading ? 8'(set_q * WORDS + 2 * step)     : wr_addr;
  assign a1   = 8'(set_q * WORDS + 2 * step + 1);
  assign csb0 = ~((loading && step < 4'(STEPS)) || (!loading && wr_en));
  assign web0 = loading ? 1'b1 : ~wr_en;
  assign csb1 = ~(loading && step < 4'(STEPS));
  assign busy = loading;

`ifdef SYNTHESIS
  `define CIM_SRAM_PARAMS
`else
  `define CIM_SRAM_PARAMS #(.VERBOSE(0))  // silence the behavioural model (not a macro parameter)
`endif
  sky130_sram_1kbyte_1rw1r_32x256_8 `CIM_SRAM_PARAMS u_sram (
    .clk0(clk), .csb0(csb0), .web0(web0), .wmask0(4'hF), .addr0(a0), .din0(wr_data), .dout0(d0),
    .clk1(clk), .csb1(csb1), .addr1(a1), .dout1(d1)
  );

  // active weights are clock-gated: they change only while a set is being captured
  logic clk_act;
  cim_icg u_icg_act (.clk(clk), .en(cap_valid | ~rst_n), .gclk(clk_act));
  always_ff @(posedge clk_act or negedge rst_n) begin
    if (!rst_n) w_active <= '0;
    else if (cap_valid) begin
      w_active[cap_step * 64 +: 32]      <= d0;
      w_active[cap_step * 64 + 32 +: 32] <= d1;
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      loading <= 1'b0; step <= '0; set_q <= '0; cap_valid <= 1'b0; cap_step <= '0;
    end else begin
      cap_valid <= loading && (step < 4'(STEPS));
      cap_step  <= step;
      if (!loading && load_req) begin
        loading <= 1'b1; step <= '0; set_q <= load_set;
      end else if (loading) begin
        if (step == 4'(STEPS)) loading <= 1'b0;
        else                   step    <= step + 1'b1;
      end
    end
  end
endmodule : wset_sram
