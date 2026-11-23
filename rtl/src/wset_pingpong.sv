// Ping-pong weight-set store: one OpenRAM macro (4 columns x M sets) + two active-weight
// buffers. The macro computes from buf[sel] while `pf_req` prefetches the next set into buf[~sel]
// through both SRAM ports (8 read cycles). `use_set` selects the set needed by the next op:
//   - equal to the active set            -> ready, no action
//   - equal to the prefetched (shadow) set -> swap buffers in 1 cycle (no SRAM access)
//   - otherwise                           -> demand load into the shadow buffer, then swap (stall)
// SRAM writes (write-while-compute) are accepted whenever no prefetch/load is reading the macro.
module wset_pingpong #(
  parameter int NCOL = 4,
  parameter int M    = 16
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  wr_en,
  input  logic [7:0]            wr_addr,
  input  logic [31:0]           wr_data,
  output logic                  wr_ready,      // write accepted this cycle
  input  logic                  pf_req,        // prefetch set pf_set into the shadow buffer
  input  logic [$clog2(M)-1:0]  pf_set,
  input  logic [$clog2(M)-1:0]  use_set,       // set required by the next op
  input  logic                  use_valid,
  output logic                  ready,         // w_active holds use_set
  output logic [NCOL*128-1:0]   w_active,
  output logic [31:0]           demand_loads,  // counters for the power/latency analysis
  output logic [31:0]           swaps
);
  localparam int WORDS = NCOL * 4;
  localparam int STEPS = WORDS / 2;

  logic [NCOL*128-1:0]   buf0, buf1;
  logic                  sel;                 // active buffer
  logic                  act_valid, sh_valid; // buffer contents valid
  logic [$clog2(M)-1:0]  act_set, sh_set;
  logic                  reading;
  logic [3:0]            step;
  logic [$clog2(M)-1:0]  rd_set;
  logic                  cap_valid;
  logic [3:0]            cap_step;
  logic [31:0]           d0, d1;
  logic [7:0]            a0, a1;
  logic                  csb0, web0, csb1;

  assign w_active = sel ? buf1 : buf0;
  assign ready    = use_valid && act_valid && (act_set == use_set);
  assign wr_ready = !reading;
  assign a0   = reading ? 8'(rd_set * WORDS + 2 * step) : wr_addr;
  assign a1   = 8'(rd_set * WORDS + 2 * step + 1);
  assign csb0 = ~((reading && step < 4'(STEPS)) || (!reading && wr_en));
  assign web0 = reading ? 1'b1 : ~wr_en;
  assign csb1 = ~(reading && step < 4'(STEPS));

`ifdef SYNTHESIS
  `define CIM_SRAM_PARAMS
`else
  `define CIM_SRAM_PARAMS #(.VERBOSE(0))
`endif
  sky130_sram_1kbyte_1rw1r_32x256_8 `CIM_SRAM_PARAMS u_sram (
    .clk0(clk), .csb0(csb0), .web0(web0), .wmask0(4'hF), .addr0(a0), .din0(wr_data), .dout0(d0),
    .clk1(clk), .csb1(csb1), .addr1(a1), .dout1(d1)
  );

  // which action does the next op need?
  logic need_swap, need_demand;
  assign need_swap   = use_valid && !ready && sh_valid && (sh_set == use_set) && !reading;
  assign need_demand = use_valid && !ready && !(sh_valid && sh_set == use_set) && !reading;

  // shadow-buffer capture (clock-gated: only while reading)
  logic clk_cap;
  cim_icg u_icg_cap (.clk(clk), .en(cap_valid | ~rst_n), .gclk(clk_cap));
  always_ff @(posedge clk_cap or negedge rst_n) begin
    if (!rst_n) begin
      buf0 <= '0; buf1 <= '0;
    end else if (cap_valid) begin
      if (sel) begin buf0[cap_step*64 +: 32] <= d0; buf0[cap_step*64 + 32 +: 32] <= d1; end
      else     begin buf1[cap_step*64 +: 32] <= d0; buf1[cap_step*64 + 32 +: 32] <= d1; end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sel <= 1'b0; act_valid <= 1'b0; sh_valid <= 1'b0; act_set <= '0; sh_set <= '0; reading <= 1'b0;
      step <= '0; rd_set <= '0; cap_valid <= 1'b0; cap_step <= '0; demand_loads <= '0; swaps <= '0;
    end else begin
      cap_valid <= reading && (step < 4'(STEPS));
      cap_step  <= step;
      if (reading) begin
        if (step == 4'(STEPS)) begin            // last capture lands this cycle
          reading <= 1'b0; sh_valid <= 1'b1; sh_set <= rd_set;
        end else step <= step + 1'b1;
      end else if (need_swap) begin             // shadow already holds the needed set
        sel <= ~sel; act_set <= sh_set; act_valid <= 1'b1; sh_valid <= 1'b0; swaps <= swaps + 1;
      end else if (need_demand) begin           // stall: read the needed set into the shadow buffer
        reading <= 1'b1; step <= '0; rd_set <= use_set; sh_valid <= 1'b0; demand_loads <= demand_loads + 1;
      end else if (pf_req && !(sh_valid && sh_set == pf_set) && !(act_valid && act_set == pf_set)) begin
        reading <= 1'b1; step <= '0; rd_set <= pf_set; sh_valid <= 1'b0;
      end
    end
  end
endmodule : wset_pingpong
