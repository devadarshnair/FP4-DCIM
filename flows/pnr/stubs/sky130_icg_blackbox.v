// Blackbox for the directly instantiated sky130 ICG (rtl/src/cim_icg.sv under SYNTHESIS).
// OpenLane2's Yosys does not preload std-cell Verilog models; the cell is resolved from the Liberty/LEF later.
(* blackbox *)
module sky130_fd_sc_hd__dlclkp_1 (input CLK, input GATE, output GCLK);
endmodule
