# Common constraints for all 4-column-group designs (WP-27). Period is overridden by the sweep.
create_clock -name clk -period $::env(CLK_PERIOD_NS) [get_ports clk]
set_input_delay  0.0 -clock clk [delete_from_list [all_inputs] [get_ports {clk rst_n}]]
set_output_delay 0.0 -clock clk [all_outputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 [all_inputs]
set_load 0.005 [all_outputs]
