# P&R / signoff SDC: same constraints as flows/synth/constraints/column.sdc with a fixed period
# (OpenLane2 substitutes $::env(CLOCK_PERIOD) from the flow config; ns).
create_clock -name clk -period $::env(CLOCK_PERIOD) [get_ports clk]
set_input_delay  0.0 -clock clk [delete_from_list [all_inputs] [get_ports {clk rst_n}]]
set_output_delay 0.0 -clock clk [all_outputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 [all_inputs]
set_load 0.005 [all_outputs]
