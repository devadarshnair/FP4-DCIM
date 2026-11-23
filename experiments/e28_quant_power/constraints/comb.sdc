# e28 (T2.3): combinational blocks (act_quant_mx/nv, int5 conversion). Virtual clock: the block sits between registers of
# its neighbours; paths input->output must fit one period (CLK_PERIOD_NS). Same I/O model as flows/synth/constraints/column.sdc.
create_clock -name vclk -period $::env(CLK_PERIOD_NS)
set_input_delay  0.0 -clock vclk [all_inputs]
set_output_delay 0.0 -clock vclk [all_outputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 [all_inputs]
set_load 0.005 [all_outputs]
