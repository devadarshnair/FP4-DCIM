# OpenSTA for macros (openlane2 image). Env: NETLIST TOP LIB SRAM_LIB CLK_PERIOD_NS SDC OUT OUT_R2R
# OUT: worst path of any kind (inputs are ideal buf_2-driven, so write-decode input paths can dominate).
# OUT_R2R: worst register-to-register path (the datapath/compute timing).
read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
report_checks -path_delay max -digits 3 > $::env(OUT)
report_worst_slack -max -digits 3 >> $::env(OUT)
report_checks -path_delay max -digits 3 -from [all_registers -clock_pins] -to [all_registers -data_pins] > $::env(OUT_R2R)
report_worst_slack -max -digits 3 >> $::env(OUT_R2R)
exit
