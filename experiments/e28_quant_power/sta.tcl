# e28 OpenSTA timing (openlane2 image). Env: NETLIST TOP LIB SDC CLK_PERIOD_NS OUT [OUT_R2R]
read_liberty $::env(LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
report_checks -path_delay max -digits 3 > $::env(OUT)
report_worst_slack -max -digits 3 >> $::env(OUT)
if {[info exists ::env(OUT_R2R)]} {
  report_checks -path_delay max -digits 3 -from [all_registers -clock_pins] -to [all_registers -data_pins] > $::env(OUT_R2R)
  report_worst_slack -max -digits 3 >> $::env(OUT_R2R)
}
exit
