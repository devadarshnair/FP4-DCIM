# OpenSTA: worst setup path of a mapped netlist (run inside the openlane2 image).
# Env: NETLIST TOP LIB SRAM_LIB CLK_PERIOD_NS SDC OUT  -> OUT (report_checks + worst slack)
read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
report_checks -path_delay max -digits 3 > $::env(OUT)
report_worst_slack -max -digits 3 >> $::env(OUT)
exit
