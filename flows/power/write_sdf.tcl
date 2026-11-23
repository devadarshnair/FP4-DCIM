# Emit SDF (cell delays, no wire parasitics) for a mapped netlist. Env: NETLIST TOP LIB SRAM_LIB SDC CLK_PERIOD_NS OUT
read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
write_sdf -divider . -digits 4 -include_typ $::env(OUT)
exit
