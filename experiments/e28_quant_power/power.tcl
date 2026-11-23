# e28 OpenSTA power from a gate-level VCD window (openlane2 image). Env: NETLIST TOP LIB SDC CLK_PERIOD_NS VCD SCOPE OUT
read_liberty $::env(LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
read_vcd -scope $::env(SCOPE) $::env(VCD)
report_power -digits 6 > $::env(OUT)
exit
