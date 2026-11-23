# OpenSTA power from a gate-level VCD window (run inside the openlane2 image).
# Env: NETLIST TOP LIB SRAM_LIB SDC CLK_PERIOD_NS VCD SCOPE OUT
read_liberty $::env(LIB)
read_liberty $::env(SRAM_LIB)
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
read_vcd -scope $::env(SCOPE) $::env(VCD)
report_power -digits 6 > $::env(OUT)
report_power -instances [get_cells -hierarchical -filter "ref_name =~ sky130_sram*"] -digits 6 >> $::env(OUT)
exit
