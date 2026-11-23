# Hierarchical magic extraction (devices with AD/PD + wiring capacitance) of the whole OpenRAM macro
# from the PDK GDS. Env: GDS, CELL (top), OUT. Output: $OUT/$CELL.ext.spice (one .subckt per layout cell).
cd $::env(OUT)
gds noduplicates true
gds readonly true
gds read $::env(GDS)
load $::env(CELL)
select top cell
extract path $::env(OUT)
extract do capacitance
extract do coupling
extract do local
extract all
ext2spice cthresh 0
ext2spice rthresh infinite
ext2spice hierarchy on
ext2spice scale off
ext2spice subcircuit top on
ext2spice -o $::env(OUT)/$::env(CELL).ext.spice
quit -noprompt
