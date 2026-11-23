# Magic parasitic-capacitance extraction of the OpenRAM dual-port bitcell used in
# sky130_sram_1kbyte_1rw1r_32x256_8 (read from the macro GDS shipped in the PDK).
# Env: GDS (macro gds), CELL (cell to extract), OUT (work dir). Run inside the openlane2 image:
#   magic -dnull -noconsole -rcfile $PDK/libs.tech/magic/sky130A.magicrc extract_cell.tcl
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
ext2spice hierarchy off
ext2spice scale off
ext2spice subcircuit top on
ext2spice -o $::env(OUT)/$::env(CELL).ext.spice
quit -noprompt
