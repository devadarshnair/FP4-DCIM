Y=../yenv/bin/yosys
for top in v0_full v0_struct v1_rowdec v2_bcastX v2_static wdec v3_split v4_onescomp v5_bitserial; do
  best=99999; bestg=""
  for g in cmos2 cmos3 cmos4 "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX,AOI3,OAI3,AOI4,OAI4"; do
   for pre in "synth -flatten -top $top" "synth -flatten -top $top; abc -g AND; opt; abc -g AND"; do
   t=$($Y -q -p "read_verilog mult.v; $pre; abc -g $g; opt_clean; tee -o /dev/stdout stat -tech cmos" 2>&1 | grep "transistors" | awk '{print $NF}')
   if [ -n "$t" ] && [ "$t" -lt "$best" ]; then best=$t; bestg=$g; fi
   done
  done
  echo "$top: best transistors=$best  (~$((best/4)) NAND2-eq) lib=${bestg:0:10}"
done
