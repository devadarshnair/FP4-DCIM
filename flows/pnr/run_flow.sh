#!/bin/bash
# Driver for the e24 P&R runs (same recipe as the v2 t14 run). Each step is its own safe_run job.
# usage: flows/pnr/run_flow.sh <design: bs_int5|dpu|v2_sram_m16> <tag> <phase: flow|resume|drc|power|save>
# flow  : generate config (14 ns, die 900x700, DRC+LVS flags) and run OpenLane2 up to the (expected) Magic.WriteLEF fatal
# resume: -F Magic.SpiceExtraction (steps 57-65; LVS/hold deferred errors make rc=2 as in v2)
# drc   : manual KLayout DRC on the Magic-streamed GDS with -rd top_cell
# power : post-route power (VCD from synthesis netlist, rebased real0) + the three STA variants
set -u
D=$1; TAG=$2; PH=$3
R=$(cd "$(dirname "$0")/../.." && pwd); B=$R/build/pnr/$D; T=$B/runs/$TAG
case $D in bs_int5) TOP=bs_column_top;; dpu) TOP=dpu_column;; v2_sram_m16) TOP=column_top;; esac
source ~/miniconda3/etc/profile.d/conda.sh; conda activate cim; export TMPDIR=~/tmp; cd $R
SR="python scripts/safe_run.py --max-rss-mb 4500 --"
DK="docker run --rm --memory=4000m --memory-swap=4000m -v $R:$R -v $HOME/.ciel:$HOME/.ciel"
case $PH in
flow)
  mkdir -p $B
  python flows/pnr/gen_config.py $D $B/config.json --clk-ns 14 --drc --lvs ${DIE:-}
  $SR $DK -w $B -u $(id -u):$(id -g) ghcr.io/efabless/openlane2:2.3.10 openlane --pdk-root $R/build/pnr/pdkroot --manual-pdk --pdk sky130A --run-tag $TAG $B/config.json ;;
resume)
  $SR $DK -w $B -u $(id -u):$(id -g) ghcr.io/efabless/openlane2:2.3.10 openlane --pdk-root $R/build/pnr/pdkroot --manual-pdk --pdk sky130A --run-tag $TAG -F Magic.SpiceExtraction $B/config.json ;;
drc)
  mkdir -p $T/drc_manual
  $SR $DK -w $T/drc_manual -u $(id -u):$(id -g) --entrypoint klayout ghcr.io/efabless/openlane2:2.3.10 -b -zz -r $R/build/pnr/pdkroot/sky130A/libs.tech/klayout/drc/sky130A_mr.drc -rd input=$T/53-magic-streamout/$TOP.gds -rd top_cell=$TOP -rd report=$T/drc_manual/drc.xml -rd feol=true -rd beol=true -rd floating_metal=false -rd offgrid=true -rd seal=true -rd threads=2 ;;
power)
  mkdir -p $B/postroute
  $SR python flows/pnr/post_route_power.py $T $B/postroute --design $D &&
  $SR python flows/pnr/sta_variants.py $T $B/postroute $D ;;
save)
  python flows/pnr/save_results.py $D $TAG ;;
esac
