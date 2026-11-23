#!/bin/bash
# T3.1 signoff P&R driver (OpenLane2 2.3.10, same PDK shim / ICG stub as e24).
# Same die (1200x1000; 1000x900 failed: DPU GRT-0118 at density 55, GPL-0302 at 45) and macro origin (60,60) for all three tiles; hold margin PL 0.1 / GRT 0.3 ns (PL 0.3 overflowed DPU global routing); density 45 % (55 % congested DPU GRT-0118 twice, s1/s2); in-flow KLayout DRC + Netgen LVS.
# usage: flows/pnr/run_t31.sh <bs_int5|dpu|v2_sram_m16> <tag> <a|b>
#   CLK=<ns> overrides the 14 ns clock (T3.2 pushed-clock fmax runs; same recipe otherwise)
#   HOLD=<ns> overrides the GRT hold slack margin 0.3 (T4.3 job D: DPU 10.5 ns had 4 nom_tt hold violations, WS -0.011 ns,
#     at 0.3; PL margin stays 0.1). Any HOLD != 0.3 is a recipe deviation from v2/INT5 and is recorded in the run config.
#   a: config + flow up to Magic.StreamOut            b: resume to the end (skips Magic.WriteLEF and the LEF-antenna check that needs its output; which cannot read the OpenRAM GDS) -> final/
set -u
D=$1; TAG=$2; PH=$3
R=$(cd "$(dirname "$0")/../.." && pwd); B=$R/build/pnr/t31/$D
if [ -z "${CIM_REMOTE:-}" ]; then source ~/miniconda3/etc/profile.d/conda.sh; conda activate cim; fi; export TMPDIR=~/tmp; cd $R
# CIM_REMOTE=1 (remote run): env already on PATH and no shared local lane, so no safe_run wrapper
if [ -n "${CIM_REMOTE:-}" ]; then SR=""; else SR="python scripts/safe_run.py --lane cpu --max-rss-mb 4500 --"; fi
OL="docker run --rm --memory=4000m --memory-swap=4000m -v $R:$R -v $HOME/.ciel:$HOME/.ciel -w $B -u $(id -u):$(id -g) ghcr.io/efabless/openlane2:2.3.10 openlane --pdk-root $R/build/pnr/pdkroot --manual-pdk --pdk sky130A --run-tag $TAG"
case $PH in
a) mkdir -p $B
   python flows/pnr/gen_config.py $D $B/config.json --clk-ns ${CLK:-14} --drc --lvs --die ${DIEW:-1200} ${DIEH:-1000} --macro-at 60 60 --hold-margin ${HOLD:-0.3} --pl-hold-margin 0.1 --signoff --density ${DENS:-45}
   $SR $OL -T Magic.StreamOut $B/config.json ;;
b) $SR $OL -F ${FROM:-KLayout.StreamOut} -S Magic.WriteLEF -S Odb.CheckDesignAntennaProperties $B/config.json ;;
esac
