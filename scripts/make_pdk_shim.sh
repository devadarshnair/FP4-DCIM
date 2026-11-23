#!/usr/bin/env bash
# Build the OpenLane2 2.3.10 PDK shim on top of a ciel sky130A install.
# OpenLane2 2.3.10 expects the older OpenLane variable names; the ciel PDK ships LibreLane names.
# Usage: scripts/make_pdk_shim.sh [ciel_home]   (default ~/.ciel)
set -euo pipefail
CIEL=${1:-$HOME/.ciel}
V=1c8019c4421c99348a0ec78f4e969eed6fb5c647
SRC=$CIEL/ciel/sky130/versions/$V/sky130A
[ -d "$SRC" ] || { echo "missing $SRC; run: ciel enable --pdk-family sky130 $V"; exit 1; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
DST=$REPO/build/pnr/pdkroot/sky130A
rm -rf "$DST"; mkdir -p "$DST/libs.tech/openlane"
ln -s "$SRC/SOURCES" "$DST/SOURCES"
ln -s "$SRC/libs.ref" "$DST/libs.ref"
for d in "$SRC"/libs.tech/*; do n=$(basename "$d"); [ "$n" = openlane ] || ln -s "$d" "$DST/libs.tech/$n"; done
ln -s "$SRC/libs.tech" "$DST/libs.tech/libs.tech"
for f in "$SRC"/libs.tech/librelane/*; do n=$(basename "$f"); [ "$n" = config.tcl ] || ln -s "$f" "$DST/libs.tech/openlane/$n"; done
{ echo "# shim: LibreLane-format PDK (ciel) used with OpenLane2 2.3.10, which requires the older variable names"
  echo "source $SRC/libs.tech/librelane/config.tcl"
  cat "$REPO/flows/pdk/openlane_shim_config.tail.tcl"; } > "$DST/libs.tech/openlane/config.tcl"
echo "shim ready: $DST"
