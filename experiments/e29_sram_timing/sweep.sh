#!/bin/bash
# sequential sweep, one ngspice run at a time through safe_run (machine-wide lock). usage: sweep.sh "40 30 ..." [extra args]
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate cim; export TMPDIR=~/tmp; cd "$(dirname "$0")/../.."
P="$1"; shift
for p in $P; do
  python scripts/safe_run.py --max-rss-mb 3200 -- python experiments/e29_sram_timing/run.py --period-ns $p "$@"
done
