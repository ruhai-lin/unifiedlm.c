#!/usr/bin/env bash
# Called at the end of run_index.sh in pods that were already running (2026-11-10): builds the 4of14 subset on
# local disk from the 2of14 copy plus the new shards if the PVC has it, then works through the given plans.
set -euo pipefail
cd /scratch/src
SRC=/workspace/data/FineWebEdu_4of14
D=/scratch/data/FineWebEdu_4of14
if [ -s $SRC/SHA256SUMS ] && [ ! -e $D/.ok ]; then
  rm -rf $D && mkdir -p $D
  for f in /scratch/data/FineWebEdu_2of14/*.bin; do b=$(basename $f); [ "$b" = 001_00000_008.bin ] && b=val_$b; ln -s $f $D/$b; done
  for f in $SRC/00[2-3]_*.bin; do cp $f $D/; done
  (cd $D && sha256sum -c --quiet $SRC/SHA256SUMS)
  touch $D/.ok
fi
for P in "$@"; do
  if [[ $P == fw110m* ]] && [ ! -e $D/.ok ]; then echo "skipping $P: FineWebEdu_4of14 is not ready"; continue; fi
  bash scripts/nrp/run_queue.sh "$P"
done
