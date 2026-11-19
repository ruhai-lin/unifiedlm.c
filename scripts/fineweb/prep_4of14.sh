# Builds FineWebEdu_4of14 on the PVC: parquet files 002 and 003 pretokenized by the unmodified fineweb.py, plus the
# 2of14 training shards, plus 2of14's validation shard renamed val_001_00000_008.bin so it sorts last (validation).
# Runs in a CPU Job from a source copy; the 2of14 subset must already be on the PVC.
set -euo pipefail
export HF_HOME=/scratch/hf
mkdir -p data
python -u - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import fineweb
fineweb.PATTERN = "sample/10BT/00[2-3]_*.parquet"
fineweb.DATA_DIR = "data/FineWebEdu_new"
files = fineweb.parquet_files()
print("parquet files:", [os.path.basename(f) for f in files])
assert len(files) == 2, files
fineweb.pretokenize(num_workers=2)
PY
OLD=/workspace/data/FineWebEdu_2of14
NEW=data/FineWebEdu_new
VAL=001_00000_008.bin
test -s $OLD/SHA256SUMS
DST=/workspace/data/FineWebEdu_4of14
rm -rf $DST.tmp && mkdir -p $DST.tmp
cp $NEW/*.bin $DST.tmp/
for f in $OLD/*.bin; do b=$(basename $f); [ "$b" = "$VAL" ] || cp $f $DST.tmp/; done
cp $OLD/$VAL $DST.tmp/val_$VAL
(cd $NEW && sha256sum *.bin) > /scratch/sha_new
grep -v " $VAL\$" $OLD/SHA256SUMS > /scratch/sha_old
grep " $VAL\$" $OLD/SHA256SUMS | sed "s/ $VAL\$/ val_$VAL/" > /scratch/sha_val
cat /scratch/sha_new /scratch/sha_old /scratch/sha_val > /scratch/sha_all
(cd $DST.tmp && sha256sum -c --quiet /scratch/sha_all)
test "$(ls $DST.tmp/*.bin | sort | tail -1)" = "$DST.tmp/val_$VAL"
cp /scratch/sha_all $DST.tmp/SHA256SUMS
rm -rf $DST && mv $DST.tmp $DST
ls $DST | wc -l; du -sh $DST
echo PREP DONE
