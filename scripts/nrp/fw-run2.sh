# Self-running training pod for plans fw110m.txt then fw15m_seeds.txt: environment on local disk, FineWebEdu_4of14
# copied off the PVC and checked (2of14 is rebuilt from it as symlinks), a 10-step preflight of 110M S and T plus the
# eval, then the claim queue. Exits when the queue is done.
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev > /dev/null
python -m venv /scratch/venv
/scratch/venv/bin/pip install -q --no-cache-dir torch==2.10.0 triton==3.6.0 numpy==2.2.6 sentencepiece==0.2.1 pyarrow==26.0.0 huggingface-hub==1.33.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
export PATH=/scratch/venv/bin:$PATH
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name())"
mkdir -p /scratch/src && tar -xzf /cfg/src.tgz -C /scratch/src 2>/dev/null && cd /scratch/src
SRC=/workspace/data/FineWebEdu_4of14
D4=/scratch/data/FineWebEdu_4of14
D2=/scratch/data/FineWebEdu_2of14
test -s $SRC/SHA256SUMS
mkdir -p $D4 $D2
cp $SRC/*.bin $D4/
(cd $D4 && sha256sum -c --quiet $SRC/SHA256SUMS)
for f in $D4/00[0-1]_*.bin; do ln -s $f $D2/; done
ln -s $D4/val_001_00000_008.bin $D2/001_00000_008.bin
(cd $D2 && sha256sum -c --quiet /workspace/data/FineWebEdu_2of14/SHA256SUMS)
ln -sfn /scratch/data data
# cards under 40 GB: same tokens per step, half the micro-batch for 110M
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MIB" -lt 40000 ]; then
  sed -i 's/--batch_size=16 --gradient_accumulation_steps=8/--batch_size=8 --gradient_accumulation_steps=16/' scripts/nrp/fw110m.txt
fi
grep -o -- "--batch_size=[0-9]* --gradient_accumulation_steps=[0-9]*" scripts/nrp/fw110m.txt
for line in 1 2; do
  F=$(sed -n "${line}p" scripts/nrp/fw110m.txt | cut -d' ' -f2- | sed 's/--batch_size=[0-9]*/--batch_size=2/; s/--gradient_accumulation_steps=[0-9]*/--gradient_accumulation_steps=1/; s/--max_iters=[0-9]*/--max_iters=10/; s/--eval_interval=[0-9]*/--eval_interval=5/; s/--eval_iters=[0-9]*/--eval_iters=2/; s/--warmup_iters=[0-9]*/--warmup_iters=2/')
  rm -rf /scratch/pre && python -u scripts/train_variant.py --out_dir=/scratch/pre --compile=False $F 2>&1 | tail -4
  python -u scripts/fineweb/eval_fineweb.py --out_dir=/scratch/pre --windows=8 --batch=4 2>&1 | tail -2
done
echo PREFLIGHT OK
bash scripts/nrp/run_queue.sh fw110m.txt fw15m_seeds.txt
echo QUEUE DONE
