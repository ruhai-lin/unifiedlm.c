# Self-running training pod for plans fw42m.txt then fw15m.txt: environment on local disk, FineWeb subset copied off the PVC and
# checked, a 10-step preflight of S and T plus the eval, then the claim queue. Exits when the queue is done.
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev > /dev/null
python -m venv /scratch/venv
/scratch/venv/bin/pip install -q --no-cache-dir torch==2.10.0 triton==3.6.0 numpy==2.2.6 sentencepiece==0.2.1 pyarrow==26.0.0 huggingface-hub==1.33.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
export PATH=/scratch/venv/bin:$PATH
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name())"
mkdir -p /scratch/src && tar -xzf /cfg/src.tgz -C /scratch/src 2>/dev/null && cd /scratch/src
SRC=/workspace/data/FineWebEdu_2of14
test -s $SRC/SHA256SUMS
mkdir -p /scratch/data/FineWebEdu_2of14
cp $SRC/*.bin /scratch/data/FineWebEdu_2of14/
(cd /scratch/data/FineWebEdu_2of14 && sha256sum -c --quiet $SRC/SHA256SUMS)
ln -sfn /scratch/data data
ls data/FineWebEdu_2of14 | wc -l
# 24 GB cards: same tokens per step, half the micro-batch (42M S at batch 32 peaks at 22 GiB)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MIB" -lt 40000 ]; then
  sed -i 's/--batch_size=32 --gradient_accumulation_steps=4/--batch_size=16 --gradient_accumulation_steps=8/' scripts/nrp/fw42m.txt
fi
grep -o -- "--batch_size=[0-9]* --gradient_accumulation_steps=[0-9]*" scripts/nrp/fw42m.txt
for line in 1 2; do
  F=$(sed -n "${line}p" scripts/nrp/fw42m.txt | cut -d' ' -f2- | sed 's/--batch_size=[0-9]*/--batch_size=4/; s/--gradient_accumulation_steps=[0-9]*/--gradient_accumulation_steps=1/; s/--max_iters=[0-9]*/--max_iters=10/; s/--eval_interval=[0-9]*/--eval_interval=5/; s/--eval_iters=[0-9]*/--eval_iters=2/; s/--warmup_iters=[0-9]*/--warmup_iters=2/')
  rm -rf /scratch/pre && python -u scripts/train_variant.py --out_dir=/scratch/pre --compile=False $F 2>&1 | tail -4
  python -u scripts/fineweb/eval_fineweb.py --out_dir=/scratch/pre --windows=8 --batch=4 2>&1 | tail -2
done
echo PREFLIGHT OK
bash scripts/nrp/run_queue.sh fw42m.txt fw15m.txt
echo QUEUE DONE
