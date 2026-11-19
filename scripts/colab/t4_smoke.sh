# T4 smoke test of the 15M model in fp16, run on a Colab VM from /content after src.tgz and babylm.tgz are uploaded.
# 1. kernel check (S configs, fp32 and fp16) with the kernel as it is (bf16 inside) and with a copy that uses fp16 inside
# 2. throughput of 15M S and T in fp16
# 3. fp16 convergence: the BabyLM S run (same flags and seed) to step 1000, against bf16's val 3.8739 / 3.5270
set -uo pipefail
cd /content
mkdir -p ulm && tar -xzf src.tgz -C ulm && tar -xzf babylm.tgz -C ulm
cp -r ulm ulm_fp16 && sed -i 's/tl\.bfloat16/tl.float16/g; s/torch\.bfloat16/torch.float16/g' ulm_fp16/kernel.py
grep -c float16 ulm_fp16/kernel.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
pip install -q uv
uv venv -q /content/venv --python 3.12
uv pip install -q --python /content/venv/bin/python torch==2.10.0 triton==3.6.0 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
PY=/content/venv/bin/python
$PY -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
echo "== STAGE kernel check, kernel as is (bf16 inside)"
(cd ulm && $PY -u scripts/colab/check_kernel_fp16.py) 2>&1 | tail -40
echo "== STAGE kernel check, fp16 inside"
(cd ulm_fp16 && $PY -u scripts/colab/check_kernel_fp16.py) 2>&1 | tail -40
K=${KERNEL:-ulm_fp16}
echo "== STAGE speed ($K)"
(cd $K && BENCH_DTYPE=float16 $PY -u scripts/bench/train_speed.py 15M 2> /content/speed.err) | tee /content/speed.jsonl
tail -5 /content/speed.err
echo "== STAGE converge ($K)"
cd $K
$PY -u scripts/train_variant.py --out_dir=runs/t4_fp16 --dataset=babylm --variant=S --hidden_dim=712 --vocab_source=custom --vocab_size=16384 --dim=288 --n_layers=12 --max_seq_len=512 --batch_size=64 --gradient_accumulation_steps=1 --learning_rate=1e-3 --warmup_iters=200 --max_iters=4630 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=9 --dtype=float16 > /content/train.log 2>&1 &
TP=$!
until grep -q "^step 1000:" /content/train.log || ! kill -0 $TP 2>/dev/null; do sleep 20; done
kill $TP 2>/dev/null
grep -E "^step |tok/s" /content/train.log | grep -E "^step |^(100|500|1000) \|" | head -20
tail -5 /content/train.log
echo "== SMOKE DONE"
