#!/usr/bin/env bash
# Runs inside the setup Job, from a source copy on local disk. Builds the environment and the 512-vocab
# TinyStories tokens on local disk, then stores them on the PVC as one tarball and one directory of .bin files.
# NRP prohibits pip and conda installs on CephFS, so the environment is only ever unpacked to local disk.
set -euo pipefail

SRC=$(pwd)
LOCK=$(sha256sum uv.lock | cut -c1-12)
ENV_TAR=/workspace/env/venv-$LOCK.tar
mkdir -p /workspace/env /workspace/data /opt/ulm /scratch/data
ln -sfn /scratch/data data

python3 --version
pip install --quiet --root-user-action=ignore uv
export UV_PROJECT_ENVIRONMENT=/opt/ulm/venv UV_CACHE_DIR=/scratch/uv-cache
uv sync --frozen --python "$(command -v python3)"
du -sh /opt/ulm/venv
/opt/ulm/venv/bin/python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
tar -C /opt/ulm -cf "$ENV_TAR.tmp" venv
mv "$ENV_TAR.tmp" "$ENV_TAR"
ls -l "$ENV_TAR"

/opt/ulm/venv/bin/python tinystories.py download
cp tok512.model data/
# ProcessPoolExecutor defaults to os.cpu_count(), which is the whole node here, and each worker holds a shard in RAM
/opt/ulm/venv/bin/python - <<'EOF'
import os, sys, runpy
os.cpu_count = lambda: 8
sys.argv = ["tinystories.py", "pretokenize", "--vocab_size=512"]
sys.path.insert(0, os.getcwd())
runpy.run_path("tinystories.py", run_name="__main__")
EOF

# pretokenize never consumes executor.map, so a failed worker is silent: check that every shard was written
n_json=$(ls data/TinyStories_all_data/*.json | wc -l)
n_bin=$(ls data/tok512/*.bin | wc -l)
echo "shards: $n_json json, $n_bin bin"
test "$n_json" -eq "$n_bin"
find data/tok512 -name '*.bin' -size 0 | (! grep .)

rm -rf /workspace/data/tok512.tmp
cp -r /scratch/data/tok512 /workspace/data/tok512.tmp
(cd /scratch/data/tok512 && sha256sum *.bin) > /workspace/data/tok512.sha256.tmp
(cd /workspace/data/tok512.tmp && sha256sum -c --quiet /workspace/data/tok512.sha256.tmp)
rm -rf /workspace/data/tok512
mv /workspace/data/tok512.tmp /workspace/data/tok512
mv /workspace/data/tok512.sha256.tmp /workspace/data/tok512.sha256
du -sh /workspace/data/tok512
echo "setup done: env $ENV_TAR, data /workspace/data/tok512"
