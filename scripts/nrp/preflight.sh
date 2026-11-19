#!/usr/bin/env bash
# Gate before any training: the repo's own tests, the real kernel against the dense reference, and the run path.
set -euo pipefail

OUT=/workspace/runs/preflight/$(git rev-parse --short HEAD 2>/dev/null || basename "$PWD")-$(date +%s)
mkdir -p "$OUT"
exec > >(tee "$OUT/preflight.log") 2>&1

nvidia-smi
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python test_all.py
python scripts/check_kernel.py
python scripts/preflight.py
echo "PREFLIGHT PASSED"
