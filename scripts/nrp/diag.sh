#!/usr/bin/env bash
# Runs one diagnostic script and keeps its output on the PVC. No training.
set -euo pipefail
OUT=/workspace/runs/diag/$(basename "$1" .py)-$(date +%s).log
mkdir -p "$(dirname "$OUT")"
python "$1" 2>&1 | tee "$OUT"
