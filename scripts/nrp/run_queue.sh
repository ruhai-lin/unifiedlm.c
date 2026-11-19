#!/usr/bin/env bash
# Works through plan files in order with any number of pods: each pod claims the next unclaimed line by
# creating a directory on the PVC (atomic), runs it, and moves on, so the environment is unpacked once per pod
# and no line waits on a pod that never gets a card. A claim left by a crashed pod is cleared by hand; a claim
# made by hand holds a line back.
set -euo pipefail

for PLAN in "$@"; do
    CLAIMS=/workspace/runs/${PLAN%.txt}/.claims
    mkdir -p "$CLAIMS"
    N=$(grep -c . "scripts/nrp/$PLAN")
    for ((i = 0; i < N; i++)); do
        NAME=$(sed -n "$((i + 1))p" "scripts/nrp/$PLAN" | cut -d' ' -f1)
        [ -e "/workspace/runs/${PLAN%.txt}/$NAME/eval.json" ] && continue
        mkdir "$CLAIMS/$NAME" 2>/dev/null || continue
        echo "${HOSTNAME:-pod} $(date -u +%FT%TZ)" > "$CLAIMS/$NAME/by"
        JOB_COMPLETION_INDEX=$i bash scripts/nrp/run_index.sh "$PLAN"
    done
done
