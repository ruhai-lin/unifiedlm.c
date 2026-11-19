#!/usr/bin/env bash
# Runs line JOB_COMPLETION_INDEX of a plan file, "<name> <flags...>". Trains on local disk, then publishes the
# finished run to the PVC with a rename, so a run directory on the PVC is always complete. A FineWeb line also
# copies each checkpoint (ckpt.pt at every eval, with train.log) to <plan>/.resume/<name> on the PVC; a pod that
# takes over a line whose pod was stopped resumes from there, and the copy is deleted once the run is published.
# train_seconds in timing.json counts this pod only.
set -euo pipefail
# compiled graphs and kernels are cached on the PVC, so a variant compiles once per GPU type instead of once per run;
# INDUCTOR_DIR overrides it (the fork session uses local /scratch: on some nodes the cephfs metadata waits stretch a
# new variant's compile to 30-45 min with the GPU idle)
export TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_DIR:-/workspace/cache/inductor} TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1

PLAN=$1
LINE=$(sed -n "$((JOB_COMPLETION_INDEX + 1))p" "scripts/nrp/$PLAN")
NAME=${LINE%% *}
FLAGS=${LINE#* }
OUT=/workspace/runs/${PLAN%.txt}/$NAME
RESUME=/workspace/runs/${PLAN%.txt}/.resume/$NAME
if [ -e "$OUT/eval.json" ]; then
    echo "$NAME is already complete at $OUT"
    exit 0
fi

LOCAL=/scratch/out
rm -rf "$LOCAL"  # a pod may run several plan lines; start each from an empty directory
mkdir -p "$LOCAL" "$(dirname "$OUT")"
nvidia-smi -L || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 > "$LOCAL/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

START=$SECONDS
if [[ $PLAN == mem* ]]; then  # synthetic memory tasks write their own eval.json
    python scripts/memory/mqar.py --out_dir="$LOCAL" $FLAGS 2>&1 | tee "$LOCAL/train.log"
    TRAIN_SECONDS=$((SECONDS - START))
else
    MIRROR="" RESUMED=""
    if [[ $PLAN == fw* ]]; then
        MIRROR=$RESUME
        if [ -s "$RESUME/ckpt.pt" ]; then  # an earlier pod was stopped: continue from its last eval
            cp "$RESUME/ckpt.pt" "$LOCAL/"
            if [ -e "$RESUME/train.log" ]; then cp "$RESUME/train.log" "$LOCAL/"; fi
            RESUMED=--init_from=resume
            echo "resuming $NAME from $RESUME"
        fi
    fi
    CKPT_MIRROR=$MIRROR python scripts/train_variant.py --out_dir="$LOCAL" $FLAGS $RESUMED 2>&1 | tee -a "$LOCAL/train.log"
    TRAIN_SECONDS=$((SECONDS - START))
    EVAL=scripts/eval_nll.py
    [[ $PLAN == babylm* ]] && EVAL=scripts/babylm/eval_babylm.py
    [[ $PLAN == fw* ]] && EVAL=scripts/fineweb/eval_fineweb.py
    if ! python $EVAL --out_dir="$LOCAL" 2>&1 | tee "$LOCAL/eval.log"; then  # keep the trained weights for a re-eval
        rm -rf "$OUT.noeval.tmp" && cp -r "$LOCAL" "$OUT.noeval.tmp" && mv "$OUT.noeval.tmp" "$OUT.noeval"
        echo "eval failed; trained run kept at $OUT.noeval"
        exit 1
    fi
    if [[ $PLAN == fwsg* || $PLAN == fwsr* || $PLAN == fwsh* || $PLAN == fwsd* || $PLAN == fwlam* ]]; then  # past-L diagnostics and the learned gates; not fatal
        python scripts/fineweb/diag_long.py --out_dir="$LOCAL" > "$LOCAL/diag_long.log" 2>&1 || true
        python scripts/fineweb/gates.py --out_dir="$LOCAL" > "$LOCAL/gates.log" 2>&1 || true
    fi
fi
echo "{\"name\": \"$NAME\", \"node\": \"${NODE_NAME:-unknown}\", \"train_seconds\": $TRAIN_SECONDS, \"resumed\": $([ -n "${RESUMED:-}" ] && echo true || echo false)}" > "$LOCAL/timing.json"

rm -rf "$OUT.tmp"
cp -r "$LOCAL" "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
rm -rf "$RESUME"
echo "published $OUT after ${TRAIN_SECONDS}s of training"
