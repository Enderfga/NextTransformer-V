#!/usr/bin/env bash
# Train and evaluate a list of pilot runs on one 8-GPU node, one after another.
# Usage: NTV_ROOT=<root> bash scripts/pilot.sh configs/pilot/jobs.txt   (run from the code directory)
#
# Each line of the job file is "<run> <method> [extra ntv.train arguments]"; '#' starts a comment. For every run:
# train 5,000 updates, score every checkpoint on the synthetic probe and on the harder probe (with the token-order
# scrambles), score the final checkpoint on COCO 5k and What'sUp, and copy the summaries to $NTV_ROOT/runs/<run>/summary.
# Expects the layout of the README: $NTV_ROOT/{data,resources,packed}. A finished run (summary present) is skipped.
# Optional: NTV_S3=s3://<bucket>/<prefix> uploads each run's summary and final checkpoint.
set -u
: "${NTV_ROOT:?set NTV_ROOT}"
JOBS=$1
DATA=$NTV_ROOT/data; P=$NTV_ROOT/packed; ENC=$DATA/siglip2-so400m-patch16-384; RES=$NTV_ROOT/resources
LOG=$NTV_ROOT/logs; mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

score() {  # <probe dir> <prefix> <run dir>: every checkpoint on one probe, one GPU each
  local probe=$1 prefix=$2 out=$3 gpu=0
  for ckpt in "$out"/checkpoint_*; do
    [ -f "$ckpt/complete.json" ] || continue
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m ntv.spatial_eval --checkpoint "$ckpt" --probe "$probe" \
      --encoder "$ENC" --resources "$RES" --out "$out/evals/${prefix}_$(basename "$ckpt").json" &
    gpu=$(( (gpu + 1) % 7 ))
  done
}

grep -v -E '^\s*(#|$)' "$JOBS" | while read -r run method extra; do
  OUT=$NTV_ROOT/runs/$run
  [ -f "$OUT/summary/metrics.jsonl" ] && { echo "skip $run (done)"; continue; }
  mkdir -p "$OUT/evals"
  echo "start $run $(date -u +%FT%TZ)"
  .venv/bin/torchrun --standalone --nproc_per_node 8 -m ntv.train --method "$method" \
    --model "configs/track/$method.json" --train-rows "$P/pilot-train" --val-rows "$P/pilot-val" --tars "$DATA" \
    --encoder "$ENC" --resources "$RES" --output "$OUT" --updates 5000 --micro-batch 2 --eval-every 250 \
    --eval-rows 128 --checkpoint-every 1000 --log-every 10 --workers 10 --flops-per-token 5.99e9 $extra \
    >> "$LOG/$run.log" 2>&1 < /dev/null || { echo "train $run failed, see $LOG/$run.log"; continue; }
  {
    score "$DATA/synth-eval" spatial "$OUT"
    wait
    score "$DATA/synth-hard" hard "$OUT"
    last=$(ls -d "$OUT"/checkpoint_* | tail -1)
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ntv.evaluate --checkpoint "$last" --coco "$DATA/coco5k" \
      --encoder "$ENC" --resources "$RES" --out "$OUT/evals/coco_$(basename "$last").json" &
    wait
  } >> "$LOG/$run.eval.log" 2>&1 < /dev/null
  mkdir -p "$OUT/summary" && cp "$OUT/metrics.jsonl" "$OUT/latest.json" "$OUT"/evals/*.json "$OUT/summary/"
  if [ -n "${NTV_S3:-}" ]; then
    .venv/bin/python scripts/s3sync.py up "$OUT/summary" "$NTV_S3/runs/$run/summary"
    .venv/bin/python scripts/s3sync.py up "$last" "$NTV_S3/runs/$run/$(basename "$last")"
  fi
  echo "done $run $(date -u +%FT%TZ)"
done
