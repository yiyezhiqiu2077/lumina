#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_CONFIG:?DATA_CONFIG must be the pre-tokenized MagicBrush train manifest}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

SAVE_STEP="${SAVE_STEP:-10}"
RESUME_STEPS="${RESUME_STEPS:-12}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR must not already exist for a fresh smoke test: $OUTPUT_DIR" >&2
  exit 2
fi

run=("${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-1}"
  "$ROOT/train/train_magicbrush_attention.py"
  --model "$MODEL_PATH" --train-manifest "$DATA_CONFIG" --output "$OUTPUT_DIR"
  --attention-layers 24 25 26 27 --attention-loss-weight 0.1
  --batch-size "${BATCH_SIZE_PER_GPU:-1}" --gradient-accumulation "${GRAD_ACCUM:-1}"
  --learning-rate "${LEARNING_RATE:-1e-5}" --warmup-steps "${WARMUP_STEPS:-2}"
  --seed "${SEED:-42}" --num-workers "${NUM_WORKERS:-0}")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${run[@]}" \
  --max-steps "$SAVE_STEP" --save-steps "$SAVE_STEP" --checkpoint-steps "$SAVE_STEP"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${run[@]}" \
  --max-steps "$RESUME_STEPS" --save-steps "$RESUME_STEPS" \
  --resume-from-checkpoint "$OUTPUT_DIR/checkpoint-$(printf '%06d' "$SAVE_STEP")"
