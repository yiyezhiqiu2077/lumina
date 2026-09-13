#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
DATA_CONFIG="${DATA_CONFIG:-$DATA_ROOT/official_tokens/train/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/A1-ATTN-smoke}"
echo "experiment_id=A1-ATTN-smoke commit=$(git -C "$ROOT" rev-parse --short HEAD) host=$(hostname)"
echo "config=$ROOT/configs/experiments/magicbrush_attention.yaml"
echo "MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT DATA_CONFIG=$DATA_CONFIG OUTPUT_ROOT=$OUTPUT_ROOT OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${run[@]}" \
  --max-steps "$SAVE_STEP" --save-steps "$SAVE_STEP" --checkpoint-steps "$SAVE_STEP"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${run[@]}" \
  --max-steps "$RESUME_STEPS" --save-steps "$RESUME_STEPS" \
  --resume-from-checkpoint "$OUTPUT_DIR/checkpoint-$(printf '%06d' "$SAVE_STEP")"
