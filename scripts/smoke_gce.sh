#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${GCE_CLUSTER_PATH:?GCE_CLUSTER_PATH is required}"
DATA_CONFIG="${DATA_CONFIG:-$DATA_ROOT/official_tokens/train/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/G1-GCE-smoke}"
echo "experiment_id=G1-GCE-smoke commit=$(git -C "$ROOT" rev-parse --short HEAD) host=$(hostname)"
echo "config=$ROOT/configs/experiments/magicbrush_gce.yaml"
echo "MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT DATA_CONFIG=$DATA_CONFIG OUTPUT_ROOT=$OUTPUT_ROOT OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

SAVE_STEP="${SAVE_STEP:-10}"
RESUME_STEPS="${RESUME_STEPS:-12}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR must not already exist for a fresh smoke test: $OUTPUT_DIR" >&2
  exit 2
fi

run=("${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-1}"
  "$ROOT/tools/smoke_gce_ddp.py"
  --model "$MODEL_PATH" --manifest "$DATA_CONFIG" --output "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE_PER_GPU:-1}" --accum "${GRAD_ACCUM:-1}"
  --max-seq-len "${MAX_SEQ_LEN:-3072}" --seed "${SEED:-42}"
  --use-gce --clusters "$GCE_CLUSTER_PATH")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${run[@]}" \
  --steps "$SAVE_STEP" --save-every "$SAVE_STEP"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${run[@]}" \
  --steps "$RESUME_STEPS" --save-every "$RESUME_STEPS" \
  --resume "$OUTPUT_DIR/checkpoint-$(printf '%06d' "$SAVE_STEP")"
