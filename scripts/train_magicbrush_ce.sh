#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
DATA_CONFIG="${DATA_CONFIG:-$DATA_ROOT/official_tokens/train/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/B0-CE}"
echo "experiment_id=B0-CE commit=$(git -C "$ROOT" rev-parse --short HEAD) host=$(hostname)"
echo "objective=ce config=$ROOT/configs/experiments/magicbrush_ce.yaml"
echo "MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT DATA_CONFIG=$DATA_CONFIG OUTPUT_DIR=$OUTPUT_DIR"
resume=(); [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]] && resume=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" "$ROOT/train/train_magicbrush.py" \
  --objective ce --model "$MODEL_PATH" --train-manifest "$DATA_CONFIG" --output "$OUTPUT_DIR" \
  --max-steps "${MAX_STEPS:-2200}" --save-steps "${SAVE_EVERY:-500}" --batch-size "${BATCH_SIZE_PER_GPU:-4}" \
  --gradient-accumulation "${GRAD_ACCUM:-2}" --learning-rate "${LEARNING_RATE:-1e-5}" \
  --warmup-steps "${WARMUP_STEPS:-0}" --weight-decay "${WEIGHT_DECAY:-0.1}" --max-grad-norm "${MAX_GRAD_NORM:-4}" \
  --max-seq-len "${MAX_SEQ_LEN:-3072}" --seed "${SEED:-42}" --num-workers "${NUM_WORKERS:-4}" "${resume[@]}"
