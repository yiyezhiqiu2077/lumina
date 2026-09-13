#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_CONFIG:?DATA_CONFIG must be the pre-tokenized MagicBrush train manifest}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
"${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  "$ROOT/train/train_magicbrush_attention.py" \
  --model "$MODEL_PATH" --train-manifest "$DATA_CONFIG" --output "$OUTPUT_DIR" \
  --attention-layers 24 25 26 27 --attention-loss-weight "${ATTENTION_LOSS_WEIGHT:-0.1}" \
  --max-steps "${MAX_STEPS:-2750}" --save-steps "${SAVE_EVERY:-275}" \
  --batch-size "${BATCH_SIZE_PER_GPU:-8}" --gradient-accumulation "${GRAD_ACCUM:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" --warmup-steps "${WARMUP_STEPS:-20}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" --max-grad-norm "${MAX_GRAD_NORM:-4.0}" \
  --lora-rank "${LORA_RANK:-16}" --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" --condition-dropout "${CONDITION_DROPOUT:-0.1}" \
  --seed "${SEED:-42}" --num-workers "${NUM_WORKERS:-4}" "${resume_args[@]}"
