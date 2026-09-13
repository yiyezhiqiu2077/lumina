#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
DATA_CONFIG="${DATA_CONFIG:-$DATA_ROOT/official_tokens/train/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/A1-ATTN}"
echo "experiment_id=A1-ATTN commit=$(git -C "$ROOT" rev-parse --short HEAD) host=$(hostname)"
echo "config=$ROOT/configs/experiments/magicbrush_attention.yaml"
echo "MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT DATA_CONFIG=$DATA_CONFIG OUTPUT_ROOT=$OUTPUT_ROOT OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
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
