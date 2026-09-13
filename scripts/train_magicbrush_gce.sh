#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_CONFIG:?DATA_CONFIG must be the pre-tokenized MagicBrush train manifest}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${GCE_CLUSTER_PATH:?GCE_CLUSTER_PATH is required}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume "$RESUME_FROM_CHECKPOINT")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  "$ROOT/tools/smoke_gce_ddp.py" \
  --model "$MODEL_PATH" --manifest "$DATA_CONFIG" --output "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE_PER_GPU:-4}" --accum "${GRAD_ACCUM:-2}" \
  --steps "${MAX_STEPS:-2200}" --save-every "${SAVE_EVERY:-500}" \
  --max-seq-len "${MAX_SEQ_LEN:-3072}" --seed "${SEED:-42}" \
  --use-gce --clusters "$GCE_CLUSTER_PATH" "${resume_args[@]}"
