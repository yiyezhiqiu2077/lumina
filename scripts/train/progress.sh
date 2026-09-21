#!/usr/bin/env bash
# Display current data-preparation, smoke, training, and GPU progress.
set -euo pipefail

: "${PROJECT_ROOT:?export PROJECT_ROOT}"
: "${DATA_ROOT:?export DATA_ROOT}"
: "${MODEL_PATH:?export MODEL_PATH}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"
: "${GCE_CLUSTER_PATH:?export GCE_CLUSTER_PATH}"

show_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        printf '%s rows=%s size=%s\n' "$file" "$(wc -l < "$file")" "$(du -h "$file" | cut -f1)"
    else
        printf '%s MISSING\n' "$file"
    fi
}

printf 'updated_at=%s\n\n' "$(date -Is)"
printf 'DATA PREPARATION\n'
show_file "$DATA_ROOT/magicbrush/raw/train.jsonl"
show_file "$DATA_ROOT/magicbrush/prepared/train.jsonl"
show_file "$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl"
printf 'images=%s token_files=%s\n' \
    "$(find "$DATA_ROOT/magicbrush/raw/images" -maxdepth 1 -type f 2>/dev/null | wc -l)" \
    "$(find "$DATA_ROOT/magicbrush/tokens/train/files" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
printf 'model_size=%s\n' "$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || printf MISSING)"
printf 'gce_cluster=%s\n' "$([[ -f "$GCE_CLUSTER_PATH" ]] && printf READY || printf MISSING)"

printf '\nPIPELINE STATUS\n'
bash "$PROJECT_ROOT/scripts/train/run_pipeline_tmux.sh" status

printf '\nSMOKE STATUS\n'
bash "$PROJECT_ROOT/scripts/train/run_smokes_tmux.sh" status

printf '\nPROBE STATUS\n'
bash "$PROJECT_ROOT/scripts/train/run_probe_tmux.sh" status

printf '\nEXPERIMENT STATUS\n'
bash "$PROJECT_ROOT/scripts/train/run_tmux.sh" status

printf '\nLATEST LOG LINES\n'
for file in \
    "$OUTPUT_ROOT/logs/pipeline.log" \
    "$OUTPUT_ROOT/smoke/logs/smokes.log" \
    "$OUTPUT_ROOT/probe/logs/ce.log" \
    "$OUTPUT_ROOT/probe/logs/attention.log" \
    "$OUTPUT_ROOT/probe/logs/gce.log" \
    "$OUTPUT_ROOT/logs/all.log" \
    "$OUTPUT_ROOT/logs/attention.log" \
    "$OUTPUT_ROOT/logs/gce.log" \
    "$OUTPUT_ROOT/logs/ce.log"; do
    if [[ -f "$file" ]]; then
        printf '\n--- %s ---\n' "$file"
        tail -n 5 "$file"
    fi
done

printf '\nGPU STATUS\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
