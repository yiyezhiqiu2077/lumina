#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/zhangyuyang/lumina_attn_supervision
REPO="$ROOT/code/Lumina-DiMOO"
MANIFEST="$ROOT/datasets/magicbrush_tokens/dev528/manifest.jsonl"
STEPS=(300 600 1000 1500)
pids=()

run_one() {
    local gpu="$1"
    local variant="$2"
    local step="$3"
    local output="$ROOT/outputs/dev528_hardlock64/${variant}_step${step}"
    local log="$ROOT/logs/dev528_hardlock64_${variant}_step${step}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$ROOT/env/bin/python" \
        "$REPO/tools/evaluate_magicbrush_hardlock.py" \
        --model "$ROOT/models/Lumina-DiMOO" \
        --checkpoint "$ROOT/outputs/${variant}_long1500_from_smoke100_v2/checkpoint-$(printf '%06d' "$step")" \
        --manifest "$MANIFEST" \
        --output "$output" \
        --timesteps 64 \
        --cfg-scale 2.5 \
        --cfg-img 4.0 \
        --temperature 1.0 \
        --seed 42 \
        > "$log" 2>&1 &
    pids+=("$!")
}

cd "$REPO"
for index in "${!STEPS[@]}"; do
    run_one "$index" A0 "${STEPS[$index]}"
done
for index in "${!STEPS[@]}"; do
    run_one "$((index + 4))" A1 "${STEPS[$index]}"
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -eq 0 ]]; then
    printf '{"complete":true,"samples":528,"timesteps":64,"seed":42}\n' \
        > "$ROOT/outputs/dev528_hardlock64/experiment_complete.json"
fi
exit "$status"
