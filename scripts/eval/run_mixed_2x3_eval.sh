#!/usr/bin/env bash
# Print or run matched MagicBrush TEST evaluations for the six formal checkpoints.
set -euo pipefail

usage() {
    printf 'Usage: %s [--print-command|--run]\n' "$0" >&2
}

mode="${1:---print-command}"
[[ "$#" -le 1 && ( "$mode" == '--print-command' || "$mode" == '--run' ) ]] || { usage; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
asset_audit="${MIXED_2X3_ASSET_AUDIT:-$root/scripts/tools/audit_formal_assets.py}"
asset_audit_override="${MIXED_2X3_ASSET_AUDIT:+1}"
evaluator_override="${MIXED_2X3_EVALUATOR:-}"
compare_override="${MIXED_2X3_COMPARE:-}"

configs=(
    'configs/train/formal/mixed_ce_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_ce_editregion_8g_b4_a1.yaml'
    'configs/train/formal/mixed_attention_postrope_region_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_attention_editregion_8g_b4_a1.yaml'
    'configs/train/formal/mixed_gce_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_gce_editregion_8g_b4_a1.yaml'
)
labels=('CE-full' 'CE-editregion' 'Attention-full' 'Attention-editregion' 'GCE-full' 'GCE-editregion')

die() { printf '%s\n' "$*" >&2; exit 1; }
run_evaluator() {
    if [[ -n "$evaluator_override" ]]; then "$evaluator_override" "$@"; else uv run python "$root/scripts/eval/evaluate_gt_mask_editing.py" "$@"; fi
}
run_compare() {
    if [[ -n "$compare_override" ]]; then "$compare_override" "$@"; else uv run python "$root/scripts/eval/compare_gt_mask_editing.py" "$@"; fi
}
run_asset_audit() {
    if [[ -n "$asset_audit_override" ]]; then "$asset_audit" "$@"; else uv run python "$asset_audit" "$@"; fi
}

if [[ -n "$asset_audit_override" ]]; then
    [[ -x "$asset_audit" ]] || die "asset audit override is not executable: $asset_audit"
else
    [[ -f "$asset_audit" ]] || die "formal asset audit script is missing: $asset_audit"
fi
if [[ -n "$evaluator_override" ]]; then [[ -x "$evaluator_override" ]] || die "evaluator override is not executable: $evaluator_override"; fi
if [[ -n "$compare_override" ]]; then [[ -x "$compare_override" ]] || die "compare override is not executable: $compare_override"; fi
command -v uv >/dev/null 2>&1 || die "uv is required for formal evaluation"

cd "$root"
for variable in MODEL_PATH DATA_ROOT DATA_CONFIG OUTPUT_ROOT EVAL_OUTPUT_ROOT GCE_CLUSTER_PATH MAGICBRUSH_TEST_CANONICAL_MANIFEST MAGICBRUSH_TEST_TOKEN_MANIFEST MAGICBRUSH_TEST_SUBSET DINO_MODEL_PATH CLIP_MODEL_PATH; do
    [[ -n "${!variable:-}" ]] || die "missing $variable"
done
[[ -f "$MODEL_PATH/config.json" ]] || die "MODEL_PATH/config.json does not exist"
[[ -d "$DATA_ROOT" && -f "$DATA_CONFIG" ]] || die "DATA_ROOT and DATA_CONFIG must name the mixed training assets used only to resolve final checkpoint steps"
[[ -f "$GCE_CLUSTER_PATH" ]] || die "GCE_CLUSTER_PATH is required to resolve the frozen GCE final checkpoint step"
[[ -d "$OUTPUT_ROOT" && -d "$EVAL_OUTPUT_ROOT" ]] || die "OUTPUT_ROOT and EVAL_OUTPUT_ROOT must already exist"
for file in "$MAGICBRUSH_TEST_CANONICAL_MANIFEST" "$MAGICBRUSH_TEST_TOKEN_MANIFEST" "$MAGICBRUSH_TEST_SUBSET"; do
    [[ -f "$file" ]] || die "MagicBrush TEST asset does not exist: $file"
done
[[ -d "$DINO_MODEL_PATH" && -d "$CLIP_MODEL_PATH" ]] || die "DINO_MODEL_PATH and CLIP_MODEL_PATH must be local model directories"
for config in "${configs[@]}"; do [[ -f "$root/$config" ]] || die "config does not exist: $config"; done

declare -a train_outputs max_steps eval_outputs
for index in "${!configs[@]}"; do
    metadata="$(uv run python - "$root/${configs[$index]}" <<'PY'
import sys
from pathlib import Path
from training.config import load_train_config
args = load_train_config(Path(sys.argv[1]))
print(f"{args.output}\t{args.max_steps}")
PY
)" || die "failed to load formal config: ${configs[$index]}"
    IFS=$'\t' read -r "train_outputs[$index]" "max_steps[$index]" <<< "$metadata"
    eval_outputs[$index]="$EVAL_OUTPUT_ROOT/${labels[$index]}"
done

for index in "${!configs[@]}"; do
    printf '[%s/6] %s\n' "$((index + 1))" "${labels[$index]}"
    printf '  checkpoint=%s/checkpoint-%06d\n' "${train_outputs[$index]}" "${max_steps[$index]}"
    printf '  test_token_manifest=%s\n  test_subset=%s\n  eval_output=%s\n' \
        "$MAGICBRUSH_TEST_TOKEN_MANIFEST" "$MAGICBRUSH_TEST_SUBSET" "${eval_outputs[$index]}"
    printf '  timesteps=%s cfg_scale=%s cfg_img=%s temperature=%s seed=%s lpips=alex roi_padding=0.10\n' \
        "${FORMAL_EVAL_TIMESTEPS:-64}" "${FORMAL_EVAL_CFG_SCALE:-2.5}" "${FORMAL_EVAL_CFG_IMG:-4.0}" "${FORMAL_EVAL_TEMPERATURE:-1.0}" "${FORMAL_EVAL_SEED:-42}"
done

[[ "$mode" == '--run' ]] || exit 0

for index in "${!configs[@]}"; do
    checkpoint="${train_outputs[$index]}/checkpoint-$(printf '%06d' "${max_steps[$index]}")"
    [[ -f "$checkpoint/_SUCCESS" ]] || die "final formal checkpoint is missing: $checkpoint/_SUCCESS"
    [[ ! -e "${eval_outputs[$index]}/summary.json" && ! -e "${eval_outputs[$index]}/per_sample.jsonl" ]] || \
        die "existing formal evaluation output detected: ${eval_outputs[$index]}"
done

run_asset_audit --mode eval --output "$OUTPUT_ROOT/formal_assets.json" --model "$MODEL_PATH" \
    --canonical-test-manifest "$MAGICBRUSH_TEST_CANONICAL_MANIFEST" \
    --test-token-manifest "$MAGICBRUSH_TEST_TOKEN_MANIFEST" --test-subset "$MAGICBRUSH_TEST_SUBSET" \
    --dino-model "$DINO_MODEL_PATH" --clip-model "$CLIP_MODEL_PATH"

for index in "${!configs[@]}"; do
    checkpoint="${train_outputs[$index]}/checkpoint-$(printf '%06d' "${max_steps[$index]}")"
    run_evaluator --manifest "$MAGICBRUSH_TEST_TOKEN_MANIFEST" --subset "$MAGICBRUSH_TEST_SUBSET" --limit 0 \
        --model "$MODEL_PATH" --checkpoint "$checkpoint" --output "${eval_outputs[$index]}" --model-label "${labels[$index]}" \
        --timesteps "${FORMAL_EVAL_TIMESTEPS:-64}" --cfg-scale "${FORMAL_EVAL_CFG_SCALE:-2.5}" \
        --cfg-img "${FORMAL_EVAL_CFG_IMG:-4.0}" --temperature "${FORMAL_EVAL_TEMPERATURE:-1.0}" --seed "${FORMAL_EVAL_SEED:-42}" \
        --lpips --lpips-net alex --dino-model "$DINO_MODEL_PATH" --clip-model "$CLIP_MODEL_PATH" --roi-padding-ratio 0.10
    [[ -s "${eval_outputs[$index]}/per_sample.jsonl" && -s "${eval_outputs[$index]}/summary.json" ]] || \
        die "formal evaluation did not produce summary/per_sample output: ${eval_outputs[$index]}"
done

compare_args=(--output "$EVAL_OUTPUT_ROOT/comparison")
for index in "${!configs[@]}"; do compare_args+=(--model "${labels[$index]}" "${eval_outputs[$index]}/summary.json"); done
run_compare "${compare_args[@]}"
