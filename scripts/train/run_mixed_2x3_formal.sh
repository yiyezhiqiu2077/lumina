#!/usr/bin/env bash
# Print or run the six matched 8-GPU mixed-editing formal experiments.
set -euo pipefail

usage() {
    printf 'Usage: %s [--print-command|--run]\n' "$0" >&2
}

mode="${1:---print-command}"
[[ "$#" -le 1 && ( "$mode" == '--print-command' || "$mode" == '--run' ) ]] || {
    usage
    exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
formal_launcher="${MIXED_2X3_FORMAL_LAUNCHER:-$root/scripts/train/run_mixed_formal.sh}"
asset_audit="${MIXED_2X3_ASSET_AUDIT:-$root/scripts/tools/audit_formal_assets.py}"
asset_audit_override="${MIXED_2X3_ASSET_AUDIT:+1}"
git_bin="${MIXED_2X3_GIT_BIN:-git}"

configs=(
    'configs/train/formal/mixed_ce_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_ce_editregion_8g_b4_a1.yaml'
    'configs/train/formal/mixed_attention_postrope_region_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_attention_editregion_8g_b4_a1.yaml'
    'configs/train/formal/mixed_gce_8g_b4_a1.yaml'
    'configs/train/ablation/mixed_gce_editregion_8g_b4_a1.yaml'
)
labels=(
    'CE / full_target'
    'CE / edit_region_hardlock'
    'Attention / full_target'
    'Attention / edit_region_hardlock'
    'GCE / full_target'
    'GCE / edit_region_hardlock'
)
expected_objectives=(ce ce attention attention gce gce)
expected_corruptions=(full_target edit_region_hardlock full_target edit_region_hardlock full_target edit_region_hardlock)

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

for command in "$formal_launcher" "$git_bin" uv; do
    if [[ "$command" == */* ]]; then
        [[ -x "$command" ]] || die "required executable is missing: $command"
    else
        command -v "$command" >/dev/null 2>&1 || die "required command is missing: $command"
    fi
done
if [[ -n "$asset_audit_override" ]]; then
    [[ -x "$asset_audit" ]] || die "asset audit override is not executable: $asset_audit"
else
    [[ -f "$asset_audit" ]] || die "formal asset audit script is missing: $asset_audit"
fi
run_asset_audit() {
    if [[ -n "$asset_audit_override" ]]; then "$asset_audit" "$@"; else uv run python "$asset_audit" "$@"; fi
}

cd "$root"
for variable in MODEL_PATH DATA_ROOT DATA_CONFIG OUTPUT_ROOT GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES; do
    [[ -n "${!variable:-}" ]] || die "missing $variable"
done
[[ -f "$MODEL_PATH/config.json" ]] || die "MODEL_PATH/config.json does not exist"
[[ -d "$DATA_ROOT" ]] || die "DATA_ROOT is not a directory"
[[ -f "$DATA_CONFIG" ]] || die "DATA_CONFIG is not a file"
[[ -d "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT must already exist"
[[ -f "$GCE_CLUSTER_PATH" ]] || die "GCE_CLUSTER_PATH must name an existing cluster file"

IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" == 8 ]] || die "CUDA_VISIBLE_DEVICES must list exactly 8 devices for the formal 2x3 workflow"

for config in "${configs[@]}"; do
    [[ -f "$root/$config" ]] || die "config does not exist: $config"
done

declare -a outputs max_steps steps_per_epoch epochs objectives corruptions
for index in "${!configs[@]}"; do
    config="$root/${configs[$index]}"
    metadata="$(uv run python - "$config" "${expected_objectives[$index]}" "${expected_corruptions[$index]}" <<'PY'
import sys
from pathlib import Path

from training.config import load_train_config

args = load_train_config(Path(sys.argv[1]))
expected_objective, expected_corruption = sys.argv[2:]
if args.objective != expected_objective or args.target_corruption_mode != expected_corruption:
    raise SystemExit(
        f"unexpected config metadata: objective={args.objective!r}, "
        f"target_corruption_mode={args.target_corruption_mode!r}"
    )
print(
    "\t".join(
        (
            args.objective,
            args.target_corruption_mode,
            str(args.output),
            str(args.max_steps),
            str(args.optimizer_steps_per_epoch),
            str(args.epochs),
        )
    )
)
PY
)" || die "failed to load formal config: ${configs[$index]}"
    IFS=$'\t' read -r "objectives[$index]" "corruptions[$index]" "outputs[$index]" \
        "max_steps[$index]" "steps_per_epoch[$index]" "epochs[$index]" <<< "$metadata"
done

if [[ "$mode" == '--run' ]]; then
    dirty="$("$git_bin" -C "$root" status --short)"
    [[ -z "$dirty" ]] || die "refusing dirty formal run; commit or stash first:\n$dirty"

    for output in "${outputs[@]}"; do
        [[ -d "$output" ]] || continue
        for artifact in train_metrics.jsonl quality_status.json experiment_config.json lora_report.json; do
            [[ ! -e "$output/$artifact" ]] || die "existing formal output detected: $output/$artifact"
        done
        checkpoint="$(find "$output" -maxdepth 1 -mindepth 1 -name 'checkpoint-*' -print -quit)"
        [[ -z "$checkpoint" ]] || die "existing formal output detected: $checkpoint"
    done
    [[ ! -e "$OUTPUT_ROOT/2x3_formal_pipeline.log" ]] || \
        die "existing formal output detected: $OUTPUT_ROOT/2x3_formal_pipeline.log"

    # This gates the immutable 8807/7804 training composition, all token files,
    # Lumina/VQ identity, and GCE cluster compatibility before group 1 starts.
    run_asset_audit --mode train --output "$OUTPUT_ROOT/formal_assets.json" \
        --model "$MODEL_PATH" --train-manifest "$DATA_CONFIG" --gce-clusters "$GCE_CLUSTER_PATH"
fi

# Verify every underlying single-run launch before any formal run can start.
for config in "${configs[@]}"; do
    "$formal_launcher" "$root/$config" --print-command
done

for index in "${!configs[@]}"; do
    printf '[%s/6] %s\n' "$((index + 1))" "${labels[$index]}"
    printf '  config=%s\n' "${configs[$index]}"
    printf '  objective=%s target_corruption=%s\n' "${objectives[$index]}" "${corruptions[$index]}"
    printf '  output=%s\n' "${outputs[$index]}"
    printf '  optimizer_steps_per_epoch=%s epochs=%s max_steps=%s\n' \
        "${steps_per_epoch[$index]}" "${epochs[$index]}" "${max_steps[$index]}"
done

[[ "$mode" == '--run' ]] || exit 0

pipeline_log="$OUTPUT_ROOT/2x3_formal_pipeline.log"
: > "$pipeline_log"
log_pipeline() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$pipeline_log"
}

verify_success() {
    local output="$1"
    local expected_max_steps="$2"
    uv run python - "$output" "$expected_max_steps" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_step = int(sys.argv[2])
quality_path = output / "quality_status.json"
metrics_path = output / "train_metrics.jsonl"
checkpoint = output / f"checkpoint-{expected_step:06d}" / "_SUCCESS"

if not quality_path.is_file():
    raise SystemExit(f"quality status is missing: {quality_path}")
quality = json.loads(quality_path.read_text(encoding="utf-8"))
if quality.get("status") != "SUCCEEDED" or quality.get("step") != expected_step:
    raise SystemExit(
        f"quality status did not succeed at step {expected_step}: "
        f"status={quality.get('status')!r} step={quality.get('step')!r}"
    )

if not metrics_path.is_file():
    raise SystemExit(f"metrics are missing: {metrics_path}")
records = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not records:
    raise SystemExit(f"metrics are empty: {metrics_path}")
last = json.loads(records[-1])
if last.get("step") != expected_step:
    raise SystemExit(f"last metric step is {last.get('step')!r}, expected {expected_step}")
if not checkpoint.is_file():
    raise SystemExit(f"final checkpoint success marker is missing: {checkpoint}")
print(
    f"quality_status={quality['status']} quality_step={quality['step']} "
    f"last_metric_step={last['step']} final_checkpoint={checkpoint}"
)
PY
}

for index in "${!configs[@]}"; do
    ordinal="$((index + 1))"
    label="${labels[$index]}"
    config="$root/${configs[$index]}"
    output="${outputs[$index]}"
    expected_max_steps="${max_steps[$index]}"
    started_at="$(date -Is)"
    log_pipeline "[$ordinal/6] $label START config=$config output=$output max_steps=$expected_max_steps start_time=$started_at"

    if ! "$formal_launcher" "$config" --run; then
        log_pipeline "[$ordinal/6] $label FAILED reason=process_exit finish_time=$(date -Is)"
        exit 1
    fi
    if ! gate="$(verify_success "$output" "$expected_max_steps")"; then
        log_pipeline "[$ordinal/6] $label FAILED reason=success_gate finish_time=$(date -Is)"
        exit 1
    fi
    log_pipeline "[$ordinal/6] $label SUCCEEDED finish_time=$(date -Is) $gate"
done
