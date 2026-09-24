#!/usr/bin/env bash
# Run 8-GPU Attention, GCE, and CE smoke tests serially in detached tmux.
set -euo pipefail

readonly SESSION='lumina_smoke'

session_exists() {
    tmux has-session -t "=${1}" 2>/dev/null
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_smokes_tmux.sh start
  bash scripts/train/run_smokes_tmux.sh status
  bash scripts/train/run_smokes_tmux.sh logs
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

smoke_name_for() {
    case "$1" in
        attention) printf '%s\n' 'SMOKE-MB-ATTN-8G-B4-A1-S42' ;;
        gce) printf '%s\n' 'SMOKE-MB-GCE-8G-B4-A1-S42' ;;
        ce) printf '%s\n' 'SMOKE-MB-CE-8G-B4-A1-S42' ;;
        *) die "unsupported smoke objective: $1" ;;
    esac
}

require_environment() {
    local variable
    for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES; do
        [[ -n "${!variable:-}" ]] || die "required environment variable is empty: $variable"
    done
    [[ -f "$MODEL_PATH/config.json" ]] || die "missing model config: $MODEL_PATH/config.json"
    [[ -d "$MODEL_PATH/vqvae" ]] || die "missing model VQ-VAE directory: $MODEL_PATH/vqvae"
    [[ -f "$DATA_CONFIG" ]] || die "DATA_CONFIG is not a file: $DATA_CONFIG"
    [[ -f "$GCE_CLUSTER_PATH" ]] || die "GCE_CLUSTER_PATH is not a file: $GCE_CLUSTER_PATH"
    [[ "$(wc -l < "$DATA_CONFIG")" == '8807' ]] || die "MagicBrush manifest must contain 8807 rows: $DATA_CONFIG"
    local -a devices
    local device
    local -A seen=()
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    [[ "${#devices[@]}" -eq 8 ]] || die "smoke requires exactly eight visible GPUs"
    for device in "${devices[@]}"; do
        [[ "$device" =~ ^[0-9]+$ ]] || die "invalid GPU ID in CUDA_VISIBLE_DEVICES"
        [[ -z "${seen[$device]:-}" ]] || die "smoke requires eight distinct visible GPUs"
        seen[$device]=1
    done
}

write_smoke_config() {
    local objective="$1"
    local source="$PROJECT_ROOT/configs/train/ablation/mb_${objective}_8g_b4_a1.yaml"
    local destination="$OUTPUT_ROOT/smoke/configs/mb_${objective}_smoke.yaml"
    local name
    name="$(smoke_name_for "$objective")"
    uv run python - "$source" "$destination" "$name" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["experiment_name"] = config["output_name"] = name
config["training"]["scheduler_horizon_steps"] = config["training"][
    "max_optimizer_steps"
]
config["training"]["max_optimizer_steps"] = 20
config["training"]["checkpoint_every_steps"] = 20
config["quality_gate"].update(
    baseline_start=1,
    baseline_end=10,
    window_size=10,
    consecutive_windows=2,
)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

write_launcher() {
    local launcher="$OUTPUT_ROOT/smoke/logs/smokes.command.sh"
    local log_file="$OUTPUT_ROOT/smoke/logs/smokes.log"
    local exit_file="$OUTPUT_ROOT/smoke/logs/smokes.exit_code"
    local running_file="$OUTPUT_ROOT/smoke/logs/smokes.running"
    local variable objective name stage_exit stage_quality

    umask 077
    mkdir -p "$OUTPUT_ROOT/smoke/logs"
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' 'set -o pipefail'
        for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES PATH; do
            printf 'export %s=%q\n' "$variable" "${!variable:-}"
        done
        printf 'unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT\n'
        printf 'cd %q\n' "$PROJECT_ROOT"
        printf 'rm -f %q\n' "$exit_file"
        printf 'touch %q\n' "$running_file"
        printf 'finish() { local status=$?; trap - EXIT; rm -f %q; printf %q "$status" > %q; exit "$status"; }\n' \
            "$running_file" '%s\n' "$exit_file"
        printf 'trap finish EXIT\n'
        printf '{\n'
        printf '  printf %q "$(date -Is)"\n' 'smokes_started_at=%s\n'
        for objective in attention gce ce; do
            name="$(smoke_name_for "$objective")"
            stage_exit="$OUTPUT_ROOT/smoke/logs/${objective}.exit_code"
            stage_quality="$OUTPUT_ROOT/$name/quality_status.json"
            printf '  rm -f %q\n' "$stage_exit"
            printf '  printf %q "$(date -Is)"\n' "objective=$objective state=starting at=%s\n"
            printf '  uv run python scripts/train/train.py --config %q\n' "$OUTPUT_ROOT/smoke/configs/mb_${objective}_smoke.yaml"
            printf '  status=$?\n'
            printf '  printf %q "$status" > %q\n' '%s\n' "$stage_exit"
            printf '  if [[ "$status" -ne 0 ]]; then\n'
            printf '    uv run python -m training.quality --mark-process-failed %q --objective %q --exit-code "$status" || true\n' "$stage_quality" "$objective"
            printf '    printf %q "$status" "$(date -Is)"\n' "objective=$objective state=failed exit_code=%s at=%s\n"
            printf '    exit "$status"\n'
            printf '  fi\n'
            printf '  uv run python -m training.quality --require-succeeded %q\n' "$stage_quality"
            printf '  status=$?\n'
            printf '  if [[ "$status" -ne 0 ]]; then printf %q "$status" > %q; exit "$status"; fi\n' '%s\n' "$stage_exit"
            printf '  printf %q "$(date -Is)"\n' "objective=$objective state=succeeded exit_code=0 quality=SUCCEEDED at=%s\n"
        done
        printf '  printf %q "$(date -Is)"\n' 'smokes_finished_at=%s\n'
        printf '} 2>&1 | tee %q\n' "$log_file"
        printf 'SMOKE_STATUS=${PIPESTATUS[0]}\n'
        printf 'exit "$SMOKE_STATUS"\n'
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

start() {
    command -v tmux >/dev/null 2>&1 || die 'required command not found: tmux'
    require_environment
    for session in lumina_prepare lumina_smoke lumina_all lumina_attn lumina_gce lumina_ce; do
        if [[ "$session" == 'lumina_prepare' && "${LUMINA_PIPELINE_PARENT:-}" == '1' ]]; then
            continue
        fi
        ! session_exists "$session" || die "another Lumina session is already active: $session"
    done
    local objective output_name
    for objective in attention gce ce; do
        output_name="$(smoke_name_for "$objective")"
        [[ ! -e "$OUTPUT_ROOT/$output_name" ]] || die "smoke output already exists: $OUTPUT_ROOT/$output_name"
        [[ ! -e "$OUTPUT_ROOT/smoke/logs/${objective}.exit_code" ]] || die "smoke stage status already exists: $objective"
        write_smoke_config "$objective"
    done
    [[ ! -f "$OUTPUT_ROOT/smoke/logs/smokes.running" ]] || die "stale smoke running marker detected"
    rm -f "$OUTPUT_ROOT/smoke/logs/smokes.exit_code"
    local launcher
    launcher="$(write_launcher)"
    tmux new-session -d -s "$SESSION" bash "$launcher"
    printf 'started objectives=attention,gce,ce session=%s\n' "$SESSION"
    printf 'log=%s\n' "$OUTPUT_ROOT/smoke/logs/smokes.log"
}

status() {
    if session_exists "$SESSION" || [[ -f "${OUTPUT_ROOT:-}/smoke/logs/smokes.running" ]]; then
        printf 'RUNNING session=%s\n' "$SESSION"
        return
    fi
    if [[ -f "${OUTPUT_ROOT:-}/smoke/logs/smokes.exit_code" ]]; then
        printf 'FINISHED exit_code=%s\n' "$(< "$OUTPUT_ROOT/smoke/logs/smokes.exit_code")"
        local objective name quality='MISSING' code='MISSING'
        for objective in attention gce ce; do
            name="$(smoke_name_for "$objective")"
            [[ ! -f "$OUTPUT_ROOT/smoke/logs/${objective}.exit_code" ]] || code="$(< "$OUTPUT_ROOT/smoke/logs/${objective}.exit_code")"
            if [[ -f "$OUTPUT_ROOT/$name/quality_status.json" ]]; then
                quality="$(python3 - "$OUTPUT_ROOT/$name/quality_status.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "UNKNOWN"))
PY
)"
            fi
            printf '%-10s exit_code=%-8s quality=%s\n' "$objective" "$code" "$quality"
            code='MISSING'
            quality='MISSING'
        done
    else
        printf 'NOT STARTED\n'
    fi
}

case "${1:-help}" in
    start) [[ "$#" == 1 ]] || die 'usage: run_smokes_tmux.sh start'; start ;;
    status) [[ "$#" == 1 ]] || die 'usage: run_smokes_tmux.sh status'; status ;;
    logs)
        [[ "$#" == 1 ]] || die 'usage: run_smokes_tmux.sh logs'
        [[ -f "${OUTPUT_ROOT:-}/smoke/logs/smokes.log" ]] || die 'smoke log does not exist'
        tail -f "$OUTPUT_ROOT/smoke/logs/smokes.log"
        ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $1" ;;
esac
