#!/usr/bin/env bash
# Launch one fresh bounded 8-GPU stability probe in detached tmux.
set -euo pipefail

readonly SESSION='lumina_probe'

session_exists() {
    tmux has-session -t "=${1}" 2>/dev/null
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_probe_tmux.sh <ce|attention|gce> [STEPS]
  bash scripts/train/run_probe_tmux.sh status
  bash scripts/train/run_probe_tmux.sh logs

The default stop horizon is 825 optimizer steps. The cosine schedule keeps the
formal 2,750-step horizon. OUTPUT_ROOT must be a fresh root. Set
LUMINA_PROBE_LEARNING_RATE to run a single-variable peak-LR control.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

name_for() {
    case "$1" in
        ce) printf 'PROBE-MB-CE-8G-B4-A1-S42\n' ;;
        attention) printf 'PROBE-MB-ATTN-8G-B4-A1-S42\n' ;;
        gce) printf 'PROBE-MB-GCE-8G-B4-A1-S42\n' ;;
        *) die "unknown probe objective: $1" ;;
    esac
}

require_environment() {
    local variable
    for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES; do
        [[ -n "${!variable:-}" ]] || die "required environment variable is empty: $variable"
    done
    [[ -f "$MODEL_PATH/config.json" ]] || die "missing model config: $MODEL_PATH/config.json"
    [[ -f "$DATA_CONFIG" ]] || die "DATA_CONFIG is not a file: $DATA_CONFIG"
    [[ "$(wc -l < "$DATA_CONFIG")" == '8807' ]] || die "MagicBrush manifest must contain 8807 rows: $DATA_CONFIG"
    local -a devices
    local device
    local -A seen=()
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    [[ "${#devices[@]}" -eq 8 ]] || die "probe requires exactly eight visible GPUs"
    for device in "${devices[@]}"; do
        [[ "$device" =~ ^[0-9]+$ ]] || die "invalid GPU ID in CUDA_VISIBLE_DEVICES"
        [[ -z "${seen[$device]:-}" ]] || die "probe requires eight distinct visible GPUs"
        seen[$device]=1
    done
}

write_probe_config() {
    local objective="$1"
    local steps="$2"
    local source="$PROJECT_ROOT/configs/train/ablation/mb_${objective}_8g_b4_a1.yaml"
    local destination="$OUTPUT_ROOT/probe/configs/mb_${objective}_${steps}.yaml"
    local name
    name="$(name_for "$objective")"
    uv run python - "$source" "$destination" "$name" "$steps" "${LUMINA_PROBE_LEARNING_RATE:-}" <<'PY' || return
from pathlib import Path
import math
import sys
import yaml

source, destination, name, steps, learning_rate = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
)
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["experiment_name"] = config["output_name"] = name
if learning_rate:
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("LUMINA_PROBE_LEARNING_RATE must be finite and positive")
    config["optimization"]["learning_rate"] = learning_rate
formal_horizon = int(config["training"]["max_optimizer_steps"])
if steps > formal_horizon:
    raise ValueError(
        f"probe stop horizon {steps} exceeds formal scheduler horizon {formal_horizon}"
    )
config["training"]["scheduler_horizon_steps"] = formal_horizon
config["training"]["max_optimizer_steps"] = steps
config["training"]["checkpoint_every_steps"] = 275
config["quality_gate"]["baseline_end"] = min(config["quality_gate"]["baseline_end"], steps)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
    printf '%s\n' "$destination"
}

write_launcher() {
    local objective="$1"
    local config="$2"
    local name="$3"
    local launcher="$OUTPUT_ROOT/probe/logs/${objective}.command.sh"
    local log_file="$OUTPUT_ROOT/probe/logs/${objective}.log"
    local exit_file="$OUTPUT_ROOT/probe/logs/${objective}.exit_code"
    local running_file="$OUTPUT_ROOT/probe/logs/${objective}.running"
    local quality_file="$OUTPUT_ROOT/$name/quality_status.json"
    local variable
    umask 077
    mkdir -p "$OUTPUT_ROOT/probe/logs"
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
        printf '  printf %q "$(date -Is)"\n' 'probe_started_at=%s\n'
        printf '  uv run python scripts/train/train.py --config %q\n' "$config"
        printf '} 2>&1 | tee %q\n' "$log_file"
        printf 'status=${PIPESTATUS[0]}\n'
        printf 'if [[ "$status" -ne 0 ]]; then\n'
        printf '  uv run python -m training.quality --mark-process-failed %q --objective %q --exit-code "$status" || true\n' "$quality_file" "$objective"
        printf '  exit "$status"\n'
        printf 'fi\n'
        printf 'uv run python -m training.quality --require-succeeded %q\n' "$quality_file"
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

start() {
    local objective="$1"
    local steps="$2"
    [[ "$steps" =~ ^[1-9][0-9]*$ ]] || die "STEPS must be a positive integer"
    (( steps >= 400 )) || die "stability probe must run at least 400 steps"
    require_environment
    command -v tmux >/dev/null 2>&1 || die 'required command not found: tmux'
    local session
    for session in lumina_prepare lumina_smoke lumina_probe lumina_all lumina_attn lumina_gce lumina_ce; do
        ! session_exists "$session" || die "another Lumina session is already active: $session"
    done
    local name output config launcher
    name="$(name_for "$objective")"
    output="$OUTPUT_ROOT/$name"
    [[ ! -e "$output" ]] || die "probe output already exists: $output"
    [[ ! -e "$OUTPUT_ROOT/probe/logs/${objective}.log" ]] || die "probe log already exists; use a fresh OUTPUT_ROOT"
    [[ ! -e "$OUTPUT_ROOT/probe/logs/${objective}.running" ]] || die "stale probe marker exists"
    config="$(write_probe_config "$objective" "$steps")"
    launcher="$(write_launcher "$objective" "$config" "$name")"
    tmux new-session -d -s "$SESSION" bash "$launcher"
    printf 'started objective=%s steps=%s session=%s\n' "$objective" "$steps" "$SESSION"
    printf 'log=%s\n' "$OUTPUT_ROOT/probe/logs/${objective}.log"
}

status() {
    if session_exists "$SESSION" || find "${OUTPUT_ROOT:-/nonexistent}/probe/logs" -maxdepth 1 -name '*.running' -type f -print -quit 2>/dev/null | grep -q .; then
        printf 'RUNNING session=%s\n' "$SESSION"
    elif [[ -d "${OUTPUT_ROOT:-/nonexistent}/probe/logs" ]]; then
        local file found=0
        for file in "$OUTPUT_ROOT"/probe/logs/*.exit_code; do
            [[ -f "$file" ]] || continue
            found=1
            printf '%s exit_code=%s\n' "$(basename "$file" .exit_code)" "$(< "$file")"
        done
        (( found )) || printf 'NOT STARTED\n'
    else
        printf 'NOT STARTED\n'
    fi
}

case "${1:-help}" in
    ce|attention|gce)
        [[ "$#" -le 2 ]] || die 'usage: run_probe_tmux.sh <ce|attention|gce> [STEPS]'
        start "$1" "${2:-825}"
        ;;
    status) [[ "$#" == 1 ]] || die 'usage: run_probe_tmux.sh status'; status ;;
    logs)
        [[ "$#" == 1 ]] || die 'usage: run_probe_tmux.sh logs'
        file="$(find "${OUTPUT_ROOT:-/nonexistent}/probe/logs" -maxdepth 1 -name '*.log' -type f -print -quit 2>/dev/null)"
        [[ -n "$file" ]] || die 'probe log does not exist'
        tail -f "$file"
        ;;
    help|-h|--help) usage ;;
    *) die "unknown command: ${1:-}" ;;
esac
