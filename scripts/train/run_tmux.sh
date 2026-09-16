#!/usr/bin/env bash
# Start one matched objective-ablation training run in a detached tmux session.
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_tmux.sh attention [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh gce [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh ce [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh status
  bash scripts/train/run_tmux.sh logs <attention|gce|ce>

Each objective starts a detached tmux session. Required paths must be exported
in the calling shell; they are snapshotted into an objective-specific launcher.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

session_for() {
    case "$1" in
        attention) printf '%s\n' 'lumina_attn' ;;
        gce) printf '%s\n' 'lumina_gce' ;;
        ce) printf '%s\n' 'lumina_ce' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

config_for() {
    case "$1" in
        attention) printf '%s\n' 'configs/train/ablation/mb_attention_2g_b8_a2.yaml' ;;
        gce) printf '%s\n' 'configs/train/ablation/mb_gce_2g_b8_a2.yaml' ;;
        ce) printf '%s\n' 'configs/train/ablation/mb_ce_2g_b8_a2.yaml' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

output_name_for() {
    case "$1" in
        attention) printf '%s\n' 'MB-ATTN-2G-B8-A2-S42' ;;
        gce) printf '%s\n' 'MB-GCE-2G-B8-A2-S42' ;;
        ce) printf '%s\n' 'MB-CE-2G-B8-A2-S42' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

validate_selected_gpus() {
    local devices="$CUDA_VISIBLE_DEVICES"
    local -a device_ids
    local device_id
    IFS=',' read -r -a device_ids <<< "$devices"
    [[ "${#device_ids[@]}" -eq 2 ]] || die "CUDA_VISIBLE_DEVICES must contain exactly 2 device IDs, got: $devices"
    for device_id in "${device_ids[@]}"; do
        [[ "$device_id" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES contains an invalid device ID: $devices"
    done
    [[ "${device_ids[0]}" != "${device_ids[1]}" ]] || die "CUDA_VISIBLE_DEVICES must contain two distinct device IDs: $devices"
    printf 'selected GPUs: %s\n' "$devices"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$devices" \
            || printf 'warning: unable to query selected GPU utilization; no processes were changed\n' >&2
    fi
}

require_environment() {
    local variable
    for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG CUDA_VISIBLE_DEVICES; do
        [[ -n "${!variable:-}" ]] || die "required environment variable is empty: $variable"
    done

    [[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT is not a directory: $PROJECT_ROOT"
    [[ -d "$MODEL_PATH" ]] || die "MODEL_PATH is not a directory: $MODEL_PATH"
    [[ -f "$MODEL_PATH/config.json" ]] || die "missing model config: $MODEL_PATH/config.json"
    [[ -d "$MODEL_PATH/vqvae" ]] || die "missing model VQ-VAE directory: $MODEL_PATH/vqvae"
    [[ -f "$DATA_CONFIG" ]] || die "DATA_CONFIG is not a file: $DATA_CONFIG"

    mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" || die "cannot create OUTPUT_ROOT: $OUTPUT_ROOT"
    [[ -w "$OUTPUT_ROOT/logs" ]] || die "OUTPUT_ROOT/logs is not writable: $OUTPUT_ROOT/logs"

    local manifest_count
    manifest_count="$(wc -l < "$DATA_CONFIG")"
    [[ "$manifest_count" == '8807' ]] || die "MagicBrush manifest must contain 8807 rows, got $manifest_count: $DATA_CONFIG"
    validate_selected_gpus
}

reject_existing_fresh_output() {
    local objective="$1"
    local output_dir="$OUTPUT_ROOT/$(output_name_for "$objective")"
    local checkpoint
    for checkpoint in train_metrics.jsonl experiment_config.json lora_report.json; do
        [[ ! -e "$output_dir/$checkpoint" ]] || die "existing experiment output detected: $output_dir (use a different OUTPUT_ROOT or use resume)"
    done
    for checkpoint in "$output_dir"/checkpoint-*; do
        [[ ! -e "$checkpoint" ]] || die "existing experiment output detected: $output_dir (use a different OUTPUT_ROOT or use resume)"
    done
}

validate_resume_checkpoint() {
    local objective="$1"
    local checkpoint="$2"
    local output_dir="$OUTPUT_ROOT/$(output_name_for "$objective")"
    [[ -d "$checkpoint" ]] || die "resume checkpoint is not a directory: $checkpoint"
    [[ -d "$output_dir" ]] || die "resume output directory does not exist: $output_dir"
    local checkpoint_real output_real
    checkpoint_real="$(realpath -e "$checkpoint")"
    output_real="$(realpath -e "$output_dir")"
    [[ "$(dirname "$checkpoint_real")" == "$output_real" && "$(basename "$checkpoint_real")" == checkpoint-* ]] \
        || die "resume checkpoint must be inside the correct $objective experiment directory: $output_dir"
}

session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

reject_active_sessions() {
    local requested_session="$1"
    if session_exists "$requested_session"; then
        die "session already exists: $requested_session"
    fi

    local session
    for session in lumina_attn lumina_gce lumina_ce; do
        [[ "$session" == "$requested_session" ]] && continue
        if session_exists "$session"; then
            die "another Lumina training session is already active: $session"
        fi
    done
}

write_launcher() {
    local objective="$1"
    local config_relative="$2"
    local resume_checkpoint="$3"
    local launcher="$OUTPUT_ROOT/logs/${objective}.command.sh"
    local log_file="$OUTPUT_ROOT/logs/${objective}.log"
    local exit_file="$OUTPUT_ROOT/logs/${objective}.exit_code"
    local running_file="$OUTPUT_ROOT/logs/${objective}.running"
    local config_path="$PROJECT_ROOT/$config_relative"
    local variable

    umask 077
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' 'set -o pipefail'
        for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES PATH; do
            printf 'export %s=%q\n' "$variable" "${!variable:-}"
        done
        printf 'cd %q\n' "$PROJECT_ROOT"
        printf 'mkdir -p %q\n' "$OUTPUT_ROOT/logs"
        printf 'rm -f %q\n' "$exit_file"
        printf 'touch %q\n' "$running_file"
        printf 'finish() { local status=$?; trap - EXIT; rm -f %q; printf %q "$status" > %q; exit "$status"; }\n' \
            "$running_file" '%s\n' "$exit_file"
        printf 'trap finish EXIT\n'
        printf '{\n'
        printf '  printf %q "$(date -Is)"\n' 'started_at=%s\n'
        printf '  printf %q "$(hostname)"\n' 'hostname=%s\n'
        printf '  printf %q "$(pwd)"\n' 'pwd=%s\n'
        printf '  printf %q "$(git rev-parse HEAD)"\n' 'git_commit=%s\n'
        printf '  printf %q "$CUDA_VISIBLE_DEVICES"\n' 'cuda_visible_devices=%s\n'
        printf '  printf %q "$MODEL_PATH"\n' 'model_path=%s\n'
        printf '  printf %q "$DATA_CONFIG"\n' 'data_config=%s\n'
        printf '  printf %q "$OUTPUT_ROOT"\n' 'output_root=%s\n'
        printf '  printf %q %q\n' 'config=%s\n' "$config_path"
        if [[ -n "$resume_checkpoint" ]]; then
            printf '  uv run python scripts/train/train.py --config %q --resume-from-checkpoint %q\n' "$config_path" "$resume_checkpoint"
        else
            printf '  uv run python scripts/train/train.py --config %q\n' "$config_path"
        fi
        printf '} 2>&1 | tee %q\n' "$log_file"
        printf 'TRAIN_STATUS=${PIPESTATUS[0]}\n'
        printf 'exit "$TRAIN_STATUS"\n'
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

start_objective() {
    local objective="$1"
    local resume_checkpoint="$2"
    local session config_relative launcher exit_file
    session="$(session_for "$objective")"
    config_relative="$(config_for "$objective")"

    require_command tmux
    require_environment
    [[ -f "$PROJECT_ROOT/$config_relative" ]] || die "missing objective config: $PROJECT_ROOT/$config_relative"
    if [[ "$objective" == 'gce' ]]; then
        [[ -n "${GCE_CLUSTER_PATH:-}" ]] || die 'required environment variable is empty: GCE_CLUSTER_PATH'
        [[ -f "$GCE_CLUSTER_PATH" ]] || die "GCE_CLUSTER_PATH is not a file: $GCE_CLUSTER_PATH"
    fi
    reject_active_sessions "$session"
    if [[ -n "$resume_checkpoint" ]]; then
        validate_resume_checkpoint "$objective" "$resume_checkpoint"
    else
        reject_existing_fresh_output "$objective"
    fi
    exit_file="$OUTPUT_ROOT/logs/${objective}.exit_code"
    rm -f "$exit_file"

    launcher="$(write_launcher "$objective" "$config_relative" "$resume_checkpoint")"
    tmux new-session -d -s "$session" bash "$launcher"
    printf 'started objective=%s session=%s\n' "$objective" "$session"
    printf 'log=%s\n' "$OUTPUT_ROOT/logs/${objective}.log"
    printf 'launcher=%s\n' "$launcher"
}

show_status() {
    require_command tmux
    local objective session exit_file running_file status='NOT STARTED'
    for objective in attention gce ce; do
        session="$(session_for "$objective")"
        running_file="${OUTPUT_ROOT:-}/logs/${objective}.running"
        if session_exists "$session" || [[ -f "$running_file" ]]; then
            status='RUNNING'
        elif [[ -n "${OUTPUT_ROOT:-}" && -f "$OUTPUT_ROOT/logs/${objective}.exit_code" ]]; then
            exit_file="$OUTPUT_ROOT/logs/${objective}.exit_code"
            if [[ "$(< "$exit_file")" == '0' ]]; then
                status='FINISHED exit_code=0'
            else
                status="FAILED exit_code=$(< "$exit_file")"
            fi
        else
            status='NOT STARTED'
        fi
        printf '%-10s %-12s %s' "$objective" "$session" "$status"
        printf '\n'
    done
}

show_logs() {
    local objective="$1"
    session_for "$objective" >/dev/null
    [[ -n "${OUTPUT_ROOT:-}" ]] || die 'required environment variable is empty: OUTPUT_ROOT'
    local log_file="$OUTPUT_ROOT/logs/${objective}.log"
    [[ -f "$log_file" ]] || die "log file does not exist: $log_file"
    tail -f "$log_file"
}

main() {
    local command="${1:-help}"
    case "$command" in
        attention|gce|ce)
            if [[ "$#" == 1 ]]; then
                start_objective "$command" ''
            elif [[ "$#" == 3 && "$2" == '--resume-from-checkpoint' ]]; then
                start_objective "$command" "$3"
            else
                die "usage: run_tmux.sh <attention|gce|ce> [--resume-from-checkpoint PATH]"
            fi
            ;;
        status)
            [[ "$#" == 1 ]] || die 'usage: run_tmux.sh status'
            show_status
            ;;
        logs)
            [[ "$#" == 2 ]] || die 'usage: run_tmux.sh logs <attention|gce|ce>'
            show_logs "$2"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            die "unknown command: $command (run 'bash scripts/train/run_tmux.sh help')"
            ;;
    esac
}

main "$@"
