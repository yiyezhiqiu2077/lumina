#!/usr/bin/env bash
# Start one matched objective-ablation training run in a detached tmux session.
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_tmux.sh attention
  bash scripts/train/run_tmux.sh gce
  bash scripts/train/run_tmux.sh ce
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
    local launcher="$OUTPUT_ROOT/logs/${objective}.command.sh"
    local log_file="$OUTPUT_ROOT/logs/${objective}.log"
    local exit_file="$OUTPUT_ROOT/logs/${objective}.exit_code"
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
        printf '  uv run python scripts/train/train.py --config %q\n' "$config_path"
        printf '} 2>&1 | tee %q\n' "$log_file"
        printf 'TRAIN_STATUS=${PIPESTATUS[0]}\n'
        printf 'printf %q "$TRAIN_STATUS" > %q\n' '%s\n' "$exit_file"
        printf 'exit "$TRAIN_STATUS"\n'
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

start_objective() {
    local objective="$1"
    local session config_relative launcher
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

    launcher="$(write_launcher "$objective" "$config_relative")"
    tmux new-session -d -s "$session" bash "$launcher"
    printf 'started objective=%s session=%s\n' "$objective" "$session"
    printf 'log=%s\n' "$OUTPUT_ROOT/logs/${objective}.log"
    printf 'launcher=%s\n' "$launcher"
}

show_status() {
    require_command tmux
    local objective session exit_file status='NOT RUNNING'
    for objective in attention gce ce; do
        session="$(session_for "$objective")"
        if session_exists "$session"; then
            status='RUNNING'
        else
            status='NOT RUNNING'
        fi
        printf '%-10s %-12s %s' "$objective" "$session" "$status"
        if [[ -n "${OUTPUT_ROOT:-}" ]]; then
            exit_file="$OUTPUT_ROOT/logs/${objective}.exit_code"
            if [[ -f "$exit_file" ]]; then
                printf ' exit_code=%s' "$(< "$exit_file")"
            fi
        fi
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
            [[ "$#" == 1 ]] || die "usage: run_tmux.sh <attention|gce|ce>"
            start_objective "$command"
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
