#!/usr/bin/env bash
# Start matched 8-GPU objective-ablation training in detached tmux sessions.
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ALL_SESSION='lumina_all'

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_tmux.sh all
  bash scripts/train/run_tmux.sh attention [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh gce [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh ce [--resume-from-checkpoint PATH]
  bash scripts/train/run_tmux.sh status
  bash scripts/train/run_tmux.sh logs <all|attention|gce|ce>

'all' runs Attention -> GCE -> CE serially in one detached tmux session and
stops on the first failure. Required paths must be exported in the calling
shell; they are snapshotted into owner-only launchers.
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
        all) printf '%s\n' "$ALL_SESSION" ;;
        attention) printf '%s\n' 'lumina_attn' ;;
        gce) printf '%s\n' 'lumina_gce' ;;
        ce) printf '%s\n' 'lumina_ce' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

config_for() {
    case "$1" in
        attention) printf '%s\n' 'configs/train/ablation/mb_attention_8g_b4_a1.yaml' ;;
        gce) printf '%s\n' 'configs/train/ablation/mb_gce_8g_b4_a1.yaml' ;;
        ce) printf '%s\n' 'configs/train/ablation/mb_ce_8g_b4_a1.yaml' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

output_name_for() {
    case "$1" in
        attention) printf '%s\n' 'MB-ATTN-8G-B4-A1-S42' ;;
        gce) printf '%s\n' 'MB-GCE-8G-B4-A1-S42' ;;
        ce) printf '%s\n' 'MB-CE-8G-B4-A1-S42' ;;
        *) die "unknown objective: $1 (expected attention, gce, or ce)" ;;
    esac
}

validate_selected_gpus() {
    local devices="$CUDA_VISIBLE_DEVICES"
    local -a device_ids
    local device_id
    local -A seen=()
    IFS=',' read -r -a device_ids <<< "$devices"
    [[ "${#device_ids[@]}" -eq 8 ]] || die "CUDA_VISIBLE_DEVICES must contain exactly 8 device IDs, got: $devices"
    for device_id in "${device_ids[@]}"; do
        [[ "$device_id" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES contains an invalid device ID: $devices"
        [[ -z "${seen[$device_id]:-}" ]] || die "CUDA_VISIBLE_DEVICES must contain 8 distinct device IDs: $devices"
        seen[$device_id]=1
    done
    printf 'selected GPUs: %s\n' "$devices"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$devices" \
            || die "unable to query every selected GPU: $devices"
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

require_objective_assets() {
    local objective="$1"
    local config_relative
    config_relative="$(config_for "$objective")"
    [[ -f "$PROJECT_ROOT/$config_relative" ]] || die "missing objective config: $PROJECT_ROOT/$config_relative"
    if [[ "$objective" == 'gce' ]]; then
        [[ -n "${GCE_CLUSTER_PATH:-}" ]] || die 'required environment variable is empty: GCE_CLUSTER_PATH'
        [[ -f "$GCE_CLUSTER_PATH" ]] || die "GCE_CLUSTER_PATH is not a file: $GCE_CLUSTER_PATH"
    fi
}

reject_existing_fresh_output() {
    local objective="$1"
    local output_dir="$OUTPUT_ROOT/$(output_name_for "$objective")"
    local checkpoint
    for checkpoint in train_metrics.jsonl experiment_config.json lora_report.json quality_status.json; do
        [[ ! -e "$output_dir/$checkpoint" ]] || die "existing experiment output detected: $output_dir (use a different OUTPUT_ROOT or use resume)"
    done
    for checkpoint in "$output_dir"/checkpoint-*; do
        [[ ! -e "$checkpoint" ]] || die "existing experiment output detected: $output_dir (use a different OUTPUT_ROOT or use resume)"
    done
    for checkpoint in \
        "$OUTPUT_ROOT/logs/${objective}.log" \
        "$OUTPUT_ROOT/logs/${objective}.command.sh" \
        "$OUTPUT_ROOT/logs/${objective}.exit_code" \
        "$OUTPUT_ROOT/logs/${objective}.running"; do
        [[ ! -e "$checkpoint" ]] || die "existing experiment launcher artifact detected: $checkpoint (use a different OUTPUT_ROOT)"
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
    [[ -f "$checkpoint/_SUCCESS" ]] || die "resume checkpoint is incomplete (missing _SUCCESS): $checkpoint"
    [[ -f "$checkpoint/checkpoint_meta.json" ]] || die "resume checkpoint metadata is missing: $checkpoint"
    [[ -f "$checkpoint/lora.pt" ]] || die "resume checkpoint LoRA state is missing: $checkpoint"
    local rank
    for rank in {0..7}; do
        printf -v rank '%02d' "$rank"
        [[ -f "$checkpoint/training_state.rank${rank}.pt" ]] \
            || die "resume checkpoint rank state is missing: training_state.rank${rank}.pt"
    done
    [[ -f "$output_dir/train_metrics.jsonl" ]] || die "resume metrics are missing: $output_dir/train_metrics.jsonl"
}

session_exists() {
    tmux has-session -t "=${1}" 2>/dev/null
}

reject_stale_marker() {
    local name="$1"
    local marker="$OUTPUT_ROOT/logs/${name}.running"
    [[ ! -f "$marker" ]] || die "stale running marker detected; inspect before retrying: $marker"
}

reject_active_sessions() {
    local requested_session="$1"
    if session_exists "$requested_session"; then
        die "session already exists: $requested_session"
    fi

    local session
    for session in lumina_prepare "$ALL_SESSION" lumina_attn lumina_gce lumina_ce lumina_smoke; do
        [[ "$session" == "$requested_session" ]] && continue
        if [[ "$session" == 'lumina_prepare' && "${LUMINA_PIPELINE_PARENT:-}" == '1' ]]; then
            continue
        fi
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
    local quality_file="$OUTPUT_ROOT/$(output_name_for "$objective")/quality_status.json"
    local config_path="$PROJECT_ROOT/$config_relative"
    local variable

    umask 077
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' 'set -o pipefail'
        for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES PATH; do
            printf 'export %s=%q\n' "$variable" "${!variable:-}"
        done
        printf 'unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT\n'
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
        printf 'if [[ "$TRAIN_STATUS" -ne 0 ]]; then\n'
        printf '  uv run python -m training.quality --mark-process-failed %q --objective %q --exit-code "$TRAIN_STATUS" || true\n' "$quality_file" "$objective"
        printf '  exit "$TRAIN_STATUS"\n'
        printf 'fi\n'
        printf 'uv run python -m training.quality --require-succeeded %q\n' "$quality_file"
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

write_all_launcher() {
    local launcher="$OUTPUT_ROOT/logs/all.command.sh"
    local log_file="$OUTPUT_ROOT/logs/all.log"
    local exit_file="$OUTPUT_ROOT/logs/all.exit_code"
    local running_file="$OUTPUT_ROOT/logs/all.running"
    local variable objective objective_launcher

    umask 077
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' 'set -o pipefail'
        for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES PATH; do
            printf 'export %s=%q\n' "$variable" "${!variable:-}"
        done
        printf 'cd %q\n' "$PROJECT_ROOT"
        printf 'rm -f %q\n' "$exit_file"
        printf 'touch %q\n' "$running_file"
        printf 'finish() { local status=$?; trap - EXIT; rm -f %q; printf %q "$status" > %q; exit "$status"; }\n' \
            "$running_file" '%s\n' "$exit_file"
        printf 'trap finish EXIT\n'
        printf '{\n'
        printf '  printf %q "$(date -Is)"\n' 'queue_started_at=%s\n'
        for objective in attention gce ce; do
            objective_launcher="$OUTPUT_ROOT/logs/${objective}.command.sh"
            printf '  printf %q "$(date -Is)"\n' "objective=$objective state=starting at=%s\n"
            printf '  if bash %q; then\n' "$objective_launcher"
            printf '    printf %q "$(date -Is)"\n' "objective=$objective state=finished exit_code=0 at=%s\n"
            printf '  else\n'
            printf '    status=$?\n'
            printf '    printf %q "$status" "$(date -Is)"\n' "objective=$objective state=failed exit_code=%s at=%s\n"
            printf '    exit "$status"\n'
            printf '  fi\n'
        done
        printf '  printf %q "$(date -Is)"\n' 'queue_finished_at=%s\n'
        printf '} 2>&1 | tee %q\n' "$log_file"
        printf 'QUEUE_STATUS=${PIPESTATUS[0]}\n'
        printf 'exit "$QUEUE_STATUS"\n'
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
    require_objective_assets "$objective"
    reject_active_sessions "$session"
    reject_stale_marker "$objective"
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

start_all() {
    local objective launcher
    require_command tmux
    require_environment
    reject_active_sessions "$ALL_SESSION"
    reject_stale_marker all
    for objective in attention gce ce; do
        reject_stale_marker "$objective"
        require_objective_assets "$objective"
        reject_existing_fresh_output "$objective"
        rm -f "$OUTPUT_ROOT/logs/${objective}.exit_code"
        write_launcher "$objective" "$(config_for "$objective")" '' >/dev/null
    done
    rm -f "$OUTPUT_ROOT/logs/all.exit_code"
    launcher="$(write_all_launcher)"
    tmux new-session -d -s "$ALL_SESSION" bash "$launcher"
    printf 'started objectives=attention,gce,ce session=%s\n' "$ALL_SESSION"
    printf 'log=%s\n' "$OUTPUT_ROOT/logs/all.log"
    printf 'launcher=%s\n' "$launcher"
}

status_for() {
    local name="$1"
    local session="$2"
    local exit_file="${OUTPUT_ROOT:-}/logs/${name}.exit_code"
    local running_file="${OUTPUT_ROOT:-}/logs/${name}.running"
    local status='NOT STARTED'
    local quality_file=''
    if [[ "$name" != 'all' && -n "${OUTPUT_ROOT:-}" ]]; then
        quality_file="$OUTPUT_ROOT/$(output_name_for "$name")/quality_status.json"
    fi
    if session_exists "$session" || [[ -f "$running_file" ]]; then
        status='RUNNING'
    elif [[ -n "${OUTPUT_ROOT:-}" && -f "$exit_file" ]]; then
        local exit_code quality_status=''
        exit_code="$(< "$exit_file")"
        if [[ -n "$quality_file" && -f "$quality_file" ]]; then
            quality_status="$(python3 - "$quality_file" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "UNKNOWN"))
PY
)"
        fi
        if [[ "$exit_code" == '0' && "$quality_status" == 'SUCCEEDED' ]]; then
            status='SUCCEEDED exit_code=0 quality=SUCCEEDED'
        elif [[ "$quality_status" == 'QUALITY_FAILED' ]]; then
            status="QUALITY_FAILED exit_code=$exit_code"
        elif [[ "$exit_code" != '0' ]]; then
            status="PROCESS_FAILED exit_code=$exit_code quality=${quality_status:-UNKNOWN}"
        elif [[ "$name" == 'all' ]]; then
            status='SUCCEEDED exit_code=0'
        else
            status="PROCESS_FAILED exit_code=0 quality=${quality_status:-MISSING}"
        fi
    fi
    printf '%-10s %-12s %s\n' "$name" "$session" "$status"
}

show_status() {
    require_command tmux
    status_for all "$ALL_SESSION"
    local objective
    for objective in attention gce ce; do
        status_for "$objective" "$(session_for "$objective")"
    done
}

show_logs() {
    local name="$1"
    case "$name" in
        all|attention|gce|ce) ;;
        *) die 'logs expects all, attention, gce, or ce' ;;
    esac
    [[ -n "${OUTPUT_ROOT:-}" ]] || die 'required environment variable is empty: OUTPUT_ROOT'
    local log_file="$OUTPUT_ROOT/logs/${name}.log"
    [[ -f "$log_file" ]] || die "log file does not exist: $log_file"
    tail -f "$log_file"
}

main() {
    local command="${1:-help}"
    case "$command" in
        all)
            [[ "$#" == 1 ]] || die 'usage: run_tmux.sh all'
            start_all
            ;;
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
            [[ "$#" == 2 ]] || die 'usage: run_tmux.sh logs <all|attention|gce|ce>'
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
