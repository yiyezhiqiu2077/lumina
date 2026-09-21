#!/usr/bin/env bash
# Prepare assets, run 8-GPU smokes, then launch the formal objective queue.
set -euo pipefail

readonly SESSION='lumina_prepare'

session_exists() {
    tmux has-session -t "=${1}" 2>/dev/null
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/run_pipeline_tmux.sh start
  bash scripts/train/run_pipeline_tmux.sh status
  bash scripts/train/run_pipeline_tmux.sh logs

The detached preparation session performs:
  shared geometry -> 8-GPU VQ tokenization -> GCE clusters -> data audit
It then starts the detached 8-GPU Attention/GCE smoke queue, waits for success,
and finally starts the detached formal Attention -> GCE -> CE queue.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_environment() {
    local variable
    for variable in PROJECT_ROOT ASSET_ROOT DATA_ROOT MODEL_PATH OUTPUT_ROOT MAGICBRUSH_DATA_CONFIG DATA_CONFIG GCE_CLUSTER_PATH CUDA_VISIBLE_DEVICES; do
        [[ -n "${!variable:-}" ]] || die "required environment variable is empty: $variable"
    done
    [[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT is not a directory: $PROJECT_ROOT"
    [[ -f "$MODEL_PATH/config.json" ]] || die "missing model config: $MODEL_PATH/config.json"
    [[ -d "$MODEL_PATH/vqvae" ]] || die "missing model VQ-VAE directory: $MODEL_PATH/vqvae"
    [[ -f "$DATA_ROOT/magicbrush/raw/train.jsonl" ]] || die "missing raw manifest: $DATA_ROOT/magicbrush/raw/train.jsonl"
    [[ "$(wc -l < "$DATA_ROOT/magicbrush/raw/train.jsonl")" == '8807' ]] || die 'raw manifest must contain 8807 rows'
    [[ "$CUDA_VISIBLE_DEVICES" == '0,1,2,3,4,5,6,7' ]] || die 'pipeline expects CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'
    command -v tmux >/dev/null 2>&1 || die 'required command not found: tmux'
    command -v uv >/dev/null 2>&1 || die 'required command not found: uv'
}

write_launcher() {
    local launcher="$OUTPUT_ROOT/logs/pipeline.command.sh"
    local log_file="$OUTPUT_ROOT/logs/pipeline.log"
    local exit_file="$OUTPUT_ROOT/logs/pipeline.exit_code"
    local running_file="$OUTPUT_ROOT/logs/pipeline.running"
    local variable

    umask 077
    mkdir -p "$OUTPUT_ROOT/logs"
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' 'set -euo pipefail'
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
        printf '  printf %q "$(date -Is)"\n' 'pipeline_started_at=%s\n'
        printf '  printf %q\n' 'stage=geometry state=starting'
        printf "  uv run python - <<'PY'\n"
        printf '%s\n' \
            'import os' \
            'from pathlib import Path' \
            'from dataset.geometry import enrich_geometry, resolve_record_paths' \
            'from dataset.utils import read_jsonl, write_jsonl' \
            '' \
            'root = Path(os.environ["DATA_ROOT"]) / "magicbrush"' \
            'raw, prepared = root / "raw/train.jsonl", root / "prepared/train.jsonl"' \
            'rows = [resolve_record_paths(row, raw) for row in read_jsonl(raw)]' \
            'if len(rows) != 8807:' \
            '    raise RuntimeError(f"Expected 8807 raw samples, got {len(rows)}")' \
            'write_jsonl(prepared, enrich_geometry(rows, seed=42, target_size=512))' \
            'print(prepared)' \
            'PY'
        printf '  test "$(wc -l < "$DATA_ROOT/magicbrush/prepared/train.jsonl")" = 8807\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=geometry state=finished at=%s\n'
        printf '  printf %q\n' 'stage=pretokenize state=starting world_size=8'
        printf '  uv run python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/data/preprocess_magicbrush.py pretokenize --manifest "$DATA_ROOT/magicbrush/prepared/train.jsonl" --model "$MODEL_PATH" --output "$DATA_ROOT/magicbrush/tokens/train"\n'
        printf '  test "$(wc -l < "$DATA_CONFIG")" = 8807\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=pretokenize state=finished at=%s\n'
        printf '  printf %q\n' 'stage=gce_clusters state=starting'
        printf '  uv run python scripts/tools/gce/build_clusters.py --model "$MODEL_PATH" --output "$GCE_CLUSTER_PATH" --device cuda:0 --levels 1024 512 --seed 0\n'
        printf '  test -f "$GCE_CLUSTER_PATH"\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=gce_clusters state=finished at=%s\n'
        printf '  printf %q\n' 'stage=data_audit state=starting'
        printf '  uv run python scripts/data/check_magicbrush.py all --manifest "$DATA_CONFIG" --model "$MODEL_PATH" --output "$OUTPUT_ROOT/data_audit"\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=data_audit state=finished at=%s\n'
        printf '  printf %q\n' 'stage=smokes state=starting'
        printf '  LUMINA_PIPELINE_PARENT=1 bash scripts/train/run_smokes_tmux.sh start\n'
        printf '  while [[ ! -f "$OUTPUT_ROOT/smoke/logs/smokes.exit_code" ]]; do sleep 10; done\n'
        printf '  smoke_status="$(< "$OUTPUT_ROOT/smoke/logs/smokes.exit_code")"\n'
        printf '  [[ "$smoke_status" == 0 ]] || { printf %q "$smoke_status"; exit "$smoke_status"; }\n' 'stage=smokes state=failed exit_code=%s\n'
        printf '  while tmux has-session -t =lumina_smoke 2>/dev/null; do sleep 1; done\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=smokes state=finished at=%s\n'
        printf '  printf %q\n' 'stage=formal_queue state=starting'
        printf '  LUMINA_PIPELINE_PARENT=1 bash scripts/train/run_tmux.sh all\n'
        printf '  printf %q "$(date -Is)"\n' 'stage=formal_queue state=launched at=%s\n'
        printf '} 2>&1 | tee %q\n' "$log_file"
    } > "$launcher"
    chmod 700 "$launcher"
    printf '%s\n' "$launcher"
}

start() {
    require_environment
    local session
    for session in lumina_prepare lumina_smoke lumina_all lumina_attn lumina_gce lumina_ce; do
        ! session_exists "$session" || die "another Lumina session is already active: $session"
    done
    [[ ! -e "$DATA_CONFIG" ]] || die "token manifest already exists; use the individual launchers instead: $DATA_CONFIG"
    [[ ! -e "$GCE_CLUSTER_PATH" ]] || die "GCE cluster already exists; use the individual launchers instead: $GCE_CLUSTER_PATH"
    rm -f "$OUTPUT_ROOT/logs/pipeline.exit_code"
    local launcher
    launcher="$(write_launcher)"
    tmux new-session -d -s "$SESSION" bash "$launcher"
    printf 'started session=%s\n' "$SESSION"
    printf 'log=%s\n' "$OUTPUT_ROOT/logs/pipeline.log"
    printf 'progress=%s\n' "$OUTPUT_ROOT/logs/progress.log"
}

status() {
    if session_exists "$SESSION" || [[ -f "${OUTPUT_ROOT:-}/logs/pipeline.running" ]]; then
        printf 'RUNNING session=%s\n' "$SESSION"
    elif [[ -f "${OUTPUT_ROOT:-}/logs/pipeline.exit_code" ]]; then
        printf 'FINISHED exit_code=%s\n' "$(< "$OUTPUT_ROOT/logs/pipeline.exit_code")"
    else
        printf 'NOT STARTED\n'
    fi
}

case "${1:-help}" in
    start) [[ "$#" == 1 ]] || die 'usage: run_pipeline_tmux.sh start'; start ;;
    status) [[ "$#" == 1 ]] || die 'usage: run_pipeline_tmux.sh status'; status ;;
    logs)
        [[ "$#" == 1 ]] || die 'usage: run_pipeline_tmux.sh logs'
        [[ -f "${OUTPUT_ROOT:-}/logs/pipeline.log" ]] || die 'pipeline log does not exist'
        tail -f "$OUTPUT_ROOT/logs/pipeline.log"
        ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $1" ;;
esac
