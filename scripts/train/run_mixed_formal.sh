#!/usr/bin/env bash
# Construct (or, with --run, start) one portable 8-GPU mixed-data formal run.
set -euo pipefail

usage() {
    printf 'Usage: %s <config.yaml> [--run]\n' "$0" >&2
}

[[ "$#" == 1 || "$#" == 2 ]] || { usage; exit 2; }
config="$1"
mode="${2:---print-command}"
[[ "$mode" == '--print-command' || "$mode" == '--run' ]] || { usage; exit 2; }
for variable in MODEL_PATH DATA_ROOT DATA_CONFIG OUTPUT_ROOT CUDA_VISIBLE_DEVICES; do
    [[ -n "${!variable:-}" ]] || { printf 'missing %s\n' "$variable" >&2; exit 1; }
done
[[ -f "$config" && -f "$DATA_CONFIG" && -d "$DATA_ROOT" && -d "$OUTPUT_ROOT" && -f "$MODEL_PATH/config.json" ]] || {
    printf 'config, data manifest, data root, output root, or model path is invalid\n' >&2; exit 1;
}
[[ -n "$(command -v uv)" ]] || { printf 'uv is required for the mixed-data workflow\n' >&2; exit 1; }
read -r nproc objective <<< "$(uv run python - "$config" <<'PY'
import sys
import yaml
payload = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
print(payload['distributed']['nproc_per_node'], payload['objective']['mode'])
PY
)"
if [[ "$objective" == 'gce' ]]; then
    [[ -n "${GCE_CLUSTER_PATH:-}" && -f "$GCE_CLUSTER_PATH" ]] || {
        printf 'GCE_CLUSTER_PATH must name an existing cluster file for GCE runs\n' >&2; exit 1;
    }
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" == "$nproc" ]] || {
    printf 'CUDA_VISIBLE_DEVICES has %s devices but config requests %s ranks\n' "${#devices[@]}" "$nproc" >&2
    exit 1
}
command=(uv run python scripts/train/train.py --config "$config")
printf 'nproc_per_node=%s\n' "$nproc"
printf 'command:'; printf ' %q' "${command[@]}"; printf '\n'
if [[ "$mode" == '--run' ]]; then
    exec "${command[@]}"
fi
