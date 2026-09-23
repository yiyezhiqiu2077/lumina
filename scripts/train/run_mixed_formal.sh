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
[[ -f "$config" && -f "$DATA_CONFIG" && -f "$MODEL_PATH/config.json" ]] || {
    printf 'config, data manifest, or model path is invalid\n' >&2; exit 1;
}
nproc="$(python - "$config" <<'PY'
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['distributed']['nproc_per_node'])
PY
)"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" == "$nproc" ]] || {
    printf 'CUDA_VISIBLE_DEVICES has %s devices but config requests %s ranks\n' "${#devices[@]}" "$nproc" >&2
    exit 1
}
command=(python scripts/train/train.py --config "$config")
printf 'nproc_per_node=%s\n' "$nproc"
printf 'command:'; printf ' %q' "${command[@]}"; printf '\n'
if [[ "$mode" == '--run' ]]; then
    exec "${command[@]}"
fi
