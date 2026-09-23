#!/usr/bin/env bash
# Serial 20-step correctness smokes.  It never chooses GPUs itself.
set -euo pipefail

mode="${1:---print-command}"
[[ "$mode" == '--print-command' || "$mode" == '--run' ]] || {
    printf 'Usage: %s [--print-command|--run]\n' "$0" >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
configs=(
    "$root/configs/train/validation/mixed_ce_4g.yaml"
    "$root/configs/train/validation/mixed_attention_4g.yaml"
    "$root/configs/train/validation/mixed_gce_4g.yaml"
)
for config in "${configs[@]}"; do
    "$root/scripts/train/run_mixed_formal.sh" "$config" --print-command
    if [[ "$mode" == '--run' ]]; then
        "$root/scripts/train/run_mixed_formal.sh" "$config" --run
    fi
done
