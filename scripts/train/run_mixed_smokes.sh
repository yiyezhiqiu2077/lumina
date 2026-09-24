#!/usr/bin/env bash
# Serial 20-step correctness smokes.  It never chooses GPUs itself.
set -euo pipefail

usage() {
    printf 'Usage: %s [full|editregion|all] [--print-command|--run]\n' "$0" >&2
}

selector="${1:-full}"
mode="${2:---print-command}"
if [[ "$selector" == '--print-command' || "$selector" == '--run' ]]; then
    mode="$selector"
    selector=full
fi
[[ "$selector" == full || "$selector" == editregion || "$selector" == all ]] || { usage; exit 2; }
[[ "$mode" == '--print-command' || "$mode" == '--run' ]] || { usage; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
full_configs=(
    "$root/configs/train/validation/mixed_ce_4g.yaml"
    "$root/configs/train/validation/mixed_attention_4g.yaml"
    "$root/configs/train/validation/mixed_gce_4g.yaml"
)
editregion_configs=(
    "$root/configs/train/validation/mixed_ce_editregion_4g.yaml"
    "$root/configs/train/validation/mixed_attention_editregion_4g.yaml"
    "$root/configs/train/validation/mixed_gce_editregion_4g.yaml"
)
configs=()
case "$selector" in
  full) configs=("${full_configs[@]}") ;;
  editregion) configs=("${editregion_configs[@]}") ;;
  all) configs=("${full_configs[@]}" "${editregion_configs[@]}") ;;
esac
for config in "${configs[@]}"; do
    "$root/scripts/train/run_mixed_formal.sh" "$config" --print-command
    if [[ "$mode" == '--run' ]]; then
        "$root/scripts/train/run_mixed_formal.sh" "$config" --run
    fi
done
