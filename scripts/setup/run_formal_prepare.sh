#!/usr/bin/env bash
# Cold-start preparation only.  It never trains a model.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
assets_root="${ASSET_ROOT:-$root/local_assets}"
config="${FORMAL_ASSETS_CONFIG:-$root/configs/formal_assets.yaml}"
mode="print"
scope="all"
for argument in "$@"; do
  case "$argument" in
    --print-command) mode=print ;;
    --run) mode=run ;;
    --train-only) scope=train ;;
    --eval-only) scope=eval ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done
mkdir -p "$assets_root/logs" "$assets_root/.formal_prepare"
log="$assets_root/logs/formal_prepare.log"

run_stage() {
  local name="$1"; shift
  local marker="$assets_root/.formal_prepare/${name}._SUCCESS"
  if [[ "$mode" == print ]]; then printf '%q ' "$@"; printf '\n'; return 0; fi
  if [[ -f "$marker" ]]; then
    printf 'SKIPPED %s (verified marker)\n' "$name" | tee -a "$log"
    return 0
  fi
  printf 'START %s\n' "$name" | tee -a "$log"
  if "$@" >>"$log" 2>&1; then
    printf 'SUCCEEDED %s\n' "$name" | tee -a "$log"
    : > "$marker"
  else
    printf 'FAILED %s\n' "$name" | tee -a "$log" >&2
    return 1
  fi
}

python_cmd=(uv run python)
if [[ "$scope" != eval ]]; then
  run_stage environment uv sync --frozen --extra dev --extra upstream --extra analysis --extra eval --extra data
  run_stage download_models "${python_cmd[@]}" "$root/scripts/setup/download_formal_models.py" --assets-root "$assets_root" --config "$config" --only lumina
  run_stage magicbrush_train_download "${python_cmd[@]}" "$root/scripts/data/download_magicbrush_train.py" --assets-root "$assets_root" --config "$config"
  run_stage magicbrush_geometry "${python_cmd[@]}" "$root/scripts/data/download_magicbrush_train.py" --assets-root "$assets_root" --config "$config" --geometry-only
  run_stage magicbrush_tokenize torchrun --nproc_per_node="${TOKENIZE_GPUS:-8}" "$root/scripts/data/preprocess_magicbrush.py" pretokenize --manifest "$assets_root/datasets/magicbrush/prepared/train.jsonl" --model "$assets_root/models/Lumina-DiMOO" --output "$assets_root/datasets/magicbrush/tokens/train"
  run_stage refedit_download "${python_cmd[@]}" "$root/scripts/data/download_refedit.py" --assets-root "$assets_root" --config "$config"
  run_stage refedit_audit "${python_cmd[@]}" "$root/scripts/data/preprocess_refedit.py" audit --raw-root "$assets_root/datasets/refedit/raw" --output "$assets_root/datasets/refedit/audit"
  run_stage refedit_tokenize "${python_cmd[@]}" "$root/scripts/data/preprocess_refedit.py" tokenize --raw-root "$assets_root/datasets/refedit/raw" --model "$assets_root/models/Lumina-DiMOO" --output "$assets_root/datasets/refedit/tokens/train" --seed 42 --target-size 512
  run_stage mixed_manifest "${python_cmd[@]}" "$root/scripts/data/build_mixed_edit_manifest.py" --magicbrush-manifest "$assets_root/datasets/magicbrush/tokens/train/manifest.jsonl" --refedit-manifest "$assets_root/datasets/refedit/tokens/train/manifest.jsonl" --output "$assets_root/datasets/mixed"
  run_stage gce_clusters "${python_cmd[@]}" "$root/scripts/tools/gce/build_clusters.py" --model "$assets_root/models/Lumina-DiMOO" --output "$assets_root/artifacts/gce_clusters_1024_512.pt" --levels 1024 512 --seed 0
  run_stage metric_models "${python_cmd[@]}" "$root/scripts/setup/download_formal_models.py" --assets-root "$assets_root" --config "$config" --only dino --only clip
  run_stage lpips "${python_cmd[@]}" "$root/scripts/setup/prefetch_lpips.py" --assets-root "$assets_root"
fi
if [[ "$scope" != train ]]; then
  # The official test archive is intentionally manual.  This stage must fail
  # clearly until MAGICBRUSH_TEST_ROOT names the real unpacked archive.
  if [[ "$mode" == run ]]; then : "${MAGICBRUSH_TEST_ROOT:?REAL TEST ARCHIVE NOT VERIFIED: set MAGICBRUSH_TEST_ROOT to the official archive}"; else MAGICBRUSH_TEST_ROOT='${MAGICBRUSH_TEST_ROOT}'; fi
  run_stage magicbrush_test "${python_cmd[@]}" "$root/scripts/data/prepare_magicbrush_test.py" --test-root "$MAGICBRUSH_TEST_ROOT" --output "$assets_root/datasets/magicbrush-test/canonical"
  run_stage test_geometry "${python_cmd[@]}" "$root/scripts/eval/prepare_magicbrush_eval.py" --manifest "$assets_root/datasets/magicbrush-test/canonical/manifest.jsonl" --output "$assets_root/datasets/magicbrush-test/geometry.jsonl" --seed 42 --target-size 512
  run_stage test_tokenize torchrun --nproc_per_node="${TOKENIZE_GPUS:-8}" "$root/scripts/data/preprocess_magicbrush.py" pretokenize --manifest "$assets_root/datasets/magicbrush-test/geometry.jsonl" --model "$assets_root/models/Lumina-DiMOO" --output "$assets_root/datasets/magicbrush-test/tokens"
  run_stage test_subset "${python_cmd[@]}" "$root/scripts/eval/evaluate_gt_mask_editing.py" --manifest "$assets_root/datasets/magicbrush-test/tokens/manifest.jsonl" --subset "$assets_root/datasets/magicbrush-test/eval_subset.jsonl" --prepare-subset-only --limit 0 --seed 42
  run_stage formal_asset_audit "${python_cmd[@]}" "$root/scripts/tools/audit_formal_assets.py" --mode all --output "$assets_root/artifacts/formal_assets.json" --asset-config "$config" --model "$assets_root/models/Lumina-DiMOO" --train-manifest "$assets_root/datasets/mixed/train/manifest.jsonl" --gce-clusters "$assets_root/artifacts/gce_clusters_1024_512.pt" --canonical-test-manifest "$assets_root/datasets/magicbrush-test/canonical/manifest.jsonl" --test-token-manifest "$assets_root/datasets/magicbrush-test/tokens/manifest.jsonl" --test-subset "$assets_root/datasets/magicbrush-test/eval_subset.jsonl" --dino-model "$assets_root/models/dinov2-base" --clip-model "$assets_root/models/clip-vit-large-patch14"
fi
