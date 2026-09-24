#!/usr/bin/env bash
# Create portable, git-ignored local_assets symlinks without touching real assets.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_local_assets.sh model <existing-model-dir>
  bash scripts/setup_local_assets.sh magicbrush <existing-magicbrush-dir>
  bash scripts/setup_local_assets.sh magicbrush-test <existing-magicbrush-test-dir>
  bash scripts/setup_local_assets.sh refedit <existing-refedit-dir>
  bash scripts/setup_local_assets.sh mixed <existing-mixed-token-dir>
  bash scripts/setup_local_assets.sh dino <existing-dinov2-base-dir>
  bash scripts/setup_local_assets.sh clip <existing-clip-vit-large-patch14-dir>
  bash scripts/setup_local_assets.sh experiments <existing-experiment-dir>

The target must already exist.  This script only replaces the corresponding
link inside local_assets/; it never deletes or modifies the real target.
EOF
}

[[ "$#" == 2 ]] || { usage >&2; exit 2; }
kind="$1"
target="$2"
case "$kind" in
    model) link='local_assets/models/Lumina-DiMOO' ;;
    magicbrush) link='local_assets/datasets/magicbrush' ;;
    magicbrush-test) link='local_assets/datasets/magicbrush-test' ;;
    refedit) link='local_assets/datasets/refedit' ;;
    mixed) link='local_assets/datasets/mixed' ;;
    dino) link='local_assets/models/dinov2-base' ;;
    clip) link='local_assets/models/clip-vit-large-patch14' ;;
    experiments) link='local_assets/experiments' ;;
    *) usage >&2; exit 2 ;;
esac

[[ -e "$target" ]] || { printf 'error: target does not exist: %s\n' "$target" >&2; exit 1; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$(cd "$target" && pwd)"
link="$root/$link"
mkdir -p "$(dirname "$link")"
if [[ -e "$link" && ! -L "$link" ]]; then
    printf 'error: refusing to replace non-symlink: %s\n' "$link" >&2
    exit 1
fi
ln -sfn "$target" "$link"
printf '%s -> %s\n' "$link" "$(readlink "$link")"
