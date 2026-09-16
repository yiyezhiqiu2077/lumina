#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec uv run python "$ROOT/scripts/train/train.py" --config "$ROOT/configs/train/magicbrush_gce.yaml" "$@"
