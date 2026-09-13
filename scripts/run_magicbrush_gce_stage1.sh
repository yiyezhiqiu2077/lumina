#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
mkdir -p "$OUTPUT_ROOT/ce" "$OUTPUT_ROOT/gce"
OUTPUT_DIR="$OUTPUT_ROOT/ce" "$ROOT/scripts/train_magicbrush_ce.sh" 2>&1 | tee "$OUTPUT_ROOT/ce/train.log"
OUTPUT_DIR="$OUTPUT_ROOT/gce" "$ROOT/scripts/train_magicbrush_gce.sh" 2>&1 | tee "$OUTPUT_ROOT/gce/train.log"
