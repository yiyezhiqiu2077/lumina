#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_DIR="$OUTPUT_ROOT/B0-CE" "$ROOT/scripts/train_magicbrush_ce.sh" 2>&1 | tee "$OUTPUT_ROOT/B0-CE.train.log"
OUTPUT_DIR="$OUTPUT_ROOT/G1-GCE" "$ROOT/scripts/train_magicbrush_gce.sh" 2>&1 | tee "$OUTPUT_ROOT/G1-GCE.train.log"
