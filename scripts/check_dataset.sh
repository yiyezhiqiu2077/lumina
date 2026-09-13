#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT is required}"
DATA_CONFIG="${DATA_CONFIG:-$DATA_ROOT/official_tokens/train/manifest.jsonl}"
: "${PYTHON:=python}"

[[ -d "$DATA_ROOT" ]] || { echo "DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 2; }
[[ -f "$DATA_CONFIG" ]] || { echo "DATA_CONFIG does not exist: $DATA_CONFIG" >&2; exit 2; }

"$PYTHON" - "$DATA_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("manifest is empty")
required = {"instruction", "source", "target", "mask_edit", "token_file", "token_height", "token_width"}
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"manifest is missing keys: {sorted(missing)}")
for row in rows[:16]:
    token = Path(row["token_file"])
    if not token.is_file():
        raise SystemExit(f"missing token file: {token}")
    if (row["token_height"], row["token_width"]) != (32, 32):
        raise SystemExit(f"unexpected VQ grid for {row.get('sample_key')}: {(row['token_height'], row['token_width'])}")
print({"manifest": str(manifest), "samples": len(rows), "checked_token_files": min(16, len(rows)), "vq_grid": [32, 32]})
PY
