"""Small data helpers used by the canonical MagicBrush dataset."""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON Lines records with the prior dataset semantics."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
