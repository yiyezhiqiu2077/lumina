"""Small, dependency-light contracts shared by cold-start asset scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_assets(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not isinstance(values.get("assets"), dict):
        raise ValueError(f"invalid formal asset config: {path}")
    for name, entry in values["assets"].items():
        if name == "lpips":
            continue
        revision = entry.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"{name} must use a full 40-character immutable revision")
    return values["assets"]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
