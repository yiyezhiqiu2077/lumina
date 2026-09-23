#!/usr/bin/env python3
"""Build a non-duplicating MagicBrush + RefEdit training manifest."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from dataset.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magicbrush-manifest", type=Path, required=True)
    parser.add_argument("--refedit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory containing train/manifest.jsonl")
    return parser.parse_args()


def _link(destination: Path, target: Path) -> None:
    if destination.is_symlink() and destination.resolve() == target.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing mixed token link: {destination}")
    destination.symlink_to(target.resolve(), target_is_directory=True)


def _payload_reason(path: Path) -> str | None:
    if not path.is_file():
        return "missing_token_file"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return "unreadable_token_payload"
    required = {"source_codes", "target_codes", "edit_mask", "token_height", "token_width"}
    if required.difference(payload):
        return "invalid_token_payload_keys"
    if tuple(payload["source_codes"].shape) != (32, 32) or tuple(payload["target_codes"].shape) != (32, 32):
        return "invalid_token_grid"
    if tuple(payload["edit_mask"].shape) != (32, 32):
        return "invalid_mask_grid"
    if not bool(payload["edit_mask"].any()):
        return "empty_mask"
    return None


def _rows(manifest: Path, dataset_name: str, token_link_name: str) -> tuple[list[dict], Counter[str]]:
    rows, failures = [], Counter()
    manifest = manifest.resolve()
    for row in read_jsonl(manifest):
        token_path = Path(row["token_file"])
        token_path = token_path if token_path.is_absolute() else manifest.parent / token_path
        reason = _payload_reason(token_path)
        if reason is not None:
            failures[reason] += 1
            continue
        key = str(row["sample_key"])
        if dataset_name == "refedit" and not key.startswith("refedit/"):
            key = f"refedit/{key}"
        if dataset_name == "magicbrush" and not key.startswith("magicbrush/"):
            key = f"magicbrush/{key}"
        rows.append(
            {
                "dataset_name": dataset_name,
                "sample_key": key,
                "instruction": row["instruction"],
                "token_file": (Path("files") / token_link_name / token_path.name).as_posix(),
            }
        )
    return rows, failures


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    train = output / "train"
    links = train / "files"
    links.mkdir(parents=True, exist_ok=True)
    magicbrush_manifest = args.magicbrush_manifest.resolve()
    refedit_manifest = args.refedit_manifest.resolve()
    magicbrush_rows, magicbrush_failures = _rows(magicbrush_manifest, "magicbrush", "magicbrush")
    refedit_rows, refedit_failures = _rows(refedit_manifest, "refedit", "refedit")
    all_rows = magicbrush_rows + refedit_rows
    keys = [row["sample_key"] for row in all_rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise ValueError(f"mixed manifest has {duplicates} duplicate sample_key values")
    _link(links / "magicbrush", magicbrush_manifest.parent / "files")
    _link(links / "refedit", refedit_manifest.parent / "files")
    manifest = train / "manifest.jsonl"
    write_jsonl(manifest, all_rows)
    audit = {
        "magicbrush_count": len(magicbrush_rows),
        "refedit_count": len(refedit_rows),
        "total_count": len(all_rows),
        "duplicate_sample_key_count": duplicates,
        "invalid": {
            "magicbrush": dict(sorted(magicbrush_failures.items())),
            "refedit": dict(sorted(refedit_failures.items())),
        },
        "token_grid": [32, 32],
        "manifest": str(manifest),
    }
    (output / "dataset_meta.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
