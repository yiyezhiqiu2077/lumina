#!/usr/bin/env python3
"""Build a non-duplicating N-way editing training manifest."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from dataset.formal_assets import sha256
from dataset.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], metavar="DATASET=MANIFEST",
                        help="repeatable input; e.g. --input magicbrush=/path/manifest.jsonl")
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


def _rows(manifest: Path, dataset_name: str) -> tuple[list[dict], Counter[str]]:
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
        if not key.startswith(f"{dataset_name}/"):
            key = f"{dataset_name}/{key}"
        rows.append(
            {
                "dataset_name": dataset_name,
                "sample_key": key,
                "instruction": row["instruction"],
                "token_file": (Path("files") / dataset_name / token_path.name).as_posix(),
            }
        )
    return rows, failures


def main() -> None:
    args = parse_args()
    components: list[tuple[str, Path]] = []
    seen = set()
    for value in args.input:
        if "=" not in value:
            raise ValueError(f"--input must be DATASET=MANIFEST, got {value!r}")
        dataset_name, path = value.split("=", 1)
        if not dataset_name or not path or dataset_name in seen:
            raise ValueError(f"invalid or duplicate dataset input: {value!r}")
        seen.add(dataset_name)
        components.append((dataset_name, Path(path).resolve()))
    if not components:
        raise ValueError("at least one --input DATASET=MANIFEST is required")
    output = args.output.resolve()
    train = output / "train"
    links = train / "files"
    links.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    failures: dict[str, dict[str, int]] = {}
    component_manifests: dict[str, dict] = {}
    for dataset_name, manifest_path in components:
        rows, rejected = _rows(manifest_path, dataset_name)
        all_rows.extend(rows)
        failures[dataset_name] = dict(sorted(rejected.items()))
        component_manifests[dataset_name] = {
            "path": str(manifest_path), "sha256": sha256(manifest_path),
            "usable_row_count": len(rows),
            "tokenization_metadata": str(manifest_path.parent / "dataset_meta.json"),
            "tokenization_metadata_sha256": sha256(manifest_path.parent / "dataset_meta.json")
            if (manifest_path.parent / "dataset_meta.json").is_file() else None,
        }
        _link(links / dataset_name, manifest_path.parent / "files")
    keys = [row["sample_key"] for row in all_rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise ValueError(f"mixed manifest has {duplicates} duplicate sample_key values")
    manifest = train / "manifest.jsonl"
    write_jsonl(manifest, all_rows)
    audit = {
        "dataset_counts": {name: info["usable_row_count"] for name, info in component_manifests.items()},
        "component_manifests": component_manifests,
        "total_count": len(all_rows),
        "duplicate_sample_key_count": duplicates,
        "invalid_token_payload_counts": failures,
        "token_grid": [32, 32],
        "manifest": str(manifest),
    }
    (output / "dataset_meta.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
