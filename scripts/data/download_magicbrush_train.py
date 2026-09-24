#!/usr/bin/env python3
"""Fetch the pinned MagicBrush train split and write its canonical raw manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from dataset.formal_assets import load_formal_assets, write_json
from dataset.geometry import enrich_geometry
from dataset.utils import write_jsonl

REQUIRED_FIELDS = ("img_id", "turn_index", "instruction", "source_img", "target_img", "mask_img")


def _save_image(value: Any, path: Path, *, mode: str) -> None:
    image = value if isinstance(value, Image.Image) else value.convert("RGB")
    image.convert(mode).save(path)


def canonicalize_rows(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    canonical = []
    keys = set()
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row or row[field] is None]
        if missing:
            raise ValueError(f"MagicBrush row {index} missing required fields: {missing}")
        img_id, turn_index = str(row["img_id"]), int(row["turn_index"])
        key = f"magicbrush/{img_id}_turn{turn_index}"
        if key in keys:
            raise ValueError(f"duplicate MagicBrush sample key: {key}")
        keys.add(key)
        stem = f"{index:05d}_{img_id}_turn{turn_index}"
        source, target, mask = images / f"{stem}_source.png", images / f"{stem}_target.png", images / f"{stem}_mask.png"
        _save_image(row["source_img"], source, mode="RGB")
        _save_image(row["target_img"], target, mode="RGB")
        _save_image(row["mask_img"], mask, mode="L")
        canonical.append({
            "index": index, "sample_key": key, "img_id": img_id, "turn_index": turn_index,
            "session_id": img_id, "instruction": str(row["instruction"]).strip(),
            "source": source.relative_to(output).as_posix(), "target": target.relative_to(output).as_posix(),
            "mask_edit": mask.relative_to(output).as_posix(),
        })
    if any(not row["instruction"] for row in canonical):
        raise ValueError("MagicBrush contains an empty instruction")
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/formal_assets.yaml"))
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    asset = load_formal_assets(args.config)["magicbrush"]
    root = args.assets_root / "datasets" / "magicbrush"
    raw, prepared = root / "raw", root / "prepared"
    if args.geometry_only:
        rows = [json.loads(line) for line in (raw / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
        prepared.mkdir(parents=True, exist_ok=True)
        write_jsonl(prepared / "train.jsonl", enrich_geometry([{**row, **{key: str(raw / row[key]) for key in ("source", "target", "mask_edit")}} for row in rows], 42, 512))
        return
    from datasets import load_dataset
    dataset = load_dataset(asset["repo_id"], split=asset["split"], revision=asset["revision"])
    rows = canonicalize_rows([dict(item) for item in dataset], raw)
    if len(rows) != asset["expected_samples"]:
        raise ValueError(f"MagicBrush count must be {asset['expected_samples']}, got {len(rows)}")
    write_jsonl(raw / "train.jsonl", rows)
    write_json(raw / "dataset_meta.json", {"repo_id": asset["repo_id"], "resolved_revision": asset["revision"], "split": asset["split"], "sample_count": len(rows), "fields": list(REQUIRED_FIELDS)})


if __name__ == "__main__":
    main()
