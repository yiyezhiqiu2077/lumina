#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets.magicbrush_dataset import (
    SharedGeometry,
    apply_shared_geometry,
    enrich_geometry,
    read_jsonl,
    resolve_record_paths,
    split_by_session,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--preview-samples", type=int, default=100)
    return parser.parse_args()


def preview(row: dict, output: Path):
    geometry = SharedGeometry(**row["geometry"])
    source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
    target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
    mask = apply_shared_geometry(Image.open(row["mask_edit"]).convert("L"), geometry, is_mask=True)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    overlay_source = source.copy()
    overlay_target = target.copy()
    red = Image.new("RGB", source.size, (255, 0, 0))
    alpha = mask.point(lambda value: 110 if value else 0)
    overlay_source.paste(red, mask=alpha)
    overlay_target.paste(red, mask=alpha)
    panels = [source, target, mask.convert("RGB"), overlay_source, overlay_target]
    header = 42
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), header + max(panel.height for panel in panels)), "white")
    ImageDraw.Draw(canvas).text((8, 8), f"{row['sample_key']} | {row['instruction']}", fill="black")
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, header))
        left += panel.width
    canvas.save(output)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [resolve_record_paths(row, args.train_manifest) for row in read_jsonl(args.train_manifest)]
    rows = enrich_geometry(rows, args.seed, args.target_size)
    train, val = split_by_session(rows, args.val_fraction, args.seed)
    calibration = random.Random(args.seed).sample(train, args.calibration_samples)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "val.jsonl", val)
    write_jsonl(args.output / "calibration200_seed42.jsonl", calibration)
    preview_dir = args.output / "alignment_preview100"
    preview_dir.mkdir(exist_ok=True)
    for row in random.Random(args.seed).sample(train, min(args.preview_samples, len(train))):
        preview(row, preview_dir / f"{int(row['index']):05d}.jpg")
    summary = {
        "seed": args.seed,
        "target_size": args.target_size,
        "all_samples": len(rows),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_sessions": len({row["session_id"] for row in train}),
        "val_sessions": len({row["session_id"] for row in val}),
        "calibration_samples": len(calibration),
        "preview_samples": len(list(preview_dir.glob("*.jpg"))),
    }
    (args.output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
