#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.magicbrush_dataset import SharedGeometry, apply_shared_geometry, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vq-stride", type=int, default=16)
    return parser.parse_args()


def orientation(width: int, height: int) -> str:
    ratio = width / height
    if ratio > 1.05:
        return "landscape"
    if ratio < 1 / 1.05:
        return "portrait"
    return "square"


def summarize(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": float(np.percentile(ordered, 95)),
        "max": ordered[-1],
    }


def main():
    args = parse_args()
    rows = read_jsonl(args.manifest)
    counts = Counter()
    mask_pixels = []
    token_spatial = []
    examples = {"portrait": [], "landscape": [], "square": []}
    failures = []
    for row in rows:
        geometry = SharedGeometry(**row["geometry"])
        source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
        target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
        mask = apply_shared_geometry(Image.open(row["mask_edit"]).convert("L"), geometry, is_mask=True)
        if source.size != target.size or source.size != mask.size:
            failures.append({"sample_key": row["sample_key"], "reason": "processed size mismatch"})
            continue
        width, height = source.size
        if width % args.vq_stride or height % args.vq_stride:
            failures.append({"sample_key": row["sample_key"], "reason": "not divisible by VQ stride"})
            continue
        kind = orientation(width, height)
        counts[kind] += 1
        token_height, token_width = height // args.vq_stride, width // args.vq_stride
        spatial = token_height * token_width
        token_spatial.append(spatial)
        mask_pixels.append(int((np.asarray(mask) >= 128).sum()))
        if len(examples[kind]) < 5:
            examples[kind].append(
                {
                    "sample_key": row["sample_key"],
                    "processed_pil_size_wh": [width, height],
                    "token_grid_hw": [token_height, token_width],
                    "source_spatial_count": spatial,
                    "reshape_count": token_height * token_width,
                }
            )
    for kind in examples:
        if len(examples[kind]) < 5:
            failures.append({"orientation": kind, "reason": "fewer than five audit examples"})
    result = {
        "manifest": str(args.manifest.resolve()),
        "samples": len(rows),
        "orientation_counts": dict(counts),
        "five_per_orientation": examples,
        "token_spatial_count": summarize(token_spatial),
        "binary_mask_positive_pixels": summarize(mask_pixels),
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
