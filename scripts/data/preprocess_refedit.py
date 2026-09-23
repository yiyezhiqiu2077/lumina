#!/usr/bin/env python3
"""Audit and VQ-tokenize the official RefEdit final-mask dataset.

The script reads image bytes directly from the official parquet shards.  It
never fabricates source/mask associations: only records passing the strict
``(source_relative_path, row_idx)`` plus ``raw.img_id == audit.img_id``,
instruction, and image-size contract are emitted.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor

from dataset.geometry import apply_shared_geometry, sample_shared_geometry
from dataset.preprocess import encode, token_mask
from dataset.refedit import audit_refedit, iter_refedit_records
from dataset.utils import write_jsonl
from dataset.magicbrush import stable_sample_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="write official and strict parser metadata")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    tokenize = sub.add_parser("tokenize", help="create the shared editing-token manifest")
    tokenize.add_argument("--raw-root", type=Path, required=True)
    tokenize.add_argument("--model", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)
    tokenize.add_argument("--seed", type=int, default=42)
    tokenize.add_argument("--target-size", type=int, default=512)
    tokenize.add_argument("--max-samples", type=int, default=0, help="0 means all valid records")
    return parser.parse_args()


def write_audit(raw_root: Path, output: Path, *, limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    meta = audit_refedit(raw_root)
    strict_skipped: Counter[str] = Counter()
    raw_mask_areas = []
    checked = 0
    for record in iter_refedit_records(
        raw_root,
        on_skip=lambda reason: strict_skipped.update([reason]),
    ):
        checked += 1
        raw_mask_areas.append(float((np.asarray(record["mask"], dtype=np.uint8) >= 128).mean()))
        if limit and checked >= limit:
            break
    meta["strict_parser"] = {
        "checked_records": checked,
        "skipped_reasons": dict(sorted(strict_skipped.items())),
        "raw_mask_area_fraction": {
            "mean": sum(raw_mask_areas) / len(raw_mask_areas) if raw_mask_areas else 0.0,
            "min": min(raw_mask_areas) if raw_mask_areas else 0.0,
            "max": max(raw_mask_areas) if raw_mask_areas else 0.0,
        },
    }
    (output / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def tokenize(args: argparse.Namespace) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    meta = audit_refedit(args.raw_root)
    strict_skipped: Counter[str] = Counter()
    device = torch.device("cuda")
    vqvae = VQModel.from_pretrained(
        args.model,
        subfolder="vqvae",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval()
    scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    processor = VaeImageProcessor(vae_scale_factor=scale, do_normalize=False)
    files = args.output / "files"
    files.mkdir(exist_ok=True)
    rows = []
    token_mask_areas = []
    with torch.no_grad():
        for index, record in enumerate(
            iter_refedit_records(
                args.raw_root,
                on_skip=lambda reason: strict_skipped.update([reason]),
            )
        ):
            if args.max_samples and index >= args.max_samples:
                break
            geometry_seed = stable_sample_seed(args.seed, 0, record["sample_key"])
            geometry = sample_shared_geometry(record["source"].size, geometry_seed, args.target_size)
            source = apply_shared_geometry(record["source"], geometry)
            target = apply_shared_geometry(record["target"], geometry)
            mask = apply_shared_geometry(record["mask"], geometry, is_mask=True)
            source_codes, source_shape = encode(vqvae, processor, source)
            target_codes, target_shape = encode(vqvae, processor, target)
            if source_shape != target_shape:
                raise ValueError(f"VQ source/target mismatch for {record['sample_key']}")
            edit_mask = token_mask(mask, source_shape, device)
            if source_shape != (32, 32) or edit_mask.shape != source_codes.shape:
                raise ValueError(f"unexpected VQ contract for {record['sample_key']}: {source_shape}")
            if not bool(edit_mask.any()):
                raise ValueError(f"empty token mask after geometry for {record['sample_key']}")
            token_file = files / f"{index:06d}.pt"
            torch.save(
                {
                    "source_codes": source_codes,
                    "target_codes": target_codes,
                    "edit_mask": edit_mask,
                    "token_height": source_shape[0],
                    "token_width": source_shape[1],
                    "processed_width": source.width,
                    "processed_height": source.height,
                },
                token_file,
            )
            rows.append(
                {
                    "dataset_name": "refedit",
                    "sample_key": record["sample_key"],
                    "instruction": record["instruction"],
                    "token_file": token_file.relative_to(args.output).as_posix(),
                    "token_height": source_shape[0],
                    "token_width": source_shape[1],
                    "geometry": geometry.as_dict(),
                    "geometry_seed": geometry_seed,
                    "row_idx": record["row_idx"],
                    "source_relative_path": record["source_relative_path"],
                }
            )
            token_mask_areas.append(float(edit_mask.float().mean()))
            if (index + 1) % 10 == 0:
                print(json.dumps({"complete": index + 1}), flush=True)
    write_jsonl(args.output / "manifest.jsonl", rows)
    meta["tokenization"] = {
        "usable_row_count": len(rows),
        "strict_skipped_reasons": dict(sorted(strict_skipped.items())),
        "token_grid": [32, 32],
        "token_mask_area_fraction": {
            "mean": sum(token_mask_areas) / len(token_mask_areas) if token_mask_areas else 0.0,
            "min": min(token_mask_areas) if token_mask_areas else 0.0,
            "max": max(token_mask_areas) if token_mask_areas else 0.0,
        },
    }
    (args.output / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "manifest": str(args.output / "manifest.jsonl")}, indent=2))
    return args.output / "manifest.jsonl"


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        print(json.dumps(write_audit(args.raw_root, args.output), indent=2))
    else:
        tokenize(args)


if __name__ == "__main__":
    main()
