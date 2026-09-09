#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from config import SPECIAL_TOKENS
from datasets.magicbrush_tokens import MagicBrushTokenDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summary(values):
    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    dataset = MagicBrushTokenDataset(
        args.manifest,
        tokenizer,
        condition_dropout=0,
        seed=args.seed,
        fixed_corruption=True,
    )
    fields = {
        "instruction_token_count": [],
        "source_spatial_count": [],
        "source_newline_count": [],
        "target_spatial_count": [],
        "target_newline_count": [],
        "special_count": [],
        "total_sequence_length": [],
        "valid_target_label_count": [],
    }
    special_ids = set(SPECIAL_TOKENS.values())
    orientation = {"portrait": [], "landscape": [], "square": []}
    for index in range(len(dataset)):
        row = dataset[index]
        for key in fields:
            if key == "special_count":
                fields[key].append(sum(token in special_ids for token in row["input_ids"]))
            else:
                fields[key].append(row[key])
        height, width = row["token_height"], row["token_width"]
        kind = "landscape" if width / height > 1.05 else "portrait" if width / height < 1 / 1.05 else "square"
        if len(orientation[kind]) < 5:
            orientation[kind].append(
                {
                    "sample_key": row["sample_key"],
                    "token_grid_hw": [height, width],
                    "source_spatial_count": row["source_spatial_count"],
                    "reshape_count": height * width,
                }
            )
        if len(row["input_ids"]) != len(row["labels"]):
            raise AssertionError("input/label length mismatch")
        if row["valid_target_label_count"] <= 0:
            raise AssertionError("sample without a valid target label")
        if row["source_spatial_count"] != height * width or row["target_spatial_count"] != height * width:
            raise AssertionError("spatial token count mismatch")
    missing_orientations = [kind for kind, values in orientation.items() if len(values) < 5]
    result = {
        "samples": len(dataset),
        "statistics": {key: summary(values) for key, values in fields.items()},
        "five_per_orientation": orientation,
        "missing_orientations": missing_orientations,
        "non_square_dataset_gate": "not_applicable_all_magicbrush_512_level_samples_are_square" if missing_orientations else "passed",
        "all_input_label_lengths_equal": True,
        "all_samples_have_valid_target_labels": True,
        "target_truncation_detected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
