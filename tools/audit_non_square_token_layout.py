#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import SPECIAL_TOKENS
from datasets.magicbrush_tokens import _spatial_with_newlines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_shape(height: int, width: int) -> dict:
    codes = torch.arange(height * width, dtype=torch.int32).reshape(height, width)
    sequence, spatial_flags = _spatial_with_newlines(codes)
    recovered = torch.tensor(
        [token - SPECIAL_TOKENS["image_token_offset"] for token, spatial in zip(sequence, spatial_flags) if spatial]
    ).reshape(height, width)
    newline_positions = [index for index, spatial in enumerate(spatial_flags) if not spatial]
    expected_newlines = [(row + 1) * (width + 1) - 1 for row in range(height)]
    if not torch.equal(codes.long(), recovered.long()):
        raise AssertionError(f"row-major recovery failed for {(height, width)}")
    if newline_positions != expected_newlines:
        raise AssertionError(f"row newline positions failed for {(height, width)}")
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[1, 2] = True
    mask[-2, -3] = True
    if torch.nonzero(mask.flatten()).flatten().tolist() != [width + 2, (height - 2) * width + width - 3]:
        raise AssertionError(f"mask flatten orientation failed for {(height, width)}")
    return {
        "token_grid_hw": [height, width],
        "source_spatial_count": sum(spatial_flags),
        "reshape_count": height * width,
        "newline_count": len(newline_positions),
        "row_major_round_trip": True,
        "mask_orientation_round_trip": True,
    }


def main():
    args = parse_args()
    shapes = {
        "portrait": [(16, 8), (24, 12), (32, 16), (28, 14), (20, 10)],
        "landscape": [(8, 16), (12, 24), (16, 32), (14, 28), (10, 20)],
        "square": [(8, 8), (12, 12), (16, 16), (24, 24), (32, 32)],
    }
    result = {
        "scope": "synthetic layout fixtures only; MagicBrush 512-level samples are all square",
        "orientations": {kind: [audit_shape(*shape) for shape in values] for kind, values in shapes.items()},
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
