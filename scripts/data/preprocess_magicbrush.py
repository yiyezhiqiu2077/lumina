#!/usr/bin/env python3
"""Prepare raw MagicBrush manifests or pretokenize a prepared split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.preprocess import prepare_magicbrush, pretokenize_magicbrush


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create deterministic train/validation manifests")
    prepare.add_argument("--train-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--target-size", type=int, default=512)
    prepare.add_argument("--calibration-samples", type=int, default=200)
    prepare.add_argument("--preview-samples", type=int, default=100)
    pretokenize = commands.add_parser("pretokenize", help="encode a prepared manifest into VQ tokens")
    pretokenize.add_argument("--manifest", type=Path, required=True)
    pretokenize.add_argument("--model", type=Path, required=True)
    pretokenize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare_magicbrush(
                    args.train_manifest,
                    args.output,
                    seed=args.seed,
                    val_fraction=args.val_fraction,
                    target_size=args.target_size,
                    calibration_samples=args.calibration_samples,
                    preview_samples=args.preview_samples,
                ),
                indent=2,
            )
        )
        return
    pretokenize_magicbrush(args.manifest, args.model, args.output)


if __name__ == "__main__":
    main()
