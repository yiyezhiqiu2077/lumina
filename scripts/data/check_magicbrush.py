#!/usr/bin/env python3
"""Run MagicBrush geometry, sequence, and synthetic-layout contract audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from dataset.audit import run_geometry_audit, run_non_square_layout_audit, run_sequence_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("geometry", "sequence", "all"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", type=Path)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--vq-stride", type=int, default=16)
    layout = commands.add_parser("layout")
    layout.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.command == "layout":
        write(args.output, run_non_square_layout_audit())
        return
    if args.command in ("geometry", "all"):
        output = args.output if args.command == "geometry" else args.output / "geometry.json"
        write(output, run_geometry_audit(args.manifest, vq_stride=args.vq_stride))
    if args.command in ("sequence", "all"):
        if args.model is None:
            raise SystemExit("--model is required for sequence and all audits")
        output = args.output if args.command == "sequence" else args.output / "sequence.json"
        write(output, run_sequence_audit(args.model, args.manifest, seed=args.seed))
    if args.command == "all":
        write(args.output / "non_square_layout.json", run_non_square_layout_audit())


if __name__ == "__main__":
    main()
