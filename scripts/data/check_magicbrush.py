#!/usr/bin/env python3
"""Run MagicBrush geometry, sequence, and synthetic-layout contract audits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


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


def run(script: str, *args: object) -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / script), *map(str, args)], check=True)


def main() -> None:
    args = parse_args()
    if args.command == "layout":
        run("audit_non_square_token_layout.py", "--output", args.output)
        return
    if args.command in ("geometry", "all"):
        output = args.output if args.command == "geometry" else args.output / "geometry.json"
        run("audit_magicbrush_geometry.py", "--manifest", args.manifest, "--output", output, "--vq-stride", args.vq_stride)
    if args.command in ("sequence", "all"):
        if args.model is None:
            raise SystemExit("--model is required for sequence and all audits")
        output = args.output if args.command == "sequence" else args.output / "sequence.json"
        run("audit_magicbrush_sequences.py", "--model", args.model, "--manifest", args.manifest, "--output", output, "--seed", args.seed)
    if args.command == "all":
        run("audit_non_square_token_layout.py", "--output", args.output / "non_square_layout.json")


if __name__ == "__main__":
    main()
