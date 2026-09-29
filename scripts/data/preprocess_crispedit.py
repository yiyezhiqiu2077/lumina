#!/usr/bin/env python3
"""Strictly audit and distributed-tokenize the pinned CrispEdit snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.crispedit import audit_crispedit, crispedit_shards, iter_crispedit_records
from dataset.labeled_edit import pretokenize_sharded_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "tokenize"):
        command = sub.add_parser(name)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    token = sub.choices["tokenize"]
    token.add_argument("--model", type=Path, required=True)
    token.add_argument("--seed", type=int, default=42)
    token.add_argument("--target-size", type=int, default=512)
    token.add_argument("--max-samples", type=int, default=0, help="0 means all strict-pass records; smoke only on one rank")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = audit_crispedit(args.raw_root)
        (args.output / "dataset_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return
    manifest = pretokenize_sharded_records(
        raw_root=args.raw_root, model=args.model, output=args.output, seed=args.seed,
        target_size=args.target_size, dataset_name="crispedit", shards=crispedit_shards(args.raw_root),
        iter_records=iter_crispedit_records,
        max_samples=args.max_samples,
    )
    print(json.dumps({"manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
