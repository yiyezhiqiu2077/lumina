#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets.magicbrush_dataset import enrich_geometry, read_jsonl, resolve_record_paths, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=512)
    args = parser.parse_args()
    rows = [resolve_record_paths(row, args.manifest) for row in read_jsonl(args.manifest)]
    rows = enrich_geometry(rows, args.seed, args.target_size)
    write_jsonl(args.output, rows)
    print({"samples": len(rows), "output": str(args.output), "seed": args.seed})


if __name__ == "__main__":
    main()
