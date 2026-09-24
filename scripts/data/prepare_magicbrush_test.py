#!/usr/bin/env python3
"""Convert an unpacked official MagicBrush TEST archive into a canonical raw manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.magicbrush_test import prepare_magicbrush_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="verified official annotation file when auto-discovery is ambiguous")
    args = parser.parse_args()
    print(json.dumps(prepare_magicbrush_test(args.test_root, args.output, annotations=args.annotations), indent=2))


if __name__ == "__main__":
    main()
