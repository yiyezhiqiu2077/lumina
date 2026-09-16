#!/usr/bin/env python3
"""Thin config-driven entrypoint for all MagicBrush objectives."""
from __future__ import annotations

import argparse
from pathlib import Path

from lumina_dimoo.training.config import load_train_config
from lumina_dimoo.training.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    run(load_train_config(args.config, args.resume_from_checkpoint))


if __name__ == "__main__":
    main()
