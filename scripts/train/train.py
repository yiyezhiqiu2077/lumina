#!/usr/bin/env python3
"""The single public, config-driven MagicBrush training entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.config import load_train_config
from training.distributed import distributed_worker_environment_present, launch_training, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    worker_environment = distributed_worker_environment_present()
    if cli.distributed_worker and not worker_environment:
        raise SystemExit(
            "train.py worker mode requires RANK, WORLD_SIZE, and LOCAL_RANK. "
            "Run scripts/train/train.py without --distributed-worker instead."
        )
    train_args = load_train_config(cli.config, cli.resume_from_checkpoint)
    if cli.distributed_worker or worker_environment:
        run(train_args)
        return
    launch_training(train_args, Path(__file__).resolve(), cli.config.resolve(), cli.resume_from_checkpoint)


if __name__ == "__main__":
    main()
