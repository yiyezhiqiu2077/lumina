#!/usr/bin/env python3
"""Populate and identify the official AlexNet trunk required by LPIPS alex."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import AlexNet_Weights

from dataset.formal_assets import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    args = parser.parse_args()
    # Asking torchvision for the enum is the official cache/download mechanism.
    AlexNet_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(AlexNet_Weights.IMAGENET1K_V1.url).name
    if not path.is_file():
        raise RuntimeError(f"AlexNet weight is not present after prefetch: {path}")
    write_json(args.assets_root / "artifacts" / "lpips_alex.json", {
        "lpips_version": "0.1.4", "net": "alex", "torchvision_weight": "AlexNet_Weights.IMAGENET1K_V1",
        "path": str(path), "sha256": sha256(path),
    })


if __name__ == "__main__":
    main()
