#!/usr/bin/env python3
"""Write a reproducible identity audit for formal mixed training or MagicBrush TEST evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluation.formal_assets import (
    audit_test_assets,
    audit_training_manifest,
    lpips_alex_identity,
    metric_model_identity,
    model_identity,
    sha256,
)
from dataset.formal_assets import load_formal_assets


def _code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")), "uv_lock_sha256": sha256(root / "uv.lock")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--gce-clusters", type=Path)
    parser.add_argument("--canonical-test-manifest", type=Path)
    parser.add_argument("--test-token-manifest", type=Path)
    parser.add_argument("--test-subset", type=Path)
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--asset-config", type=Path, default=Path("configs/formal_assets.yaml"))
    args = parser.parse_args()

    output = Path(args.output)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    previous["code"] = _code_identity()
    previous["pinned_downloads"] = load_formal_assets(args.asset_config)
    previous["lumina"] = model_identity(args.model)
    if args.mode in {"train", "all"}:
        if args.train_manifest is None or args.gce_clusters is None:
            parser.error("--mode train requires --train-manifest and --gce-clusters")
        previous["training"] = audit_training_manifest(args.train_manifest)
        inspect = Path(__file__).with_name("gce") / "inspect_clusters.py"
        result = subprocess.run(
            [sys.executable, str(inspect), "--model", str(args.model), "--clusters", str(args.gce_clusters), "--levels", "1024", "512"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"GCE cluster inspection failed:\n{result.stdout}\n{result.stderr}")
        previous["gce"] = {
            "cluster_file": {"path": str(args.gce_clusters.resolve()), "sha256": sha256(args.gce_clusters)},
            "levels": [1024, 512],
            "inspection": json.loads(result.stdout),
        }
    if args.mode in {"eval", "all"}:
        required = (args.canonical_test_manifest, args.test_token_manifest, args.test_subset, args.dino_model, args.clip_model)
        if any(value is None for value in required):
            parser.error("--mode eval requires canonical/test-token/subset manifests plus --dino-model and --clip-model")
        previous["evaluation"] = audit_test_assets(
            args.canonical_test_manifest, args.test_token_manifest, args.test_subset
        )
        previous["metrics"] = {
            "lpips": lpips_alex_identity(),
            "dino": metric_model_identity(args.dino_model, model_id="facebook/dinov2-base"),
            "clip": metric_model_identity(args.clip_model, model_id="openai/clip-vit-large-patch14"),
            "roi_padding_ratio": 0.10,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(previous, indent=2))


if __name__ == "__main__":
    main()
