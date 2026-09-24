#!/usr/bin/env python3
"""Download pinned Lumina/DINO/CLIP snapshots into a cold-start asset root."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.formal_assets import load_formal_assets, sha256, write_json


def _required_snapshot_files(path: Path) -> list[Path]:
    return [path / "config.json", path / "model_index.json"]


def _verify_snapshot(path: Path, *, repo_id: str, revision: str) -> dict:
    manifest = path / "download_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"partial snapshot has no manifest: {path}")
    import json

    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("repo_id") != repo_id or value.get("resolved_revision") != revision:
        raise RuntimeError(f"snapshot identity does not match pinned config: {path}")
    missing = [str(item) for item in _required_snapshot_files(path) if not item.is_file()]
    if missing:
        raise RuntimeError(f"partial snapshot is missing required files: {missing}")
    return value


def download_snapshot(*, repo_id: str, revision: str, destination: Path) -> dict:
    """Download once, then write an identity marker only after verification.

    The marker makes a rerun safe.  A half-written directory intentionally
    fails rather than being silently treated as a usable model.
    """
    if destination.exists():
        return {**_verify_snapshot(destination, repo_id=repo_id, revision=revision), "skipped_verified": True}
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(destination))
    missing = [str(item) for item in _required_snapshot_files(destination) if not item.is_file()]
    # Lumina keeps the Diffusers index at root; Transformers metric snapshots
    # have config.json only.  Do not impose Lumina's index on metric models.
    if repo_id != "Alpha-VLLM/Lumina-DiMOO":
        missing = [str(destination / "config.json")] if not (destination / "config.json").is_file() else []
    if missing:
        raise RuntimeError(f"downloaded snapshot is incomplete: {missing}")
    files = {item.relative_to(destination).as_posix(): sha256(item) for item in sorted(destination.rglob("*")) if item.is_file()}
    record = {"repo_id": repo_id, "requested_revision": revision, "resolved_revision": revision, "files": files}
    write_json(destination / "download_manifest.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/formal_assets.yaml"))
    parser.add_argument("--only", choices=("lumina", "dino", "clip"), action="append")
    args = parser.parse_args()
    assets = load_formal_assets(args.config)
    result = {}
    for name in ("lumina", "dino", "clip"):
        if args.only and name not in args.only:
            continue
        item = assets[name]
        result[name] = download_snapshot(
            repo_id=item["repo_id"], revision=item["revision"], destination=args.assets_root / item["local_dir"]
        )
    write_json(args.assets_root / "artifacts" / "model_downloads.json", result)


if __name__ == "__main__":
    main()
