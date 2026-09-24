#!/usr/bin/env python3
"""Download pinned Lumina/DINO/CLIP snapshots into a cold-start asset root."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.formal_assets import load_formal_assets, sha256, write_json


def _missing_required(path: Path, kind: str) -> list[str]:
    required = ["config.json"]
    if kind == "lumina": required += ["model_index.json", "model.safetensors.index.json", "vqvae/config.json"]
    else: required += ["preprocessor_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    weights = list(path.rglob("*.safetensors")) + list(path.rglob("*.bin"))
    if not weights: missing.append("model weights")
    if kind == "lumina" and not list(path.rglob("*tokenizer*")): missing.append("tokenizer")
    if kind == "lumina" and (path / "model.safetensors.index.json").is_file():
        import json
        for shard in set(json.loads((path / "model.safetensors.index.json").read_text()).get("weight_map", {}).values()):
            if not (path / shard).is_file(): missing.append(f"missing shard {shard}")
    return missing


def _verify_snapshot(path: Path, *, repo_id: str, revision: str, kind: str) -> dict:
    manifest = path / "download_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"partial snapshot has no manifest: {path}")
    import json

    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("repo_id") != repo_id or value.get("resolved_revision") != revision:
        raise RuntimeError(f"snapshot identity does not match pinned config: {path}")
    missing = _missing_required(path, kind)
    if missing:
        raise RuntimeError(f"partial snapshot is missing required files: {missing}")
    return value


def download_snapshot(*, repo_id: str, revision: str, destination: Path, kind: str) -> dict:
    """Download once, then write an identity marker only after verification.

    The marker makes a rerun safe.  A half-written directory intentionally
    fails rather than being silently treated as a usable model.
    """
    if (destination / "download_manifest.json").is_file():
        return {**_verify_snapshot(destination, repo_id=repo_id, revision=revision, kind=kind), "skipped_verified": True}
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(destination))
    missing = _missing_required(destination, kind)
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
            repo_id=item["repo_id"], revision=item["revision"], destination=args.assets_root / item["local_dir"], kind=name
        )
    write_json(args.assets_root / "artifacts" / "model_downloads.json", result)


if __name__ == "__main__":
    main()
