#!/usr/bin/env python3
"""Download the immutable RefEdit source snapshot without copying local assets."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.formal_assets import load_formal_assets, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/formal_assets.yaml"))
    args = parser.parse_args()
    entry = load_formal_assets(args.config)["refedit"]
    destination = args.assets_root / entry["local_dir"]
    marker = destination / "download_manifest.json"
    if marker.is_file():
        import json
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("repo_id") == entry["repo_id"] and existing.get("resolved_revision") == entry["revision"]:
            if any(destination.iterdir()): return
        raise RuntimeError(f"RefEdit destination identity mismatch: {destination}")
    from huggingface_hub import snapshot_download
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=entry["repo_id"], repo_type="dataset", revision=entry["revision"], local_dir=str(destination))
    files = {item.relative_to(destination).as_posix(): sha256(item) for item in sorted(destination.rglob("*")) if item.is_file()}
    if not files:
        raise RuntimeError("RefEdit snapshot has no files")
    write_json(marker, {"repo_id": entry["repo_id"], "requested_revision": entry["revision"], "resolved_revision": entry["revision"], "files": files})


if __name__ == "__main__":
    main()
