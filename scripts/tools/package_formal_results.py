#!/usr/bin/env python3
"""Create a small, whitelisted archive of formal logs and scalar results."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
RUN_FILES = ("experiment_config.json", "run_provenance.json", "lora_report.json", "quality_status.json", "eval_args.json", "summary.json")
EXCLUDED_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(source: Path) -> bool:
    parts = set(source.parts)
    return not (source.suffix in EXCLUDED_SUFFIXES or "checkpoints" in parts or "checkpoint" in source.name or "images" in parts or "cache" in parts)


def _collect(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [path for name in names for path in sorted(root.rglob(name)) if path.is_file() and _safe(path)]


def _gzip_jsonl(source: Path, destination: Path) -> Path:
    target = destination / f"{source.name}.gz"
    with source.open("rb") as reader, gzip.open(target, "wb") as writer:
        writer.writelines(reader)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lumina_formal_package_") as temporary:
        stage = Path(temporary)
        copied: list[Path] = []
        for origin, label in ((args.train_root, "train"), (args.eval_root, "eval")):
            if not origin.exists():
                continue
            for item in _collect(origin, RUN_FILES):
                target = stage / label / item.relative_to(origin)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes()); copied.append(target)
            for item in sorted(origin.rglob("*.jsonl")):
                if _safe(item):
                    target_dir = stage / label / item.relative_to(origin).parent; target_dir.mkdir(parents=True, exist_ok=True)
                    copied.append(_gzip_jsonl(item, target_dir))
            for item in (origin / "2x3_formal_pipeline.log", origin / "evaluation_pipeline.log"):
                if item.is_file() and _safe(item):
                    target = stage / label / item.name; target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(item.read_bytes()); copied.append(target)
        for root, label in ((args.train_root, "train"), (args.eval_root, "eval")):
            audit = root / "formal_assets.json"
            if audit.is_file() and _safe(audit):
                target = stage / label / audit.name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(audit.read_bytes()); copied.append(target)
        repo = Path(__file__).resolve().parents[2]
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        (stage / "FORMAL_EXPERIMENT_CODE_SHA.txt").write_text(code_sha + "\n", encoding="utf-8")
        (stage / "git_status.txt").write_text(status, encoding="utf-8")
        configs = stage / "configs_used"; configs.mkdir()
        for config in sorted((repo / "configs/train").rglob("mixed_*8g_b4_a1.yaml")):
            target = configs / config.name; target.write_bytes(config.read_bytes())
        copied.extend([stage / "FORMAL_EXPERIMENT_CODE_SHA.txt", stage / "git_status.txt", *configs.iterdir()])
        files = {path.relative_to(stage).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size} for path in copied}
        manifest = {"files": files, "git_sha": code_sha, "run_names": sorted({path.parent.name for path in copied if path.parent.name}), "test_identity": next((json.loads(path.read_text(encoding="utf-8")) for path in copied if path.name == "formal_assets.json"), None)}
        (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            for item in sorted(stage.rglob("*")):
                if item.is_file(): archive.add(item, arcname=item.relative_to(stage))
    if args.output.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"result package exceeds 50 MiB: {args.output.stat().st_size} bytes")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
