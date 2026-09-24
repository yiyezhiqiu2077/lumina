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
import os
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
    parser.add_argument("--formal-assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.formal_assets or Path(os.environ.get("FORMAL_ASSET_AUDIT_PATH", args.train_root / "formal_assets.json"))
    if not audit_path.is_file(): raise RuntimeError(f"formal asset authority is missing: {audit_path}")
    for root, log_name in ((args.train_root, "2x3_formal_pipeline.log"), (args.eval_root, "2x3_eval_pipeline.log")):
        log = root / log_name
        if not log.is_file() or sum("SUCCEEDED" in line for line in log.read_text().splitlines()) != 6:
            raise RuntimeError(f"package requires six successful runs in {log}")
    comparison_json, comparison_csv = args.eval_root / "comparison/comparison.json", args.eval_root / "comparison/comparison.csv"
    if not comparison_json.is_file() or not comparison_csv.is_file(): raise RuntimeError("package requires comparison.json and comparison.csv")
    eval_rows = list(args.eval_root.glob("*/per_sample.jsonl"))
    if len(eval_rows) != 6 or any(sum(1 for line in path.read_text().splitlines() if line) != 1053 for path in eval_rows):
        raise RuntimeError("package requires six 1053-row evaluation outputs")
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
            for item in (origin / "2x3_formal_pipeline.log", origin / "2x3_eval_pipeline.log"):
                if item.is_file() and _safe(item):
                    target = stage / label / item.name; target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(item.read_bytes()); copied.append(target)
        for root, label in ((args.train_root, "train"), (args.eval_root, "eval")):
            audit = root / "formal_assets.json"
            if audit.is_file() and _safe(audit):
                target = stage / label / audit.name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(audit.read_bytes()); copied.append(target)
        target = stage / "formal_assets.json"; target.write_bytes(audit_path.read_bytes()); copied.append(target)
        for item in (comparison_json, comparison_csv):
            target = stage / item.name; target.write_bytes(item.read_bytes()); copied.append(target)
        repo = Path(__file__).resolve().parents[2]
        packaging_code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        provenance_shas=[]
        for path in args.train_root.glob("*/run_provenance.json"):
            value=json.loads(path.read_text()); provenance_shas.append(value.get("git_sha") or value.get("git_commit"))
        experiment_code_sha=next((value for value in provenance_shas if value), json.loads(audit_path.read_text()).get("code",{}).get("git_sha"))
        if not experiment_code_sha or len({value for value in provenance_shas if value}) > 1: raise RuntimeError("train provenance git SHA is missing or inconsistent")
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        (stage / "FORMAL_EXPERIMENT_CODE_SHA.txt").write_text(experiment_code_sha + "\n", encoding="utf-8")
        (stage / "git_status.txt").write_text(status, encoding="utf-8")
        configs = stage / "configs_used"; configs.mkdir()
        for config in sorted((repo / "configs/train").rglob("mixed_*8g_b4_a1.yaml")):
            target = configs / config.name; target.write_bytes(config.read_bytes())
        copied.extend([stage / "FORMAL_EXPERIMENT_CODE_SHA.txt", stage / "git_status.txt", *configs.iterdir()])
        files = {path.relative_to(stage).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size} for path in copied}
        manifest = {"files": files, "experiment_code_sha": experiment_code_sha, "packaging_code_sha": packaging_code_sha, "run_names": sorted({path.parent.name for path in copied if path.parent.name}), "test_identity": json.loads(audit_path.read_text())}
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
