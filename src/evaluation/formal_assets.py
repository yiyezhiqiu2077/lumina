"""Portable formal train/evaluation asset audits."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset.utils import read_jsonl


FORMAL_TRAIN_COUNTS = {"magicbrush": 8807, "refedit": 7804}
FORMAL_TEST_SESSIONS = 535
FORMAL_TEST_TURNS = 1053


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required formal asset is missing: {path}")
        return None
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _weights_identity(directory: Path) -> dict[str, Any]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(payload.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"model index has no weight_map shards: {index}")
        return {"index": _file(index, required=True), "shards": [_file(directory / shard, required=True) for shard in shards]}
    for name in (
        "model.safetensors", "pytorch_model.bin", "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.bin",
    ):
        identity = _file(directory / name)
        if identity is not None:
            return {"single_weight": identity}
    raise FileNotFoundError(f"no recognized model weights under {directory}")


def model_identity(model: Path) -> dict[str, Any]:
    model = Path(model).resolve()
    tokenizer_files = {
        name: value
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model")
        if (value := _file(model / name)) is not None
    }
    vqvae = model / "vqvae"
    vq_weights = _weights_identity(vqvae)
    return {
        "model_root": str(model),
        "config": _file(model / "config.json", required=True),
        "weights": _weights_identity(model),
        "tokenizer": tokenizer_files,
        "vqvae": {"config": _file(vqvae / "config.json", required=True), "weights": vq_weights},
    }


def audit_training_manifest(
    manifest: Path, *, expected_counts: dict[str, int] | None = None, require_token_files: bool = True,
) -> dict[str, Any]:
    """Require the frozen 8807/7804 composition and portable token-file integrity."""
    manifest = Path(manifest).resolve()
    rows = read_jsonl(manifest)
    expected = expected_counts or FORMAL_TRAIN_COUNTS
    counts = Counter(str(row.get("dataset_name", "")) for row in rows)
    keys = [str(row.get("sample_key", "")) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    missing_tokens: list[str] = []
    if require_token_files:
        for row in rows:
            token = Path(str(row.get("token_file", "")))
            token = token if token.is_absolute() else manifest.parent / token
            if not token.is_file():
                missing_tokens.append(str(token))
                if len(missing_tokens) >= 3:
                    break
    result = {
        "manifest": str(manifest),
        "sha256": sha256(manifest),
        "dataset_counts": dict(sorted(counts.items())),
        "total": len(rows),
        "duplicate_sample_key_count": duplicate_count,
        "missing_token_files": missing_tokens,
    }
    if dict(counts) != expected or len(rows) != sum(expected.values()) or duplicate_count or missing_tokens:
        raise ValueError(
            "formal training manifest composition gate failed: "
            f"expected={expected} actual={dict(counts)} total={len(rows)} "
            f"duplicate_sample_key={duplicate_count} missing_token_files={missing_tokens}"
        )
    return result


def audit_test_assets(canonical_manifest: Path, token_manifest: Path, subset: Path) -> dict[str, Any]:
    canonical_manifest = Path(canonical_manifest).resolve()
    token_manifest = Path(token_manifest).resolve()
    subset = Path(subset).resolve()
    meta_path = canonical_manifest.parent / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if metadata.get("split") != "test":
        raise ValueError(f"formal evaluation requires MagicBrush TEST metadata, got split={metadata.get('split')!r}")
    canonical_rows = read_jsonl(canonical_manifest)
    if metadata.get("session_count") != FORMAL_TEST_SESSIONS or len(canonical_rows) != FORMAL_TEST_TURNS:
        raise ValueError(
            "REAL TEST ARCHIVE NOT VERIFIED: expected official MagicBrush TEST "
            f"{FORMAL_TEST_SESSIONS} sessions / {FORMAL_TEST_TURNS} turns, got "
            f"{metadata.get('session_count')} sessions / {len(canonical_rows)} turns"
        )
    rows = read_jsonl(token_manifest)
    subset_rows = read_jsonl(subset)
    canonical_keys = [str(row.get("sample_key", "")) for row in canonical_rows]
    keys = [str(row.get("sample_key", "")) for row in rows]
    subset_keys = [str(row.get("sample_key", "")) for row in subset_rows]
    if len(canonical_keys) != len(set(canonical_keys)) or len(keys) != len(set(keys)) or len(subset_keys) != len(set(subset_keys)):
        raise ValueError("formal MagicBrush test manifest/subset contains duplicate sample_key values")
    if canonical_keys != keys or canonical_keys != subset_keys:
        raise ValueError("formal TEST canonical/token/subset sample_key order must be identical")
    if len(rows) != FORMAL_TEST_TURNS or len(subset_rows) != FORMAL_TEST_TURNS:
        raise ValueError("REAL TEST ARCHIVE NOT VERIFIED: token manifest and eval subset must both contain 1053 turns")
    missing_tokens = []
    invalid_payloads = []
    for row in rows:
        token = Path(str(row.get("token_file", "")))
        token = token if token.is_absolute() else token_manifest.parent / token
        if not token.is_file():
            missing_tokens.append(str(token))
            if len(missing_tokens) >= 3:
                break
            continue
        import torch
        payload = torch.load(token, map_location="cpu", weights_only=True)
        if (tuple(getattr(payload.get("source_codes"), "shape", ())) != (32, 32) or tuple(getattr(payload.get("target_codes"), "shape", ())) != (32, 32)
                or tuple(getattr(payload.get("edit_mask"), "shape", ())) != (32, 32) or not bool(payload.get("edit_mask", torch.zeros(())).any())
                or payload.get("token_height") != 32 or payload.get("token_width") != 32):
            invalid_payloads.append(str(token))
            if len(invalid_payloads) >= 3:
                break
    if missing_tokens:
        raise FileNotFoundError(f"formal MagicBrush TEST token files are missing: {missing_tokens}")
    if invalid_payloads:
        raise ValueError(f"formal MagicBrush TEST token payload contract failed: {invalid_payloads}")
    return {
        "split": "test",
        "canonical_manifest": {"path": str(canonical_manifest), "sha256": sha256(canonical_manifest), "sample_count": len(canonical_rows)},
        "token_manifest": {"path": str(token_manifest), "sha256": sha256(token_manifest), "sample_count": len(rows)},
        "eval_subset": {"path": str(subset), "sha256": sha256(subset), "sample_count": len(subset_rows)},
        "duplicate_sample_key_count": 0,
        "official_session_count": FORMAL_TEST_SESSIONS,
        "official_turn_count": FORMAL_TEST_TURNS,
    }


def metric_model_identity(model: Path, *, model_id: str) -> dict[str, Any]:
    model = Path(model).resolve()
    config = _file(model / "config.json", required=True)
    payload = json.loads((model / "config.json").read_text(encoding="utf-8"))
    return {
        "model_id": model_id,
        "model_root": str(model),
        "revision": payload.get("_commit_hash") or payload.get("revision"),
        "config": config,
        "weights": _weights_identity(model),
    }


def lpips_alex_identity() -> dict[str, Any]:
    checkpoint = Path(torch_hub_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    return {
        "net": "alex",
        "lpips_version": importlib.metadata.version("lpips"),
        "torchvision_alexnet_checkpoint": _file(checkpoint, required=True),
    }


def torch_hub_dir() -> str:
    # Import torchvision/torch lazily so the audit's manifest-only helpers are lightweight.
    import torch

    return torch.hub.get_dir()
