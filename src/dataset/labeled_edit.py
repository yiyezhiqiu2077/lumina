"""Shared parquet and resumable tokenization helpers for audited edit datasets."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from dataset.geometry import (
    apply_shared_geometry, geometry_policy, mask_retention, sample_shared_geometry, vq_stride_from_vqvae,
)
from dataset.magicbrush import stable_sample_seed
from dataset.preprocess import encode, token_mask
from dataset.token_contract import atomic_save_token, grid_key, validate_reusable_token, validate_token_payload
from dataset.utils import read_jsonl


def parquet_rows(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict]]:
    import pyarrow.parquet as pq

    for path in paths:
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                yield path, row_index, row
                row_index += 1


def decode_image(value, *, mode: str) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("expected embedded image bytes")
    with Image.open(BytesIO(value)) as image:
        return image.convert(mode).copy()


def decode_mask(value) -> Image.Image:
    return decode_image(value, mode="L")


def _token_name(sample_key: str) -> str:
    return hashlib.sha256(sample_key.encode("utf-8")).hexdigest() + ".pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "min", "p01", "p05", "p50")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)), "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
    }


def _load_identity(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def pretokenize_sharded_records(
    *, raw_root: Path, model: Path, output: Path, seed: int, target_size: int,
    dataset_name: str, shards: list[Path], iter_records: Callable[[Path, Iterable[Path]], Iterator[dict]],
    max_samples: int = 0, strict_audit_metadata: Path | None = None,
) -> Path:
    """Tokenize deterministic shard slices; valid existing tokens are reused.

    Every rank recreates its own manifest. Token files are safe to reuse only
    after their complete payload, identity, geometry and variable-grid contract
    have been verified. A corrupt existing file is a hard error rather than a
    silent overwrite.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if max_samples and world_size != 1:
        raise ValueError("max_samples is only supported for single-rank smoke tokenization")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output.mkdir(parents=True, exist_ok=True)
    files = output / "files"
    files.mkdir(exist_ok=True)
    vqvae = VQModel.from_pretrained(model, subfolder="vqvae", torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    vq_stride = vq_stride_from_vqvae(vqvae)
    processor = VaeImageProcessor(vae_scale_factor=vq_stride, do_normalize=False)
    rows: list[dict] = []
    skipped: Counter[str] = Counter()
    retentions: list[float] = []
    observed_grids: Counter[str] = Counter()
    # Each rank owns an exclusive, deterministic set of shards: no token file
    # can be produced concurrently by two ranks.
    owned_shards = [shards[position] for position in range(rank, len(shards), world_size)]
    with torch.no_grad():
        for record in iter_records(raw_root, owned_shards, on_skip=lambda reason: skipped.update([reason])):
            if max_samples and len(rows) >= max_samples:
                break
            geometry_seed = stable_sample_seed(seed, 0, record["sample_key"])
            geometry = sample_shared_geometry(record["source"].size, geometry_seed, target_size)
            retention = mask_retention(record["mask"], geometry)
            if retention["post_geometry_empty_mask"]:
                skipped["post_geometry_empty_mask"] += 1
                continue
            token_file = files / _token_name(record["sample_key"])
            if token_file.exists():
                source_shape = validate_reusable_token(
                    token_file, sample_key=record["sample_key"], dataset_name=dataset_name,
                    geometry_seed=geometry_seed, geometry=geometry.as_dict(),
                )
            else:
                source = apply_shared_geometry(record["source"], geometry)
                target = apply_shared_geometry(record["target"], geometry)
                mask = apply_shared_geometry(record["mask"], geometry, is_mask=True)
                source_codes, source_shape = encode(vqvae, processor, source)
                target_codes, target_shape = encode(vqvae, processor, target)
                edit_mask = token_mask(mask, source_shape, device)
                if source_shape != target_shape:
                    raise ValueError(f"VQ source/target mismatch for {record['sample_key']}")
                payload = {
                    "dataset_name": dataset_name, "sample_key": record["sample_key"],
                    "geometry_seed": geometry_seed, "geometry": geometry.as_dict(), "vq_stride": vq_stride,
                    "source_codes": source_codes, "target_codes": target_codes, "edit_mask": edit_mask,
                    "token_height": source_shape[0], "token_width": source_shape[1],
                    "processed_width": source.width, "processed_height": source.height,
                }
                try:
                    validate_token_payload(payload, sample_key=record["sample_key"], dataset_name=dataset_name, require_processed_geometry=True)
                except ValueError as error:
                    if "empty edit mask" in str(error):
                        skipped["post_geometry_empty_mask"] += 1
                        continue
                    raise
                atomic_save_token(payload, token_file, sample_key=record["sample_key"], dataset_name=dataset_name)
            height, width = source_shape
            retentions.append(float(retention["mask_retention_ratio"]))
            observed_grids[grid_key(height, width)] += 1
            rows.append({
                "dataset_name": dataset_name, "sample_key": record["sample_key"],
                "instruction": record["instruction"], "task": record.get("task", ""),
                "token_file": token_file.relative_to(output).as_posix(),
                "token_height": height, "token_width": width, "geometry": geometry.as_dict(),
                "geometry_seed": geometry_seed, "mask_retention": retention, "_sort_key": record["sort_key"],
            })
    _atomic_write_jsonl(output / f"manifest.rank{rank:02d}.jsonl", rows)
    rank_summary = {
        "skipped_reasons": dict(sorted(skipped.items())), "mask_retentions": retentions,
        "observed_token_grids": dict(sorted(observed_grids.items())),
    }
    (output / f"summary.rank{rank:02d}.json").write_text(json.dumps(rank_summary), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged: list[dict] = []
        all_skipped: Counter[str] = Counter()
        all_retentions: list[float] = []
        all_grids: Counter[str] = Counter()
        for shard_rank in range(world_size):
            merged.extend(read_jsonl(output / f"manifest.rank{shard_rank:02d}.jsonl"))
            summary = _load_identity(output / f"summary.rank{shard_rank:02d}.json")
            all_skipped.update(summary.get("skipped_reasons", {}))
            all_retentions.extend(summary.get("mask_retentions", []))
            all_grids.update(summary.get("observed_token_grids", {}))
        merged.sort(key=lambda row: tuple(row["_sort_key"]))
        if len({row["sample_key"] for row in merged}) != len(merged):
            raise ValueError("tokenization produced duplicate sample_key values")
        for row in merged:
            token = output / row["token_file"]
            payload = torch.load(token, map_location="cpu", weights_only=True)
            validate_token_payload(payload, sample_key=row["sample_key"], dataset_name=dataset_name, require_processed_geometry=True)
            row.pop("_sort_key")
        manifest = output / "manifest.jsonl"
        _atomic_write_jsonl(manifest, merged)
        download = _load_identity(Path(raw_root) / "download_manifest.json")
        audit_path = Path(strict_audit_metadata) if strict_audit_metadata else Path(raw_root).parent / "audit" / "dataset_meta.json"
        audit = _load_identity(audit_path)
        metadata = {
            "dataset_name": dataset_name,
            "repo_id": download.get("repo_id") or audit.get("repo_id"),
            "revision": download.get("resolved_revision") or audit.get("revision"),
            "parser_version": audit.get("parser_version"),
            "seed": seed, "target_size": target_size, "vq_stride": vq_stride,
            "geometry_policy": geometry_policy(target_size=target_size),
            "raw_strict_usable_count": audit.get("usable_row_count"),
            "post_geometry_rejected_count": int(all_skipped.get("post_geometry_empty_mask", 0)),
            "tokenized_usable_count": len(merged), "usable_row_count": len(merged),
            "skip_reasons": dict(sorted(all_skipped.items())),
            "observed_token_grids": dict(sorted(all_grids.items())),
            "mask_retention": {
                **_quantiles(all_retentions),
                "retention_lt_0_25_count": sum(value < 0.25 for value in all_retentions),
                "retention_lt_0_50_count": sum(value < 0.50 for value in all_retentions),
            },
            "download_manifest": {"path": str(Path(raw_root) / "download_manifest.json"), "sha256": _sha256(Path(raw_root) / "download_manifest.json")} if (Path(raw_root) / "download_manifest.json").is_file() else None,
            "strict_audit_metadata": {"path": str(audit_path), "sha256": _sha256(audit_path)} if audit_path.is_file() else None,
            "manifest": "manifest.jsonl", "manifest_sha256": _sha256(manifest),
        }
        temporary = output / f".dataset_meta.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output / "dataset_meta.json")
    if world_size > 1:
        dist.destroy_process_group()
    return output / "manifest.jsonl"
