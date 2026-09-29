"""Shared parquet and tokenization helpers for strictly audited edit datasets."""
from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from dataset.geometry import apply_shared_geometry, sample_shared_geometry
from dataset.magicbrush import stable_sample_seed
from dataset.preprocess import encode, token_mask
from dataset.utils import read_jsonl, write_jsonl


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


def pretokenize_sharded_records(
    *,
    raw_root: Path,
    model: Path,
    output: Path,
    seed: int,
    target_size: int,
    dataset_name: str,
    shards: list[Path],
    iter_records: Callable[[Path, Iterable[Path]], Iterator[dict]],
    max_samples: int = 0,
) -> Path:
    """Tokenize deterministic shard slices; rank zero merges a stable manifest."""
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
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vqvae.config.block_out_channels) - 1), do_normalize=False)
    rows: list[dict] = []
    # Each rank owns a disjoint, deterministic slice of parquet shards.  This
    # avoids concurrent token writes and avoids every rank decoding every row.
    owned_shards = [shards[position] for position in range(rank, len(shards), world_size)]
    with torch.no_grad():
        for record in iter_records(raw_root, owned_shards):
            if max_samples and len(rows) >= max_samples:
                break
            geometry_seed = stable_sample_seed(seed, 0, record["sample_key"])
            geometry = sample_shared_geometry(record["source"].size, geometry_seed, target_size)
            source = apply_shared_geometry(record["source"], geometry)
            target = apply_shared_geometry(record["target"], geometry)
            mask = apply_shared_geometry(record["mask"], geometry, is_mask=True)
            source_codes, source_shape = encode(vqvae, processor, source)
            target_codes, target_shape = encode(vqvae, processor, target)
            edit_mask = token_mask(mask, source_shape, device)
            if source_shape != target_shape or source_shape != (32, 32) or tuple(edit_mask.shape) != (32, 32):
                raise ValueError(f"unexpected VQ contract for {record['sample_key']}")
            if not bool(edit_mask.any()):
                raise ValueError(f"empty token mask after geometry for {record['sample_key']}")
            token_file = files / _token_name(record["sample_key"])
            if token_file.exists():
                raise FileExistsError(f"refusing to overwrite token file: {token_file}")
            torch.save({
                "source_codes": source_codes, "target_codes": target_codes, "edit_mask": edit_mask,
                "token_height": 32, "token_width": 32,
                "processed_width": source.width, "processed_height": source.height,
            }, token_file)
            rows.append({
                "dataset_name": dataset_name, "sample_key": record["sample_key"],
                "instruction": record["instruction"], "task": record.get("task", ""),
                "token_file": token_file.relative_to(output).as_posix(),
                "token_height": 32, "token_width": 32, "geometry": geometry.as_dict(),
                "geometry_seed": geometry_seed, "_sort_key": record["sort_key"],
            })
    write_jsonl(output / f"manifest.rank{rank:02d}.jsonl", rows)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for shard_rank in range(world_size):
            merged.extend(read_jsonl(output / f"manifest.rank{shard_rank:02d}.jsonl"))
        merged.sort(key=lambda row: tuple(row["_sort_key"]))
        for row in merged:
            row.pop("_sort_key")
        write_jsonl(output / "manifest.jsonl", merged)
        (output / "dataset_meta.json").write_text(json.dumps({
            "dataset_name": dataset_name, "usable_row_count": len(merged),
            "token_grid": [32, 32], "manifest": "manifest.jsonl",
        }, indent=2) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.destroy_process_group()
    return output / "manifest.jsonl"
