#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageDraw

from dataset.geometry import (
    SharedGeometry,
    apply_shared_geometry,
    enrich_geometry,
    resolve_record_paths,
    split_by_session,
)
from dataset.utils import read_jsonl, write_jsonl


def encode(vqvae, processor, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    pixels = processor.preprocess(image).to(vqvae.device, dtype=vqvae.dtype)
    latents = vqvae.encode(pixels).latents
    height, width = latents.shape[-2:]
    indices = vqvae.quantize(latents)[2][2].reshape(height, width).to(torch.int32).cpu()
    return indices, (height, width)


def token_mask(mask: Image.Image, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    values = torch.from_numpy(__import__("numpy").asarray(mask, dtype="uint8").copy()).to(device)
    values = (values >= 128).float()[None, None]
    pooled = F.adaptive_max_pool2d(values, shape)[0, 0]
    return pooled.bool().cpu()


def pretokenize_magicbrush(manifest: Path, model: Path, output: Path) -> Path:
    """Encode a prepared manifest into VQ tokens without duplicating dataset contracts."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rows = read_jsonl(manifest)
    output.mkdir(parents=True, exist_ok=True)
    files = output / "files"
    files.mkdir(exist_ok=True)
    vqvae = VQModel.from_pretrained(
        model,
        subfolder="vqvae",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval()
    scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    processor = VaeImageProcessor(vae_scale_factor=scale, do_normalize=False)
    output_rows = []
    with torch.no_grad():
        for position in range(rank, len(rows), world_size):
            row = rows[position]
            destination = files / f"{position:06d}.pt"
            geometry = SharedGeometry(**row["geometry"])
            source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
            target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
            mask = apply_shared_geometry(Image.open(row["mask_edit"]).convert("L"), geometry, is_mask=True)
            source_codes, source_shape = encode(vqvae, processor, source)
            target_codes, target_shape = encode(vqvae, processor, target)
            if source_shape != target_shape:
                raise ValueError(f"VQ shape mismatch for {row['sample_key']}: {source_shape} vs {target_shape}")
            edit_mask = token_mask(mask, source_shape, device)
            if edit_mask.numel() != source_codes.numel():
                raise AssertionError("mask and VQ token counts differ")
            torch.save(
                {
                    "source_codes": source_codes,
                    "target_codes": target_codes,
                    "edit_mask": edit_mask,
                    "token_height": source_shape[0],
                    "token_width": source_shape[1],
                    "processed_width": source.width,
                    "processed_height": source.height,
                },
                destination,
            )
            output_rows.append(
                {
                    **row,
                    "token_file": str(destination),
                    "token_height": source_shape[0],
                    "token_width": source_shape[1],
                }
            )
            if len(output_rows) % 10 == 0:
                print(json.dumps({"rank": rank, "complete": len(output_rows), "last_position": position}), flush=True)
    shard = output / f"manifest.rank{rank:02d}.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        combined = []
        for shard_rank in range(world_size):
            combined.extend(read_jsonl(output / f"manifest.rank{shard_rank:02d}.jsonl"))
        combined.sort(key=lambda row: int(Path(row["token_file"]).stem))
        with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in combined:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"samples": len(combined), "output": str(output / "manifest.jsonl")}, indent=2))
    if world_size > 1:
        dist.destroy_process_group()
    return output / "manifest.jsonl"


def _preview(row: dict, output: Path) -> None:
    geometry = SharedGeometry(**row["geometry"])
    source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
    target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
    mask = apply_shared_geometry(Image.open(row["mask_edit"]).convert("L"), geometry, is_mask=True)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    red = Image.new("RGB", source.size, (255, 0, 0))
    alpha = mask.point(lambda value: 110 if value else 0)
    overlay_source, overlay_target = source.copy(), target.copy()
    overlay_source.paste(red, mask=alpha)
    overlay_target.paste(red, mask=alpha)
    panels = [source, target, mask.convert("RGB"), overlay_source, overlay_target]
    header = 42
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), header + max(panel.height for panel in panels)), "white")
    ImageDraw.Draw(canvas).text((8, 8), f"{row['sample_key']} | {row['instruction']}", fill="black")
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, header))
        left += panel.width
    canvas.save(output)


def prepare_magicbrush(
    train_manifest: Path,
    output: Path,
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    target_size: int = 512,
    calibration_samples: int = 200,
    preview_samples: int = 100,
) -> dict:
    """Create deterministic train/validation/probe manifests from raw MagicBrush records."""
    output.mkdir(parents=True, exist_ok=True)
    rows = [resolve_record_paths(row, train_manifest) for row in read_jsonl(train_manifest)]
    rows = enrich_geometry(rows, seed, target_size)
    train, val = split_by_session(rows, val_fraction, seed)
    calibration = random.Random(seed).sample(train, calibration_samples)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "val.jsonl", val)
    write_jsonl(output / f"calibration{calibration_samples}_seed{seed}.jsonl", calibration)
    preview_dir = output / f"alignment_preview{preview_samples}"
    preview_dir.mkdir(exist_ok=True)
    for row in random.Random(seed).sample(train, min(preview_samples, len(train))):
        _preview(row, preview_dir / f"{int(row['index']):05d}.jpg")
    summary = {
        "seed": seed,
        "target_size": target_size,
        "all_samples": len(rows),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_sessions": len({row["session_id"] for row in train}),
        "val_sessions": len({row["session_id"] for row in val}),
        "calibration_samples": len(calibration),
        "preview_samples": len(list(preview_dir.glob("*.jpg"))),
    }
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
