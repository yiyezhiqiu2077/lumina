#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from datasets.magicbrush_dataset import SharedGeometry, apply_shared_geometry, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rows = read_jsonl(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    files = args.output / "files"
    files.mkdir(exist_ok=True)
    vqvae = VQModel.from_pretrained(
        args.model,
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
    shard = args.output / f"manifest.rank{rank:02d}.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        combined = []
        for shard_rank in range(world_size):
            combined.extend(read_jsonl(args.output / f"manifest.rank{shard_rank:02d}.jsonl"))
        combined.sort(key=lambda row: int(Path(row["token_file"]).stem))
        with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in combined:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"samples": len(combined), "output": str(args.output / "manifest.jsonl")}, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
