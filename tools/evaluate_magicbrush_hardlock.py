#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from diffusers import VQModel
from PIL import Image
from transformers import AutoTokenizer

from training.lora import inject_lora, load_lora_state_dict
from config import SPECIAL_TOKENS
from datasets.magicbrush_dataset import SharedGeometry, apply_shared_geometry, read_jsonl, write_jsonl
from generators.masked_image_edit_generator import generate_i2i_gt_mask_hard_lock
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration
from utils.image_utils import add_break_line, decode_vq_to_image
from utils.prompt_utils import create_prompt_templates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--cfg-img", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def spatial_tokens(codes: torch.Tensor) -> list[int]:
    height, width = codes.shape
    return [SPECIAL_TOKENS["boi"]] + add_break_line(
        (codes.flatten().long() + SPECIAL_TOKENS["image_token_offset"]).tolist(),
        height,
        width,
        SPECIAL_TOKENS["newline_token"],
    ) + [SPECIAL_TOKENS["eoi"]]


def target_tokens(source: torch.Tensor, edit_mask: torch.Tensor) -> list[int]:
    values = source.flatten().long() + SPECIAL_TOKENS["image_token_offset"]
    values[edit_mask.flatten().bool()] = SPECIAL_TOKENS["mask_token"]
    height, width = source.shape
    return add_break_line(values.tolist(), height, width, SPECIAL_TOKENS["newline_token"])


def pixel_metrics(prediction: Image.Image, source: Image.Image, target: Image.Image, token_mask: torch.Tensor):
    pred = np.asarray(prediction, dtype=np.float32) / 255.0
    src = np.asarray(source, dtype=np.float32) / 255.0
    tgt = np.asarray(target, dtype=np.float32) / 255.0
    mask = np.asarray(
        Image.fromarray((token_mask.numpy().astype(np.uint8) * 255)).resize(prediction.size, Image.Resampling.NEAREST)
    ) > 0
    inside = np.abs(pred - tgt)[mask].mean() if mask.any() else float("nan")
    outside = np.abs(pred - src)[~mask].mean() if (~mask).any() else float("nan")
    mse = np.square(pred - tgt).mean()
    return {"inside_l1": float(inside), "outside_l1": float(outside), "psnr": float(-10 * np.log10(max(mse, 1e-12)))}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    (args.output / "eval_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    inject_lora(model, rank=16, alpha=16.0, dropout=0.05)
    load_lora_state_dict(model, torch.load(args.checkpoint / "lora.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    vqvae = VQModel.from_pretrained(args.model, subfolder="vqvae", local_files_only=True).to(device).eval()
    system = create_prompt_templates()["image_editing"]
    output_rows = []
    for position, row in enumerate(rows):
        payload = torch.load(row["token_file"], map_location="cpu", weights_only=True)
        source_codes = payload["source_codes"].long()
        edit_mask = payload["edit_mask"].bool()
        conditional_text = f"<system>{system}</system><user>{row['instruction']}</user>"
        unconditional_text = f"<system>{system}</system><user><uncondition></user>"
        conditional_ids = tokenizer(conditional_text, truncation=False, padding=False)["input_ids"]
        unconditional_ids = tokenizer(unconditional_text, truncation=False, padding=False)["input_ids"]
        source_sequence = spatial_tokens(source_codes)
        conditional_prefix = conditional_ids[:-1] + source_sequence + conditional_ids[-1:]
        unconditional_prefix = unconditional_ids[:-1] + source_sequence + unconditional_ids[-1:]
        target_sequence = [SPECIAL_TOKENS["answer_start"], SPECIAL_TOKENS["boi"]]
        target_sequence += target_tokens(source_codes, edit_mask)
        target_sequence += [SPECIAL_TOKENS["eoi"], SPECIAL_TOKENS["answer_end"]]
        code_start = len(conditional_prefix) + 2
        prompt = torch.tensor([conditional_prefix + target_sequence], device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed + position)
        started = time.perf_counter()
        generated = generate_i2i_gt_mask_hard_lock(
            model,
            prompt,
            source_codes=source_codes,
            edit_mask=edit_mask,
            code_start=code_start,
            timesteps=args.timesteps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            cfg_img=args.cfg_img,
            uncon_text=torch.tensor([unconditional_prefix], device=device),
            uncon_image=torch.tensor([conditional_ids], device=device),
            generator=generator,
        )
        prediction = decode_vq_to_image(
            generated,
            "unused.png",
            str(args.model),
            512,
            512,
            vqvae=vqvae,
        )
        elapsed = time.perf_counter() - started
        prediction.save(image_dir / f"{position:05d}.png")
        geometry = SharedGeometry(**row["geometry"])
        source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
        target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
        metrics = pixel_metrics(prediction, source, target, edit_mask)
        record = {
            "eval_index": position,
            "sample_key": row["sample_key"],
            "instruction": row["instruction"],
            "prediction": str(image_dir / f"{position:05d}.png"),
            "source": row["source"],
            "target": row["target"],
            "mask_edit": row["mask_edit"],
            "seed": args.seed + position,
            "seconds": elapsed,
            **metrics,
        }
        output_rows.append(record)
        write_jsonl(args.output / "per_sample.jsonl", output_rows)
        print(json.dumps({"sample": position + 1, "total": len(rows), **metrics, "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
