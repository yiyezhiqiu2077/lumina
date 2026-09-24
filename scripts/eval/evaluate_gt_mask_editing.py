#!/usr/bin/env python3
"""Evaluate GT-mask hard-lock editing on a fixed held-out MagicBrush subset."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import VQModel
from PIL import Image
from transformers import AutoTokenizer

from dataset.geometry import SharedGeometry, apply_shared_geometry
from dataset.utils import write_jsonl
from evaluation.perceptual_metrics import (
    CLIPImageMetric,
    DINOImageMetric,
    LPIPSMetric,
    image_similarity_metrics,
)
from evaluation.gt_mask_editing import (
    IMAGE_TOKEN_OFFSET,
    effective_pixel_mask,
    load_lora_recipe,
    load_token_payload,
    mask_ratio_binned_summary,
    metric_summary,
    oracle_codes,
    oracle_token_accuracies,
    pixel_metrics,
    prepare_eval_subset,
    read_eval_subset,
    token_accuracies,
)
from generators.masked_image_edit_generator import generate_i2i_gt_mask_hard_lock
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration
from training.lora import inject_lora, load_lora_state_dict
from utils.constants import SPECIAL_TOKENS
from utils.image_utils import add_break_line, decode_vq_to_image
from utils.prompt_utils import create_prompt_templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="held-out token manifest")
    parser.add_argument("--subset", type=Path, required=True, help="persisted fixed eval_subset.jsonl")
    parser.add_argument("--prepare-subset-only", action="store_true")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-label", type=str)
    parser.add_argument("--oracle-only", action="store_true", help="decode/evaluate the GT-mask token oracle without loading a denoiser")
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--cfg-img", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lpips", action="store_true", help="compute local pretrained LPIPS metrics")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--dino-model", type=Path, help="local Hugging Face DINO/DINOv2 image model")
    parser.add_argument("--clip-model", type=Path, help="local Hugging Face CLIP image model")
    parser.add_argument("--roi-padding-ratio", type=float, default=0.10)
    return parser.parse_args()


def _spatial_tokens(codes: torch.Tensor) -> list[int]:
    height, width = codes.shape
    return [SPECIAL_TOKENS["boi"]] + add_break_line(
        (codes.flatten().long() + IMAGE_TOKEN_OFFSET).tolist(), height, width, SPECIAL_TOKENS["newline_token"]
    ) + [SPECIAL_TOKENS["eoi"]]


def _target_tokens(source: torch.Tensor, edit_mask: torch.Tensor) -> list[int]:
    values = source.flatten().long() + IMAGE_TOKEN_OFFSET
    values[edit_mask.flatten().bool()] = SPECIAL_TOKENS["mask_token"]
    height, width = source.shape
    return add_break_line(values.tolist(), height, width, SPECIAL_TOKENS["newline_token"])


def _image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _release_cuda() -> None:
    """Release one evaluation stage before loading the next optional metric model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def perceptual_metrics_requested(args: argparse.Namespace) -> bool:
    """Keep the pre-existing evaluator path untouched unless a new metric is requested."""
    return bool(args.lpips or args.dino_model is not None or args.clip_model is not None)


def _load_model(model_path: Path, checkpoint: Path | None, device: torch.device):
    model = LLaDAForMultiModalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    if checkpoint is not None:
        recipe = load_lora_recipe(checkpoint)
        inject_lora(model, targets=recipe["targets"], rank=recipe["rank"], alpha=recipe["alpha"], dropout=recipe["dropout"])
        state_path = checkpoint / "lora.pt"
        load_lora_state_dict(model, torch.load(state_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.roi_padding_ratio < 0:
        raise SystemExit("--roi-padding-ratio must be non-negative")
    perceptual_requested = perceptual_metrics_requested(args)
    if args.prepare_subset_only:
        rows = prepare_eval_subset(args.manifest, args.subset, seed=args.seed, limit=args.limit)
        print(json.dumps({"subset": str(args.subset), "samples": len(rows), "seed": args.seed}, indent=2))
        return
    if args.model is None or args.output is None or args.model_label is None:
        raise SystemExit("--model, --output, and --model-label are required for evaluation")
    if args.oracle_only and args.checkpoint is not None:
        raise SystemExit("--oracle-only cannot be combined with --checkpoint")
    rows = read_eval_subset(args.subset)
    if len(rows) > args.limit:
        raise ValueError(f"fixed subset has {len(rows)} rows, limit={args.limit}")
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    source_recon_dir = args.output / "source_recon"
    target_recon_dir = args.output / "target_recon"
    target_dir = args.output / "targets"
    mask_dir = args.output / "effective_masks"
    oracle_dir = args.output / "oracle_hardlock"
    directories = [image_dir, source_recon_dir, target_recon_dir, oracle_dir]
    if perceptual_requested:
        directories += [target_dir, mask_dir]
    for directory in directories:
        directory.mkdir(exist_ok=True)
    (args.output / "eval_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    device = torch.device("cuda")
    tokenizer = None
    model = None
    if not args.oracle_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        model = _load_model(args.model, args.checkpoint, device)
    vqvae = VQModel.from_pretrained(args.model, subfolder="vqvae", local_files_only=True).to(device).eval()
    system = create_prompt_templates()["image_editing"]
    output_rows = []
    for row in rows:
        payload = load_token_payload(row, args.subset)
        source_codes = payload["source_codes"].long()
        target_codes = payload["target_codes"].long()
        edit_mask = payload["edit_mask"].bool()
        token_height, token_width = int(payload["token_height"]), int(payload["token_width"])
        processed_size = (int(payload["processed_width"]), int(payload["processed_height"]))
        if tuple(source_codes.shape) != (token_height, token_width):
            raise AssertionError("source code geometry is inconsistent")
        oracle = oracle_codes(source_codes, target_codes, edit_mask)
        oracle_with_offset = (oracle.flatten()[None] + IMAGE_TOKEN_OFFSET).to(device)
        if args.oracle_only:
            generated = oracle_with_offset
            seconds = 0.0
            accuracies = oracle_token_accuracies(source_codes, target_codes, edit_mask)
        else:
            assert tokenizer is not None and model is not None
            conditional_ids = tokenizer(f"<system>{system}</system><user>{row['instruction']}</user>", truncation=False, padding=False)["input_ids"]
            unconditional_ids = tokenizer(f"<system>{system}</system><user><uncondition></user>", truncation=False, padding=False)["input_ids"]
            source_sequence = _spatial_tokens(source_codes)
            conditional_prefix = conditional_ids[:-1] + source_sequence + conditional_ids[-1:]
            unconditional_prefix = unconditional_ids[:-1] + source_sequence + unconditional_ids[-1:]
            target_sequence = [SPECIAL_TOKENS["answer_start"], SPECIAL_TOKENS["boi"]]
            target_sequence += _target_tokens(source_codes, edit_mask)
            target_sequence += [SPECIAL_TOKENS["eoi"], SPECIAL_TOKENS["answer_end"]]
            code_start = len(conditional_prefix) + 2
            prompt = torch.tensor([conditional_prefix + target_sequence], device=device)
            generator = torch.Generator(device=device).manual_seed(int(row["inference_seed"]))
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            generated = generate_i2i_gt_mask_hard_lock(
                model, prompt, source_codes=source_codes, edit_mask=edit_mask, code_start=code_start,
                timesteps=args.timesteps, temperature=args.temperature, cfg_scale=args.cfg_scale, cfg_img=args.cfg_img,
                uncon_text=torch.tensor([unconditional_prefix], device=device),
                uncon_image=torch.tensor([conditional_ids], device=device), generator=generator,
            )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            accuracies = token_accuracies(generated, source_codes, target_codes, edit_mask)
        prediction = decode_vq_to_image(generated, "unused.png", str(args.model), processed_size[1], processed_size[0], vqvae=vqvae)
        source_reconstruction = decode_vq_to_image((source_codes.flatten()[None] + IMAGE_TOKEN_OFFSET).to(device), "unused.png", str(args.model), processed_size[1], processed_size[0], vqvae=vqvae)
        target_reconstruction = decode_vq_to_image((target_codes.flatten()[None] + IMAGE_TOKEN_OFFSET).to(device), "unused.png", str(args.model), processed_size[1], processed_size[0], vqvae=vqvae)
        oracle_image = decode_vq_to_image(oracle_with_offset, "unused.png", str(args.model), processed_size[1], processed_size[0], vqvae=vqvae)
        geometry = SharedGeometry(**row["geometry"])
        target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
        if target.size != processed_size:
            raise AssertionError(f"processed target geometry {target.size} != payload {processed_size}")
        effective = effective_pixel_mask(edit_mask, prediction.size)
        metrics = pixel_metrics(_image_array(prediction), _image_array(target), _image_array(target_reconstruction), _image_array(source_reconstruction), effective)
        oracle_metrics = pixel_metrics(_image_array(oracle_image), _image_array(target), _image_array(target_reconstruction), _image_array(source_reconstruction), effective)
        position = int(row["eval_index"])
        filename = f"{position:03d}.png"
        prediction.save(image_dir / filename)
        source_reconstruction.save(source_recon_dir / filename)
        target_reconstruction.save(target_recon_dir / filename)
        if perceptual_requested:
            target.save(target_dir / filename)
            Image.fromarray((effective.astype(np.uint8) * 255), mode="L").save(mask_dir / filename)
        oracle_image.save(oracle_dir / filename)
        record = {
            "model": args.model_label, "eval_index": position, "sample_key": row["sample_key"],
            "inference_seed": int(row["inference_seed"]), "instruction": row["instruction"],
            "token_height": token_height, "token_width": token_width,
            "processed_width": processed_size[0], "processed_height": processed_size[1],
            "seconds": seconds, "mask_ratio": float(edit_mask.float().mean()),
            **accuracies, **metrics,
            "oracle_metrics": oracle_metrics,
        }
        output_rows.append(record)
        print(json.dumps(record), flush=True)

    # Persist the generation-stage diagnostics before freeing Lumina/VQ memory.
    write_jsonl(args.output / "per_sample.jsonl", output_rows)
    if perceptual_requested:
        del model
        del vqvae
        del tokenizer
        _release_cuda()

        def image_inputs(record: dict) -> tuple[Image.Image, Image.Image, np.ndarray]:
            filename = f"{int(record['eval_index']):03d}.png"
            prediction = Image.open(image_dir / filename).convert("RGB")
            target = Image.open(target_dir / filename).convert("RGB")
            mask = np.asarray(Image.open(mask_dir / filename).convert("L"), dtype=np.uint8) > 0
            return prediction, target, mask

        if args.lpips:
            metric = LPIPSMetric(net=args.lpips_net, device=device)
            for record in output_rows:
                prediction, target, mask = image_inputs(record)
                record.update(metric.scores(prediction, target, mask))
            del metric
            _release_cuda()
        if args.dino_model is not None:
            metric = DINOImageMetric(args.dino_model, device=device)
            for record in output_rows:
                prediction, target, mask = image_inputs(record)
                record.update(
                    image_similarity_metrics(
                        metric, prediction, target, mask, metric_name="dino_i", roi_padding_ratio=args.roi_padding_ratio
                    )
                )
            del metric
            _release_cuda()
        if args.clip_model is not None:
            metric = CLIPImageMetric(args.clip_model, device=device)
            for record in output_rows:
                prediction, target, mask = image_inputs(record)
                record.update(
                    image_similarity_metrics(
                        metric, prediction, target, mask, metric_name="clip_i", roi_padding_ratio=args.roi_padding_ratio
                    )
                )
            del metric
            _release_cuda()

    # Rewrite the same compatible output after optional metric stages finish.
    write_jsonl(args.output / "per_sample.jsonl", output_rows)
    summary = {
        "model": args.model_label,
        **metric_summary(output_rows),
        "mask_ratio_bins": mask_ratio_binned_summary(output_rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
