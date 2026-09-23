"""Pure helpers for GT-mask hard-lock editing evaluation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset.utils import read_jsonl, write_jsonl


IMAGE_TOKEN_OFFSET = 126356


def resolve_token_file(row: dict[str, Any], manifest: Path) -> Path:
    """Resolve both legacy absolute and portable manifest-relative token paths."""
    token_file = Path(row["token_file"])
    return token_file if token_file.is_absolute() else Path(manifest).parent / token_file


def load_token_payload(row: dict[str, Any], manifest: Path) -> dict[str, Any]:
    token_file = resolve_token_file(row, manifest)
    if not token_file.is_file():
        raise FileNotFoundError(f"missing token file for {row.get('sample_key')}: {token_file}")
    payload = torch.load(token_file, map_location="cpu", weights_only=True)
    required = {
        "source_codes", "target_codes", "edit_mask", "token_height", "token_width",
        "processed_height", "processed_width",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"token payload {token_file} is missing {sorted(missing)}")
    shape = (int(payload["token_height"]), int(payload["token_width"]))
    if tuple(payload["source_codes"].shape) != shape or tuple(payload["target_codes"].shape) != shape:
        raise ValueError(f"token geometry does not match codes in {token_file}")
    if tuple(payload["edit_mask"].shape) != shape:
        raise ValueError(f"token geometry does not match edit mask in {token_file}")
    return payload


def prepare_eval_subset(manifest: Path, subset: Path, *, seed: int, limit: int) -> list[dict[str, Any]]:
    """Persist a deterministic held-out subset with stable per-sample seeds."""
    manifest = Path(manifest)
    eligible = []
    for row in read_jsonl(manifest):
        payload = load_token_payload(row, manifest)
        if bool(payload["edit_mask"].any()):
            eligible.append(dict(row))
    if len(eligible) < limit:
        raise ValueError(f"only {len(eligible)} non-empty-mask samples, need {limit}")
    selected = random.Random(seed).sample(eligible, limit)
    output = []
    for eval_index, row in enumerate(selected):
        row = dict(row)
        row["eval_index"] = eval_index
        row["inference_seed"] = seed + eval_index
        output.append(row)
    subset.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset, output)
    return output


def read_eval_subset(subset: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(subset))
    expected = list(range(len(rows)))
    if [row.get("eval_index") for row in rows] != expected:
        raise ValueError("eval_subset rows must have contiguous eval_index values")
    if len({row.get("sample_key") for row in rows}) != len(rows):
        raise ValueError("eval_subset contains duplicate sample_key values")
    return rows


def load_lora_recipe(checkpoint: Path | None) -> dict[str, Any] | None:
    """Read the checkpoint recipe without requiring model construction."""
    if checkpoint is None:
        return None
    checkpoint = Path(checkpoint)
    meta = checkpoint / "checkpoint_meta.json"
    if meta.is_file():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        recipe = payload.get("fingerprint", {}).get("payload", {}).get("lora")
        if recipe is not None:
            return {
                "rank": int(recipe["rank"]),
                "alpha": float(recipe["alpha"]),
                "dropout": float(recipe["dropout"]),
                "targets": tuple(recipe["targets"]),
            }
    experiment = checkpoint.parent / "experiment_config.json"
    if experiment.is_file():
        args = json.loads(experiment.read_text(encoding="utf-8")).get("args", {})
        keys = ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets")
        if all(key in args for key in keys):
            return {
                "rank": int(args["lora_rank"]),
                "alpha": float(args["lora_alpha"]),
                "dropout": float(args["lora_dropout"]),
                "targets": tuple(args["lora_targets"]),
            }
    raise ValueError(f"cannot determine LoRA recipe from checkpoint {checkpoint}")


def token_accuracies(
    generated_with_offset: torch.Tensor,
    source_codes: torch.Tensor,
    target_codes: torch.Tensor,
    edit_mask: torch.Tensor,
    *,
    image_token_offset: int = IMAGE_TOKEN_OFFSET,
) -> dict[str, float]:
    generated = generated_with_offset.reshape(-1).long() - image_token_offset
    source = source_codes.reshape(-1).long()
    target = target_codes.reshape(-1).long()
    mask = edit_mask.reshape(-1).bool()
    if generated.numel() != source.numel() or source.shape != target.shape or source.shape != mask.shape:
        raise ValueError("generated/source/target/mask token shapes differ")
    outside = ~mask
    outside_accuracy = float((generated[outside] == source[outside]).float().mean()) if outside.any() else 1.0
    if outside_accuracy != 1.0:
        raise AssertionError(f"hard-lock violation: outside_token_accuracy={outside_accuracy}")
    edit_accuracy = float((generated[mask] == target[mask]).float().mean()) if mask.any() else float("nan")
    return {
        "edit_token_accuracy": edit_accuracy,
        "outside_token_accuracy": outside_accuracy,
    }


def effective_pixel_mask(token_mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor token-mask upsampling, returned as HxW boolean."""
    width, height = map(int, size)
    values = token_mask.bool().float()[None, None]
    return F.interpolate(values, size=(height, width), mode="nearest")[0, 0].bool().cpu().numpy()


def boundary_ring(mask: np.ndarray, radius: int = 32) -> np.ndarray:
    """Return dilate(mask, radius) minus mask without external morphology deps."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    mask = np.asarray(mask, dtype=bool)
    if radius == 0:
        return np.zeros_like(mask)
    values = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=radius)[0, 0].bool().numpy()
    return dilated & ~mask


def _masked_l1_mse_psnr(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    if not bool(mask.any()):
        return {"l1": None, "mse": None, "psnr": None}
    difference = prediction[mask] - reference[mask]
    l1 = float(np.abs(difference).mean())
    mse = float(np.square(difference).mean())
    return {"l1": l1, "mse": mse, "psnr": float(-10.0 * math.log10(max(mse, 1e-12)))}


def pixel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_reconstruction: np.ndarray,
    source_reconstruction: np.ndarray,
    effective_mask: np.ndarray,
    *,
    boundary_radius: int = 32,
) -> dict[str, float | None]:
    """Core hard-lock metrics; all images are HxWx3 floats in [0, 1]."""
    arrays = (prediction, target, target_reconstruction, source_reconstruction)
    if any(array.shape != prediction.shape for array in arrays) or prediction.ndim != 3:
        raise ValueError("pixel metric images must share HxWxC shape")
    mask = np.asarray(effective_mask, dtype=bool)
    if mask.shape != prediction.shape[:2]:
        raise ValueError("effective mask/image geometry mismatch")
    inside_target = _masked_l1_mse_psnr(prediction, target, mask)
    inside_recon = _masked_l1_mse_psnr(prediction, target_reconstruction, mask)
    full = _masked_l1_mse_psnr(prediction, target, np.ones_like(mask, dtype=bool))
    ring = boundary_ring(mask, radius=boundary_radius)
    boundary = _masked_l1_mse_psnr(prediction, source_reconstruction, ring)
    return {
        "inside_l1_target": inside_target["l1"],
        "inside_mse_target": inside_target["mse"],
        "inside_psnr_target": inside_target["psnr"],
        "inside_l1_target_recon": inside_recon["l1"],
        "full_l1_target": full["l1"],
        "full_mse_target": full["mse"],
        "full_psnr_target": full["psnr"],
        "boundary_l1_source_recon": boundary["l1"],
        "effective_mask_fraction": float(mask.mean()),
        "boundary_ring_fraction": float(ring.mean()),
    }


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Mean every numeric per-sample metric while leaving identifiers untouched."""
    summary: dict[str, float | int | None] = {"samples": len(rows)}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if finite:
            summary[key] = float(np.mean(finite))
    return summary
