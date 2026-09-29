from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GEOMETRY_POLICY_NAME = "lumina_aspect_ratio"
DEFAULT_TARGET_SIZE = 512
DEFAULT_BUCKET_PATCH_SIZE = 32
DEFAULT_MAX_RATIO = 4.0
# Lumina-DiMOO's VQ-VAE has four downsampling blocks. Keep this in one
# contract module; runtime encoders derive the same value from VQ config.
DEFAULT_VQ_STRIDE = 16


@dataclass(frozen=True)
class SharedGeometry:
    crop_width: int
    crop_height: int
    downsample_width: int
    downsample_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int

    def as_dict(self) -> dict:
        return asdict(self)


def geometry_policy(
    *, target_size: int = DEFAULT_TARGET_SIZE,
    bucket_patch_size: int = DEFAULT_BUCKET_PATCH_SIZE,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> dict:
    return {
        "name": GEOMETRY_POLICY_NAME,
        "target_size": int(target_size),
        "bucket_patch_size": int(bucket_patch_size),
        "max_ratio": float(max_ratio),
    }


def resolve_record_paths(record: dict, manifest: Path) -> dict:
    output = dict(record)
    for key in ("source", "target", "mask_edit"):
        path = Path(output[key])
        output[key] = str(path if path.is_absolute() else manifest.parent / path)
    output["session_id"] = str(output.get("session_id", output["img_id"]))
    return output


def generate_crop_size_list(num_patches: int, patch_size: int, max_ratio: float = DEFAULT_MAX_RATIO) -> list[tuple[int, int]]:
    """Upstream Lumina crop-bucket enumeration (width, height)."""
    if num_patches <= 0 or patch_size <= 0 or max_ratio < 1.0:
        raise ValueError("invalid crop bucket parameters")
    sizes: list[tuple[int, int]] = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            sizes.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return sizes


def valid_crop_sizes(
    target_size: int = DEFAULT_TARGET_SIZE,
    bucket_patch_size: int = DEFAULT_BUCKET_PATCH_SIZE,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> list[tuple[int, int]]:
    return generate_crop_size_list((target_size // bucket_patch_size) ** 2, bucket_patch_size, max_ratio)


def vq_stride_from_vqvae(vqvae) -> int:
    return 2 ** (len(vqvae.config.block_out_channels) - 1)


def choose_crop_size(
    size: tuple[int, int], *, target_size: int = DEFAULT_TARGET_SIZE,
    bucket_patch_size: int = DEFAULT_BUCKET_PATCH_SIZE,
    max_ratio: float = DEFAULT_MAX_RATIO,
    random_top_k: int = 1,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Match upstream ``var_center_crop`` while making its RNG explicit."""
    width, height = size
    if width <= 0 or height <= 0 or random_top_k <= 0:
        raise ValueError("image dimensions and random_top_k must be positive")
    buckets = valid_crop_sizes(target_size, bucket_patch_size, max_ratio)
    scores = [min(cw / width, ch / height) / max(cw / width, ch / height) for cw, ch in buckets]
    ranked = sorted(zip(scores, buckets), reverse=True)
    choices = [bucket for _, bucket in ranked[:random_top_k]]
    return (rng or random.Random()).choice(choices)


def sample_shared_geometry(
    size: tuple[int, int], seed: int, target_size: int = DEFAULT_TARGET_SIZE,
    *, bucket_patch_size: int = DEFAULT_BUCKET_PATCH_SIZE, max_ratio: float = DEFAULT_MAX_RATIO,
) -> SharedGeometry:
    """Sample one deterministic Lumina crop shared by source, target and mask.

    The bucket is chosen by source aspect ratio, exactly as upstream
    ``var_center_crop(..., random_top_k=1)``. The fixed seed controls only the
    crop offset, never target or mask content.
    """
    width, height = size
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError("image dimensions and target_size must be positive")
    rng = random.Random(seed)
    crop_width, crop_height = choose_crop_size(
        size, target_size=target_size, bucket_patch_size=bucket_patch_size,
        max_ratio=max_ratio, random_top_k=1, rng=rng,
    )
    down_width, down_height = width, height
    while down_width >= 2 * crop_width and down_height >= 2 * crop_height:
        down_width //= 2
        down_height //= 2
    scale = max(crop_width / down_width, crop_height / down_height)
    resized_width = round(down_width * scale)
    resized_height = round(down_height * scale)
    if resized_width < crop_width or resized_height < crop_height:
        raise AssertionError("resize-to-fit must cover the crop bucket")
    return SharedGeometry(
        crop_width=crop_width, crop_height=crop_height,
        downsample_width=down_width, downsample_height=down_height,
        resized_width=resized_width, resized_height=resized_height,
        crop_left=rng.randint(0, resized_width - crop_width),
        crop_top=rng.randint(0, resized_height - crop_height),
    )


def apply_shared_geometry(image: Image.Image, geometry: SharedGeometry, is_mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    if image.size != (geometry.downsample_width, geometry.downsample_height):
        image = image.resize((geometry.downsample_width, geometry.downsample_height), resample=resample)
    image = image.resize((geometry.resized_width, geometry.resized_height), resample=resample)
    left, top = geometry.crop_left, geometry.crop_top
    return image.crop((left, top, left + geometry.crop_width, top + geometry.crop_height))


def mask_retention(mask: Image.Image, geometry: SharedGeometry) -> dict[str, float | int | bool]:
    """Measure foreground surviving crop after the same resize transform.

    The denominator is foreground after downsample/resize but before crop, so
    this is a crop-retention metric rather than an apparent-area scale metric.
    """
    raw = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    resized = mask.convert("L")
    if resized.size != (geometry.downsample_width, geometry.downsample_height):
        resized = resized.resize((geometry.downsample_width, geometry.downsample_height), Image.Resampling.NEAREST)
    resized = resized.resize((geometry.resized_width, geometry.resized_height), Image.Resampling.NEAREST)
    pre_crop = np.asarray(resized, dtype=np.uint8) >= 128
    processed = apply_shared_geometry(mask.convert("L"), geometry, is_mask=True)
    post_crop = np.asarray(processed, dtype=np.uint8) >= 128
    raw_pixels = int(raw.sum())
    pre_crop_pixels = int(pre_crop.sum())
    processed_pixels = int(post_crop.sum())
    return {
        "raw_mask_pixels": raw_pixels,
        "raw_mask_fraction": float(raw.mean()),
        "processed_mask_pixels": processed_pixels,
        "processed_mask_fraction": float(post_crop.mean()),
        "resized_pre_crop_mask_pixels": pre_crop_pixels,
        "mask_retention_ratio": float(processed_pixels / pre_crop_pixels) if pre_crop_pixels else 0.0,
        "post_geometry_empty_mask": not bool(processed_pixels),
    }


def summarize_mask_retentions(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 0.0,
                "retention_lt_0_25_count": 0, "retention_lt_0_50_count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "min": float(array.min()), "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)), "p50": float(np.quantile(array, 0.50)),
        "retention_lt_0_25_count": int((array < 0.25).sum()),
        "retention_lt_0_50_count": int((array < 0.50).sum()),
    }


def split_by_session(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    sessions = sorted({str(row["session_id"]) for row in rows})
    random.Random(seed).shuffle(sessions)
    val_session_count = max(1, round(len(sessions) * val_fraction))
    val_sessions = set(sessions[:val_session_count])
    train = [row for row in rows if str(row["session_id"]) not in val_sessions]
    val = [row for row in rows if str(row["session_id"]) in val_sessions]
    assert {row["session_id"] for row in train}.isdisjoint({row["session_id"] for row in val})
    return train, val


def enrich_geometry(rows: list[dict], seed: int, target_size: int = DEFAULT_TARGET_SIZE) -> list[dict]:
    enriched = []
    for row in rows:
        source = Image.open(row["source"])
        target = Image.open(row["target"])
        mask = Image.open(row["mask_edit"])
        aspect_ratios = [image.width / image.height for image in (source, target, mask)]
        if max(aspect_ratios) - min(aspect_ratios) > 1e-2:
            raise ValueError(
                f"unaligned raw aspect ratios for {row['sample_key']}: "
                f"source={source.size}, target={target.size}, mask={mask.size}"
            )
        geometry_seed = seed + int(row["index"])
        geometry = sample_shared_geometry(source.size, geometry_seed, target_size)
        output = dict(row)
        output["geometry"] = geometry.as_dict()
        output["geometry_seed"] = geometry_seed
        output["mask_retention"] = mask_retention(mask, geometry)
        enriched.append(output)
    return enriched


def expected_token_grid(geometry: SharedGeometry, vq_stride: int = DEFAULT_VQ_STRIDE) -> tuple[int, int]:
    if geometry.crop_height % vq_stride or geometry.crop_width % vq_stride:
        raise ValueError("crop dimensions must be divisible by the VQ stride")
    return geometry.crop_height // vq_stride, geometry.crop_width // vq_stride
