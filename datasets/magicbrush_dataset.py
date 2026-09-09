from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_record_paths(record: dict, manifest: Path) -> dict:
    output = dict(record)
    for key in ("source", "target", "mask_edit"):
        path = Path(output[key])
        output[key] = str(path if path.is_absolute() else manifest.parent / path)
    output["session_id"] = str(output.get("session_id", output["img_id"]))
    return output


def generate_crop_size_list(num_patches: int, patch_size: int, max_ratio: float = 4.0):
    sizes = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            sizes.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return sizes


def sample_shared_geometry(size: tuple[int, int], seed: int, target_size: int = 512) -> SharedGeometry:
    width, height = size
    candidates = generate_crop_size_list((target_size // 32) ** 2, 32)
    crop_width, crop_height = max(
        candidates,
        key=lambda candidate: min(candidate[0] / width, candidate[1] / height)
        / max(candidate[0] / width, candidate[1] / height),
    )
    down_width, down_height = width, height
    while down_width >= 2 * crop_width and down_height >= 2 * crop_height:
        down_width //= 2
        down_height //= 2
    scale = max(crop_width / down_width, crop_height / down_height)
    resized_width = round(down_width * scale)
    resized_height = round(down_height * scale)
    rng = random.Random(seed)
    crop_left = rng.randint(0, resized_width - crop_width)
    crop_top = rng.randint(0, resized_height - crop_height)
    return SharedGeometry(
        crop_width=crop_width,
        crop_height=crop_height,
        downsample_width=down_width,
        downsample_height=down_height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=crop_left,
        crop_top=crop_top,
    )


def apply_shared_geometry(image: Image.Image, geometry: SharedGeometry, is_mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    if image.size != (geometry.downsample_width, geometry.downsample_height):
        image = image.resize((geometry.downsample_width, geometry.downsample_height), resample=resample)
    image = image.resize((geometry.resized_width, geometry.resized_height), resample=resample)
    left, top = geometry.crop_left, geometry.crop_top
    return image.crop((left, top, left + geometry.crop_width, top + geometry.crop_height))


def split_by_session(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    sessions = sorted({str(row["session_id"]) for row in rows})
    random.Random(seed).shuffle(sessions)
    val_session_count = max(1, round(len(sessions) * val_fraction))
    val_sessions = set(sessions[:val_session_count])
    train = [row for row in rows if str(row["session_id"]) not in val_sessions]
    val = [row for row in rows if str(row["session_id"]) in val_sessions]
    assert {row["session_id"] for row in train}.isdisjoint({row["session_id"] for row in val})
    return train, val


def enrich_geometry(rows: list[dict], seed: int, target_size: int = 512) -> list[dict]:
    enriched = []
    for position, row in enumerate(rows):
        source = Image.open(row["source"])
        target = Image.open(row["target"])
        mask = Image.open(row["mask_edit"])
        aspect_ratios = [image.width / image.height for image in (source, target, mask)]
        if max(aspect_ratios) - min(aspect_ratios) > 1e-2:
            raise ValueError(
                f"unaligned raw aspect ratios for {row['sample_key']}: "
                f"source={source.size}, target={target.size}, mask={mask.size}"
            )
        geometry = sample_shared_geometry(source.size, seed + int(row["index"]), target_size)
        output = dict(row)
        output["geometry"] = geometry.as_dict()
        output["geometry_seed"] = seed + int(row["index"])
        enriched.append(output)
    return enriched


def expected_token_grid(geometry: SharedGeometry, vq_stride: int = 16) -> tuple[int, int]:
    if geometry.crop_height % vq_stride or geometry.crop_width % vq_stride:
        raise ValueError("crop dimensions must be divisible by the VQ stride")
    return geometry.crop_height // vq_stride, geometry.crop_width // vq_stride
