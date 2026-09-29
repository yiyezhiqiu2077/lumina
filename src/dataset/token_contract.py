"""One variable-grid token payload contract for all editing datasets."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import torch

from dataset.geometry import DEFAULT_VQ_STRIDE, SharedGeometry, expected_token_grid, valid_crop_sizes


REQUIRED_TOKEN_KEYS = frozenset({"source_codes", "target_codes", "edit_mask", "token_height", "token_width"})


def grid_key(height: int, width: int) -> str:
    return f"{height}x{width}"


def validate_token_payload(
    payload: dict[str, Any], *, sample_key: str | None = None, dataset_name: str | None = None,
    require_processed_geometry: bool = False, vq_stride: int = DEFAULT_VQ_STRIDE,
) -> tuple[int, int]:
    missing = REQUIRED_TOKEN_KEYS.difference(payload)
    if missing:
        raise ValueError(f"invalid token payload: missing keys {sorted(missing)}")
    source, target, mask = payload["source_codes"], payload["target_codes"], payload["edit_mask"]
    if not all(isinstance(value, torch.Tensor) and value.ndim == 2 for value in (source, target, mask)):
        raise ValueError("invalid token payload: source_codes, target_codes and edit_mask must be 2D tensors")
    shape = tuple(source.shape)
    if shape != tuple(target.shape) or shape != tuple(mask.shape):
        raise ValueError(f"invalid token payload: source/target/mask shape mismatch {shape}/{tuple(target.shape)}/{tuple(mask.shape)}")
    height, width = shape
    if height <= 0 or width <= 0 or payload["token_height"] != height or payload["token_width"] != width:
        raise ValueError("invalid token payload: token_height/token_width do not match tensors")
    if not bool(mask.bool().any()):
        raise ValueError("invalid token payload: empty edit mask")
    payload_stride = payload.get("vq_stride", vq_stride)
    if not isinstance(payload_stride, int) or payload_stride <= 0:
        raise ValueError("invalid token payload: vq_stride must be positive")
    processed_width, processed_height = payload.get("processed_width"), payload.get("processed_height")
    if require_processed_geometry and (not isinstance(processed_width, int) or not isinstance(processed_height, int)):
        raise ValueError("invalid token payload: processed_width/processed_height are required")
    if processed_width is not None or processed_height is not None:
        if (not isinstance(processed_width, int) or not isinstance(processed_height, int)
                or processed_width != width * payload_stride or processed_height != height * payload_stride):
            raise ValueError("invalid token payload: processed dimensions do not match VQ grid/stride")
        if (processed_width, processed_height) not in set(valid_crop_sizes()):
            raise ValueError("invalid token payload: processed dimensions are not a legal Lumina bucket")
    geometry = payload.get("geometry")
    if geometry is not None:
        try:
            expected = expected_token_grid(SharedGeometry(**geometry), payload_stride)
        except Exception as error:
            raise ValueError(f"invalid token payload geometry: {error}") from error
        if expected != shape:
            raise ValueError("invalid token payload: geometry and VQ grid mismatch")
    if sample_key is not None and payload.get("sample_key") not in (None, sample_key):
        raise ValueError("invalid existing token: sample identity mismatch")
    if dataset_name is not None and payload.get("dataset_name") not in (None, dataset_name):
        raise ValueError("invalid existing token: dataset identity mismatch")
    return height, width


def atomic_save_token(payload: dict[str, Any], destination: Path, *, sample_key: str, dataset_name: str) -> None:
    """Write, validate, then atomically publish a token payload."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        written = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_token_payload(written, sample_key=sample_key, dataset_name=dataset_name, require_processed_geometry=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_reusable_token(
    destination: Path, *, sample_key: str, dataset_name: str, geometry_seed: int, geometry: dict,
) -> tuple[int, int]:
    """Return a verified existing grid or fail closed on stale/corrupt files."""
    try:
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        shape = validate_token_payload(
            payload, sample_key=sample_key, dataset_name=dataset_name,
            require_processed_geometry=True,
        )
    except Exception as error:
        raise ValueError(f"invalid existing token {destination}: {error}") from error
    if payload.get("geometry_seed") != geometry_seed or payload.get("geometry") != geometry:
        raise ValueError(f"invalid existing token {destination}: geometry identity mismatch")
    return shape
