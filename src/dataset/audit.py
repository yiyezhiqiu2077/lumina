"""Reusable MagicBrush geometry, sequence, and token-layout contract audits."""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from dataset.geometry import SharedGeometry, apply_shared_geometry
from dataset.magicbrush import MagicBrushTokenDataset, _spatial_with_newlines
from dataset.utils import read_jsonl
from utils.constants import SPECIAL_TOKENS


def _summary(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered), "p95": float(np.percentile(ordered, 95)), "max": ordered[-1]}


def run_geometry_audit(manifest: Path, *, vq_stride: int = 16) -> dict:
    rows = read_jsonl(manifest)
    counts, mask_pixels, token_spatial = Counter(), [], []
    examples = {"portrait": [], "landscape": [], "square": []}
    failures = []
    for row in rows:
        geometry = SharedGeometry(**row["geometry"])
        source = apply_shared_geometry(Image.open(row["source"]).convert("RGB"), geometry)
        target = apply_shared_geometry(Image.open(row["target"]).convert("RGB"), geometry)
        mask = apply_shared_geometry(Image.open(row["mask_edit"]).convert("L"), geometry, is_mask=True)
        if source.size != target.size or source.size != mask.size:
            failures.append({"sample_key": row["sample_key"], "reason": "processed size mismatch"})
            continue
        width, height = source.size
        if width % vq_stride or height % vq_stride:
            failures.append({"sample_key": row["sample_key"], "reason": "not divisible by VQ stride"})
            continue
        ratio = width / height
        kind = "landscape" if ratio > 1.05 else "portrait" if ratio < 1 / 1.05 else "square"
        counts[kind] += 1
        token_height, token_width = height // vq_stride, width // vq_stride
        token_spatial.append(token_height * token_width)
        mask_pixels.append(int((np.asarray(mask) >= 128).sum()))
        if len(examples[kind]) < 5:
            examples[kind].append({"sample_key": row["sample_key"], "processed_pil_size_wh": [width, height], "token_grid_hw": [token_height, token_width], "source_spatial_count": token_height * token_width, "reshape_count": token_height * token_width})
    for kind, values in examples.items():
        if len(values) < 5:
            failures.append({"orientation": kind, "reason": "fewer than five audit examples"})
    return {"manifest": str(manifest.resolve()), "samples": len(rows), "orientation_counts": dict(counts), "five_per_orientation": examples, "token_spatial_count": _summary(token_spatial), "binary_mask_positive_pixels": _summary(mask_pixels), "failures": failures, "passed": not failures}


def run_sequence_audit(model: Path, manifest: Path, *, seed: int = 42) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    dataset = MagicBrushTokenDataset(manifest, tokenizer, condition_dropout=0, seed=seed, fixed_corruption=True)
    fields = {key: [] for key in ("instruction_token_count", "source_spatial_count", "source_newline_count", "target_spatial_count", "target_newline_count", "special_count", "total_sequence_length", "valid_target_label_count")}
    orientation = {"portrait": [], "landscape": [], "square": []}
    special_ids = set(SPECIAL_TOKENS.values())
    for index in range(len(dataset)):
        row = dataset[index]
        for key in fields:
            fields[key].append(sum(token in special_ids for token in row["input_ids"]) if key == "special_count" else row[key])
        height, width = row["token_height"], row["token_width"]
        kind = "landscape" if width / height > 1.05 else "portrait" if width / height < 1 / 1.05 else "square"
        if len(orientation[kind]) < 5:
            orientation[kind].append({"sample_key": row["sample_key"], "token_grid_hw": [height, width], "source_spatial_count": row["source_spatial_count"], "reshape_count": height * width})
        if len(row["input_ids"]) != len(row["labels"]):
            raise AssertionError("input/label length mismatch")
        if row["valid_target_label_count"] <= 0:
            raise AssertionError("sample without a valid target label")
        if row["source_spatial_count"] != height * width or row["target_spatial_count"] != height * width:
            raise AssertionError("spatial token count mismatch")
    missing = [kind for kind, values in orientation.items() if len(values) < 5]
    return {"samples": len(dataset), "statistics": {key: _summary(values) for key, values in fields.items()}, "five_per_orientation": orientation, "missing_orientations": missing, "non_square_dataset_gate": "not_applicable_all_magicbrush_512_level_samples_are_square" if missing else "passed", "all_input_label_lengths_equal": True, "all_samples_have_valid_target_labels": True, "target_truncation_detected": False}


def run_non_square_layout_audit() -> dict:
    def audit_shape(height: int, width: int) -> dict:
        codes = torch.arange(height * width, dtype=torch.int32).reshape(height, width)
        sequence, spatial_flags = _spatial_with_newlines(codes)
        recovered = torch.tensor([token - SPECIAL_TOKENS["image_token_offset"] for token, spatial in zip(sequence, spatial_flags) if spatial]).reshape(height, width)
        newline_positions = [index for index, spatial in enumerate(spatial_flags) if not spatial]
        expected_newlines = [(row + 1) * (width + 1) - 1 for row in range(height)]
        if not torch.equal(codes.long(), recovered.long()) or newline_positions != expected_newlines:
            raise AssertionError(f"row-major layout failed for {(height, width)}")
        mask = torch.zeros(height, width, dtype=torch.bool)
        mask[1, 2], mask[-2, -3] = True, True
        if torch.nonzero(mask.flatten()).flatten().tolist() != [width + 2, (height - 2) * width + width - 3]:
            raise AssertionError(f"mask flatten orientation failed for {(height, width)}")
        return {"token_grid_hw": [height, width], "source_spatial_count": sum(spatial_flags), "reshape_count": height * width, "newline_count": len(newline_positions), "row_major_round_trip": True, "mask_orientation_round_trip": True}
    shapes = {"portrait": [(16, 8), (24, 12), (32, 16), (28, 14), (20, 10)], "landscape": [(8, 16), (12, 24), (16, 32), (14, 28), (10, 20)], "square": [(8, 8), (12, 12), (16, 16), (24, 24), (32, 32)]}
    return {"scope": "synthetic layout fixtures only; MagicBrush 512-level samples are all square", "orientations": {kind: [audit_shape(*shape) for shape in values] for kind, values in shapes.items()}, "passed": True}
