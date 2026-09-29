"""Strict parser for the pinned CrispEdit labeling snapshot."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from dataset.labeled_edit import decode_image, decode_mask, parquet_rows


DATASET_NAME = "crispedit"


def crispedit_shards(raw_root: Path) -> list[Path]:
    paths = sorted((Path(raw_root) / "shards").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"CrispEdit parquet shards not found below {raw_root}")
    return paths


def reject_reason(row: dict) -> str | None:
    checks = (
        (row.get("quality__prefilter_verdict") == "PASS", "quality_failed"),
        (row.get("quality__filter_decision") == "keep", "filter_rejected"),
        (row.get("scene__scene_decision") == "PASS" and row.get("scene__scene_pass") is True, "scene_failed"),
        (row.get("grounding__grounding_status") == "OK" and row.get("grounding__qc_flag") == "OK", "grounding_failed"),
        (row.get("mask__qc_flag") == "OK", "mask_qc_failed"),
        (row.get("mask__mask_selection_reason") == "SELECTED", "mask_not_selected"),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
        return "missing_instruction"
    return None


def record_from_row(row: dict, *, parquet_path: Path, parquet_row_index: int) -> tuple[dict | None, str | None]:
    reason = reject_reason(row)
    if reason:
        return None, reason
    try:
        source = decode_image(row.get("input_img"), mode="RGB")
    except (ValueError, OSError):
        return None, "source_decode_failed"
    try:
        target = decode_image(row.get("output_img"), mode="RGB")
    except (ValueError, OSError):
        return None, "target_decode_failed"
    try:
        mask = decode_mask(row.get("mask__mask_png"))
    except (ValueError, OSError):
        return None, "mask_decode_failed"
    if mask.getbbox() is None:
        return None, "empty_mask"
    if source.size != target.size or source.size != mask.size:
        return None, "size_mismatch"
    source_shard, row_idx = str(row.get("source_shard", parquet_path.name)), int(row.get("row_idx", parquet_row_index))
    return {
        "dataset_name": DATASET_NAME, "sample_key": f"crispedit/{source_shard}:{row_idx}",
        "instruction": row["instruction"].strip(), "task": str(row.get("type") or ""),
        "source": source, "target": target, "mask": mask,
        "sort_key": (source_shard, row_idx), "source_shard": source_shard, "row_idx": row_idx,
    }, None


def iter_crispedit_records(raw_root: Path, paths: Iterable[Path] | None = None, *, on_skip: Callable[[str], None] | None = None) -> Iterator[dict]:
    for path, position, row in parquet_rows(paths if paths is not None else crispedit_shards(raw_root)):
        record, reason = record_from_row(row, parquet_path=path, parquet_row_index=position)
        if reason is not None:
            if on_skip: on_skip(reason)
            continue
        yield record


def audit_crispedit(raw_root: Path) -> dict:
    raw_count, accepted = 0, 0
    skipped: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    areas: list[float] = []
    for path, position, row in parquet_rows(crispedit_shards(raw_root)):
        raw_count += 1
        record, reason = record_from_row(row, parquet_path=path, parquet_row_index=position)
        if reason:
            skipped[reason] += 1
            continue
        accepted += 1
        tasks[record["task"]] += 1
        areas.append(float((__import__("numpy").asarray(record["mask"], dtype="uint8") >= 128).mean()))
    provenance_path = Path(raw_root) / "download_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    return {
        "dataset_name": DATASET_NAME, "raw_row_count": raw_count, "usable_row_count": accepted,
        "skipped_count": raw_count - accepted, "skipped_reasons": dict(sorted(skipped.items())),
        "task_counts": dict(sorted(tasks.items())), "embedded_source_count": accepted,
        "mask_area_fraction": {"mean": sum(areas) / len(areas) if areas else 0.0, "min": min(areas) if areas else 0.0, "max": max(areas) if areas else 0.0},
        "parser_version": 1,
        "repo_id": provenance.get("repo_id"), "revision": provenance.get("resolved_revision"),
    }
