"""Strict parser for the pinned ScaleEdit labeling snapshot."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from dataset.labeled_edit import decode_image, decode_mask, parquet_rows


DATASET_NAME = "scaleedit"


def scaleedit_shards(raw_root: Path) -> list[Path]:
    paths = sorted((Path(raw_root) / "shards").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"ScaleEdit parquet shards not found below {raw_root}")
    return paths


def reject_reason(row: dict) -> str | None:
    checks = (
        (row.get("quality__verdict") == "PASS" and row.get("quality__keep") is True, "quality_failed"),
        (row.get("scene__verdict") == "PASS" and row.get("scene__keep") is True, "scene_failed"),
        (row.get("grounding__qc_flag") == "OK", "grounding_failed"),
        (row.get("mask__qc_flag") == "OK", "mask_qc_failed"),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    if not isinstance(row.get("final_instruction"), str) or not row["final_instruction"].strip():
        return "missing_instruction"
    if not row.get("source_image") and row.get("source_image_url"):
        # URL fallback is deliberately not guessed.  The audit reports this
        # population, after which a separately tested fetcher may be enabled.
        return "source_url_fallback_not_enabled"
    return None


def record_from_row(row: dict, *, parquet_path: Path, parquet_row_index: int) -> tuple[dict | None, str | None]:
    reason = reject_reason(row)
    if reason:
        return None, reason
    try:
        source = decode_image(row.get("source_image"), mode="RGB")
    except (ValueError, OSError):
        return None, "source_decode_failed"
    try:
        target = decode_image(row.get("edited_image"), mode="RGB")
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
    sample_id = str(row.get("sample_id") or f"{parquet_path.name}:{parquet_row_index}")
    return {
        "dataset_name": DATASET_NAME, "sample_key": f"scaleedit/{sample_id}",
        "instruction": row["final_instruction"].strip(), "task": str(row.get("final_task") or ""),
        "source": source, "target": target, "mask": mask,
        "sort_key": (str(row.get("source_shard", parquet_path.name)), int(row.get("row_idx", parquet_row_index)), sample_id),
        "sample_id": sample_id,
    }, None


def iter_scaleedit_records(raw_root: Path, paths: Iterable[Path] | None = None, *, on_skip: Callable[[str], None] | None = None) -> Iterator[dict]:
    for path, position, row in parquet_rows(paths if paths is not None else scaleedit_shards(raw_root)):
        record, reason = record_from_row(row, parquet_path=path, parquet_row_index=position)
        if reason is not None:
            if on_skip: on_skip(reason)
            continue
        yield record


def audit_scaleedit(raw_root: Path) -> dict:
    raw_count, accepted, embedded, url_only = 0, 0, 0, 0
    skipped: Counter[str] = Counter(); tasks: Counter[str] = Counter(); areas: list[float] = []
    for path, position, row in parquet_rows(scaleedit_shards(raw_root)):
        raw_count += 1
        if row.get("source_image"): embedded += 1
        elif row.get("source_image_url"): url_only += 1
        record, reason = record_from_row(row, parquet_path=path, parquet_row_index=position)
        if reason:
            skipped[reason] += 1
            continue
        accepted += 1; tasks[record["task"]] += 1
        areas.append(float((__import__("numpy").asarray(record["mask"], dtype="uint8") >= 128).mean()))
    provenance_path = Path(raw_root) / "download_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    return {
        "dataset_name": DATASET_NAME, "raw_row_count": raw_count, "usable_row_count": accepted,
        "skipped_count": raw_count - accepted, "skipped_reasons": dict(sorted(skipped.items())),
        "task_counts": dict(sorted(tasks.items())), "embedded_source_count": embedded, "url_source_count": url_only,
        "mask_area_fraction": {"mean": sum(areas) / len(areas) if areas else 0.0, "min": min(areas) if areas else 0.0, "max": max(areas) if areas else 0.0},
        "parser_version": 1,
        "repo_id": provenance.get("repo_id"), "revision": provenance.get("resolved_revision"),
    }
