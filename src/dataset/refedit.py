"""Strict RefEdit parquet parsing shared by audit and token preprocessing.

The upstream dataset stores source/target image pairs in ``data/*.parquet``
and final instructions/masks in ``final/data/*.parquet``.  ``row_idx`` is
shard-local, so rows are joined by ``(source_relative_path, row_idx)`` and
then verified with ``raw.img_id == audit.img_id``.  This module deliberately
does not guess a mapping.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image


DATASET_REPO = "TTangenty/refedit-final-mask"
DATASET_REVISION = "45028821c8d787b3532070d787e72f56d16796a0"


def _pq():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:  # pragma: no cover - environment error
        raise RuntimeError(
            "RefEdit preprocessing requires pyarrow; install the repository data extra."
        ) from error
    return pq


def decode_image(value, *, mode: str) -> Image.Image:
    """Decode HuggingFace Image parquet structs without relying on path fields."""
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), (bytes, bytearray)):
        raise ValueError("expected HuggingFace image struct with in-parquet bytes")
    with Image.open(BytesIO(value["bytes"])) as image:
        return image.convert(mode).copy()


def decode_mask(value: bytes) -> Image.Image:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("expected PNG mask bytes")
    with Image.open(BytesIO(value)) as image:
        return image.convert("L").copy()


def audit_row_reason(row: dict) -> str | None:
    """Return an explicit reject reason from final/audit metadata, if any."""
    if row.get("prefilter_verdict") != "PASS":
        return f"prefilter_verdict={row.get('prefilter_verdict')!r}"
    if row.get("grounding_status") != "OK":
        return f"grounding_status={row.get('grounding_status')!r}"
    if row.get("mask_qc_flag") != "OK":
        return f"mask_qc_flag={row.get('mask_qc_flag')!r}"
    if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
        return "missing_instruction"
    if float(row.get("area_frac", 0.0)) <= 0.0:
        return "empty_audit_mask"
    return None


def _read_rows(path: Path) -> list[dict]:
    return _pq().read_table(path).to_pylist()


def audit_refedit(raw_root: Path) -> dict:
    """Read the official audit table and produce stable, machine-readable metadata."""
    raw_root = Path(raw_root)
    audit_path = raw_root / "final/audit/final_manifest.parquet"
    audit_rows = _read_rows(audit_path)
    rejected = Counter()
    usable = []
    areas = []
    for row in audit_rows:
        reason = audit_row_reason(row)
        if reason is not None:
            rejected[reason] += 1
            continue
        usable.append(row)
        areas.append(float(row["area_frac"]))
    source_files = sorted((raw_root / "data").glob("*.parquet"))
    raw_count = sum(_pq().ParquetFile(path).metadata.num_rows for path in source_files)
    return {
        "dataset_repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "raw_row_count": raw_count,
        "final_audit_row_count": len(audit_rows),
        "usable_row_count": len(usable),
        "skipped_count": len(audit_rows) - len(usable),
        "skipped_reasons": dict(sorted(rejected.items())),
        "fields": {
            "source": "source_img",
            "target": "target_img",
            "raw_instruction": "instruction",
            "final_instruction": "final_instruction",
            "mask": "mask_png",
            "join": "(source_relative_path, shard_local_row_idx), verify raw.img_id == audit.img_id",
        },
        "mask_area_fraction": {
            "mean": sum(areas) / len(areas) if areas else 0.0,
            "min": min(areas) if areas else 0.0,
            "max": max(areas) if areas else 0.0,
        },
    }


def iter_refedit_records(
    raw_root: Path,
    *,
    on_skip: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Yield only fully validated RefEdit source/target/instruction/mask records."""
    raw_root = Path(raw_root)
    audit_rows = _read_rows(raw_root / "final/audit/final_manifest.parquet")
    allowed = {
        (str(row["source_relative_path"]), int(row["row_idx"])): row
        for row in audit_rows
        if audit_row_reason(row) is None
    }
    def skip(reason: str) -> None:
        if on_skip is not None:
            on_skip(reason)
    for final_path in sorted((raw_root / "final/data").glob("*.parquet")):
        raw_rows_by_path: dict[Path, list[dict]] = {}
        for final in _read_rows(final_path):
            row_idx = int(final["row_idx"])
            source_relative_path = str(final["source_relative_path"])
            audit = allowed.get((source_relative_path, row_idx))
            if audit is None:
                skip("not_accepted_by_official_audit")
                continue
            source_path = raw_root / final["source_relative_path"]
            if not source_path.is_file():
                skip("missing_source_parquet")
                continue
            if source_path not in raw_rows_by_path:
                raw_rows_by_path[source_path] = _read_rows(source_path)
            raw_rows = raw_rows_by_path[source_path]
            if row_idx < 0 or row_idx >= len(raw_rows):
                skip("row_idx_out_of_source_shard_bounds")
                continue
            raw = raw_rows[row_idx]
            if int(raw.get("img_id", -1)) != int(audit["img_id"]):
                skip("raw_img_id_audit_mismatch")
                continue
            instruction = final.get("final_instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                skip("missing_final_instruction")
                continue
            if final.get("original_instruction") != raw.get("instruction"):
                skip("raw_final_instruction_mismatch")
                continue
            if audit.get("instruction") != instruction:
                skip("audit_final_instruction_mismatch")
                continue
            if final.get("sample_id") != audit.get("sample_id"):
                skip("final_audit_sample_id_mismatch")
                continue
            try:
                source = decode_image(raw["source_img"], mode="RGB")
                target = decode_image(raw["target_img"], mode="RGB")
                mask = decode_mask(final["mask_png"])
            except (KeyError, ValueError, OSError):
                skip("image_or_mask_decode_failed")
                continue
            if source.size != target.size or source.size != mask.size:
                skip("source_target_mask_size_mismatch")
                continue
            if mask.getbbox() is None:
                skip("empty_mask")
                continue
            sample_id = str(final.get("sample_id", f"refedit:{row_idx}"))
            yield {
                "dataset_name": "refedit",
                "sample_key": f"refedit/{sample_id}",
                "instruction": instruction,
                "source": source,
                "target": target,
                "mask": mask,
                "row_idx": row_idx,
                "img_id": int(raw["img_id"]),
                "source_relative_path": final["source_relative_path"],
            }
