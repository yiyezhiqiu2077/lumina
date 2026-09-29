from io import BytesIO
from pathlib import Path

from PIL import Image

from dataset.scaleedit import record_from_row


def _png(mode="RGB", size=(8, 6), fill=255):
    image = Image.new(mode, size, fill)
    output = BytesIO(); image.save(output, format="PNG")
    return output.getvalue()


def _accepted():
    return {
        "sample_id": "sample-7", "source_image": _png(), "edited_image": _png(),
        "final_instruction": "add a bird", "final_task": "object_addition",
        "mask__mask_png": _png("L", fill=255),
        "quality__verdict": "PASS", "quality__keep": True,
        "scene__verdict": "PASS", "scene__keep": True,
        "grounding__qc_flag": "OK", "mask__qc_flag": "OK",
    }


def test_scaleedit_accepts_embedded_source_and_stable_sample_id():
    record, reason = record_from_row(_accepted(), parquet_path=Path("x.parquet"), parquet_row_index=0)
    assert reason is None
    assert record["sample_key"] == "scaleedit/sample-7"
    assert record["target"].size == (8, 6)


def test_scaleedit_strict_gates_and_url_only_source_are_explicit():
    cases = (("quality__keep", False, "quality_failed"), ("scene__keep", False, "scene_failed"),
             ("grounding__qc_flag", "BAD", "grounding_failed"), ("mask__qc_flag", "BAD", "mask_qc_failed"))
    for key, value, expected in cases:
        assert record_from_row({**_accepted(), key: value}, parquet_path=Path("x"), parquet_row_index=0)[1] == expected
    assert record_from_row({**_accepted(), "source_image": None, "source_image_url": "https://example.test/a.jpg"}, parquet_path=Path("x"), parquet_row_index=0)[1] == "source_url_fallback_not_enabled"


def test_scaleedit_rejects_empty_mask_and_size_mismatch():
    assert record_from_row({**_accepted(), "mask__mask_png": _png("L", fill=0)}, parquet_path=Path("x"), parquet_row_index=0)[1] == "empty_mask"
    assert record_from_row({**_accepted(), "edited_image": _png(size=(9, 6))}, parquet_path=Path("x"), parquet_row_index=0)[1] == "size_mismatch"
