from io import BytesIO
from pathlib import Path

from PIL import Image

from dataset.crispedit import record_from_row


def _png(mode="RGB", size=(8, 6), fill=255):
    image = Image.new(mode, size, fill)
    output = BytesIO(); image.save(output, format="PNG")
    return output.getvalue()


def _accepted():
    return {
        "source_shard": "add_00000.parquet", "row_idx": 7,
        "input_img": {"bytes": _png()}, "output_img": {"bytes": _png()},
        "instruction": "make it blue", "type": "color",
        "mask__mask_png": _png("L", fill=255),
        "quality__prefilter_verdict": "PASS", "quality__filter_decision": "keep",
        "scene__scene_decision": "PASS", "scene__scene_pass": True,
        "grounding__grounding_status": "OK", "grounding__qc_flag": "OK",
        "mask__qc_flag": "OK", "mask__mask_selection_reason": "SELECTED",
    }


def test_crispedit_accepts_strict_pass_row_and_has_stable_key():
    record, reason = record_from_row(_accepted(), parquet_path=Path("x.parquet"), parquet_row_index=0)
    assert reason is None
    assert record["sample_key"] == "crispedit/add_00000.parquet:7"
    assert record["source"].size == (8, 6)


def test_crispedit_quality_scene_grounding_and_mask_gates():
    cases = (("quality__prefilter_verdict", "FAIL", "quality_failed"),
             ("quality__filter_decision", "drop", "filter_rejected"),
             ("scene__scene_pass", False, "scene_failed"),
             ("grounding__qc_flag", "BAD", "grounding_failed"),
             ("mask__qc_flag", "BAD", "mask_qc_failed"),
             ("mask__mask_selection_reason", "REJECTED", "mask_not_selected"))
    for key, value, expected in cases:
        record, reason = record_from_row({**_accepted(), key: value}, parquet_path=Path("x.parquet"), parquet_row_index=0)
        assert record is None and reason == expected


def test_crispedit_rejects_empty_mask_and_size_mismatch():
    empty = _png("L", fill=0)
    assert record_from_row({**_accepted(), "mask__mask_png": empty}, parquet_path=Path("x"), parquet_row_index=0)[1] == "empty_mask"
    assert record_from_row({**_accepted(), "output_img": {"bytes": _png(size=(9, 6))}}, parquet_path=Path("x"), parquet_row_index=0)[1] == "size_mismatch"
