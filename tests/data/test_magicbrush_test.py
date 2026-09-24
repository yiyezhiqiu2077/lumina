from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from dataset.magicbrush_test import prepare_magicbrush_test
from dataset.utils import read_jsonl


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "white").save(path)


def _record() -> dict[str, object]:
    return {
        "img_id": "42",
        "turn_index": 2,
        "session_id": "42",
        "source": "images/source.png",
        "target": "images/target.png",
        "mask_edit": "masks/edit.png",
        "instruction": "make it blue",
    }


def test_official_test_records_are_canonicalized_with_portable_paths(tmp_path):
    root = tmp_path / "MagicBrush-test"
    for relative in ("images/source.png", "images/target.png", "masks/edit.png"):
        _image(root / relative)
    (root / "test.json").write_text(json.dumps([_record()]), encoding="utf-8")
    output = tmp_path / "canonical"
    metadata = prepare_magicbrush_test(root, output)
    rows = read_jsonl(output / "manifest.jsonl")
    assert rows == [{
        "index": 0, "sample_key": "magicbrush-test/42_turn2", "img_id": "42", "turn_index": 2, "session_id": "42",
        "source": "../MagicBrush-test/images/source.png", "target": "../MagicBrush-test/images/target.png",
        "mask_edit": "../MagicBrush-test/masks/edit.png", "instruction": "make it blue",
    }]
    assert metadata["split"] == "test"
    assert metadata["sample_count"] == 1
    assert metadata["duplicate_sample_key_count"] == 0


@pytest.mark.parametrize("field", ("source", "target", "mask_edit"))
def test_missing_required_test_assets_fail_clearly(tmp_path, field):
    root = tmp_path / "MagicBrush-test"
    for relative in ("images/source.png", "images/target.png", "masks/edit.png"):
        if relative != _record()[field]:
            _image(root / relative)
    (root / "test.json").write_text(json.dumps([_record()]), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid official MagicBrush test archive"):
        prepare_magicbrush_test(root, tmp_path / "canonical")


def test_duplicate_canonical_test_sample_keys_fail(tmp_path):
    root = tmp_path / "MagicBrush-test"
    for relative in ("images/source.png", "images/target.png", "masks/edit.png"):
        _image(root / relative)
    (root / "test.json").write_text(json.dumps([_record(), _record()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_sample_key=1"):
        prepare_magicbrush_test(root, tmp_path / "canonical")
