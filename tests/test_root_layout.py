"""Regression checks for the canonical repository layout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_root_implementation_paths_are_absent():
    legacy_paths = (
        "config.py",
        "data",
        "datasets",
        "examples",
        "generators",
        "ht_grpo",
        "inference",
        "model",
        "pre_tokenizer",
        "train",
        "xllmx",
        "requirements.txt",
        "ENVIRONMENT.md",
        "Technical-Report.pdf",
        "VLMEvalKit",
    )
    assert not [path for path in legacy_paths if (ROOT / path).exists()]


def test_canonical_layout_paths_exist():
    expected_paths = (
        "src/dataset",
        "src/generators",
        "src/models",
        "src/training",
        "src/utils",
        "src/xllmx",
        "scripts/data",
        "scripts/inference",
        "scripts/train/upstream",
        "third_party/VLMEvalKit",
    )
    assert all((ROOT / path).exists() for path in expected_paths)
