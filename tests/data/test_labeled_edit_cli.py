import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", ("preprocess_crispedit.py", "preprocess_scaleedit.py"))
def test_labeled_edit_tokenize_max_samples_defaults_to_full(monkeypatch, script):
    module = runpy.run_path(str(ROOT / "scripts/data" / script))
    monkeypatch.setattr(sys, "argv", [script, "tokenize", "--raw-root", "/raw", "--output", "/out", "--model", "/model"])
    assert module["parse_args"]().max_samples == 0
    monkeypatch.setattr(sys, "argv", [script, "tokenize", "--raw-root", "/raw", "--output", "/out", "--model", "/model", "--max-samples", "10"])
    assert module["parse_args"]().max_samples == 10
