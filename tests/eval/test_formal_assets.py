from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.formal_assets import audit_test_assets, audit_training_manifest, metric_model_identity


def _manifest(path: Path, rows: list[dict]) -> Path:
    for row in rows:
        token = path.parent / row["token_file"]
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_bytes(b"token")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_training_composition_gate_and_token_files(tmp_path):
    rows = [
        {"dataset_name": "magicbrush", "sample_key": "m-0", "token_file": "tokens/m.pt"},
        {"dataset_name": "refedit", "sample_key": "r-0", "token_file": "tokens/r.pt"},
    ]
    result = audit_training_manifest(_manifest(tmp_path / "manifest.jsonl", rows), expected_counts={"magicbrush": 1, "refedit": 1})
    assert result["dataset_counts"] == {"magicbrush": 1, "refedit": 1}
    assert result["duplicate_sample_key_count"] == 0


def test_training_composition_gate_rejects_wrong_counts_duplicates_and_missing_tokens(tmp_path):
    rows = [
        {"dataset_name": "magicbrush", "sample_key": "same", "token_file": "tokens/m.pt"},
        {"dataset_name": "magicbrush", "sample_key": "same", "token_file": "tokens/missing.pt"},
    ]
    manifest = _manifest(tmp_path / "manifest.jsonl", rows[:1])
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="composition gate failed"):
        audit_training_manifest(manifest, expected_counts={"magicbrush": 1, "refedit": 1})


def test_dino_clip_identity_records_local_config_revision_and_weight_hash(tmp_path):
    model = tmp_path / "metric-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"_commit_hash": "frozen-revision"}), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    identity = metric_model_identity(model, model_id="facebook/dinov2-base")
    assert identity["model_id"] == "facebook/dinov2-base"
    assert identity["revision"] == "frozen-revision"
    assert identity["config"]["sha256"]
    assert identity["weights"]["single_weight"]["sha256"]


def test_test_asset_audit_requires_test_metadata_unique_rows_and_existing_tokens(tmp_path):
    canonical = tmp_path / "canonical" / "manifest.jsonl"
    canonical.parent.mkdir()
    canonical.write_text(json.dumps({"sample_key": "test-0"}) + "\n", encoding="utf-8")
    (canonical.parent / "dataset_meta.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    token = tmp_path / "tokens" / "0.pt"
    token.parent.mkdir()
    token.write_bytes(b"token")
    tokens = token.parent / "manifest.jsonl"
    tokens.write_text(json.dumps({"sample_key": "test-0", "token_file": "0.pt"}) + "\n", encoding="utf-8")
    subset = tmp_path / "eval_subset.jsonl"
    subset.write_text(json.dumps({"sample_key": "test-0", "token_file": "tokens/0.pt"}) + "\n", encoding="utf-8")
    audit = audit_test_assets(canonical, tokens, subset)
    assert audit["split"] == "test"
    assert audit["token_manifest"]["sample_count"] == 1
