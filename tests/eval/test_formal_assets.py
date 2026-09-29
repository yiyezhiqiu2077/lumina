from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dataset.geometry import geometry_policy
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
    result = audit_training_manifest(_manifest(tmp_path / "manifest.jsonl", rows), expected_counts={"magicbrush": 1, "refedit": 1}, require_token_files=False)
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


def test_mixed4_composition_uses_dynamic_new_dataset_counts_and_token_metadata(tmp_path):
    rows = []
    component = {}
    for name, count in {"magicbrush": 8807, "refedit": 7804, "crispedit": 2, "scaleedit": 3}.items():
        token = tmp_path / name / "files" / "token.pt"
        token.parent.mkdir(parents=True)
        torch.save({"source_codes": torch.zeros(26, 38, dtype=torch.int32), "target_codes": torch.zeros(26, 38, dtype=torch.int32), "edit_mask": torch.ones(26, 38, dtype=torch.bool), "token_height": 26, "token_width": 38, "processed_width": 608, "processed_height": 416, "vq_stride": 16}, token)
        component_meta = tmp_path / name / "dataset_meta.json"
        component_meta.write_text(json.dumps({"usable_row_count": count, "geometry_policy": geometry_policy(), "observed_token_grids": {"26x38": count}, "repo_id": "pinned" if name in {"crispedit", "scaleedit"} else None, "revision": "revision" if name in {"crispedit", "scaleedit"} else None}), encoding="utf-8")
        component[name] = {"tokenization_metadata": str(component_meta), "tokenization_metadata_sha256": __import__("hashlib").sha256(component_meta.read_bytes()).hexdigest()}
        rows.extend({"dataset_name": name, "sample_key": f"{name}/{index}", "token_file": str(token)} for index in range(count))
    manifest = tmp_path / "mixed" / "train" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = tmp_path / "mixed" / "dataset_meta.json"
    metadata.write_text(json.dumps({"dataset_counts": {"magicbrush": 8807, "refedit": 7804, "crispedit": 2, "scaleedit": 3}, "component_manifests": component, "duplicate_sample_key_count": 0, "geometry_policy": geometry_policy(), "observed_token_grids": {name: {"26x38": count} for name, count in {"magicbrush": 8807, "refedit": 7804, "crispedit": 2, "scaleedit": 3}.items()}, "manifest": str(manifest)}), encoding="utf-8")
    result = audit_training_manifest(manifest, mixed_metadata=metadata)
    assert result["dataset_counts"]["scaleedit"] == 3


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


def test_test_asset_audit_rejects_unverified_official_test_count(tmp_path):
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
    with pytest.raises(ValueError, match="REAL TEST ARCHIVE NOT VERIFIED"):
        audit_test_assets(canonical, tokens, subset)
