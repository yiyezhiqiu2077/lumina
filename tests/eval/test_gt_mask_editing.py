import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.gt_mask_editing import (
    boundary_ring,
    load_lora_recipe,
    pixel_metrics,
    prepare_eval_subset,
    read_eval_subset,
    resolve_token_file,
    token_accuracies,
)


def _payload(mask: torch.Tensor) -> dict:
    return {
        "source_codes": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "target_codes": torch.tensor([[5, 2], [3, 6]], dtype=torch.int32),
        "edit_mask": mask.bool(),
        "token_height": 2,
        "token_width": 2,
        "processed_height": 4,
        "processed_width": 4,
    }


def test_base_mode_requires_no_checkpoint_recipe():
    assert load_lora_recipe(None) is None


def test_lora_recipe_is_loaded_from_checkpoint_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint-000005"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps({"fingerprint": {"payload": {"lora": {
            "rank": 8, "alpha": 12.0, "dropout": 0.2, "targets": ["q_proj", "k_proj"],
        }}}}),
        encoding="utf-8",
    )
    assert load_lora_recipe(checkpoint) == {
        "rank": 8, "alpha": 12.0, "dropout": 0.2, "targets": ("q_proj", "k_proj"),
    }


def test_token_file_resolution_accepts_absolute_and_relative_paths(tmp_path):
    absolute = tmp_path / "files" / "one.pt"
    assert resolve_token_file({"token_file": str(absolute)}, tmp_path / "manifest.jsonl") == absolute
    assert resolve_token_file({"token_file": "files/one.pt"}, tmp_path / "manifest.jsonl") == absolute


def test_token_accuracy_and_hard_lock_assertion():
    source = torch.tensor([[1, 2], [3, 4]])
    target = torch.tensor([[5, 2], [3, 6]])
    mask = torch.tensor([[True, False], [False, True]])
    generated = torch.tensor([[126361, 126358, 126359, 126362]])
    metrics = token_accuracies(generated, source, target, mask)
    assert metrics == {"edit_token_accuracy": 1.0, "outside_token_accuracy": 1.0}
    with pytest.raises(AssertionError, match="hard-lock violation"):
        token_accuracies(torch.tensor([[126361, 126399, 126359, 126362]]), source, target, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only evaluator boundary")
def test_token_accuracy_accepts_cuda_generated_tokens_and_cpu_payload():
    source = torch.tensor([[1, 2], [3, 4]])
    target = torch.tensor([[5, 2], [3, 6]])
    mask = torch.tensor([[True, False], [False, True]])
    generated = torch.tensor([[126361, 126358, 126359, 126362]], device="cuda")
    assert token_accuracies(generated, source, target, mask)["outside_token_accuracy"] == 1.0


def test_masked_pixel_metrics_and_empty_full_masks_do_not_crash():
    prediction = np.zeros((4, 4, 3), dtype=np.float32)
    target = np.ones_like(prediction)
    source = np.zeros_like(prediction)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    metrics = pixel_metrics(prediction, target, target, source, mask, boundary_radius=1)
    assert metrics["inside_l1_target"] == pytest.approx(1.0)
    assert metrics["inside_mse_target"] == pytest.approx(1.0)
    assert metrics["inside_psnr_target"] == pytest.approx(0.0)
    assert metrics["full_l1_target"] == pytest.approx(1.0)
    empty = pixel_metrics(prediction, target, target, source, np.zeros((4, 4), dtype=bool), boundary_radius=1)
    assert empty["inside_l1_target"] is None
    full = pixel_metrics(prediction, target, target, source, np.ones((4, 4), dtype=bool), boundary_radius=1)
    assert full["boundary_l1_source_recon"] is None


def test_boundary_ring_has_expected_semantics():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    ring = boundary_ring(mask, radius=1)
    assert ring.sum() == 8
    assert not ring[3, 3]
    assert not boundary_ring(mask, radius=0).any()


def test_subset_is_fixed_and_preserves_identical_keys_and_seeds(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    rows = []
    for index in range(4):
        token = files / f"{index:06d}.pt"
        torch.save(_payload(torch.tensor([[index != 0, False], [False, False]])), token)
        rows.append({"sample_key": f"sample-{index}", "token_file": f"files/{token.name}"})
    manifest = tmp_path / "heldout.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    subset = tmp_path / "eval_subset.jsonl"
    first = prepare_eval_subset(manifest, subset, seed=42, limit=3)
    second = prepare_eval_subset(manifest, subset, seed=42, limit=3)
    assert [row["sample_key"] for row in first] == [row["sample_key"] for row in second]
    assert [row["inference_seed"] for row in first] == [42, 43, 44]
    assert read_eval_subset(subset) == second


def test_subset_preparation_only_writes_the_requested_subset(tmp_path):
    token = tmp_path / "token.pt"
    torch.save(_payload(torch.ones(2, 2, dtype=torch.bool)), token)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_key": "one", "token_file": str(token)}) + "\n", encoding="utf-8")
    before = token.stat().st_mtime_ns
    prepare_eval_subset(manifest, tmp_path / "subset.jsonl", seed=42, limit=1)
    assert token.stat().st_mtime_ns == before
