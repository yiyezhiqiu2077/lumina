from __future__ import annotations

import math
import random

import pytest
import torch

from dataset.magicbrush import EditTokenDataset, corrupt_target_spatial, stable_sample_seed
from utils.constants import SPECIAL_TOKENS


def _legacy_masked_target(codes: torch.Tensor, rng: random.Random):
    flat = codes.flatten().long() + SPECIAL_TOKENS["image_token_offset"]
    ratio = math.cos(rng.random() * math.pi / 2)
    count = max(1, int(flat.numel() * ratio))
    selected = set(rng.sample(range(flat.numel()), count))
    tokens, labels = [], []
    width = codes.shape[1]
    for index, token in enumerate(flat.tolist()):
        tokens.append(SPECIAL_TOKENS["mask_token"] if index in selected else token)
        labels.append(token if index in selected else -100)
        if (index + 1) % width == 0:
            tokens.append(SPECIAL_TOKENS["newline_token"])
            labels.append(-100)
    return tokens, labels, count


def _payload(shape=(3, 4), mask=None):
    source = torch.arange(math.prod(shape), dtype=torch.long).reshape(shape)
    target = source + 100
    if mask is None:
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[0, 1] = True
        mask[-1, -1] = True
    return source, target, mask.bool()


def test_full_target_is_exact_legacy_regression():
    source, target, mask = _payload()
    seed = stable_sample_seed(42, 3, "magicbrush/a")
    expected = _legacy_masked_target(target, random.Random(seed))
    actual = corrupt_target_spatial(source, target, mask, random.Random(seed), "full_target")
    assert actual["tokens"] == expected[0]
    assert actual["labels"] == expected[1]
    assert actual["masked_count"] == expected[2]
    assert int(actual["spatial_labels"].ne(-100).sum()) == expected[2]
    expected_input = target.long().clone()
    expected_input[actual["selected_spatial"]] = (
        SPECIAL_TOKENS["mask_token"] - SPECIAL_TOKENS["image_token_offset"]
    )
    assert torch.equal(actual["input_spatial"], expected_input)


def test_editregion_hardlock_has_exact_source_target_mask_and_label_semantics():
    source, target, mask = _payload()
    result = corrupt_target_spatial(source, target, mask, random.Random(7), "edit_region_hardlock")
    selected = result["selected_spatial"]
    values = result["input_spatial"]
    labels = result["spatial_labels"]
    assert bool((selected & ~mask).any()) is False
    assert torch.equal(values[~mask], source[~mask])
    assert torch.equal(labels[~mask], torch.full_like(labels[~mask], -100))
    assert torch.equal(values[mask & ~selected], target[mask & ~selected])
    assert torch.equal(labels[mask & ~selected], torch.full_like(labels[mask & ~selected], -100))
    assert torch.equal(values[selected], torch.full_like(values[selected], SPECIAL_TOKENS["mask_token"] - SPECIAL_TOKENS["image_token_offset"]))
    assert torch.equal(labels[selected], target[selected] + SPECIAL_TOKENS["image_token_offset"])


def test_editregion_count_is_based_on_edit_tokens_and_is_deterministic():
    source, target, mask = _payload((4, 4))
    mask[:] = False
    mask[0, 0], mask[0, 1], mask[1, 1] = True, True, True
    one = corrupt_target_spatial(source, target, mask, random.Random(19), "edit_region_hardlock")
    two = corrupt_target_spatial(source, target, mask, random.Random(19), "edit_region_hardlock")
    assert one["candidate_count"] == 3
    assert one["masked_count"] <= 3
    assert one["masked_count"] >= 1
    assert torch.equal(one["selected_spatial"], two["selected_spatial"])


def test_tiny_and_full_masks_are_valid_and_empty_fails_fast():
    source, target, tiny = _payload((2, 3), torch.tensor([[True, False, False], [False, False, False]]))
    tiny_result = corrupt_target_spatial(source, target, tiny, random.Random(1), "edit_region_hardlock")
    assert tiny_result["masked_count"] == 1
    full = torch.ones_like(tiny)
    full_result = corrupt_target_spatial(source, target, full, random.Random(2), "edit_region_hardlock")
    assert full_result["candidate_count"] == source.numel()
    with pytest.raises(ValueError, match="non-empty"):
        corrupt_target_spatial(source, target, torch.zeros_like(tiny), random.Random(3), "edit_region_hardlock")


def test_newline_mapping_is_row_major_and_never_supervised():
    source, target, mask = _payload((2, 3), torch.tensor([[True, False, True], [False, False, False]]))
    result = corrupt_target_spatial(source, target, mask, random.Random(4), "edit_region_hardlock")
    tokens, labels = result["tokens"], result["labels"]
    assert len(tokens) == len(labels) == 2 * (3 + 1)
    assert tokens[3] == tokens[7] == SPECIAL_TOKENS["newline_token"]
    assert labels[3] == labels[7] == -100


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 9, 2] if text == "edit" else [1, 9, 2, 3]}


def test_dataset_is_reorder_invariant_and_fails_fast_on_empty_editregion(tmp_path):
    source, target, mask = _payload()
    token = tmp_path / "token.pt"
    torch.save({"source_codes": source, "target_codes": target, "edit_mask": mask, "token_height": 3, "token_width": 4}, token)
    row = '{"dataset_name":"magicbrush","sample_key":"same","instruction":"edit","token_file":"token.pt"}\n'
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(row, encoding="utf-8")
    first = EditTokenDataset(manifest, _Tokenizer(), condition_dropout=0, seed=42, target_corruption_mode="edit_region_hardlock")[0]
    second = EditTokenDataset(manifest, _Tokenizer(), condition_dropout=0, seed=42, target_corruption_mode="edit_region_hardlock")[0]
    assert first["input_ids"] == second["input_ids"]
    assert first["labels"] == second["labels"]
    torch.save({"source_codes": source, "target_codes": target, "edit_mask": torch.zeros_like(mask), "token_height": 3, "token_width": 4}, token)
    with pytest.raises(ValueError, match="dataset_name=magicbrush sample_key=same"):
        EditTokenDataset(manifest, _Tokenizer(), condition_dropout=0, target_corruption_mode="edit_region_hardlock")[0]
