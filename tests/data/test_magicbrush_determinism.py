import random

import torch

from dataset.magicbrush import EditTokenDataset, stable_sample_seed


def test_sample_identity_seed_is_order_and_index_independent():
    first = stable_sample_seed(42, 3, "sample-abc")
    reordered = stable_sample_seed(42, 3, "sample-abc")
    assert first == reordered
    assert random.Random(first).sample(range(128), 8) == random.Random(reordered).sample(range(128), 8)


def test_epoch_and_global_seed_change_the_corruption_stream():
    base = stable_sample_seed(42, 3, "sample-abc")
    assert base != stable_sample_seed(42, 4, "sample-abc")
    assert base != stable_sample_seed(43, 3, "sample-abc")
    assert base != stable_sample_seed(42, 3, "sample-def")


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 9, 2] if text == "edit" else [1, 9, 2, 3]}


def test_edit_dataset_accepts_legacy_absolute_and_new_relative_token_paths(tmp_path):
    payload = {
        "source_codes": torch.zeros(32, 32, dtype=torch.int32),
        "target_codes": torch.zeros(32, 32, dtype=torch.int32),
        "edit_mask": torch.ones(32, 32, dtype=torch.bool),
        "token_height": 32,
        "token_width": 32,
    }
    token = tmp_path / "files" / "000000.pt"
    token.parent.mkdir()
    torch.save(payload, token)
    for token_file in (str(token), "files/000000.pt"):
        manifest = tmp_path / ("absolute.jsonl" if token_file == str(token) else "relative.jsonl")
        manifest.write_text(
            '{"dataset_name":"refedit","sample_key":"refedit/0","instruction":"edit","token_file":"'
            + token_file
            + '"}\n',
            encoding="utf-8",
        )
        item = EditTokenDataset(manifest, _Tokenizer(), condition_dropout=0)[0]
        assert item["dataset_name"] == "refedit"
        assert item["source_spatial_count"] == 1024
