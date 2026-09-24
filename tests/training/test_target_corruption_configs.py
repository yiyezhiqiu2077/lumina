from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.config import load_train_config


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture()
def environment(monkeypatch, tmp_path):
    manifest = tmp_path / "mixed" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text('{"dataset_name":"magicbrush"}\n' * 16611, encoding="utf-8")
    for key, value in {
        "MODEL_PATH": tmp_path / "model",
        "DATA_ROOT": tmp_path,
        "DATA_CONFIG": manifest,
        "OUTPUT_ROOT": tmp_path / "outputs",
        "GCE_CLUSTER_PATH": tmp_path / "clusters.pt",
    }.items():
        monkeypatch.setenv(key, str(value))


@pytest.mark.parametrize(
    ("full_name", "edit_name"),
    (
        ("mixed_ce_8g_b4_a1.yaml", "mixed_ce_editregion_8g_b4_a1.yaml"),
        ("mixed_attention_postrope_region_8g_b4_a1.yaml", "mixed_attention_editregion_8g_b4_a1.yaml"),
        ("mixed_gce_8g_b4_a1.yaml", "mixed_gce_editregion_8g_b4_a1.yaml"),
    ),
)
def test_formal_target_corruption_pairs_have_no_scientific_drift(full_name, edit_name):
    full = yaml.safe_load((REPOSITORY / "configs/train/formal" / full_name).read_text())
    edit = yaml.safe_load((REPOSITORY / "configs/train/ablation" / edit_name).read_text())
    for config in (full, edit):
        config.pop("experiment_name")
        config.pop("output_name")
        config.pop("target_corruption")
    assert full == edit


@pytest.mark.parametrize("name", ("mixed_ce_editregion_4g.yaml", "mixed_attention_editregion_4g.yaml", "mixed_gce_editregion_4g.yaml"))
def test_editregion_validation_configs_are_four_gpu_correctness_runs(name, environment):
    args = load_train_config(REPOSITORY / "configs/train/validation" / name)
    assert args.target_corruption_mode == "edit_region_hardlock"
    assert (args.nproc_per_node, args.batch_size, args.gradient_accumulation, args.global_batch_size) == (4, 4, 2, 32)
    assert args.max_steps == args.save_steps == 20
    assert args.scheduler_horizon_steps == 5190
