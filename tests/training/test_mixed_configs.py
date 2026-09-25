from pathlib import Path

import pytest

from training.config import load_train_config


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture()
def mixed_environment(monkeypatch, tmp_path):
    manifest = tmp_path / "mixed/train/manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    # Deliberately not divisible by 8: this checks the exact drop_last recipe.
    manifest.write_text('{"dataset_name":"magicbrush"}\n' * 16611, encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_CONFIG", str(manifest))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GCE_CLUSTER_PATH", str(tmp_path / "clusters.pt"))


@pytest.mark.parametrize("name", ("mixed_ce_8g_b4_a1.yaml", "mixed_attention_postrope_region_8g_b4_a1.yaml", "mixed_gce_8g_b4_a1.yaml"))
def test_formal_mixed_configs_compute_steps_from_manifest(name, mixed_environment):
    args = load_train_config(REPOSITORY / "configs/train/formal" / name)
    assert (args.nproc_per_node, args.batch_size, args.gradient_accumulation) == (8, 4, 1)
    assert args.global_batch_size == 32
    assert args.optimizer_steps_per_epoch == 519
    assert args.max_steps == args.scheduler_horizon_steps == 5190
    assert args.save_steps == 519
    assert args.loss_reduction == "token_mean"


@pytest.mark.parametrize("name", ("mixed_ce_4g.yaml", "mixed_attention_4g.yaml", "mixed_gce_4g.yaml"))
def test_validation_mixed_configs_match_global_batch_and_formal_horizon(name, mixed_environment):
    args = load_train_config(REPOSITORY / "configs/train/validation" / name)
    assert (args.nproc_per_node, args.batch_size, args.gradient_accumulation) == (4, 4, 2)
    assert args.global_batch_size == 32
    assert args.optimizer_steps_per_epoch == 519
    assert args.max_steps == args.save_steps == 20
    assert args.scheduler_horizon_steps == 5190
    assert args.loss_reduction == "token_mean"


def test_mixed_attention_recipe_is_postrope_region_mass(mixed_environment):
    args = load_train_config(
        REPOSITORY / "configs/train/formal/mixed_attention_postrope_region_8g_b4_a1.yaml"
    )
    assert args.attention_qk_stage == "post_rope"
    assert args.attention_loss_mode == "region_mass"
    assert args.attention_loss_weight == pytest.approx(0.3)
