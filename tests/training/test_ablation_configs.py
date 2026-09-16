from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.config import load_train_config


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY / "configs/train/ablation"
CONFIG_NAMES = ("mb_ce_2g_b8_a2.yaml", "mb_attention_2g_b8_a2.yaml", "mb_gce_2g_b8_a2.yaml")


@pytest.fixture(autouse=True)
def required_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("DATA_ROOT", "/tmp/data")
    monkeypatch.setenv("DATA_CONFIG", "/tmp/data/manifest.jsonl")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GCE_CLUSTER_PATH", "/tmp/gce_clusters.pt")


def _raw(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_ablation_configs_have_identical_matched_common_fields():
    configs = [_raw(name) for name in CONFIG_NAMES]
    excluded = {"experiment_name", "output_name", "objective", "attention_loss", "gce"}
    common = [{key: value for key, value in config.items() if key not in excluded} for config in configs]
    assert common[0] == common[1] == common[2]


def test_ablation_configs_load_to_the_requested_effective_batch_and_schedule():
    for name in CONFIG_NAMES:
        args = load_train_config(CONFIG_ROOT / name)
        assert (args.nproc_per_node, args.batch_size, args.gradient_accumulation, args.global_batch_size) == (2, 8, 2, 32)
        assert args.max_seq_len == 5120
        assert args.max_steps == 2750
        assert args.save_steps == 275
        assert args.learning_rate == pytest.approx(1e-5)
        assert args.warmup_steps == 20
        assert args.lora_rank == 16
        assert args.lora_alpha == pytest.approx(16.0)
        assert args.lora_dropout == pytest.approx(0.05)
        assert args.precision == "bf16"


def test_ablation_objective_specific_sections_are_explicit_and_isolated():
    ce, attention, gce = (_raw(name) for name in CONFIG_NAMES)
    assert ce["objective"] == {"mode": "ce"}
    assert "attention_loss" not in ce and "gce" not in ce
    assert attention["attention_loss"]["weight"] == pytest.approx(0.1)
    assert attention["attention_loss"]["layers"] == [24, 25, 26, 27]
    assert "gce" not in attention
    assert gce["gce"]["weight"] == pytest.approx(1.0)
    assert gce["gce"]["levels"] == [1024, 512]
    assert "attention_loss" not in gce
