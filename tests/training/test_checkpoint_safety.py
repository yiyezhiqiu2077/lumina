from argparse import Namespace
import json
from pathlib import Path

import pytest

from training.checkpoint import _validate_metrics_continuity, build_run_fingerprint


def _write(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _args(tmp_path: Path) -> Namespace:
    model = tmp_path / "model"
    _write(model / "config.json", "{}\n")
    config = _write(tmp_path / "train.yaml", "objective: ce\n")
    dataset = _write(tmp_path / "dataset.yaml", "vq_grid: [32, 32]\n")
    distributed = _write(tmp_path / "distributed.yaml", "launcher: torchrun\n")
    manifest = _write(tmp_path / "manifest.jsonl", "{}\n")
    return Namespace(
        objective="ce",
        model=model,
        config_file=config,
        dataset_config_file=dataset,
        distributed_config_file=distributed,
        train_manifest=manifest,
        gce_clusters=None,
        max_steps=2750,
        scheduler_horizon_steps=2750,
        save_steps=275,
        batch_size=4,
        gradient_accumulation=1,
        warmup_steps=20,
        learning_rate=1e-5,
        optimizer_betas=(0.9, 0.95),
        weight_decay=0.1,
        max_grad_norm=4.0,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_targets=("q_proj", "k_proj", "v_proj", "attn_out"),
        loss_reduction="sample_mean",
        z_loss_weight=1e-5,
        attention_loss_weight=0.1,
        attention_layers=[],
        gce_weight=1.0,
        gce_levels=[1024, 512],
        seed=42,
        condition_dropout=0.1,
        max_seq_len=5120,
        quality_gate_enabled=True,
        quality_baseline_start=101,
        quality_baseline_end=400,
        quality_window_size=50,
        quality_max_loss_ratio=1.75,
        quality_consecutive_windows=2,
        quality_max_clip_fraction=0.95,
        quality_max_abs_logit=1e4,
        quality_max_update_ratio=0.25,
    )


def test_fingerprint_changes_with_training_semantics_but_not_config_path(tmp_path):
    first = _args(tmp_path / "first")
    second = Namespace(**vars(first))
    second.config_file = _write(
        tmp_path / "renamed-train.yaml",
        first.config_file.read_text(encoding="utf-8"),
    )
    fingerprint = build_run_fingerprint(first, 8)
    same_recipe = build_run_fingerprint(second, 8)
    assert fingerprint["digest"] == same_recipe["digest"]
    second.loss_reduction = "token_mean"
    assert build_run_fingerprint(second, 8)["digest"] != fingerprint["digest"]


def test_fingerprint_distinguishes_stop_and_scheduler_horizons(tmp_path):
    formal = _args(tmp_path / "formal")
    formal_fingerprint = build_run_fingerprint(formal, 8)

    probe = Namespace(**vars(formal))
    probe.max_steps = 825
    probe_fingerprint = build_run_fingerprint(probe, 8)
    schedule = probe_fingerprint["payload"]["schedule"]
    assert schedule["stop_after_steps"] == 825
    assert schedule["scheduler_horizon_steps"] == 2750
    assert probe_fingerprint["digest"] != formal_fingerprint["digest"]

    compressed = Namespace(**vars(probe))
    compressed.scheduler_horizon_steps = 825
    assert build_run_fingerprint(compressed, 8)["digest"] != probe_fingerprint["digest"]


def test_fingerprint_distinguishes_target_corruption_modes(tmp_path):
    full = _args(tmp_path / "full")
    full.target_corruption_mode = "full_target"
    edit = Namespace(**vars(full))
    edit.target_corruption_mode = "edit_region_hardlock"
    assert build_run_fingerprint(full, 8)["digest"] != build_run_fingerprint(edit, 8)["digest"]


def test_metrics_continuity_requires_exact_prefix(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (1, 2, 3)),
        encoding="utf-8",
    )
    _validate_metrics_continuity(metrics, 3)
    metrics.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (1, 3)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous steps"):
        _validate_metrics_continuity(metrics, 3)
