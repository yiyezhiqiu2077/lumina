from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY / "scripts/eval/plot_training_curves.py"
SPEC = importlib.util.spec_from_file_location("plot_training_curves", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def _write_metrics(path: Path, objective: str) -> Path:
    records = []
    for step in range(1, 4):
        record = {"step": step, "objective": objective, "L_gen": 3.0 / step, "L_total": 4.0 / step}
        if objective == "attention":
            record.update(
                L_attn_raw=2.0 / step,
                weighted_attn_loss=0.2 / step,
                attn_to_gen_ratio=0.1,
                conditional_localization_mass=0.4 + step * 0.01,
                actual_full_attention_edit_mask_mass=0.05 + step * 0.01,
                attention_entropy=6.5 - step * 0.01,
                per_layer={
                    str(layer): {"conditional_localization_mass": 0.4 + step * 0.01 + layer * 0.0001}
                    for layer in (24, 25, 26, 27)
                },
            )
        if objective == "gce":
            record.update(gce_loss=1.0 / step, gce_loss_k1024=0.6 / step, gce_loss_k512=0.4 / step)
        records.append(record)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("objective", ("ce", "attention", "gce"))
def test_objective_plots_are_created(objective: str, tmp_path: Path):
    metrics = _write_metrics(tmp_path / f"{objective}.jsonl", objective)
    generated = PLOT.plot_objective(
        metrics, objective=objective, steps_per_epoch=2, smooth_window=2, output=tmp_path / objective
    )
    assert generated
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)


def test_rolling_mean_and_missing_metric_errors_are_clear(tmp_path: Path):
    assert PLOT.rolling_mean([1.0, 3.0, 5.0], 1).tolist() == [1.0, 3.0, 5.0]
    metrics = _write_metrics(tmp_path / "ce.jsonl", "ce")
    with pytest.raises(ValueError, match="missing metric"):
        PLOT.metric_values(PLOT.read_metrics(metrics), "gce_loss")


def test_compare_mode_creates_generation_loss_plot(tmp_path: Path):
    paths = {objective: _write_metrics(tmp_path / f"{objective}.jsonl", objective) for objective in ("ce", "attention", "gce")}
    generated = PLOT.plot_comparison(
        ce=paths["ce"], attention=paths["attention"], gce=paths["gce"],
        steps_per_epoch=2, smooth_window=2, output=tmp_path / "comparison",
    )
    assert generated.is_file() and generated.stat().st_size > 0


def test_cli_defaults_to_regular_plot_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metrics = _write_metrics(tmp_path / "ce.jsonl", "ce")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--input", str(metrics),
            "--objective", "ce",
            "--output", str(tmp_path / "cli-output"),
        ],
    )
    args = PLOT.parse_args()
    assert args.mode == "plot"
