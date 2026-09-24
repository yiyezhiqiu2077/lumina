import json

import pytest

from training.quality import (
    QualityGateError,
    TrainingQualityGate,
    require_quality_success,
    write_quality_status,
)


def make_gate(**overrides):
    values = {
        "baseline_start": 1,
        "baseline_end": 4,
        "window_size": 2,
        "max_loss_ratio": 1.5,
        "consecutive_windows": 2,
        "max_clip_fraction": 1.0,
        "max_abs_logit": 100.0,
        "max_update_ratio": 0.5,
    }
    values.update(overrides)
    return TrainingQualityGate(**values)


def observe(gate, step, loss, **overrides):
    values = {
        "generation_loss": loss,
        "grad_norm": 1.0,
        "max_abs_logit": 10.0,
        "update_ratio": 0.01,
        "clipped": False,
    }
    values.update(overrides)
    return gate.observe(step=step, **values)


def test_gate_uses_non_overlapping_windows_for_sustained_regression():
    gate = make_gate()
    for step in range(1, 5):
        observe(gate, step, 2.0)
    observe(gate, 5, 4.0)
    first = observe(gate, 6, 4.0)
    assert first["quality_bad_loss_windows"] == 1
    observe(gate, 7, 4.0)
    with pytest.raises(QualityGateError, match="non-overlapping windows"):
        observe(gate, 8, 4.0)


def test_gate_fails_fast_on_nonfinite_and_runaway_update():
    gate = make_gate()
    with pytest.raises(QualityGateError, match="non-finite generation_loss"):
        observe(gate, 1, float("nan"))
    with pytest.raises(QualityGateError, match="update_ratio"):
        observe(gate, 1, 2.0, update_ratio=0.6)


def test_gate_state_round_trip_preserves_resume_window():
    gate = make_gate()
    for step in range(1, 6):
        observe(gate, step, 2.0)
    restored = make_gate()
    restored.load_state_dict(gate.state_dict())
    assert restored.state_dict() == gate.state_dict()
    assert restored.baseline == gate.baseline


def test_quality_status_must_explicitly_succeed(tmp_path):
    path = tmp_path / "quality_status.json"
    write_quality_status(path, "RUNNING", step=3)
    with pytest.raises(SystemExit, match="not SUCCEEDED"):
        require_quality_success(path)
    write_quality_status(path, "SUCCEEDED", step=20)
    require_quality_success(path)
    assert json.loads(path.read_text())["step"] == 20
