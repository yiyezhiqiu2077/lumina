from __future__ import annotations

import pytest
import torch

import training.distributed as distributed


def test_gradient_decomposition_reports_raw_and_weighted_auxiliary_scales(monkeypatch):
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    generation = (4.0 * parameter).sum()
    raw_auxiliary = (3.0 * parameter).sum()
    weighted_auxiliary = 0.3 * raw_auxiliary
    values = distributed.gradient_decomposition(
        generation, raw_auxiliary, weighted_auxiliary, [parameter], torch.device("cpu")
    )
    assert values["generation_gradient_norm"].item() == pytest.approx(4.0)
    assert values["raw_auxiliary_gradient_norm"].item() == pytest.approx(3.0)
    assert values["weighted_auxiliary_gradient_norm"].item() == pytest.approx(0.9)
    assert values["raw_auxiliary_to_generation_gradient_ratio"].item() == pytest.approx(0.75)
    assert values["weighted_auxiliary_to_generation_gradient_ratio"].item() == pytest.approx(0.225)
    # Scaling the weighted objective has not altered the raw auxiliary loss.
    assert raw_auxiliary.item() == pytest.approx(3.0)


def test_global_peak_memory_stats_uses_global_max(monkeypatch):
    monkeypatch.setattr(distributed.torch.cuda, "max_memory_allocated", lambda device: 3 * 1024**3)
    monkeypatch.setattr(distributed.torch.cuda, "max_memory_reserved", lambda device: 5 * 1024**3)
    monkeypatch.setattr(distributed, "reduce_max", lambda value: value)
    stats = distributed.global_peak_memory_stats(torch.device("cpu"))
    assert stats["peak_allocated_gib"].item() == pytest.approx(3.0)
    assert stats["peak_reserved_gib"].item() == pytest.approx(5.0)
