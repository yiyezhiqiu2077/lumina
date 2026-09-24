from __future__ import annotations

import pytest
import torch

import training.distributed as distributed


def test_scaled_auxiliary_gradient_vectors_preserve_exact_weighting():
    raw = (torch.tensor([1.0, -2.0]), None, torch.tensor([3.0]))
    weighted = distributed.scale_reduced_auxiliary_gradients(raw, 0.3)
    assert weighted[1] is None
    assert torch.equal(weighted[0], 0.3 * raw[0])
    assert torch.equal(weighted[2], 0.3 * raw[2])


@pytest.mark.parametrize(
    ("weight", "expected_weighted"),
    ((0.3, 0.9), (1.0, 3.0), (0.0, 0.0)),
)
def test_gradient_decomposition_scales_the_ddp_reduced_raw_vector(
    monkeypatch, weight, expected_weighted
):
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 1)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    generation = (4.0 * parameter).sum()
    raw_auxiliary = (3.0 * parameter).sum()
    values = distributed.gradient_decomposition(
        generation, raw_auxiliary, weight, [parameter], torch.device("cpu")
    )
    # The test all_reduce is an identity rank, so these are post-reduction
    # vectors. The implementation must scale that vector directly.
    assert values["shared_generation_gradient_norm"].item() == pytest.approx(4.0)
    assert values["generation_gradient_norm"].item() == pytest.approx(4.0)
    assert values["raw_auxiliary_gradient_norm"].item() == pytest.approx(3.0)
    assert values["weighted_auxiliary_gradient_norm"].item() == pytest.approx(expected_weighted)
    assert values["raw_auxiliary_to_generation_gradient_ratio"].item() == pytest.approx(0.75)
    assert values["weighted_auxiliary_to_generation_gradient_ratio"].item() == pytest.approx(
        expected_weighted / 4.0
    )
    assert values["auxiliary_gradient_norm"].item() == pytest.approx(expected_weighted)
    # Scaling the weighted objective has not altered the raw auxiliary loss.
    assert raw_auxiliary.item() == pytest.approx(3.0)


def test_global_peak_memory_stats_uses_global_max(monkeypatch):
    monkeypatch.setattr(distributed.torch.cuda, "max_memory_allocated", lambda device: 3 * 1024**3)
    monkeypatch.setattr(distributed.torch.cuda, "max_memory_reserved", lambda device: 5 * 1024**3)
    monkeypatch.setattr(distributed, "reduce_max", lambda value: value)
    stats = distributed.global_peak_memory_stats(torch.device("cpu"))
    assert stats["peak_allocated_gib"].item() == pytest.approx(3.0)
    assert stats["peak_reserved_gib"].item() == pytest.approx(5.0)
