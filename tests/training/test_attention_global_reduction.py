import pytest
import torch

from training import distributed


def test_global_attention_reduction_preserves_all_logged_metrics(monkeypatch):
    """The reducer must retain enrichment/mask-entropy/effective-KL fields."""
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda value, *args, **kwargs: value)
    auxiliary_by_layer = torch.tensor(
        [
            [2.0, 0.2, 1.2, 0.3, 1.0, 2.5, 0.7, 1.3],
            [4.0, 0.4, 1.4, 0.5, 1.0, 3.5, 0.9, 3.1],
        ],
        requires_grad=True,
    )

    backward_loss, metrics, local = distributed._global_attention_loss(auxiliary_by_layer, world_size=1)

    assert local.shape == (8,)
    assert metrics.shape == (8,)
    assert metrics.tolist() == pytest.approx([3.0, 0.3, 1.3, 0.4, 1.0, 3.0, 0.8, 2.2])
    backward_loss.backward()
    assert auxiliary_by_layer.grad is not None
    assert auxiliary_by_layer.grad[:, 0].abs().sum() > 0
