from argparse import Namespace

import pytest
import torch

from training.distributed import build_scheduler


def _learning_rates(stop_after_steps: int, scheduler_horizon_steps: int) -> list[float]:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-5)
    args = Namespace(
        max_steps=stop_after_steps,
        scheduler_horizon_steps=scheduler_horizon_steps,
        warmup_steps=20,
    )
    scheduler = build_scheduler(optimizer, args)
    values = []
    for _ in range(stop_after_steps):
        optimizer.step()
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    return values


def test_bounded_probe_uses_formal_cosine_scheduler_horizon():
    formal = _learning_rates(2750, 2750)
    probe = _learning_rates(825, 2750)
    assert probe == pytest.approx(formal[:825])
    for step in (300, 550, 825):
        assert probe[step - 1] == pytest.approx(formal[step - 1])
    assert probe[-1] > 8.0e-6


def test_compressed_horizon_would_not_match_formal_recipe():
    formal_prefix = _learning_rates(825, 2750)
    compressed = _learning_rates(825, 825)
    assert compressed[549] < 3.0e-6
    assert compressed[549] != pytest.approx(formal_prefix[549])
