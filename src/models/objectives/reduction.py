"""Reduction helpers for losses over variably masked image tokens."""
from __future__ import annotations

import torch


LOSS_REDUCTIONS = ("sample_mean", "token_mean")


def reduce_supervised_values(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Reduce tokenwise values while ignoring unsupervised positions.

    ``sample_mean`` gives every sample with at least one supervised token equal
    weight. ``token_mean`` retains the historical flattened-token behavior.
    """
    if reduction not in LOSS_REDUCTIONS:
        raise ValueError(f"loss reduction must be one of {LOSS_REDUCTIONS}, got {reduction!r}")
    if values.shape != valid_mask.shape:
        raise ValueError(
            f"values and valid_mask must have the same shape, got {values.shape} and {valid_mask.shape}"
        )
    valid_mask = valid_mask.bool()
    if not bool(valid_mask.any()):
        return values.sum() * 0.0
    if reduction == "token_mean":
        return values[valid_mask].mean()
    if values.ndim < 2:
        raise ValueError("sample_mean requires a batch dimension and at least one value dimension")
    flattened_values = values.reshape(values.shape[0], -1)
    flattened_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    counts = flattened_mask.sum(dim=1)
    active = counts > 0
    per_sample = (flattened_values * flattened_mask).sum(dim=1) / counts.clamp_min(1)
    return per_sample[active].mean()
