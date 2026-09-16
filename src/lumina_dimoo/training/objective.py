"""Strict three-way objective dispatch for unified MagicBrush training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from lumina_dimoo.models import MagicBrushModelOutput

OBJECTIVE_MODES = ("ce", "attention", "gce")


@dataclass
class ObjectiveForwardResult:
    output: MagicBrushModelOutput
    attention_auxiliary: Optional[torch.Tensor] = None
    gce_loss: Optional[torch.Tensor] = None
    gce_metrics: Optional[dict[str, torch.Tensor]] = None


def validate_objective(objective: str) -> str:
    if objective not in OBJECTIVE_MODES:
        raise ValueError(f"objective must be one of {OBJECTIVE_MODES}, got {objective!r}")
    return objective


def run_model_for_objective(
    model,
    *,
    objective: str,
    input_ids,
    labels,
    attention_layers: tuple[int, ...] | list[int] = (),
    attention_masks: Optional[dict[str, Any]] = None,
    gce_objective=None,
) -> ObjectiveForwardResult:
    """Run exactly the model/objective work allowed by one objective mode."""
    validate_objective(objective)
    common = {"input_ids": input_ids, "labels": labels, "return_training_output": True}
    if objective == "ce":
        return ObjectiveForwardResult(output=model(**common))
    if objective == "attention":
        if not attention_layers or attention_masks is None:
            raise ValueError("attention objective requires layers and supervision masks")
        output = model(
            **common,
            attention_supervision_layers=list(attention_layers),
            **attention_masks,
        )
        if output.attention_auxiliary is None:
            raise RuntimeError("attention objective requested auxiliary data but model returned none")
        return ObjectiveForwardResult(output=output, attention_auxiliary=output.attention_auxiliary)
    if gce_objective is None:
        raise ValueError("gce objective requires a configured GCE objective")
    output = model(**common)
    gce_loss, gce_metrics = gce_objective(output.logits, output.labels)
    return ObjectiveForwardResult(output=output, gce_loss=gce_loss, gce_metrics=gce_metrics)


def compose_total_loss(
    objective: str,
    generation_loss: torch.Tensor,
    *,
    attention_loss: Optional[torch.Tensor] = None,
    attention_weight: float = 0.1,
    gce_loss: Optional[torch.Tensor] = None,
    gce_weight: float = 1.0,
) -> torch.Tensor:
    """Combine only the loss term permitted by the selected objective."""
    validate_objective(objective)
    if objective == "ce":
        return generation_loss
    if objective == "attention":
        if attention_loss is None:
            raise ValueError("attention objective requires attention_loss")
        return generation_loss + attention_weight * attention_loss
    if gce_loss is None:
        raise ValueError("gce objective requires gce_loss")
    return generation_loss + gce_weight * gce_loss
