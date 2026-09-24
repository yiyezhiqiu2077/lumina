"""Strict three-way objective dispatch for unified MagicBrush training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from models.lumina.modeling_xllmx_dimoo import MagicBrushModelOutput

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
    loss_reduction: str = "sample_mean",
    attention_layers: tuple[int, ...] | list[int] = (),
    attention_qk_stage: str = "post_rope",
    attention_loss_mode: str = "normalized_mask_ce",
    attention_masks: Optional[dict[str, Any]] = None,
    gce_objective=None,
) -> ObjectiveForwardResult:
    """Run exactly the model/objective work allowed by one objective mode."""
    validate_objective(objective)
    common = {
        "input_ids": input_ids,
        "labels": labels,
        "return_training_output": True,
        "loss_reduction": loss_reduction,
    }
    if objective == "ce":
        return ObjectiveForwardResult(output=model(**common))
    if objective == "attention":
        if not attention_layers or attention_masks is None:
            raise ValueError("attention objective requires layers and supervision masks")
        output = model(
            **common,
            attention_supervision_layers=list(attention_layers),
            attention_qk_stage=attention_qk_stage,
            attention_loss_mode=attention_loss_mode,
            **attention_masks,
        )
        if output.attention_auxiliary is None:
            raise RuntimeError("attention objective requested auxiliary data but model returned none")
        return ObjectiveForwardResult(output=output, attention_auxiliary=output.attention_auxiliary)
    if gce_objective is None:
        raise ValueError("gce objective requires a configured GCE objective")
    output = model(**common)
    gce_loss, gce_metrics = gce_objective(
        output.logits,
        output.labels,
        reduction=loss_reduction,
    )
    return ObjectiveForwardResult(output=output, gce_loss=gce_loss, gce_metrics=gce_metrics)


def compose_total_loss(
    objective: str,
    generation_loss: torch.Tensor,
    *,
    generation_z_loss: Optional[torch.Tensor] = None,
    z_loss_weight: float = 0.0,
    attention_loss: Optional[torch.Tensor] = None,
    attention_weight: float = 0.1,
    gce_loss: Optional[torch.Tensor] = None,
    gce_weight: float = 1.0,
) -> torch.Tensor:
    """Combine the shared generation terms and only the selected auxiliary."""
    validate_objective(objective)
    total = generation_loss
    if z_loss_weight:
        if generation_z_loss is None:
            raise ValueError("nonzero z_loss_weight requires generation_z_loss")
        total = total + z_loss_weight * generation_z_loss
    if objective == "ce":
        return total
    if objective == "attention":
        if attention_loss is None:
            raise ValueError("attention objective requires attention_loss")
        return total + attention_weight * attention_loss
    if gce_loss is None:
        raise ValueError("gce objective requires gce_loss")
    return total + gce_weight * gce_loss
