from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lora = self.lora_b(self.lora_a(self.dropout(inputs).float())) * self.scaling
        return self.base(inputs) + lora.to(inputs.dtype)


@dataclass(frozen=True)
class LoRAReport:
    matched_modules: tuple[str, ...]
    trainable_names: tuple[str, ...]
    trainable_parameters: int


def inject_lora(
    model: nn.Module,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "attn_out"),
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> LoRAReport:
    model.requires_grad_(False)
    matched = []
    for module_name, module in list(model.named_modules()):
        for target in targets:
            if module_name.rsplit(".", 1)[-1] != target:
                continue
            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            original = getattr(parent, child_name)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"LoRA target {module_name} is {type(original).__name__}, expected Linear")
            setattr(parent, child_name, LoRALinear(original, rank, alpha, dropout))
            matched.append(module_name)
            break
    if not matched:
        raise ValueError(f"no LoRA modules matched targets {targets}")
    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if any(not (name.endswith("lora_a.weight") or name.endswith("lora_b.weight")) for name in trainable):
        raise AssertionError("non-LoRA parameters unexpectedly trainable")
    return LoRAReport(
        matched_modules=tuple(matched),
        trainable_names=trainable,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {
        name
        for name in parameters
        if name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
    }
    received = set(state)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(f"LoRA state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
