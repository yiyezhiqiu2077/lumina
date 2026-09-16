"""Image-only grouped cross entropy, isolated from the model wrapper."""
from __future__ import annotations

import torch
from torch import nn

from utils.constants import SPECIAL_TOKENS, VISUAL_CODEBOOK_SIZE


class GroupedCrossEntropyLoss(nn.Module):
    def __init__(self, levels: dict[int, dict[str, torch.Tensor]]):
        super().__init__()
        self.levels = tuple(sorted(int(level) for level in levels))
        for level in self.levels:
            value = levels[level]
            self.register_buffer(f"token_to_cluster_{level}", value["token_to_cluster"].long(), persistent=True)
            self.register_buffer(f"map_{level}", value["cluster_map"].long(), persistent=True)
            self.register_buffer(f"sizes_{level}", value["cluster_sizes"].long(), persistent=True)

    @classmethod
    def from_file(cls, path: str, requested_levels: tuple[int, ...] = (1024, 512)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["codebook_size"]) != VISUAL_CODEBOOK_SIZE:
            raise ValueError(f"GCE clusters must describe {VISUAL_CODEBOOK_SIZE} visual codes")
        levels = {int(level): payload["levels"][int(level)] for level in requested_levels}
        return cls(levels)

    def forward(self, image_logits: torch.Tensor, image_targets: torch.Tensor):
        if image_logits.numel() == 0:
            zero = image_logits.sum() * 0.0
            return zero, {level: zero.detach() for level in self.levels}
        logits = image_logits.float()
        denominator = torch.logsumexp(logits, dim=-1)
        values = {}
        for level in self.levels:
            cluster_map = getattr(self, f"map_{level}")
            cluster_sizes = getattr(self, f"sizes_{level}")
            cluster_ids = getattr(self, f"token_to_cluster_{level}")[image_targets]
            members = cluster_map[cluster_ids]
            valid = torch.arange(members.shape[-1], device=members.device)[None] < cluster_sizes[cluster_ids, None]
            selected = logits.gather(1, members.clamp_min(0)).masked_fill(~valid, float("-inf"))
            values[level] = (denominator - torch.logsumexp(selected, dim=-1)).mean()
        return sum(values.values()), values


class GCEObjective(nn.Module):
    """Adds grouped image-token CE without changing the base-model forward."""

    def __init__(self, grouped_loss: GroupedCrossEntropyLoss):
        super().__init__()
        self.grouped_loss = grouped_loss

    @classmethod
    def from_clusters(cls, cluster_path: str, levels: tuple[int, ...] = (1024, 512)) -> "GCEObjective":
        return cls(GroupedCrossEntropyLoss.from_file(cluster_path, levels))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        offset = SPECIAL_TOKENS["image_token_offset"]
        image_valid = (labels != -100) & (labels >= offset) & (labels < offset + VISUAL_CODEBOOK_SIZE)
        image_logits = logits[image_valid][:, offset : offset + VISUAL_CODEBOOK_SIZE]
        image_targets = labels[image_valid] - offset
        gce, per_level = self.grouped_loss(image_logits, image_targets)
        metrics = {
            "gce_loss": gce.detach(),
            **{f"gce_loss_k{level}": value.detach() for level, value in per_level.items()},
            "image_token_count": image_valid.sum().detach(),
        }
        return gce, metrics
