"""Image-only grouped cross entropy over Lumina's visual codebook."""
from __future__ import annotations

import torch
from torch import nn


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
        if int(payload["codebook_size"]) != 8192:
            raise ValueError("GCE clusters must describe 8192 visual codes")
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
