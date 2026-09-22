from __future__ import annotations

import math

import torch


def expand_grouped_keys(keys: torch.Tensor, query_heads: int) -> torch.Tensor:
    if keys.shape[1] == query_heads:
        return keys
    if query_heads % keys.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    return keys.repeat_interleave(query_heads // keys.shape[1], dim=1)


def source_conditional_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    instruction_masks: torch.Tensor,
    source_spatial_masks: torch.Tensor,
) -> list[torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, heads, sequence, head_dim]")
    k = expand_grouped_keys(k, q.shape[1])
    maps = []
    for sample in range(q.shape[0]):
        instruction_q = q[sample, :, instruction_masks[sample], :].float()
        source_k = k[sample, :, source_spatial_masks[sample], :].float()
        if not instruction_q.shape[1] or not source_k.shape[1]:
            raise ValueError("instruction and source spatial selections must be non-empty")
        scores = torch.einsum("hid,hjd->hij", instruction_q, source_k) / math.sqrt(q.shape[-1])
        maps.append(scores.softmax(dim=-1).mean(dim=(0, 1)))
    return maps


def spatial_cross_entropy(
    maps: list[torch.Tensor],
    edit_masks: list[torch.Tensor],
    active: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    masses = []
    supervised = []
    for index, (attention, mask) in enumerate(zip(maps, edit_masks)):
        mask = mask.to(device=attention.device, dtype=torch.float32).flatten()
        enabled = bool(active[index]) and bool(mask.sum() > 0)
        if not enabled:
            continue
        target = mask / mask.sum()
        losses.append(-(target * attention.clamp_min(eps).log()).sum())
        masses.append((attention * mask).sum())
        supervised.append(index)
    if losses:
        loss = torch.stack(losses).mean()
        mass = torch.stack(masses).mean()
    else:
        loss = sum((attention.sum() * 0.0 for attention in maps))
        mass = loss.detach()
    return loss, {
        "conditional_localization_mass": mass,
        "active_samples": torch.tensor(len(supervised), device=loss.device),
    }


def layer_attention_auxiliary(
    q: torch.Tensor,
    k: torch.Tensor,
    instruction_masks: torch.Tensor,
    source_spatial_masks: torch.Tensor,
    source_edit_masks: torch.Tensor,
    active: torch.Tensor,
    mode: str = "normalized_mask_ce",
    eps: float = 1e-8,
) -> torch.Tensor:
    if mode not in {"normalized_mask_ce", "region_mass"}:
        raise ValueError(f"unsupported attention loss mode: {mode}")
    losses = []
    conditional_masses = []
    entropies = []
    full_masses = []
    enrichments = []
    mask_entropies = []
    effective_kls = []
    expanded_k = expand_grouped_keys(k, q.shape[1])
    for sample in range(q.shape[0]):
        if not bool(active[sample]):
            continue
        source_edit = source_edit_masks[sample, source_spatial_masks[sample]].float()
        if not bool(source_edit.sum() > 0):
            continue
        instruction_q = q[sample, :, instruction_masks[sample], :].float()
        source_k = expanded_k[sample, :, source_spatial_masks[sample], :].float()
        if not instruction_q.shape[1] or not source_k.shape[1]:
            raise ValueError("active samples require non-empty instruction and source spatial selections")
        conditional_map = (
            torch.einsum("hid,hjd->hij", instruction_q, source_k) / math.sqrt(q.shape[-1])
        ).softmax(dim=-1).mean(dim=(0, 1))
        probability = conditional_map.float().clamp_min(eps)
        mask_count = source_edit.sum()
        probability_mass = (probability * source_edit).sum()
        normalized_ce = -((source_edit / mask_count) * probability.log()).sum()
        if mode == "normalized_mask_ce":
            losses.append(normalized_ce)
        else:
            losses.append(-(probability_mass + eps).log())
        conditional_masses.append(probability_mass)
        entropies.append(-(probability * probability.log()).sum())
        mask_entropy = mask_count.log()
        mask_entropies.append(mask_entropy)
        enrichments.append(probability_mass / (mask_count / source_edit.numel()))
        effective_kls.append(normalized_ce - mask_entropy)

        full_k = expanded_k[sample].float()
        full_probability = (
            torch.einsum("hid,hjd->hij", instruction_q, full_k) / math.sqrt(q.shape[-1])
        ).softmax(dim=-1)
        full_masses.append((full_probability * source_edit_masks[sample].float()[None, None, :]).sum(-1).mean())

    if losses:
        return torch.stack(
            [
                torch.stack(losses).mean(),
                torch.stack(conditional_masses).mean().detach(),
                torch.stack(entropies).mean().detach(),
                torch.stack(full_masses).mean().detach(),
                q.new_tensor(float(len(losses))),
                torch.stack(enrichments).mean().detach(),
                torch.stack(mask_entropies).mean().detach(),
                torch.stack(effective_kls).mean().detach(),
            ]
        )
    zero = q.sum() * 0.0
    return torch.stack([zero, zero, zero, zero, zero, zero, zero, zero])
