from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk


@torch.no_grad()
def generate_i2i_gt_mask_hard_lock(
    model,
    prompt: torch.LongTensor,
    *,
    source_codes: torch.LongTensor,
    edit_mask: torch.BoolTensor,
    code_start: int,
    timesteps: int = 64,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 2.5,
    cfg_img: float = 4.0,
    uncon_text: torch.LongTensor,
    uncon_image: torch.LongTensor,
    codebook_size: int = 8192,
    vocab_offset: int = 126356,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    device = next(model.parameters()).device
    x = prompt.to(device).clone()
    source_codes = source_codes.to(device).long()
    edit_mask = edit_mask.to(device).bool()
    if source_codes.shape != edit_mask.shape:
        raise ValueError(f"source/edit grid mismatch: {source_codes.shape} vs {edit_mask.shape}")
    height, width = source_codes.shape
    spatial_offsets = torch.arange(height * width, device=device)
    target_positions = code_start + spatial_offsets + torch.div(spatial_offsets, width, rounding_mode="floor")
    keep_positions = target_positions[~edit_mask.flatten()]
    keep_values = source_codes.flatten()[~edit_mask.flatten()] + vocab_offset
    source_target_positions = target_positions
    source_target_values = source_codes.flatten() + vocab_offset

    x[0, keep_positions] = keep_values
    x[0, target_positions[edit_mask.flatten()]] = mask_token_id
    initial_unknown = int(edit_mask.sum().item())
    if initial_unknown <= 0:
        return source_codes.flatten().unsqueeze(0) + vocab_offset

    for step in range(timesteps):
        x[0, keep_positions] = keep_values
        vq_mask = x == mask_token_id
        unknown_positions = vq_mask.nonzero(as_tuple=False)[:, 1]
        if not unknown_positions.numel():
            break
        if step < timesteps - 1:
            fraction = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = max(1, int(math.floor(initial_unknown * float(fraction.item()))))
        else:
            keep_n = 0

        suffix = x[:, code_start - 2 :]
        uncond_text_input = torch.cat((uncon_text.to(device), suffix), dim=1)
        uncond_image_input = torch.cat((uncon_image.to(device), suffix), dim=1)
        uncond_text_mask = uncond_text_input == mask_token_id
        uncond_image_mask = uncond_image_input == mask_token_id
        conditional_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
        text_logits = model(uncond_text_input, infer=True).logits[
            :, uncond_text_mask[0], vocab_offset : vocab_offset + codebook_size
        ]
        image_logits = model(uncond_image_input, infer=True).logits[
            :, uncond_image_mask[0], vocab_offset : vocab_offset + codebook_size
        ]
        logits = conditional_logits + cfg_scale * (conditional_logits - text_logits)
        logits = logits + cfg_img * (conditional_logits - image_logits)
        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        probabilities = logits.softmax(dim=-1)
        confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        x.view(-1)[unknown_positions] = sampled.view(-1) + vocab_offset
        if keep_n:
            remask = mask_by_random_topk(
                torch.tensor([keep_n], device=device),
                confidence,
                temperature=temperature,
                generator=generator,
            )
            x.view(-1)[unknown_positions[remask.view(-1)]] = mask_token_id
        x[0, keep_positions] = keep_values

    if bool((x[0, target_positions] == mask_token_id).any()):
        raise AssertionError("final target grid still contains mask tokens")
    x[0, keep_positions] = keep_values
    if not torch.equal(x[0, keep_positions], keep_values):
        raise AssertionError("GT-mask hard lock failed outside the edit region")
    return x[0, source_target_positions].view(1, height * width)
