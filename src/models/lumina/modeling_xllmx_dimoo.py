import functools
import logging
import math
from dataclasses import dataclass
from typing import List, Optional
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.objectives.reduction import reduce_supervised_values

__all__ = [
    "LLaDAForMultiModalGeneration",
    "MagicBrushModelOutput",
    "generation_ce_and_z_loss",
]


@dataclass
class MagicBrushModelOutput:
    """Training-only output shared by CE, attention, and GCE objectives."""

    generation_loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    attention_auxiliary: Optional[torch.Tensor] = None
    generation_z_loss: Optional[torch.Tensor] = None
    valid_target_count: Optional[torch.Tensor] = None
    max_abs_valid_logit: Optional[torch.Tensor] = None


def generation_ce_and_z_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CE, raw log-normalizer z-loss, target count, and max valid logit."""
    valid = labels != -100
    valid_logits = logits[valid].float()
    valid_targets = labels[valid]
    if not valid_targets.numel():
        zero = logits.float().sum() * 0.0
        return zero, zero, valid.sum(), zero.detach()
    token_ce = F.cross_entropy(valid_logits, valid_targets, reduction="none")
    token_z = torch.logsumexp(valid_logits, dim=-1).square()
    per_position_ce = logits.new_zeros(labels.shape, dtype=torch.float32)
    per_position_z = logits.new_zeros(labels.shape, dtype=torch.float32)
    per_position_ce[valid] = token_ce
    per_position_z[valid] = token_z
    ce = reduce_supervised_values(per_position_ce, valid, reduction)
    z_loss = reduce_supervised_values(per_position_z, valid, reduction)
    return ce, z_loss, valid.sum(), valid_logits.detach().abs().max()


def pad_batch_sequences(sequences, pad_value: int, max_tokens: int | None = None) -> list[list[int]]:
    max_tokens = max_tokens or max(len(sequence) for sequence in sequences)
    return [list(sequence) + [pad_value] * (max_tokens - len(sequence)) for sequence in sequences]


def pad_batch_labels(labels, max_tokens: int) -> list[list[int]]:
    return [list(label) + [-100] * (max_tokens - len(label)) for label in labels]

def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1  # 有效位置设为1
    return attention_mask

class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

    def forward(self, input_ids=None, labels=None, infer=False, use_cache=False, to_compute_mask=None, cat='', **kwargs):
        return_training_output = bool(kwargs.pop("return_training_output", False))
        loss_reduction = kwargs.pop("loss_reduction", "sample_mean")
        attention_supervision_layers = kwargs.pop("attention_supervision_layers", None)
        instruction_token_mask = kwargs.pop("instruction_token_mask", None)
        source_spatial_mask = kwargs.pop("source_spatial_mask", None)
        source_edit_mask = kwargs.pop("source_edit_mask", None)
        attention_active = kwargs.pop("attention_active", None)
        attention_qk_stage = kwargs.pop("attention_qk_stage", "post_rope")
        attention_loss_mode = kwargs.pop("attention_loss_mode", "normalized_mask_ce")
        return_attention_auxiliary = bool(attention_supervision_layers)
        if infer:
            input_ids = input_ids.tolist()
        # ========================================================
        # padding input batch len & attention bias for attention mask
        # ========================================================
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
        input_ids = pad_batch_sequences(input_ids, self.config.pad_token_id, max_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        # attn mask
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        def pad_boolean_masks(masks):
            if masks is None:
                return None
            padded = [list(mask) + [False] * (max_tokens - len(mask)) for mask in masks]
            return torch.tensor(padded, dtype=torch.bool, device=self.device)

        instruction_token_mask = pad_boolean_masks(instruction_token_mask)
        source_spatial_mask = pad_boolean_masks(source_spatial_mask)
        source_edit_mask = pad_boolean_masks(source_edit_mask)
        if attention_active is not None:
            attention_active = torch.as_tensor(attention_active, dtype=torch.bool, device=self.device)
        # ========================================================
        # model output
        # ========================================================
        output = LLaDAModelLM.forward(
            self,
            input_ids=input_ids,
            attention_bias=attention_bias,
            use_cache=use_cache,
            to_compute_mask=to_compute_mask,
            cat=cat,
            attention_supervision_layers=attention_supervision_layers,
            instruction_token_mask=instruction_token_mask,
            source_spatial_mask=source_spatial_mask,
            source_edit_mask=source_edit_mask,
            attention_active=attention_active,
            attention_qk_stage=attention_qk_stage,
            attention_loss_mode=attention_loss_mode,
            return_attention_auxiliary=return_attention_auxiliary,
        )
        if return_attention_auxiliary:
            output, attention_auxiliary = output
        if infer:
            return (output, attention_auxiliary) if return_attention_auxiliary else output

        # ========================================================
        # padding label batch len & loss
        # ========================================================
        if labels is None:
            raise ValueError("labels are required when infer=False")
        labels = pad_batch_labels(labels, max_tokens)
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        logits = output.logits
        loss, z_loss, valid_target_count, max_abs_valid_logit = generation_ce_and_z_loss(
            logits,
            labels,
            loss_reduction,
        )
        if return_training_output:
            return MagicBrushModelOutput(
                generation_loss=loss,
                logits=logits,
                labels=labels,
                attention_auxiliary=attention_auxiliary if return_attention_auxiliary else None,
                generation_z_loss=z_loss,
                valid_target_count=valid_target_count,
                max_abs_valid_logit=max_abs_valid_logit,
            )
        return (loss, attention_auxiliary) if return_attention_auxiliary else loss

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules
