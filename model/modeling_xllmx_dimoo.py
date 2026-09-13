import functools
import logging
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from .gce_loss import GroupedCrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
__all__ = ["LLaDAForMultiModalGeneration"]

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
        self.gce_loss = None

    def configure_gce(self, cluster_path, levels=(1024, 512)):
        self.gce_loss = GroupedCrossEntropyLoss.from_file(str(cluster_path), tuple(levels)).to(self.device)
    
    def forward(self, input_ids=None, labels=None, infer=False, use_cache=False, to_compute_mask=None, cat='', **kwargs):
        use_gce = kwargs.pop("use_gce", False)
        gce_weight = kwargs.pop("gce_weight", 1.0)
        gce_logit_grad_probe = kwargs.pop("gce_logit_grad_probe", False)
        attention_supervision_layers = kwargs.pop("attention_supervision_layers", None)
        instruction_token_mask = kwargs.pop("instruction_token_mask", None)
        source_spatial_mask = kwargs.pop("source_spatial_mask", None)
        source_edit_mask = kwargs.pop("source_edit_mask", None)
        attention_active = kwargs.pop("attention_active", None)
        return_attention_auxiliary = bool(attention_supervision_layers)
        if infer:
            input_ids = input_ids.tolist()
        # ========================================================
        # padding input batch len & attention bias for attention mask
        # ========================================================
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids] # padding 0 to right --> max length
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
            return_attention_auxiliary=return_attention_auxiliary,
        )
        if return_attention_auxiliary:
            output, attention_auxiliary = output
        if infer:
            return (output, attention_auxiliary) if return_attention_auxiliary else output
        
        # ========================================================
        # padding label batch len & loss
        # ========================================================
        labels = [label + [-100] * (max_tokens - len(label)) for label in labels] # padding -100 to right --> max length
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        logits = output.logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=-100,)
        if use_gce:
            if self.gce_loss is None:
                raise RuntimeError("use_gce=True requires model.configure_gce(cluster_path) before DDP wrapping")
            offset, codebook_size = 126356, 8192
            image_valid = (labels != -100) & (labels >= offset) & (labels < offset + codebook_size)
            image_logits = logits[image_valid][:, offset : offset + codebook_size]
            image_targets = labels[image_valid] - offset
            gce, per_level = self.gce_loss(image_logits, image_targets)
            total = loss + gce_weight * gce
            metrics = {"ce_loss": loss.detach(), "gce_loss": gce.detach(), **{f"gce_loss_k{k}": v.detach() for k, v in per_level.items()}, "image_token_count": image_valid.sum().detach()}
            if gce_logit_grad_probe:
                ce_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                gce_grad = torch.autograd.grad(gce, logits, retain_graph=True)[0]
                metrics["gce_to_ce_logit_grad_norm"] = gce_grad.float().norm().detach() / ce_grad.float().norm().detach().clamp_min(1e-12)
            return total, metrics
        return (loss, attention_auxiliary) if return_attention_auxiliary else loss
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules
