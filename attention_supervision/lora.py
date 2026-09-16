"""Backward-compatible import path for the shared LoRA implementation."""

from lumina_dimoo.training.lora import LoRALinear, LoRAReport, inject_lora, load_lora_state_dict, lora_state_dict

__all__ = ["LoRALinear", "LoRAReport", "inject_lora", "load_lora_state_dict", "lora_state_dict"]
