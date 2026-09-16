"""Checkpoint persistence for the unified MagicBrush trainer."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from training.lora import load_lora_state_dict, lora_state_dict


def save_checkpoint(output: Path, step: int, model, optimizer, scheduler, rank: int, objective: str) -> None:
    destination = output / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(lora_state_dict(model.module), destination / "lora.pt")
        (destination / "lora_config.json").write_text(
            json.dumps({"step": step, "objective": objective, "targets": ["q_proj", "k_proj", "v_proj", "attn_out"]}, indent=2)
            + "\n"
        )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "step": step,
            "objective": objective,
        },
        destination / f"training_state.rank{rank:02d}.pt",
    )
    dist.barrier()


def restore_checkpoint(checkpoint: Path, model, optimizer, scheduler, rank: int, max_steps: int, objective: str):
    load_lora_state_dict(model, torch.load(checkpoint / "lora.pt", map_location="cpu", weights_only=True))
    state = torch.load(checkpoint / f"training_state.rank{rank:02d}.pt", map_location="cpu")
    if state.get("objective") not in (None, objective):
        raise ValueError(f"checkpoint objective {state['objective']!r} does not match {objective!r}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if step >= max_steps:
        raise ValueError(f"resume step {step} must be below max_steps {max_steps}")
    resumed_lrs = []
    for index, group in enumerate(optimizer.param_groups):
        resumed_lr = scheduler.base_lrs[index] * scheduler.lr_lambdas[index](scheduler.last_epoch)
        group["lr"] = resumed_lr
        resumed_lrs.append(resumed_lr)
    scheduler._last_lr = resumed_lrs
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return step, {
        "checkpoint": str(checkpoint),
        "restored_step": step,
        "restored_optimizer": True,
        "restored_scheduler_progress": True,
        "scheduler_horizon": max_steps,
        "resumed_lr": resumed_lrs,
        "data_cursor_strategy": "derived_from_global_step_and_gradient_accumulation",
    }
