#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from attention_supervision.lora import inject_lora, load_lora_state_dict, lora_state_dict
from datasets.magicbrush_tokens import MagicBrushTokenDataset
from model import LLaDAForMultiModalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-layers", type=int, nargs="+", required=True)
    parser.add_argument("--attention-loss-weight", type=float, choices=(0.0, 0.1), required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=4.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().clone()
    dist.all_reduce(value)
    return value / dist.get_world_size()


def named_gradient_norm(model, fragment: str, device: torch.device) -> torch.Tensor:
    squared = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        if fragment in name and ".lora_" in name and parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum()
    return squared.sqrt()


def save_checkpoint(output: Path, step: int, model, optimizer, scheduler, rank: int):
    destination = output / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(lora_state_dict(model.module), destination / "lora.pt")
        (destination / "lora_config.json").write_text(
            json.dumps({"step": step, "targets": ["q_proj", "k_proj", "v_proj", "attn_out"]}, indent=2) + "\n"
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
        },
        destination / f"training_state.rank{rank:02d}.pt",
    )
    dist.barrier()


def restore_checkpoint(checkpoint: Path, model, optimizer, scheduler, rank: int, max_steps: int):
    lora = torch.load(checkpoint / "lora.pt", map_location="cpu", weights_only=True)
    load_lora_state_dict(model, lora)
    state = torch.load(checkpoint / f"training_state.rank{rank:02d}.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if step >= max_steps:
        raise ValueError(f"resume step {step} must be below max_steps {max_steps}")

    # LambdaLR serializes its progress but not its lambda closure. Rebuilding the
    # closure with max_steps extends the completed smoke schedule to the long-run
    # horizon while preserving optimizer moments and scheduler/global step.
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
        "data_cursor_strategy": "derived_from_global_step",
    }


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if model.config.block_type != "llama" or model.config.n_layers != 32:
        raise AssertionError(f"unexpected runtime architecture: {model.config.block_type}, {model.config.n_layers}")
    report = inject_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.model.set_activation_checkpointing("whole_layer")
    model.to(device)

    dataset = MagicBrushTokenDataset(
        args.train_manifest,
        tokenizer,
        condition_dropout=args.condition_dropout,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda rows: rows,
        pin_memory=True,
        drop_last=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    step = 0
    resume_report = None
    if args.resume_from_checkpoint is not None:
        step, resume_report = restore_checkpoint(
            args.resume_from_checkpoint, model, optimizer, scheduler, rank, args.max_steps
        )
    resume_start_step = step
    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    if rank == 0:
        (args.output / "lora_report.json").write_text(
            json.dumps(
                {
                    "matched_modules": report.matched_modules,
                    "trainable_names": report.trainable_names,
                    "trainable_parameters": report.trainable_parameters,
                    "effective_batch": args.batch_size * args.gradient_accumulation * world_size,
                },
                indent=2,
            )
            + "\n"
        )
        if resume_report is not None:
            (args.output / "resume_report.json").write_text(json.dumps(resume_report, indent=2) + "\n")
        print(json.dumps({"layers": args.attention_layers, "matched": len(report.matched_modules), "trainable": report.trainable_parameters, "resume": resume_report}))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = step * args.gradient_accumulation
    accumulated_metrics = torch.zeros(9, device=device, dtype=torch.float32)
    accumulated_layer_metrics = torch.zeros(len(args.attention_layers), 5, device=device, dtype=torch.float32)
    batches_per_epoch = len(loader)
    consumed_batches = micro_step
    epoch = consumed_batches // batches_per_epoch
    resume_batch_in_epoch = consumed_batches % batches_per_epoch
    start_time = time.time()
    log_path = args.output / "train_metrics.jsonl"
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch_index, rows in enumerate(loader):
            if epoch == consumed_batches // batches_per_epoch and batch_index < resume_batch_in_epoch:
                continue
            examples = [row["input_ids"] for row in rows]
            labels = [row["labels"] for row in rows]
            masks = {
                "instruction_token_mask": [row["instruction_token_mask"] for row in rows],
                "source_spatial_mask": [row["source_spatial_mask"] for row in rows],
                "source_edit_mask": [row["source_edit_mask"] for row in rows],
                "attention_active": [row["attention_active"] for row in rows],
            }
            boundary = (micro_step + 1) % args.gradient_accumulation == 0
            sync_context = contextlib.nullcontext() if boundary else model.no_sync()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                generation_loss, auxiliary_by_layer = model(
                    input_ids=examples,
                    labels=labels,
                    attention_supervision_layers=args.attention_layers,
                    **masks,
                )
                auxiliary = auxiliary_by_layer.mean(0)
                local_active = auxiliary[4].detach()
                global_active = local_active.clone()
                dist.all_reduce(global_active)
                global_aux_metrics = auxiliary[:4].detach().float() * local_active.float()
                dist.all_reduce(global_aux_metrics)
                if global_active.item() > 0:
                    attention_for_backward = auxiliary[0] * local_active * world_size / global_active
                    global_aux_metrics /= global_active
                else:
                    attention_for_backward = auxiliary[0] * 0.0
                    global_aux_metrics.zero_()
                weighted_attention = args.attention_loss_weight * attention_for_backward
                total_loss = generation_loss + weighted_attention
                (total_loss / args.gradient_accumulation).backward()
            accumulated_metrics += torch.tensor(
                [
                    generation_loss.detach().float().item(),
                    global_aux_metrics[0].item(),
                    global_aux_metrics[1].item(),
                    global_aux_metrics[2].item(),
                    global_aux_metrics[3].item(),
                    sum(row["conditional"] for row in rows),
                    sum(not row["conditional"] for row in rows),
                    sum(row["empty_mask"] for row in rows),
                    sum(row["attention_active"] for row in rows),
                ],
                device=device,
            )
            layer_values = auxiliary_by_layer.detach().float()
            accumulated_layer_metrics[:, :4] += layer_values[:, :4] * layer_values[:, 4:5]
            accumulated_layer_metrics[:, 4] += layer_values[:, 4]
            micro_step += 1
            if not boundary:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            q_lora_grad_norm = named_gradient_norm(model, ".q_proj.", device)
            k_lora_grad_norm = named_gradient_norm(model, ".k_proj.", device)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = accumulated_metrics.clone()
            dist.all_reduce(metrics)
            layer_metrics = accumulated_layer_metrics.clone()
            dist.all_reduce(layer_metrics)
            layer_metrics[:, :4] /= layer_metrics[:, 4:5].clamp_min(1.0)
            metric_denominator = world_size * args.gradient_accumulation
            metrics[:5] /= metric_denominator
            global_grad_norm = reduce_mean(grad_norm.detach())
            global_q_lora_grad_norm = reduce_mean(q_lora_grad_norm)
            global_k_lora_grad_norm = reduce_mean(k_lora_grad_norm)
            accumulated_metrics.zero_()
            accumulated_layer_metrics.zero_()
            if rank == 0:
                record = {
                    "step": step,
                    "L_gen": metrics[0].item(),
                    "L_attn_raw": metrics[1].item(),
                    "weighted_attn_loss": args.attention_loss_weight * metrics[1].item(),
                    "L_total": metrics[0].item() + args.attention_loss_weight * metrics[1].item(),
                    "attn_to_gen_ratio": args.attention_loss_weight * metrics[1].item() / max(metrics[0].item(), 1e-8),
                    "conditional_localization_mass": metrics[2].item(),
                    "attention_entropy": metrics[3].item(),
                    "actual_full_attention_edit_mask_mass": metrics[4].item(),
                    "conditional_count": int(metrics[5].item()),
                    "unconditional_count": int(metrics[6].item()),
                    "empty_mask_count": int(metrics[7].item()),
                    "attn_supervised_count": int(metrics[8].item()),
                    "grad_norm": global_grad_norm.item(),
                    "q_lora_grad_norm": global_q_lora_grad_norm.item(),
                    "k_lora_grad_norm": global_k_lora_grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds_per_step": (time.time() - start_time) / max(step - resume_start_step, 1),
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    "per_layer": {
                        str(layer): {
                            "conditional_spatial_ce": layer_metrics[index, 0].item(),
                            "conditional_localization_mass": layer_metrics[index, 1].item(),
                            "attention_entropy": layer_metrics[index, 2].item(),
                            "actual_full_attention_edit_mask_mass": layer_metrics[index, 3].item(),
                            "active_sample_count": int(layer_metrics[index, 4].item()),
                        }
                        for index, layer in enumerate(args.attention_layers)
                    },
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            should_save = step in set(args.checkpoint_steps)
            should_save = should_save or (args.save_steps > 0 and step % args.save_steps == 0)
            if should_save or step == args.max_steps:
                save_checkpoint(args.output, step, model, optimizer, scheduler, rank)
            if step >= args.max_steps:
                break
        epoch += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
