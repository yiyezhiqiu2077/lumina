#!/usr/bin/env python3
"""Unified, mutually-exclusive CE / attention / GCE MagicBrush trainer."""
from __future__ import annotations

import argparse
import contextlib
import json
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

from datasets.magicbrush_tokens import MagicBrushTokenDataset
from model import LLaDAForMultiModalGeneration
from training.lora import inject_lora, load_lora_state_dict, lora_state_dict
from training.objective_dispatch import OBJECTIVE_MODES, compose_total_loss, run_model_for_objective


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVE_MODES, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
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
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--attention-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--attention-loss-weight", type=float, default=0.1)
    parser.add_argument("--gce-clusters", type=Path)
    parser.add_argument("--gce-weight", type=float, default=1.0)
    parser.add_argument("--gce-levels", type=int, nargs="+", default=[1024, 512])
    args = parser.parse_args()
    if args.objective == "gce" and args.gce_clusters is None:
        parser.error("--gce-clusters is required only for --objective gce")
    if args.objective != "gce" and args.gce_clusters is not None:
        parser.error("--gce-clusters is only valid for --objective gce")
    if args.objective != "attention" and args.attention_loss_weight != 0.1:
        parser.error("--attention-loss-weight is only valid for --objective attention")
    return args


def seed_all(seed: int) -> None:
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


def _json_args(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def objective_banner(args) -> dict:
    banner = {
        "mode": args.objective,
        "generation_ce": "ON",
        "attention_supervision": "ON" if args.objective == "attention" else "OFF",
        "gce": "ON" if args.objective == "gce" else "OFF",
    }
    if args.objective == "attention":
        banner.update(attention_weight=args.attention_loss_weight, attention_layers=args.attention_layers)
    if args.objective == "gce":
        banner.update(gce_weight=args.gce_weight, cluster_levels=args.gce_levels)
    return banner


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


def _global_attention_loss(auxiliary_by_layer: torch.Tensor, world_size: int):
    """Preserve the existing per-sample/layer attention-loss reduction."""
    auxiliary = auxiliary_by_layer.mean(0)
    local_active = auxiliary[4].detach()
    global_active = local_active.clone()
    dist.all_reduce(global_active)
    global_metrics = auxiliary[:4].detach().float() * local_active.float()
    dist.all_reduce(global_metrics)
    if global_active.item() > 0:
        backward_loss = auxiliary[0] * local_active * world_size / global_active
        global_metrics /= global_active
    else:
        backward_loss = auxiliary[0] * 0.0
        global_metrics.zero_()
    return backward_loss, global_metrics, auxiliary


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        banner = objective_banner(args)
        (args.output / "experiment_config.json").write_text(
            json.dumps({"objective": banner, "args": _json_args(args)}, indent=2) + "\n"
        )
        print("=" * 40)
        print("MagicBrush Training Objective")
        print("=" * 40)
        for key, value in banner.items():
            print(f"{key}: {value}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    if model.config.block_type != "llama" or model.config.n_layers != 32:
        raise AssertionError(f"unexpected runtime architecture: {model.config.block_type}, {model.config.n_layers}")
    report = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.model.set_activation_checkpointing("whole_layer")
    model.to(device)

    # Deliberately lazy: CE/attention never import, instantiate, or load GCE.
    gce_objective = None
    if args.objective == "gce":
        from objectives.gce import GCEObjective

        gce_objective = GCEObjective.from_clusters(str(args.gce_clusters), tuple(args.gce_levels)).to(device)

    dataset = MagicBrushTokenDataset(
        args.train_manifest,
        tokenizer,
        max_sequence_length=args.max_seq_len,
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
    if len(loader) == 0:
        raise ValueError("no full batches available; reduce batch size or provide more data")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    step = 0
    resume_report = None
    if args.resume_from_checkpoint is not None:
        step, resume_report = restore_checkpoint(
            args.resume_from_checkpoint, model, optimizer, scheduler, rank, args.max_steps, args.objective
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

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = step * args.gradient_accumulation
    batches_per_epoch = len(loader)
    consumed_batches = micro_step
    epoch = consumed_batches // batches_per_epoch
    resume_batch_in_epoch = consumed_batches % batches_per_epoch
    log_path = args.output / "train_metrics.jsonl"
    start_time = time.time()
    sums = {"L_gen": torch.zeros((), device=device), "L_total": torch.zeros((), device=device)}
    if args.objective == "attention":
        sums.update(L_attn_raw=torch.zeros((), device=device), conditional_localization_mass=torch.zeros((), device=device), attention_entropy=torch.zeros((), device=device), actual_full_attention_edit_mask_mass=torch.zeros((), device=device))
        layer_sums = torch.zeros(len(args.attention_layers), 5, device=device)
    else:
        layer_sums = None
    if args.objective == "gce":
        sums.update(gce_loss=torch.zeros((), device=device), **{f"gce_loss_k{level}": torch.zeros((), device=device) for level in args.gce_levels})

    while step < args.max_steps:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch_index, rows in enumerate(loader):
            if epoch == consumed_batches // batches_per_epoch and batch_index < resume_batch_in_epoch:
                continue
            examples = [row["input_ids"] for row in rows]
            labels = [row["labels"] for row in rows]
            attention_masks = None
            if args.objective == "attention":
                attention_masks = {
                    "instruction_token_mask": [row["instruction_token_mask"] for row in rows],
                    "source_spatial_mask": [row["source_spatial_mask"] for row in rows],
                    "source_edit_mask": [row["source_edit_mask"] for row in rows],
                    "attention_active": [row["attention_active"] for row in rows],
                }
            boundary = (micro_step + 1) % args.gradient_accumulation == 0
            sync_context = contextlib.nullcontext() if boundary else model.no_sync()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                result = run_model_for_objective(
                    model,
                    objective=args.objective,
                    input_ids=examples,
                    labels=labels,
                    attention_layers=args.attention_layers,
                    attention_masks=attention_masks,
                    gce_objective=gce_objective,
                )
                generation_loss = result.output.generation_loss
                if args.objective == "attention":
                    attention_for_backward, attention_metrics, auxiliary = _global_attention_loss(
                        result.attention_auxiliary, world_size
                    )
                    total_loss = compose_total_loss(
                        "attention", generation_loss, attention_loss=attention_for_backward,
                        attention_weight=args.attention_loss_weight,
                    )
                elif args.objective == "gce":
                    total_loss = compose_total_loss(
                        "gce", generation_loss, gce_loss=result.gce_loss, gce_weight=args.gce_weight
                    )
                else:
                    total_loss = compose_total_loss("ce", generation_loss)
                (total_loss / args.gradient_accumulation).backward()

            sums["L_gen"] += generation_loss.detach().float()
            sums["L_total"] += total_loss.detach().float()
            if args.objective == "attention":
                sums["L_attn_raw"] += attention_metrics[0]
                sums["conditional_localization_mass"] += attention_metrics[1]
                sums["attention_entropy"] += attention_metrics[2]
                sums["actual_full_attention_edit_mask_mass"] += attention_metrics[3]
                values = result.attention_auxiliary.detach().float()
                layer_sums[:, :4] += values[:, :4] * values[:, 4:5]
                layer_sums[:, 4] += values[:, 4]
            elif args.objective == "gce":
                sums["gce_loss"] += result.gce_loss.detach().float()
                for key, value in result.gce_metrics.items():
                    if key in sums and key != "gce_loss":
                        sums[key] += value.detach().float()

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

            metrics = {key: value.clone() for key, value in sums.items()}
            for value in metrics.values():
                dist.all_reduce(value)
                value /= world_size * args.gradient_accumulation
            for value in sums.values():
                value.zero_()
            if layer_sums is not None:
                layers = layer_sums.clone()
                dist.all_reduce(layers)
                layers[:, :4] /= layers[:, 4:5].clamp_min(1.0)
                layer_sums.zero_()
            else:
                layers = None
            global_grad_norm = reduce_mean(grad_norm)
            global_q_lora_grad_norm = reduce_mean(q_lora_grad_norm)
            global_k_lora_grad_norm = reduce_mean(k_lora_grad_norm)
            if rank == 0:
                record = {
                    "step": step,
                    **{key: value.item() for key, value in metrics.items()},
                    "objective": args.objective,
                    "grad_norm": global_grad_norm.item(),
                    "q_lora_grad_norm": global_q_lora_grad_norm.item(),
                    "k_lora_grad_norm": global_k_lora_grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds_per_step": (time.time() - start_time) / max(step - resume_start_step, 1),
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                }
                if args.objective == "attention":
                    record.update(
                        weighted_attn_loss=args.attention_loss_weight * metrics["L_attn_raw"].item(),
                        attn_to_gen_ratio=args.attention_loss_weight * metrics["L_attn_raw"].item() / max(metrics["L_gen"].item(), 1e-8),
                        per_layer={
                            str(layer): {
                                "conditional_spatial_ce": layers[index, 0].item(),
                                "conditional_localization_mass": layers[index, 1].item(),
                                "attention_entropy": layers[index, 2].item(),
                                "actual_full_attention_edit_mask_mass": layers[index, 3].item(),
                                "active_sample_count": int(layers[index, 4].item()),
                            }
                            for index, layer in enumerate(args.attention_layers)
                        },
                    )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            should_save = step in set(args.checkpoint_steps)
            should_save = should_save or (args.save_steps > 0 and step % args.save_steps == 0)
            if should_save or step == args.max_steps:
                save_checkpoint(args.output, step, model, optimizer, scheduler, rank, args.objective)
            if step >= args.max_steps:
                break
        epoch += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
