#!/usr/bin/env python3
"""Unified, mutually-exclusive CE / attention / GCE MagicBrush trainer."""
from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import EditTokenDataset
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration
from training.checkpoint import (
    build_run_fingerprint,
    restore_checkpoint,
    save_checkpoint,
)
from training.config import launch_summary
from training.lora import inject_lora
from training.objective import OBJECTIVE_MODES, compose_total_loss, run_model_for_objective
from training.quality import QualityGateError, TrainingQualityGate, write_quality_status


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().clone()
    dist.all_reduce(value)
    return value / dist.get_world_size()


def reduce_max(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().clone()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def named_gradient_norm(model, fragment: str, device: torch.device) -> torch.Tensor:
    squared = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        if fragment in name and ".lora_" in name and parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum()
    return squared.sqrt()


def parameter_group_norms(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    targets: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_squared = torch.zeros((), device=device)
    target_squared: dict[str, torch.Tensor] = {}
    for name, parameter in named_parameters:
        value = parameter.detach().float().pow(2).sum()
        total_squared += value
        for target in targets:
            if f".{target}." in name:
                target_squared.setdefault(target, torch.zeros((), device=device))
                target_squared[target] += value
    return total_squared.sqrt(), {
        target: value.sqrt() for target, value in target_squared.items()
    }


def gradient_decomposition(
    generation_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Measure globally reduced component gradients without changing ``.grad``."""
    generation_gradients = torch.autograd.grad(
        generation_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    generation_squared = torch.zeros((), device=device)
    auxiliary_squared = torch.zeros((), device=device)
    dot = torch.zeros((), device=device)
    world_size = dist.get_world_size()
    for generation_gradient, auxiliary_gradient in zip(
        generation_gradients,
        auxiliary_gradients,
    ):
        if generation_gradient is not None:
            generation_gradient = generation_gradient.detach().float()
            dist.all_reduce(generation_gradient)
            generation_gradient /= world_size
            generation_squared += generation_gradient.pow(2).sum()
        if auxiliary_gradient is not None:
            auxiliary_gradient = auxiliary_gradient.detach().float()
            dist.all_reduce(auxiliary_gradient)
            auxiliary_gradient /= world_size
            auxiliary_squared += auxiliary_gradient.pow(2).sum()
        if generation_gradient is not None and auxiliary_gradient is not None:
            dot += (generation_gradient * auxiliary_gradient).sum()
    generation_norm = generation_squared.sqrt()
    auxiliary_norm = auxiliary_squared.sqrt()
    cosine = dot / (generation_norm * auxiliary_norm).clamp_min(1e-12)
    return {
        "generation_gradient_norm": generation_norm,
        "auxiliary_gradient_norm": auxiliary_norm,
        "auxiliary_to_generation_gradient_ratio": auxiliary_norm / generation_norm.clamp_min(1e-12),
        "gradient_cosine": cosine,
    }


def snapshot_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }


def update_norm_and_finiteness(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    update_squared = torch.zeros((), device=device)
    finite = torch.ones((), device=device, dtype=torch.int32)
    for name, parameter in named_parameters:
        current = parameter.detach()
        update_squared += (current.float() - before[name].float()).pow(2).sum()
        if not bool(torch.isfinite(current).all()):
            finite.zero_()
    return update_squared.sqrt(), finite


def _json_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def objective_banner(args) -> dict:
    banner = {
        "mode": args.objective,
        "generation_ce": "ON",
        "attention_supervision": "ON" if args.objective == "attention" else "OFF",
        "gce": "ON" if args.objective == "gce" else "OFF",
        "loss_reduction": args.loss_reduction,
        "z_loss_weight": args.z_loss_weight,
    }
    if args.objective == "attention":
        banner.update(
            attention_weight=args.attention_loss_weight,
            attention_layers=args.attention_layers,
        )
    if args.objective == "gce":
        banner.update(gce_weight=args.gce_weight, cluster_levels=args.gce_levels)
    return banner


def distributed_worker_environment_present() -> bool:
    return all(os.environ.get(name) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def build_torchrun_command(
    args,
    worker_script: Path,
    config_path: Path,
    resume_from_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(worker_script),
        "--config",
        str(config_path),
        "--distributed-worker",
    ]
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    return command


def launch_training(
    args,
    worker_script: Path,
    config_path: Path,
    resume_from_checkpoint: Path | None,
) -> None:
    print(json.dumps(launch_summary(args), sort_keys=True), flush=True)
    subprocess.run(
        build_torchrun_command(args, worker_script, config_path, resume_from_checkpoint),
        check=True,
    )


def _global_attention_loss(auxiliary_by_layer: torch.Tensor, world_size: int):
    """Preserve the existing per-sample/layer attention-loss reduction."""
    auxiliary = auxiliary_by_layer.mean(0)
    local_active = auxiliary[4].detach()
    global_active = local_active.clone()
    dist.all_reduce(global_active)
    # Keep the complete auxiliary layout for logging.  Entries 0--3 and 5--7
    # are per-active-sample means, while entry 4 is the active-sample count.
    # Reducing only the first four entries makes the later enrichment / entropy
    # logging index an invalid four-element tensor.
    global_metrics = auxiliary.detach().float().clone()
    global_metrics[:4] *= local_active.float()
    global_metrics[5:] *= local_active.float()
    dist.all_reduce(global_metrics)
    if global_active.item() > 0:
        backward_loss = auxiliary[0] * local_active * world_size / global_active
        global_metrics[:4] /= global_active
        global_metrics[5:] /= global_active
        global_metrics[4] = global_active
    else:
        backward_loss = auxiliary[0] * 0.0
        global_metrics.zero_()
    return backward_loss, global_metrics, auxiliary


def build_scheduler(optimizer, args):
    """Build the schedule independently from a bounded run's stop horizon."""
    return get_cosine_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        args.scheduler_horizon_steps,
    )


def _quality_gate(args) -> TrainingQualityGate:
    return TrainingQualityGate(
        baseline_start=args.quality_baseline_start,
        baseline_end=args.quality_baseline_end,
        window_size=args.quality_window_size,
        max_loss_ratio=args.quality_max_loss_ratio,
        consecutive_windows=args.quality_consecutive_windows,
        max_clip_fraction=args.quality_max_clip_fraction,
        max_abs_logit=args.quality_max_abs_logit,
        max_update_ratio=args.quality_max_update_ratio,
    )


def _quality_failure_message(local_message: str | None, device: torch.device) -> str | None:
    local_failed = torch.tensor(int(local_message is not None), device=device)
    dist.all_reduce(local_failed, op=dist.ReduceOp.MAX)
    if not local_failed.item():
        return None
    messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(messages, local_message)
    return next(message for message in messages if message is not None)


def run(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.nproc_per_node:
        raise RuntimeError(
            f"runtime world size {world_size} does not match configured "
            f"nproc_per_node {args.nproc_per_node}"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    runtime_dtype = torch.bfloat16
    seed_all(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    quality_path = args.output / "quality_status.json"
    if rank == 0:
        banner = objective_banner(args)
        config_path = args.output / "experiment_config.json"
        if args.resume_from_checkpoint is None:
            if config_path.exists():
                raise FileExistsError(f"fresh run metadata already exists: {config_path}")
            config_path.write_text(
                json.dumps({"objective": banner, "args": _json_args(args)}, indent=2) + "\n",
                encoding="utf-8",
            )
        write_quality_status(quality_path, "RUNNING", objective=args.objective, step=0)
        print("=" * 40)
        print("MagicBrush Training Objective")
        print("=" * 40)
        for key, value in banner.items():
            print(f"{key}: {value}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model,
        torch_dtype=runtime_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if model.config.block_type != "llama" or model.config.n_layers != 32:
        raise AssertionError(
            f"unexpected runtime architecture: {model.config.block_type}, {model.config.n_layers}"
        )
    report = inject_lora(
        model,
        targets=args.lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.model.set_activation_checkpointing("whole_layer")
    model.to(device)

    gce_objective = None
    if args.objective == "gce":
        from models.objectives.gce import GCEObjective

        gce_objective = GCEObjective.from_clusters(
            str(args.gce_clusters),
            tuple(args.gce_levels),
        ).to(device)

    dataset = EditTokenDataset(
        args.train_manifest,
        tokenizer,
        max_sequence_length=args.max_seq_len,
        condition_dropout=args.condition_dropout,
        seed=args.seed,
    )
    dataset_composition = Counter(
        str(row.get("dataset_name", "magicbrush")) for row in dataset.rows
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
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=args.optimizer_betas,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    run_fingerprint = build_run_fingerprint(args, world_size)
    gate = _quality_gate(args) if args.quality_gate_enabled else None
    log_path = args.output / "train_metrics.jsonl"
    step = 0
    resume_report = None
    if args.resume_from_checkpoint is not None:
        step, resume_report, quality_state = restore_checkpoint(
            args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            rank,
            args.max_steps,
            args.objective,
            run_fingerprint,
            metrics_path=log_path,
        )
        if gate is not None:
            if quality_state is None:
                raise ValueError("quality-gated resume checkpoint has no quality state")
            gate.load_state_dict(quality_state)
    resume_start_step = step
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    if rank == 0:
        report_path = args.output / "lora_report.json"
        if args.resume_from_checkpoint is None:
            report_path.write_text(
                json.dumps(
                    {
                        "matched_modules": report.matched_modules,
                        "trainable_names": report.trainable_names,
                        "trainable_parameters": report.trainable_parameters,
                        "effective_batch": args.batch_size * args.gradient_accumulation * world_size,
                        "targets": args.lora_targets,
                        "run_fingerprint": run_fingerprint,
                        "dataset_composition": dict(sorted(dataset_composition.items())),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "dataset_sample_count": len(dataset),
                    "dataset_composition": dict(sorted(dataset_composition.items())),
                    "optimizer_steps_per_epoch": args.optimizer_steps_per_epoch,
                }
            ),
            flush=True,
        )
        if resume_report is not None:
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            (args.output / f"resume_report.{timestamp}.json").write_text(
                json.dumps(resume_report, indent=2) + "\n",
                encoding="utf-8",
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = step * args.gradient_accumulation
    batches_per_epoch = len(loader)
    consumed_batches = micro_step
    epoch = consumed_batches // batches_per_epoch
    resume_batch_in_epoch = consumed_batches % batches_per_epoch
    start_time = time.time()
    sums = {
        "L_gen": torch.zeros((), device=device),
        "L_z_raw": torch.zeros((), device=device),
        "L_total": torch.zeros((), device=device),
        "valid_target_count": torch.zeros((), device=device),
        "max_abs_logit": torch.zeros((), device=device),
    }
    if args.objective == "attention":
        sums.update(
            L_attn_raw=torch.zeros((), device=device),
            conditional_localization_mass=torch.zeros((), device=device),
            attention_entropy=torch.zeros((), device=device),
            actual_full_attention_edit_mask_mass=torch.zeros((), device=device),
            enrichment=torch.zeros((), device=device),
            mask_entropy=torch.zeros((), device=device),
            effective_KL=torch.zeros((), device=device),
        )
        layer_sums = torch.zeros(len(args.attention_layers), 8, device=device)
    else:
        layer_sums = None
    if args.objective == "gce":
        sums.update(
            gce_loss=torch.zeros((), device=device),
            **{
                f"gce_loss_k{level}": torch.zeros((), device=device)
                for level in args.gce_levels
            },
        )

    try:
        while step < args.max_steps:
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            for batch_index, rows in enumerate(loader):
                if (
                    epoch == consumed_batches // batches_per_epoch
                    and batch_index < resume_batch_in_epoch
                ):
                    continue
                examples = [row["input_ids"] for row in rows]
                labels = [row["labels"] for row in rows]
                attention_masks = None
                if args.objective == "attention":
                    attention_masks = {
                        "instruction_token_mask": [
                            row["instruction_token_mask"] for row in rows
                        ],
                        "source_spatial_mask": [row["source_spatial_mask"] for row in rows],
                        "source_edit_mask": [row["source_edit_mask"] for row in rows],
                        "attention_active": [row["attention_active"] for row in rows],
                    }
                boundary = (micro_step + 1) % args.gradient_accumulation == 0
                optimizer_step = micro_step // args.gradient_accumulation + 1
                diagnostic_boundary = (
                    args.objective != "ce"
                    and boundary
                    and args.gradient_accumulation == 1
                    and args.gradient_decomposition_every_steps > 0
                    and optimizer_step % args.gradient_decomposition_every_steps == 0
                )
                sync_context = (
                    contextlib.nullcontext()
                    if boundary and not diagnostic_boundary
                    else model.no_sync()
                )
                with sync_context, torch.autocast("cuda", dtype=runtime_dtype):
                    result = run_model_for_objective(
                        model,
                        objective=args.objective,
                        input_ids=examples,
                        labels=labels,
                        loss_reduction=args.loss_reduction,
                        attention_layers=args.attention_layers,
                        attention_qk_stage=args.attention_qk_stage,
                        attention_loss_mode=args.attention_loss_mode,
                        attention_masks=attention_masks,
                        gce_objective=gce_objective,
                    )
                    generation_loss = result.output.generation_loss
                    generation_z_loss = result.output.generation_z_loss
                    if generation_z_loss is None:
                        raise RuntimeError("model did not return generation_z_loss")
                    if args.objective == "attention":
                        attention_for_backward, attention_metrics, auxiliary = _global_attention_loss(
                            result.attention_auxiliary,
                            world_size,
                        )
                        weighted_auxiliary_loss = (
                            args.attention_loss_weight * attention_for_backward
                        )
                        total_loss = compose_total_loss(
                            "attention",
                            generation_loss,
                            generation_z_loss=generation_z_loss,
                            z_loss_weight=args.z_loss_weight,
                            attention_loss=attention_for_backward,
                            attention_weight=args.attention_loss_weight,
                        )
                    elif args.objective == "gce":
                        weighted_auxiliary_loss = args.gce_weight * result.gce_loss
                        total_loss = compose_total_loss(
                            "gce",
                            generation_loss,
                            generation_z_loss=generation_z_loss,
                            z_loss_weight=args.z_loss_weight,
                            gce_loss=result.gce_loss,
                            gce_weight=args.gce_weight,
                        )
                    else:
                        weighted_auxiliary_loss = None
                        total_loss = compose_total_loss(
                            "ce",
                            generation_loss,
                            generation_z_loss=generation_z_loss,
                            z_loss_weight=args.z_loss_weight,
                        )
                    decomposition = None
                    if diagnostic_boundary:
                        shared_generation_loss = (
                            generation_loss
                            + args.z_loss_weight * generation_z_loss
                        )
                        decomposition = gradient_decomposition(
                            shared_generation_loss,
                            weighted_auxiliary_loss,
                            trainable,
                            device,
                        )
                    (total_loss / args.gradient_accumulation).backward()
                    if diagnostic_boundary:
                        for parameter in trainable:
                            if parameter.grad is not None:
                                dist.all_reduce(parameter.grad)
                                parameter.grad /= world_size

                sums["L_gen"] += generation_loss.detach().float()
                sums["L_z_raw"] += generation_z_loss.detach().float()
                sums["L_total"] += total_loss.detach().float()
                sums["valid_target_count"] += result.output.valid_target_count.detach().float()
                sums["max_abs_logit"] = torch.maximum(
                    sums["max_abs_logit"],
                    result.output.max_abs_valid_logit.detach().float(),
                )
                if args.objective == "attention":
                    sums["L_attn_raw"] += attention_metrics[0]
                    sums["conditional_localization_mass"] += attention_metrics[1]
                    sums["attention_entropy"] += attention_metrics[2]
                    sums["actual_full_attention_edit_mask_mass"] += attention_metrics[3]
                    sums["enrichment"] += attention_metrics[5]
                    sums["mask_entropy"] += attention_metrics[6]
                    sums["effective_KL"] += attention_metrics[7]
                    values = result.attention_auxiliary.detach().float()
                    layer_sums[:, :4] += values[:, :4] * values[:, 4:5]
                    layer_sums[:, 5:] += values[:, 5:] * values[:, 4:5]
                    layer_sums[:, 4] += values[:, 4]
                elif args.objective == "gce":
                    sums["gce_loss"] += result.gce_loss.detach().float()
                    for key, value in result.gce_metrics.items():
                        if key in sums and key != "gce_loss":
                            sums[key] += value.detach().float()

                micro_step += 1
                if not boundary:
                    continue
                next_step = step + 1
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                    error_if_nonfinite=False,
                )
                clipped = pre_clip_grad_norm.detach() > args.max_grad_norm
                post_clip_grad_norm = torch.minimum(
                    pre_clip_grad_norm.detach().float(),
                    torch.tensor(float(args.max_grad_norm), device=device),
                )
                q_lora_grad_norm = named_gradient_norm(model, ".q_proj.", device)
                k_lora_grad_norm = named_gradient_norm(model, ".k_proj.", device)
                parameter_norm_before, target_norms_before = parameter_group_norms(
                    named_trainable,
                    args.lora_targets,
                    device,
                )
                capture_update = (
                    next_step == 1
                    or next_step % args.diagnostic_every_steps == 0
                    or next_step % args.save_steps == 0
                    or next_step == args.max_steps
                )
                before = snapshot_parameters(named_trainable) if capture_update else None
                optimizer.step()
                scheduler.step()
                if before is not None:
                    update_norm, local_parameters_finite = update_norm_and_finiteness(
                        named_trainable,
                        before,
                        device,
                    )
                    update_ratio = update_norm / parameter_norm_before.clamp_min(1e-12)
                else:
                    update_norm = torch.tensor(float("nan"), device=device)
                    update_ratio = torch.tensor(float("nan"), device=device)
                    local_parameters_finite = torch.ones((), device=device, dtype=torch.int32)
                optimizer.zero_grad(set_to_none=True)
                step = next_step

                metrics = {key: value.clone() for key, value in sums.items()}
                for key, value in metrics.items():
                    if key == "max_abs_logit":
                        dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    elif key == "valid_target_count":
                        dist.all_reduce(value)
                    else:
                        dist.all_reduce(value)
                        value /= world_size * args.gradient_accumulation
                for value in sums.values():
                    value.zero_()
                if layer_sums is not None:
                    layers = layer_sums.clone()
                    dist.all_reduce(layers)
                    layers[:, :4] /= layers[:, 4:5].clamp_min(1.0)
                    layers[:, 5:] /= layers[:, 4:5].clamp_min(1.0)
                    layer_sums.zero_()
                else:
                    layers = None
                global_pre_clip_grad_norm = reduce_max(pre_clip_grad_norm)
                global_post_clip_grad_norm = reduce_max(post_clip_grad_norm)
                global_q_lora_grad_norm = reduce_mean(q_lora_grad_norm)
                global_k_lora_grad_norm = reduce_mean(k_lora_grad_norm)
                global_parameter_norm = reduce_mean(parameter_norm_before)
                global_target_norms = {
                    target: reduce_mean(value)
                    for target, value in target_norms_before.items()
                }
                global_decomposition = decomposition
                global_clipped = reduce_max(clipped.float())
                if before is not None:
                    global_update_norm = reduce_mean(update_norm)
                    global_update_ratio = reduce_max(update_ratio)
                else:
                    global_update_norm = update_norm
                    global_update_ratio = update_ratio
                dist.all_reduce(local_parameters_finite, op=dist.ReduceOp.MIN)
                parameters_finite = bool(local_parameters_finite.item())

                local_quality_error = None
                quality_metrics: dict[str, float | int | bool | None] = {}
                if gate is not None:
                    try:
                        quality_metrics = gate.observe(
                            step=step,
                            generation_loss=metrics["L_gen"].item(),
                            grad_norm=global_pre_clip_grad_norm.item(),
                            max_abs_logit=metrics["max_abs_logit"].item(),
                            update_ratio=(
                                global_update_ratio.item()
                                if before is not None
                                else None
                            ),
                            clipped=bool(global_clipped.item()),
                            parameters_finite=parameters_finite,
                        )
                    except QualityGateError as error:
                        local_quality_error = str(error)
                quality_error = _quality_failure_message(local_quality_error, device)

                if rank == 0:
                    record = {
                        "step": step,
                        **{key: value.item() for key, value in metrics.items()},
                        "weighted_z_loss": args.z_loss_weight * metrics["L_z_raw"].item(),
                        "objective": args.objective,
                        "grad_norm": global_pre_clip_grad_norm.item(),
                        "pre_clip_grad_norm": global_pre_clip_grad_norm.item(),
                        "post_clip_grad_norm": global_post_clip_grad_norm.item(),
                        "gradient_clipped": bool(global_clipped.item()),
                        "q_lora_grad_norm": global_q_lora_grad_norm.item(),
                        "k_lora_grad_norm": global_k_lora_grad_norm.item(),
                        "lora_parameter_norm": global_parameter_norm.item(),
                        "lora_parameter_norm_by_target": {
                            key: value.item() for key, value in global_target_norms.items()
                        },
                        "update_norm": (
                            global_update_norm.item() if before is not None else None
                        ),
                        "update_ratio": (
                            global_update_ratio.item() if before is not None else None
                        ),
                        "parameters_finite": parameters_finite,
                        "gradient_decomposition": (
                            {
                                key: value.item()
                                for key, value in global_decomposition.items()
                            }
                            if global_decomposition is not None
                            else None
                        ),
                        "lr": scheduler.get_last_lr()[0],
                        "seconds_per_step": (time.time() - start_time)
                        / max(step - resume_start_step, 1),
                        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                        **quality_metrics,
                    }
                    if args.objective == "attention":
                        record.update(
                            weighted_attn_loss=args.attention_loss_weight
                            * metrics["L_attn_raw"].item(),
                            attn_to_gen_ratio=args.attention_loss_weight
                            * metrics["L_attn_raw"].item()
                            / max(metrics["L_gen"].item(), 1e-8),
                            P_G=metrics["conditional_localization_mass"].item(),
                            enrichment=metrics["enrichment"].item(),
                            mask_entropy=metrics["mask_entropy"].item(),
                            effective_KL=(
                                metrics["effective_KL"].item()
                                if args.attention_loss_mode == "normalized_mask_ce"
                                else None
                            ),
                            attention_qk_stage=args.attention_qk_stage,
                            attention_loss_mode=args.attention_loss_mode,
                            per_layer={
                                str(layer): {
                                    "conditional_spatial_ce": layers[index, 0].item(),
                                    "conditional_localization_mass": layers[index, 1].item(),
                                    "attention_entropy": layers[index, 2].item(),
                                    "actual_full_attention_edit_mask_mass": layers[index, 3].item(),
                                    "active_sample_count": int(layers[index, 4].item()),
                                    "enrichment": layers[index, 5].item(),
                                    "mask_entropy": layers[index, 6].item(),
                                    "effective_KL": (
                                        layers[index, 7].item()
                                        if args.attention_loss_mode == "normalized_mask_ce"
                                        else None
                                    ),
                                }
                                for index, layer in enumerate(args.attention_layers)
                            },
                        )
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    if quality_error is not None:
                        write_quality_status(
                            quality_path,
                            "QUALITY_FAILED",
                            objective=args.objective,
                            step=step,
                            reason=quality_error,
                        )
                if quality_error is not None:
                    raise QualityGateError(quality_error)

                should_save = step in set(args.checkpoint_steps)
                should_save = should_save or (
                    args.save_steps > 0 and step % args.save_steps == 0
                )
                if should_save or step == args.max_steps:
                    save_checkpoint(
                        args.output,
                        step,
                        model,
                        optimizer,
                        scheduler,
                        rank,
                        args.objective,
                        run_fingerprint,
                        gate.state_dict() if gate is not None else None,
                    )
                if step >= args.max_steps:
                    break
            epoch += 1
        if rank == 0:
            write_quality_status(
                quality_path,
                "SUCCEEDED",
                objective=args.objective,
                step=step,
                baseline=gate.baseline if gate is not None else None,
                fingerprint_digest=run_fingerprint["digest"],
            )
    except BaseException as error:
        if rank == 0 and not isinstance(error, QualityGateError):
            write_quality_status(
                quality_path,
                "PROCESS_FAILED",
                objective=args.objective,
                step=step,
                reason=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        dist.destroy_process_group()
