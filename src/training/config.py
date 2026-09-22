"""YAML loading and validation for the canonical MagicBrush train entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _read_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def validate_train_config(args: argparse.Namespace) -> None:
    expected = args.nproc_per_node * args.batch_size * args.gradient_accumulation
    if expected != args.global_batch_size:
        raise ValueError(
            "distributed.global_batch_size mismatch: "
            f"{args.nproc_per_node} * {args.batch_size} * {args.gradient_accumulation} "
            f"= {expected}, declared {args.global_batch_size}"
        )
    if args.vq_grid != [32, 32]:
        raise ValueError(f"MagicBrush vq_grid must be [32, 32], got {args.vq_grid}")
    if args.launcher != "torchrun" or args.rdzv != "standalone":
        raise ValueError(f"unsupported single-node launcher policy: launcher={args.launcher!r}, rdzv={args.rdzv!r}")
    if args.precision != "bf16":
        raise ValueError(f"only the validated bf16 runtime is supported, got {args.precision!r}")
    loss_reduction = getattr(args, "loss_reduction", "sample_mean")
    if loss_reduction not in ("sample_mean", "token_mean"):
        raise ValueError(f"unsupported loss reduction: {loss_reduction!r}")
    optimizer_betas = getattr(args, "optimizer_betas", (0.9, 0.95))
    if len(optimizer_betas) != 2 or not all(0.0 <= value < 1.0 for value in optimizer_betas):
        raise ValueError(f"optimizer betas must contain two values in [0, 1), got {optimizer_betas}")
    lora_targets = getattr(args, "lora_targets", ("q_proj", "k_proj", "v_proj", "attn_out"))
    if not lora_targets or len(set(lora_targets)) != len(lora_targets):
        raise ValueError(f"LoRA targets must be non-empty and unique, got {lora_targets}")
    if getattr(args, "diagnostic_every_steps", 1) <= 0:
        raise ValueError("diagnostic every_steps must be positive")
    if getattr(args, "gradient_decomposition_every_steps", 0) < 0:
        raise ValueError("gradient decomposition cadence cannot be negative")
    if getattr(args, "z_loss_weight", 0.0) < 0:
        raise ValueError("z_loss_weight cannot be negative")
    if getattr(args, "attention_qk_stage", "post_rope") not in ("pre_rope", "post_rope"):
        raise ValueError("attention_qk_stage must be pre_rope or post_rope")
    if getattr(args, "attention_loss_mode", "normalized_mask_ce") not in ("normalized_mask_ce", "region_mass"):
        raise ValueError("attention_loss_mode must be normalized_mask_ce or region_mass")
    scheduler_horizon_steps = getattr(args, "scheduler_horizon_steps", args.max_steps)
    if scheduler_horizon_steps < args.max_steps:
        raise ValueError(
            "scheduler_horizon_steps must be greater than or equal to "
            "max_optimizer_steps"
        )
    if not (0 <= args.warmup_steps < scheduler_horizon_steps):
        raise ValueError(
            "warmup_steps must be non-negative and below scheduler_horizon_steps"
        )
    if getattr(args, "quality_gate_enabled", False):
        if not (0 < args.quality_baseline_start <= args.quality_baseline_end <= args.max_steps):
            raise ValueError("quality baseline must be a non-empty range within max_optimizer_steps")
        if args.quality_window_size <= 0 or args.quality_consecutive_windows <= 0:
            raise ValueError("quality window size and consecutive-window count must be positive")
        if args.quality_max_loss_ratio <= 1.0:
            raise ValueError("quality max_loss_ratio must be greater than 1")
        if not 0.0 <= args.quality_max_clip_fraction <= 1.0:
            raise ValueError("quality max_clip_fraction must be in [0, 1]")
        if args.quality_max_abs_logit <= 0 or args.quality_max_update_ratio <= 0:
            raise ValueError("quality logit and update thresholds must be positive")


def load_train_config(config_path: Path, resume_from_checkpoint: Path | None = None) -> argparse.Namespace:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    dataset_config_path = _resolve_repository_path(config["dataset_config"])
    distributed_config_path = _resolve_repository_path(config["distributed_config"])
    dataset_config = _read_yaml(dataset_config_path)
    runtime_config = _read_yaml(distributed_config_path)

    objective = config["objective"]["mode"]
    paths = config["paths"]
    data_root = _required_env(paths["data_root_env"])
    manifest = os.environ.get(
        paths["train_manifest_env"],
        str(Path(data_root) / dataset_config["token_manifest_default"]),
    )
    output_root = _required_env(paths["output_root_env"])
    training = config["training"]
    optimization = config["optimization"]
    lora = config.get("lora", {})
    diagnostics = config.get("diagnostics", {})
    quality_gate = config.get("quality_gate", {})
    distributed = config["distributed"]
    values = dict(
        config_file=config_path,
        dataset_config_file=dataset_config_path,
        distributed_config_file=distributed_config_path,
        model=Path(_required_env(paths["model_env"])),
        train_manifest=Path(manifest),
        output=Path(output_root) / config["output_name"],
        objective=objective,
        max_steps=training["max_optimizer_steps"],
        scheduler_horizon_steps=training.get(
            "scheduler_horizon_steps",
            training["max_optimizer_steps"],
        ),
        save_steps=training["checkpoint_every_steps"], checkpoint_steps=[],
        batch_size=training["batch_size_per_gpu"], gradient_accumulation=training["gradient_accumulation"],
        learning_rate=optimization["learning_rate"], warmup_steps=optimization.get("warmup_steps", 0),
        optimizer_betas=tuple(optimization.get("betas", (0.9, 0.95))),
        weight_decay=optimization["weight_decay"], max_grad_norm=optimization["clip_grad_norm"],
        z_loss_weight=optimization.get("z_loss_weight", 0.0),
        loss_reduction=optimization.get("loss_reduction", "sample_mean"),
        lora_rank=lora.get("rank", 16), lora_alpha=lora.get("alpha", 16.0), lora_dropout=lora.get("dropout", 0.05),
        lora_targets=tuple(lora.get("targets", ("q_proj", "k_proj", "v_proj", "attn_out"))),
        diagnostic_every_steps=int(diagnostics.get("every_steps", 25)),
        gradient_decomposition_every_steps=int(diagnostics.get("gradient_decomposition_every_steps", 100)),
        quality_gate_enabled=bool(quality_gate.get("enabled", True)),
        quality_baseline_start=int(quality_gate.get("baseline_start", 101)),
        quality_baseline_end=int(quality_gate.get("baseline_end", 400)),
        quality_window_size=int(quality_gate.get("window_size", 50)),
        quality_max_loss_ratio=float(quality_gate.get("max_loss_ratio", 1.75)),
        quality_consecutive_windows=int(quality_gate.get("consecutive_windows", 2)),
        quality_max_clip_fraction=float(quality_gate.get("max_clip_fraction", 0.95)),
        quality_max_abs_logit=float(quality_gate.get("max_abs_logit", 1.0e4)),
        quality_max_update_ratio=float(quality_gate.get("max_update_ratio", 0.25)),
        condition_dropout=config.get("condition_dropout", 0.1), max_seq_len=config["data"]["max_seq_len"],
        seed=config["seed"], num_workers=config.get("num_workers", 4), resume_from_checkpoint=resume_from_checkpoint,
        attention_layers=[], attention_loss_weight=0.1, attention_qk_stage="post_rope",
        attention_loss_mode="normalized_mask_ce", gce_clusters=None, gce_weight=1.0, gce_levels=[1024, 512],
        nproc_per_node=int(distributed["nproc_per_node"]),
        global_batch_size=int(distributed["global_batch_size"]),
        launcher=runtime_config["launcher"], rdzv=runtime_config["rdzv"], precision=runtime_config["precision"],
        vq_grid=list(dataset_config["vq_grid"]),
    )
    if objective == "attention":
        values["attention_layers"] = config["attention_loss"]["layers"]
        values["attention_loss_weight"] = config["attention_loss"]["weight"]
        values["attention_qk_stage"] = config["attention_loss"].get("qk_stage", "post_rope")
        values["attention_loss_mode"] = config["attention_loss"].get("mode", "normalized_mask_ce")
    if objective == "gce":
        gce = config["gce"]
        values["gce_clusters"] = Path(_required_env(gce["cluster_path_env"]))
        values["gce_weight"] = gce["weight"]
        values["gce_levels"] = gce["levels"]
    args = argparse.Namespace(**values)
    validate_train_config(args)
    return args


def launch_summary(args: argparse.Namespace) -> dict:
    return {
        "config_file": str(args.config_file),
        "objective": args.objective,
        "nproc_per_node": args.nproc_per_node,
        "batch_per_gpu": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "computed_global_batch": args.nproc_per_node * args.batch_size * args.gradient_accumulation,
        "declared_global_batch": args.global_batch_size,
        "precision": args.precision,
        "max_seq_len": args.max_seq_len,
        "optimizer_steps": args.max_steps,
        "scheduler_horizon_steps": args.scheduler_horizon_steps,
        "checkpoint_interval": args.save_steps,
    }
