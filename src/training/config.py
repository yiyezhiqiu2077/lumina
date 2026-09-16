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
    distributed = config["distributed"]
    values = dict(
        config_file=config_path,
        dataset_config_file=dataset_config_path,
        distributed_config_file=distributed_config_path,
        model=Path(_required_env(paths["model_env"])),
        train_manifest=Path(manifest),
        output=Path(output_root) / config["output_name"],
        objective=objective,
        max_steps=training["max_optimizer_steps"], save_steps=training["checkpoint_every_steps"], checkpoint_steps=[],
        batch_size=training["batch_size_per_gpu"], gradient_accumulation=training["gradient_accumulation"],
        learning_rate=optimization["learning_rate"], warmup_steps=optimization.get("warmup_steps", 0),
        weight_decay=optimization["weight_decay"], max_grad_norm=optimization["clip_grad_norm"],
        lora_rank=lora.get("rank", 16), lora_alpha=lora.get("alpha", 16.0), lora_dropout=lora.get("dropout", 0.05),
        condition_dropout=config.get("condition_dropout", 0.1), max_seq_len=config["data"]["max_seq_len"],
        seed=config["seed"], num_workers=config.get("num_workers", 4), resume_from_checkpoint=resume_from_checkpoint,
        attention_layers=[], attention_loss_weight=0.1, gce_clusters=None, gce_weight=1.0, gce_levels=[1024, 512],
        nproc_per_node=int(distributed["nproc_per_node"]),
        global_batch_size=int(distributed["global_batch_size"]),
        launcher=runtime_config["launcher"], rdzv=runtime_config["rdzv"], precision=runtime_config["precision"],
        vq_grid=list(dataset_config["vq_grid"]),
    )
    if objective == "attention":
        values["attention_layers"] = config["attention_loss"]["layers"]
        values["attention_loss_weight"] = config["attention_loss"]["weight"]
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
        "checkpoint_interval": args.save_steps,
    }
