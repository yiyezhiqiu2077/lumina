"""Checkpoint persistence and resume validation for MagicBrush training."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from training.lora import load_lora_state_dict, lora_state_dict


_FINGERPRINT_SOURCES = (
    "src/dataset/utils.py",
    "src/dataset/magicbrush.py",
    "src/dataset/refedit.py",
    "src/training/checkpoint.py",
    "src/training/config.py",
    "src/training/distributed.py",
    "src/training/lora.py",
    "src/training/objective.py",
    "src/training/quality.py",
    "src/models/lumina/modeling_llada.py",
    "src/models/lumina/modeling_xllmx_dimoo.py",
    "src/models/objectives/attention.py",
    "src/models/objectives/gce.py",
    "src/models/objectives/reduction.py",
    "src/utils/constants.py",
    "src/utils/prompt_utils.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        "size": stat.st_size,
        "sha256": _sha256(resolved),
    }
    if include_path:
        identity["path"] = str(resolved)
    return identity


def _manifest_composition(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            dataset_name = json.loads(line).get("dataset_name", "magicbrush")
            counts[str(dataset_name)] = counts.get(str(dataset_name), 0) + 1
    return dict(sorted(counts.items()))


def build_run_fingerprint(args, world_size: int) -> dict[str, Any]:
    """Build the immutable recipe identity required for safe resume."""
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {
        relative: _sha256(repository / relative)
        for relative in _FINGERPRINT_SOURCES
    }
    payload = {
        "schema_version": 2,
        "objective": args.objective,
        "world_size": world_size,
        "effective_batch": args.batch_size * args.gradient_accumulation * world_size,
        "model": {
            "config": _file_identity(
                Path(args.model) / "config.json",
                include_path=False,
            ),
        },
        "dataset_config": _file_identity(
            Path(args.dataset_config_file),
            include_path=False,
        ),
        "distributed_config": _file_identity(
            Path(args.distributed_config_file),
            include_path=False,
        ),
        "train_manifest": _file_identity(
            Path(args.train_manifest),
            include_path=False,
        ),
        "gce_clusters": (
            _file_identity(Path(args.gce_clusters), include_path=False)
            if args.objective == "gce"
            else None
        ),
        "schedule": {
            "stop_after_steps": args.max_steps,
            "scheduler_horizon_steps": getattr(
                args,
                "scheduler_horizon_steps",
                args.max_steps,
            ),
            "save_steps": args.save_steps,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "warmup_steps": args.warmup_steps,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": list(args.optimizer_betas),
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": list(args.lora_targets),
        },
        "loss": {
            "reduction": args.loss_reduction,
            "z_loss_weight": args.z_loss_weight,
            "attention_weight": args.attention_loss_weight,
            "attention_layers": list(args.attention_layers),
            "attention_qk_stage": getattr(args, "attention_qk_stage", "pre_rope"),
            "attention_loss_mode": getattr(args, "attention_loss_mode", "normalized_mask_ce"),
            "gce_weight": args.gce_weight,
            "gce_levels": list(args.gce_levels),
        },
        "data": {
            "seed": args.seed,
            "condition_dropout": args.condition_dropout,
            "max_seq_len": args.max_seq_len,
            "target_corruption": {
                "mode": getattr(args, "target_corruption_mode", "full_target"),
                "schedule": "cosine_random_ratio",
                "minimum_masked_tokens": 1,
                "candidate_policy": getattr(args, "target_corruption_mode", "full_target"),
            },
            "cursor_strategy": "global_step_times_gradient_accumulation",
            "sampler": "DistributedSampler(drop_last=True)",
            "sample_count": getattr(args, "dataset_sample_count", None),
            "composition": _manifest_composition(Path(args.train_manifest)),
        },
        "quality_gate": {
            "enabled": args.quality_gate_enabled,
            "baseline_start": args.quality_baseline_start,
            "baseline_end": args.quality_baseline_end,
            "window_size": args.quality_window_size,
            "max_loss_ratio": args.quality_max_loss_ratio,
            "consecutive_windows": args.quality_consecutive_windows,
            "max_clip_fraction": args.quality_max_clip_fraction,
            "max_abs_logit": args.quality_max_abs_logit,
            "max_update_ratio": args.quality_max_update_ratio,
        },
        "source_sha256": source_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "digest": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    output: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    rank: int,
    objective: str,
    run_fingerprint: dict[str, Any],
    quality_state: dict[str, Any] | None = None,
) -> None:
    destination = output / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    success = destination / "_SUCCESS"
    if rank == 0:
        success.unlink(missing_ok=True)
        checkpoint_model = getattr(model, "module", model)
        _atomic_torch_save(lora_state_dict(checkpoint_model), destination / "lora.pt")
        _atomic_json_write(
            {
                "step": step,
                "objective": objective,
                "targets": run_fingerprint["payload"]["lora"]["targets"],
            },
            destination / "lora_config.json",
        )
    _atomic_torch_save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "step": step,
            "objective": objective,
            "fingerprint_digest": run_fingerprint["digest"],
        },
        destination / f"training_state.rank{rank:02d}.pt",
    )
    dist.barrier()
    if rank == 0:
        _atomic_json_write(
            {
                "step": step,
                "objective": objective,
                "world_size": dist.get_world_size(),
                "fingerprint": run_fingerprint,
                "quality_state": quality_state,
            },
            destination / "checkpoint_meta.json",
        )
        success.write_text("complete\n", encoding="utf-8")
    dist.barrier()


def _validate_metrics_continuity(metrics_path: Path, expected_step: int) -> None:
    if not metrics_path.is_file():
        raise ValueError(f"resume metrics file is missing: {metrics_path}")
    steps = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                steps.append(int(record["step"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid metrics record at {metrics_path}:{line_number}: {error}"
                ) from error
    if not steps:
        raise ValueError(f"resume metrics file is empty: {metrics_path}")
    expected = list(range(1, expected_step + 1))
    if steps != expected:
        raise ValueError(
            f"resume metrics must contain contiguous steps 1..{expected_step}; "
            f"got {len(steps)} rows ending at {steps[-1]}"
        )


def validate_checkpoint(
    checkpoint: Path,
    objective: str,
    run_fingerprint: dict[str, Any],
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if not (checkpoint / "_SUCCESS").is_file():
        raise ValueError(f"checkpoint is incomplete (missing _SUCCESS): {checkpoint}")
    meta_path = checkpoint / "checkpoint_meta.json"
    if not meta_path.is_file():
        raise ValueError(f"checkpoint metadata is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("objective") != objective:
        raise ValueError(
            f"checkpoint objective {meta.get('objective')!r} does not match {objective!r}"
        )
    saved_fingerprint = meta.get("fingerprint", {})
    if saved_fingerprint.get("digest") != run_fingerprint["digest"]:
        raise ValueError(
            "checkpoint run fingerprint does not match the current recipe: "
            f"saved={saved_fingerprint.get('digest')!r}, current={run_fingerprint['digest']!r}"
        )
    world_size = int(meta["world_size"])
    required = [checkpoint / "lora.pt", checkpoint / "lora_config.json"]
    required.extend(checkpoint / f"training_state.rank{rank:02d}.pt" for rank in range(world_size))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint is incomplete; missing files: {missing}")
    step = int(meta["step"])
    _validate_metrics_continuity(
        metrics_path or checkpoint.parent / "train_metrics.jsonl",
        step,
    )
    return meta


def restore_checkpoint(
    checkpoint: Path,
    model,
    optimizer,
    scheduler,
    rank: int,
    max_steps: int,
    objective: str,
    run_fingerprint: dict[str, Any],
    metrics_path: Path | None = None,
):
    meta = validate_checkpoint(
        checkpoint,
        objective,
        run_fingerprint,
        metrics_path=metrics_path,
    )
    load_lora_state_dict(
        model,
        torch.load(checkpoint / "lora.pt", map_location="cpu", weights_only=True),
    )
    state = torch.load(
        checkpoint / f"training_state.rank{rank:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if state.get("fingerprint_digest") != run_fingerprint["digest"]:
        raise ValueError(f"rank {rank} checkpoint state has a mismatched fingerprint")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state["step"])
    if step != int(meta["step"]):
        raise ValueError(
            f"rank {rank} checkpoint step {step} does not match metadata step {meta['step']}"
        )
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
        "scheduler_horizon": run_fingerprint["payload"]["schedule"][
            "scheduler_horizon_steps"
        ],
        "stop_after_steps": max_steps,
        "resumed_lr": resumed_lrs,
        "data_cursor_strategy": "derived_from_global_step_and_gradient_accumulation",
        "fingerprint_digest": run_fingerprint["digest"],
    }, meta.get("quality_state")
