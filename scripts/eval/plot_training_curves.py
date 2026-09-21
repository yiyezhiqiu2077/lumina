#!/usr/bin/env python3
"""Plot objective-specific curves from a MagicBrush ``train_metrics.jsonl`` file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBJECTIVE_METRICS = {
    "ce": {
        "loss_generation.png": ("Generation loss", ("L_gen",)),
        "loss_total.png": ("Total loss", ("L_total",)),
    },
    "attention": {
        "loss_generation.png": ("Generation loss", ("L_gen",)),
        "loss_attention.png": ("Attention loss", ("L_attn_raw", "weighted_attn_loss")),
        "loss_total.png": ("Total loss", ("L_total",)),
        "attention_ratio.png": ("Attention / generation ratio", ("attn_to_gen_ratio",)),
        "attention_localization.png": (
            "Attention localization",
            ("conditional_localization_mass", "actual_full_attention_edit_mask_mass"),
        ),
        "attention_entropy.png": ("Attention entropy", ("attention_entropy",)),
    },
    "gce": {
        "loss_generation.png": ("Generation loss", ("L_gen",)),
        "loss_gce.png": ("GCE loss", ("gce_loss",)),
        "loss_gce_levels.png": ("GCE loss by level", ("gce_loss_k1024", "gce_loss_k512")),
        "loss_total.png": ("Total loss", ("L_total",)),
    },
}


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics file does not exist: {path}")
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        if not isinstance(value.get("step"), (int, float)) or not math.isfinite(float(value["step"])):
            raise ValueError(f"missing or non-finite step at {path}:{line_number}")
        records.append(value)
    if not records:
        raise ValueError(f"metrics file is empty: {path}")
    return records


def rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if window < 1:
        raise ValueError("smooth window must be at least 1")
    if values.size == 0:
        raise ValueError("cannot smooth an empty curve")
    width = min(window, values.size)
    kernel = np.ones(width, dtype=float)
    totals = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones_like(values), kernel, mode="same")
    return totals / counts


def metric_values(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    for index, record in enumerate(records):
        if key not in record:
            raise ValueError(f"missing metric {key!r} in record {index + 1}")
        value = record[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite metric {key!r} in record {index + 1}")
        steps.append(float(record["step"]))
        values.append(float(value))
    return np.asarray(steps), np.asarray(values)


def _style_epoch_axis(axis, steps: np.ndarray, steps_per_epoch: int) -> None:
    if steps_per_epoch < 1:
        raise ValueError("steps-per-epoch must be at least 1")
    last_step = int(max(steps))
    for boundary in range(steps_per_epoch, last_step + 1, steps_per_epoch):
        axis.axvline(boundary, color="0.6", linestyle="--", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Optimizer step")
    axis.grid(alpha=0.2)
    secondary = axis.secondary_xaxis(
        "top",
        functions=(lambda step: step / steps_per_epoch, lambda epoch: epoch * steps_per_epoch),
    )
    secondary.set_xlabel("Epoch")


def plot_metrics(
    records: list[dict],
    *,
    title: str,
    metrics: tuple[str, ...],
    steps_per_epoch: int,
    smooth_window: int,
    output: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    plotted_steps = None
    for metric in metrics:
        steps, values = metric_values(records, metric)
        plotted_steps = steps
        axis.plot(steps, values, alpha=0.25, linewidth=1.0, label=f"{metric} raw")
        axis.plot(steps, rolling_mean(values, smooth_window), linewidth=2.0, label=f"{metric} mean")
    assert plotted_steps is not None
    _style_epoch_axis(axis, plotted_steps, steps_per_epoch)
    axis.set_title(title)
    axis.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_attention_per_layer(records: list[dict], *, steps_per_epoch: int, smooth_window: int, output: Path) -> Path:
    layers = sorted({str(layer) for record in records for layer in record.get("per_layer", {})}, key=int)
    if not layers:
        raise ValueError("attention records do not contain per_layer metrics")
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    steps = np.asarray([float(record["step"]) for record in records])
    for layer in layers:
        values = []
        for index, record in enumerate(records):
            try:
                value = record["per_layer"][layer]["conditional_localization_mass"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing per_layer localization for layer {layer!r} in record {index + 1}") from error
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite per_layer localization for layer {layer!r} in record {index + 1}")
            values.append(float(value))
        axis.plot(steps, rolling_mean(values, smooth_window), linewidth=2.0, label=f"layer {layer}")
    _style_epoch_axis(axis, steps, steps_per_epoch)
    axis.set_title("Per-layer conditional localization mass")
    axis.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_objective(
    metrics_path: Path,
    *,
    objective: str,
    steps_per_epoch: int,
    smooth_window: int,
    output: Path,
    title: str | None = None,
) -> list[Path]:
    if objective not in OBJECTIVE_METRICS:
        raise ValueError(f"unsupported objective {objective!r}; choose from {sorted(OBJECTIVE_METRICS)}")
    records = read_metrics(metrics_path)
    generated = []
    for filename, (metric_title, metrics) in OBJECTIVE_METRICS[objective].items():
        generated.append(
            plot_metrics(
                records,
                title=f"{title}: {metric_title}" if title else metric_title,
                metrics=metrics,
                steps_per_epoch=steps_per_epoch,
                smooth_window=smooth_window,
                output=output / filename,
            )
        )
    if objective == "attention":
        generated.append(
            plot_attention_per_layer(
                records,
                steps_per_epoch=steps_per_epoch,
                smooth_window=smooth_window,
                output=output / "attention_per_layer.png",
            )
        )
    return generated


def plot_comparison(
    *,
    ce: Path,
    attention: Path,
    gce: Path,
    steps_per_epoch: int,
    smooth_window: int,
    output: Path,
    title: str | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    final_steps = []
    for label, path in (("CE", ce), ("Attention", attention), ("GCE", gce)):
        steps, values = metric_values(read_metrics(path), "L_gen")
        final_steps.extend(steps)
        axis.plot(steps, values, alpha=0.2, linewidth=1.0, label=f"{label} raw")
        axis.plot(steps, rolling_mean(values, smooth_window), linewidth=2.0, label=f"{label} mean")
    _style_epoch_axis(axis, np.asarray(final_steps), steps_per_epoch)
    axis.set_title(title or "Generation-loss comparison")
    axis.set_ylabel("L_gen")
    axis.legend(fontsize="small", ncol=2)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "compare_generation_loss.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep regular plotting as the default while reserving the optional
    # positional subcommand for a three-objective comparison.
    parser.add_argument("mode", nargs="?", choices=("plot", "compare"), default="plot")
    parser.add_argument("--input", type=Path, help="Path to one train_metrics.jsonl file")
    parser.add_argument("--objective", choices=tuple(OBJECTIVE_METRICS))
    parser.add_argument("--ce", type=Path, help="CE train_metrics.jsonl for compare mode")
    parser.add_argument("--attention", type=Path, help="Attention train_metrics.jsonl for compare mode")
    parser.add_argument("--gce", type=Path, help="GCE train_metrics.jsonl for compare mode")
    parser.add_argument("--steps-per-epoch", type=int, default=275)
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    if args.mode == "compare":
        missing = [name for name in ("ce", "attention", "gce") if getattr(args, name) is None]
    else:
        missing = [name for name in ("input", "objective") if getattr(args, name) is None]
    if missing:
        parser.error(f"{args.mode} mode requires: {', '.join('--' + name for name in missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "compare":
        generated = [
            plot_comparison(
                ce=args.ce,
                attention=args.attention,
                gce=args.gce,
                steps_per_epoch=args.steps_per_epoch,
                smooth_window=args.smooth_window,
                output=args.output,
                title=args.title,
            )
        ]
    else:
        generated = plot_objective(
            args.input,
            objective=args.objective,
            steps_per_epoch=args.steps_per_epoch,
            smooth_window=args.smooth_window,
            output=args.output,
            title=args.title,
        )
    print(json.dumps({"output": [str(path) for path in generated]}, indent=2))


if __name__ == "__main__":
    main()
