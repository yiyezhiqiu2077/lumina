"""Online scientific-quality checks for long MagicBrush runs."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


class QualityGateError(RuntimeError):
    """Raised when a run is mechanically healthy but scientifically invalid."""


@dataclass
class TrainingQualityGate:
    baseline_start: int
    baseline_end: int
    window_size: int
    max_loss_ratio: float
    consecutive_windows: int
    max_clip_fraction: float
    max_abs_logit: float
    max_update_ratio: float
    baseline_losses: list[float] = field(default_factory=list)
    recent_losses: deque[float] = field(init=False)
    recent_clipped: deque[bool] = field(init=False)
    bad_loss_windows: int = 0

    def __post_init__(self) -> None:
        self.recent_losses = deque(maxlen=self.window_size)
        self.recent_clipped = deque(maxlen=self.window_size)

    @property
    def baseline(self) -> float | None:
        if not self.baseline_losses:
            return None
        return sum(self.baseline_losses) / len(self.baseline_losses)

    def state_dict(self) -> dict[str, Any]:
        return {
            "baseline_losses": list(self.baseline_losses),
            "recent_losses": list(self.recent_losses),
            "recent_clipped": list(self.recent_clipped),
            "bad_loss_windows": self.bad_loss_windows,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.baseline_losses = [float(value) for value in state["baseline_losses"]]
        self.recent_losses = deque(
            (float(value) for value in state["recent_losses"]),
            maxlen=self.window_size,
        )
        self.recent_clipped = deque(
            (bool(value) for value in state["recent_clipped"]),
            maxlen=self.window_size,
        )
        self.bad_loss_windows = int(state["bad_loss_windows"])

    def observe(
        self,
        *,
        step: int,
        generation_loss: float,
        grad_norm: float,
        max_abs_logit: float,
        update_ratio: float | None,
        clipped: bool,
        parameters_finite: bool = True,
    ) -> dict[str, float | int | bool | None]:
        scalars = {
            "generation_loss": generation_loss,
            "grad_norm": grad_norm,
            "max_abs_logit": max_abs_logit,
        }
        if update_ratio is not None:
            scalars["update_ratio"] = update_ratio
        nonfinite = [name for name, value in scalars.items() if not math.isfinite(value)]
        if not parameters_finite:
            nonfinite.append("trainable_parameters")
        if nonfinite:
            raise QualityGateError(
                f"quality gate failed at step {step}: non-finite {', '.join(nonfinite)}"
            )
        if max_abs_logit > self.max_abs_logit:
            raise QualityGateError(
                f"quality gate failed at step {step}: max_abs_logit={max_abs_logit:.6g} "
                f"> {self.max_abs_logit:.6g}"
            )
        if update_ratio is not None and update_ratio > self.max_update_ratio:
            raise QualityGateError(
                f"quality gate failed at step {step}: update_ratio={update_ratio:.6g} "
                f"> {self.max_update_ratio:.6g}"
            )

        if self.baseline_start <= step <= self.baseline_end:
            self.baseline_losses.append(generation_loss)
        self.recent_losses.append(generation_loss)
        self.recent_clipped.append(clipped)
        baseline = self.baseline
        window_loss = None
        clip_fraction = None
        window_boundary = (
            len(self.recent_losses) == self.window_size
            and step % self.window_size == 0
        )
        if window_boundary:
            window_loss = sum(self.recent_losses) / self.window_size
            clip_fraction = sum(self.recent_clipped) / self.window_size
            if step > self.baseline_end and baseline is not None:
                if window_loss > baseline * self.max_loss_ratio:
                    self.bad_loss_windows += 1
                else:
                    self.bad_loss_windows = 0
                if self.bad_loss_windows >= self.consecutive_windows:
                    raise QualityGateError(
                        f"quality gate failed at step {step}: generation-loss window "
                        f"{window_loss:.6g} > baseline {baseline:.6g} * {self.max_loss_ratio:.6g} "
                        f"for {self.bad_loss_windows} consecutive non-overlapping windows"
                    )
            if clip_fraction > self.max_clip_fraction:
                raise QualityGateError(
                    f"quality gate failed at step {step}: clipping fraction "
                    f"{clip_fraction:.3f} > {self.max_clip_fraction:.3f}"
                )
        return {
            "quality_baseline": baseline,
            "quality_window_loss": window_loss,
            "quality_clip_fraction": clip_fraction,
            "quality_bad_loss_windows": self.bad_loss_windows,
        }


def write_quality_status(path: Path, status: str, **details: Any) -> None:
    """Atomically publish the current run-quality state."""
    payload = {"status": status, **details}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_quality_success(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"quality status is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCEEDED":
        raise SystemExit(
            f"quality status is not SUCCEEDED: {path}: {payload.get('status')!r}"
        )


def mark_process_failed(path: Path, objective: str, exit_code: int) -> None:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "QUALITY_FAILED":
            return
        step = payload.get("step", 0)
    else:
        step = 0
    write_quality_status(
        path,
        "PROCESS_FAILED",
        objective=objective,
        step=step,
        exit_code=exit_code,
        reason=f"training process exited with code {exit_code}",
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--require-succeeded", type=Path)
    action.add_argument("--mark-process-failed", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    if args.require_succeeded is not None:
        require_quality_success(args.require_succeeded)
        return
    if args.objective is None or args.exit_code is None:
        parser.error("--mark-process-failed requires --objective and --exit-code")
    mark_process_failed(args.mark_process_failed, args.objective, args.exit_code)


if __name__ == "__main__":
    _main()
