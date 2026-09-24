from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY / "scripts/eval/evaluate_gt_mask_editing.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gt_mask_editing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_perceptual_metrics_are_opt_in_and_preserve_default_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH), "--manifest", "manifest.jsonl", "--subset", "subset.jsonl"])
    args = EVALUATOR.parse_args()
    assert not EVALUATOR.perceptual_metrics_requested(args)
    assert args.lpips is False
    assert args.dino_model is None
    assert args.clip_model is None
    assert args.lpips_net == "alex"
    assert args.roi_padding_ratio == 0.10
