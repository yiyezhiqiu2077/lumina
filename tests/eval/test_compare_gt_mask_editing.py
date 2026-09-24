from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts/eval/compare_gt_mask_editing.py"


def _summary(path: Path, metrics: dict[str, float]) -> Path:
    path.write_text(
        json.dumps({"metrics": {key: {"mean": value} for key, value in metrics.items()}}),
        encoding="utf-8",
    )
    return path


def test_compare_includes_perceptual_metrics_and_accepts_legacy_summary(tmp_path):
    current = _summary(tmp_path / "current.json", {"roi_lpips": 0.2, "full_dino_i": 0.8})
    legacy = _summary(tmp_path / "legacy.json", {"full_psnr_target": 30.0})
    output = tmp_path / "comparison"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--model", "current", str(current), "--model", "legacy", str(legacy)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert rows[0]["ROI LPIPS"] == 0.2
    assert rows[0]["Full DINO-I"] == 0.8
    assert rows[1]["ROI LPIPS"] is None
    assert "ROI CLIP-I" in rows[1]
