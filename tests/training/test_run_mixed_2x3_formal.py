from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "scripts/train/run_mixed_2x3_formal.sh"

ORDER = [
    "mixed_ce_8g_b4_a1.yaml",
    "mixed_ce_editregion_8g_b4_a1.yaml",
    "mixed_attention_postrope_region_8g_b4_a1.yaml",
    "mixed_attention_editregion_8g_b4_a1.yaml",
    "mixed_gce_8g_b4_a1.yaml",
    "mixed_gce_editregion_8g_b4_a1.yaml",
]
OUTPUTS = {
    "mixed_ce_8g_b4_a1.yaml": "MIXED-CE-8G-B4-A1-S42",
    "mixed_ce_editregion_8g_b4_a1.yaml": "MIXED-CE-EDITREGION-8G-B4-A1-S42",
    "mixed_attention_postrope_region_8g_b4_a1.yaml": "MIXED-ATTN-POSTROPE-REGION-8G-B4-A1-S42",
    "mixed_attention_editregion_8g_b4_a1.yaml": "MIXED-ATTN-POSTROPE-REGION-EDITREGION-8G-B4-A1-S42",
    "mixed_gce_8g_b4_a1.yaml": "MIXED-GCE-8G-B4-A1-S42",
    "mixed_gce_editregion_8g_b4_a1.yaml": "MIXED-GCE-EDITREGION-8G-B4-A1-S42",
}


@pytest.fixture
def formal_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "manifest.jsonl"
    # 50 optimizer steps/epoch × 10 epochs: enough for formal quality-gate
    # validation while remaining a synthetic manifest.
    manifest.write_text("".join('{"id": %d}\n' % index for index in range(1600)), encoding="utf-8")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    gce = tmp_path / "gce_clusters.pt"
    gce.write_bytes(b"test")
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fake_formal.sh"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
config=$1
mode=$2
name=$(basename "$config")
printf '%s %s\\n' "$name" "$mode" >> "$FAKE_CALLS"
[[ "$mode" == --print-command ]] && exit 0
if [[ "${FAKE_SCENARIO:-success}" == process_failure ]]; then exit 23; fi
python - "$name" <<'PY'
import json, os, sys
from pathlib import Path
name = sys.argv[1]
outputs = json.loads(os.environ["FAKE_OUTPUTS"])
out = Path(os.environ["OUTPUT_ROOT"]) / outputs[name]
out.mkdir(parents=True, exist_ok=True)
scenario = os.environ.get("FAKE_SCENARIO", "success")
step = 500
status = "SUCCEEDED"
metric_step = step
make_checkpoint = True
if scenario == "quality_failure": status = "QUALITY_FAILED"
elif scenario == "quality_step_failure": step -= 1
elif scenario == "metric_failure": metric_step -= 1
elif scenario == "checkpoint_failure": make_checkpoint = False
(out / "quality_status.json").write_text(json.dumps({"status": status, "step": step}) + "\\n")
(out / "train_metrics.jsonl").write_text(json.dumps({"step": metric_step}) + "\\n")
if make_checkpoint:
    checkpoint = out / "checkpoint-000500"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "_SUCCESS").write_text("ok\\n")
PY
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    clean_git = tmp_path / "clean_git.sh"
    clean_git.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    clean_git.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "MODEL_PATH": str(model),
            "DATA_ROOT": str(data_root),
            "DATA_CONFIG": str(manifest),
            "OUTPUT_ROOT": str(output_root),
            "GCE_CLUSTER_PATH": str(gce),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "MIXED_2X3_FORMAL_LAUNCHER": str(fake),
            "MIXED_2X3_GIT_BIN": str(clean_git),
            "FAKE_CALLS": str(calls),
            "FAKE_OUTPUTS": json.dumps(OUTPUTS),
        }
    )
    return environment, calls


def run_launcher(environment: dict[str, str], *arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=cwd or REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def called(calls: Path) -> list[tuple[str, str]]:
    if not calls.exists():
        return []
    return [tuple(line.split()) for line in calls.read_text(encoding="utf-8").splitlines()]


def test_default_is_dry_run_and_preserves_required_order(formal_environment):
    environment, calls = formal_environment
    result = run_launcher(environment)
    assert result.returncode == 0, result.stderr
    assert called(calls) == [(name, "--print-command") for name in ORDER]
    assert not (Path(environment["OUTPUT_ROOT"]) / "2x3_formal_pipeline.log").exists()


def test_print_command_does_not_start_training_and_runs_from_arbitrary_cwd(formal_environment, tmp_path):
    environment, calls = formal_environment
    result = run_launcher(environment, "--print-command", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert called(calls) == [(name, "--print-command") for name in ORDER]
    assert not any((Path(environment["OUTPUT_ROOT"]) / name).exists() for name in OUTPUTS.values())


def test_all_configs_are_preflighted_before_first_run(formal_environment):
    environment, calls = formal_environment
    result = run_launcher(environment, "--run")
    assert result.returncode == 0, result.stderr
    assert called(calls) == [(name, "--print-command") for name in ORDER] + [(name, "--run") for name in ORDER]


@pytest.mark.parametrize(
    ("scenario", "needle"),
    [
        ("process_failure", "process_exit"),
        ("quality_failure", "success_gate"),
        ("quality_step_failure", "success_gate"),
        ("metric_failure", "success_gate"),
        ("checkpoint_failure", "success_gate"),
    ],
)
def test_run_stops_after_process_or_success_gate_failure(formal_environment, scenario, needle):
    environment, calls = formal_environment
    environment["FAKE_SCENARIO"] = scenario
    result = run_launcher(environment, "--run")
    assert result.returncode != 0
    assert needle in (Path(environment["OUTPUT_ROOT"]) / "2x3_formal_pipeline.log").read_text(encoding="utf-8")
    assert called(calls) == [(name, "--print-command") for name in ORDER] + [(ORDER[0], "--run")]


def test_existing_output_fails_before_any_child_launcher(formal_environment):
    environment, calls = formal_environment
    existing = Path(environment["OUTPUT_ROOT"]) / OUTPUTS[ORDER[0]]
    existing.mkdir()
    (existing / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    result = run_launcher(environment, "--run")
    assert result.returncode != 0
    assert "existing formal output detected" in result.stderr
    assert called(calls) == []


def test_missing_gce_asset_fails_before_any_child_launcher(formal_environment):
    environment, calls = formal_environment
    Path(environment["GCE_CLUSTER_PATH"]).unlink()
    result = run_launcher(environment, "--run")
    assert result.returncode != 0
    assert "GCE_CLUSTER_PATH" in result.stderr
    assert called(calls) == []


def test_dirty_worktree_is_rejected_before_any_child_launcher(formal_environment, tmp_path):
    environment, calls = formal_environment
    fake_git = tmp_path / "fake_git.sh"
    fake_git.write_text("#!/usr/bin/env bash\nprintf ' M tracked-file\\n'\n", encoding="utf-8")
    fake_git.chmod(0o755)
    environment["MIXED_2X3_GIT_BIN"] = str(fake_git)
    result = run_launcher(environment, "--run")
    assert result.returncode != 0
    assert "refusing dirty formal run" in result.stderr
    assert called(calls) == []
