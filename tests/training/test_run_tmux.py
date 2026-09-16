from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts/train/run_tmux.sh"


def _fake_tmux(tmp_path: Path) -> Path:
    executable = tmp_path / "bin" / "tmux"
    executable.parent.mkdir()
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  has-session)
    [[ "${FAKE_TMUX_ACTIVE:-}" == "$3" ]] && exit 0
    exit 1
    ;;
  new-session)
    printf '%s\\n' "$@" >> "$FAKE_TMUX_RECORD"
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    uv = executable.parent / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
printf 'fake uv invoked: %s\\n' "$*"
exit "${FAKE_UV_EXIT:-0}"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return executable.parent


def _environment(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "model"
    (model / "vqvae").mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n" * 8807, encoding="utf-8")
    cluster = tmp_path / "gce_clusters.pt"
    cluster.write_bytes(b"test")
    record = tmp_path / "tmux.record"
    fake_bin = _fake_tmux(tmp_path)
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(REPOSITORY),
        "ASSET_ROOT": str(tmp_path / "assets"),
        "DATA_ROOT": str(tmp_path / "data"),
        "MODEL_PATH": str(model),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "MAGICBRUSH_DATA_CONFIG": str(manifest),
        "DATA_CONFIG": str(manifest),
        "GCE_CLUSTER_PATH": str(cluster),
        "CUDA_VISIBLE_DEVICES": "0,1",
        "FAKE_TMUX_RECORD": str(record),
    }


def _run(*arguments: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("objective", "session", "config"),
    (
        ("attention", "lumina_attn", "mb_attention_2g_b8_a2.yaml"),
        ("gce", "lumina_gce", "mb_gce_2g_b8_a2.yaml"),
        ("ce", "lumina_ce", "mb_ce_2g_b8_a2.yaml"),
    ),
)
def test_objective_creates_snapshot_launcher(
    tmp_path: Path, objective: str, session: str, config: str
):
    environment = _environment(tmp_path)
    result = _run(objective, environment=environment)
    assert result.returncode == 0, result.stderr
    assert f"session={session}" in result.stdout
    assert "selected GPUs: 0,1" in result.stdout
    launcher = Path(environment["OUTPUT_ROOT"]) / "logs" / f"{objective}.command.sh"
    content = launcher.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(launcher)], check=False).returncode == 0
    assert "set -o pipefail" in content
    assert "TRAIN_STATUS=${PIPESTATUS[0]}" in content
    assert "export PROJECT_ROOT=" in content
    assert "export CUDA_VISIBLE_DEVICES=0\\,1" in content
    assert config in content
    assert f"{objective}.exit_code" in content
    record = Path(environment["FAKE_TMUX_RECORD"]).read_text(encoding="utf-8")
    assert f"-s\n{session}\n" in record
    assert str(launcher) in record


def test_launcher_preserves_python_exit_status(tmp_path: Path):
    environment = _environment(tmp_path)
    assert _run("attention", environment=environment).returncode == 0
    launcher = Path(environment["OUTPUT_ROOT"]) / "logs" / "attention.command.sh"
    environment["FAKE_UV_EXIT"] = "23"
    launched = subprocess.run(["bash", str(launcher)], env=environment, text=True, capture_output=True, check=False)
    assert launched.returncode == 23
    exit_code = Path(environment["OUTPUT_ROOT"]) / "logs" / "attention.exit_code"
    assert exit_code.read_text(encoding="utf-8") == "23\n"
    assert not (Path(environment["OUTPUT_ROOT"]) / "logs" / "attention.running").exists()
    assert "fake uv invoked" in (Path(environment["OUTPUT_ROOT"]) / "logs" / "attention.log").read_text(encoding="utf-8")


def test_unknown_objective_is_rejected(tmp_path: Path):
    result = _run("unknown", environment=_environment(tmp_path))
    assert result.returncode != 0
    assert "unknown command" in result.stderr


def test_missing_environment_is_rejected(tmp_path: Path):
    environment = _environment(tmp_path)
    del environment["MODEL_PATH"]
    result = _run("attention", environment=environment)
    assert result.returncode != 0
    assert "MODEL_PATH" in result.stderr


def test_missing_objective_config_is_rejected(tmp_path: Path):
    environment = _environment(tmp_path)
    environment["PROJECT_ROOT"] = str(tmp_path / "missing-project")
    Path(environment["PROJECT_ROOT"]).mkdir()
    result = _run("attention", environment=environment)
    assert result.returncode != 0
    assert "missing objective config" in result.stderr


def test_gce_requires_cluster_asset(tmp_path: Path):
    environment = _environment(tmp_path)
    del environment["GCE_CLUSTER_PATH"]
    result = _run("gce", environment=environment)
    assert result.returncode != 0
    assert "GCE_CLUSTER_PATH" in result.stderr


@pytest.mark.parametrize("artifact", ("train_metrics.jsonl", "experiment_config.json", "lora_report.json"))
def test_fresh_run_rejects_existing_experiment_artifacts(tmp_path: Path, artifact: str):
    environment = _environment(tmp_path)
    output = Path(environment["OUTPUT_ROOT"]) / "MB-ATTN-2G-B8-A2-S42"
    output.mkdir(parents=True)
    (output / artifact).write_text("old run\n", encoding="utf-8")
    result = _run("attention", environment=environment)
    assert result.returncode != 0
    assert "existing experiment output detected" in result.stderr
    assert "use resume" in result.stderr


def test_fresh_run_rejects_existing_checkpoint(tmp_path: Path):
    environment = _environment(tmp_path)
    checkpoint = Path(environment["OUTPUT_ROOT"]) / "MB-ATTN-2G-B8-A2-S42" / "checkpoint-001375"
    checkpoint.mkdir(parents=True)
    result = _run("attention", environment=environment)
    assert result.returncode != 0
    assert "existing experiment output detected" in result.stderr


def test_valid_resume_is_accepted_and_stale_exit_code_is_removed(tmp_path: Path):
    environment = _environment(tmp_path)
    output = Path(environment["OUTPUT_ROOT"]) / "MB-ATTN-2G-B8-A2-S42"
    checkpoint = output / "checkpoint-001375"
    checkpoint.mkdir(parents=True)
    logs = Path(environment["OUTPUT_ROOT"]) / "logs"
    logs.mkdir(parents=True)
    stale_exit = logs / "attention.exit_code"
    stale_exit.write_text("0\n", encoding="utf-8")
    result = _run("attention", "--resume-from-checkpoint", str(checkpoint), environment=environment)
    assert result.returncode == 0, result.stderr
    assert not stale_exit.exists()
    launcher = logs / "attention.command.sh"
    assert f"--resume-from-checkpoint {checkpoint}" in launcher.read_text(encoding="utf-8")


def test_resume_must_belong_to_requested_objective_output(tmp_path: Path):
    environment = _environment(tmp_path)
    (Path(environment["OUTPUT_ROOT"]) / "MB-ATTN-2G-B8-A2-S42").mkdir(parents=True)
    checkpoint = Path(environment["OUTPUT_ROOT"]) / "MB-GCE-2G-B8-A2-S42" / "checkpoint-001375"
    checkpoint.mkdir(parents=True)
    result = _run("attention", "--resume-from-checkpoint", str(checkpoint), environment=environment)
    assert result.returncode != 0
    assert "correct attention experiment directory" in result.stderr


def test_one_gpu_is_rejected(tmp_path: Path):
    environment = _environment(tmp_path)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    result = _run("attention", environment=environment)
    assert result.returncode != 0
    assert "exactly 2 device IDs" in result.stderr


def test_duplicate_and_other_active_sessions_are_rejected(tmp_path: Path):
    environment = _environment(tmp_path)
    environment["FAKE_TMUX_ACTIVE"] = "lumina_attn"
    duplicate = _run("attention", environment=environment)
    assert duplicate.returncode != 0
    assert "session already exists" in duplicate.stderr

    other = _run("gce", environment=environment)
    assert other.returncode != 0
    assert "another Lumina training session is already active" in other.stderr


def test_help_and_status_do_not_start_training(tmp_path: Path):
    environment = _environment(tmp_path)
    help_result = _run("help", environment=environment)
    status_result = _run("status", environment=environment)
    assert help_result.returncode == 0
    assert "Usage:" in help_result.stdout
    assert status_result.returncode == 0
    assert "NOT STARTED" in status_result.stdout
    assert not Path(environment["FAKE_TMUX_RECORD"]).exists()


def test_status_reports_finished_and_failed_exit_codes(tmp_path: Path):
    environment = _environment(tmp_path)
    logs = Path(environment["OUTPUT_ROOT"]) / "logs"
    logs.mkdir(parents=True)
    (logs / "attention.exit_code").write_text("0\n", encoding="utf-8")
    (logs / "gce.exit_code").write_text("7\n", encoding="utf-8")
    result = _run("status", environment=environment)
    assert result.returncode == 0
    assert "FINISHED exit_code=0" in result.stdout
    assert "FAILED exit_code=7" in result.stdout
