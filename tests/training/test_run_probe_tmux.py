from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts/train/run_probe_tmux.sh"


def _fake_commands(tmp_path: Path) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    tmux = directory / "tmux"
    tmux.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" >> "$FAKE_TMUX_RECORD"
case "$1" in
  has-session) exit 1 ;;
  new-session) exit 0 ;;
esac
""",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    uv = directory / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == run ]]
shift
exec "$@"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return directory


def _environment(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n" * 8807, encoding="utf-8")
    cluster = tmp_path / "gce_clusters.pt"
    cluster.write_bytes(b"test")
    fake_bin = _fake_commands(tmp_path)
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
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "FAKE_TMUX_RECORD": str(tmp_path / "tmux.record"),
    }


def _run(*arguments: str, environment: dict[str, str]):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_probe_separates_stop_and_scheduler_horizons(tmp_path):
    environment = _environment(tmp_path)
    result = _run("ce", "825", environment=environment)
    assert result.returncode == 0, result.stderr
    generated = (
        Path(environment["OUTPUT_ROOT"])
        / "probe/configs/mb_ce_825.yaml"
    )
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["training"]["max_optimizer_steps"] == 825
    assert config["training"]["scheduler_horizon_steps"] == 2750
    assert config["training"]["checkpoint_every_steps"] == 275


def test_probe_allows_single_variable_learning_rate_control(tmp_path):
    environment = _environment(tmp_path)
    environment["LUMINA_PROBE_LEARNING_RATE"] = "5.0e-6"
    result = _run("ce", "825", environment=environment)
    assert result.returncode == 0, result.stderr
    generated = Path(environment["OUTPUT_ROOT"]) / "probe/configs/mb_ce_825.yaml"
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["optimization"]["learning_rate"] == 5.0e-6
    assert config["training"]["scheduler_horizon_steps"] == 2750


def test_probe_rejects_invalid_learning_rate_control(tmp_path):
    environment = _environment(tmp_path)
    environment["LUMINA_PROBE_LEARNING_RATE"] = "nan"
    result = _run("ce", "825", environment=environment)
    assert result.returncode != 0
    assert "must be finite and positive" in result.stderr


def test_probe_rejects_stop_beyond_formal_horizon(tmp_path):
    environment = _environment(tmp_path)
    result = _run("ce", "2751", environment=environment)
    assert result.returncode != 0
    assert "exceeds formal scheduler horizon" in result.stderr


def test_probe_checks_tmux_sessions_by_exact_name(tmp_path):
    environment = _environment(tmp_path)
    result = _run("ce", "825", environment=environment)
    assert result.returncode == 0, result.stderr
    record = Path(environment["FAKE_TMUX_RECORD"]).read_text(encoding="utf-8")
    assert "has-session\n-t\n=lumina_probe\n" in record
