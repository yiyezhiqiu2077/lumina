from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.config import load_train_config, validate_train_config
from training.distributed import build_torchrun_command


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def required_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("DATA_ROOT", "/tmp/data")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GCE_CLUSTER_PATH", "/tmp/gce_clusters.pt")


def _load(name: str):
    return load_train_config(REPOSITORY / "configs/train" / name)


def test_train_configs_define_actual_launch_world_sizes():
    expected = {
        "magicbrush_ce.yaml": (4, 4, 2, 32),
        "magicbrush_attention.yaml": (2, 8, 4, 64),
        "magicbrush_gce.yaml": (4, 4, 2, 32),
    }
    for name, values in expected.items():
        args = _load(name)
        assert (args.nproc_per_node, args.batch_size, args.gradient_accumulation, args.global_batch_size) == values


def test_torchrun_command_and_resume_are_preserved(tmp_path):
    args = _load("magicbrush_attention.yaml")
    resume = tmp_path / "checkpoint-000001"
    command = build_torchrun_command(args, REPOSITORY / "scripts/train/train.py", args.config_file, resume)
    assert command[:3] == [command[0], "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert command[command.index("--nproc_per_node") + 1] == "2"
    assert command[command.index("--resume-from-checkpoint") + 1] == str(resume)
    assert "--distributed-worker" in command


def test_global_batch_validation_rejects_mismatch():
    args = argparse.Namespace(
        nproc_per_node=2,
        batch_size=8,
        gradient_accumulation=4,
        global_batch_size=63,
        vq_grid=[32, 32],
        launcher="torchrun",
        rdzv="standalone",
    )
    with pytest.raises(ValueError, match="global_batch_size mismatch"):
        validate_train_config(args)


def test_direct_worker_without_distributed_environment_has_actionable_error():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/train/train.py"), "--config", "unused.yaml", "--distributed-worker"],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert result.returncode != 0
    assert "worker mode requires RANK, WORLD_SIZE, and LOCAL_RANK" in result.stderr
