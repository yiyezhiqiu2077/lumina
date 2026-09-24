from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "scripts/eval/run_mixed_2x3_eval.sh"
LABELS = ("CE-full", "CE-editregion", "Attention-full", "Attention-editregion", "GCE-full", "GCE-editregion")
TRAIN_OUTPUTS = (
    "MIXED-CE-8G-B4-A1-S42", "MIXED-CE-EDITREGION-8G-B4-A1-S42",
    "MIXED-ATTN-POSTROPE-REGION-8G-B4-A1-S42", "MIXED-ATTN-POSTROPE-REGION-EDITREGION-8G-B4-A1-S42",
    "MIXED-GCE-8G-B4-A1-S42", "MIXED-GCE-EDITREGION-8G-B4-A1-S42",
)


@pytest.fixture
def eval_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "mixed.jsonl"
    manifest.write_text("".join('{"id": %d}\n' % index for index in range(1600)), encoding="utf-8")
    gce = tmp_path / "gce.pt"
    gce.write_bytes(b"clusters")
    output = tmp_path / "training"
    output.mkdir()
    for name in TRAIN_OUTPUTS:
        checkpoint = output / name / "checkpoint-000500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    eval_output = tmp_path / "evaluation"
    eval_output.mkdir()
    canonical = tmp_path / "canonical.jsonl"
    tokens = tmp_path / "tokens.jsonl"
    subset = tmp_path / "subset.jsonl"
    canonical.write_text("{}\n", encoding="utf-8")
    tokens.write_text("{}\n", encoding="utf-8")
    subset.write_text("{}\n", encoding="utf-8")
    dino, clip = tmp_path / "dino", tmp_path / "clip"
    dino.mkdir()
    clip.mkdir()
    calls = tmp_path / "calls.txt"
    audit = tmp_path / "audit.sh"
    audit.write_text("#!/usr/bin/env bash\nprintf 'audit\\n' >> \"$FAKE_CALLS\"\n", encoding="utf-8")
    audit.chmod(0o755)
    evaluator = tmp_path / "evaluator.sh"
    evaluator.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'eval %s\\n' "$*" >> "$FAKE_CALLS"
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == --output ]]; then output=$2; fi
  shift
done
mkdir -p "$output"
printf '{"step": 1}\\n' > "$output/per_sample.jsonl"
printf '{"metrics": {}}\\n' > "$output/summary.json"
""",
        encoding="utf-8",
    )
    evaluator.chmod(0o755)
    compare = tmp_path / "compare.sh"
    compare.write_text("#!/usr/bin/env bash\nprintf 'compare\\n' >> \"$FAKE_CALLS\"\n", encoding="utf-8")
    compare.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "MODEL_PATH": str(model), "DATA_ROOT": str(data), "DATA_CONFIG": str(manifest), "GCE_CLUSTER_PATH": str(gce), "OUTPUT_ROOT": str(output),
        "EVAL_OUTPUT_ROOT": str(eval_output), "MAGICBRUSH_TEST_CANONICAL_MANIFEST": str(canonical),
        "MAGICBRUSH_TEST_TOKEN_MANIFEST": str(tokens), "MAGICBRUSH_TEST_SUBSET": str(subset),
        "DINO_MODEL_PATH": str(dino), "CLIP_MODEL_PATH": str(clip), "FAKE_CALLS": str(calls),
        "MIXED_2X3_ASSET_AUDIT": str(audit), "MIXED_2X3_EVALUATOR": str(evaluator), "MIXED_2X3_COMPARE": str(compare),
    })
    return environment, calls


def _run(environment: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(LAUNCHER), *args], cwd=REPOSITORY, env=environment, text=True, capture_output=True, check=False)


def _calls(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_eval_print_command_has_fixed_six_run_order_and_no_execution(eval_environment):
    environment, calls = eval_environment
    result = _run(environment, "--print-command")
    assert result.returncode == 0, result.stderr
    assert _calls(calls) == []
    for label in LABELS:
        assert label in result.stdout
    assert "test_token_manifest=" in result.stdout


def test_eval_run_reuses_same_test_subset_for_all_six_runs(eval_environment):
    environment, calls = eval_environment
    result = _run(environment, "--run")
    assert result.returncode == 0, result.stderr
    recorded = _calls(calls)
    assert recorded[0] == "audit"
    assert [line.split()[0] for line in recorded[1:7]] == ["eval"] * 6
    assert recorded[-1] == "compare"
    assert all(environment["MAGICBRUSH_TEST_SUBSET"] in line for line in recorded[1:7])
