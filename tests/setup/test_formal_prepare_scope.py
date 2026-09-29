import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/setup/run_formal_prepare.sh"


def _printed(*arguments: str) -> str:
    return subprocess.check_output(["bash", str(SCRIPT), *arguments, "--print-command"], text=True, cwd=ROOT)


def test_formal_prepare_scopes_emit_matching_audit_and_dependencies():
    all_output = _printed()
    train_output = _printed("--train-only")
    eval_output = _printed("--eval-only")
    for output in (all_output, train_output, eval_output):
        assert "download_formal_models.py" in output
    assert "--mode all" in all_output
    assert "--mode train" in train_output
    assert "--mode eval" in eval_output
    assert "download_magicbrush_train.py" not in eval_output
    assert "download_magicbrush_test.py" not in train_output
    assert "--only dino" not in train_output
