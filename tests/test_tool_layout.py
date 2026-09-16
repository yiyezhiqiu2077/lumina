"""Keep project-owned MagicBrush CLIs out of the repository-root tools directory."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_magicbrush_tool_entrypoints_are_present_and_compilable():
    expected = [
        "scripts/data/check_magicbrush.py",
        "scripts/data/preprocess_magicbrush.py",
        "scripts/tools/gce/build_clusters.py",
        "scripts/tools/gce/inspect_clusters.py",
        "scripts/tools/gce/probe_scale.py",
    ]
    for relative in expected:
        source = (ROOT / relative).read_text(encoding="utf-8")
        compile(source, relative, "exec")
    assert not (ROOT / "tools").exists()
