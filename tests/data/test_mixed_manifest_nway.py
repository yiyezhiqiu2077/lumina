import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _component(root: Path, name: str) -> Path:
    files = root / name / "files"; files.mkdir(parents=True)
    token = files / "one.pt"
    torch.save({"source_codes": torch.zeros(26, 38, dtype=torch.int32), "target_codes": torch.zeros(26, 38, dtype=torch.int32), "edit_mask": torch.ones(26, 38, dtype=torch.bool), "token_height": 26, "token_width": 38, "processed_width": 608, "processed_height": 416, "vq_stride": 16}, token)
    manifest = root / name / "manifest.jsonl"
    manifest.write_text(json.dumps({"dataset_name": name, "sample_key": f"{name}/one", "instruction": "edit", "token_file": "files/one.pt"}) + "\n", encoding="utf-8")
    (root / name / "dataset_meta.json").write_text(json.dumps({"usable_row_count": 1}) + "\n", encoding="utf-8")
    return manifest


def test_nway_builder_keeps_all_components_and_relative_token_links(tmp_path):
    manifests = {name: _component(tmp_path, name) for name in ("magicbrush", "refedit", "crispedit", "scaleedit")}
    output = tmp_path / "mixed"
    command = [sys.executable, str(ROOT / "scripts/data/build_mixed_edit_manifest.py"), "--output", str(output)]
    for name, manifest in manifests.items(): command += ["--input", f"{name}={manifest}"]
    subprocess.run(command, cwd=ROOT, check=True)
    rows = [json.loads(line) for line in (output / "train/manifest.jsonl").read_text().splitlines()]
    metadata = json.loads((output / "dataset_meta.json").read_text())
    assert [row["dataset_name"] for row in rows] == list(manifests)
    assert metadata["dataset_counts"] == {name: 1 for name in manifests}
    assert all((output / "train" / row["token_file"]).is_file() for row in rows)
