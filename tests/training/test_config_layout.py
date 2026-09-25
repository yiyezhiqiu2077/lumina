from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "configs"


def test_final_config_layout_contains_only_supported_presets():
    assert sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.yaml")) == [
        "dataset/mixed_edit.yaml", "distributed/single_node.yaml", "formal_assets.yaml",
        "train/ablation/mixed_attention_editregion_8g_b4_a1.yaml",
        "train/ablation/mixed_ce_editregion_8g_b4_a1.yaml",
        "train/ablation/mixed_gce_editregion_8g_b4_a1.yaml",
        "train/formal/mixed_attention_postrope_region_8g_b4_a1.yaml",
        "train/formal/mixed_ce_8g_b4_a1.yaml", "train/formal/mixed_gce_8g_b4_a1.yaml",
        "train/validation/mixed_attention_4g.yaml", "train/validation/mixed_attention_editregion_4g.yaml",
        "train/validation/mixed_ce_4g.yaml", "train/validation/mixed_ce_editregion_4g.yaml",
        "train/validation/mixed_gce_4g.yaml", "train/validation/mixed_gce_editregion_4g.yaml",
        "upstream/data.yaml",
    ]
