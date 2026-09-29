from pathlib import Path

import pytest
import torch

from dataset.geometry import sample_shared_geometry
from dataset.token_contract import atomic_save_token, validate_reusable_token, validate_token_payload


def _payload(shape=(26, 38), *, sample_key="crispedit/a", dataset_name="crispedit"):
    geometry = sample_shared_geometry((1536, 1024), 42)
    assert (geometry.crop_height // 16, geometry.crop_width // 16) == shape
    return {
        "dataset_name": dataset_name, "sample_key": sample_key, "geometry_seed": 42,
        "geometry": geometry.as_dict(), "vq_stride": 16,
        "source_codes": torch.zeros(shape, dtype=torch.int32),
        "target_codes": torch.ones(shape, dtype=torch.int32),
        "edit_mask": torch.ones(shape, dtype=torch.bool),
        "token_height": shape[0], "token_width": shape[1],
        "processed_width": geometry.crop_width, "processed_height": geometry.crop_height,
    }


def test_variable_grid_payload_contract_and_atomic_write(tmp_path):
    payload = _payload()
    assert validate_token_payload(payload, sample_key="crispedit/a", dataset_name="crispedit", require_processed_geometry=True) == (26, 38)
    destination = tmp_path / "token.pt"
    atomic_save_token(payload, destination, sample_key="crispedit/a", dataset_name="crispedit")
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert validate_token_payload(loaded, sample_key="crispedit/a", dataset_name="crispedit", require_processed_geometry=True) == (26, 38)
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.mark.parametrize("field,value", [("token_width", 37), ("processed_width", 512)])
def test_variable_grid_contract_rejects_inconsistent_identity_or_geometry(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_token_payload(payload, require_processed_geometry=True)


def test_contract_rejects_empty_mask_and_identity_mismatch():
    payload = _payload()
    payload["edit_mask"] = torch.zeros((26, 38), dtype=torch.bool)
    with pytest.raises(ValueError, match="empty edit mask"):
        validate_token_payload(payload, require_processed_geometry=True)
    payload = _payload()
    with pytest.raises(ValueError, match="sample identity mismatch"):
        validate_token_payload(payload, sample_key="crispedit/other", require_processed_geometry=True)


def test_reusable_token_rejects_corruption_and_geometry_mismatch(tmp_path):
    payload = _payload()
    destination = tmp_path / "token.pt"
    atomic_save_token(payload, destination, sample_key="crispedit/a", dataset_name="crispedit")
    assert validate_reusable_token(destination, sample_key="crispedit/a", dataset_name="crispedit", geometry_seed=42, geometry=payload["geometry"]) == (26, 38)
    with pytest.raises(ValueError, match="geometry identity mismatch"):
        validate_reusable_token(destination, sample_key="crispedit/a", dataset_name="crispedit", geometry_seed=43, geometry=payload["geometry"])
    destination.write_bytes(b"broken")
    with pytest.raises(ValueError, match="invalid existing token"):
        validate_reusable_token(destination, sample_key="crispedit/a", dataset_name="crispedit", geometry_seed=42, geometry=payload["geometry"])


def test_atomic_write_does_not_publish_a_partial_file(monkeypatch, tmp_path):
    payload = _payload()
    destination = tmp_path / "token.pt"
    monkeypatch.setattr(torch, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError, match="disk full"):
        atomic_save_token(payload, destination, sample_key="crispedit/a", dataset_name="crispedit")
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
