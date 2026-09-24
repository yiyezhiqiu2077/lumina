from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture()
def downloader():
    path = Path(__file__).resolve().parents[2] / "scripts/data/download_magicbrush_test.py"
    spec = importlib.util.spec_from_file_location("magicbrush_test_downloader", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(path: Path, *, unsafe: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt" if unsafe else "wrapped/test.json", "[]")


def test_zip_path_traversal_is_rejected(tmp_path, downloader):
    archive = tmp_path / "bad.zip"
    _zip(archive, unsafe=True)
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        downloader.safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_offline_archive_fallback_extracts_and_writes_provenance(tmp_path, downloader):
    archive, output = tmp_path / "official.zip", tmp_path / "raw"
    _zip(archive)
    result = downloader.prepare_archive(output=output, archive=archive, source_url=None, force=False)
    assert result["status"] == "prepared"
    assert (output / "wrapped/test.json").is_file()
    assert (output / "_SUCCESS").is_file()
    provenance = json.loads((output / "download_provenance.json").read_text(encoding="utf-8"))
    assert provenance["archive_sha256"] == downloader.sha256(archive)
    assert downloader.prepare_archive(output=output, archive=archive, source_url=None, force=False)["status"] == "skipped_verified"


def test_online_path_is_mocked_and_passes_public_url_to_browser(tmp_path, downloader, monkeypatch):
    calls = []

    def fake_download(url, password, destination):
        calls.append((url, password))
        _zip(destination)

    monkeypatch.setattr(downloader, "download_from_sharepoint", fake_download)
    monkeypatch.setattr(sys, "argv", ["download_magicbrush_test.py", "--output", str(tmp_path / "raw")])
    downloader.main()
    assert calls == [(downloader.OFFICIAL_TEST_URL, downloader.OFFICIAL_TEST_PASSWORD)]
