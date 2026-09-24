#!/usr/bin/env python3
"""Download (or safely unpack) the official password-protected MagicBrush TEST archive.

The archive remains a local asset.  This tool never uploads, mirrors, or
redistributes it.  A completed extraction is identified by ``_SUCCESS`` plus
``download_provenance.json``; partial archives and extraction directories are
never accepted as success.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath


OFFICIAL_TEST_URL = (
    "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/"
    "zhang_13253_buckeyemail_osu_edu/EUaRCBCR9sNClTGlRVG2-uIBXvPIzyYBjnoBs0ylKs8y3Q"
)
OFFICIAL_TEST_PASSWORD = "MagicBrush"
ANNOTATION_NAMES = {
    "edit_sessions.json", "test.jsonl", "test.json", "metadata_test.jsonl",
    "metadata_test.json", "metadata.jsonl", "metadata.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unsafe_member(member: zipfile.ZipInfo) -> bool:
    name = PurePosixPath(member.filename)
    # ZIP entries always use '/', but reject Windows drive spelling too.
    is_link = (member.external_attr >> 16) & 0o170000 == 0o120000
    first = name.parts[0] if name.parts else ""
    return name.is_absolute() or ".." in name.parts or ":" in first or is_link


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP only after rejecting traversal and symlink entries."""
    try:
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if not members:
                raise ValueError("official MagicBrush TEST archive is empty")
            unsafe = [member.filename for member in members if _unsafe_member(member)]
            if unsafe:
                raise ValueError(f"unsafe ZIP path/symlink entry rejected: {unsafe[:3]}")
            for member in members:
                target = destination / PurePosixPath(member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as error:
        raise ValueError(f"official MagicBrush TEST archive is not a valid ZIP: {archive}") from error


async def _playwright_download(url: str, password: str, destination: Path) -> None:
    """Use the public SharePoint password form and save its browser download."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:  # pragma: no cover - packaging contract
        raise RuntimeError("install the data extra: uv sync --extra data") from error
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            password_box = page.locator("input[type=password]").first
            if await password_box.count():
                await password_box.fill(password)
                submit = page.get_by_role("button", name="Submit").or_(page.get_by_role("button", name="Continue"))
                if await submit.count():
                    await submit.first.click()
                else:
                    await password_box.press("Enter")
            # SharePoint labels vary slightly; prefer semantic selectors, then
            # fail explicitly rather than falling back to an unknown endpoint.
            download = page.get_by_role("button", name="Download").or_(page.get_by_role("link", name="Download"))
            if not await download.count():
                await page.wait_for_timeout(1_000)
            if not await download.count():
                raise RuntimeError("SharePoint download control was not found after public password submission")
            async with page.expect_download(timeout=120_000) as event:
                await download.first.click()
            result = await event.value
            await result.save_as(destination)
        finally:
            await context.close()
            await browser.close()


def download_from_sharepoint(url: str, password: str, destination: Path) -> None:
    asyncio.run(_playwright_download(url, password, destination))


def _verified_complete(output: Path, *, archive: Path | None) -> bool:
    marker, provenance = output / "_SUCCESS", output / "download_provenance.json"
    if not marker.is_file() or not provenance.is_file():
        return False
    try:
        record = json.loads(provenance.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return archive is None or record.get("archive_sha256") == sha256(archive)


def prepare_archive(*, output: Path, archive: Path, source_url: str | None, force: bool) -> dict:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"official MagicBrush TEST archive does not exist: {archive}")
    if output.exists() and _verified_complete(output, archive=archive) and not force:
        return {"status": "skipped_verified", **json.loads((output / "download_provenance.json").read_text(encoding="utf-8"))}
    if output.exists() and not force:
        raise RuntimeError(f"TEST output is incomplete or differs from archive; use --force after inspection: {output}")
    parent = output.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.extracting.", dir=parent))
    try:
        safe_extract_zip(archive, staging)
        provenance = {
            "source_url": source_url,
            "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_filename": archive.name,
            "downloaded_at_unix": time.time(),
            "extraction": "zip-safe-path-check",
        }
        (staging / "download_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "_SUCCESS").write_text("official MagicBrush TEST archive extracted\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "prepared", **provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="local raw TEST extraction directory")
    parser.add_argument("--archive", type=Path, help="offline official ZIP fallback; no network access")
    parser.add_argument("--url", default=OFFICIAL_TEST_URL, help="official SharePoint URL")
    parser.add_argument("--force", action="store_true", help="replace this tool's existing local extraction")
    args = parser.parse_args()
    if args.archive is not None:
        record = prepare_archive(output=args.output, archive=args.archive, source_url=None, force=args.force)
    else:
        if args.output.exists() and _verified_complete(args.output, archive=None) and not args.force:
            print(json.dumps({"status": "skipped_verified", **json.loads((args.output / "download_provenance.json").read_text(encoding="utf-8"))}, indent=2))
            return
        if args.output.exists() and not args.force:
            raise RuntimeError(f"TEST output is incomplete; use --force after inspection: {args.output}")
        with tempfile.TemporaryDirectory(prefix="magicbrush_test_download_") as temporary:
            archive = Path(temporary) / "official_magicbrush_test.zip"
            download_from_sharepoint(args.url, OFFICIAL_TEST_PASSWORD, archive)
            record = prepare_archive(output=args.output, archive=archive, source_url=args.url, force=args.force)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
