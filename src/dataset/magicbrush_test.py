"""Strict importer for an unpacked official MagicBrush test archive."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from dataset.utils import write_jsonl


PARSER_VERSION = "magicbrush-test-v2"
ANNOTATION_NAMES = ("edit_sessions.json", "test.jsonl", "test.json", "metadata_test.jsonl", "metadata_test.json", "metadata.jsonl", "metadata.json")
FIELD_ALIASES = {
    "source": ("source", "source_image", "input", "input_image"),
    "target": ("target", "target_image", "output", "output_image", "edited_image"),
    "mask_edit": ("mask_edit", "edit_mask", "mask", "mask_path"),
    "instruction": ("instruction", "edit_instruction", "edit", "prompt"),
    "img_id": ("img_id", "image_id"),
    "turn_index": ("turn_index", "turn", "turn_id"),
    "session_id": ("session_id", "session", "session_name"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "edit_sessions.json" and isinstance(payload, dict) and "edit_sessions" not in payload and payload and all(isinstance(v, list) for v in payload.values()):
            values = [
                {**turn, "img_id": str(img_id), "session_id": str(img_id), "turn_index": turn_index}
                for img_id, turns in payload.items() for turn_index, turn in enumerate(turns) if isinstance(turn, dict)
            ]
            if sum(len(turns) for turns in payload.values()) != len(values): raise ValueError("official dict edit_sessions contains a non-object turn")
        elif path.name == "edit_sessions.json":
            sessions = payload.get("edit_sessions", payload) if isinstance(payload, dict) else payload
            if not isinstance(sessions, list): raise ValueError("official edit_sessions.json must contain dict[img_id]->list[turn] or a session list")
            values = []
            for session_index, session in enumerate(sessions):
                if not isinstance(session, dict):
                    raise ValueError(f"edit session {session_index} is not an object")
                turns = session.get("edits", session.get("turns"))
                if not isinstance(turns, list):
                    raise ValueError(f"edit session {session_index} has no explicit edits/turns list")
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        raise ValueError(f"edit session {session_index} turn {turn_index} is not an object")
                    values.append({**session, **turn, "session_id": session.get("session_id", session.get("img_id")), "turn_index": turn.get("turn_index", turn_index)})
        elif isinstance(payload, dict) and payload and all(isinstance(turns, list) for turns in payload.values()):
            # Real official TEST layout: {img_id: [turn, ...]}.  The archive
            # records each turn's own GT source; no generated previous turn is
            # ever introduced here.
            values = [
                {**turn, "img_id": str(img_id), "session_id": str(img_id), "turn_index": turn_index}
                for img_id, turns in payload.items()
                for turn_index, turn in enumerate(turns)
                if isinstance(turn, dict)
            ]
            if sum(len(turns) for turns in payload.values()) != len(values):
                raise ValueError("official dict edit_sessions contains a non-object turn")
        elif isinstance(payload, dict):
            candidates = [value for value in payload.values() if isinstance(value, list)]
            if len(candidates) != 1:
                raise ValueError("test annotation JSON must contain exactly one list-valued record field")
            values = candidates[0]
        else:
            values = payload
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError("test annotations must be a list of object records")
    return values


def discover_test_annotations(test_root: Path, annotations: Path | None = None) -> Path:
    root = Path(test_root).resolve()
    if annotations is not None:
        candidate = Path(annotations).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"explicit test annotation file does not exist: {candidate}")
        return candidate
    candidates = sorted({path for name in ANNOTATION_NAMES for path in root.rglob(name) if path.is_file()})
    if len(candidates) != 1:
        raise ValueError(
            "cannot uniquely identify official MagicBrush test annotations; expected exactly one of "
            f"{list(ANNOTATION_NAMES)} under {root}, found {[str(path) for path in candidates]}. "
            "Pass --annotations for a verified archive layout."
        )
    return candidates[0]


def _field(record: dict[str, Any], name: str) -> Any:
    found = [key for key in FIELD_ALIASES[name] if key in record and record[key] not in (None, "")]
    if len(found) != 1:
        raise ValueError(f"record must contain exactly one of {FIELD_ALIASES[name]} for {name}, found {found}")
    return record[found[0]]


def _relative_existing_path(value: Any, root: Path, *, field: str, img_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string path")
    supplied = Path(value)
    candidates = [root / supplied]
    if not supplied.is_absolute(): candidates.append(root / "images" / img_id / supplied.name)
    candidate = next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} must resolve inside MagicBrush test root: {value}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {field} file: {candidate}")
    return str(relative)


def canonicalize_test_records(test_root: Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(test_root).resolve()
    canonical: list[dict[str, Any]] = []
    invalid: list[str] = []
    session_fallbacks = 0
    for index, record in enumerate(records):
        try:
            img_id = str(_field(record, "img_id"))
            turn_index = int(_field(record, "turn_index"))
            session_fields = [key for key in FIELD_ALIASES["session_id"] if key in record and record[key] not in (None, "")]
            if not session_fields:
                # Official per-turn records can omit a separate session field; img_id is then the
                # archive-provided session identity, rather than a guessed cross-session mapping.
                session_id = img_id
                session_fallbacks += 1
            elif len(session_fields) == 1:
                session_id = str(record[session_fields[0]])
            else:
                raise ValueError(f"record has ambiguous session fields: {session_fields}")
            instruction = str(_field(record, "instruction")).strip()
            if not instruction:
                raise ValueError("instruction is empty")
            canonical.append(
                {
                    "index": index,
                    "sample_key": f"magicbrush-test/{img_id}_turn{turn_index}",
                    "img_id": img_id,
                    "turn_index": turn_index,
                    "session_id": session_id,
                    "source": _relative_existing_path(_field(record, "source"), root, field="source", img_id=img_id),
                    "target": _relative_existing_path(_field(record, "target"), root, field="target", img_id=img_id),
                    "mask_edit": _relative_existing_path(_field(record, "mask_edit"), root, field="mask_edit", img_id=img_id),
                    "instruction": instruction,
                }
            )
            mask_path = root / canonical[-1]["mask_edit"]
            if not bool((np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128).any()):
                raise ValueError("mask_edit is empty")
        except (TypeError, ValueError, FileNotFoundError, OSError) as error:
            invalid.append(f"record {index}: {error}")
    duplicate_count = len(canonical) - len({row["sample_key"] for row in canonical})
    metadata = {
        "split": "test",
        "sample_count": len(canonical),
        "session_count": len({row["session_id"] for row in canonical}),
        "empty_missing_invalid_count": len(invalid),
        "duplicate_sample_key_count": duplicate_count,
        "source_target_mask_complete": not invalid,
        "parser_version": PARSER_VERSION,
        "original_test_root": str(root),
        "session_id_fallback_to_img_id_count": session_fallbacks,
    }
    if invalid or duplicate_count:
        detail = invalid[:3]
        raise ValueError(
            "invalid official MagicBrush test archive: "
            f"invalid={len(invalid)} duplicate_sample_key={duplicate_count}; examples={detail}"
        )
    return canonical, metadata


def prepare_magicbrush_test(test_root: Path, output: Path, *, annotations: Path | None = None) -> dict[str, Any]:
    extraction_root = Path(test_root).resolve()
    annotation_path = discover_test_annotations(extraction_root, annotations)
    root = annotation_path.parent
    rows, metadata = canonicalize_test_records(root, _read_annotations(annotation_path))
    metadata["annotation_file"] = str(annotation_path.relative_to(extraction_root))
    metadata["annotation_sha256"] = _sha256(annotation_path)
    metadata["archive_content_root"] = str(root.relative_to(extraction_root))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Canonical manifests are portable relative to their own location, not tied
    # to where the official archive happened to be unpacked.
    for row in rows:
        for field in ("source", "target", "mask_edit"):
            row[field] = os.path.relpath(root / row[field], output.resolve())
    write_jsonl(output / "manifest.jsonl", rows)
    (output / "dataset_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
