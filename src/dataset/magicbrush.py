from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.constants import SPECIAL_TOKENS
from dataset.utils import read_jsonl
from utils.prompt_utils import create_prompt_templates


def stable_sample_seed(global_seed: int, epoch: int, sample_key: str) -> int:
    """Return a process-independent RNG seed for one logical training sample.

    `sample_key`, rather than manifest index, makes corruption invariant to
    sampler order, DataLoader workers, rank, and manifest reordering.
    """
    payload = f"{int(global_seed)}\0{int(epoch)}\0{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _find_unique_subsequence(sequence: list[int], needle: list[int]) -> tuple[int, int]:
    starts = [index for index in range(len(sequence) - len(needle) + 1) if sequence[index : index + len(needle)] == needle]
    if len(starts) != 1:
        raise ValueError(f"expected one instruction-token match, found {len(starts)}")
    return starts[0], starts[0] + len(needle)


def _spatial_with_newlines(codes: torch.Tensor) -> tuple[list[int], list[bool]]:
    height, width = codes.shape
    sequence = []
    spatial = []
    for row in range(height):
        sequence.extend((codes[row].long() + SPECIAL_TOKENS["image_token_offset"]).tolist())
        spatial.extend([True] * width)
        sequence.append(SPECIAL_TOKENS["newline_token"])
        spatial.append(False)
    return sequence, spatial


def _masked_target(codes: torch.Tensor, rng: random.Random) -> tuple[list[int], list[int], int]:
    """Legacy full-target corruption.  Keep this byte-for-byte behavior."""
    flat = codes.flatten().long() + SPECIAL_TOKENS["image_token_offset"]
    ratio = math.cos(rng.random() * math.pi / 2)
    count = max(1, int(flat.numel() * ratio))
    selected = set(rng.sample(range(flat.numel()), count))
    tokens = []
    labels = []
    width = codes.shape[1]
    for index, token in enumerate(flat.tolist()):
        if index in selected:
            tokens.append(SPECIAL_TOKENS["mask_token"])
            labels.append(token)
        else:
            tokens.append(token)
            labels.append(-100)
        if (index + 1) % width == 0:
            tokens.append(SPECIAL_TOKENS["newline_token"])
            labels.append(-100)
    return tokens, labels, count


def _spatial_labels_from_layout(labels: list[int], height: int, width: int) -> torch.Tensor:
    """Invert the row-major spatial/newline layout without touching special tokens."""
    values = []
    for row in range(height):
        start = row * (width + 1)
        values.extend(labels[start : start + width])
        if labels[start + width] != -100:
            raise AssertionError("newline must never carry a generation label")
    return torch.tensor(values, dtype=torch.long).reshape(height, width)


def _layout_target_spatial(input_codes: torch.Tensor, labels: torch.Tensor) -> tuple[list[int], list[int]]:
    """Lay out target spatial codes and labels, restoring one newline per row."""
    height, width = input_codes.shape
    tokens: list[int] = []
    layout_labels: list[int] = []
    for row in range(height):
        tokens.extend((input_codes[row].long() + SPECIAL_TOKENS["image_token_offset"]).tolist())
        layout_labels.extend(labels[row].long().tolist())
        tokens.append(SPECIAL_TOKENS["newline_token"])
        layout_labels.append(-100)
    return tokens, layout_labels


def corrupt_target_spatial(
    source_codes: torch.Tensor,
    target_codes: torch.Tensor,
    edit_mask: torch.Tensor,
    rng: random.Random,
    mode: str = "full_target",
) -> dict[str, torch.Tensor | list[int] | int]:
    """Build the target input/labels for one of the two controlled modes."""
    if source_codes.shape != target_codes.shape or target_codes.shape != edit_mask.shape:
        raise ValueError("source, target, and edit-mask token geometry must match")
    height, width = target_codes.shape
    if mode == "full_target":
        # Calling the legacy helper preserves its exact RNG stream, cosine
        # schedule, count rounding, and target/newline sequence construction.
        tokens, labels, count = _masked_target(target_codes, rng)
        spatial_labels = _spatial_labels_from_layout(labels, height, width)
        selected = spatial_labels.ne(-100)
        input_spatial = target_codes.long().clone()
        input_spatial[selected] = (
            SPECIAL_TOKENS["mask_token"] - SPECIAL_TOKENS["image_token_offset"]
        )
        return {
            "tokens": tokens,
            "labels": labels,
            "masked_count": count,
            "candidate_count": target_codes.numel(),
            "selected_spatial": selected,
            # This diagnostic value must describe the code-domain spatial input
            # actually seen by the model, rather than the clean target.
            "input_spatial": input_spatial,
            "spatial_labels": spatial_labels,
        }
    if mode != "edit_region_hardlock":
        raise ValueError(f"unsupported target corruption mode: {mode!r}")

    candidate = edit_mask.bool().flatten()
    candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten().tolist()
    if not candidate_indices:
        raise ValueError("edit_region_hardlock requires a non-empty edit mask")
    ratio = math.cos(rng.random() * math.pi / 2)
    count = max(1, int(len(candidate_indices) * ratio))
    selected_indices = set(rng.sample(candidate_indices, count))
    selected = torch.zeros_like(candidate).bool()
    selected[list(selected_indices)] = True
    selected = selected.reshape_as(edit_mask)

    input_spatial = torch.where(edit_mask.bool(), target_codes.long(), source_codes.long())
    input_spatial[selected] = SPECIAL_TOKENS["mask_token"] - SPECIAL_TOKENS["image_token_offset"]
    spatial_labels = torch.full_like(target_codes, -100, dtype=torch.long)
    spatial_labels[selected] = target_codes.long()[selected] + SPECIAL_TOKENS["image_token_offset"]
    tokens, labels = _layout_target_spatial(input_spatial, spatial_labels)
    return {
        "tokens": tokens,
        "labels": labels,
        "masked_count": count,
        "candidate_count": len(candidate_indices),
        "selected_spatial": selected,
        "input_spatial": input_spatial,
        "spatial_labels": spatial_labels,
    }


class EditTokenDataset(Dataset):
    """One editing-token protocol shared by MagicBrush and RefEdit.

    Old MagicBrush manifests store absolute ``token_file`` paths.  New,
    portable manifests store paths relative to the manifest.  Resolving at
    load time keeps both formats valid without duplicating sequence assembly.
    """
    def __init__(
        self,
        manifest: Path,
        tokenizer,
        *,
        max_sequence_length: int = 5120,
        condition_dropout: float = 0.1,
        seed: int = 42,
        fixed_corruption: bool = False,
        target_corruption_mode: str = "full_target",
    ):
        self.manifest = Path(manifest).resolve()
        self.rows = read_jsonl(self.manifest)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.condition_dropout = condition_dropout
        self.seed = seed
        self.fixed_corruption = fixed_corruption
        if target_corruption_mode not in {"full_target", "edit_region_hardlock"}:
            raise ValueError(f"unsupported target corruption mode: {target_corruption_mode!r}")
        self.target_corruption_mode = target_corruption_mode
        self.epoch = 0
        self.system_prompt = create_prompt_templates()["image_editing"]

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, sample_key: str) -> random.Random:
        # Fixed probes intentionally remain fixed across epochs, while normal
        # training changes corruption only with the logical epoch.
        epoch = 0 if self.fixed_corruption else self.epoch
        return random.Random(stable_sample_seed(self.seed, epoch, sample_key))

    def _token_file(self, row: dict) -> Path:
        token_file = Path(row["token_file"])
        return token_file if token_file.is_absolute() else self.manifest.parent / token_file

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        token_file = self._token_file(row)
        if not token_file.is_file():
            raise FileNotFoundError(
                f"token file for {row.get('sample_key', index)!r} does not exist: {token_file}"
            )
        payload = torch.load(token_file, map_location="cpu", weights_only=True)
        required_payload = {"source_codes", "target_codes", "edit_mask", "token_height", "token_width"}
        missing_payload = required_payload.difference(payload)
        if missing_payload:
            raise ValueError(f"token payload {token_file} is missing keys: {sorted(missing_payload)}")
        edit_mask = payload["edit_mask"].bool()
        if self.target_corruption_mode == "edit_region_hardlock" and not bool(edit_mask.any()):
            raise ValueError(
                "empty edit mask is invalid for edit_region_hardlock: "
                f"dataset_name={row.get('dataset_name', 'magicbrush')} sample_key={row['sample_key']}"
            )
        rng = self._rng(str(row["sample_key"]))
        conditional = rng.random() >= self.condition_dropout
        instruction = row["instruction"] if conditional else "<uncondition>"
        formatted = f"<system>{self.system_prompt}</system><user>{instruction}</user>"
        text_ids = self.tokenizer(formatted, truncation=False, padding=False)["input_ids"]
        instruction_positions = set()
        if conditional:
            instruction_ids = self.tokenizer(row["instruction"], add_special_tokens=False)["input_ids"]
            start, end = _find_unique_subsequence(text_ids, instruction_ids)
            instruction_positions = set(range(start, end))

        source_tokens, source_spatial_local = _spatial_with_newlines(payload["source_codes"])
        prefix_ids = list(text_ids)
        prefix_instruction = [position in instruction_positions for position in range(len(text_ids))]
        prefix_source_spatial = [False] * len(text_ids)
        prefix_source_edit = [False] * len(text_ids)
        if conditional:
            insertion = len(prefix_ids) - 1
            source_wrapped = [SPECIAL_TOKENS["boi"]] + source_tokens + [SPECIAL_TOKENS["eoi"]]
            source_spatial = [False] + source_spatial_local + [False]
            flat_edit = edit_mask.flatten().tolist()
            source_edit = [False]
            flat_index = 0
            for is_spatial in source_spatial_local:
                source_edit.append(bool(flat_edit[flat_index]) if is_spatial else False)
                flat_index += int(is_spatial)
            source_edit.append(False)
            prefix_ids[insertion:insertion] = source_wrapped
            prefix_instruction[insertion:insertion] = [False] * len(source_wrapped)
            prefix_source_spatial[insertion:insertion] = source_spatial
            prefix_source_edit[insertion:insertion] = source_edit

        corruption = corrupt_target_spatial(
            payload["source_codes"], payload["target_codes"], edit_mask, rng,
            mode=self.target_corruption_mode,
        )
        target_tokens = corruption["tokens"]
        target_labels = corruption["labels"]
        masked_count = int(corruption["masked_count"])
        if self.target_corruption_mode == "edit_region_hardlock":
            selected = corruption["selected_spatial"].bool()
            input_spatial = corruption["input_spatial"].long()
            spatial_labels = corruption["spatial_labels"].long()
            if bool((selected & ~edit_mask).any()):
                raise AssertionError("masked positions must be a subset of the edit mask")
            if not torch.equal(input_spatial[~edit_mask], payload["source_codes"].long()[~edit_mask]):
                raise AssertionError("outside edit region must use source codes")
            if not torch.equal(spatial_labels[~edit_mask], torch.full_like(spatial_labels[~edit_mask], -100)):
                raise AssertionError("outside edit region must have ignored labels")
            if not torch.equal(input_spatial[edit_mask & ~selected], payload["target_codes"].long()[edit_mask & ~selected]):
                raise AssertionError("unmasked edit positions must use target codes")
            if not torch.equal(spatial_labels[edit_mask & ~selected], torch.full_like(spatial_labels[edit_mask & ~selected], -100)):
                raise AssertionError("unmasked edit positions must have ignored labels")
        suffix = [SPECIAL_TOKENS["answer_start"], SPECIAL_TOKENS["boi"]] + target_tokens + [
            SPECIAL_TOKENS["eoi"],
            SPECIAL_TOKENS["answer_end"],
        ]
        suffix_labels = [-100, -100] + target_labels + [-100, -100]
        input_ids = prefix_ids + suffix
        labels = [-100] * len(prefix_ids) + suffix_labels
        instruction_mask = prefix_instruction + [False] * len(suffix)
        source_spatial_mask = prefix_source_spatial + [False] * len(suffix)
        source_edit_mask = prefix_source_edit + [False] * len(suffix)
        if not (len(input_ids) == len(labels) == len(instruction_mask) == len(source_spatial_mask) == len(source_edit_mask)):
            raise AssertionError("sequence and mask lengths differ")
        if len(input_ids) > self.max_sequence_length:
            raise ValueError(f"sequence {row['sample_key']} has length {len(input_ids)} > {self.max_sequence_length}")
        if masked_count <= 0 or sum(label != -100 for label in labels) != masked_count:
            raise AssertionError("valid target label count mismatch")
        if conditional and sum(source_spatial_mask) != payload["source_codes"].numel():
            raise AssertionError("source spatial position count mismatch")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "instruction_token_mask": instruction_mask,
            "source_spatial_mask": source_spatial_mask,
            "source_edit_mask": source_edit_mask,
            "attention_active": conditional and bool(edit_mask.any()),
            "conditional": conditional,
            "empty_mask": not bool(edit_mask.any()),
            "sample_key": row["sample_key"],
            "dataset_name": row.get("dataset_name", "magicbrush"),
            "instruction_token_count": sum(instruction_mask),
            "source_spatial_count": sum(source_spatial_mask),
            "source_newline_count": payload["token_height"] if conditional else 0,
            "target_spatial_count": payload["target_codes"].numel(),
            "target_newline_count": payload["token_height"],
            "total_sequence_length": len(input_ids),
            "valid_target_label_count": masked_count,
            "target_corruption_mode": self.target_corruption_mode,
            "edit_token_count": int(edit_mask.sum()),
            "edit_fraction": float(edit_mask.float().mean()),
            "masked_target_token_count": masked_count,
            "masked_target_fraction": float(masked_count / max(payload["target_codes"].numel(), 1)),
            "masked_edit_token_count": masked_count if self.target_corruption_mode == "edit_region_hardlock" else None,
            "masked_edit_fraction": (
                float(masked_count / max(int(edit_mask.sum()), 1))
                if self.target_corruption_mode == "edit_region_hardlock"
                else None
            ),
            "token_height": payload["token_height"],
            "token_width": payload["token_width"],
        }


# Kept as a public compatibility name for old configs, tools, and checkpoints.
MagicBrushTokenDataset = EditTokenDataset
