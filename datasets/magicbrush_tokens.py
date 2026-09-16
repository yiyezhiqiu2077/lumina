from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import SPECIAL_TOKENS
from datasets.magicbrush_dataset import read_jsonl
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


class MagicBrushTokenDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        tokenizer,
        *,
        max_sequence_length: int = 5120,
        condition_dropout: float = 0.1,
        seed: int = 42,
        fixed_corruption: bool = False,
    ):
        self.rows = read_jsonl(manifest)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.condition_dropout = condition_dropout
        self.seed = seed
        self.fixed_corruption = fixed_corruption
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

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        payload = torch.load(row["token_file"], map_location="cpu", weights_only=True)
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
            flat_edit = payload["edit_mask"].flatten().tolist()
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

        target_tokens, target_labels, masked_count = _masked_target(payload["target_codes"], rng)
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
            "attention_active": conditional and bool(payload["edit_mask"].any()),
            "conditional": conditional,
            "empty_mask": not bool(payload["edit_mask"].any()),
            "sample_key": row["sample_key"],
            "instruction_token_count": sum(instruction_mask),
            "source_spatial_count": sum(source_spatial_mask),
            "source_newline_count": payload["token_height"] if conditional else 0,
            "target_spatial_count": payload["target_codes"].numel(),
            "target_newline_count": payload["token_height"],
            "total_sequence_length": len(input_ids),
            "valid_target_label_count": masked_count,
            "token_height": payload["token_height"],
            "token_width": payload["token_width"],
        }
