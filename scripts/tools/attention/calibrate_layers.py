#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from dataset.magicbrush import MagicBrushTokenDataset
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    dataset = MagicBrushTokenDataset(
        args.manifest,
        tokenizer,
        condition_dropout=0,
        seed=args.seed,
        fixed_corruption=True,
    )
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    layers = list(range(model.config.n_layers))
    shard_path = args.output / f"calibration.rank{rank:02d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for index in range(rank, len(dataset), world):
            row = dataset[index]
            masks = {
                "instruction_token_mask": [row["instruction_token_mask"]],
                "source_spatial_mask": [row["source_spatial_mask"]],
                "source_edit_mask": [row["source_edit_mask"]],
                "attention_active": [row["attention_active"]],
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, auxiliary = model(
                    input_ids=[row["input_ids"]],
                    labels=[row["labels"]],
                    attention_supervision_layers=layers,
                    **masks,
                )
            if auxiliary.shape != (model.config.n_layers, 5):
                raise AssertionError(f"unexpected auxiliary shape {tuple(auxiliary.shape)}")
            for layer, values in enumerate(auxiliary.float().cpu().tolist()):
                handle.write(
                    json.dumps(
                        {
                            "sample_key": row["sample_key"],
                            "layer": layer,
                            "spatial_ce": values[0],
                            "conditional_mass": values[1],
                            "entropy": values[2],
                            "actual_full_mass": values[3],
                        }
                    )
                    + "\n"
                )
            print(json.dumps({"rank": rank, "sample": index}), flush=True)
    dist.barrier()
    if rank == 0:
        records = []
        for shard_rank in range(world):
            records.extend(
                json.loads(line)
                for line in (args.output / f"calibration.rank{shard_rank:02d}.jsonl").read_text().splitlines()
                if line
            )
        layer_rows = []
        for layer in layers:
            values = [record for record in records if record["layer"] == layer]
            layer_rows.append(
                {
                    "layer": layer,
                    "samples": len(values),
                    "spatial_ce_mean": np.mean([value["spatial_ce"] for value in values]),
                    "spatial_ce_std": np.std([value["spatial_ce"] for value in values]),
                    "conditional_mass_mean": np.mean([value["conditional_mass"] for value in values]),
                    "entropy_mean": np.mean([value["entropy"] for value in values]),
                    "actual_full_mass_mean": np.mean([value["actual_full_mass"] for value in values]),
                }
            )
        with (args.output / "layer_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=layer_rows[0].keys())
            writer.writeheader()
            writer.writerows(layer_rows)
        windows = []
        for start in range(len(layers) - args.window_size + 1):
            selected = layer_rows[start : start + args.window_size]
            windows.append(
                {
                    "start": start,
                    "end": start + args.window_size - 1,
                    "spatial_ce_mean": np.mean([row["spatial_ce_mean"] for row in selected]),
                    "spatial_ce_std": np.mean([row["spatial_ce_std"] for row in selected]),
                    "conditional_mass_mean": np.mean([row["conditional_mass_mean"] for row in selected]),
                }
            )
        windows.sort(key=lambda row: (row["spatial_ce_mean"], row["spatial_ce_std"], -row["conditional_mass_mean"]))
        with (args.output / "window_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=windows[0].keys())
            writer.writeheader()
            writer.writerows(windows)
        best = windows[0]
        selection = {
            "seed": args.seed,
            "calibration_samples": len(dataset),
            "window_size": args.window_size,
            "selected_layers": list(range(best["start"], best["end"] + 1)),
            "selection_rule": "lowest mean spatial CE, then lower CE std, then higher conditional mass",
            **best,
        }
        (args.output / "selected_layers.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
