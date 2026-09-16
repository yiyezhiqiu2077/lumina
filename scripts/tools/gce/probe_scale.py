#!/usr/bin/env python3
"""Probe canonical GCE/CE loss and logit-gradient scale on MagicBrush batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset import MagicBrushTokenDataset
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration
from models.objectives.gce import GCEObjective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    dataset = MagicBrushTokenDataset(args.manifest, tokenizer, seed=args.seed)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).train()
    gce_objective = GCEObjective.from_clusters(str(args.clusters)).to(device)
    rows = []
    for index in range(min(args.batches, len(dataset))):
        sample = dataset[index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=[sample["input_ids"]], labels=[sample["labels"]], return_training_output=True)
            gce_loss, _ = gce_objective(output.logits, output.labels)
        ce_grad = torch.autograd.grad(output.generation_loss, output.logits, retain_graph=True)[0]
        gce_grad = torch.autograd.grad(gce_loss, output.logits)[0]
        rows.append(
            {
                "generation_loss": float(output.generation_loss.detach()),
                "gce_loss": float(gce_loss.detach()),
                "gce_to_ce_logit_grad_norm": float(gce_grad.float().norm() / ce_grad.float().norm().clamp_min(1e-12)),
            }
        )
    result = {
        "summary": {key: summary([row[key] for row in rows]) for key in rows[0]},
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if result["summary"]["gce_to_ce_logit_grad_norm"]["p90"] > 10:
        raise SystemExit("GCE scale gate failed")


if __name__ == "__main__":
    main()
