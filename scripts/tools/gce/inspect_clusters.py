#!/usr/bin/env python3
"""Validate a GCE cluster artifact against the VQ codebook and canonical loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import VQModel

from models.objectives.gce import GroupedCrossEntropyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This is the same loader that training uses; it validates codebook size and requested levels.
    grouped_loss = GroupedCrossEntropyLoss.from_file(str(args.clusters), tuple(args.levels))
    payload = torch.load(args.clusters, map_location="cpu", weights_only=True)
    vectors = VQModel.from_pretrained(args.model, subfolder="vqvae", local_files_only=True).quantize.embedding.weight.float()
    generator = torch.Generator().manual_seed(0)
    result: dict[str, object] = {"codebook_size": len(vectors), "levels": {}}
    for level in grouped_loss.levels:
        ids = getattr(grouped_loss, f"token_to_cluster_{level}")
        cluster_map = getattr(grouped_loss, f"map_{level}")
        sizes = getattr(grouped_loss, f"sizes_{level}")
        intra = []
        for token in torch.randperm(len(ids), generator=generator)[: args.samples]:
            cluster = ids[token]
            members = cluster_map[cluster, : sizes[cluster]]
            intra.append((vectors[token] - vectors[members]).norm(dim=-1).mean())
        pairs = torch.randint(len(vectors), (args.samples, 2), generator=generator)
        random_distance = (vectors[pairs[:, 0]] - vectors[pairs[:, 1]]).norm(dim=-1).mean()
        mean_intra = torch.stack(intra).mean()
        result["levels"][str(level)] = {
            "cluster_count": int(payload["levels"][level]["cluster_sizes"].numel()),
            "mean_intra_distance": float(mean_intra),
            "mean_random_distance": float(random_distance),
            "intra_lt_random": bool(mean_intra < random_distance),
        }
    print(json.dumps(result, indent=2))
    if not all(value["intra_lt_random"] for value in result["levels"].values()):
        raise SystemExit("cluster quality sanity failed")


if __name__ == "__main__":
    main()
