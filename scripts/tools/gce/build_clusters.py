#!/usr/bin/env python3
"""Build deterministic 1024/512 VQ-code clusters for the canonical GCE objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import VQModel


def kmeans(vectors: torch.Tensor, clusters: int, seed: int, *, n_init: int, iterations: int) -> torch.Tensor:
    """Return a non-empty assignment for a small, fixed set of VQ cluster levels."""
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    for trial in range(n_init):
        generator = torch.Generator(device=vectors.device).manual_seed(seed + trial)
        centers = vectors[torch.randperm(len(vectors), generator=generator, device=vectors.device)[:clusters]].clone()
        for _ in range(iterations):
            labels = torch.cdist(vectors, centers).argmin(dim=1)
            counts = torch.bincount(labels, minlength=clusters)
            sums = torch.zeros_like(centers).index_add_(0, labels, vectors)
            updated = sums / counts.clamp_min(1)[:, None]
            empty = counts == 0
            if empty.any():
                replacement = torch.randperm(len(vectors), generator=generator, device=vectors.device)[: int(empty.sum())]
                updated[empty] = vectors[replacement]
            if torch.allclose(centers, updated, atol=1e-5):
                break
            centers = updated
        labels = torch.cdist(vectors, centers).argmin(dim=1)
        inertia = (vectors - centers[labels]).square().sum()
        if best is None or inertia < best[0]:
            best = inertia, labels
    assert best is not None
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--levels", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vqvae = VQModel.from_pretrained(args.model, subfolder="vqvae", local_files_only=True).to(args.device).eval()
    vectors = vqvae.quantize.embedding.weight.detach().float()
    report: dict[str, object] = {
        "embedding_path": "vqvae.quantize.embedding.weight",
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "levels": {},
    }
    levels: dict[int, dict[str, torch.Tensor]] = {}
    for count in args.levels:
        assignment = kmeans(vectors, count, args.seed, n_init=args.n_init, iterations=args.iterations)
        groups = [torch.where(assignment == cluster)[0] for cluster in range(count)]
        if any(len(group) == 0 for group in groups):
            raise RuntimeError(f"empty k-means cluster at level {count}")
        sizes = torch.tensor([len(group) for group in groups], dtype=torch.long)
        cluster_map = torch.full((count, int(sizes.max())), -1, dtype=torch.long)
        for cluster, group in enumerate(groups):
            cluster_map[cluster, : len(group)] = group.cpu()
        levels[count] = {
            "token_to_cluster": assignment.cpu().long(),
            "cluster_map": cluster_map,
            "cluster_sizes": sizes,
        }
        report["levels"][str(count)] = {
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
            "mean_size": float(sizes.float().mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"codebook_size": len(vectors), "embedding_dim": vectors.shape[1], "embedding_path": report["embedding_path"], "levels": levels},
        args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
