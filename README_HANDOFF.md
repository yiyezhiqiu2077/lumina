# MagicBrush image-only Grouped Cross-Entropy (GCE)

This experiment extends Lumina-DiMOO with image-only GCE while retaining the
official full-vocabulary masked-token cross-entropy and leaving inference
unchanged. It is based on `1475c9739cd6f5e2dacabbd82f907e71504e7109`.

## Loss

For each supervised visual target token, logits are sliced to the 8,192-code
visual vocabulary (offset `126356`). For each codebook clustering level, GCE is
the negative log probability mass of the cluster containing the target code:

```text
L_total = L_CE_full + 1.0 * (L_GCE_K1024 + L_GCE_K512)
```

The denominator is over visual vocabulary logits only. `model/gce_loss.py`
implements the loss, and `tools/build_gce_clusters.py` deterministically
clusters the real VQ codebook. `tools/smoke_gce_ddp.py` is the training entry
point; it logs `ce_loss`, aggregate `gce_loss`, and each cluster-level loss.

## Training

The matched CE and GCE scripts use 4 GPUs, batch 4/GPU, accumulation 2, global
batch 32, BF16, AdamW (`lr=1e-5`, `betas=(0.9,0.95)`, `weight_decay=0.1`), LoRA
rank/alpha 16/16, dropout 0.05, gradient clipping 4.0, seed 42, and maximum
sequence length 3,072. GCE uses K=[1024,512] and lambda=1.0.

```bash
MODEL_PATH=/models/Lumina-DiMOO \
DATA_CONFIG=/datasets/magicbrush/tokens/train/manifest.jsonl \
GCE_CLUSTER_PATH=/assets/gce_clusters_1024_512.pt \
OUTPUT_DIR=/runs/gce \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/train_magicbrush_gce.sh
```

For a strict one-GPU smoke with a fresh process checkpoint resume, run:

```bash
MODEL_PATH=/models/Lumina-DiMOO \
DATA_CONFIG=/datasets/magicbrush/tokens/train/manifest.jsonl \
GCE_CLUSTER_PATH=/assets/gce_clusters_1024_512.pt \
OUTPUT_DIR=/runs/gce_smoke \
bash scripts/smoke_gce.sh
```

The target-cluster manifest must contain target-cluster token-file paths. The
source manifest embeds local paths, so regenerate or rewrite it as part of
approved data transfer; do not commit token caches or images to Git.
