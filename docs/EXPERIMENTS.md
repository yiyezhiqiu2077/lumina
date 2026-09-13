# MagicBrush experiment list

The shared experiment baseline is commit `1475c9739cd6f5e2dacabbd82f907e71504e7109`.
Full-attention supervision and ML-Cache are intentionally outside these branches.

| Exp ID | Method | Branch | Global BS | LR | Special loss | Step budget |
| --- | --- | --- | ---: | ---: | --- | ---: |
| B0-CE | masked-token CE | `experiment/gce` | 32 | 1e-5 | none | 2200 |
| G1-GCE | CE + image-only GCE | `experiment/gce` | 32 | 1e-5 | weight 1.0; K=[1024,512] | 2200 |
| A1-ATTN | CE + source-only attention supervision + padding fix | `experiment/attention` | 64 | 1e-5 | weight 0.1; layers 24-27 | 2750 |

All listed runs use BF16, seed 42, AdamW betas `(0.9, 0.95)`, weight decay
0.1, LoRA rank/alpha 16/16, LoRA dropout 0.05, and gradient clip 4.0. GCE uses
batch 4/GPU with accumulation 2. Attention uses batch 8/GPU with accumulation 4.
