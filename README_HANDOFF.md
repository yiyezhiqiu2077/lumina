# MagicBrush attention-supervision experiment

This branch contains the source-only attention-supervision experiment and the
batch-padding correction used by the padding-fix A1 run. It is based on
`1475c9739cd6f5e2dacabbd82f907e71504e7109`; it does not include the separate
full-attention experiment or ML-Cache inference work.

## Objective and loss

The normal Lumina masked target-token cross-entropy is unchanged. For selected
layers `24, 25, 26, 27`, the auxiliary loss uses real instruction-token queries
and clean source-image spatial keys. For each conditional sample and layer, it
computes a softmax over the source spatial keys only, averages heads and
instruction tokens, and applies cross-entropy against the normalized binary GT
edit mask. Unconditional or empty-mask samples contribute zero auxiliary loss.

The padding-fix A1 configuration is `L = L_gen + 0.1 * L_attn`.

## Padding correction

Variable-length batches are padded using `config.pad_token_id`, not token ID
zero. The constructed boolean pairwise attention mask is passed to SDPA with
`is_causal=False`, so real queries cannot attend to padded K/V positions.
Without padding, its all-true mask is equivalent to unmasked non-causal
attention.

## Training

The portable entry point is `scripts/train_magicbrush_attention_loss.sh`.
It requires a target-cluster model path, a target-cluster pre-tokenized
MagicBrush manifest, and an empty output path:

```bash
MODEL_PATH=/models/Lumina-DiMOO \
DATA_CONFIG=/datasets/magicbrush/tokens/train/manifest.jsonl \
OUTPUT_DIR=/runs/attn_a1 \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
bash scripts/train_magicbrush_attention_loss.sh
```

The default is batch 8/GPU, gradient accumulation 4, effective batch 64,
BF16, AdamW (`lr=1e-5`, `weight_decay=0.1`), LoRA rank/alpha 16/16, dropout
0.05, gradient clip 4.0, and seed 42. The source manifest stores token-file
locations; generate it for the target cluster or rewrite it during data
transfer rather than relying on the source server paths.

Run the one-GPU smoke and true cross-process resume test before a multi-GPU
run:

```bash
MODEL_PATH=/models/Lumina-DiMOO \
DATA_CONFIG=/datasets/magicbrush/tokens/train/manifest.jsonl \
OUTPUT_DIR=/runs/attn_smoke \
bash scripts/smoke_attention_loss.sh
```
