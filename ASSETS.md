# External assets

No model weights, MagicBrush images, pre-tokenized data, LoRA checkpoints,
optimizer states, or training logs are tracked in this repository.

## Lumina-DiMOO

Use the official `Alpha-VLLM/Lumina-DiMOO` model with tokenizer and VQ-VAE from
the same revision. Validate the target download using its configuration and
tokenizer metadata before training.

The token contract is: image offset `126356`, visual vocabulary `8192`, mask
token `126336`, BOI `126349`, EOI `126350`, newline `126084`, answer start
`126354`, and answer end `126355`. The VQ codebook used by GCE is
`vqvae.quantize.embedding.weight` with 8,192 entries of dimension 64.

## MagicBrush

The source run used a 8,807-example pre-tokenized train manifest, seed 42,
512x512 images, 32x32 VQ tokens, and a maximum sequence length of 3,072. A
target cluster needs a compliant MagicBrush copy plus token files and a manifest
whose `token_file` references are valid on that cluster. The preprocessing and
data audit helpers are `datasets/magicbrush_tokens.py` and
`tools/audit_gce_magicbrush_data.py`.

## GCE clusters

The K=[1024,512] cluster asset is deliberately not versioned in Git. Build it
from the target model's VQ codebook with:

```bash
python tools/build_gce_clusters.py --model "$MODEL_PATH" \
  --output "$GCE_CLUSTER_PATH" --device cuda:0
```

Transfer a prebuilt asset only when its SHA256 and source VQ configuration match
the target model. The asset used by the source run was 639 KiB with SHA256
`5c0eac07247298915f76a74c09365a12fb17a6cbc28a583d068c6c598e6cc162`.
