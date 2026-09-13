# External assets

No model weights, MagicBrush images, token cache, LoRA checkpoints, optimizer
states, or training outputs are tracked in this repository.

## Lumina-DiMOO

Obtain the official `Alpha-VLLM/Lumina-DiMOO` model and its tokenizer/VQ-VAE
using the same revision on the target cluster. Verify the model configuration,
tokenizer files, and `vqvae/config.json` against the source-run metadata before
training.

The visual-token contract used here is: image offset `126356`, visual vocabulary
size `8192`, mask token `126336`, BOI `126349`, EOI `126350`, newline `126084`,
answer start `126354`, and answer end `126355`. MagicBrush uses 512x512 images
and a 32x32 VQ grid with row-ending newline tokens.

## MagicBrush

The training input is a pre-tokenized JSONL manifest plus per-example token
files. The source experiment used 8,807 train examples, seed 42, a 32x32 token
grid, and a maximum sequence length of 3,072 for the batch-8 run. The target
cluster must either copy the compliant token cache and regenerate a manifest
with target paths, or rerun the project preprocessing to create both. Do not
commit this data to Git.
