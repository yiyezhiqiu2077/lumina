# MagicBrush dataset setup

The training code consumes a **pre-tokenized JSONL manifest**, not raw images at
training time. The source run used the official splits below:

| Split | Samples | Final manifest |
| --- | ---: | --- |
| train | 8,807 | `official_tokens/train/manifest.jsonl` |
| dev | 528 | `official_tokens/dev/manifest.jsonl` |
| test | 1,053 | `official_tokens/test/manifest.jsonl` |

## Target directory convention

```text
$DATA_ROOT/
├── official_raw/                 # source/target/edit-mask images and raw JSONL
├── prepared/                     # optional deterministic split manifests
└── official_tokens/
    ├── train/{manifest.jsonl,files/*.pt}
    ├── dev/{manifest.jsonl,files/*.pt}
    └── test/{manifest.jsonl,files/*.pt}
```

Set `DATA_CONFIG` to the training manifest. If it is unset, the portable
training scripts use `$DATA_ROOT/official_tokens/train/manifest.jsonl`.
Every manifest row must include `instruction`, `source`, `target`, `mask_edit`,
`session_id`, `geometry`, `token_file`, `token_height`, and `token_width`.
`token_file` must resolve on the target cluster; regenerate the manifest after a
copy if it embeds paths from another filesystem.

## Prepare and tokenize

For a raw dataset without its own train/validation manifests, create a
session-disjoint split and deterministic 512px geometry:

```bash
python tools/prepare_magicbrush.py \
  --train-manifest "$DATA_ROOT/official_raw/train_manifest.jsonl" \
  --output "$DATA_ROOT/prepared" --seed 42 --val-fraction 0.1 --target-size 512
```

Pretokenize each desired manifest with the model's `vqvae` subfolder. The
following is a real single-node DDP invocation; adjust the GPU count only after
the one-GPU check passes:

```bash
torchrun --standalone --nproc_per_node=4 tools/pretokenize_magicbrush.py \
  --manifest "$DATA_ROOT/prepared/train.jsonl" --model "$MODEL_PATH" \
  --output "$DATA_ROOT/official_tokens/train"
```

The pretokenizer applies shared geometry to source, target, and edit mask:
images use bicubic resize, masks use nearest-neighbor resize, then masks are
thresholded at 128 and adaptive-max-pooled to the VQ grid. The source run used
512x512 crops, VQ stride 16, and therefore a 32x32 code grid. Each `.pt` cache
file stores `source_codes`, `target_codes`, and `edit_mask` as 32x32 tensors.

## Token contract and sequence layout

The visual vocabulary has 8,192 codes at image token offset 126356. IDs are:
mask 126336, BOI 126349, EOI 126350, newline 126084, answer-start 126354, and
answer-end 126355. A conditional sequence is text with the source image inserted
before the final text token, followed by answer-start, BOI, masked target image
tokens with row-ending newline tokens, EOI, and answer-end. Training supervises
only selected target visual tokens. The GCE trainer passes `max_seq_len=3072`;
the current attention trainer leaves the dataset default at `5120` (the audited
source sequences are about 2.1k tokens).

Run a lightweight integrity check before training:

```bash
DATA_ROOT=/path/to/MagicBrush uv run bash scripts/check_dataset.sh
```
