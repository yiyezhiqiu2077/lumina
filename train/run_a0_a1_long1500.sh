#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/zhangyuyang/lumina_attn_supervision
PYTHONPATH="$ROOT/code/Lumina-DiMOO"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

run_variant() {
    local variant="$1"
    local weight="$2"
    local source_checkpoint="$ROOT/outputs/${variant}_smoke100/checkpoint-000100"
    local destination="$ROOT/outputs/${variant}_long1500_from_smoke100_v2"
    local log="$ROOT/logs/${variant}_long1500_from_smoke100_v2.log"
    "$ROOT/env/bin/torchrun" \
        --standalone \
        --nproc_per_node=6 \
        train/train_magicbrush_attention.py \
        --model "$ROOT/models/Lumina-DiMOO" \
        --train-manifest "$ROOT/datasets/magicbrush_tokens/train/manifest.jsonl" \
        --output "$destination" \
        --attention-layers 24 25 26 27 \
        --attention-loss-weight "$weight" \
        --max-steps 1500 \
        --save-steps 0 \
        --checkpoint-steps 300 600 1000 1500 \
        --batch-size 1 \
        --gradient-accumulation 1 \
        --learning-rate 2e-5 \
        --warmup-steps 20 \
        --weight-decay 0.1 \
        --max-grad-norm 4.0 \
        --lora-rank 16 \
        --lora-alpha 16 \
        --lora-dropout 0.05 \
        --condition-dropout 0.1 \
        --seed 42 \
        --num-workers 2 \
        --resume-from-checkpoint "$source_checkpoint" \
        2>&1 | tee "$log"
}

cd "$ROOT/code/Lumina-DiMOO"
run_variant A0 0.0
run_variant A1 0.1
