# Lumina-DiMOO Mixed Editing Experiments

本页给出空服务器上的正式 Mixed 2×3 实验流程。模型、训练数据、官方 MagicBrush TEST、token、GCE cluster 和评测权重均由仓库脚本下载或生成。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git checkout <FORMAL_EXPERIMENT_CODE_SHA>
uv sync --frozen --extra dev --extra upstream --extra analysis --extra eval --extra data
```

网络受限可显式设 `HF_ENDPOINT=https://hf-mirror.com`；repo id、不可变 revision 和最终 hash 不变。

## 空服务器目录

```bash
export PROJECT_ROOT=$PWD
export ASSET_ROOT="$PROJECT_ROOT/local_assets"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

```text
local_assets/
├── models/
├── datasets/{magicbrush,refedit,mixed,magicbrush-test}/
├── artifacts/
├── experiments/
└── logs/
```

全部 pinned public assets 位于 `configs/formal_assets.yaml`。

## 资产审计

一键准备支持失败停止、阶段 `_SUCCESS` marker、验证后跳过与任意 cwd：

```bash
bash scripts/setup/run_formal_prepare.sh --print-command
bash scripts/setup/run_formal_prepare.sh --run
```

这一个命令自动下载 Lumina、MagicBrush train、RefEdit、DINO、CLIP、LPIPS 与官方 MagicBrush TEST，并完成 tokenization、mixed manifest、GCE 和 TEST audit。无需 `MAGICBRUSH_TEST_ROOT`。正式 TEST 要求 535 sessions / 1053 turns，失败明确报 `REAL TEST ARCHIVE NOT VERIFIED`。`formal_assets.json` 记录所有 identity/hash。

## 六组正式训练

```bash
export MODEL_PATH="$ASSET_ROOT/models/Lumina-DiMOO"
export DATA_ROOT="$ASSET_ROOT/datasets/mixed"
export DATA_CONFIG="$DATA_ROOT/train/manifest.jsonl"
export GCE_CLUSTER_PATH="$ASSET_ROOT/artifacts/gce_clusters_1024_512.pt"
export OUTPUT_ROOT="$ASSET_ROOT/experiments/lumina_mixed_2x3"
export EVAL_OUTPUT_ROOT="$ASSET_ROOT/experiments/lumina_mixed_2x3_eval"
mkdir -p "$OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"
bash scripts/train/run_mixed_2x3_formal.sh --print-command
bash scripts/train/run_mixed_2x3_formal.sh --run
```

顺序固定 CE full → CE editregion → Attention full → Attention editregion → GCE full → GCE editregion。六份 YAML 的 LR、batch、epochs、`token_mean`、loss 权重及 corruption semantics 均不由该流程改变。

## MagicBrush TEST 评测

```bash
export EVAL_OUTPUT_ROOT="$ASSET_ROOT/experiments/lumina_mixed_2x3_eval"
export MAGICBRUSH_TEST_CANONICAL_MANIFEST="$ASSET_ROOT/datasets/magicbrush-test/canonical/manifest.jsonl"
export MAGICBRUSH_TEST_TOKEN_MANIFEST="$ASSET_ROOT/datasets/magicbrush-test/tokens/manifest.jsonl"
export MAGICBRUSH_TEST_SUBSET="$ASSET_ROOT/datasets/magicbrush-test/eval_subset.jsonl"
export DINO_MODEL_PATH="$ASSET_ROOT/models/dinov2-base"
export CLIP_MODEL_PATH="$ASSET_ROOT/models/clip-vit-large-patch14"
bash scripts/eval/run_mixed_2x3_eval.sh --print-command
bash scripts/eval/run_mixed_2x3_eval.sh --run
```

## 结果比较

评测输出六组 summary、per-sample JSONL 与 comparison JSON/CSV。只使用同一 1053-turn TEST、相同 seed、sampling、metric weights 与 ROI padding。

## 轻量结果打包

```bash
uv run python scripts/tools/package_formal_results.py --train-root "$OUTPUT_ROOT" --eval-root "$EVAL_OUTPUT_ROOT" --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" --output "$ASSET_ROOT/archives/lumina_mixed_2x3_results.tar.gz"
```

checkpoint、weights、token、raw data、images、cache 均排除。超过 50 MiB 明确失败。

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
```
